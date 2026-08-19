//! The plan / ingest / search surface. chops-search-wasm wraps this 1:1;
//! integration tests drive it natively with the exact bytes the browser
//! would see.
//!
//! Contract with the JS pump:
//!   1. construct with meta + index bytes (vocab must be complete)
//!   2. ingest(0, prefix_bytes) for the eager prefix
//!   3. per query: plan() → range-fetch → ingest() each → search()
//!
//! search() never fails: if any needed row is still unloaded (or the fetch
//! never happened — offline, CSP, whatever), it degrades to keyword-only
//! and reports that via used_semantic(). Degrading loudly-but-gracefully
//! beats a quietly shrunken mean.

use crate::format::{Index, ModelMeta};
use crate::keyword::KeywordIndex;
use crate::plan::{ByteRange, coalesce};
use crate::store::RowStore;
use crate::wordpiece::Vocab;
use crate::{FormatError, StoreError, rrf, score};

/// Gap (in rows) below which two needed rows share one range request.
/// Public because `chops-search plan` reads it as the shipped cell of
/// its --max-gap axis; the CLI must not carry its own copy of this.
pub const MAX_GAP_ROWS: u32 = 8;

pub struct Engine {
    vocab: Vocab,
    store: RowStore,
    index: Index,
    kw: KeywordIndex,
    dim: usize,
    prefix_rows: u32,
    used_semantic: bool,
    /// doc id → chunk that produced its score in the last search;
    /// u32::MAX where the semantic side didn't rank the doc.
    best_chunk: Vec<u32>,
    /// Chunks per doc, for the report's chunks/penalty columns on
    /// keyword-only paths where no DocScores exists.
    chunk_counts: Vec<u32>,
    /// First chunk index of each doc — chunks are contiguous per doc
    /// because the builder emits them in doc order. Used for the
    /// keyword-only snippet fallback.
    doc_first_chunk: Vec<u32>,
    opts: crate::score::ScoreOpts,
}
/// One term the keyword side resolved, with the numbers behind its score.
pub struct TermEvidence {
    pub term: Box<str>,
    pub weight: f32,
    /// True when this term came from prefix-expanding the trailing word.
    pub expanded: bool,
    pub df: u32,
    pub idf: f32,
}

/// What the semantic side did for this query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SemanticStatus {
    Unavailable,
    BelowFloor,
    /// Embedded fine, but the corroboration gate suppressed the list: no
    /// keyword evidence, top − median below min_gap, and no top strong
    /// enough to clear strong_cos.
    Suppressed,
    Ranked,
}

/// One document's line in the fused order, with per-engine contributions.
pub struct DocEvidence {
    pub doc: u16,
    /// The actual RRF sum this doc fused at.
    pub fused: f32,
    /// Position in the keyword list, when it contributed (0-based).
    pub kw_rank: Option<u16>,
    /// BM25F evidence — populated even when the confidence gate suppressed
    /// the list, so a suppressed run still shows what the evidence was.
    pub kw_score: f32,
    /// Position in the semantic list, when it contributed (0-based).
    pub sem_rank: Option<u16>,
    /// Best raw chunk cosine; None when the semantic side never ran or
    /// the doc has no chunks.
    pub best_cos: Option<f32>,
    pub penalty: f32,
    pub chunks: u32,
}

/// The full evidence behind one search. `search()` is a view of this;
/// explain prints it; nothing restates the arithmetic.
pub struct SearchReport {
    pub kw_words: Vec<Box<str>>,
    pub terms: Vec<TermEvidence>,
    /// Typed words that matched no corpus term (and, for the trailing
    /// word, produced no expansions either).
    pub unmatched: Vec<Box<str>>,
    /// Mean field lengths BM25F normalized against. Four numbers, not
    /// one: a title hit's score only makes sense against the average
    /// title, and explain has to be able to show that a 4-word title in
    /// a corpus averaging 6 was the reason a doc won.
    pub avg_title: f32,
    pub avg_tag: f32,
    pub avg_desc: f32,
    pub avg_body: f32,
    pub kw_confidence: f32,
    pub kw_gated: bool,
    /// The weight the keyword list fused at: 1.0 under plain RRF, higher
    /// once rrf_alpha is armed. Reported rather than recomputed by the
    /// caller, so explain cannot print a weight the ranker did not use.
    /// Meaningless when the list was gated, and 1.0 there by convention.
    pub kw_rrf_weight: f32,
    pub semantic: SemanticStatus,
    /// Fused order, untruncated — callers apply their own limit.
    pub docs: Vec<DocEvidence>,
    pub gap: Option<f32>,
    /// Best raw cosine across the field; None when the query never
    /// embedded. The number the strong_cos hatch judged.
    pub top: Option<f32>,
}

impl SearchReport {
    pub fn ids(&self, limit: usize) -> Vec<u16> {
        self.docs.iter().take(limit).map(|d| d.doc).collect()
    }
}

impl Engine {
    pub fn new(meta_bytes: &[u8], index_bytes: &[u8]) -> Result<Self, FormatError> {
        let meta = ModelMeta::read(meta_bytes)?;
        let index = Index::read(index_bytes)?;
        if index.dim != meta.dim {
            return Err(FormatError::Inconsistent("index dim != model dim"));
        }
        let dim = meta.dim as usize;
        let vocab = Vocab::from_tokens(&meta.tokens);
        // Full matrix buffer reserved here, once, before anything else in
        // the session allocates — wasm memory never grows because of us.
        let store = RowStore::new(dim, meta.tokens.len(), meta.scales.clone());
        let kw = index.keyword_index();
        let n_docs = index.docs.len();
        let mut doc_first_chunk = vec![u32::MAX; n_docs];
        let mut chunk_counts = vec![0u32; n_docs];
        for (c, &doc) in index.chunk_doc.iter().enumerate() {
            let d = doc as usize;
            if d < n_docs && doc_first_chunk[d] == u32::MAX {
                doc_first_chunk[d] = c as u32;
            }
            chunk_counts[d] += 1;
        }
        // Read off `index` before it moves into the struct literal below.
        let weights = index.weights;
        let min_gap = index.min_gap;
        let rrf_alpha = index.rrf_alpha;
        let min_cos_override = index.min_cos;
        let chunk_penalty = index.chunk_penalty;
        Ok(Engine {
            vocab,
            store,
            index,
            kw,
            dim,
            prefix_rows: meta.prefix_rows,
            used_semantic: false,
            best_chunk: vec![u32::MAX; n_docs],
            chunk_counts,
            doc_first_chunk,
            // Two provenances in one struct, deliberately:
            //
            // min_cos is DERIVED from geometry unless the artifact says
            // otherwise. The floor scales with dimensionality, so the
            // default cannot be a constant: PCA raises noise cosines,
            // and a value calibrated at 256 dims is too permissive at
            // 128. An index-shipped override pins it instead — and an
            // explicit 0.0 there means "floor off", which is why the
            // format keeps absent and zero distinct.
            //
            // min_gap, rrf_alpha, and the field weights are READ FROM
            // THE ARTIFACT. They're corpus-calibrated, not derivable,
            // and index.bin is the only way the browser can learn what
            // chops-search.toml said. Their compiled defaults (0.0,
            // inert) are what an unconfigured corpus writes, so the
            // browser, a bare eval, and CI construct the same ScoreOpts
            // from the same bytes.
            opts: crate::score::ScoreOpts {
                min_cos: min_cos_override.unwrap_or_else(|| crate::score::min_cos_for(dim)),
                min_gap,
                rrf_alpha,
                weights,
                chunk_penalty,
                ..Default::default()
            },
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn prefix_rows(&self) -> u32 {
        self.prefix_rows
    }

    pub fn n_rows(&self) -> usize {
        self.store.n_rows()
    }

    pub fn doc_count(&self) -> usize {
        self.index.docs.len()
    }

    pub fn doc_url(&self, id: u16) -> Option<&str> {
        self.index.docs.get(id as usize).map(|d| d.url.as_str())
    }

    pub fn doc_title(&self, id: u16) -> Option<&str> {
        self.index.docs.get(id as usize).map(|d| d.title.as_str())
    }

    /// Byte ranges of `model.rows.i8` needed before this query can be
    /// answered semantically. Empty when everything is already loaded —
    /// including the all-out-of-vocabulary case, where no rows exist to
    /// fetch and search() will correctly go keyword-only.
    pub fn plan(&self, query: &str) -> Vec<ByteRange> {
        let ids = self.vocab.tokenize(query);
        let missing = self.store.missing(&ids);
        coalesce(&missing, self.dim as u32, MAX_GAP_ROWS)
    }

    /// Feed back bytes fetched from `model.rows.i8` at `byte_start`
    /// (the prefix file is byte_start 0).
    pub fn ingest(&mut self, byte_start: u32, bytes: &[u8]) -> Result<(), StoreError> {
        self.store.ingest(byte_start as usize, bytes)
    }
    /// The scoring thresholds currently in effect. `min_cos` is derived
    /// from the index's dimensionality and the BM25F field weights come
    /// from index.bin, so this — not `ScoreOpts::default()` — is the
    /// right base for a caller that wants to override one field.
    pub fn score_opts(&self) -> crate::score::ScoreOpts {
        self.opts
    }

    /// Override scoring thresholds (eval sweeps these; the browser uses
    /// the defaults). Passing a bare `ScoreOpts::default()` here would
    /// silently discard the index's own field weights — build from
    /// `score_opts()` instead.
    pub fn set_score_opts(&mut self, opts: crate::score::ScoreOpts) {
        self.opts = opts;
    }

    /// Hybrid search: keyword BM25F and semantic ranked lists fused with
    /// RRF. Returns ranked doc ids, truncated to `limit`.
    pub fn search(&mut self, query: &str, limit: usize) -> Vec<u16> {
        self.search_detailed(query).ids(limit)
    }

    /// The full evidence behind one search: term-level keyword scoring,
    /// the confidence gate's verdict, pre-floor semantic cosines, and the
    /// fused order with per-engine contributions. `search()` is a view of
    /// this; explain prints it; nothing restates the arithmetic.
    pub fn search_detailed(&mut self, query: &str) -> SearchReport {
        // Keyword side works on word-level tokens (pre-WordPiece) so
        // out-of-vocabulary terms are first-class here.
        let norm = Vocab::normalize(query);
        let words: Vec<&str> = crate::keyword::keyword_words(&norm);
        let terms = self.kw.resolve(&words, true);
        let kw_confidence = self.kw.confidence(&words, &terms);
        let kw_gated = kw_confidence < self.opts.kw_confidence;
        // Scores are computed even when gated: the report shows the
        // evidence that WAS suppressed, which is the diagnostic point.
        // Weights come from opts, not from the index, so an eval sweep
        // re-ranks without rebuilding.
        let kw_scores = self.kw.score_terms(&terms, self.opts.weights);
        let kw_ranked = if kw_gated {
            Vec::new()
        } else {
            // Title-cover tier: docs whose titles contain every typed
            // word rank ahead of docs whose titles don't, BM25F order
            // within each tier. Cover is computed from the SAME resolved
            // terms the scores came from, so the trailing word's
            // expansions can cover mid-typing, and the kw# positions the
            // report shows are the tiered positions the ranker actually
            // used.
            let cover = self.kw.title_cover(&words, &terms);
            KeywordIndex::rank_from_scores_covered(&kw_scores, &cover)
        };

        let term_evidence: Vec<TermEvidence> = terms
            .iter()
            .map(|t| {
                let df = self.kw.terms.get(&t.text).map_or(0, |p| p.len());
                TermEvidence {
                    term: t.text.clone(),
                    weight: t.weight,
                    expanded: t.expanded,
                    df: df as u32,
                    idf: self.kw.idf(df),
                }
            })
            .collect();
        let unmatched: Vec<Box<str>> = words
            .iter()
            .filter(|w| !terms.iter().any(|t| t.text.as_ref() == **w))
            .map(|w| Box::from(*w))
            .collect();

        let ids = self.vocab.tokenize(query);
        let (semantic, doc_scores, sem_ranked, gap, top) = match self.store.embed(&ids) {
            None => (SemanticStatus::Unavailable, None, Vec::new(), None, None),
            Some(q) => {
                let ds = score::score_docs(
                    &q,
                    &self.index.chunk_vecs,
                    self.dim,
                    self.index.global_scale,
                    &self.index.chunk_doc,
                    self.index.docs.len(),
                );
                let gap = score::top_median_gap(&ds.best);
                let top = ds.best.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                // Corroboration gate: no keyword evidence, nothing standing out from
                // the pack, and no clearly-relevant top. That combination is the noise
                // signature on a homogeneous corpus. A flat field whose best doc is
                // strongly relevant is a broad-but-real query, not noise, so strong_cos
                // exempts it. Only the semantic list is suppressed; kw_ranked is
                // already empty.
                let gated = kw_ranked.is_empty()
                    && self.opts.min_gap > 0.0
                    && gap < self.opts.min_gap
                    && top < self.opts.strong_cos;
                let ranked = if gated {
                    Vec::new()
                } else {
                    score::rank_scored(&ds, self.opts)
                };
                let status = if gated {
                    SemanticStatus::Suppressed
                } else if ranked.is_empty() {
                    SemanticStatus::BelowFloor
                } else {
                    SemanticStatus::Ranked
                };
                (status, Some(ds), ranked, Some(gap), Some(top))
            }
        };

        // Reset before filling: a doc ranked last query but not this one
        // must fall back rather than show a stale snippet.
        self.best_chunk.iter_mut().for_each(|c| *c = u32::MAX);
        for r in &sem_ranked {
            self.best_chunk[r.doc as usize] = r.chunk;
        }
        let sem_ids: Vec<u16> = sem_ranked.iter().map(|r| r.doc).collect();
        // used_semantic tracks CONTRIBUTION, not merely that embedding
        // succeeded — the floor can empty the list.
        self.used_semantic = !sem_ids.is_empty();

        // Weighted fusion, fully parameterized by opts — this call is the
        // one place the fusion happens, so both knobs land here. The
        // keyword list's weight comes from rrf_alpha via kw_confidence:
        // plain RRF treats both engines as equally credible on every
        // query, and scaling the keyword vote by how much of the query's
        // idf mass it actually matched lets a df-1 exact hit outvote a
        // topical semantic first place, without giving a stopword-heavy
        // query the same licence. rrf_alpha 0 (the default) makes this
        // weight exactly 1.0 and the arithmetic identical to unweighted
        // RRF. The curve's k comes from rrf_k rather than the rrf::K
        // constant so eval can sweep the discount — at corpus scale the
        // conventional 60 is nearly flat across a dozen-deep list, which
        // is a choice to measure, not to hardcode.
        let kw_rrf_weight = self.opts.kw_rrf_weight(kw_confidence);
        let fused = rrf::fuse_scored(
            &[(&kw_ranked[..], kw_rrf_weight), (&sem_ids[..], 1.0)],
            self.opts.rrf_k,
        );
        let pos = |list: &[u16], d: u16| list.iter().position(|&x| x == d).map(|p| p as u16);
        let docs: Vec<DocEvidence> = fused
            .into_iter()
            .map(|(d, f)| {
                let du = d as usize;
                let chunks = doc_scores
                    .as_ref()
                    .map_or(self.chunk_counts[du], |ds| ds.counts[du] as u32);
                DocEvidence {
                    doc: d,
                    fused: f,
                    kw_rank: pos(&kw_ranked, d),
                    kw_score: kw_scores[du],
                    sem_rank: pos(&sem_ids, d),
                    best_cos: doc_scores
                        .as_ref()
                        .map(|ds| ds.best[du])
                        .filter(|c| *c > f32::NEG_INFINITY),
                    penalty: score::chunk_correction(chunks as usize, self.opts.chunk_penalty),
                    chunks,
                }
            })
            .collect();

        SearchReport {
            kw_words: words.iter().map(|w| Box::from(*w)).collect(),
            terms: term_evidence,
            unmatched,
            avg_title: self.kw.avg_title,
            avg_tag: self.kw.avg_tag,
            avg_desc: self.kw.avg_desc,
            avg_body: self.kw.avg_body,
            kw_confidence,
            kw_gated,
            kw_rrf_weight,
            semantic,
            docs,
            gap,
            top,
        }
    }
    /// Whether the last search() actually used the vector side. False
    /// means keyword-only: all-OOV query, or rows not yet loaded.
    pub fn used_semantic(&self) -> bool {
        self.used_semantic
    }
    /// Chunk to snippet for `doc`. Falls back to the doc's SECOND chunk —
    /// its first body chunk — when the semantic side didn't rank it,
    /// because chunk 0 is the synthetic title+tags chunk and repeating the
    /// title under the title makes a useless snippet.
    pub fn best_chunk(&self, doc: u16) -> u32 {
        let d = doc as usize;
        if let Some(&c) = self.best_chunk.get(d)
            && c != u32::MAX
        {
            return c;
        }
        let first = self.doc_first_chunk.get(d).copied().unwrap_or(0);
        if first == u32::MAX {
            return 0;
        }
        let next = first as usize + 1;
        if next < self.index.chunk_doc.len() && self.index.chunk_doc[next] == doc {
            next as u32
        } else {
            first
        }
    }

    /// Total chunks — the client needs this to size the snippet offset
    /// header before requesting it.
    pub fn chunk_count(&self) -> usize {
        self.index.chunk_doc.len()
    }
}
