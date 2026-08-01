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
use crate::plan::{coalesce, ByteRange};
use crate::store::RowStore;
use crate::wordpiece::Vocab;
use crate::{rrf, score, FormatError, StoreError};

/// Gap (in rows) below which two needed rows share one range request.
const MAX_GAP_ROWS: u32 = 8;

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
    /// First chunk index of each doc — chunks are contiguous per doc
    /// because the builder emits them in doc order. Used for the
    /// keyword-only snippet fallback.
    doc_first_chunk: Vec<u32>,
    opts: crate::score::ScoreOpts,
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
        for (c, &doc) in index.chunk_doc.iter().enumerate() {
            let d = doc as usize;
            if d < n_docs && doc_first_chunk[d] == u32::MAX {
                doc_first_chunk[d] = c as u32;
            }
        }
        Ok(Engine {
            vocab,
            store,
            index,
            kw,
            dim,
            prefix_rows: meta.prefix_rows,
            used_semantic: false,
            best_chunk: vec![u32::MAX; n_docs],
            doc_first_chunk,
            opts: crate::score::ScoreOpts::default(),
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

    /// Override scoring thresholds (eval sweeps these; the browser uses
    /// the defaults).
    pub fn set_score_opts(&mut self, opts: crate::score::ScoreOpts) {
        self.opts = opts;
    }

    /// Hybrid search: keyword tf-idf and semantic ranked lists fused with
    /// RRF. Returns ranked doc ids, truncated to `limit`.
    pub fn search(&mut self, query: &str, limit: usize) -> Vec<u16> {
        // Keyword side works on word-level tokens (pre-WordPiece) so
        // out-of-vocabulary terms are first-class here.
        let norm = Vocab::normalize(query);
        let words: Vec<&str> = crate::keyword::keyword_words(&norm);
        let kw_ranked = self.kw.rank(&words);

        let ids = self.vocab.tokenize(query);
        let fused = match self.store.embed(&ids) {
            Some(q) => {
                let detailed = score::rank_docs_detailed(
                    &q,
                    &self.index.chunk_vecs,
                    self.dim,
                    self.index.global_scale,
                    &self.index.chunk_doc,
                    self.index.docs.len(),
                    self.opts,
                );
                // Reset before filling: a doc ranked last query but not
                // this one must fall back rather than show a stale snippet.
                self.best_chunk.iter_mut().for_each(|c| *c = u32::MAX);
                for r in &detailed {
                    self.best_chunk[r.doc as usize] = r.chunk;
                }
                let sem_ranked: Vec<u16> = detailed.iter().map(|r| r.doc).collect();
                // The floor can empty this list: the query embedded fine,
                // nothing was relevant. Reporting "hybrid" then would be a
                // lie to the UI, so used_semantic tracks CONTRIBUTION, not
                // merely that embedding succeeded.
                self.used_semantic = !sem_ranked.is_empty();
                if sem_ranked.is_empty() {
                    kw_ranked
                } else {
                    rrf::fuse(&[&kw_ranked, &sem_ranked], rrf::K)
                }
            }
            None => {
                self.best_chunk.iter_mut().for_each(|c| *c = u32::MAX);
                self.used_semantic = false;
                kw_ranked
            }
        };
        fused.into_iter().take(limit).collect()
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
        if let Some(&c) = self.best_chunk.get(d) {
            if c != u32::MAX {
                return c;
            }
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
