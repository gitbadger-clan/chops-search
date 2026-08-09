//! Brute-force semantic scoring: a few hundred dot products, then group
//! chunks by post and take each post's best chunk.
//!
//! No ANN index on purpose — at static-site scale the linear scan is
//! sub-millisecond and an HNSW graph would be slower to load than to beat.
//!
//! The query is f32 (already dequantized + normalized); chunk vectors are
//! int8 with ONE global scale. Because cosine similarity is invariant to a
//! positive scale, the global scale doesn't even change the ranking — we
//! multiply it in anyway so scores are comparable across builds. The f32
//! accumulator sidesteps the int8×int8 JS overflow trap by construction;
//! if this ever moves to wasm SIMD, accumulate i32 (i16x8 extmul + pairwise
//! add), never i8.
//!
//! `ScoreOpts` lives here but is the tuning surface for BOTH halves of the
//! hybrid: the semantic floor and chunk correction below, plus the keyword
//! side's confidence gate and BM25F field weights. One struct because eval
//! sweeps them together and the engine carries exactly one of them.
//!
//! Two corrections sit on top of raw max-pooling, both driven by measured
//! behavior rather than taste:
//!
//! RELEVANCE FLOOR (`min_cos`). Max-pooling always returns every document,
//! so a query about nothing in the corpus still produces a confident-looking
//! ranking of noise — observed: an off-topic query scored 0.040/0.017/-0.003
//! across the corpus while on-topic queries score 0.29–0.45. The floor
//! filters the semantic list only. The keyword list is deliberately NOT
//! floored: BM25 already requires a document to literally contain a query
//! term, and an exact rare-term hit must not be vetoed by vectors that
//! never learned the word (the whole reason the hybrid exists).
//!
//! CHUNK-COUNT CORRECTION (`chunk_penalty`). max over n chunks is a biased
//! estimator of relevance: for noise-like scores the expected maximum grows
//! as √(2 ln n), so a 23-chunk post outranks a 1-chunk post on sampling
//! alone. Subtracting `coeff · √(2 ln n)` corrects the bias with an
//! extreme-value anchor instead of an arbitrary length fudge. n = 1 costs
//! nothing (ln 1 = 0), which is the property a pure length penalty lacks.
//!
//! FUSION WEIGHT (`rrf_alpha`). Plain RRF treats both engines as equally
//! credible on every query. Observed failure: for a query on a unique
//! compound term the keyword engine ranks the right document first on
//! df-1 evidence, the semantic engine ranks a topically adjacent page
//! first, and the arithmetic prefers the semantic winner because
//! 1/(60+2) + 1/(60+1) > 1/(60+1) + 1/(60+3). Both engines were correct;
//! the fusion was not. `rrf_alpha` scales the keyword list's vote by
//! `1 + alpha * kw_confidence`, so a query whose keyword evidence is
//! strong gets a louder keyword vote and a stopword-heavy one does not.
//! Sweep with `chops-search eval --rrf-alpha`; 0.0 is plain RRF.
//!
//! FUSION CURVE (`rrf_k`). The RRF rank discount. Unlike every other knob
//! it has no disabling value: fusion always happens on some curve, and k
//! only sets its shape. The conventional 60 comes from TREC-scale runs
//! fusing thousand-deep lists; at corpus scale — tens of documents, lists
//! a dozen deep — that curve is nearly flat (rank 1 vs rank 2 is 1/61 vs
//! 1/62, a gap of 0.0003), so fusion degenerates toward "best average
//! rank wins" and a decisive #1 in one engine cannot survive mediocrity
//! in the other. Small k re-sharpens the top. Sweep with
//! `chops-search eval --sweep-rrf-k`, pin with `--rrf-k`; the default
//! stays `rrf::K` so plain conventional RRF remains the shipped behavior
//! until a sweep says otherwise.
//!
//! Note the division of labor: the floor tests RAW similarity (is this
//! document related at all?), while ranking uses the ADJUSTED score (given
//! that several are related, which deserves to be first?). Conflating them
//! would let a long document's penalty push it below the relevance floor,
//! which is a different claim than the penalty is licensed to make.

use crate::keyword::FieldWeights;

/// Minimum raw best-chunk similarity for a document to count as
/// semantically relevant. Calibrated at the model's native 256 dims,
/// where on-topic queries score 0.29–0.45 and pure noise stays under
/// 0.04.
///
/// PCA changes the geometry: fewer dimensions means less room for two
/// unrelated vectors to be far apart, so noise cosines rise. Scaling by
/// √(256/dim) keeps the floor at the same distance from the noise floor
/// instead of leaving it behind — at 128 dims that is 0.28.
pub const MIN_COS: f32 = 0.20;

/// Coefficient on the √(2 ln n) chunk-count correction. Sweep with
/// `chops-search eval --chunk-penalty`; 0.0 disables the correction entirely.
pub const CHUNK_PENALTY: f32 = 0.02;

/// Minimum fraction of the query's potential idf mass that must have
/// matched for the keyword ranking to be trusted at all. Below this, the
/// keyword engine submits nothing to fusion: a ranking assembled from
/// stopword coincidences or a single prefix expansion is worse than no
/// ranking, because RRF consumes ranks and launders away how weak the
/// evidence was. Sweep with `chops-search eval --kw-floor`; 0.0 disables.
pub const KW_CONFIDENCE: f32 = 0.30;

/// Coefficient scaling the keyword list's RRF vote by its confidence.
/// Ships at 0.0, which is plain unweighted RRF: this knob changes the
/// order of results on queries that currently rank correctly, so it
/// lands inert and is armed only once a sweep says what it is worth.
/// Sweep with `chops-search eval --rrf-alpha`.
pub const RRF_ALPHA: f32 = 0.0;

#[derive(Debug, Clone, Copy)]
pub struct ScoreOpts {
    pub min_cos: f32,
    pub chunk_penalty: f32,
    pub kw_confidence: f32,
    pub min_gap: f32,
    pub strong_cos: f32,
    /// BM25F field weights: what a length-normalized title, tag, or
    /// description occurrence is worth against a body occurrence (body is
    /// fixed at 1.0). Defaults come from `keyword`, but the engine
    /// overwrites them from `index.bin` at construction, since the values
    /// a corpus was built with belong with the corpus. Sweep with
    /// `chops-search eval --w-title/--w-tag/--w-desc`.
    ///
    /// One struct rather than three fields, because three consecutive f32
    /// parameters is where a transposition compiles, type-checks, and
    /// shows up only as "ranking got slightly worse".
    pub weights: FieldWeights,
    /// How much a confident keyword list outvotes the semantic one in
    /// RRF: the keyword list fuses at `1 + rrf_alpha * kw_confidence`,
    /// the semantic list always at 1.0.
    ///
    /// A FLOOR-style knob: 0.0 disables it and reproduces plain RRF
    /// exactly. Note the scale is not free-form, and it is coupled to
    /// `rrf_k`: at the default k = 60 the curve is flat enough that the
    /// keyword weight must reach about 2 before it can overturn a
    /// semantic first place two ranks above it, so values under 1.0 are
    /// mostly inert on the case this exists for. A smaller k sharpens
    /// the curve and moves that crossover down — sweep the two jointly.
    pub rrf_alpha: f32,
    /// The RRF rank discount k. A SHAPE parameter, not a gate: there is
    /// no disabling value, only the question of how fast a list's vote
    /// decays with rank. Defaults to `rrf::K` (the conventional 60) —
    /// see the module header for why that convention is suspect at
    /// corpus scale. Sweep with `chops-search eval --sweep-rrf-k`, pin
    /// with `--rrf-k`.
    pub rrf_k: f32,
}

/// A ranked document plus the chunk that earned it its score — the chunk
/// whose text becomes the snippet.
#[derive(Debug, Clone, Copy)]
pub struct Ranked {
    pub doc: u16,
    pub chunk: u32,
    pub score: f32,
}

/// Raw per-doc evidence, before the floor and penalty judge it. This is
/// what the report layer needs: sub-floor cosines are exactly the numbers
/// explain must print for documents the semantic side rejected.
pub struct DocScores {
    /// Best raw cosine per doc; NEG_INFINITY where a doc has no chunks.
    pub best: Vec<f32>,
    /// Chunk that produced it; u32::MAX where a doc has no chunks.
    pub best_chunk: Vec<u32>,
    /// Chunks per doc, counted in the same pass.
    pub counts: Vec<usize>,
}

impl Default for ScoreOpts {
    fn default() -> Self {
        ScoreOpts {
            min_cos: MIN_COS,
            chunk_penalty: CHUNK_PENALTY,
            kw_confidence: KW_CONFIDENCE,
            min_gap: 0.0,
            strong_cos: f32::INFINITY,
            weights: FieldWeights::default(),
            rrf_alpha: RRF_ALPHA,
            // The one knob whose default lives in another module: rrf.rs
            // owns the conventional constant, and a literal 60.0 here
            // would be a second copy waiting to drift when a swept value
            // gets pinned.
            rrf_k: crate::rrf::K,
        }
    }
}

impl ScoreOpts {
    /// Raw max-pooling with no floor and no correction — the pre-correction
    /// behavior, kept for A/B comparison in eval.
    ///
    /// Field weights are deliberately NOT zeroed here. Every other field
    /// this disables is a gate or a correction sitting on top of a
    /// ranking; the weights ARE the keyword ranking function, and
    /// stripping them would make the baseline a different scorer rather
    /// than the same scorer unjudged.
    pub fn raw() -> Self {
        ScoreOpts {
            min_cos: f32::NEG_INFINITY,
            chunk_penalty: 0.0,
            kw_confidence: 0.0,
            min_gap: 0.0,
            strong_cos: f32::INFINITY,
            weights: FieldWeights::default(),
            // A gate-like knob, so raw() disables it: raw() is "the
            // ranking function with nothing sitting on top", and a
            // weighted fusion is something sitting on top.
            rrf_alpha: 0.0,
            // Like the field weights, NOT reset to anything special: k
            // is part of the fusion function itself, not something
            // sitting on top of it. A raw() that fused on a different
            // curve than the shipped engine would be a different scorer,
            // not the same scorer unjudged.
            rrf_k: crate::rrf::K,
        }
    }

    /// The weight the keyword list fuses at, given how much of the
    /// query's idf mass it actually matched. Lives here rather than in
    /// the engine so eval, explain, and the browser cannot disagree
    /// about it, and so rrf.rs stays a pure function of ranks.
    pub fn kw_rrf_weight(&self, kw_confidence: f32) -> f32 {
        1.0 + self.rrf_alpha * kw_confidence
    }
}
/// Contrast of the best raw cosine against the corpus pack.
/// Homogeneous noise is flat (small gap); a real match stands out.
/// Computed on RAW best-cos, pre-floor and pre-chunk-penalty: the gap
/// asks "does anything stand out from the whole corpus", so filtering
/// or penalizing before measuring would distort the statistic.
pub fn top_median_gap(best_cos: &[f32]) -> f32 {
    let mut v: Vec<f32> = best_cos.iter().copied().filter(|c| c.is_finite()).collect();
    if v.len() < 2 {
        return f32::INFINITY; // one doc can't be "flat"; never gate
    }
    v.sort_unstable_by(|a, b| a.total_cmp(b));
    v[v.len() - 1] - v[v.len() / 2]
}
/// Expected-maximum correction for a document with `n` chunks.
/// Zero for n ≤ 1; grows slowly (n=4 → 1.66·coeff, n=32 → 2.63·coeff).
pub fn chunk_correction(n: usize, coeff: f32) -> f32 {
    if n <= 1 || coeff == 0.0 {
        return 0.0;
    }
    coeff * (2.0 * (n as f32).ln()).sqrt()
}

pub fn min_cos_for(dim: usize) -> f32 {
    MIN_COS * (256.0 / dim.max(1) as f32).sqrt()
}
/// Rank documents by their best-scoring chunk, descending.
///
/// Doc ids only. Callers that need to know WHICH chunk won — for
/// snippets — want `rank_docs_detailed`.
pub fn rank_docs(
    q: &[f32],
    chunk_vecs: &[i8],
    dim: usize,
    global_scale: f32,
    chunk_doc: &[u16],
    n_docs: usize,
    opts: ScoreOpts,
) -> Vec<u16> {
    rank_docs_detailed(q, chunk_vecs, dim, global_scale, chunk_doc, n_docs, opts)
        .into_iter()
        .map(|r| r.doc)
        .collect()
}
/// Max-pooling only — the measurement half, ending where the floor begins.
///
/// - `q`: normalized f32 query vector, length `dim`
/// - `chunk_vecs`: n_chunks × dim int8
/// - `chunk_doc`: chunk index → doc id
/// - `n_docs`: total docs (doc ids are 0..n_docs)
pub fn score_docs(
    q: &[f32],
    chunk_vecs: &[i8],
    dim: usize,
    global_scale: f32,
    chunk_doc: &[u16],
    n_docs: usize,
) -> DocScores {
    debug_assert_eq!(q.len(), dim);
    debug_assert_eq!(chunk_vecs.len(), chunk_doc.len() * dim);

    let mut best = vec![f32::NEG_INFINITY; n_docs];
    let mut best_chunk = vec![u32::MAX; n_docs];
    let mut counts = vec![0usize; n_docs];
    for (c, &doc) in chunk_doc.iter().enumerate() {
        let d = doc as usize;
        if d >= n_docs {
            continue;
        }
        let row = &chunk_vecs[c * dim..(c + 1) * dim];
        let mut acc = 0f32;
        for (&qi, &vi) in q.iter().zip(row) {
            acc += qi * vi as f32;
        }
        let s = acc * global_scale;
        counts[d] += 1;
        if s > best[d] {
            best[d] = s;
            best_chunk[d] = c as u32;
        }
    }
    DocScores {
        best,
        best_chunk,
        counts,
    }
}

/// Floor + chunk penalty + sort — the judgment half.
///
/// Floor on RAW similarity, rank on ADJUSTED. Docs with no chunks are
/// omitted, as are docs whose raw best similarity is below
/// `opts.min_cos`. Ties break on doc id for byte-stable output.
pub fn rank_scored(scores: &DocScores, opts: ScoreOpts) -> Vec<Ranked> {
    let n_docs = scores.best.len();
    let mut out: Vec<Ranked> = (0..n_docs)
        .filter(|&d| scores.best[d] > f32::NEG_INFINITY && scores.best[d] >= opts.min_cos)
        .map(|d| Ranked {
            doc: d as u16,
            chunk: scores.best_chunk[d],
            score: scores.best[d] - chunk_correction(scores.counts[d], opts.chunk_penalty),
        })
        .collect();
    out.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.doc.cmp(&b.doc))
    });
    out
}

/// As `rank_docs`, but also reports the winning chunk and the adjusted
/// score per document.
///
/// - `q`: normalized f32 query vector, length `dim`
/// - `chunk_vecs`: n_chunks × dim int8
/// - `chunk_doc`: chunk index → doc id
/// - `n_docs`: total docs (doc ids are 0..n_docs)
///
/// Docs with no chunks are omitted, as are docs whose raw best similarity
/// is below `opts.min_cos`. Ties break on doc id for byte-stable output.
pub fn rank_docs_detailed(
    q: &[f32],
    chunk_vecs: &[i8],
    dim: usize,
    global_scale: f32,
    chunk_doc: &[u16],
    n_docs: usize,
    opts: ScoreOpts,
) -> Vec<Ranked> {
    rank_scored(
        &score_docs(q, chunk_vecs, dim, global_scale, chunk_doc, n_docs),
        opts,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn best_chunk_wins_for_doc() {
        // dim 2, 3 chunks: doc0 has a weak and a strong chunk, doc1 medium.
        let q = [1.0f32, 0.0];
        let chunks: Vec<i8> = vec![
            10, 0, // chunk0 → doc0, score 10
            100, 0, // chunk1 → doc0, score 100
            50, 0, // chunk2 → doc1, score 50
        ];
        let ranked = rank_docs(&q, &chunks, 2, 1.0, &[0, 0, 1], 2, ScoreOpts::raw());
        assert_eq!(ranked, vec![0, 1]);
    }

    #[test]
    fn docs_without_chunks_omitted() {
        let q = [1.0f32];
        let ranked = rank_docs(&q, &[5], 1, 1.0, &[2], 4, ScoreOpts::raw());
        assert_eq!(ranked, vec![2]);
    }

    #[test]
    fn floor_drops_noise_keeps_signal() {
        // dim 1, scale 0.01: doc0 best 0.45, doc1 best 0.04.
        let q = [1.0f32];
        let chunks: Vec<i8> = vec![45, 4];
        let opts = ScoreOpts {
            min_cos: 0.20,
            ..Default::default()
        };
        let ranked = rank_docs(&q, &chunks, 1, 0.01, &[0, 1], 2, opts);
        assert_eq!(ranked, vec![0], "0.04 should be below the floor");
    }

    #[test]
    fn floor_can_empty_the_list() {
        let q = [1.0f32];
        let chunks: Vec<i8> = vec![4, 2];
        let opts = ScoreOpts {
            min_cos: 0.20,
            ..Default::default()
        };
        assert!(rank_docs(&q, &chunks, 1, 0.01, &[0, 1], 2, opts).is_empty());
    }

    #[test]
    fn chunk_correction_demotes_the_lottery_winner() {
        // doc0: 16 chunks, best 0.50. doc1: 1 chunk, 0.42.
        // Raw → doc0 first. Corrected at coeff 0.05: doc0 loses
        // 0.05·√(2·ln16) ≈ 0.118 → 0.382, so doc1 wins.
        let q = [1.0f32];
        let mut chunks: Vec<i8> = vec![50];
        chunks.extend(std::iter::repeat_n(10i8, 15));
        chunks.push(42);
        let mut chunk_doc = vec![0u16; 16];
        chunk_doc.push(1);

        let raw = rank_docs(&q, &chunks, 1, 0.01, &chunk_doc, 2, ScoreOpts::raw());
        assert_eq!(raw, vec![0, 1]);

        let opts = ScoreOpts {
            min_cos: f32::NEG_INFINITY,
            chunk_penalty: 0.05,
            ..Default::default()
        };
        let corrected = rank_docs(&q, &chunks, 1, 0.01, &chunk_doc, 2, opts);
        assert_eq!(corrected, vec![1, 0]);
    }

    #[test]
    fn single_chunk_docs_are_never_penalized() {
        assert_eq!(chunk_correction(1, 0.5), 0.0);
        assert_eq!(chunk_correction(0, 0.5), 0.0);
        assert!(chunk_correction(2, 0.5) > 0.0);
    }

    #[test]
    fn correction_grows_slowly() {
        // Doubling chunk count must not double the penalty — the whole
        // point of √(2 ln n) over a linear or log-linear penalty.
        let a = chunk_correction(4, 1.0);
        let b = chunk_correction(32, 1.0);
        assert!(b / a < 1.7, "penalty grew too fast: {a} → {b}");
    }

    #[test]
    fn floor_uses_raw_not_adjusted() {
        // doc0: 32 chunks, best 0.25 — above a 0.20 floor. A big penalty
        // must reorder it, never filter it out.
        let q = [1.0f32];
        let mut chunks: Vec<i8> = vec![25];
        chunks.extend(std::iter::repeat_n(1i8, 31));
        let chunk_doc = vec![0u16; 32];
        let opts = ScoreOpts {
            min_cos: 0.20,
            chunk_penalty: 0.5,
            ..Default::default()
        };
        assert_eq!(
            rank_docs(&q, &chunks, 1, 0.01, &chunk_doc, 1, opts),
            vec![0]
        );
    }

    #[test]
    fn detailed_reports_the_winning_chunk() {
        let q = [1.0f32, 0.0];
        let chunks: Vec<i8> = vec![
            10, 0, // chunk0 → doc0
            100, 0, // chunk1 → doc0, the winner
            50, 0, // chunk2 → doc1
        ];
        let out = rank_docs_detailed(&q, &chunks, 2, 1.0, &[0, 0, 1], 2, ScoreOpts::raw());
        assert_eq!(out[0].doc, 0);
        assert_eq!(
            out[0].chunk, 1,
            "should name the strong chunk, not the first"
        );
        assert_eq!(out[1].doc, 1);
        assert_eq!(out[1].chunk, 2);
    }

    #[test]
    fn tied_chunks_pick_the_earliest() {
        // Deterministic snippet selection: equal scores must not depend on
        // iteration incidentals, or the artifact stops being byte-stable.
        let q = [1.0f32];
        let chunks: Vec<i8> = vec![50, 50, 50];
        let out = rank_docs_detailed(&q, &chunks, 1, 1.0, &[0, 0, 0], 1, ScoreOpts::raw());
        assert_eq!(out[0].chunk, 0);
    }

    #[test]
    fn score_docs_preserves_sub_floor_evidence() {
        // The floor rejects doc1, but the measurement must still report
        // its cosine — that's what explain prints for rejected docs.
        let q = [1.0f32];
        let chunks: Vec<i8> = vec![45, 4];
        let ds = score_docs(&q, &chunks, 1, 0.01, &[0, 1], 2);
        assert!((ds.best[1] - 0.04).abs() < 1e-6);
        assert_eq!(ds.counts, vec![1, 1]);
        let opts = ScoreOpts {
            min_cos: 0.20,
            ..Default::default()
        };
        assert_eq!(rank_scored(&ds, opts).len(), 1);
        assert_eq!(rank_scored(&ds, opts)[0].doc, 0);
    }

    #[test]
    fn gap_ignores_chunkless_docs() {
        // NEG_INFINITY entries (docs without chunks) are not part of the pack.
        let gap = top_median_gap(&[0.40, 0.10, 0.12, f32::NEG_INFINITY]);
        assert!((gap - 0.28).abs() < 1e-6); // median of {0.10,0.12,0.40} is 0.12
    }

    #[test]
    fn gap_is_measured_pre_floor() {
        // A floor that rejects every doc must not change the gap: the gap
        // reads the measurement (score_docs), never the judgment (rank_scored).
        let q = [1.0f32];
        let chunks: Vec<i8> = vec![40, 10, 12];
        let ds = score_docs(&q, &chunks, 1, 0.01, &[0, 1, 2], 3);
        let gap = top_median_gap(&ds.best);
        assert!((gap - 0.28).abs() < 1e-6); // 0.40 − median 0.12

        // Sanity: a high floor empties the ranking while the gap stands.
        let opts = ScoreOpts {
            min_cos: 0.9,
            ..Default::default()
        };
        assert!(rank_scored(&ds, opts).is_empty());
        assert!((top_median_gap(&ds.best) - 0.28).abs() < 1e-6);
    }

    #[test]
    fn raw_keeps_the_field_weights() {
        // raw() disables gates and corrections, not the ranking function.
        // A raw() that zeroed the weights would make the eval baseline a
        // body-only scorer and quietly overstate what BM25F bought.
        let raw = ScoreOpts::raw();
        assert_eq!(raw.weights, FieldWeights::default());
        assert_eq!(raw.weights, ScoreOpts::default().weights);
    }

    #[test]
    fn rrf_alpha_ships_inert() {
        // The knob changes the order of results that currently rank
        // correctly, so the default must reproduce plain RRF exactly.
        assert_eq!(ScoreOpts::default().rrf_alpha, 0.0);
        assert_eq!(ScoreOpts::raw().rrf_alpha, 0.0);
        for conf in [0.0f32, 0.3, 1.0] {
            assert_eq!(ScoreOpts::default().kw_rrf_weight(conf), 1.0);
        }
    }

    #[test]
    fn kw_weight_scales_with_confidence() {
        // The point of routing through confidence: a query whose keyword
        // evidence is weak gets no louder a vote than plain RRF gave it,
        // while a fully matched rare term reaches the ~2 needed to
        // overturn a semantic first place AT THE DEFAULT k = 60 (see the
        // module header; a smaller rrf_k moves this crossover down).
        let o = ScoreOpts {
            rrf_alpha: 1.0,
            ..Default::default()
        };
        assert_eq!(o.kw_rrf_weight(0.0), 1.0, "no evidence, no boost");
        assert_eq!(o.kw_rrf_weight(0.5), 1.5);
        assert_eq!(
            o.kw_rrf_weight(1.0),
            2.0,
            "the crossover for kw#1 vs sem#1 at the default k = 60"
        );
    }

    #[test]
    fn rrf_k_defaults_to_the_shared_constant() {
        // One source of truth: the engine and explain fuse with whatever
        // ScoreOpts carries, and ScoreOpts starts from rrf::K. A literal
        // 60.0 in Default would be a second copy waiting to drift the
        // day a swept value gets pinned.
        assert_eq!(ScoreOpts::default().rrf_k, crate::rrf::K);
    }

    #[test]
    fn raw_keeps_the_fusion_curve() {
        // Same claim as raw_keeps_the_field_weights: raw() strips gates
        // and corrections, never the ranking or fusion functions
        // themselves. A raw() on a different curve would be a different
        // scorer, not the same scorer unjudged.
        assert_eq!(ScoreOpts::raw().rrf_k, ScoreOpts::default().rrf_k);
    }

    #[test]
    fn weights_are_independent_fields() {
        // Grouping them in a struct is packaging, not coupling: each
        // weight still lands on its own field's normalized tf, and
        // overriding one leaves the others where the index put them.
        let base = ScoreOpts {
            weights: FieldWeights {
                title: 3.0,
                tag: 7.0,
                desc: 0.5,
            },
            ..Default::default()
        };
        let swept = ScoreOpts {
            weights: FieldWeights {
                desc: 0.0,
                ..base.weights
            },
            ..base
        };
        assert_eq!(swept.weights.title, 3.0);
        assert_eq!(swept.weights.tag, 7.0);
        assert_eq!(swept.weights.desc, 0.0);
    }
}
