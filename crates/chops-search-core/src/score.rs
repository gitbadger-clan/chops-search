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
//! Note the division of labor: the floor tests RAW similarity (is this
//! document related at all?), while ranking uses the ADJUSTED score (given
//! that several are related, which deserves to be first?). Conflating them
//! would let a long document's penalty push it below the relevance floor,
//! which is a different claim than the penalty is licensed to make.

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

#[derive(Debug, Clone, Copy)]
pub struct ScoreOpts {
    pub min_cos: f32,
    pub chunk_penalty: f32,
}

/// A ranked document plus the chunk that earned it its score — the chunk
/// whose text becomes the snippet.
#[derive(Debug, Clone, Copy)]
pub struct Ranked {
    pub doc: u16,
    pub chunk: u32,
    pub score: f32,
}

impl Default for ScoreOpts {
    fn default() -> Self {
        ScoreOpts {
            min_cos: MIN_COS,
            chunk_penalty: CHUNK_PENALTY,
        }
    }
}

impl ScoreOpts {
    /// Raw max-pooling with no floor and no correction — the pre-correction
    /// behavior, kept for A/B comparison in eval.
    pub fn raw() -> Self {
        ScoreOpts {
            min_cos: f32::NEG_INFINITY,
            chunk_penalty: 0.0,
        }
    }
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

    // Floor on RAW similarity, rank on ADJUSTED.
    let mut out: Vec<Ranked> = (0..n_docs)
        .filter(|&d| best[d] > f32::NEG_INFINITY && best[d] >= opts.min_cos)
        .map(|d| Ranked {
            doc: d as u16,
            chunk: best_chunk[d],
            score: best[d] - chunk_correction(counts[d], opts.chunk_penalty),
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
            chunk_penalty: 0.0,
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
            chunk_penalty: 0.0,
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
}
