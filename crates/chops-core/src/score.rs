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

/// Rank documents by their best-scoring chunk, descending.
///
/// - `q`: normalized f32 query vector, length `dim`
/// - `chunk_vecs`: n_chunks × dim int8
/// - `chunk_doc`: chunk index → doc id
/// - `n_docs`: total docs (doc ids are 0..n_docs)
///
/// Returns ranked doc ids; docs with no chunks are omitted. Ties break on
/// doc id for byte-stable output.
pub fn rank_docs(
    q: &[f32],
    chunk_vecs: &[i8],
    dim: usize,
    global_scale: f32,
    chunk_doc: &[u16],
    n_docs: usize,
) -> Vec<u16> {
    debug_assert_eq!(q.len(), dim);
    debug_assert_eq!(chunk_vecs.len(), chunk_doc.len() * dim);

    let mut best = vec![f32::NEG_INFINITY; n_docs];
    for (c, &doc) in chunk_doc.iter().enumerate() {
        let row = &chunk_vecs[c * dim..(c + 1) * dim];
        let mut acc = 0f32;
        for (&qi, &vi) in q.iter().zip(row) {
            acc += qi * vi as f32;
        }
        let s = acc * global_scale;
        let d = doc as usize;
        if d < n_docs && s > best[d] {
            best[d] = s;
        }
    }

    let mut ranked: Vec<u16> = (0..n_docs as u16)
        .filter(|&d| best[d as usize] > f32::NEG_INFINITY)
        .collect();
    ranked.sort_unstable_by(|&a, &b| {
        best[b as usize]
            .partial_cmp(&best[a as usize])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    ranked
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
        let ranked = rank_docs(&q, &chunks, 2, 1.0, &[0, 0, 1], 2);
        assert_eq!(ranked, vec![0, 1]);
    }

    #[test]
    fn docs_without_chunks_omitted() {
        let q = [1.0f32];
        let ranked = rank_docs(&q, &[5], 1, 1.0, &[2], 4);
        assert_eq!(ranked, vec![2]);
    }
}
