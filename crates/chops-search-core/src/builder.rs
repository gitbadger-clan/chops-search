//! Build-time transforms. Pure functions over in-memory matrices so the
//! CLI stays a thin I/O shell and everything here is unit-testable.

/// Per-row symmetric int8 quantization: scale = max_abs / 127.
/// Per-row (not global) because the model's stopword list is its row
/// magnitudes — "the" is a near-zero vector, "guantanamo" a huge one —
/// and per-row scales preserve relative magnitudes exactly.
/// Returns (int8 rows, per-row scales).
pub fn quantize_rows(rows_f32: &[f32], dim: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(dim > 0 && rows_f32.len().is_multiple_of(dim));
    let n = rows_f32.len() / dim;
    let mut data = Vec::with_capacity(rows_f32.len());
    let mut scales = Vec::with_capacity(n);
    for r in 0..n {
        let row = &rows_f32[r * dim..(r + 1) * dim];
        let max_abs = row.iter().fold(0f32, |m, &v| m.max(v.abs()));
        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        scales.push(scale);
        for &v in row {
            data.push((v / scale).round().clamp(-127.0, 127.0) as i8);
        }
    }
    (data, scales)
}

/// Global-scale int8 quantization for the (normalized) chunk vectors.
/// One scale is fine here: cosine similarity is invariant to a positive
/// scale, and normalized vectors share a value range anyway.
pub fn quantize_global(vecs_f32: &[f32]) -> (Vec<i8>, f32) {
    let max_abs = vecs_f32.iter().fold(0f32, |m, &v| m.max(v.abs()));
    let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
    let data = vecs_f32
        .iter()
        .map(|&v| (v / scale).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (data, scale)
}

/// Frequency ordering: permutation such that the most corpus-frequent
/// token gets the lowest new id. new_id_of_old[old] = new.
///
/// This is what makes the eager prefix pull its weight: the rows cheap
/// enough to bundle up front are the frequent ones, and — because the
/// model learned that frequent words carry tiny vectors while rare words
/// carry the discriminative signal — the rows worth a network round trip
/// are exactly the rare ones. Ties break on old id so the build is
/// byte-stable.
pub fn frequency_permutation(counts: &[u64]) -> Vec<u32> {
    let mut order: Vec<u32> = (0..counts.len() as u32).collect();
    order.sort_by(|&a, &b| counts[b as usize].cmp(&counts[a as usize]).then(a.cmp(&b)));
    // order[new] = old  →  invert
    let mut new_id_of_old = vec![0u32; counts.len()];
    for (new, &old) in order.iter().enumerate() {
        new_id_of_old[old as usize] = new as u32;
    }
    new_id_of_old
}

/// Apply a permutation to rows (and, at the call site, the token list and
/// any already-tokenized id streams).
pub fn permute_rows_f32(rows: &[f32], dim: usize, new_id_of_old: &[u32]) -> Vec<f32> {
    let n = rows.len() / dim;
    assert_eq!(n, new_id_of_old.len());
    let mut out = vec![0f32; rows.len()];
    for old in 0..n {
        let new = new_id_of_old[old] as usize;
        out[new * dim..(new + 1) * dim].copy_from_slice(&rows[old * dim..(old + 1) * dim]);
    }
    out
}

/// Build-time embedding against the FULL f32 table (index quality doesn't
/// pay the quantization tax; only the shipped table does). Mean + L2 norm,
/// mirroring RowStore::embed. None for an empty id list.
pub fn embed_f32(ids: &[u32], rows: &[f32], dim: usize) -> Option<Vec<f32>> {
    if ids.is_empty() {
        return None;
    }
    let mut acc = vec![0f32; dim];
    for &id in ids {
        let row = &rows[id as usize * dim..(id as usize + 1) * dim];
        for (a, &v) in acc.iter_mut().zip(row) {
            *a += v;
        }
    }
    let inv = 1.0 / ids.len() as f32;
    for a in &mut acc {
        *a *= inv;
    }
    let norm = acc.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm <= f32::EPSILON {
        return None;
    }
    for a in &mut acc {
        *a /= norm;
    }
    Some(acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn per_row_quantization_preserves_direction() {
        // A tiny "stopword" row next to a huge "rare word" row.
        let rows = vec![
            0.001, -0.002, 0.0005, 0.0, // near-zero row
            3.0, -1.5, 2.25, 0.75, // big row
        ];
        let (q, scales) = quantize_rows(&rows, 4);
        assert_eq!(scales.len(), 2);
        for r in 0..2 {
            let deq: Vec<f32> = q[r * 4..(r + 1) * 4]
                .iter()
                .map(|&v| v as f32 * scales[r])
                .collect();
            let c = cosine(&deq, &rows[r * 4..(r + 1) * 4]);
            assert!(c > 0.999, "row {r} cosine {c}");
        }
    }

    #[test]
    fn frequency_permutation_orders_desc_stable() {
        let counts = [5u64, 100, 5, 0];
        let perm = frequency_permutation(&counts);
        // old 1 (count 100) → new 0; old 0 and 2 tie at 5, old 0 first.
        assert_eq!(perm, vec![1, 0, 2, 3]);
    }

    #[test]
    fn permute_roundtrip() {
        let rows = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 rows dim 2
        let perm = vec![2u32, 0, 1]; // old0→new2, old1→new0, old2→new1
        let p = permute_rows_f32(&rows, 2, &perm);
        assert_eq!(p, vec![3.0, 4.0, 5.0, 6.0, 1.0, 2.0]);
    }

    #[test]
    fn embed_f32_normalizes() {
        let rows = vec![2.0, 0.0, 0.0, 2.0];
        let v = embed_f32(&[0, 1], &rows, 2).unwrap();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }
}
