//! Partially-loaded token embedding table.
//!
//! The full matrix buffer is allocated ONCE at construction. In the wasm
//! build this is load-bearing: reserving up front (before other allocations)
//! means wasm linear memory never grows because of us, so JS-side typed
//! array views don't detach mid-session.
//!
//! `embed` returns None — never a quietly shrunken mean — when any needed
//! row hasn't been ingested yet. model2vec deletes unknown tokens, so a
//! missing row would otherwise look identical to an out-of-vocabulary
//! token: a quieter, wrong answer with no signal. The vocab is always
//! fully loaded, so the two cases are distinguishable, and we distinguish
//! them.

use crate::StoreError;

pub struct RowStore {
    dim: usize,
    n_rows: usize,
    /// Per-row dequantization scales ("cheap insurance", 4 bytes/row).
    scales: Vec<f32>,
    /// n_rows * dim, zero until ingested.
    data: Vec<i8>,
    /// One bit per row.
    loaded: Vec<u64>,
}

impl RowStore {
    pub fn new(dim: usize, n_rows: usize, scales: Vec<f32>) -> Self {
        assert_eq!(scales.len(), n_rows, "one scale per row");
        RowStore {
            dim,
            n_rows,
            scales,
            data: vec![0i8; dim * n_rows],
            loaded: vec![0u64; n_rows.div_ceil(64)],
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn n_rows(&self) -> usize {
        self.n_rows
    }

    pub fn is_loaded(&self, row: u32) -> bool {
        let r = row as usize;
        r < self.n_rows && (self.loaded[r / 64] >> (r % 64)) & 1 == 1
    }

    fn mark_loaded(&mut self, row: usize) {
        self.loaded[row / 64] |= 1u64 << (row % 64);
    }

    /// Ingest raw i8 row bytes fetched from `model.rows.i8`, starting at
    /// `byte_start` in that file. Must be row-aligned and whole rows.
    pub fn ingest(&mut self, byte_start: usize, bytes: &[u8]) -> Result<(), StoreError> {
        if !byte_start.is_multiple_of(self.dim) {
            return Err(StoreError::Unaligned);
        }
        if !bytes.len().is_multiple_of(self.dim) {
            return Err(StoreError::PartialRow);
        }
        let row_start = byte_start / self.dim;
        let rows = bytes.len() / self.dim;
        if row_start + rows > self.n_rows {
            return Err(StoreError::OutOfBounds);
        }
        let dst = &mut self.data[byte_start..byte_start + bytes.len()];
        for (d, &s) in dst.iter_mut().zip(bytes) {
            *d = s as i8;
        }
        for r in row_start..row_start + rows {
            self.mark_loaded(r);
        }
        Ok(())
    }

    /// Row ids (deduplicated, sorted) that are needed but not yet loaded.
    pub fn missing(&self, ids: &[u32]) -> Vec<u32> {
        let mut m: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| (id as usize) < self.n_rows && !self.is_loaded(id))
            .collect();
        m.sort_unstable();
        m.dedup();
        m
    }

    /// Mean-of-rows embedding, dequantized per row, L2-normalized.
    ///
    /// None when: ids is empty (all-OOV query — no semantic signal exists),
    /// any row is unloaded (semantic search unavailable, fall back to
    /// keyword), or the mean is the zero vector.
    pub fn embed(&self, ids: &[u32]) -> Option<Vec<f32>> {
        if ids.is_empty() {
            return None;
        }
        if ids.iter().any(|&id| !self.is_loaded(id)) {
            return None;
        }
        let mut acc = vec![0f32; self.dim];
        for &id in ids {
            let i = id as usize;
            let scale = self.scales[i];
            let row = &self.data[i * self.dim..(i + 1) * self.dim];
            for (a, &v) in acc.iter_mut().zip(row) {
                *a += scale * v as f32;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> RowStore {
        // 3 rows, dim 4, scale 1.0 everywhere for easy math.
        RowStore::new(4, 3, vec![1.0; 3])
    }

    #[test]
    fn embed_none_until_loaded() {
        let mut s = store();
        assert_eq!(s.embed(&[0]), None);
        s.ingest(0, &[127, 0, 0, 0]).unwrap();
        let v = s.embed(&[0]).unwrap();
        assert!((v[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embed_none_if_any_row_missing() {
        let mut s = store();
        s.ingest(0, &[127, 0, 0, 0]).unwrap();
        // row 2 never ingested
        assert_eq!(s.embed(&[0, 2]), None);
    }

    #[test]
    fn embed_none_on_empty_ids() {
        let s = store();
        assert_eq!(s.embed(&[]), None);
    }

    #[test]
    fn mean_and_normalize() {
        let mut s = store();
        // scales are 1.0, so raw bytes are the values: row0 = (2,0,0,0)
        s.ingest(0, &[2, 0, 0, 0]).unwrap();
        // row1 = (0,2,0,0)
        s.ingest(4, &[0, 2, 0, 0]).unwrap();
        let v = s.embed(&[0, 1]).unwrap();
        // mean = (1,1,0,0), normalized = (0.7071, 0.7071, 0, 0)
        assert!((v[0] - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
        assert!((v[1] - core::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5);
    }

    #[test]
    fn ingest_alignment_checks() {
        let mut s = store();
        assert_eq!(s.ingest(2, &[0, 0, 0, 0]), Err(StoreError::Unaligned));
        assert_eq!(s.ingest(0, &[0, 0, 0]), Err(StoreError::PartialRow));
        assert_eq!(s.ingest(8, &[0; 8]), Err(StoreError::OutOfBounds));
    }

    #[test]
    fn missing_dedups_and_sorts() {
        let mut s = store();
        s.ingest(4, &[0, 0, 0, 1]).unwrap();
        assert_eq!(s.missing(&[2, 0, 2, 1]), vec![0, 2]);
    }
}
