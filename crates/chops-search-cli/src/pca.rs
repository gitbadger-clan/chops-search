//! PCA reduction of the token embedding table, for `--dims`.
//!
//! Why this exists instead of column truncation: model2vec applies PCA at
//! distillation time, but the potion models are trained FURTHER afterward
//! (Tokenlearn), so nothing guarantees the stored coordinates are still
//! variance-ordered. Truncating columns of potion-base-8M discards an
//! unknown amount of signal; re-running PCA on the token matrix is exact
//! and costs a one-off 256×256 eigendecomposition at build time.
//!
//! This is centered PCA (matching sklearn / model2vec's own reduction).
//! One honest caveat: centering shifts every row by -mean, which perturbs
//! absolute row norms — and row magnitude is the model's learned stopword
//! weighting. The structure survives in practice (Bart's --dims 128 run
//! measured fine), but if you ever see stopwords gaining weight after
//! reduction, uncentered PCA (plain SVD of X) is the knob to try.
//!
//! f64 throughout: the covariance accumulates ~30K terms per entry, and
//! f32 accumulation there costs real precision for zero benefit at build
//! time.

use nalgebra::{DMatrix, SymmetricEigen};

/// Project `rows` (n × dim, row-major) onto its top-k principal
/// components. Returns n × k row-major f32.
///
/// Panics if k == 0, k > dim, or rows is not a whole number of rows —
/// caller (the CLI) validates the flag before this point.
pub fn pca_reduce(rows: &[f32], dim: usize, k: usize) -> Vec<f32> {
    assert!(k > 0 && k <= dim, "k must be in 1..=dim");
    assert!(
        dim > 0 && rows.len() % dim == 0,
        "rows not a multiple of dim"
    );
    let n = rows.len() / dim;
    assert!(n > 1, "need at least two rows for PCA");

    let x = DMatrix::<f64>::from_row_iterator(n, dim, rows.iter().map(|&v| v as f64));

    // Column means over all rows.
    let mut mean = vec![0f64; dim];
    for r in 0..n {
        for c in 0..dim {
            mean[c] += x[(r, c)];
        }
    }
    for m in &mut mean {
        *m /= n as f64;
    }
    let centered = DMatrix::<f64>::from_fn(n, dim, |r, c| x[(r, c)] - mean[c]);

    // Covariance (dim × dim, symmetric) and its eigendecomposition.
    let cov = (centered.transpose() * &centered) / (n as f64 - 1.0);
    let eig = SymmetricEigen::new(cov);

    // nalgebra does not guarantee eigenvalue ordering; sort descending.
    // PANIC-SAFETY: eigenvalues of a real symmetric matrix built from
    // finite input are finite reals, so partial_cmp cannot be None.
    let mut order: Vec<usize> = (0..dim).collect();
    order.sort_by(|&a, &b| {
        eig.eigenvalues[b]
            .partial_cmp(&eig.eigenvalues[a])
            .expect("symmetric eigenvalues are finite")
    });

    let mut proj = DMatrix::<f64>::zeros(dim, k);
    for (j, &i) in order.iter().take(k).enumerate() {
        proj.set_column(j, &eig.eigenvectors.column(i));
    }

    // n × k, back to row-major f32.
    let reduced = centered * proj;
    let mut out = Vec::with_capacity(n * k);
    for r in 0..n {
        for c in 0..k {
            out.push(reduced[(r, c)] as f32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dist(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f32>()
            .sqrt()
    }

    /// Data confined to a 2-D plane inside 3-D (third coordinate constant):
    /// reducing to k=2 must preserve all pairwise distances exactly, since
    /// the dropped component carries zero variance.
    #[test]
    fn planar_data_reduces_losslessly() {
        let pts: Vec<[f32; 3]> = vec![
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [0.0, 2.0, 5.0],
            [3.0, 1.0, 5.0],
        ];
        let rows: Vec<f32> = pts.iter().flatten().copied().collect();
        let red = pca_reduce(&rows, 3, 2);
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let d0 = dist(&pts[i], &pts[j]);
                let d1 = dist(&red[i * 2..(i + 1) * 2], &red[j * 2..(j + 1) * 2]);
                assert!((d0 - d1).abs() < 1e-4, "pair ({i},{j}): {d0} vs {d1}");
            }
        }
    }

    /// k == dim is a pure rotation (+ centering): distances preserved.
    #[test]
    fn full_rank_is_isometric() {
        let rows = vec![
            1.0f32, 2.0, 3.0, //
            4.0, 6.0, 5.0, //
            -1.0, 0.5, 2.0, //
            0.0, -2.0, 1.0,
        ];
        let red = pca_reduce(&rows, 3, 3);
        for i in 0..4 {
            for j in (i + 1)..4 {
                let d0 = dist(&rows[i * 3..(i + 1) * 3], &rows[j * 3..(j + 1) * 3]);
                let d1 = dist(&red[i * 3..(i + 1) * 3], &red[j * 3..(j + 1) * 3]);
                assert!((d0 - d1).abs() < 1e-4);
            }
        }
    }

    /// Variance actually concentrates: with one dominant direction, the
    /// first reduced coordinate must carry (almost) all the spread.
    #[test]
    fn dominant_direction_lands_first() {
        // Points along (1,1)/√2 with tiny orthogonal jitter.
        let rows = vec![
            0.0f32, 0.0, //
            10.0, 10.1, //
            20.0, 19.9, //
            30.0, 30.0,
        ];
        let red = pca_reduce(&rows, 2, 2);
        let var = |c: usize| {
            let m: f32 = (0..4).map(|r| red[r * 2 + c]).sum::<f32>() / 4.0;
            (0..4).map(|r| (red[r * 2 + c] - m).powi(2)).sum::<f32>() / 3.0
        };
        assert!(var(0) > 100.0 * var(1), "first component should dominate");
    }
}
