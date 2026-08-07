//! Reciprocal Rank Fusion.
//!
//! Both engines' scores are thrown away entirely; only ranks are kept.
//! fused(doc) = Σ over engines of 1 / (k + rank), rank starting at 1,
//! k conventionally 60. The k flattens the top of the curve so "both
//! engines quite liked this" beats "one engine loved it".
//!
//! This also makes late arrival graceful: a keyword-only list is just RRF
//! with one engine, and when the semantic list shows up the fusion is
//! purely additive — no rescaling problem, because there are no scales.

use std::collections::HashMap;

pub const K: f32 = 60.0;

/// Fused (doc, RRF score) pairs, score descending, ties on doc id. The
/// scores exist for the report layer — explain prints contributions from
/// here instead of restating the formula.
pub fn fuse_scored(lists: &[&[u16]], k: f32) -> Vec<(u16, f32)> {
    let mut scores: HashMap<u16, f32> = HashMap::new();
    for list in lists {
        for (i, &doc) in list.iter().enumerate() {
            *scores.entry(doc).or_insert(0.0) += 1.0 / (k + (i + 1) as f32);
        }
    }
    let mut out: Vec<(u16, f32)> = scores.into_iter().collect();
    out.sort_unstable_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    out
}

/// Fuse ranked lists of doc ids. Ties break on doc id for determinism.
pub fn fuse(lists: &[&[u16]], k: f32) -> Vec<u16> {
    fuse_scored(lists, k).into_iter().map(|(d, _)| d).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranked_third_by_both_beats_first_by_one() {
        // doc 9 is third in both lists: 1/63 + 1/63.
        // doc 1 is first in one list only: 1/61.
        // 2/63 > 1/61, so doc 9 wins.
        let kw = [1u16, 5, 9];
        let sem = [7u16, 5, 9];
        let fused = fuse(&[&kw, &sem], K);
        let pos = |d: u16| fused.iter().position(|&x| x == d).unwrap();
        assert!(pos(9) < pos(1));
        // doc 5 (second in both) beats doc 9 (third in both)
        assert!(pos(5) < pos(9));
    }

    #[test]
    fn single_list_passthrough_order() {
        let kw = [3u16, 1, 2];
        assert_eq!(fuse(&[&kw], K), vec![3, 1, 2]);
    }

    #[test]
    fn empty_lists_ok() {
        assert!(fuse(&[], K).is_empty());
        let empty: [u16; 0] = [];
        assert!(fuse(&[&empty, &empty], K).is_empty());
    }

    #[test]
    fn fused_scores_are_the_rrf_sums() {
        let kw = [1u16, 5];
        let sem = [5u16];
        let scored = fuse_scored(&[&kw, &sem], K);
        let get = |d: u16| scored.iter().find(|(x, _)| *x == d).unwrap().1;
        assert!((get(5) - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-6);
        assert!((get(1) - 1.0 / 61.0).abs() < 1e-6);
    }
}
