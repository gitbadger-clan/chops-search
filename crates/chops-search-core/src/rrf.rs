//! Reciprocal Rank Fusion.
//!
//! Both engines' scores are thrown away entirely; only ranks are kept.
//! fused(doc) = Σ over engines of w / (k + rank), rank starting at 1,
//! k conventionally 60. The k flattens the top of the curve so "both
//! engines quite liked this" beats "one engine loved it".
//!
//! This also makes late arrival graceful: a keyword-only list is just RRF
//! with one engine, and when the semantic list shows up the fusion is
//! purely additive — no rescaling problem, because there are no scales.
//!
//! WEIGHTS. Plain RRF treats both engines as equally credible on every
//! query, which is wrong in a specific, observable way. For a query on a
//! unique compound term, the keyword engine ranks the right document
//! first on df-1 evidence while the semantic engine ranks a topically
//! adjacent page first, and unweighted fusion prefers the semantic
//! winner:
//!
//!   1/(60+2) + 1/(60+1)  >  1/(60+1) + 1/(60+3)
//!
//! Both engines behaved correctly; the arithmetic resolved them the wrong
//! way. A per-list weight lets the caller say how much this particular
//! query's keyword evidence is worth, which the engine already knows as
//! `kw_confidence`.
//!
//! The crossover is worth knowing before tuning: for the shape above,
//! w/61 + 1/63 > w/62 + 1/61 needs w > 1.97. So a keyword weight has to
//! reach roughly 2 to overturn a semantic first place two ranks up, and
//! weights below that change nothing at all. k = 60 flattens the curve
//! deliberately, and the weight has to climb out of that flattening.

use std::collections::HashMap;

pub const K: f32 = 60.0;

/// One engine's ranked output and how much its opinion counts.
///
/// A tuple rather than two parallel slices: the types differ, so a
/// transposed argument will not compile, and a list can never be paired
/// with the wrong weight.
pub type Contribution<'a> = (&'a [u16], f32);

/// Fused (doc, RRF score) pairs, score descending, ties on doc id. The
/// scores exist for the report layer — explain prints contributions from
/// here instead of restating the formula.
///
/// A weight of 0.0 removes a list from the fusion entirely, which is the
/// same as not passing it: worth relying on, because it makes "disable
/// this engine" expressible without a branch at the call site.
pub fn fuse_scored(lists: &[Contribution<'_>], k: f32) -> Vec<(u16, f32)> {
    let mut scores: HashMap<u16, f32> = HashMap::new();
    for (list, weight) in lists {
        if *weight == 0.0 {
            continue;
        }
        for (i, &doc) in list.iter().enumerate() {
            *scores.entry(doc).or_insert(0.0) += weight / (k + (i + 1) as f32);
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

/// Fuse ranked lists of doc ids at equal weight. Ties break on doc id for
/// determinism.
pub fn fuse(lists: &[&[u16]], k: f32) -> Vec<u16> {
    let weighted: Vec<Contribution<'_>> = lists.iter().map(|l| (*l, 1.0)).collect();
    fuse_scored(&weighted, k)
        .into_iter()
        .map(|(d, _)| d)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(scored: &[(u16, f32)]) -> Vec<u16> {
        scored.iter().map(|(d, _)| *d).collect()
    }

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
        let scored = fuse_scored(&[(&kw, 1.0), (&sem, 1.0)], K);
        let get = |d: u16| scored.iter().find(|(x, _)| *x == d).unwrap().1;
        assert!((get(5) - (1.0 / 62.0 + 1.0 / 61.0)).abs() < 1e-6);
        assert!((get(1) - 1.0 / 61.0).abs() < 1e-6);
    }

    #[test]
    fn unit_weights_are_the_unweighted_behaviour() {
        // The tripwire for shipping this inert: at weight 1.0 every score
        // must be bit-identical to plain RRF, or "rrf_alpha defaults to
        // 0" stops meaning "nothing changed".
        let kw = [4u16, 2, 7];
        let sem = [2u16, 9];
        let weighted = fuse_scored(&[(&kw, 1.0), (&sem, 1.0)], K);
        assert_eq!(ids(&weighted), fuse(&[&kw, &sem], K));
        for (doc, score) in &weighted {
            let expect: f32 = [&kw[..], &sem[..]]
                .iter()
                .filter_map(|l| l.iter().position(|d| d == doc))
                .map(|i| 1.0 / (K + (i + 1) as f32))
                .sum();
            assert_eq!(*score, expect, "doc {doc}");
        }
    }

    #[test]
    fn a_weighted_keyword_first_place_overturns_a_semantic_one() {
        // The model2vec-rs shape: doc 0 is kw#1 and sem#3, doc 1 is kw#2
        // and sem#1. Unweighted, doc 1 wins on the arithmetic in the
        // module header. Weighting the keyword list past ~1.97 flips it,
        // which is the entire point of the knob.
        let kw = [0u16, 1];
        let sem = [1u16, 7, 0];

        assert_eq!(ids(&fuse_scored(&[(&kw, 1.0), (&sem, 1.0)], K))[0], 1);
        assert_eq!(ids(&fuse_scored(&[(&kw, 1.9), (&sem, 1.0)], K))[0], 1);
        assert_eq!(ids(&fuse_scored(&[(&kw, 2.0), (&sem, 1.0)], K))[0], 0);
    }

    #[test]
    fn zero_weight_removes_a_list_entirely() {
        // Not merely "contributes little": a doc appearing only in the
        // zero-weighted list must not appear at all, so a caller can
        // disable an engine without branching.
        let kw = [3u16, 8];
        let sem = [5u16];
        let fused = fuse_scored(&[(&kw, 0.0), (&sem, 1.0)], K);
        assert_eq!(ids(&fused), vec![5]);
        assert_eq!(ids(&fuse_scored(&[(&kw, 0.0)], K)), Vec::<u16>::new());
    }

    #[test]
    fn weighting_does_not_reorder_within_one_list() {
        // A weight scales every rank in its list by the same factor, so
        // it can only change how the list trades against another one,
        // never the order inside it.
        let kw = [3u16, 1, 2];
        for w in [0.5f32, 1.0, 4.0] {
            assert_eq!(ids(&fuse_scored(&[(&kw, w)], K)), vec![3, 1, 2], "w {w}");
        }
    }

    #[test]
    fn ties_still_break_on_doc_id() {
        // Equal contributions from symmetric lists: the order must not
        // depend on HashMap iteration, or native and wasm can disagree.
        let a = [5u16, 2];
        let b = [2u16, 5];
        assert_eq!(ids(&fuse_scored(&[(&a, 1.0), (&b, 1.0)], K)), vec![2, 5]);
    }
}
