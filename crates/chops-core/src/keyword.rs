//! Minimal keyword engine: BM25 over whole documents.
//!
//! This is the exact-token half of the hybrid — the one that nails `pydub`
//! when the vector side shatters it into subword confetti. It indexes the
//! *word-level* tokens (post-normalization, pre-WordPiece), so an
//! out-of-vocabulary word is a first-class term here even though the
//! semantic side deleted it.
//!
//! Why BM25 and not the earlier idf · (1 + ln tf): no length
//! normalization means a long document outscores an exact-title match on
//! generic terms alone — observed live, where a 9 KB post took kw#1 for
//! "losing tasks when the server dies" on the strength of its tf for
//! "when" and "the", beating the 300-byte page actually titled "Where Do
//! the Tasks Go?". BM25's saturation term caps what raw tf can buy, and
//! its length term makes a mention in a short doc worth more than the
//! same mention diluted across a long one. This is the keyword-side twin
//! of the chunk-count lottery on the semantic side.
//!
//! Document lengths are DERIVED from the postings (Σ tf per doc) rather
//! than stored, so the artifact format is unchanged — index.bin bytes are
//! identical before and after this change. "Length" is therefore weighted
//! length (title/tag weights inflate it slightly); that's fine, it's the
//! same corpus-consistent quantity for every doc.
//!
//! Deliberately not Fuse.js-shaped: no distance windows, no field-start
//! bias — a term matches wherever it appears in the post. That windowing
//! is precisely the bug that hid `pydub` at char 3,700 for years.

use std::collections::HashMap;

/// Standard BM25 parameters. k1 caps term-frequency saturation; b sets
/// how hard document length bites (0 = none, 1 = full).
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

pub struct KeywordIndex {
    pub n_docs: u16,
    /// term → postings (doc id, term frequency), doc ids ascending.
    pub terms: HashMap<Box<str>, Vec<(u16, u16)>>,
    /// Weighted length per doc (Σ tf over all terms), derived at
    /// construction — never serialized.
    pub dl: Vec<f32>,
    /// Mean of dl, floored at 1 so the ratio is always defined.
    pub avgdl: f32,
}

impl KeywordIndex {
    pub fn new(n_docs: u16, terms: HashMap<Box<str>, Vec<(u16, u16)>>) -> Self {
        let mut dl = vec![0f32; n_docs as usize];
        for postings in terms.values() {
            for &(doc, tf) in postings {
                if let Some(d) = dl.get_mut(doc as usize) {
                    *d += tf as f32;
                }
            }
        }
        let avgdl = if n_docs == 0 {
            1.0
        } else {
            (dl.iter().sum::<f32>() / n_docs as f32).max(1.0)
        };
        KeywordIndex {
            n_docs,
            terms,
            dl,
            avgdl,
        }
    }

    /// Rank docs for pre-normalized query words. Score is Σ over terms of
    /// idf · (tf·(k1+1)) / (tf + k1·(1 − b + b·dl/avgdl)). Ties break on
    /// doc id.
    pub fn rank(&self, query_words: &[&str]) -> Vec<u16> {
        let n = self.n_docs as f32;
        let mut scores: HashMap<u16, f32> = HashMap::new();
        let mut seen: Vec<&str> = Vec::new();
        for &w in query_words {
            if seen.contains(&w) {
                continue; // don't double-count repeated query terms
            }
            seen.push(w);
            let Some(postings) = self.terms.get(w) else {
                continue;
            };
            let df = postings.len() as f32;
            // BM25 idf, floored at ~0 for terms in every doc.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc, tf) in postings {
                let tf = tf as f32;
                let norm = K1 * (1.0 - B + B * self.dl[doc as usize] / self.avgdl);
                *scores.entry(doc).or_insert(0.0) += idf * (tf * (K1 + 1.0)) / (tf + norm);
            }
        }
        let mut out: Vec<u16> = scores.keys().copied().collect();
        out.sort_unstable_by(|&a, &b| {
            scores[&b]
                .partial_cmp(&scores[&a])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index() -> KeywordIndex {
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        // "pydub" only in doc 2; "the" in every doc.
        terms.insert(Box::from("pydub"), vec![(2, 3)]);
        terms.insert(Box::from("the"), vec![(0, 10), (1, 8), (2, 12)]);
        terms.insert(Box::from("bloom"), vec![(0, 2), (1, 1)]);
        KeywordIndex::new(3, terms)
    }

    #[test]
    fn rare_term_dominates() {
        let idx = index();
        let ranked = idx.rank(&["pydub", "the"]);
        assert_eq!(ranked[0], 2);
    }

    #[test]
    fn unknown_term_scores_nothing() {
        let idx = index();
        assert!(idx.rank(&["zzz"]).is_empty());
    }

    #[test]
    fn repeated_query_term_counted_once() {
        let idx = index();
        assert_eq!(idx.rank(&["bloom"]), idx.rank(&["bloom", "bloom"]));
    }

    #[test]
    fn length_normalization_beats_tf_inflation() {
        // The observed failure, in miniature: a long doc mentions the term
        // more times in absolute terms, but a short doc that's actually
        // ABOUT it must win. doc 0: tiny page, "tasks" tf 2 of dl 6.
        // doc 1: long page, "tasks" tf 6 of dl 300.
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        terms.insert(Box::from("tasks"), vec![(0, 2), (1, 6)]);
        terms.insert(Box::from("filler"), vec![(0, 4), (1, 294)]);
        let idx = KeywordIndex::new(2, terms);
        assert_eq!(idx.rank(&["tasks"])[0], 0);
    }

    #[test]
    fn tf_saturates() {
        // Under 1+ln(tf) doc 1's tf=100 would dwarf doc 0's tf=3 (~5.6 vs
        // ~2.1). Under BM25 with equal lengths both approach the k1+1
        // asymptote; the gap must be small, not 2.5x.
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        terms.insert(Box::from("x"), vec![(0, 3), (1, 100)]);
        // Equalize lengths so only saturation differs.
        terms.insert(Box::from("pad"), vec![(0, 97)]);
        let idx = KeywordIndex::new(2, terms);
        // doc 1 still wins (higher tf) …
        assert_eq!(idx.rank(&["x"])[0], 1);
        // … but by saturation, not multiples: recompute both scores.
        let score = |tf: f32, dl: f32, avgdl: f32| {
            (tf * (K1 + 1.0)) / (tf + K1 * (1.0 - B + B * dl / avgdl))
        };
        let s0 = score(3.0, idx.dl[0], idx.avgdl);
        let s1 = score(100.0, idx.dl[1], idx.avgdl);
        assert!(s1 / s0 < 1.5, "saturation failed: {s1} vs {s0}");
    }

    #[test]
    fn empty_index_is_sane() {
        let idx = KeywordIndex::new(0, HashMap::new());
        assert!(idx.rank(&["anything"]).is_empty());
        assert_eq!(idx.avgdl, 1.0);
    }
}
