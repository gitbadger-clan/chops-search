//! Minimal keyword engine: tf-idf over whole documents.
//!
//! This is the exact-token half of the hybrid — the one that nails `pydub`
//! when the vector side shatters it into subword confetti. It indexes the
//! *word-level* tokens (post-normalization, pre-WordPiece), so an
//! out-of-vocabulary word is a first-class term here even though the
//! semantic side deleted it.
//!
//! Deliberately not Fuse.js-shaped: no distance windows, no field-start
//! bias — a term matches wherever it appears in the post. That windowing
//! is precisely the bug that hid `pydub` at char 3,700 for years.

use std::collections::HashMap;

pub struct KeywordIndex {
    pub n_docs: u16,
    /// term → postings (doc id, term frequency), doc ids ascending.
    pub terms: HashMap<Box<str>, Vec<(u16, u16)>>,
}

impl KeywordIndex {
    /// Rank docs for pre-normalized query words. Score is
    /// Σ idf(term) · (1 + ln tf). Ties break on doc id.
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
            // BM25-style idf, floored at ~0 for terms in every doc.
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for &(doc, tf) in postings {
                *scores.entry(doc).or_insert(0.0) += idf * (1.0 + (tf as f32).ln());
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
        KeywordIndex { n_docs: 3, terms }
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
}
