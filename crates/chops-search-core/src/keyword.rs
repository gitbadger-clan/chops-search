//! Minimal keyword engine: BM25 over whole documents, with prefix
//! expansion on the term the user is still typing.
//!
//! This is the exact-token half of the hybrid — the one that nails `pydub`
//! when the vector side shatters it into subword confetti. It indexes the
//! *word-level* tokens (post-normalization, pre-WordPiece), so an
//! out-of-vocabulary word is a first-class term here even though the
//! semantic side deleted it.
//!
//! BM25 rather than idf · (1 + ln tf): without length normalization a long
//! document outscores an exact-title match on generic terms alone —
//! observed live, where a 9 KB post took kw#1 for "losing tasks when the
//! server dies" on the strength of its tf for "when" and "the", beating
//! the 300-byte page actually titled "Where Do the Tasks Go?".
//!
//! PREFIX EXPANSION. In an as-you-type box, a query is a complete phrase
//! plus one half-typed word. Scoring that last word exactly means
//! "chromiumox" matches nothing until the final character lands — results
//! blink out mid-word, which reads as broken. So the LAST word (only) also
//! matches terms it prefixes. Three constraints keep that from swamping
//! the real signal:
//!
//!   - a minimum prefix length, so "a" doesn't expand to a third of the
//!     vocabulary
//!   - a cap on expansions, taking the LOWEST-df matches: rare terms are
//!     the distinctive ones, and they're what the user is reaching for
//!   - a damping factor, so an expansion is worth strictly less than the
//!     term it expands to would be if typed in full
//!
//! Note the last word is expanded even when the query is "finished" (the
//! user hit Enter): the engine can't tell typing from submitting, and
//! damped expansion of a complete word is a small, bounded cost against a
//! large gain on every keystroke before it.
//!
//! Deliberately not Fuse.js-shaped: no distance windows, no field-start
//! bias — a term matches wherever it appears in the post. That windowing
//! is precisely the bug that hid `pydub` at char 3,700 for years.

use std::collections::HashMap;

/// Standard BM25 parameters. k1 caps term-frequency saturation; b sets
/// how hard document length bites (0 = none, 1 = full).
pub const K1: f32 = 1.2;
pub const B: f32 = 0.75;

/// Shortest trailing word that may expand. Two characters already matches
/// far too much of an English vocabulary to be informative.
pub const PREFIX_MIN_CHARS: usize = 3;
/// Most expansions considered, lowest-df first.
pub const PREFIX_MAX_EXPANSIONS: usize = 8;
/// Weight multiplier for an expanded term relative to an exact match.
pub const PREFIX_DAMP: f32 = 0.5;

pub struct KeywordIndex {
    pub n_docs: u16,
    /// term → postings (doc id, term frequency), doc ids ascending.
    pub terms: HashMap<Box<str>, Vec<(u16, u16)>>,
    /// Terms in sorted order, so a prefix range is a binary search. Built
    /// here rather than serialized: index.bin already stores them sorted,
    /// but the engine reads them into a map, and re-sorting once at load
    /// is cheaper than a second copy in the artifact.
    pub sorted: Vec<Box<str>>,
    /// Weighted length per doc (Σ tf over all terms), derived at
    /// construction — never serialized.
    pub dl: Vec<f32>,
    /// Mean of dl, floored at 1 so the ratio is always defined.
    pub avgdl: f32,
}

/// One scored query term, after prefix expansion has been resolved.
/// `weight` is 1.0 for a term the user typed and PREFIX_DAMP for one
/// reached by expansion.
#[derive(Debug, Clone)]
pub struct Term {
    pub text: Box<str>,
    pub weight: f32,
    /// True when this term came from expanding the trailing word.
    pub expanded: bool,
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
        let mut sorted: Vec<Box<str>> = terms.keys().cloned().collect();
        sorted.sort_unstable();
        KeywordIndex {
            n_docs,
            terms,
            sorted,
            dl,
            avgdl,
        }
    }

    /// Fraction of the query's potential idf mass that actually matched.
    ///
    /// Each distinct typed word contributes its idf to `potential` if it
    /// is a corpus term, else idf(0) — an unmatched word is evidence
    /// against keyword confidence at full strength. EXCEPTION: the
    /// trailing word, when it produced expansions, contributes nothing
    /// to potential. It is still being typed, and judging an incomplete
    /// word as a miss would gate type-ahead on small corpora, where
    /// idf(1)/idf(0) is too coarse for any fixed floor to separate
    /// mid-typing from junk. Its expansions still count in `matched`.
    ///
    /// When potential is zero — the query is nothing but an in-progress
    /// word — expansions are the only possible evidence: full confidence
    /// if they exist, zero if not.
    pub fn confidence(&self, query_words: &[&str], terms: &[Term]) -> f32 {
        let max_idf = self.idf(0);
        let has_expansions = terms.iter().any(|t| t.expanded);
        let last = query_words.last().copied();
        let mut seen: Vec<&str> = Vec::new();
        let mut potential = 0.0f32;
        for &w in query_words {
            if seen.contains(&w) {
                continue;
            }
            seen.push(w);
            potential += match self.terms.get(w) {
                Some(p) => self.idf(p.len()),
                None if Some(w) == last && has_expansions => 0.0,
                None => max_idf,
            };
        }
        let matched: f32 = terms
            .iter()
            .filter_map(|t| {
                self.terms
                    .get(&t.text)
                    .map(|p| t.weight * self.idf(p.len()))
            })
            .sum();
        if potential <= 0.0 {
            return if matched > 0.0 { 1.0 } else { 0.0 };
        }
        (matched / potential).min(1.0)
    }

    /// BM25 idf. Floors at ~0 for terms present in every document.
    pub fn idf(&self, df: usize) -> f32 {
        let n = self.n_docs as f32;
        let df = df as f32;
        ((n - df + 0.5) / (df + 0.5) + 1.0).ln()
    }

    /// One term's contribution to one document. Exposed so `chops-search query`
    /// reports the same numbers the ranker used — the duplicated formula
    /// in explain.rs drifted once already when BM25 landed.
    pub fn term_score(&self, doc: u16, tf: u16, idf: f32) -> f32 {
        let tf = tf as f32;
        let norm = K1 * (1.0 - B + B * self.dl[doc as usize] / self.avgdl);
        idf * (tf * (K1 + 1.0)) / (tf + norm)
    }

    /// Terms in `sorted` that begin with `p`, as a contiguous slice.
    /// Sorted order makes prefix matches adjacent, so this is two binary
    /// searches and no allocation.
    fn prefix_range(&self, p: &str) -> &[Box<str>] {
        let lo = self.sorted.partition_point(|t| t.as_ref() < p);
        let hi = lo + self.sorted[lo..].partition_point(|t| t.starts_with(p));
        &self.sorted[lo..hi]
    }

    /// Resolve query words into scored terms: every word at full weight,
    /// plus damped expansions of the last word. Duplicate words are
    /// collapsed (a repeated query term shouldn't count twice), and an
    /// expansion that duplicates a typed word is dropped rather than
    /// double-counted at a lower weight.
    pub fn resolve(&self, query_words: &[&str], expand_last: bool) -> Vec<Term> {
        let mut out: Vec<Term> = Vec::new();
        let mut seen: Vec<&str> = Vec::new();
        for &w in query_words {
            if seen.contains(&w) {
                continue;
            }
            seen.push(w);
            if self.terms.contains_key(w) {
                out.push(Term {
                    text: Box::from(w),
                    weight: 1.0,
                    expanded: false,
                });
            }
        }

        let Some(&last) = query_words.last() else {
            return out;
        };
        if !expand_last || last.chars().count() < PREFIX_MIN_CHARS {
            return out;
        }

        // Lowest df first: the rare completions are the informative ones.
        // A user typing "chromiumox" wants "chromiumoxide", not whichever
        // common word happens to sort first.
        let mut cands: Vec<(usize, &Box<str>)> = self
            .prefix_range(last)
            .iter()
            .filter(|t| t.as_ref() != last)
            .map(|t| (self.terms[t.as_ref()].len(), t))
            .collect();
        cands.sort_unstable_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(b.1)));

        for (_, t) in cands.into_iter().take(PREFIX_MAX_EXPANSIONS) {
            out.push(Term {
                text: t.clone(),
                weight: PREFIX_DAMP,
                expanded: true,
            });
        }
        out
    }

    /// Rank docs for pre-normalized query words, with prefix expansion on
    /// the trailing term. Ties break on doc id.
    pub fn rank(&self, query_words: &[&str]) -> Vec<u16> {
        self.rank_terms(&self.resolve(query_words, true))
    }

    /// Rank with expansion disabled — exact terms only.
    pub fn rank_exact(&self, query_words: &[&str]) -> Vec<u16> {
        self.rank_terms(&self.resolve(query_words, false))
    }

    /// Per-document scores for resolved terms, densely indexed by doc id.
    /// Dense rather than a HashMap for two reasons: f32 summation order
    /// over HashMap iteration is not deterministic, and native and wasm
    /// must rank identically; and the report needs the scores after
    /// ranking, which rank_terms used to throw away.
    pub fn score_terms(&self, terms: &[Term]) -> Vec<f32> {
        let mut scores = vec![0f32; self.n_docs as usize];
        for t in terms {
            let Some(postings) = self.terms.get(&t.text) else {
                continue;
            };
            let idf = self.idf(postings.len());
            for &(doc, tf) in postings {
                if let Some(s) = scores.get_mut(doc as usize) {
                    *s += t.weight * self.term_score(doc, tf, idf);
                }
            }
        }
        scores
    }

    /// Order docs by score descending, ties on doc id. Docs with no
    /// evidence (score 0) are excluded, matching the old HashMap behavior
    /// where they never entered the map.
    pub fn rank_from_scores(scores: &[f32]) -> Vec<u16> {
        let mut out: Vec<u16> = (0..scores.len() as u16)
            .filter(|&d| scores[d as usize] > 0.0)
            .collect();
        out.sort_unstable_by(|&a, &b| {
            scores[b as usize]
                .partial_cmp(&scores[a as usize])
                .unwrap_or(core::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        out
    }

    pub fn rank_terms(&self, terms: &[Term]) -> Vec<u16> {
        Self::rank_from_scores(&self.score_terms(terms))
    }
}

/// Longest compound worth indexing, in bytes. A 70-char kebab-case slug
/// is a URL, not a search term, and unbounded compounds would let one
/// pathological line mint arbitrarily large terms.
const MAX_COMPOUND: usize = 64;
/// Joiners that glue identifier compounds: data-chops-open, prefix_rows,
/// model.rows. Single joiner between alphanumeric runs only.
const JOINERS: &[char] = &['-', '_', '.'];

/// Word split for the KEYWORD index — deliberately not BertPreTokenizer.
/// Keyword search wants every non-alphanumeric to be a boundary, so
/// `25°C` yields "25" and "c" rather than one unsearchable term. Sharing
/// `Vocab::words` coupled keyword recall to BERT's symbol/punctuation
/// asymmetry, which exists for embedding parity and has no business
/// shaping an inverted index.
///
/// COMPOUNDS: identifiers joined by single `-`, `_`, or `.` between
/// alphanumeric runs are additionally emitted whole ("data-chops-open",
/// "prefix_rows", "v0.2.10"), joiners preserved. Without this, an
/// identifier query decomposes into its parts and any part that happens
/// to be corpus-common ("chops" on a chops-search docs site) turns the
/// ranking into stopword noise; the whole compound is a rare term with
/// the idf to match. Both the builder and the query path call this
/// function, so emission is symmetric by construction, and the sorted
/// term list picks compounds up for trailing-term prefix expansion —
/// typing "data-cho" completes into the attribute name.
///
/// Rules: a joiner must have alphanumerics on BOTH sides to count
/// (doubled, leading, and trailing joiners break the candidate), and
/// spans longer than MAX_COMPOUND are dropped. A compound always
/// contains a joiner, so it can never duplicate a part.
pub fn keyword_words(normalized: &str) -> Vec<&str> {
    let mut out = Vec::new();

    // Pass 1: parts. Every non-alphanumeric is a boundary.
    let mut start: Option<usize> = None;
    for (i, c) in normalized.char_indices() {
        if c.is_alphanumeric() {
            if start.is_none() {
                start = Some(i);
            }
        } else if let Some(s) = start.take() {
            out.push(&normalized[s..i]);
        }
    }
    if let Some(s) = start {
        out.push(&normalized[s..]);
    }

    // Pass 2: compounds. A candidate opens at an alphanumeric and
    // accumulates `alnum+ (joiner alnum+)*`; anything else flushes it.
    // `pending` is a joiner waiting for an alphanumeric to legitimize
    // it; `interior` counts joiners that got one; `alnum_end` trims a
    // dangling joiner off the emitted span ("model." emits nothing,
    // "model.rows" emits whole).
    let mut c_start: Option<usize> = None;
    let mut alnum_end = 0usize;
    let mut pending = false;
    let mut interior = 0usize;

    for (i, c) in normalized.char_indices() {
        if c.is_alphanumeric() {
            if c_start.is_none() {
                c_start = Some(i);
                interior = 0;
            }
            if pending {
                interior += 1;
                pending = false;
            }
            alnum_end = i + c.len_utf8();
        } else if JOINERS.contains(&c) && c_start.is_some() && !pending {
            pending = true;
        } else {
            // Boundary, doubled joiner, or joiner with nothing before it.
            if let Some(s) = c_start.take()
                && interior > 0
                && alnum_end - s <= MAX_COMPOUND
            {
                out.push(&normalized[s..alnum_end]);
            }
            pending = false;
            interior = 0;
        }
    }
    if let Some(s) = c_start
        && interior > 0
        && alnum_end - s <= MAX_COMPOUND
    {
        out.push(&normalized[s..alnum_end]);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::score::KW_CONFIDENCE;

    /// Corpus shaped like the docs site: a term for everything common, a
    /// couple of mid-frequency words, and one rare term ("routinely")
    /// reachable only by prefix-expanding "routine".
    fn confidence_index() -> KeywordIndex {
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        terms.insert(Box::from("the"), vec![(0, 9), (1, 7), (2, 8), (3, 6)]);
        terms.insert(Box::from("to"), vec![(0, 5), (1, 4), (2, 3), (3, 5)]);
        terms.insert(Box::from("what"), vec![(0, 2), (1, 1), (2, 2)]);
        terms.insert(Box::from("when"), vec![(0, 1), (1, 2), (3, 1)]);
        terms.insert(Box::from("happens"), vec![(1, 1)]);
        terms.insert(Box::from("process"), vec![(0, 2), (2, 1)]);
        terms.insert(Box::from("routinely"), vec![(2, 1)]);
        terms.insert(Box::from("search"), vec![(0, 6), (1, 5), (2, 4), (3, 7)]);
        KeywordIndex::new(4, terms)
    }

    fn index() -> KeywordIndex {
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        terms.insert(Box::from("pydub"), vec![(2, 3)]);
        terms.insert(Box::from("the"), vec![(0, 10), (1, 8), (2, 12)]);
        terms.insert(Box::from("bloom"), vec![(0, 2), (1, 1)]);
        KeywordIndex::new(3, terms)
    }

    fn prefix_index() -> KeywordIndex {
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        // doc0 chromiumoxide (rare), doc1 chromium (less rare),
        // doc2 chrome (common-ish), all sharing the "chro" prefix.
        terms.insert(Box::from("chromiumoxide"), vec![(0, 4)]);
        terms.insert(Box::from("chromium"), vec![(1, 4), (2, 1)]);
        terms.insert(Box::from("chrome"), vec![(0, 1), (1, 1), (2, 4)]);
        terms.insert(Box::from("filler"), vec![(0, 20), (1, 20), (2, 20)]);
        KeywordIndex::new(3, terms)
    }

    #[test]
    fn rare_term_dominates() {
        assert_eq!(index().rank(&["pydub", "the"])[0], 2);
    }

    #[test]
    fn unknown_term_scores_nothing() {
        assert!(index().rank(&["zzz"]).is_empty());
    }

    #[test]
    fn repeated_query_term_counted_once() {
        let idx = index();
        assert_eq!(idx.rank(&["bloom"]), idx.rank(&["bloom", "bloom"]));
    }

    #[test]
    fn length_normalization_beats_tf_inflation() {
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        terms.insert(Box::from("tasks"), vec![(0, 2), (1, 6)]);
        terms.insert(Box::from("filler"), vec![(0, 4), (1, 294)]);
        let idx = KeywordIndex::new(2, terms);
        assert_eq!(idx.rank_exact(&["tasks"])[0], 0);
    }

    #[test]
    fn tf_saturates() {
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        terms.insert(Box::from("x"), vec![(0, 3), (1, 100)]);
        terms.insert(Box::from("pad"), vec![(0, 97)]);
        let idx = KeywordIndex::new(2, terms);
        assert_eq!(idx.rank_exact(&["x"])[0], 1);
        let s0 = idx.term_score(0, 3, 1.0);
        let s1 = idx.term_score(1, 100, 1.0);
        assert!(s1 / s0 < 1.5, "saturation failed: {s1} vs {s0}");
    }

    #[test]
    fn empty_index_is_sane() {
        let idx = KeywordIndex::new(0, HashMap::new());
        assert!(idx.rank(&["anything"]).is_empty());
        assert_eq!(idx.avgdl, 1.0);
    }

    // ---- prefix expansion --------------------------------------------

    #[test]
    fn partial_word_finds_documents() {
        // The headline behavior: mid-word, results exist.
        let idx = prefix_index();
        assert!(!idx.rank(&["chromiumox"]).is_empty());
        assert!(idx.rank_exact(&["chromiumox"]).is_empty());
        assert_eq!(idx.rank(&["chromiumox"])[0], 0);
    }

    #[test]
    fn prefix_range_is_contiguous_and_correct() {
        let idx = prefix_index();
        let got: Vec<&str> = idx
            .prefix_range("chromium")
            .iter()
            .map(|t| t.as_ref())
            .collect();
        assert_eq!(got, vec!["chromium", "chromiumoxide"]);
        assert!(idx.prefix_range("zzz").is_empty());
        // A prefix matching nothing between two real terms.
        assert!(idx.prefix_range("chromiv").is_empty());
    }

    #[test]
    fn expansion_prefers_rare_completions() {
        let idx = prefix_index();
        let terms = idx.resolve(&["chro"], true);
        let expanded: Vec<&str> = terms
            .iter()
            .filter(|t| t.expanded)
            .map(|t| t.text.as_ref())
            .collect();
        // df order: chromiumoxide(1), chromium(2), chrome(3).
        assert_eq!(expanded, vec!["chromiumoxide", "chromium", "chrome"]);
    }

    #[test]
    fn exact_match_outranks_prefix_match() {
        // Typing "chrome" in full must favor the doc that's about chrome,
        // not the rare completion the expansion also reaches.
        let idx = prefix_index();
        assert_eq!(idx.rank(&["chrome"])[0], 2);
    }

    #[test]
    fn typed_term_is_not_double_counted_as_expansion() {
        let idx = prefix_index();
        let terms = idx.resolve(&["chromium"], true);
        let n_chromium = terms
            .iter()
            .filter(|t| t.text.as_ref() == "chromium")
            .count();
        assert_eq!(n_chromium, 1);
        assert!(
            !terms
                .iter()
                .any(|t| t.text.as_ref() == "chromium" && t.expanded)
        );
    }

    #[test]
    fn only_the_last_word_expands() {
        let idx = prefix_index();
        let terms = idx.resolve(&["chro", "filler"], true);
        // "chro" is not last, so nothing expands from it; "filler" is
        // last but has no completions beyond itself.
        assert!(terms.iter().all(|t| !t.expanded));
        assert_eq!(terms.len(), 1); // only "filler" is a real term
    }

    #[test]
    fn short_prefixes_do_not_expand() {
        let idx = prefix_index();
        assert!(idx.resolve(&["ch"], true).iter().all(|t| !t.expanded));
        assert!(idx.resolve(&["chr"], true).iter().any(|t| t.expanded));
    }

    #[test]
    fn expansion_count_is_capped() {
        let mut terms: HashMap<Box<str>, Vec<(u16, u16)>> = HashMap::new();
        for i in 0..50 {
            terms.insert(Box::from(format!("prefix{i:02}").as_str()), vec![(0, 1)]);
        }
        let idx = KeywordIndex::new(1, terms);
        let n = idx
            .resolve(&["prefix"], true)
            .iter()
            .filter(|t| t.expanded)
            .count();
        assert_eq!(n, PREFIX_MAX_EXPANSIONS);
    }

    #[test]
    fn keyword_words_splits_on_symbols() {
        assert_eq!(keyword_words("25°c"), vec!["25", "c"]);
        assert_eq!(keyword_words("a×b"), vec!["a", "b"]);
        assert_eq!(keyword_words("hello world"), vec!["hello", "world"]);
        assert!(keyword_words("...").is_empty());
    }
    #[test]
    fn compounds_emit_whole_and_parts() {
        assert_eq!(
            keyword_words("data-chops-open"),
            vec!["data", "chops", "open", "data-chops-open"]
        );
        assert_eq!(
            keyword_words("prefix_rows"),
            vec!["prefix", "rows", "prefix_rows"]
        );
        assert_eq!(keyword_words("v0.2.10"), vec!["v0", "2", "10", "v0.2.10"]);
    }

    #[test]
    fn dangling_and_doubled_joiners_break_the_candidate() {
        // Trailing joiner: sentence-final "model." is prose, not an identifier.
        assert_eq!(keyword_words("model."), vec!["model"]);
        // Doubled joiner kills the left candidate; the right side restarts.
        assert_eq!(keyword_words("a--b-c"), vec!["a", "b", "c", "b-c"]);
        // Leading joiner is not a compound opener (CLI flags still work out:
        // "--min-cos" emits the useful "min-cos").
        assert_eq!(keyword_words("--min-cos"), vec!["min", "cos", "min-cos"]);
    }

    #[test]
    fn non_joiner_symbols_still_split_without_compounding() {
        assert_eq!(keyword_words("25°c"), vec!["25", "c"]);
    }

    #[test]
    fn oversized_compounds_are_dropped() {
        let long = ["part"; 20].join("-"); // 99 bytes of kebab
        let words = keyword_words(&long);
        assert_eq!(words.iter().filter(|w| w.contains('-')).count(), 0);
        assert_eq!(words.len(), 20);
    }

    #[test]
    fn compound_emission_is_query_build_symmetric() {
        // The property the whole change rests on: both sides tokenize alike.
        let doc = keyword_words("wire data-chops-open onto the trigger");
        let q = keyword_words("data-chops-open");
        assert!(doc.contains(&"data-chops-open"));
        assert_eq!(q.last(), Some(&"data-chops-open"));
    }

    #[test]
    fn expansion_only_multiword_query_is_suppressed() {
        // toddler-shape: three typed words, none a corpus term, one
        // damped expansion of the trailing word. A lone prefix expansion
        // cannot carry a 3-word query.
        let idx = confidence_index();
        let words = ["toddler", "bedtime", "routine"];
        let terms = idx.resolve(&words, true);
        assert!(!terms.is_empty(), "expected the routinely expansion");
        assert!(terms.iter().all(|t| t.expanded));
        assert!(idx.confidence(&words, &terms) < KW_CONFIDENCE);
    }

    #[test]
    fn stopword_matches_do_not_carry_a_query() {
        // unicow-shape: the glue words match, every discriminating word
        // ("queued", "jobs", "dies") misses at max idf.
        let idx = confidence_index();
        let words = [
            "what", "happens", "to", "queued", "jobs", "when", "the", "process", "dies",
        ];
        let terms = idx.resolve(&words, true);
        assert!(idx.confidence(&words, &terms) < KW_CONFIDENCE);
    }

    #[test]
    fn single_word_typing_survives_the_gate() {
        // Mid-type: one typed word, unmatched, rare completions at
        // PREFIX_DAMP. The floor is chosen to sit under this ratio.
        let idx = prefix_index();
        let words = ["chromiumox"];
        let terms = idx.resolve(&words, true);
        assert!(terms.iter().all(|t| t.expanded));
        assert!(idx.confidence(&words, &terms) >= KW_CONFIDENCE);
    }

    #[test]
    fn fully_matched_query_is_confident() {
        let idx = index();
        let words = ["pydub"];
        let terms = idx.resolve(&words, true);
        assert!(idx.confidence(&words, &terms) > 0.9);
    }

    #[test]
    fn confidence_of_empty_query_is_zero() {
        let idx = index();
        assert_eq!(idx.confidence(&[], &[]), 0.0);
    }

    #[test]
    fn trailing_expansion_does_not_mask_earlier_misses() {
        // "toddler routin": the trailing word is being typed and expands,
        // but "toddler" is a completed miss at max idf — one live
        // expansion must not launder a query whose other words all fail.
        let idx = confidence_index();
        let words = ["toddler", "routin"];
        let terms = idx.resolve(&words, true);
        assert!(terms.iter().any(|t| t.expanded));
        assert!(idx.confidence(&words, &terms) < KW_CONFIDENCE);
    }
}
