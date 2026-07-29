//! BERT WordPiece, specialized to how model2vec ("potion") actually uses it.
//!
//! Three deliberate deviations from a stock BERT tokenizer, all of which
//! silently poison the query vector if you get them wrong:
//!
//! 1. **No [CLS]/[SEP].** model2vec calls the tokenizer with
//!    add_special_tokens=False; adding the markers averages two vectors
//!    into the query that shouldn't be there.
//! 2. **Unknown words are DELETED, not mapped to [UNK].** A word that
//!    can't be fully tokenized drops out of the sequence entirely.
//! 3. **Accents are stripped.** strip_accents is null in the config, which
//!    inherits from lowercase=true. "café" tokenizes as "cafe".

use std::collections::HashMap;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

pub struct Vocab {
    map: HashMap<Box<str>, u32>,
    /// Longest vocab entry in chars (## prefix not counted), bounds the
    /// longest-match scan.
    max_len: usize,
}

impl Vocab {
    /// Build from tokens ordered by id (index == token id == matrix row).
    pub fn from_tokens<S: AsRef<str>>(tokens: &[S]) -> Self {
        let mut map = HashMap::with_capacity(tokens.len());
        let mut max_len = 1;
        for (id, tok) in tokens.iter().enumerate() {
            let t = tok.as_ref();
            let plain_len = t.strip_prefix("##").unwrap_or(t).chars().count();
            if plain_len > max_len {
                max_len = plain_len;
            }
            map.insert(Box::from(t), id as u32);
        }
        Vocab { map, max_len }
    }

    pub fn len(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Lowercase + NFD + drop combining marks. Matches HuggingFace's
    /// BertNormalizer with lowercase=true, strip_accents=None (inherits on).
    pub fn normalize(text: &str) -> String {
        let mut out = String::with_capacity(text.len());
        for c in text.nfd() {
            if is_combining_mark(c) {
                continue;
            }
            for lc in c.to_lowercase() {
                out.push(lc);
            }
        }
        out
    }

    /// BERT basic-tokenizer word split over *normalized* text: runs of
    /// alphanumerics are words, each punctuation char is its own word,
    /// whitespace and other separators are dropped.
    pub fn words(normalized: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut start: Option<usize> = None;
        for (i, c) in normalized.char_indices() {
            if c.is_alphanumeric() {
                if start.is_none() {
                    start = Some(i);
                }
            } else {
                if let Some(s) = start.take() {
                    out.push(&normalized[s..i]);
                }
                if !c.is_whitespace() && !c.is_control() {
                    // punctuation / symbol: its own token
                    out.push(&normalized[i..i + c.len_utf8()]);
                }
            }
        }
        if let Some(s) = start {
            out.push(&normalized[s..]);
        }
        out
    }

    /// Longest-match-first WordPiece over one word. Returns None when the
    /// word cannot be fully tokenized — the caller must DROP it (gotcha 2),
    /// never substitute.
    fn wordpiece(&self, word: &str) -> Option<Vec<u32>> {
        let chars: Vec<char> = word.chars().collect();
        let mut ids = Vec::new();
        let mut start = 0usize;
        let mut piece = String::new();
        while start < chars.len() {
            let mut end = chars.len().min(start + self.max_len);
            let mut hit: Option<(u32, usize)> = None;
            while end > start {
                piece.clear();
                if start > 0 {
                    piece.push_str("##");
                }
                for &c in &chars[start..end] {
                    piece.push(c);
                }
                if let Some(&id) = self.map.get(piece.as_str()) {
                    hit = Some((id, end));
                    break;
                }
                end -= 1;
            }
            match hit {
                Some((id, e)) => {
                    ids.push(id);
                    start = e;
                }
                None => return None,
            }
        }
        Some(ids)
    }

    /// Full pipeline: normalize → split → wordpiece each word, deleting
    /// untokenizable words. No special tokens (gotcha 1). An empty result
    /// is a legitimate outcome for an all-out-of-vocabulary query.
    pub fn tokenize(&self, text: &str) -> Vec<u32> {
        let norm = Self::normalize(text);
        let mut ids = Vec::new();
        for w in Self::words(&norm) {
            if let Some(mut piece_ids) = self.wordpiece(w) {
                ids.append(&mut piece_ids);
            }
        }
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab() -> Vocab {
        // ids:      0      1      2       3     4      5     6
        Vocab::from_tokens(&["the", "bloom", "filter", "##s", "cafe", ".", "un"])
    }

    #[test]
    fn longest_match_and_continuation() {
        let v = vocab();
        assert_eq!(v.tokenize("the filters"), vec![0, 2, 3]);
    }

    #[test]
    fn unknown_word_is_deleted_not_unked() {
        let v = vocab();
        // "pydub" is untokenizable → dropped entirely; neighbors survive.
        assert_eq!(v.tokenize("the pydub filter"), vec![0, 2]);
    }

    #[test]
    fn partial_match_still_deletes_whole_word() {
        let v = vocab();
        // "unfoo": "un" matches but "##foo" doesn't → whole word dropped,
        // NOT a dangling [un] fragment.
        assert_eq!(v.tokenize("unfoo the"), vec![0]);
    }

    #[test]
    fn accents_stripped_and_lowercased() {
        let v = vocab();
        assert_eq!(v.tokenize("Café"), vec![4]);
    }

    #[test]
    fn punctuation_is_its_own_token() {
        let v = vocab();
        assert_eq!(v.tokenize("the."), vec![0, 5]);
    }

    #[test]
    fn all_oov_yields_empty() {
        let v = vocab();
        assert!(v.tokenize("zzz qqq").is_empty());
    }
}
