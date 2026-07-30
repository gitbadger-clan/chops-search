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
//!
//! `normalize` reproduces HuggingFace's BertNormalizer in its own order —
//! clean_text → handle_chinese_chars → strip_accents → lowercase — and
//! `words` reproduces BertPreTokenizer. Both were originally written for
//! Latin text only and diverged three ways on everything else; see
//! tests/tokenizer_parity.rs, which sweeps codepoints against the real
//! tokenizer rather than trusting hand-picked fixtures.
//!
//! Known remaining divergence: BERT's strip_accents removes category Mn
//! only, while `is_combining_mark` also covers Mc and Me. Indic spacing
//! marks (Devanagari matras) and the Hangul tone marks U+302E/U+302F are
//! therefore dropped where HF keeps them — where HF would fuse them into
//! the word and delete the whole word as untokenizable, chops keeps the
//! base characters. Fixing it needs an Mn-only category table in the wasm
//! blob; the parity survey reports the gap instead of hiding it.

use std::collections::HashMap;
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;
/// HF WordPiece refuses words longer than this, emitting [UNK] without
/// attempting to segment. Since chops deletes rather than [UNK]s, the
/// equivalent is returning None. Value is potion's
/// `model.max_input_chars_per_word`; not read from the JSON because Vocab
/// is built from a token list, not the file.
const MAX_INPUT_CHARS_PER_WORD: usize = 100;
pub struct Vocab {
    map: HashMap<Box<str>, u32>,
    /// Longest vocab entry in chars (## prefix not counted), bounds the
    /// longest-match scan.
    max_len: usize,
}

/// BERT's `_is_chinese_char`: the CJK *ideograph* blocks, and only those.
/// Hiragana, katakana, and hangul are deliberately absent — HF does not
/// space those out, so neither may we, or Japanese and Korean tokenize
/// differently from the reference.
fn is_cjk_ideograph(c: char) -> bool {
    matches!(c as u32,
        0x4E00..=0x9FFF
        | 0x3400..=0x4DBF
        | 0xF900..=0xFAFF
        | 0x20000..=0x2A6DF
        | 0x2A700..=0x2B73F
        | 0x2B740..=0x2B81F
        | 0x2B820..=0x2CEAF
        | 0x2F800..=0x2FA1F
    )
}

/// Characters `clean_text` deletes outright: category Cc (minus the three
/// whitespace escapes, which become spaces) and category Cf. Cf matters
/// more than it looks — a zero-width joiner inside an emoji sequence, or a
/// soft hyphen inside a word, changes the word boundary if left in.
///
/// The Cf list is the commonly-occurring subset rather than the full
/// category; the parity survey flags anything missed.
fn is_deleted_by_clean_text(c: char) -> bool {
    if matches!(c, '\t' | '\n' | '\r') {
        return false;
    }
    if c.is_control() {
        return true; // Cc
    }
    matches!(c as u32,
        0x00AD                  // soft hyphen
        | 0x0600..=0x0605       // Arabic number signs
        | 0x061C
        | 0x06DD
        | 0x070F
        | 0x180E
        | 0x200B..=0x200F       // ZWSP, ZWNJ, ZWJ, LRM, RLM
        | 0x202A..=0x202E       // bidi embedding
        | 0x2060..=0x2064       // word joiner, invisible operators
        | 0x2066..=0x206F
        | 0xFEFF                // BOM
        | 0xFFF9..=0xFFFB
        | 0xE000..=0xF8FF       // Co, private use
        | 0xE0001 | 0xE0020..=0xE007F  // Cf, tag characters
    )
}

/// BERT's `_is_punctuation`: every printable non-alphanumeric ASCII char,
/// plus Unicode category P. Category S (math, currency, modifier symbols,
/// emoji) is NOT punctuation and stays glued to its word — that asymmetry
/// is the whole reason `hello😅` behaves differently from `hello!`.
///
/// The non-ASCII half is a range table over the punctuation blocks that
/// occur in real prose, not a full category lookup. tests/tokenizer_parity
/// asserts exactness over these blocks and surveys the rest.
fn is_bert_punctuation(c: char) -> bool {
    if c.is_ascii() {
        return c.is_ascii_punctuation();
    }
    matches!(c as u32,
        0x00A1 | 0x00A7 | 0x00AB | 0x00B6 | 0x00B7 | 0x00BB | 0x00BF
        | 0x05BE | 0x05C0 | 0x05C3 | 0x05C6 | 0x05F3 | 0x05F4   // Hebrew
        | 0x060C | 0x060D | 0x061B | 0x061E | 0x061F | 0x06D4   // Arabic
        | 0x0964 | 0x0965                                        // danda
        | 0x2010..=0x2027
        | 0x2030..=0x2043
        | 0x2045..=0x2051
        | 0x2053..=0x205E
        | 0x207D | 0x207E | 0x208D | 0x208E
        | 0x2308..=0x230B
        | 0x2329 | 0x232A
        | 0x2768..=0x2775
        | 0x27C5 | 0x27C6
        | 0x27E6..=0x27EF
        | 0x2983..=0x2998
        | 0x29D8..=0x29DB
        | 0x29FC | 0x29FD
        | 0x2CF9..=0x2CFC
        | 0x2E00..=0x2E2E
        | 0x2E30..=0x2E4F
        | 0x3001..=0x3003
        | 0x3008..=0x3011
        | 0x3014..=0x301F
        | 0x3030
        | 0x303D
        | 0x30A0 | 0x30FB
        | 0xFE10..=0xFE19
        | 0xFE30..=0xFE52
        | 0xFE54..=0xFE61
        | 0xFE63 | 0xFE68 | 0xFE6A | 0xFE6B
        | 0xFF01..=0xFF03
        | 0xFF05..=0xFF0A
        | 0xFF0C..=0xFF0F
        | 0xFF1A | 0xFF1B | 0xFF1F | 0xFF20
        | 0xFF3B..=0xFF3D
        | 0xFF3F | 0xFF5B | 0xFF5D
        | 0xFF5F..=0xFF65
    )
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

    /// HuggingFace BertNormalizer with clean_text=true,
    /// handle_chinese_chars=true, strip_accents=None (inherits on),
    /// lowercase=true — applied in that order.
    pub fn normalize(text: &str) -> String {
        // Pass 1: clean_text + CJK padding. Must precede NFD so that the
        // spaces land around the composed ideograph.
        let mut staged = String::with_capacity(text.len() + 16);
        for c in text.chars() {
            if c == '\u{FFFD}' {
                continue;
            }
            if matches!(c, '\t' | '\n' | '\r') {
                staged.push(' ');
            } else if is_deleted_by_clean_text(c) {
                continue;
            } else if is_cjk_ideograph(c) {
                staged.push(' ');
                staged.push(c);
                staged.push(' ');
            } else {
                staged.push(c);
            }
        }

        // Pass 2: strip_accents (NFD, drop combining) then lowercase.
        let mut out = String::with_capacity(staged.len());
        for c in staged.nfd() {
            if is_combining_mark(c) {
                continue;
            }
            for lc in c.to_lowercase() {
                out.push(lc);
            }
        }
        out
    }

    /// BertPreTokenizer over *normalized* text: split on whitespace, then
    /// split each run on punctuation, emitting every punctuation character
    /// as its own word. Symbols and letters accumulate into words.
    pub fn words(normalized: &str) -> Vec<&str> {
        let mut out = Vec::new();
        let mut start: Option<usize> = None;
        for (i, c) in normalized.char_indices() {
            if c.is_whitespace() {
                if let Some(s) = start.take() {
                    out.push(&normalized[s..i]);
                }
            } else if is_bert_punctuation(c) {
                if let Some(s) = start.take() {
                    out.push(&normalized[s..i]);
                }
                out.push(&normalized[i..i + c.len_utf8()]);
            } else if start.is_none() {
                start = Some(i);
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
        if chars.len() > MAX_INPUT_CHARS_PER_WORD {
            return None;
        }
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
        // ids:      0      1        2         3      4       5     6
        Vocab::from_tokens(&[
            "the", "bloom", "filter", "##s", "cafe", ".", "un", "东", "京", "の",
        ])
    }

    #[test]
    fn longest_match_and_continuation() {
        let v = vocab();
        assert_eq!(v.tokenize("the filters"), vec![0, 2, 3]);
    }

    #[test]
    fn unknown_word_is_deleted_not_unked() {
        let v = vocab();
        assert_eq!(v.tokenize("the pydub filter"), vec![0, 2]);
    }

    #[test]
    fn partial_match_still_deletes_whole_word() {
        let v = vocab();
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

    // ---- the three divergences this rewrite fixes --------------------

    #[test]
    fn cjk_ideographs_split_per_character() {
        let v = vocab();
        // Without handle_chinese_chars this was ONE word, no ##京
        // continuation existed, and the whole thing was deleted.
        assert_eq!(Vocab::words(&Vocab::normalize("东京")), vec!["东", "京"]);
        assert_eq!(v.tokenize("东京"), vec![7, 8]);
    }

    #[test]
    fn kana_and_hangul_are_not_split() {
        // BERT spaces ideographs only; kana and hangul runs stay whole.
        assert_eq!(
            Vocab::words(&Vocab::normalize("こんにちは")),
            vec!["こんにちは"]
        );
        // Hangul syllables NFD-decompose into conjoining jamo, so a
        // precomposed literal won't compare equal even though nothing was
        // split — and HF's BertNormalizer decomposes identically. Assert
        // the invariant (one word), not the byte sequence.
        assert_eq!(Vocab::words(&Vocab::normalize("안녕하세요")).len(), 1);
    }

    #[test]
    fn ideograph_adjacent_to_latin_still_splits() {
        assert_eq!(
            Vocab::words(&Vocab::normalize("tokyo东京city")),
            vec!["tokyo", "东", "京", "city"]
        );
    }

    #[test]
    fn symbols_stay_attached_punctuation_does_not() {
        // Category S: glued to the word (usually making it OOV → deleted).
        assert_eq!(Vocab::words(&Vocab::normalize("a×b")), vec!["a×b"]);
        assert_eq!(Vocab::words(&Vocab::normalize("hello😅")), vec!["hello😅"]);
        assert_eq!(Vocab::words(&Vocab::normalize("€5")), vec!["€5"]);
        // Category P and ASCII punctuation: separate tokens.
        assert_eq!(Vocab::words(&Vocab::normalize("a-b")), vec!["a", "-", "b"]);
        assert_eq!(Vocab::words(&Vocab::normalize("a…b")), vec!["a", "…", "b"]);
        assert_eq!(Vocab::words(&Vocab::normalize("$5")), vec!["$", "5"]);
    }

    #[test]
    fn clean_text_deletes_rather_than_separates() {
        // Control and format chars vanish, JOINING the neighbors — the old
        // code emitted two words here.
        assert_eq!(Vocab::words(&Vocab::normalize("a\u{7}b")), vec!["ab"]);
        assert_eq!(
            Vocab::words(&Vocab::normalize("soft\u{AD}hyphen")),
            vec!["softhyphen"]
        );
        assert_eq!(
            Vocab::words(&Vocab::normalize("zero\u{200B}width")),
            vec!["zerowidth"]
        );
        // Real whitespace still separates.
        assert_eq!(
            Vocab::words(&Vocab::normalize("a\tb\nc")),
            vec!["a", "b", "c"]
        );
    }

    #[test]
    fn fullwidth_punctuation_splits_cjk() {
        let v = vocab();
        // 、is category P: without it in the table, 京 would fuse with the
        // comma and be deleted.
        assert_eq!(
            Vocab::words(&Vocab::normalize("东、京")),
            vec!["东", "、", "京"]
        );
        assert_eq!(v.tokenize("东、京"), vec![7, 8]);
    }

    #[test]
    fn mixed_script_sentence() {
        let v = vocab();
        assert_eq!(v.tokenize("The 东京 cafe."), vec![0, 7, 8, 4, 5]);
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert!(Vocab::words(&Vocab::normalize("")).is_empty());
        assert!(Vocab::words(&Vocab::normalize("   \t\n ")).is_empty());
        assert!(Vocab::words(&Vocab::normalize("\u{200B}\u{FEFF}")).is_empty());
    }
}
