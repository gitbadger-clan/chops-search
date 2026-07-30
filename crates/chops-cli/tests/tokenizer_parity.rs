//! Tokenizer parity against the real HuggingFace tokenizer, by exact token
//! id — not by embedding cosine.
//!
//! model2vec_parity.rs already checks that whole-sentence *embeddings*
//! match model2vec-rs. That's the right end-to-end assertion, but it can't
//! say WHICH codepoint diverged, and a single wrong token inside a long
//! sentence barely moves the cosine. So this file goes at the tokenizer
//! directly: same tokenizer.json, exact id sequences, and a sweep over
//! codepoint ranges instead of hand-picked fixtures. Fixtures only prove
//! what someone thought to write down; three real divergences (CJK
//! spacing, symbol-vs-punctuation, clean_text) survived a fixture suite
//! that looked thorough.
//!
//! The one transformation applied to the reference output is dropping
//! [UNK] ids: HF emits [UNK] for an untokenizable word, chops deletes the
//! word (gotcha 2 in wordpiece.rs). Filtering [UNK] from HF's ids is
//! therefore the definition of agreement, and it doubles as a check that
//! deletion happens at word granularity rather than piece granularity.
//!
//! Needs real model files:
//!
//!   hf download minishlab/potion-base-8M tokenizer.json \
//!       --local-dir model/
//!   CHOPS_MODEL_DIR=model cargo test -p chops-cli --test tokenizer_parity -- --ignored

use chops_core::wordpiece::Vocab;
use tokenizers::Tokenizer;

fn load() -> (Tokenizer, Vocab, u32) {
    let dir = std::env::var("CHOPS_MODEL_DIR")
        .expect("set CHOPS_MODEL_DIR to a directory containing tokenizer.json");
    let tk = Tokenizer::from_file(format!("{dir}/tokenizer.json")).expect("tokenizer.json");
    let unk = tk.token_to_id("[UNK]").expect("vocab has no [UNK]");

    // Rebuild chops's vocab from the same file, ordered by id so index ==
    // row. Uses the tokenizer's own vocab rather than the safetensors
    // loader: this test is about tokenization, not about the matrix.
    let vocab_map = tk.get_vocab(false);
    let mut tokens = vec![String::new(); vocab_map.len()];
    for (tok, id) in vocab_map {
        tokens[id as usize] = tok;
    }
    let vocab = Vocab::from_tokens(&tokens);
    (tk, vocab, unk)
}

/// Reference ids with [UNK] removed — the sequence chops must reproduce.
fn reference(tk: &Tokenizer, unk: u32, s: &str) -> Vec<u32> {
    tk.encode(s, false)
        .expect("encode")
        .get_ids()
        .iter()
        .copied()
        .filter(|&id| id != unk)
        .collect()
}

/// Inputs that exercise script and category boundaries rather than English
/// prose. Every entry here corresponds to a rule in wordpiece.rs.
const CASES: &[&str] = &[
    // Latin baseline (should already have passed before the rewrite)
    "The London Beer Flood of 1814 killed eight people.",
    "Café au lait, s'il vous plaît — RÉSUMÉ naïveté",
    "pydub silently truncates your audio files!",
    "won't can't shouldn't",
    "a . , - ) the to and of in",
    // CJK ideographs: must be spaced per character
    "东京都",
    "北京大学在哪里",
    "The 东京 office",
    "东、京",
    "汉字hanzi混合mixed",
    // Astral-plane CJK (SIP). Escaped rather than literal: these render as
    // tofu in most fonts, and is_cjk_ideograph's U+20000+ ranges would
    // otherwise go unverified.
    "\u{20000}\u{20001} test",
    "\u{2A700}\u{2F800} mixed",
    // Kana and hangul: must NOT be spaced per character
    "こんにちは世界",
    "カタカナのテスト",
    "안녕하세요 세계",
    "ひらがな漢字カタカナ",
    "安녕",
    // Symbols (category S) stay attached; punctuation (P) splits
    "a×b",
    "€5 and $5 and £5",
    "hello😅 world",
    "emoji 👨‍👩‍👧‍👦 family sequence",
    "temperature 25°C ±2",
    "math: 3≤4 and 5≠6",
    // clean_text: control and format chars are deleted, not separators
    "a\u{7}b",
    "soft\u{AD}hyphen",
    "zero\u{200B}width\u{200B}space",
    "bom\u{FEFF}inside",
    "tab\there\nnewline",
    // Combining marks
    "e\u{301}gal", // decomposed é
    "Ångström",    // precomposed
    "ﬁligree",     // ligature (compatibility, NOT decomposed by NFD)
    // Mixed everything
    "Rust 🦀 в 东京 café ±1 — done.",
    // Degenerate
    "",
    "   ",
    "\u{200B}",
    "!!!",
];

#[test]
#[ignore = "needs tokenizer.json; set CHOPS_MODEL_DIR and run with --ignored"]
fn exact_ids_match_reference() {
    let (tk, vocab, unk) = load();
    let mut failures = Vec::new();
    for &s in CASES {
        let theirs = reference(&tk, unk, s);
        let ours = vocab.tokenize(s);
        if ours != theirs {
            failures.push(format!(
                "  {s:?}\n      ours: {ours:?}\n      ref:  {theirs:?}\n      words: {:?}",
                Vocab::words(&Vocab::normalize(s))
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} cases diverged:\n{}",
        failures.len(),
        CASES.len(),
        failures.join("\n")
    );
}

/// Codepoint ranges chops claims exact parity over. A char is probed both
/// standalone and embedded between letters, because the two exercise
/// different rules: standalone catches misclassification, embedded catches
/// word-boundary errors.
const CLAIMED: &[(u32, u32, &str)] = &[
    (0x0020, 0x007F, "ASCII"),
    (0x00A0, 0x00FF, "Latin-1 supplement"),
    (0x0100, 0x017F, "Latin Extended-A"),
    (0x2000, 0x206F, "General Punctuation"),
    (0x3000, 0x303F, "CJK Symbols and Punctuation"),
    (0x4E00, 0x4E80, "CJK Unified Ideographs (sample)"),
    (0x3040, 0x30FF, "Hiragana + Katakana"),
    (0xFF00, 0xFF65, "Fullwidth forms"),
    (0xE000, 0xE080, "Private use (sample)"),
    (0x20000, 0x20080, "CJK Extension B (sample)"),
];

/// Codepoints where chops knowingly diverges (see wordpiece.rs header).
/// Listing them keeps the sweep honest: a new divergence still fails.
///
/// U+302E/U+302F are Hangul tone marks, category Mc. BERT's strip_accents
/// removes Mn only; `is_combining_mark` also covers Mc, so chops drops
/// them and keeps the surrounding letters where HF fuses them into the
/// word and deletes the whole word. Fixing it needs an Mn-only category
/// table in the wasm blob.
const KNOWN_GAPS: &[u32] = &[0x302E, 0x302F];

#[test]
#[ignore = "needs tokenizer.json; set CHOPS_MODEL_DIR and run with --ignored"]
fn codepoint_sweep_over_claimed_ranges() {
    let (tk, vocab, unk) = load();
    let mut diverged: Vec<(char, &str, String)> = Vec::new();

    for &(lo, hi, name) in CLAIMED {
        for cp in lo..=hi {
            let Some(c) = char::from_u32(cp) else {
                continue;
            };
            if KNOWN_GAPS.contains(&cp) {
                continue;
            }
            for probe in [format!("{c}"), format!("a{c}b"), format!("东{c}京")] {
                let theirs = reference(&tk, unk, &probe);
                let ours = vocab.tokenize(&probe);
                if ours != theirs {
                    diverged.push((c, name, format!("{probe:?} ours {ours:?} ref {theirs:?}")));
                    break;
                }
            }
        }
    }

    if !diverged.is_empty() {
        let mut msg = format!("{} codepoints diverged:\n", diverged.len());
        for (c, name, detail) in diverged.iter().take(40) {
            msg += &format!("  U+{:04X} {c:?} [{name}] {detail}\n", *c as u32);
        }
        if diverged.len() > 40 {
            msg += &format!("  … and {} more\n", diverged.len() - 40);
        }
        panic!("{msg}");
    }
}

/// Not an assertion — a map of where parity ends. Run it when extending
/// the punctuation table or before claiming support for a new script:
///
///   CHOPS_MODEL_DIR=model cargo test -p chops-cli --test tokenizer_parity \
///       -- --ignored survey --nocapture
///
/// Expected known gaps: Indic spacing marks (Mc), which BERT keeps and
/// chops strips along with Mn.
#[test]
#[ignore = "survey, not an assertion; run explicitly with --nocapture"]
fn survey_full_bmp() {
    let (tk, vocab, unk) = load();
    let mut runs: Vec<(u32, u32)> = Vec::new();
    let mut open: Option<(u32, u32)> = None;
    let mut total = 0usize;

    for cp in 0x0020u32..=0xFFFFu32 {
        let Some(c) = char::from_u32(cp) else {
            continue;
        };
        let probe = format!("a{c}b");
        let bad = vocab.tokenize(&probe) != reference(&tk, unk, &probe);
        if bad {
            total += 1;
            match open {
                Some((_, ref mut end)) => *end = cp,
                None => open = Some((cp, cp)),
            }
        } else if let Some(run) = open.take() {
            runs.push(run);
        }
    }
    if let Some(run) = open {
        runs.push(run);
    }

    println!("\n{total} diverging codepoints in {} runs:", runs.len());
    for (lo, hi) in &runs {
        if lo == hi {
            println!("  U+{lo:04X}");
        } else {
            println!("  U+{lo:04X}..U+{hi:04X}  ({} chars)", hi - lo + 1);
        }
    }
}

#[test]
#[ignore = "needs tokenizer.json; set CHOPS_MODEL_DIR and run with --ignored"]
fn overlong_words_match_reference() {
    let (tk, vocab, unk) = load();
    for n in [99usize, 100, 101, 150] {
        let w = "a".repeat(n);
        assert_eq!(vocab.tokenize(&w), reference(&tk, unk, &w), "length {n}");
    }
}
