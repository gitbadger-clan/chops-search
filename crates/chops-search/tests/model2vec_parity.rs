//! Golden parity against MinishLab's official model2vec-rs implementation.
//!
//! This is the test that upgrades chops-search-core's tokenizer + embedding from
//! "three gotchas transcribed from a blog post" to "verified against the
//! reference". If it passes, the browser-side query vectors are the same
//! vectors the official inference engine would produce.
//!
//! Needs real model files (network to fetch them once), so it is #[ignore]
//! gated per the usual hermeticity rule. Run with:
//!
//!   huggingface-cli download minishlab/potion-base-8M \
//!       tokenizer.json model.safetensors config.json --local-dir model/
//!   CHOPS_SEARCH_MODEL_DIR=model cargo test -p chops-search -- --ignored
//!
//! (config.json is needed by model2vec-rs's loader, not by chops itself.)

use chops_search::model_loader::load_model2vec;
use chops_search_core::builder::embed_f32;
use chops_search_core::wordpiece::Vocab;
use model2vec_rs::model::StaticModel;

/// Sentences chosen to exercise every behavior that could diverge:
/// case folding, accent stripping, punctuation splitting, OOV deletion,
/// subword continuation, and plain long-form prose.
const SENTENCES: &[&str] = &[
    "The London Beer Flood of 1814 killed eight people.",
    "Café au lait, s'il vous plaît — RÉSUMÉ naïveté",
    "pydub silently truncates your audio files!",
    "reciprocal rank fusion, with k=60, flattens the curve",
    "TOKENIZATION MUST BE CASE INSENSITIVE",
    "unfathomable antidisestablishmentarianism supercalifragilistic",
    "a . , - ) the to and of in",
    "turkmenistan seychelles guantanamo hemingway vanuatu",
    "Static embeddings reduce a forward pass to tokenize, look up, and \
     average. The model's stopword list is its row magnitudes: frequent \
     words carry tiny vectors while rare words dominate the mean.",
];

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

#[test]
#[ignore = "needs model files; set CHOPS_SEARCH_MODEL_DIR and run with --ignored"]
fn embeddings_match_official_implementation() {
    let dir = std::env::var("CHOPS_SEARCH_MODEL_DIR").expect(
        "set CHOPS_SEARCH_MODEL_DIR to a dir with tokenizer.json + model.safetensors + config.json",
    );

    // Our path: the exact code the browser runs (chops-search-core), fed by the
    // exact loader the build tool runs (chops-search).
    let (tokens, rows, dim) = load_model2vec(dir.as_ref()).expect("chops loader failed");
    let vocab = Vocab::from_tokens(&tokens);

    // Reference path: official crate, normalization forced ON to match
    // embed_f32's L2 norm.
    let official = StaticModel::from_pretrained(&dir, None, Some(true), None)
        .expect("model2vec-rs loader failed");
    let inputs: Vec<String> = SENTENCES.iter().map(|s| s.to_string()).collect();
    let theirs = official.encode(&inputs);
    assert_eq!(theirs.len(), SENTENCES.len());

    for (s, ref_vec) in SENTENCES.iter().zip(&theirs) {
        let ids = vocab.tokenize(s);
        match embed_f32(&ids, &rows, dim) {
            Some(ours) => {
                assert_eq!(ours.len(), ref_vec.len(), "dim mismatch on {s:?}");
                let c = cosine(&ours, ref_vec);
                assert!(
                    c > 0.9999,
                    "divergence on {s:?}: cosine {c} (token ids: {ids:?})"
                );
            }
            None => {
                // We produced nothing → every word must have been deleted
                // as untokenizable. The official implementation must agree
                // (a zero/empty vector), or our deletion logic is wrong.
                let norm: f32 = ref_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
                assert!(
                    norm < 1e-6,
                    "we deleted everything in {s:?} but the official \
                     implementation embedded it (norm {norm})"
                );
            }
        }
    }
}

/// Same oracle, sharper lens: per-sentence cosine is forgiving of a single
/// wrong token in a long sentence. Short inputs isolate the tokenizer.
#[test]
#[ignore = "needs model files; set CHOPS_SEARCH_MODEL_DIR and run with --ignored"]
fn single_words_match_official_implementation() {
    let dir = std::env::var("CHOPS_SEARCH_MODEL_DIR").expect("set CHOPS_SEARCH_MODEL_DIR");
    let (tokens, rows, dim) = load_model2vec(dir.as_ref()).expect("load");
    let vocab = Vocab::from_tokens(&tokens);
    let official = StaticModel::from_pretrained(&dir, None, Some(true), None).expect("official");

    let words = ["café", "Running", "filters", "guantanamo", "the", "won't"];
    let inputs: Vec<String> = words.iter().map(|s| s.to_string()).collect();
    let theirs = official.encode(&inputs);

    for (w, ref_vec) in words.iter().zip(&theirs) {
        let ids = vocab.tokenize(w);
        if let Some(ours) = embed_f32(&ids, &rows, dim) {
            let c = cosine(&ours, ref_vec);
            assert!(c > 0.9999, "tokenizer divergence on {w:?}: cosine {c}");
        } else {
            let norm: f32 = ref_vec.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(norm < 1e-6, "deletion disagreement on {w:?}");
        }
    }
}
