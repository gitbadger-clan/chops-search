//! Full loop with synthetic artifacts: exactly the bytes the browser
//! would fetch, driven through the same plan → ingest → search surface
//! the worker uses.

use chops_core::builder::{embed_f32, quantize_global, quantize_rows};
use chops_core::engine::Engine;
use chops_core::format::{Doc, Index, ModelMeta};
use chops_core::wordpiece::Vocab;

const DIM: usize = 4;

/// tokens 0..5; "beer" and "flood" carry orthogonal-ish signal.
fn tokens() -> Vec<String> {
    ["the", "beer", "flood", "audio", "pipeline"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn rows_f32() -> Vec<f32> {
    vec![
        0.01, 0.01, 0.0, 0.0, // the: near-zero stopword-ish
        1.0, 0.1, 0.0, 0.0, // beer
        0.9, 0.2, 0.1, 0.0, // flood (close to beer)
        0.0, 0.0, 1.0, 0.1, // audio
        0.0, 0.1, 0.9, 0.3, // pipeline (close to audio)
    ]
}

fn build_artifacts() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let tokens = tokens();
    let rows = rows_f32();
    let vocab = Vocab::from_tokens(&tokens);

    // Two docs: one about beer floods, one about audio pipelines.
    let doc_texts = ["the beer flood", "the audio pipeline"];
    let mut chunk_vecs_f32 = Vec::new();
    let mut chunk_doc = Vec::new();
    for (d, text) in doc_texts.iter().enumerate() {
        let ids = vocab.tokenize(text);
        let v = embed_f32(&ids, &rows, DIM).unwrap();
        chunk_vecs_f32.extend(v);
        chunk_doc.push(d as u16);
    }
    let (chunk_vecs, gscale) = quantize_global(&chunk_vecs_f32);

    let (data_i8, scales) = quantize_rows(&rows, DIM);

    let prefix_rows = 2u32; // "the", "beer" eager; rest range-fetched
    let meta = ModelMeta {
        dim: DIM as u16,
        prefix_rows,
        scales,
        tokens: tokens.clone(),
    };

    let index = Index {
        dim: DIM as u16,
        global_scale: gscale,
        docs: vec![
            Doc { url: "/beer-flood/".into(), title: "The London Beer Flood".into() },
            Doc { url: "/audio/".into(), title: "Audio pipelines".into() },
        ],
        chunk_doc,
        chunk_vecs,
        terms: vec![
            ("beer".into(), vec![(0, 2)]),
            ("flood".into(), vec![(0, 1)]),
            ("audio".into(), vec![(1, 2)]),
            ("pipeline".into(), vec![(1, 1)]),
            ("pydub".into(), vec![(1, 3)]), // OOV term only keyword knows
        ],
    };

    let rows_file: Vec<u8> = data_i8.iter().map(|&v| v as u8).collect();
    let prefix_file = rows_file[..prefix_rows as usize * DIM].to_vec();
    (meta.write(), index.write(), rows_file, prefix_file)
}

#[test]
fn plan_ingest_search_hybrid() {
    let (meta, index, rows_file, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();

    // "flood" (row 2) is outside the prefix → plan demands a fetch.
    let ranges = e.plan("beer flood");
    assert!(!ranges.is_empty());
    for r in &ranges {
        e.ingest(r.start, &rows_file[r.start as usize..r.end as usize])
            .unwrap();
    }
    let results = e.search("beer flood", 5);
    assert!(e.used_semantic());
    assert_eq!(results[0], 0, "beer-flood doc should rank first");

    // Semantic paraphrase: "pipeline" alone should surface the audio doc.
    for r in e.plan("pipeline") {
        e.ingest(r.start, &rows_file[r.start as usize..r.end as usize])
            .unwrap();
    }
    let results = e.search("pipeline", 5);
    assert!(e.used_semantic());
    assert_eq!(results[0], 1);
}

#[test]
fn unloaded_rows_degrade_to_keyword_only() {
    let (meta, index, _rows_file, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();

    // "flood" needs row 2, which we deliberately never ingest.
    let results = e.search("beer flood", 5);
    assert!(!e.used_semantic(), "must not quietly shrink the mean");
    assert_eq!(results[0], 0, "keyword side still finds the doc");
}

#[test]
fn oov_query_is_keyword_only_with_empty_plan() {
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();

    // "pydub" is out of vocabulary: nothing to fetch, keyword still hits.
    assert!(e.plan("pydub").is_empty());
    let results = e.search("pydub", 5);
    assert!(!e.used_semantic());
    assert_eq!(results[0], 1);
}

#[test]
fn quantization_fidelity_query_vs_f32() {
    // Cosine between the browser-side (int8 per-row) query embedding and
    // the build-side f32 embedding should be ~1.
    let (meta, index, rows_file, _prefix) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &rows_file).unwrap(); // load everything

    let tokens = tokens();
    let vocab = Vocab::from_tokens(&tokens);
    let ids = vocab.tokenize("beer flood pipeline");

    // Browser-side path (int8): reconstruct via search internals is
    // private; recompute with the same quantized data instead.
    let rows = rows_f32();
    let f32_vec = embed_f32(&ids, &rows, DIM).unwrap();

    // int8 path through the public engine: use plan/ingest already done,
    // then compare rankings as a proxy plus direct cosine via store-level
    // math re-derived from artifacts.
    let m = chops_core::format::ModelMeta::read(&meta).unwrap();
    let mut store =
        chops_core::store::RowStore::new(DIM, m.tokens.len(), m.scales.clone());
    store.ingest(0, &rows_file).unwrap();
    let q_vec = store.embed(&ids).unwrap();

    let dot: f32 = f32_vec.iter().zip(&q_vec).map(|(a, b)| a * b).sum();
    assert!(dot > 0.999, "int8 query drifted from f32: cosine {dot}");
}
