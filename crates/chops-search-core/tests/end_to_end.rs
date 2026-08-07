//! Full loop with synthetic artifacts: exactly the bytes the browser
//! would fetch, driven through the same plan → ingest → search surface
//! the worker uses.

use chops_search_core::builder::{embed_f32, quantize_global, quantize_rows};
use chops_search_core::engine::{Engine, SemanticStatus};
use chops_search_core::format::{Doc, Index, ModelMeta};
use chops_search_core::score::{KW_CONFIDENCE, ScoreOpts};
use chops_search_core::wordpiece::Vocab;

const DIM: usize = 4;

/// Six tokens. "beer"/"flood" and "audio"/"pipeline" form two related
/// clusters; "granite" sits alone on the fourth axis and has no keyword
/// postings, which is what makes it the standout for the gate tests.
fn tokens() -> Vec<String> {
    ["the", "beer", "flood", "audio", "pipeline", "granite"]
        .iter()
        .map(|s| s.to_string())
        .collect()
}

fn rows_f32() -> Vec<f32> {
    vec![
        0.01, 0.01, 0.0, 0.0, // the
        1.0, 0.1, 0.0, 0.0, // beer
        0.9, 0.2, 0.1, 0.0, // flood
        0.0, 0.0, 1.0, 0.1, // audio
        0.0, 0.1, 0.9, 0.3, // pipeline
        0.0, 0.0, 0.0, 1.0, // granite: the standout axis
    ]
}

fn build_artifacts() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    let tokens = tokens();
    let rows = rows_f32();
    let vocab = Vocab::from_tokens(&tokens);

    // Three docs. doc0 has TWO chunks so the semantic winner (chunk 0)
    // and the keyword-only fallback (first body chunk, 1) differ — that
    // difference is what the best_chunk reset tests observe. doc2 is the
    // kw-empty standout. Chunks stay contiguous per doc; the fallback
    // logic assumes it.
    let chunk_texts: [(&str, u16); 4] = [
        ("beer flood", 0),
        ("the audio", 0),
        ("the audio pipeline", 1),
        ("granite", 2),
    ];
    let mut chunk_vecs_f32 = Vec::new();
    let mut chunk_doc = Vec::new();
    for (text, d) in chunk_texts {
        let ids = vocab.tokenize(text);
        chunk_vecs_f32.extend(embed_f32(&ids, &rows, DIM).unwrap());
        chunk_doc.push(d);
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
            Doc {
                url: "/beer-flood/".into(),
                title: "The London Beer Flood".into(),
            },
            Doc {
                url: "/audio/".into(),
                title: "Audio pipelines".into(),
            },
            Doc {
                url: "/granite/".into(),
                title: "Granite".into(),
            },
        ],
        chunk_doc,
        chunk_vecs,
        // "granite" deliberately absent: a granite query must arrive at
        // fusion with zero keyword corroboration.
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
    // This test is about the byte path, not about relevance. The
    // synthetic corpus has arbitrary cosines, so leave the floor out of
    // it rather than tuning fixture vectors to clear a threshold that
    // exists for real content.
    e.set_score_opts(ScoreOpts::raw());
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
    let m = chops_search_core::format::ModelMeta::read(&meta).unwrap();
    let mut store = chops_search_core::store::RowStore::new(DIM, m.tokens.len(), m.scales.clone());
    store.ingest(0, &rows_file).unwrap();
    let q_vec = store.embed(&ids).unwrap();

    let dot: f32 = f32_vec.iter().zip(&q_vec).map(|(a, b)| a * b).sum();
    assert!(dot > 0.999, "int8 query drifted from f32: cosine {dot}");
}

#[test]
fn flat_uncorroborated_field_is_suppressed() {
    // "the": no keyword postings, no expansions, embeds from the prefix.
    // min_gap 10 makes ANY field flat — this tests the wiring, not the
    // margin; the margin's power is eval's job on real corpora.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    let mut opts = ScoreOpts::raw();
    opts.min_gap = 10.0;
    e.set_score_opts(opts);

    let report = e.search_detailed("the");
    assert_eq!(report.semantic, SemanticStatus::Suppressed);
    assert!(report.ids(5).is_empty());
    assert!(!e.used_semantic());
    assert!(report.gap.unwrap() < 10.0);
}

#[test]
fn corroborated_field_bypasses_the_gate() {
    // Same absurd margin, but "beer" has postings: with keyword
    // corroboration the gate is never consulted.
    let (meta, index, rows_file, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    for r in e.plan("beer") {
        e.ingest(r.start, &rows_file[r.start as usize..r.end as usize])
            .unwrap();
    }
    let mut opts = ScoreOpts::raw();
    opts.min_gap = 10.0;
    e.set_score_opts(opts);

    let report = e.search_detailed("beer");
    assert_eq!(report.semantic, SemanticStatus::Ranked);
    assert!(!report.ids(5).is_empty());
    assert!(e.used_semantic());
}

#[test]
fn min_gap_zero_never_gates() {
    // The shipped default: pre-gate behavior, bit for bit.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    e.set_score_opts(ScoreOpts::raw()); // min_gap 0.0

    let report = e.search_detailed("the");
    assert_eq!(report.semantic, SemanticStatus::Ranked);
    assert!(report.gap.is_some(), "gap is reported even when unused");
}

#[test]
fn standout_passes_an_active_gate() {
    // "granite": kw-empty AND one doc clearly stands out. A realistic
    // margin must not gate it. This catches an inverted comparison —
    // the suppression test alone passes if the gate fires on everything.
    let (meta, index, rows_file, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    for r in e.plan("granite") {
        e.ingest(r.start, &rows_file[r.start as usize..r.end as usize])
            .unwrap();
    }
    let mut opts = ScoreOpts::raw();
    opts.min_gap = 0.5; // doc2's gap over the pack is ~0.8
    e.set_score_opts(opts);

    let report = e.search_detailed("granite");
    assert_eq!(report.semantic, SemanticStatus::Ranked);
    assert_eq!(report.ids(5)[0], 2);
    assert!(report.gap.unwrap() > 0.5);
}

#[test]
fn search_is_a_view_of_search_detailed() {
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    e.set_score_opts(ScoreOpts::raw());
    let via_report = e.search_detailed("the audio").ids(5);
    let via_search = e.search("the audio", 5);
    assert_eq!(via_report, via_search);
}

#[test]
fn kw_gated_report_still_shows_kw_evidence() {
    // One matched rare term drowned by three misses: confidence gates
    // the LIST, but the report keeps the per-doc BM25 evidence.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    let mut opts = ScoreOpts::raw();
    opts.kw_confidence = KW_CONFIDENCE; // re-arm just this gate
    e.set_score_opts(opts);

    // "beer" matches (df 1 of 3); zzz/qqq/www miss at max idf →
    // confidence ≈ 0.14 < 0.30. Embeds as pure "beer" (OOV deleted).
    let report = e.search_detailed("beer zzz qqq www");
    assert!(report.kw_gated);
    assert!(report.docs.iter().all(|d| d.kw_rank.is_none()));
    let doc0 = report.docs.iter().find(|d| d.doc == 0).unwrap();
    assert!(doc0.kw_score > 0.0, "suppressed evidence must still show");
}

#[test]
fn below_floor_is_distinct_from_suppressed() {
    // Impossible floor, gate disabled: the status must say BelowFloor,
    // and the gap is still measured. Guards the status-arm ordering.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    let mut opts = ScoreOpts::raw();
    opts.min_cos = 2.0;
    e.set_score_opts(opts);

    let report = e.search_detailed("the");
    assert_eq!(report.semantic, SemanticStatus::BelowFloor);
    assert!(report.gap.is_some());
    assert!(!e.used_semantic());
}

#[test]
fn best_chunk_resets_between_queries() {
    // Query A: "beer flood" wins on doc0's chunk 0. Query B: "pydub"
    // (all-OOV, semantic never runs) — best_chunk(0) must be the
    // FALLBACK (first body chunk, 1), not query A's stale winner (0).
    let (meta, index, rows_file, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    e.set_score_opts(ScoreOpts::raw());
    for r in e.plan("beer flood") {
        e.ingest(r.start, &rows_file[r.start as usize..r.end as usize])
            .unwrap();
    }
    e.search("beer flood", 5);
    assert_eq!(e.best_chunk(0), 0, "precondition: chunk 0 won query A");

    e.search("pydub", 5);
    assert_eq!(e.best_chunk(0), 1, "stale winner survived the reset");
}

#[test]
fn suppressed_query_resets_best_chunk_too() {
    // Same shape, but query B dies at the GATE rather than at embed:
    // the reset must run before the (empty) fill on that path as well.
    let (meta, index, rows_file, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    e.set_score_opts(ScoreOpts::raw());
    for r in e.plan("beer flood") {
        e.ingest(r.start, &rows_file[r.start as usize..r.end as usize])
            .unwrap();
    }
    e.search("beer flood", 5);
    assert_eq!(e.best_chunk(0), 0);

    let mut opts = ScoreOpts::raw();
    opts.min_gap = 10.0;
    e.set_score_opts(opts);
    e.search("the", 5); // uncorroborated → gated
    assert_eq!(e.best_chunk(0), 1);
}

#[test]
fn strong_top_bypasses_an_active_gate() {
    // Gate armed so hard everything is "flat"; the hatch must rescue a
    // field whose best doc is clearly relevant in absolute terms.
    let (meta, index, rows_file, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    for r in e.plan("granite") {
        e.ingest(r.start, &rows_file[r.start as usize..r.end as usize])
            .unwrap();
    }
    let mut opts = ScoreOpts::raw();
    opts.min_gap = 10.0; // nothing can clear this
    opts.strong_cos = 0.5; // but granite's top is ~1.0
    e.set_score_opts(opts);

    let report = e.search_detailed("granite");
    assert_eq!(report.semantic, SemanticStatus::Ranked);
    assert!(report.top.unwrap() > 0.5);
}
