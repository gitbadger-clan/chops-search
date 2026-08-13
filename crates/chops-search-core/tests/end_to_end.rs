//! Full loop with synthetic artifacts: exactly the bytes the browser
//! would fetch, driven through the same plan → ingest → search surface
//! the worker uses.

use chops_search_core::builder::{embed_f32, quantize_global, quantize_rows};
use chops_search_core::engine::{Engine, SemanticStatus};
use chops_search_core::format::{Doc, Index, ModelMeta, Posting};
use chops_search_core::keyword::FieldWeights;
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

/// Field order matches the struct and the wire format, so a call site
/// reads the same as a hexdump.
fn post(doc: u16, title: u16, tag: u16, desc: u16, body: u16) -> Posting {
    Posting {
        doc,
        title,
        tag,
        desc,
        body,
    }
}

fn build_artifacts() -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
    build_artifacts_with(FieldWeights::default())
}

fn build_artifacts_with(weights: FieldWeights) -> (Vec<u8>, Vec<u8>, Vec<u8>, Vec<u8>) {
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
        weights,
        // v5 scoring calibration, inert: this fixture is an
        // UNCONFIGURED corpus, so it writes exactly what a
        // chops-search.toml with no scoring keys writes — gate
        // disarmed, plain RRF, floor derived from dims. The
        // calibrated path is exercised by
        // calibrated_scoring_reaches_the_engine, which mutates
        // these on a read-back Index.
        min_gap: 0.0,
        rrf_alpha: 0.0,
        min_cos: None,
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
        // Field tfs are consistent with the doc titles above, so the
        // BM25F path gets exercised rather than degenerating to
        // body-only postings. No tags on this corpus — that's the
        // avg_tag floor doing its job, not an oversight.
        //
        // "granite" deliberately absent: a granite query must arrive at
        // fusion with zero keyword corroboration.
        terms: vec![
            ("audio".into(), vec![post(1, 1, 0, 1, 2)]),
            ("beer".into(), vec![post(0, 1, 0, 1, 2)]),
            ("flood".into(), vec![post(0, 1, 0, 0, 1)]),
            // Title says "pipelines"; this index has no stemmer, so the
            // singular term legitimately has a title tf of 0.
            ("pipeline".into(), vec![post(1, 0, 0, 0, 1)]),
            ("pydub".into(), vec![post(1, 0, 0, 0, 3)]), // OOV term only keyword knows
            // Description-only: appears in doc2's front-matter
            // description and nowhere in its prose. Out of vocabulary
            // too, so a query for it is keyword-only by construction and
            // isolates the desc field from the semantic side.
            ("quarry".into(), vec![post(2, 0, 0, 2, 0)]),
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

// ---- BM25F wiring -----------------------------------------------------

#[test]
fn engine_takes_field_weights_from_the_index() {
    // The trap this guards: seeding opts with a bare
    // `ScoreOpts::default()` would compile fine and silently ignore what
    // the corpus was built with. Non-default weights make that visible,
    // and all three are distinct so a transposed pair cannot pass.
    let built = FieldWeights {
        title: 7.0,
        tag: 9.0,
        desc: 5.0,
    };
    let (meta, index, _rows, _prefix) = build_artifacts_with(built);
    let e = Engine::new(&meta, &index).unwrap();
    assert_eq!(e.score_opts().weights, built);
}

#[test]
fn field_weights_survive_the_byte_roundtrip_and_move_scores() {
    // "beer" sits in doc0's title (tf 1), description (tf 1), and body
    // (tf 2). Dropping the title weight to zero must lower its BM25F
    // score — end to end, through index.bin's header rather than through
    // a constant.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut weighted = Engine::new(&meta, &index).unwrap();
    weighted.ingest(0, &prefix_file).unwrap();
    let with_title = weighted.search_detailed("beer").docs[0].kw_score;

    let (meta0, index0, _rows0, prefix0) = build_artifacts_with(FieldWeights {
        title: 0.0,
        ..FieldWeights::default()
    });
    let mut bodyonly = Engine::new(&meta0, &index0).unwrap();
    bodyonly.ingest(0, &prefix0).unwrap();
    let without_title = bodyonly.search_detailed("beer").docs[0].kw_score;

    assert!(with_title > 0.0 && without_title > 0.0);
    assert!(
        with_title > without_title,
        "title weight did not reach the ranker: {with_title} vs {without_title}"
    );
}

#[test]
fn report_carries_all_four_field_averages() {
    // Titles, descriptions, and bodies are all populated here; no tags
    // anywhere, so avg_tag rides its floor. Explain prints all four, so
    // all four must be finite.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    let report = e.search_detailed("beer");
    assert!(report.avg_title > 0.0);
    assert_eq!(report.avg_tag, 1.0, "no tags anywhere → the floor");
    assert!(
        report.avg_desc > 1.0,
        "descriptions are real on this corpus"
    );
    assert!(report.avg_body > 0.0);
}

// ---- the description field -------------------------------------------

#[test]
fn description_only_term_is_findable() {
    // "quarry" appears in doc2's description and nowhere else: not in
    // its prose, not in the vocab. If the desc field is dropped anywhere
    // between the builder and the ranker — write, read, dl_desc, or
    // term_score — this query returns nothing.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    e.set_score_opts(ScoreOpts::raw());

    assert!(e.plan("quarry").is_empty(), "out of vocabulary by design");
    let results = e.search("quarry", 5);
    assert!(
        !e.used_semantic(),
        "keyword-only, so the desc field is alone"
    );
    assert_eq!(results, vec![2]);
}

#[test]
fn desc_weight_zero_removes_description_evidence() {
    // The property the field exists for: whether descriptions count is a
    // query-time question, answerable without a rebuild. At w_desc 0 the
    // only evidence "quarry" had is gone and the list is empty.
    let (meta, index, _rows, prefix_file) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &prefix_file).unwrap();
    e.set_score_opts(ScoreOpts {
        weights: FieldWeights {
            desc: 0.0,
            ..e.score_opts().weights
        },
        ..ScoreOpts::raw()
    });

    assert!(
        e.search("quarry", 5).is_empty(),
        "a desc-only term must carry no evidence at w_desc 0"
    );
    // Sibling check: a doc with body evidence is unaffected by the same
    // override, so this really is the desc field and not a dead ranker.
    assert_eq!(e.search("pydub", 5), vec![1]);
}

// ---- weighted fusion --------------------------------------------------

/// The query that reproduces the shape weighted RRF exists for.
///
/// "quarry" is a df-1 description term on doc2 and out of vocabulary, so
/// the semantic side embeds "pipeline" alone: it ranks doc1 first (the
/// pipeline chunk), doc0 second, doc2 third. The keyword side ranks doc2
/// first, because a tf-2 hit in a 2-token description beats a tf-1 hit in
/// a 6-token body. So doc2 is kw#1 / sem#3 and doc1 is kw#2 / sem#1,
/// which unweighted RRF resolves in favour of doc1:
///
///   doc1: 1/62 + 1/61 = 0.032522
///   doc2: 1/61 + 1/63 = 0.032266
const SPLIT_QUERY: &str = "quarry pipeline";

#[test]
fn unweighted_fusion_prefers_the_semantic_winner() {
    // The precondition for the next test, asserted rather than assumed:
    // if the fixture ever stops producing this disagreement, that test
    // would pass vacuously and stop testing anything.
    let (meta, index, rows_file, _prefix) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &rows_file).unwrap();
    e.set_score_opts(ScoreOpts::raw()); // rrf_alpha 0

    let r = e.search_detailed(SPLIT_QUERY);
    assert_eq!(r.kw_rrf_weight, 1.0, "raw() must fuse at plain RRF");

    let kw_first = r.docs.iter().find(|d| d.kw_rank == Some(0)).unwrap().doc;
    let sem_first = r.docs.iter().find(|d| d.sem_rank == Some(0)).unwrap().doc;
    assert_eq!(kw_first, 2, "keyword side should back the desc-only hit");
    assert_eq!(sem_first, 1, "semantic side should back the pipeline chunk");
    let doc2 = r.docs.iter().find(|d| d.doc == 2).unwrap();
    assert_eq!(
        doc2.sem_rank,
        Some(2),
        "doc2 must be sem#3 for the arithmetic"
    );

    assert_eq!(r.docs[0].doc, sem_first, "unweighted RRF prefers sem#1");
}

#[test]
fn rrf_alpha_lets_a_confident_keyword_list_outvote_the_semantic_one() {
    // Same query, same corpus, one knob. Both words are corpus terms so
    // confidence is 1.0 and the weight is 1 + alpha.
    let (meta, index, rows_file, _prefix) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &rows_file).unwrap();
    e.set_score_opts(ScoreOpts {
        rrf_alpha: 4.0,
        ..ScoreOpts::raw()
    });

    let r = e.search_detailed(SPLIT_QUERY);
    assert!(
        (r.kw_rrf_weight - 5.0).abs() < 1e-5,
        "expected 1 + 4 * confidence, got {}",
        r.kw_rrf_weight
    );
    assert_eq!(r.docs[0].doc, 2, "the df-1 keyword hit should now win");
    // The semantic winner is demoted, not discarded: weighting trades
    // between engines, it does not silence one.
    assert!(r.docs.iter().any(|d| d.doc == 1 && d.sem_rank == Some(0)));
}

#[test]
fn a_weak_keyword_list_gets_no_boost() {
    // The reason the weight routes through kw_confidence rather than
    // being a flat multiplier: a query whose keyword evidence is mostly
    // misses must fuse at close to plain RRF even with alpha armed, or
    // the knob would amplify exactly the stopword rankings the
    // confidence gate exists to distrust.
    let (meta, index, rows_file, _prefix) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &rows_file).unwrap();
    e.set_score_opts(ScoreOpts {
        rrf_alpha: 4.0,
        ..ScoreOpts::raw()
    });

    // "beer" matches; zzz/qqq/www miss at max idf → confidence ~0.14.
    let weak = e.search_detailed("beer zzz qqq www");
    let strong = e.search_detailed(SPLIT_QUERY);
    assert!(weak.kw_confidence < 0.2, "{}", weak.kw_confidence);
    assert!(
        weak.kw_rrf_weight < 2.0,
        "weak evidence must not reach the ~2 that overturns a ranking: {}",
        weak.kw_rrf_weight
    );
    assert!(strong.kw_rrf_weight > weak.kw_rrf_weight);
}

#[test]
fn a_gated_keyword_list_cannot_be_amplified() {
    // Confidence gating happens before fusion, so an armed alpha must not
    // resurrect a list the gate rejected: the weight multiplies an empty
    // list and changes nothing.
    let (meta, index, rows_file, _prefix) = build_artifacts();
    let mut e = Engine::new(&meta, &index).unwrap();
    e.ingest(0, &rows_file).unwrap();

    let gated = ScoreOpts {
        kw_confidence: KW_CONFIDENCE,
        rrf_alpha: 4.0,
        ..ScoreOpts::raw()
    };
    e.set_score_opts(gated);
    let r = e.search_detailed("beer zzz qqq www");
    assert!(r.kw_gated);
    assert!(r.docs.iter().all(|d| d.kw_rank.is_none()));

    // Identical order to the same run with fusion weighting off.
    e.set_score_opts(ScoreOpts {
        kw_confidence: KW_CONFIDENCE,
        ..ScoreOpts::raw()
    });
    let plain = e.search_detailed("beer zzz qqq www");
    assert_eq!(r.ids(5), plain.ids(5));
}

// ---- corroboration gate ----------------------------------------------

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
    // the LIST, but the report keeps the per-doc BM25F evidence.
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

#[test]
fn calibrated_scoring_reaches_the_engine() {
    // The honesty-gap test: values written into index.bin must be the
    // values the engine scores with, and an uncalibrated index must
    // yield the derived floor and inert knobs.
    let (meta_bytes, index_bytes, _rows, _prefix) = build_artifacts();
    let mut index = Index::read(&index_bytes).unwrap();
    index.min_gap = 0.08;
    index.rrf_alpha = 1.0;
    index.min_cos = Some(0.34);
    let engine = Engine::new(&meta_bytes, &index.write()).unwrap();
    let o = engine.score_opts();
    assert_eq!(o.min_gap, 0.08);
    assert_eq!(o.rrf_alpha, 1.0);
    assert_eq!(o.min_cos, 0.34, "override must beat the derived floor");

    index.min_cos = None;
    let engine = Engine::new(&meta_bytes, &index.write()).unwrap();
    assert_eq!(
        engine.score_opts().min_cos,
        chops_search_core::score::min_cos_for(engine.dim()),
        "absent override must derive from dims"
    );
}
