//! `chops-search query` — explain a query against built artifacts.
//!
//! Loads meta, index, and the full row matrix, runs the identical
//! chops-search-core surface natively, and prints the evidence behind the
//! ranking: keyword scores, best-chunk cosine, chunk count, and each
//! engine's RRF contribution per document.
//!
//! Keyword scoring goes through `KeywordIndex::idf`/`term_score`, so it
//! cannot drift from the ranker — it did once, when BM25 landed here a
//! commit later than in core. Only the RRF contribution arithmetic is
//! still restated, because core discards scores after ranking.

use std::fs;
use std::path::Path;

use crate::eval::ScoreArgs;
use anyhow::{Context, Result};
use chops_search_core::engine::{Engine, SemanticStatus};
use chops_search_core::format::{Index, ModelMeta};
use chops_search_core::wordpiece::Vocab;

/// List indexed documents with their URLs — what you need to write
/// `expect` entries in a query set after adding a post.
pub fn list_docs(artifacts: &Path) -> Result<()> {
    let a = crate::artifacts::resolve(artifacts)?;
    let index = Index::read(&fs::read(&a.index)?).context("parsing index")?;
    let mut chunks = vec![0u32; index.docs.len()];
    for &d in &index.chunk_doc {
        chunks[d as usize] += 1;
    }
    println!("{:<4} {:>6}  {:<44} title", "doc", "chunks", "url");
    for (i, d) in index.docs.iter().enumerate() {
        println!("{:<4} {:>6}  {:<44} {}", i, chunks[i], d.url, d.title);
    }
    Ok(())
}

pub fn explain(artifacts: &Path, query: &str, limit: usize, args: ScoreArgs) -> Result<()> {
    let a = crate::artifacts::resolve(artifacts)?;
    let meta_bytes = fs::read(&a.meta).with_context(|| format!("{}", a.meta.display()))?;
    let index_bytes = fs::read(&a.index).with_context(|| format!("{}", a.index.display()))?;
    let rows_bytes = fs::read(&a.rows).with_context(|| format!("{}", a.rows.display()))?;

    let mut engine = Engine::new(&meta_bytes, &index_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    // The whole matrix: `query` explains a ranking, so it must never
    // degrade to keyword-only the way a cold browser session can.
    engine
        .ingest(0, &rows_bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let opts = args.apply(engine.score_opts());
    engine.set_score_opts(opts);

    println!(
        "corpus:    {} docs, {} chunks, dim {}",
        engine.doc_count(),
        engine.chunk_count(),
        engine.dim()
    );
    println!("scoring:   {}", ScoreArgs::describe(&opts));
    println!("query:     {query:?}");

    // Vocab rebuilt only to SHOW the wordpiece split; the engine
    // tokenizes with its own copy. Display, not arithmetic.
    let meta = ModelMeta::read(&meta_bytes).context("parsing model.meta.bin")?;
    let vocab = Vocab::from_tokens(&meta.tokens);
    let pieces: Vec<&str> = vocab
        .tokenize(query)
        .iter()
        .map(|&i| meta.tokens[i as usize].as_str())
        .collect();

    let report = engine.search_detailed(query);

    println!("kw words:  {:?}", report.kw_words);
    println!("wordpiece: {pieces:?}");
    println!(
        "kw:        avgdl={:.1}, confidence {:.2} (floor {:.2}){}",
        report.avgdl,
        report.kw_confidence,
        opts.kw_confidence,
        if report.kw_gated {
            " — keyword list SUPPRESSED"
        } else {
            ""
        }
    );
    for t in &report.terms {
        println!(
            "keyword:   {:?} df={} idf={:.3}{}",
            t.term,
            t.df,
            t.idf,
            if t.expanded {
                format!("  (prefix ×{})", t.weight)
            } else {
                String::new()
            }
        );
    }
    for w in &report.unmatched {
        println!("keyword:   {w:?} matches no documents");
    }

    match report.semantic {
        SemanticStatus::Unavailable => println!("semantic:  unavailable (no in-vocabulary tokens)"),
        SemanticStatus::BelowFloor => println!(
            "semantic:  nothing cleared the floor (top {:.3} < min_cos {:.2})",
            report.top.unwrap_or(f32::NAN),
            opts.min_cos
        ),
        SemanticStatus::Suppressed => println!(
            "semantic:  SUPPRESSED — no keyword corroboration, gap {:.3} < min_gap {:.2}, \
             top {:.3} < strong_cos {}",
            report.gap.unwrap_or(f32::NAN),
            opts.min_gap,
            report.top.unwrap_or(f32::NAN),
            if opts.strong_cos.is_finite() {
                format!("{:.2}", opts.strong_cos)
            } else {
                "∞".into()
            }
        ),
        SemanticStatus::Ranked => {}
    }
    if let (Some(top), Some(gap)) = (report.top, report.gap) {
        println!("sem:       top {top:.3}, top-median gap {gap:.3}");
    }

    println!();
    println!(
        "{:<4} {:>8} {:>4} {:>9} {:>5} {:>9} {:>8} {:>7}  title",
        "doc", "fused", "kw#", "kw-score", "sem#", "best-cos", "penalty", "chunks"
    );
    let rank = |r: Option<u16>| r.map_or_else(|| "-".to_string(), |r| (r + 1).to_string());
    for d in report.docs.iter().take(limit) {
        println!(
            "{:<4} {:>8.5} {:>4} {:>9} {:>5} {:>9} {:>8} {:>7}  {}",
            d.doc,
            d.fused,
            rank(d.kw_rank),
            if d.kw_score > 0.0 {
                format!("{:.3}", d.kw_score)
            } else {
                "-".into()
            },
            rank(d.sem_rank),
            d.best_cos.map_or("-".into(), |c| format!("{c:.3}")),
            if d.penalty > 0.0 {
                format!("-{:.3}", d.penalty)
            } else {
                "0".into()
            },
            d.chunks,
            engine.doc_title(d.doc).unwrap_or("<missing>"),
        );
    }
    Ok(())
}
