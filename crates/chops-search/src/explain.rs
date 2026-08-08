//! `chops-search query` — explain a query against built artifacts.
//!
//! Loads meta, index, and the full row matrix, runs the identical
//! chops-search-core surface natively, and prints the evidence behind the
//! ranking: keyword scores with the fields they came from, best-chunk
//! cosine, chunk count, and each engine's RRF contribution per document.
//!
//! Keyword scoring goes through `KeywordIndex::idf`/`term_score`, so it
//! cannot drift from the ranker — it did once, when BM25 landed here a
//! commit later than in core. Only the RRF contribution arithmetic is
//! still restated, because core discards scores after ranking.
//!
//! The per-field term frequencies come from `index.bin` directly rather
//! than from the report: `SearchReport` carries df and idf per term, but
//! not per-doc field tfs, and the engine's keyword index is private. With
//! BM25F, "which field did this term turn up in" is the first question
//! worth asking about a surprising ranking, so it's worth the second
//! parse of a file already in memory.

use std::collections::HashMap;
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

/// Per-doc field term frequencies, summed over every resolved query term.
/// Raw counts, not weighted: the weights are printed in the scoring line
/// and applying them here would produce a number that appears nowhere in
/// the ranker.
#[derive(Default, Clone, Copy)]
struct FieldTotals {
    title: u32,
    tag: u32,
    desc: u32,
    body: u32,
}

impl FieldTotals {
    fn is_empty(&self) -> bool {
        self.title == 0 && self.tag == 0 && self.desc == 0 && self.body == 0
    }
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
        "kw:        avg len title {:.1}, tag {:.1}, desc {:.1}, body {:.1}; \
         confidence {:.2} (floor {:.2}){}",
        report.avg_title,
        report.avg_tag,
        report.avg_desc,
        report.avg_body,
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

    // Postings for the resolved terms, straight out of the artifact. The
    // report knows df but not where in a document a term landed, and
    // under BM25F that's the difference between a title match and a
    // passing mention in paragraph nine.
    let index = Index::read(&index_bytes).context("parsing index")?;
    let postings: HashMap<&str, _> = index
        .terms
        .iter()
        .map(|(term, p)| (term.as_str(), p))
        .collect();
    let mut fields = vec![FieldTotals::default(); index.docs.len()];
    for t in &report.terms {
        let Some(list) = postings.get(t.term.as_ref()) else {
            continue;
        };
        for p in list.iter() {
            let Some(f) = fields.get_mut(p.doc as usize) else {
                continue;
            };
            f.title += p.title as u32;
            f.tag += p.tag as u32;
            f.desc += p.desc as u32;
            f.body += p.body as u32;
        }
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
    println!("t/g/d/b:   title/tag/desc/body term frequencies, summed over the terms above");
    println!(
        "{:<4} {:>8} {:>4} {:>9} {:>11} {:>5} {:>9} {:>8} {:>7}  title",
        "doc", "fused", "kw#", "kw-score", "t/g/d/b", "sem#", "best-cos", "penalty", "chunks"
    );
    let rank = |r: Option<u16>| r.map_or_else(|| "-".to_string(), |r| (r + 1).to_string());
    for d in report.docs.iter().take(limit) {
        let f = fields.get(d.doc as usize).copied().unwrap_or_default();
        println!(
            "{:<4} {:>8.5} {:>4} {:>9} {:>11} {:>5} {:>9} {:>8} {:>7}  {}",
            d.doc,
            d.fused,
            rank(d.kw_rank),
            if d.kw_score > 0.0 {
                format!("{:.3}", d.kw_score)
            } else {
                "-".into()
            },
            if f.is_empty() {
                "-".to_string()
            } else {
                format!("{}/{}/{}/{}", f.title, f.tag, f.desc, f.body)
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
