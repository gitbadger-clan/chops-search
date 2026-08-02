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

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chops_search_core::format::{Index, ModelMeta};
use chops_search_core::keyword::keyword_words;
use chops_search_core::rrf;
use chops_search_core::score;
use chops_search_core::store::RowStore;
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

pub fn explain(artifacts: &Path, query: &str, limit: usize) -> Result<()> {
    // ---- Load exactly what the browser would ---------------------------
    let a = crate::artifacts::resolve(artifacts)?;
    let meta_bytes = fs::read(&a.meta).with_context(|| format!("{}", a.meta.display()))?;
    let index_bytes = fs::read(&a.index).with_context(|| format!("{}", a.index.display()))?;
    let rows_bytes = fs::read(&a.rows).with_context(|| format!("{}", a.rows.display()))?;

    let meta = ModelMeta::read(&meta_bytes).context("parsing model.meta.bin")?;
    let index = Index::read(&index_bytes).context("parsing index.bin")?;
    let dim = meta.dim as usize;

    let vocab = Vocab::from_tokens(&meta.tokens);
    let mut store = RowStore::new(dim, meta.tokens.len(), meta.scales.clone());
    store.ingest(0, &rows_bytes).context("ingesting rows")?;

    println!(
        "corpus:    {} docs, {} chunks, dim {}",
        index.docs.len(),
        index.chunk_doc.len(),
        dim
    );

    // ---- Tokenization report (both pipelines) --------------------------
    let norm = Vocab::normalize(query);
    let words: Vec<&str> = keyword_words(&norm).into_iter().collect();
    let ids = vocab.tokenize(query);
    println!("query:     {query:?}");
    println!("kw words:  {words:?}");
    println!(
        "wordpiece: {:?}",
        ids.iter()
            .map(|&i| meta.tokens[i as usize].as_str())
            .collect::<Vec<_>>()
    );

    // ---- Keyword side, with scores (same formula as core::keyword) -----
    let kw = index.keyword_index();
    println!("kw:        avgdl={:.1}", kw.avgdl);
    let terms = kw.resolve(&words, true);
    let mut kw_scores: HashMap<u16, f32> = HashMap::new();
    for t in &terms {
        let postings = &kw.terms[&t.text];
        let idf = kw.idf(postings.len());
        println!(
            "keyword:   {:?} df={} idf={idf:.3}{}",
            t.text,
            postings.len(),
            if t.expanded {
                format!("  (prefix ×{})", t.weight)
            } else {
                String::new()
            }
        );
        for &(doc, tf) in postings {
            *kw_scores.entry(doc).or_insert(0.0) += t.weight * kw.term_score(doc, tf, idf);
        }
    }
    for &w in &words {
        if !terms.iter().any(|t| t.text.as_ref() == w) {
            println!("keyword:   {w:?} matches no documents");
        }
    }
    let kw_ranked = kw.rank_terms(&terms);

    // ---- Semantic side, with per-doc best cosine + chunk counts --------
    let mut best_cos = vec![f32::NEG_INFINITY; index.docs.len()];
    let mut chunk_count = vec![0u32; index.docs.len()];
    for &doc in &index.chunk_doc {
        chunk_count[doc as usize] += 1;
    }
    let q = store.embed(&ids);
    let sem_ranked: Vec<u16> = match &q {
        None => {
            println!("semantic:  unavailable (no in-vocabulary tokens)");
            Vec::new()
        }
        Some(qv) => {
            for (c, &doc) in index.chunk_doc.iter().enumerate() {
                let row = &index.chunk_vecs[c * dim..(c + 1) * dim];
                let mut acc = 0f32;
                for (&qi, &vi) in qv.iter().zip(row) {
                    acc += qi * vi as f32;
                }
                let s = acc * index.global_scale;
                if s > best_cos[doc as usize] {
                    best_cos[doc as usize] = s;
                }
            }
            score::rank_docs(
                qv,
                &index.chunk_vecs,
                dim,
                index.global_scale,
                &index.chunk_doc,
                index.docs.len(),
                score::ScoreOpts::default(),
            )
        }
    };

    // ---- Fuse and print, with contributions ----------------------------
    let lists: Vec<&[u16]> = if sem_ranked.is_empty() {
        vec![&kw_ranked]
    } else {
        vec![&kw_ranked, &sem_ranked]
    };
    let fused = rrf::fuse(&lists, rrf::K);

    let rank_of = |list: &[u16], d: u16| list.iter().position(|&x| x == d);
    println!();
    println!(
        "{:<4} {:>8} {:>4} {:>9} {:>5} {:>9} {:>8} {:>7}  title",
        "doc", "fused", "kw#", "kw-score", "sem#", "best-cos", "penalty", "chunks"
    );
    for &d in fused.iter().take(limit) {
        let kwr = rank_of(&kw_ranked, d);
        let smr = rank_of(&sem_ranked, d);
        let fused_score = kwr.map_or(0.0, |r| 1.0 / (rrf::K + (r + 1) as f32))
            + smr.map_or(0.0, |r| 1.0 / (rrf::K + (r + 1) as f32));
        let opt_rank = |r: Option<usize>| r.map_or_else(|| "-".into(), |r| (r + 1).to_string());

        let n_chunks = chunk_count[d as usize] as usize;
        let raw = best_cos[d as usize];
        let (cos_s, pen_s) = if raw > f32::NEG_INFINITY {
            let p = score::chunk_correction(n_chunks, score::CHUNK_PENALTY);
            let pen = if p > 0.0 {
                format!("-{p:.3}")
            } else {
                "0".to_string()
            };
            (format!("{raw:.3}"), pen)
        } else {
            ("-".to_string(), "-".to_string())
        };

        println!(
            "{:<4} {:>8.5} {:>4} {:>9} {:>5} {:>9} {:>8} {:>7}  {}",
            d,
            fused_score,
            opt_rank(kwr),
            kw_scores
                .get(&d)
                .map_or_else(|| "-".into(), |s| format!("{s:.3}")),
            opt_rank(smr),
            cos_s,
            pen_s,
            n_chunks,
            index.docs[d as usize].title,
        );
    }
    Ok(())
}
