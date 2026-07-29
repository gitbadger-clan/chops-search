//! `chops query` — explain a query against built artifacts.
//!
//! Loads the same four artifacts the browser fetches, runs the identical
//! chops-core surface natively, and prints the evidence behind the final
//! ranking: keyword scores (recomputed with the same formula), best-chunk
//! cosine + chunk count per doc, and each engine's RRF contribution.
//!
//! The keyword-score and RRF-contribution math is duplicated from
//! chops-core here because the core deliberately discards scores after
//! ranking; if either formula ever changes in core, change it here too.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chops_core::format::{Index, ModelMeta};
use chops_core::rrf;
use chops_core::score;
use chops_core::store::RowStore;
use chops_core::wordpiece::Vocab;

pub fn explain(artifacts: &Path, query: &str, limit: usize) -> Result<()> {
    // ---- Load exactly what the browser would ---------------------------
    let meta_bytes =
        fs::read(artifacts.join("model.meta.bin")).context("reading model.meta.bin")?;
    let index_bytes = fs::read(artifacts.join("index.bin")).context("reading index.bin")?;
    let rows_bytes = fs::read(artifacts.join("model.rows.i8")).context("reading model.rows.i8")?;

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
    let words: Vec<&str> = Vocab::words(&norm)
        .into_iter()
        .filter(|w| w.chars().any(|c| c.is_alphanumeric()))
        .collect();
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
    let n = kw.n_docs as f32;
    let mut kw_scores: HashMap<u16, f32> = HashMap::new();
    let mut seen: Vec<&str> = Vec::new();
    for &w in &words {
        if seen.contains(&w) {
            continue;
        }
        seen.push(w);
        let Some(postings) = kw.terms.get(w) else {
            println!("keyword:   {w:?} matches no documents");
            continue;
        };
        let df = postings.len() as f32;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
        println!("keyword:   {w:?} df={df} idf={idf:.3}");
        for &(doc, tf) in postings {
            *kw_scores.entry(doc).or_insert(0.0) += idf * (1.0 + (tf as f32).ln());
        }
    }
    let kw_ranked = kw.rank(&words);

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
        "{:<4} {:>8} {:>4} {:>9} {:>5} {:>9} {:>7}  {}",
        "doc", "fused", "kw#", "kw-score", "sem#", "best-cos", "chunks", "title"
    );
    for &d in fused.iter().take(limit) {
        let kwr = rank_of(&kw_ranked, d);
        let smr = rank_of(&sem_ranked, d);
        let fused_score = kwr.map_or(0.0, |r| 1.0 / (rrf::K + (r + 1) as f32))
            + smr.map_or(0.0, |r| 1.0 / (rrf::K + (r + 1) as f32));
        let opt_rank = |r: Option<usize>| r.map_or_else(|| "-".into(), |r| (r + 1).to_string());
        println!(
            "{:<4} {:>8.5} {:>4} {:>9} {:>5} {:>9} {:>7}  {}",
            d,
            fused_score,
            opt_rank(kwr),
            kw_scores
                .get(&d)
                .map_or_else(|| "-".into(), |s| format!("{s:.3}")),
            opt_rank(smr),
            if best_cos[d as usize] > f32::NEG_INFINITY {
                format!("{:.3}", best_cos[d as usize])
            } else {
                "-".into()
            },
            chunk_count[d as usize],
            index.docs[d as usize].title,
        );
    }
    Ok(())
}
