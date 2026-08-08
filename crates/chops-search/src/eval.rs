//! `chops-search eval` — score the engine against a labeled query set.
//!
//! Deliberately drives `chops_search_core::engine::Engine` over the SAME byte
//! path the browser uses: construct from meta + index, ingest only the
//! eager prefix, then per query plan() → slice the rows file → ingest()
//! each range → search(). A reimplementation of the ranking here would
//! score a thing the browser doesn't run; this way a bug in plan(),
//! coalesce(), or RowStore::missing shows up as a recall drop, and the
//! bytes-per-query numbers are the real ones.
//!
//! Recall@1 is the headline because that's what a search box is: people
//! click the first result. Recall@3 is reported alongside so you can see
//! whether a regression moved the right answer off the podium or merely
//! off the top step.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use chops_search_core::engine::Engine;
use chops_search_core::score::ScoreOpts;

struct Case {
    q: String,
    expect: Vec<String>,
    kind: String,
}

#[derive(Default)]
struct Tally {
    n: usize,
    hit1: usize,
    hit3: usize,
}
/// Scoring overrides from the command line. `None` means "whatever the
/// engine derived from the artifacts" — min_cos scales with
/// dimensionality and the field weights are read out of index.bin, so
/// starting from ScoreOpts::default() would reset the floor to its
/// 256-dim value and the weights to the compiled-in defaults on a run
/// that overrides only one other field.
///
/// Flat `Option<f32>` fields rather than an `Option<FieldWeights>`: these
/// map one-to-one onto CLI flags, each is independently overridable, and
/// clap names them, so the transposition hazard that justifies the struct
/// inside the engine does not exist here.
#[derive(Debug, Default, Clone, Copy)]
pub struct ScoreArgs {
    pub min_cos: Option<f32>,
    pub chunk_penalty: Option<f32>,
    pub kw_floor: Option<f32>,
    pub min_gap: Option<f32>,
    pub strong_cos: Option<f32>,
    pub w_title: Option<f32>,
    pub w_tag: Option<f32>,
    pub w_desc: Option<f32>,
}

impl ScoreArgs {
    /// Apply to a base — always `engine.score_opts()`, never `default()`.
    pub fn apply(&self, mut o: ScoreOpts) -> ScoreOpts {
        if let Some(v) = self.min_cos {
            o.min_cos = v;
        }
        if let Some(v) = self.chunk_penalty {
            o.chunk_penalty = v;
        }
        if let Some(v) = self.kw_floor {
            o.kw_confidence = v;
        }
        if let Some(v) = self.min_gap {
            o.min_gap = v;
        }
        if let Some(v) = self.strong_cos {
            o.strong_cos = v;
        }
        // Mutated in place rather than rebuilding the struct: overriding
        // one weight must leave the other two exactly as index.bin set
        // them, and a `FieldWeights { title: v, ..Default::default() }`
        // here would quietly reset the others to the compiled-in values.
        if let Some(v) = self.w_title {
            o.weights.title = v;
        }
        if let Some(v) = self.w_tag {
            o.weights.tag = v;
        }
        if let Some(v) = self.w_desc {
            o.weights.desc = v;
        }
        o
    }

    /// The `scoring:` header. Shared so `eval` and `query` can't report
    /// different thresholds for the same flags — they diverged once,
    /// when this existed and `eval` printed its own copy anyway.
    ///
    /// Keyword knobs first, then the semantic ones. strong_cos prints
    /// "off" at infinity rather than `inf`: it disables at infinity while
    /// every other knob disables at zero.
    pub fn describe(o: &ScoreOpts) -> String {
        format!(
            "kw_floor {:.2}, w_title {:.2}, w_tag {:.2}, w_desc {:.2}, \
             min_cos {:.2}, chunk_penalty {:.3}, min_gap {:.2}, strong_cos {}",
            o.kw_confidence,
            o.weights.title,
            o.weights.tag,
            o.weights.desc,
            o.min_cos,
            o.chunk_penalty,
            o.min_gap,
            if o.strong_cos.is_finite() {
                format!("{:.2}", o.strong_cos)
            } else {
                "off".into()
            }
        )
    }
}

pub fn eval(
    artifacts: &Path,
    queries: &Path,
    kind_filter: Option<&str>,
    fail_under: f32,
    args: ScoreArgs,
) -> Result<()> {
    let cases = load_cases(queries, kind_filter)?;
    if cases.is_empty() {
        bail!("no cases matched (check --kind)");
    }

    // ---- Artifacts, exactly as the worker fetches them -----------------
    let a = crate::artifacts::resolve(artifacts)?;
    let meta_bytes = fs::read(&a.meta).with_context(|| format!("{}", a.meta.display()))?;
    let index_bytes = fs::read(&a.index).with_context(|| format!("{}", a.index.display()))?;
    let prefix_bytes = fs::read(&a.prefix).with_context(|| format!("{}", a.prefix.display()))?;
    let rows_bytes = fs::read(&a.rows).with_context(|| format!("{}", a.rows.display()))?;

    let mut engine = Engine::new(&meta_bytes, &index_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    // Start from what the engine derived, not from Default: min_cos
    // scales with dimensionality and the BM25F weights come from
    // index.bin, neither of which Default knows. Starting from Default
    // would silently reset the floor to its 256-dim value and the
    // weights to the compiled-in constants on any run that passes only
    // --chunk-penalty.
    let opts = args.apply(engine.score_opts());
    engine.set_score_opts(opts);
    println!("scoring:   {}", ScoreArgs::describe(&opts));
    engine
        .ingest(0, &prefix_bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))?;

    // An `expect` URL that isn't in the corpus can never pass, and it
    // reads as a ranking failure rather than a typo. Fail loudly instead:
    // this exact class of bug — URLs that shifted under the index — once
    // looked like a total recall collapse for an entire run.
    let known: std::collections::HashSet<&str> = (0..engine.doc_count() as u16)
        .filter_map(|id| engine.doc_url(id))
        .collect();
    let mut unknown: Vec<String> = Vec::new();
    for case in &cases {
        for url in &case.expect {
            if !known.contains(url.as_str()) {
                unknown.push(format!("  {:?} expects {url}", case.q));
            }
        }
    }
    if !unknown.is_empty() {
        eprintln!("query set references URLs not in the index:");
        for u in &unknown {
            eprintln!("{u}");
        }
        bail!(
            "{} unknown URL(s) — run `chops-search docs` to see what's indexed",
            unknown.len()
        );
    }
    println!(
        "corpus:  {} docs, {} rows, prefix {} rows",
        engine.doc_count(),
        engine.n_rows(),
        engine.prefix_rows()
    );
    println!(
        "cases:   {} {}\n",
        cases.len(),
        kind_filter.map_or(String::new(), |k| format!("(kind = {k})"))
    );

    let mut by_kind: BTreeMap<String, Tally> = BTreeMap::new();
    let mut overall = Tally::default();
    let mut failures: Vec<(String, String, Vec<String>)> = Vec::new();
    let mut fetched: Vec<u64> = Vec::new();
    let mut cold = 0usize;

    for case in &cases {
        // Range-fetch simulation: the plan decides, we slice, engine ingests.
        let mut bytes_this_query = 0u64;
        for r in engine.plan(&case.q) {
            let (start, end) = (r.start as usize, r.end as usize);
            if end > rows_bytes.len() {
                bail!(
                    "plan asked for bytes {start}..{end} of a {}-byte rows file",
                    rows_bytes.len()
                );
            }
            bytes_this_query += (end - start) as u64;
            engine
                .ingest(r.start, &rows_bytes[start..end])
                .map_err(|e| anyhow::anyhow!("ingest at {start}: {e}"))?;
        }
        if bytes_this_query > 0 {
            cold += 1;
        }
        fetched.push(bytes_this_query);

        let ids = engine.search(&case.q, 3);
        let urls: Vec<String> = ids
            .iter()
            .map(|&id| engine.doc_url(id).unwrap_or("<missing>").to_string())
            .collect();

        // Negative controls invert the test: nothing should come back.
        let (hit1, hit3) = if case.expect.is_empty() {
            (urls.is_empty(), urls.is_empty())
        } else {
            (
                urls.first().is_some_and(|u| case.expect.contains(u)),
                urls.iter().any(|u| case.expect.contains(u)),
            )
        };

        let t = by_kind.entry(case.kind.clone()).or_default();
        t.n += 1;
        overall.n += 1;
        if hit1 {
            t.hit1 += 1;
            overall.hit1 += 1;
        }
        if hit3 {
            t.hit3 += 1;
            overall.hit3 += 1;
        }

        let mark = if hit1 {
            "PASS"
        } else if hit3 {
            "top3"
        } else {
            "FAIL"
        };
        println!(
            "{mark}  {:<13} {:<44} {}  [{}]",
            case.kind,
            truncate(&case.q, 44),
            urls.first().map_or("(no results)", String::as_str),
            if engine.used_semantic() {
                "hybrid"
            } else {
                "kw"
            }
        );
        if !hit1 {
            failures.push((case.kind.clone(), case.q.clone(), urls));
        }
    }

    // ---- Summary -------------------------------------------------------
    println!(
        "\n{:<14} {:>4} {:>10} {:>10}",
        "kind", "n", "recall@1", "recall@3"
    );
    for (kind, t) in &by_kind {
        println!(
            "{:<14} {:>4} {:>9.0}% {:>9.0}%",
            kind,
            t.n,
            pct(t.hit1, t.n),
            pct(t.hit3, t.n)
        );
    }
    println!(
        "{:<14} {:>4} {:>9.0}% {:>9.0}%",
        "OVERALL",
        overall.n,
        pct(overall.hit1, overall.n),
        pct(overall.hit3, overall.n)
    );

    // Bytes: the product claim ("~1 KB per query") made checkable. Note
    // these shrink over a session as rows stay warm, so the first
    // occurrence of a term pays and later ones don't — same as a browser.
    let total: u64 = fetched.iter().sum();
    let max = fetched.iter().copied().max().unwrap_or(0);
    println!(
        "\nrange-fetched: {:.1} KB total, {:.1} KB mean, {:.1} KB worst; \
         {}/{} queries needed no fetch (prefix hit or warm)",
        total as f64 / 1024.0,
        total as f64 / 1024.0 / fetched.len() as f64,
        max as f64 / 1024.0,
        fetched.len() - cold,
        fetched.len()
    );

    if !failures.is_empty() {
        println!("\nfailures:");
        for (kind, q, urls) in &failures {
            println!("  [{kind}] {q:?}");
            for (i, u) in urls.iter().enumerate() {
                println!("      {}. {u}", i + 1);
            }
            println!("      explain: cargo run -p chops-search --release -- query {q:?}");
        }
    }

    let score = pct(overall.hit1, overall.n) / 100.0;
    if score < fail_under {
        bail!(
            "recall@1 {:.0}% below --fail-under {:.0}%",
            score * 100.0,
            fail_under * 100.0
        );
    }
    Ok(())
}

fn pct(hit: usize, n: usize) -> f32 {
    if n == 0 {
        0.0
    } else {
        hit as f32 * 100.0 / n as f32
    }
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}

/// Parse the labeled set. Hand-navigated rather than serde-derived: the
/// schema is four fields and this keeps serde out of the dependency tree.
fn load_cases(path: &Path, kind_filter: Option<&str>) -> Result<Vec<Case>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let doc: toml::Table = text.parse().context("query set is not valid TOML")?;
    let arr = doc
        .get("query")
        .and_then(|v| v.as_array())
        .context("expected an array of [[query]] tables")?;

    let mut out = Vec::new();
    for (i, item) in arr.iter().enumerate() {
        let t = item
            .as_table()
            .with_context(|| format!("query {i} is not a table"))?;
        let q = t
            .get("q")
            .and_then(|v| v.as_str())
            .with_context(|| format!("query {i} has no string `q`"))?
            .to_string();
        // `expect` present-but-empty is meaningful (negative control), so
        // a MISSING key is the error case, not an empty list.
        let expect_val = t
            .get("expect")
            .and_then(|v| v.as_array())
            .with_context(|| format!("query {q:?} has no `expect` array"))?;
        let expect: Vec<String> = expect_val
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
        let kind = t
            .get("kind")
            .and_then(|v| v.as_str())
            .unwrap_or("unlabeled")
            .to_string();
        if kind_filter.is_some_and(|k| k != kind) {
            continue;
        }
        out.push(Case { q, expect, kind });
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chops_search_core::keyword::FieldWeights;

    /// A base that resembles what an engine hands over: nothing at its
    /// compiled-in default, so any field apply() forgets shows up.
    fn base() -> ScoreOpts {
        ScoreOpts {
            min_cos: 0.34,
            chunk_penalty: 0.05,
            kw_confidence: 0.25,
            min_gap: 0.11,
            strong_cos: 0.55,
            weights: FieldWeights {
                title: 3.0,
                tag: 7.0,
                desc: 0.5,
            },
        }
    }

    #[test]
    fn no_flags_is_the_identity() {
        // The whole reason apply() takes a base: a run with no scoring
        // flags must score exactly as the artifacts say, not as the
        // compiled-in defaults say.
        let o = ScoreArgs::default().apply(base());
        let b = base();
        assert_eq!(o.min_cos, b.min_cos);
        assert_eq!(o.chunk_penalty, b.chunk_penalty);
        assert_eq!(o.kw_confidence, b.kw_confidence);
        assert_eq!(o.min_gap, b.min_gap);
        assert_eq!(o.strong_cos, b.strong_cos);
        assert_eq!(o.weights, b.weights);
    }

    #[test]
    fn one_flag_leaves_the_rest_alone() {
        let args = ScoreArgs {
            w_title: Some(0.0),
            ..Default::default()
        };
        let o = args.apply(base());
        assert_eq!(o.weights.title, 0.0, "zero is a real sweep point");
        assert_eq!(o.weights.tag, 7.0, "sibling weights must survive");
        assert_eq!(o.weights.desc, 0.5, "sibling weights must survive");
        assert_eq!(o.min_cos, 0.34, "the index-derived floor must survive");
    }

    #[test]
    fn each_weight_overrides_only_itself() {
        // Three same-typed flags landing on three same-typed fields is
        // exactly where a copy-paste error hides. Set each in turn and
        // assert the other two are untouched.
        let cases = [
            (
                ScoreArgs {
                    w_title: Some(9.0),
                    ..Default::default()
                },
                (9.0, 7.0, 0.5),
            ),
            (
                ScoreArgs {
                    w_tag: Some(9.0),
                    ..Default::default()
                },
                (3.0, 9.0, 0.5),
            ),
            (
                ScoreArgs {
                    w_desc: Some(9.0),
                    ..Default::default()
                },
                (3.0, 7.0, 9.0),
            ),
        ];
        for (args, (title, tag, desc)) in cases {
            let o = args.apply(base());
            assert_eq!(
                (o.weights.title, o.weights.tag, o.weights.desc),
                (title, tag, desc)
            );
        }
    }

    #[test]
    fn describe_reports_every_knob() {
        let s = ScoreArgs::describe(&base());
        for knob in [
            "kw_floor",
            "w_title",
            "w_tag",
            "w_desc",
            "min_cos",
            "chunk_penalty",
            "min_gap",
            "strong_cos",
        ] {
            assert!(s.contains(knob), "{knob} missing from {s:?}");
        }
    }

    #[test]
    fn describe_prints_an_infinite_strong_cos_as_off() {
        let opts = ScoreOpts {
            strong_cos: f32::INFINITY,
            ..base()
        };
        let s = ScoreArgs::describe(&opts);
        assert!(s.contains("strong_cos off"), "{s}");
        assert!(!s.contains("inf"), "{s}");
    }
}
