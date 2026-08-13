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
//!
//! Sweep mode (`--sweep-rrf-k`, `--sweep-rrf-alpha`) runs the full case
//! set once per point of the k × alpha grid and prints recall per cell
//! instead of per-case lines, then re-runs the best cell verbosely so
//! its per-kind table and failure list come out of the same pass that
//! scored it. Every other scoring flag acts as the fixed base for every
//! cell. Rows warm on the first cell — plan() returns empty once a row
//! is resident — so a sweep costs one cold pass plus N−1 in-memory
//! passes, same as a browser session issuing many queries.

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

/// One full run of the case set under one ScoreOpts: the tallies, the
/// misses, and what the byte path cost. Sweep mode produces many of
/// these; normal mode exactly one.
#[derive(Default)]
struct Pass {
    by_kind: BTreeMap<String, Tally>,
    overall: Tally,
    failures: Vec<(String, String, Vec<String>)>,
    fetched: Vec<u64>,
    cold: usize,
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
///
/// `rrf_k` has no disabling value, unlike every other knob here: it is
/// the RRF rank discount (conventionally 60), always in effect, and only
/// its magnitude is in question. Small k sharpens the top of the curve
/// so a decisive #1 in one list can survive mediocrity in the other;
/// large k flattens toward "best average rank wins".
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
    pub rrf_alpha: Option<f32>,
    pub rrf_k: Option<f32>,
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
        if let Some(v) = self.rrf_alpha {
            o.rrf_alpha = v;
        }
        if let Some(v) = self.rrf_k {
            o.rrf_k = v;
        }
        o
    }

    /// The `scoring:` header. Shared so `eval` and `query` can't report
    /// different thresholds for the same flags — they diverged once.
    ///
    /// Flag-overridden knobs are marked `*`, because a transcript that
    /// can't distinguish "the index shipped 0.08" from "a flag injected
    /// 0.08" is the measured-at-0.34 header debt in a new denomination.
    /// Unmarked provenance is static and stated in the legend: weights,
    /// min_gap, rrf_alpha, and min_cos ride in index.bin; the rest are
    /// compiled defaults.
    pub fn describe(&self, o: &ScoreOpts) -> String {
        let m = |set: bool| if set { "*" } else { "" };
        let any_flag = [
            self.min_cos,
            self.chunk_penalty,
            self.kw_floor,
            self.min_gap,
            self.strong_cos,
            self.w_title,
            self.w_tag,
            self.w_desc,
            self.rrf_alpha,
            self.rrf_k,
        ]
        .iter()
        .any(Option::is_some);
        let mut s = format!(
            "kw_floor {:.2}{}, w_title {:.2}{}, w_tag {:.2}{}, w_desc {:.2}{}, \
             min_cos {:.2}{}, chunk_penalty {:.3}{}, min_gap {:.2}{}, strong_cos {}{}, \
             rrf_alpha {:.2}{}, rrf_k {:.1}{}",
            o.kw_confidence,
            m(self.kw_floor.is_some()),
            o.weights.title,
            m(self.w_title.is_some()),
            o.weights.tag,
            m(self.w_tag.is_some()),
            o.weights.desc,
            m(self.w_desc.is_some()),
            o.min_cos,
            m(self.min_cos.is_some()),
            o.chunk_penalty,
            m(self.chunk_penalty.is_some()),
            o.min_gap,
            m(self.min_gap.is_some()),
            if o.strong_cos.is_finite() {
                format!("{:.2}", o.strong_cos)
            } else {
                "off".into()
            },
            m(self.strong_cos.is_some()),
            o.rrf_alpha,
            m(self.rrf_alpha.is_some()),
            o.rrf_k,
            m(self.rrf_k.is_some()),
        );
        if any_flag {
            s.push_str("  (* = flag override)");
        }
        s
    }
}

#[allow(clippy::too_many_arguments)]
pub fn eval(
    artifacts: &Path,
    queries: &Path,
    kind_filter: Option<&str>,
    fail_under: f32,
    args: ScoreArgs,
    sweep_rrf_k: &[f32],
    sweep_rrf_alpha: &[f32],
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
    println!("scoring:   {}", args.describe(&opts));
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

    if !sweep_rrf_k.is_empty() || !sweep_rrf_alpha.is_empty() {
        // Gating an exploration would be meaningless: half the grid is
        // supposed to be worse than the baseline, that's what a sweep is.
        if fail_under > 0.0 {
            println!("note: --fail-under is ignored in sweep mode\n");
        }
        return sweep(
            &mut engine,
            &cases,
            &rows_bytes,
            opts,
            sweep_rrf_k,
            sweep_rrf_alpha,
        );
    }

    let pass = run_pass(&mut engine, &cases, &rows_bytes, true)?;
    print_summary(&pass);
    print_bytes(&pass);
    print_failures(&pass);

    let score = pct(pass.overall.hit1, pass.overall.n) / 100.0;
    if score < fail_under {
        bail!(
            "recall@1 {:.0}% below --fail-under {:.0}%",
            score * 100.0,
            fail_under * 100.0
        );
    }
    Ok(())
}

/// Run every case through the engine over the real byte path, under
/// whatever ScoreOpts the engine currently holds. Verbose prints the
/// per-case PASS/top3/FAIL lines; sweep mode runs quiet.
fn run_pass(engine: &mut Engine, cases: &[Case], rows_bytes: &[u8], verbose: bool) -> Result<Pass> {
    let mut pass = Pass::default();

    for case in cases {
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
            pass.cold += 1;
        }
        pass.fetched.push(bytes_this_query);

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

        let t = pass.by_kind.entry(case.kind.clone()).or_default();
        t.n += 1;
        pass.overall.n += 1;
        if hit1 {
            t.hit1 += 1;
            pass.overall.hit1 += 1;
        }
        if hit3 {
            t.hit3 += 1;
            pass.overall.hit3 += 1;
        }

        if verbose {
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
        }
        if !hit1 {
            pass.failures
                .push((case.kind.clone(), case.q.clone(), urls));
        }
    }
    Ok(pass)
}

/// The k × alpha grid. An empty axis is a singleton at the base value,
/// so `--sweep-rrf-k` alone sweeps one knob while the other stays put.
/// The best cell (recall@1, ties on recall@3, then first in grid order
/// for determinism) is re-run at the end for its per-kind breakdown and
/// failure list — that pass is warm, so it re-scores rather than
/// re-fetches, and its numbers are the same pass that won the grid.
fn sweep(
    engine: &mut Engine,
    cases: &[Case],
    rows_bytes: &[u8],
    base: ScoreOpts,
    ks: &[f32],
    alphas: &[f32],
) -> Result<()> {
    let ks: Vec<f32> = if ks.is_empty() {
        vec![base.rrf_k]
    } else {
        ks.to_vec()
    };
    let alphas: Vec<f32> = if alphas.is_empty() {
        vec![base.rrf_alpha]
    } else {
        alphas.to_vec()
    };

    println!(
        "sweep:   recall@1 (recall@3), {} cases per cell\n",
        cases.len()
    );
    print!("{:>11}", "");
    for a in &alphas {
        print!("{:>13}", format!("alpha {a:.2}"));
    }
    println!();

    // (k, alpha, hit1, hit3)
    let mut best: Option<(f32, f32, usize, usize)> = None;
    for &k in &ks {
        print!("{:>11}", format!("rrf_k {k:.1}"));
        for &a in &alphas {
            let mut o = base;
            o.rrf_k = k;
            o.rrf_alpha = a;
            engine.set_score_opts(o);
            let p = run_pass(engine, cases, rows_bytes, false)?;
            print!(
                "{:>13}",
                format!(
                    "{:.0}% ({:.0}%)",
                    pct(p.overall.hit1, p.overall.n),
                    pct(p.overall.hit3, p.overall.n)
                )
            );
            let better = match best {
                None => true,
                Some((_, _, h1, h3)) => {
                    p.overall.hit1 > h1 || (p.overall.hit1 == h1 && p.overall.hit3 > h3)
                }
            };
            if better {
                best = Some((k, a, p.overall.hit1, p.overall.hit3));
            }
        }
        println!();
    }

    let (k, a, h1, h3) = best.expect("grid has at least one cell");
    println!(
        "\nbest:    rrf_k {k:.1}, rrf_alpha {a:.2} → recall@1 {:.0}%, recall@3 {:.0}%\n",
        pct(h1, cases.len()),
        pct(h3, cases.len())
    );

    let mut o = base;
    o.rrf_k = k;
    o.rrf_alpha = a;
    engine.set_score_opts(o);
    let p = run_pass(engine, cases, rows_bytes, false)?;
    print_summary(&p);
    print_failures(&p);
    println!(
        "\nlock in alpha: set `rrf_alpha = {a}` in chops-search.toml and rebuild — \
         the value then reaches the browser, CI, and a bare eval alike.\n\
         rrf_k has no config key: if {k} ≠ 60 keeps winning sweeps, that is a \
         format-version conversation, not a flag to carry around."
    );
    Ok(())
}

fn print_summary(pass: &Pass) {
    println!(
        "\n{:<14} {:>4} {:>10} {:>10}",
        "kind", "n", "recall@1", "recall@3"
    );
    for (kind, t) in &pass.by_kind {
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
        pass.overall.n,
        pct(pass.overall.hit1, pass.overall.n),
        pct(pass.overall.hit3, pass.overall.n)
    );
}

fn print_bytes(pass: &Pass) {
    // Bytes: the product claim ("~1 KB per query") made checkable. Note
    // these shrink over a session as rows stay warm, so the first
    // occurrence of a term pays and later ones don't — same as a browser.
    let total: u64 = pass.fetched.iter().sum();
    let max = pass.fetched.iter().copied().max().unwrap_or(0);
    println!(
        "\nrange-fetched: {:.1} KB total, {:.1} KB mean, {:.1} KB worst; \
         {}/{} queries needed no fetch (prefix hit or warm)",
        total as f64 / 1024.0,
        total as f64 / 1024.0 / pass.fetched.len() as f64,
        max as f64 / 1024.0,
        pass.fetched.len() - pass.cold,
        pass.fetched.len()
    );
}

fn print_failures(pass: &Pass) {
    if pass.failures.is_empty() {
        return;
    }
    println!("\nfailures:");
    for (kind, q, urls) in &pass.failures {
        println!("  [{kind}] {q:?}");
        for (i, u) in urls.iter().enumerate() {
            println!("      {}. {u}", i + 1);
        }
        println!("      explain: cargo run -p chops-search --release -- query {q:?}");
    }
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
            rrf_alpha: 0.75,
            rrf_k: 12.0,
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
        assert_eq!(o.rrf_alpha, b.rrf_alpha);
        assert_eq!(o.rrf_k, b.rrf_k);
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
    fn rrf_alpha_overrides_alone() {
        // Fusion weighting is orthogonal to both engines: arming it must
        // not disturb a floor or a field weight, and sweeping a field
        // weight must not silently re-arm fusion.
        let armed = ScoreArgs {
            rrf_alpha: Some(2.0),
            ..Default::default()
        }
        .apply(base());
        assert_eq!(armed.rrf_alpha, 2.0);
        assert_eq!(armed.weights, base().weights);
        assert_eq!(armed.min_cos, base().min_cos);

        let swept = ScoreArgs {
            w_title: Some(1.0),
            ..Default::default()
        }
        .apply(base());
        assert_eq!(swept.rrf_alpha, base().rrf_alpha);
    }

    #[test]
    fn rrf_alpha_zero_is_an_override_not_an_absence() {
        // The knob's disabled value is 0, so `Some(0.0)` must reach
        // ScoreOpts rather than being mistaken for "unset". Turning
        // fusion weighting OFF against an index that shipped it on is a
        // real thing to want to measure.
        let o = ScoreArgs {
            rrf_alpha: Some(0.0),
            ..Default::default()
        }
        .apply(base());
        assert_eq!(o.rrf_alpha, 0.0);
    }

    #[test]
    fn rrf_k_overrides_alone() {
        // The other half of the fusion pair. Same orthogonality claim:

        // it must not drag rrf_alpha along.
        let o = ScoreArgs {
            rrf_k: Some(5.0),
            ..Default::default()
        }
        .apply(base());
        assert_eq!(o.rrf_k, 5.0);
        assert_eq!(o.rrf_alpha, base().rrf_alpha);
        assert_eq!(o.weights, base().weights);
        assert_eq!(o.min_cos, base().min_cos);

        let swept = ScoreArgs {
            rrf_alpha: Some(1.0),
            ..Default::default()
        }
        .apply(base());
        assert_eq!(swept.rrf_k, base().rrf_k);
    }

    #[test]
    fn describe_reports_every_knob() {
        let s = ScoreArgs::default().describe(&base());
        for knob in [
            "kw_floor",
            "w_title",
            "w_tag",
            "w_desc",
            "min_cos",
            "chunk_penalty",
            "min_gap",
            "strong_cos",
            "rrf_alpha",
            "rrf_k",
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
        let s = ScoreArgs::default().describe(&opts);
        assert!(s.contains("strong_cos off"), "{s}");
        assert!(!s.contains("inf"), "{s}");
    }

    #[test]
    fn describe_marks_flag_overrides_and_only_those() {
        // Provenance in the transcript: a sweep log must distinguish
        // "the index shipped 0.08" from "a flag injected 0.08" — the
        // measured-at-0.34 header debt in a new denomination.
        let flagged = ScoreArgs {
            min_gap: Some(0.11),
            ..Default::default()
        };
        let s = flagged.describe(&base());
        assert!(s.contains("min_gap 0.11*"), "{s}");
        assert!(
            !s.contains("rrf_alpha 0.75*"),
            "sibling must be unmarked: {s}"
        );
        assert!(s.contains("(* = flag override)"), "{s}");

        let clean = ScoreArgs::default().describe(&base());
        assert!(!clean.contains('*'), "no flags, no markers: {clean}");
    }
}
