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

use anstyle::{AnsiColor, Style};
use anyhow::{Context, Result, bail};
use chops_search_core::engine::Engine;
use chops_search_core::score::ScoreOpts;

// Color renders only on a tty (anstream strips otherwise), so the
// transcripts the sweep protocol diffs and greps stay byte-plain.
const GREEN: Style = AnsiColor::Green.on_default();
const YELLOW: Style = AnsiColor::Yellow.on_default();
const RED: Style = AnsiColor::Red.on_default().bold();
const HEADING: Style = Style::new().bold();

pub(crate) struct Case {
    pub(crate) q: String,
    pub(crate) expect: Vec<String>,
    pub(crate) kind: String,
}

#[derive(Default)]
pub(crate) struct Tally {
    pub(crate) n: usize,
    pub(crate) hit1: usize,
    pub(crate) hit3: usize,
}
/// One case's verdict under one ScoreOpts — the diffing primitive.
/// Passes over the same list diff by index; totals are forbidden as a
/// comparison basis, since a +2/−2 wash prints as zero.
#[derive(Debug, Clone)]
pub(crate) struct CaseOutcome {
    pub(crate) kind: String,
    pub(crate) q: String,
    pub(crate) hit1: bool,
    pub(crate) hit3: bool,
}

/// One full run of the case set under one ScoreOpts: the tallies, the
/// misses, and what the byte path cost. Sweep mode produces many of
/// these; normal mode exactly one.
#[derive(Default)]
pub(crate) struct Pass {
    by_kind: BTreeMap<String, Tally>,
    pub(crate) overall: Tally,
    failures: Vec<(String, String, Vec<String>)>,
    fetched: Vec<u64>,
    cold: usize,
    /// Per-case verdicts in case order. See CaseOutcome.
    pub(crate) outcomes: Vec<CaseOutcome>,
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

/// Everything one eval run needs beyond the artifacts and the case
/// file, as named fields rather than a parade of positional parameters.
/// Same transposition argument as ScoringFlags: `fail_under` and a
/// sweep value are both f32-shaped, clap names them at the flag layer,
/// and nothing protects the order once main.rs forwards them.
#[derive(Debug, Default, Clone)]
pub struct EvalArgs {
    pub kind_filter: Option<String>,
    pub fail_under: f32,
    /// Print the full explain for every failure, inline, on this run's
    /// own engine — flags applied, sweep cells included.
    pub explain: bool,
    pub scoring: ScoreArgs,
    pub sweep_rrf_k: Vec<f32>,
    pub sweep_rrf_alpha: Vec<f32>,
}

/// The engine plus the artifact bytes it was constructed from. One
/// struct because they are one thing: print_report's display inputs
/// must be the bytes behind the engine that ranked, and four separate
/// parameters are exactly how a call site pairs an engine with someone
/// else's index — the stale-artifact bug as an API shape.
pub(crate) struct EvalCtx<'a> {
    pub(crate) engine: &'a mut Engine,
    meta_bytes: &'a [u8],
    index_bytes: &'a [u8],
    pub(crate) rows_bytes: &'a [u8],
}

impl EvalCtx<'_> {
    /// Explain on this ctx's own engine and the bytes behind it — the
    /// pairing invariant as a method instead of three parameters a
    /// caller could mismatch.
    pub(crate) fn explain(&mut self, q: &str, limit: usize) -> Result<()> {
        crate::explain::print_report(self.engine, self.meta_bytes, self.index_bytes, q, limit)
    }
}

/// The owning counterpart of EvalCtx: the engine plus the bytes it was
/// built from, for callers that outlive one stack frame — calibrate
/// runs dozens of passes over a single load. EvalCtx stays the borrowed
/// view every pass and explain goes through, so the two commands cannot
/// pair an engine with someone else's index.
pub(crate) struct LoadedCtx {
    engine: Engine,
    meta_bytes: Vec<u8>,
    index_bytes: Vec<u8>,
    rows_bytes: Vec<u8>,
}

impl LoadedCtx {
    /// Artifacts exactly as the worker fetches them, prefix ingested,
    /// scoring applied on top of what the engine derived — same
    /// no-Default rationale as eval(): the floor and weights come from
    /// index.bin.
    pub(crate) fn load(artifacts: &Path, args: &ScoreArgs) -> Result<(LoadedCtx, ScoreOpts)> {
        let a = crate::artifacts::resolve(artifacts)?;
        let meta_bytes = fs::read(&a.meta).with_context(|| format!("{}", a.meta.display()))?;
        let index_bytes = fs::read(&a.index).with_context(|| format!("{}", a.index.display()))?;
        let prefix_bytes =
            fs::read(&a.prefix).with_context(|| format!("{}", a.prefix.display()))?;
        let rows_bytes = fs::read(&a.rows).with_context(|| format!("{}", a.rows.display()))?;

        let mut engine =
            Engine::new(&meta_bytes, &index_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
        let opts = args.apply(engine.score_opts());
        engine.set_score_opts(opts);
        engine
            .ingest(0, &prefix_bytes)
            .map_err(|e| anyhow::anyhow!("{e}"))?;
        Ok((
            LoadedCtx {
                engine,
                meta_bytes,
                index_bytes,
                rows_bytes,
            },
            opts,
        ))
    }

    /// The borrowed view run_pass, sweep, and explain consume.
    pub(crate) fn ctx(&mut self) -> EvalCtx<'_> {
        EvalCtx {
            engine: &mut self.engine,
            meta_bytes: &self.meta_bytes,
            index_bytes: &self.index_bytes,
            rows_bytes: &self.rows_bytes,
        }
    }

    /// Read access for the pre-walk tripwires (verify_expectations,
    /// corpus header) that must not hold the mutable view yet.
    pub(crate) fn engine(&self) -> &Engine {
        &self.engine
    }
}

/// The unknown-URL tripwire, extracted verbatim from eval(). Calibrate
/// runs it on the gate file AND the collateral file — a typo'd
/// expectation in known-failures would read as a phantom casualty.
pub(crate) fn verify_expectations(engine: &Engine, cases: &[Case]) -> Result<()> {
    let known: std::collections::HashSet<&str> = (0..engine.doc_count() as u16)
        .filter_map(|id| engine.doc_url(id))
        .collect();
    let mut unknown: Vec<String> = Vec::new();
    for case in cases {
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
    Ok(())
}

pub fn eval(artifacts: &Path, queries: &Path, args: &EvalArgs) -> Result<()> {
    let cases = load_cases(queries, args.kind_filter.as_deref())?;
    if cases.is_empty() {
        bail!("no cases matched (check --kind)");
    }

    // ---- Artifacts, exactly as the worker fetches them -----------------
    // One loader for eval and calibrate, so the two commands cannot load
    // different engines; the no-Default rationale (min_cos scales with
    // dimensionality, weights ride in index.bin) lives on LoadedCtx::load.
    let (mut loaded, opts) = LoadedCtx::load(artifacts, &args.scoring)?;
    println!("scoring:   {}", args.scoring.describe(&opts));

    // An `expect` URL that isn't in the corpus can never pass, and it
    // reads as a ranking failure rather than a typo. Fail loudly instead:
    // this exact class of bug — URLs that shifted under the index — once
    // looked like a total recall collapse for an entire run.
    verify_expectations(loaded.engine(), &cases)?;

    println!(
        "corpus:  {} docs, {} rows, prefix {} rows",
        loaded.engine().doc_count(),
        loaded.engine().n_rows(),
        loaded.engine().prefix_rows()
    );
    println!(
        "cases:   {} {}\n",
        cases.len(),
        args.kind_filter
            .as_deref()
            .map_or(String::new(), |k| format!("(kind = {k})"))
    );

    let mut ctx = loaded.ctx();

    if !args.sweep_rrf_k.is_empty() || !args.sweep_rrf_alpha.is_empty() {
        if args.fail_under > 0.0 {
            println!("note: --fail-under is ignored in sweep mode\n");
        }
        return sweep(
            &mut ctx,
            &cases,
            opts,
            &args.sweep_rrf_k,
            &args.sweep_rrf_alpha,
            args.explain,
        );
    }

    let pass = run_pass(&mut ctx, &cases, true)?;
    print_summary(&pass);
    print_bytes(&pass);
    print_failures(&pass, !args.explain);
    if args.explain {
        print_explains(&mut ctx, &pass)?;
        // The explains bury the numbers under pages of evidence;
        // restate the verdict so the last screen of the transcript is
        // the summary, not doc 12's cosine.
        anstream::println!("\n{HEADING}──── totals ────{HEADING:#}");
        print_summary(&pass);
    }

    let score = pct(pass.overall.hit1, pass.overall.n) / 100.0;
    if score < args.fail_under {
        bail!(
            "recall@1 {:.0}% below --fail-under {:.0}%",
            score * 100.0,
            args.fail_under * 100.0
        );
    }
    Ok(())
}

/// Grade one case's results. Negative controls (empty expect) invert
/// the test and pass only when nothing comes back, at BOTH depths:
/// "the junk was only at rank 3" is not partial credit for a query
/// that should return nothing.
fn judge(expect: &[String], urls: &[String]) -> (bool, bool) {
    if expect.is_empty() {
        let clean = urls.is_empty();
        (clean, clean)
    } else {
        (
            urls.first().is_some_and(|u| expect.contains(u)),
            urls.iter().any(|u| expect.contains(u)),
        )
    }
}

/// Run every case through the engine over the real byte path, under
/// whatever ScoreOpts the engine currently holds. Verbose prints the
/// per-case PASS/top3/FAIL lines; sweep mode runs quiet.
pub(crate) fn run_pass(ctx: &mut EvalCtx, cases: &[Case], verbose: bool) -> Result<Pass> {
    let mut pass = Pass::default();

    for case in cases {
        // Range-fetch simulation: the plan decides, we slice, engine ingests.
        let mut bytes_this_query = 0u64;
        for r in ctx.engine.plan(&case.q) {
            let (start, end) = (r.start as usize, r.end as usize);
            if end > ctx.rows_bytes.len() {
                bail!(
                    "plan asked for bytes {start}..{end} of a {}-byte rows file",
                    ctx.rows_bytes.len()
                );
            }
            bytes_this_query += (end - start) as u64;
            ctx.engine
                .ingest(r.start, &ctx.rows_bytes[start..end])
                .map_err(|e| anyhow::anyhow!("ingest at {start}: {e}"))?;
        }
        if bytes_this_query > 0 {
            pass.cold += 1;
        }
        pass.fetched.push(bytes_this_query);

        let ids = ctx.engine.search(&case.q, 3);
        let urls: Vec<String> = ids
            .iter()
            .map(|&id| ctx.engine.doc_url(id).unwrap_or("<missing>").to_string())
            .collect();

        // Negative controls invert the test: nothing should come back.
        let (hit1, hit3) = judge(&case.expect, &urls);

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

        pass.outcomes.push(CaseOutcome {
            kind: case.kind.clone(),
            q: case.q.clone(),
            hit1,
            hit3,
        });

        if verbose {
            let (mark, style) = if hit1 {
                ("PASS", GREEN)
            } else if hit3 {
                ("top3", YELLOW)
            } else {
                ("FAIL", RED)
            };
            anstream::println!(
                "{style}{mark}{style:#}  {:<13} {:<44} {}  [{}]",
                case.kind,
                truncate(&case.q, 44),
                urls.first().map_or("(no results)", String::as_str),
                if ctx.engine.used_semantic() {
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
    ctx: &mut EvalCtx,
    cases: &[Case],
    base: ScoreOpts,
    ks: &[f32],
    alphas: &[f32],
    explain: bool,
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
            ctx.engine.set_score_opts(o);
            let p = run_pass(ctx, cases, false)?;
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
    ctx.engine.set_score_opts(o);
    let p = run_pass(ctx, cases, false)?;
    print_summary(&p);
    print_failures(&p, !explain);
    if explain {
        // The engine holds the best cell's opts right now — the state
        // no standalone command can reproduce.
        print_explains(ctx, &p)?;
        anstream::println!(
            "{HEADING}{:<14} {:>4} {:>9.0}% {:>9.0}%{HEADING:#}",
            "OVERALL",
            p.overall.n,
            pct(p.overall.hit1, p.overall.n),
            pct(p.overall.hit3, p.overall.n)
        );
        print_summary(&p);
    }
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

/// The failure list as a string, so the hint-suppression rule is
/// testable without capturing stdout. The standalone-command hint is
/// dropped when inline explains follow: the hint reproduces neither
/// the flags nor a sweep cell, and printing it above the better tool's
/// output would recommend the worse one.
fn format_failures(pass: &Pass, hint: bool) -> String {
    if pass.failures.is_empty() {
        return String::new();
    }
    let mut out = String::from("\nfailures:\n");
    for (kind, q, urls) in &pass.failures {
        out.push_str(&format!("  [{kind}] {q:?}\n"));
        for (i, u) in urls.iter().enumerate() {
            out.push_str(&format!("      {}. {u}\n", i + 1));
        }
        if hint {
            out.push_str(&format!(
                "      explain: cargo run -p chops-search --release -- query {q:?}\n"
            ));
        }
    }
    out
}

fn print_failures(pass: &Pass, hint: bool) {
    print!("{}", format_failures(pass, hint));
}

/// Inline explains on eval's own engine, whose opts print_report reads
/// back rather than receiving. Rows are fully warm by the time any
/// failure exists. Limit 5, not query's 20: the diagnostic action is
/// the top handful plus the gate and floor lines.
fn print_explains(ctx: &mut EvalCtx, pass: &Pass) -> Result<()> {
    for (kind, q, _) in &pass.failures {
        anstream::println!("\n{HEADING}──── explain [{kind}] {q:?} ────{HEADING:#}");
        ctx.explain(q, 5)?;
    }
    Ok(())
}

pub(crate) fn pct(hit: usize, n: usize) -> f32 {
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

pub(crate) fn load_cases(path: &Path, kind_filter: Option<&str>) -> Result<Vec<Case>> {
    let text = fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    parse_cases(&text, kind_filter).with_context(|| format!("in {}", path.display()))
}

/// Parse the labeled set. Hand-navigated rather than serde-derived: the
/// schema is four fields and this keeps serde out of the dependency tree.
fn parse_cases(text: &str, kind_filter: Option<&str>) -> Result<Vec<Case>> {
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

    // ---- grading -----------------------------------------------------

    #[test]
    fn negative_controls_invert_at_both_depths() {
        let none: Vec<String> = vec![];
        assert_eq!(judge(&none, &[]), (true, true));
        // Junk at rank 3 is a full failure, not a recall@1-only one.
        let junk = vec!["/a/".to_string()];
        assert_eq!(judge(&none, &junk), (false, false));
    }

    #[test]
    fn hit1_needs_first_place_hit3_any_podium() {
        let expect = vec!["/right/".to_string()];
        let first = vec!["/right/".to_string(), "/other/".to_string()];
        assert_eq!(judge(&expect, &first), (true, true));
        let third = vec!["/a/".to_string(), "/b/".to_string(), "/right/".to_string()];
        assert_eq!(judge(&expect, &third), (false, true));
        assert_eq!(judge(&expect, &[]), (false, false));
    }

    // ---- fixture parsing ----------------------------------------------

    #[test]
    fn empty_expect_is_a_negative_missing_expect_is_an_error() {
        // Present-but-empty is the negative-control encoding; absent is
        // a typo. The distinction is the whole reason expect is required.
        let ok = parse_cases("[[query]]\nq = \"x\"\nexpect = []\n", None).unwrap();
        assert_eq!(ok.len(), 1);
        assert!(ok[0].expect.is_empty());
        assert!(parse_cases("[[query]]\nq = \"x\"\n", None).is_err());
    }

    #[test]
    fn kind_filter_selects_and_absent_kind_defaults_to_unlabeled() {
        let text = "[[query]]\nq = \"a\"\nexpect = [\"/a/\"]\nkind = \"exact\"\n\
                    [[query]]\nq = \"b\"\nexpect = [\"/b/\"]\n";
        let all = parse_cases(text, None).unwrap();
        assert_eq!(all.len(), 2);
        assert_eq!(all[1].kind, "unlabeled");
        let filtered = parse_cases(text, Some("exact")).unwrap();
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered[0].q, "a");
    }

    // ---- failure output -----------------------------------------------

    fn one_failure() -> Pass {
        Pass {
            failures: vec![(
                "exact".into(),
                "model2vec-rs".into(),
                vec!["/wrong/".into()],
            )],
            ..Default::default()
        }
    }

    #[test]
    fn failure_hint_is_suppressed_when_explains_follow() {
        // The hint reproduces neither flags nor sweep cells; printing it
        // above an inline explain would recommend the worse tool.
        let with = format_failures(&one_failure(), true);
        assert!(with.contains("explain: cargo run"), "{with}");
        let without = format_failures(&one_failure(), false);
        assert!(without.contains("model2vec-rs"), "{without}");
        assert!(!without.contains("cargo run"), "{without}");
        assert!(format_failures(&Pass::default(), true).is_empty());
    }

    #[test]
    fn pct_of_an_empty_set_is_zero_not_nan() {
        assert_eq!(pct(0, 0), 0.0);
    }
}
