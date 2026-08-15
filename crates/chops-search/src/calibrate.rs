//! `chops-search calibrate` — walk each scoring knob one value at a
//! time against the gate fixtures and report, with evidence, whether the
//! current value should stay.
//!
//! The design constraint is the whole feature: a tool that sweeps and
//! suggests is an argmax machine unless the suggestion rules encode the
//! measurement discipline. Encoded here, auditable here:
//!
//! 1. KEEP is the expected verdict. A plateau containing the current
//!    value, with no candidate beating it, prints as `keep <current>`
//!    with the plateau bounds and the nearest measured cliff. That is
//!    the tool working, not failing.
//! 2. A candidate (net ≥ +2 recall@1 flips on the gate cases) is
//!    NOMINATED, never adopted: the printout names every gained and
//!    lost case, re-runs the candidate against the collateral fixture
//!    and names casualties and rehabilitations there, and ends at
//!    "explain each flip before pinning". Promotion requires the
//!    per-case explain step — which `--explain` prints inline, on the
//!    engine holding the candidate's opts, the state no standalone
//!    `query` invocation can reproduce. The +2 floor is instrument
//!    resolution: on a ~40-case set one flipped case is a coin toss,
//!    and a grid rewards coin tosses.
//! 3. A candidate on a knob without a config key is a format-boundary
//!    conversation, not a config edit, and the printout says so.
//!
//! The tool never writes chops-search.toml.
//!
//! Comparison is per-case, never by totals: every pass carries its
//! outcome vector and cells diff against the baseline by index, so a
//! +2/−2 wash cannot print as "no change". Flips at both depths print
//! by name under their table row (`+`/`−` for recall@1, `+3`/`−3` for
//! recall@3) and both break a plateau, but only recall@1 nominates a
//! candidate: recall@3 movement is evidence worth naming, not a headline
//! worth acting on. One knob moves at a time and
//! the base is restored between knobs; the coupled rrf_k × rrf_alpha
//!
//! Cost: the baseline passes warm every row both fixture sets need
//! (plan() returns empty once a row is resident), so a full walk is one
//! cold pass plus in-memory re-scores — same shape as a long browser
//! session.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};

use crate::eval::{
    Case, CaseOutcome, LoadedCtx, Pass, ScoreArgs, load_cases, pct, run_pass, verify_expectations,
};
use crate::knob::Knob;

/// Smallest net recall@1 gain that earns a REVIEW nomination. See the
/// module header for why +1 is below instrument resolution.
const CANDIDATE_NET: i64 = 2;

pub struct CalibrateArgs {
    /// Knobs to walk; empty means all of `Knob::ALL`.
    pub knobs: Vec<Knob>,
    /// Explicit axis, allowed only with exactly one knob — an explicit
    /// axis applied to several knobs at once is how a min_gap value ends
    /// up swept over a weight scale. The current value is spliced in as
    /// an anchor cell (marked >) so the plateau always has something to
    /// be measured around, so a three-value list shows four rows.
    pub values: Vec<f32>,
    /// The known-failures fixture, for casualty checks on candidates.
    /// None runs candidates casualty-unchecked, and says so loudly.
    pub collateral: Option<PathBuf>,
    /// Print inline explains for every case a candidate flips, while
    /// the engine holds the candidate's opts.
    pub explain: bool,
    /// Fixed base under the walk, same flags as `eval`. Every cell of
    /// every knob starts from this base with exactly one field changed.
    pub score: ScoreArgs,
}

pub fn calibrate(artifacts: &Path, queries: &Path, args: CalibrateArgs) -> Result<()> {
    let knobs: Vec<Knob> = if args.knobs.is_empty() {
        Knob::ALL.to_vec()
    } else {
        args.knobs.clone()
    };
    if !args.values.is_empty() && knobs.len() != 1 {
        bail!("--values needs exactly one --knob, or the axis is ambiguous");
    }

    let cases = load_cases(queries, None)?;
    if cases.is_empty() {
        bail!("no cases in {}", queries.display());
    }
    let coll_cases: Option<(PathBuf, Vec<Case>)> = match &args.collateral {
        Some(p) => {
            let c = load_cases(p, None).with_context(|| format!("collateral {}", p.display()))?;
            Some((p.clone(), c))
        }
        None => None,
    };

    // Everything below runs on ONE loaded engine: the collateral check
    // and the gate walk must judge the same bytes, and the borrowed ctx
    // is how run_pass and explain stay on eval's exact path.
    let (mut loaded, base_opts) = LoadedCtx::load(artifacts, &args.score)?;
    println!("scoring:   {}", args.score.describe(&base_opts));
    verify_expectations(loaded.engine(), &cases)?;
    if let Some((_, c)) = &coll_cases {
        // A typo'd expectation in known-failures would read as a
        // phantom casualty, so the tripwire covers both files.
        verify_expectations(loaded.engine(), c)?;
    }
    println!("gate:    {} ({} cases)", queries.display(), cases.len());

    let mut ctx = loaded.ctx();

    // Collateral baselines up front, under the same base as everything
    // else; doing it first also warms its rows before any cell runs.
    let collateral: Option<(Vec<Case>, Pass)> = match coll_cases {
        Some((p, c)) => {
            println!("collateral: {} ({} cases)", p.display(), c.len());
            ctx.engine.set_score_opts(base_opts);
            let cpass = run_pass(&mut ctx, &c, false)?;
            Some((c, cpass))
        }
        None => {
            println!("collateral: none — candidates below are casualty-UNCHECKED");
            None
        }
    };

    ctx.engine.set_score_opts(base_opts);
    let baseline = run_pass(&mut ctx, &cases, false)?;
    println!(
        "baseline: recall@1 {:.0}% ({}/{}), recall@3 {:.0}%\n",
        pct(baseline.overall.hit1, baseline.overall.n),
        baseline.overall.hit1,
        baseline.overall.n,
        pct(baseline.overall.hit3, baseline.overall.n),
    );

    let mut suggestions: Vec<String> = Vec::new();

    for &knob in &knobs {
        let axis_raw = if args.values.is_empty() {
            knob.default_axis()
        } else {
            args.values.clone()
        };
        let current = knob.current(&base_opts);
        let (axis, cur_idx) = insert_current(axis_raw, current);

        // The walk. One field of one copy of the base changes per cell;
        // sweeping through ScoreOpts values rather than CLI flags is
        // what keeps the header's provenance story true (nothing here is
        // a flag override — it is the same engine re-judged).
        let mut cells: Vec<Cell> = Vec::with_capacity(axis.len());
        for &v in &axis {
            let mut o = base_opts;
            knob.apply(&mut o, v);
            ctx.engine.set_score_opts(o);
            let p = run_pass(&mut ctx, &cases, false)?;
            let flips = diff_outcomes(&baseline.outcomes, &p.outcomes);
            cells.push(Cell {
                value: v,
                hit1: p.overall.hit1,
                hit3: p.overall.hit3,
                flips,
            });
        }
        ctx.engine.set_score_opts(base_opts);

        let key = match knob.config_key() {
            Some(k) => format!("config key `{k}`"),
            None => "no config key".to_string(),
        };
        println!(
            "──── {} (current {}, {}) ────",
            knob.name(),
            fv(current),
            key
        );
        print!(
            "{}",
            knob_table(&cells, cur_idx, baseline.overall.n, &baseline.outcomes)
        );

        // Instrument tripwire: the cell at the current value re-runs the
        // baseline configuration, so any flip there is nondeterminism in
        // the instrument, not a finding.
        if !cells[cur_idx].flips.inert() {
            println!(
                "WARNING: re-running the current value flipped cases — \
                 nondeterminism in the instrument; distrust this table"
            );
        }

        // Candidates, casualty-checked against the collateral fixture,
        // and (with --explain) explained on the candidate's own engine
        // state while it still exists.
        let mut reports: Vec<CandidateReport> = Vec::new();
        for cell in cells.iter().filter(|c| c.flips.net1() >= CANDIDATE_NET) {
            let mut o = base_opts;
            knob.apply(&mut o, cell.value);
            ctx.engine.set_score_opts(o);

            let coll = match &collateral {
                Some((ccases, cbase)) => {
                    let p = run_pass(&mut ctx, ccases, false)?;
                    let f = diff_outcomes(&cbase.outcomes, &p.outcomes);
                    if args.explain {
                        for &i in &f.lost1 {
                            println!(
                                "\n──── explain casualty [{}] {:?} @ {} = {} ────",
                                cbase.outcomes[i].kind,
                                cbase.outcomes[i].q,
                                knob.name(),
                                fv(cell.value)
                            );
                            ctx.explain(&cbase.outcomes[i].q, 5)?;
                        }
                        for &i in &f.gained1 {
                            println!(
                                "\n──── explain rehabilitation [{}] {:?} @ {} = {} ────",
                                cbase.outcomes[i].kind,
                                cbase.outcomes[i].q,
                                knob.name(),
                                fv(cell.value)
                            );
                            ctx.explain(&cbase.outcomes[i].q, 5)?;
                        }
                    }
                    Some(CollateralFlips {
                        casualties: f.lost1.iter().map(|&i| label(&cbase.outcomes[i])).collect(),
                        rehabilitated: f
                            .gained1
                            .iter()
                            .map(|&i| label(&cbase.outcomes[i]))
                            .collect(),
                    })
                }
                None => None,
            };

            if args.explain {
                // Gate flips, both directions: a gain's explain shows
                // what the candidate ranking looks like when it wins, a
                // loss's shows what it broke. The baseline side of each
                // is one `eval --explain` away at the base config.
                for &i in cell.flips.gained1.iter().chain(&cell.flips.lost1) {
                    let dir = if cell.flips.gained1.contains(&i) {
                        "gained"
                    } else {
                        "lost"
                    };
                    println!(
                        "\n──── explain {dir} [{}] {:?} @ {} = {} ────",
                        baseline.outcomes[i].kind,
                        baseline.outcomes[i].q,
                        knob.name(),
                        fv(cell.value)
                    );
                    ctx.explain(&baseline.outcomes[i].q, 5)?;
                }
            }

            reports.push(CandidateReport {
                value: cell.value,
                gains: cell
                    .flips
                    .gained1
                    .iter()
                    .map(|&i| label(&baseline.outcomes[i]))
                    .collect(),
                losses: cell
                    .flips
                    .lost1
                    .iter()
                    .map(|&i| label(&baseline.outcomes[i]))
                    .collect(),
                collateral: coll,
            });
        }
        ctx.engine.set_score_opts(base_opts);

        let inert: Vec<bool> = cells.iter().map(|c| c.flips.inert()).collect();
        let s = suggestion(knob, &axis, &inert, cur_idx, &reports);
        println!("{s}\n");
        suggestions.push(format!(
            "{:<14} {}",
            knob.name(),
            s.replace('\n', "\n               ")
        ));
    }

    println!("──── suggestions ────");
    for s in &suggestions {
        println!("{s}");
    }
    println!(
        "\nNothing above was written anywhere. A REVIEW line is a nomination: \
         explain each named flip, then pin the value in chops-search.toml and \
         rebuild so it ships in index.bin."
    );
    Ok(())
}

/// One axis cell: the value, the tallies, and the per-case flips against
/// the baseline. Tallies ride along for the table; the flips are the
/// verdict currency.
struct Cell {
    value: f32,
    hit1: usize,
    hit3: usize,
    flips: Flips,
}

/// Per-case differences between two passes over the same case list, by
/// index into the baseline's outcome vector. Totals are deliberately not
/// stored here: a +2/−2 wash must stay visible as four names.
pub(crate) struct Flips {
    pub(crate) gained1: Vec<usize>,
    pub(crate) lost1: Vec<usize>,
    pub(crate) gained3: Vec<usize>,
    pub(crate) lost3: Vec<usize>,
}

impl Flips {
    pub(crate) fn inert(&self) -> bool {
        self.gained1.is_empty()
            && self.lost1.is_empty()
            && self.gained3.is_empty()
            && self.lost3.is_empty()
    }

    pub(crate) fn net1(&self) -> i64 {
        self.gained1.len() as i64 - self.lost1.len() as i64
    }

    fn summary(&self) -> String {
        let mut parts = Vec::new();
        if !self.gained1.is_empty() || !self.lost1.is_empty() {
            parts.push(format!("+{}/−{}@1", self.gained1.len(), self.lost1.len()));
        }
        if !self.gained3.is_empty() || !self.lost3.is_empty() {
            parts.push(format!("+{}/−{}@3", self.gained3.len(), self.lost3.len()));
        }
        if parts.is_empty() {
            "·".into()
        } else {
            parts.join(" ")
        }
    }
}

/// Diff two outcome vectors from passes over the same case list. Panics
/// on a list mismatch: that is a bug in the caller, not a measurement.
pub(crate) fn diff_outcomes(base: &[CaseOutcome], cand: &[CaseOutcome]) -> Flips {
    assert_eq!(base.len(), cand.len(), "passes ran different case lists");
    let mut f = Flips {
        gained1: Vec::new(),
        lost1: Vec::new(),
        gained3: Vec::new(),
        lost3: Vec::new(),
    };
    for (i, (b, c)) in base.iter().zip(cand).enumerate() {
        assert_eq!(b.q, c.q, "passes ran different case lists");
        match (b.hit1, c.hit1) {
            (false, true) => f.gained1.push(i),
            (true, false) => f.lost1.push(i),
            _ => {}
        }
        match (b.hit3, c.hit3) {
            (false, true) => f.gained3.push(i),
            (true, false) => f.lost3.push(i),
            _ => {}
        }
    }
    f
}

/// Sort the axis and splice the current value in, so the plateau math
/// always has a cell to anchor on and the table always shows where the
/// shipped value sits. Values within 1e-6 collapse to one cell.
pub(crate) fn insert_current(mut axis: Vec<f32>, current: f32) -> (Vec<f32>, usize) {
    axis.retain(|v| v.is_finite());
    axis.push(current);
    axis.sort_by(|a, b| a.partial_cmp(b).expect("finite by construction"));
    axis.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let idx = axis
        .iter()
        .position(|v| (v - current).abs() < 1e-6)
        .expect("current was just inserted");
    (axis, idx)
}

/// The maximal contiguous inert run containing cell `i`. Caller
/// guarantees `inert[i]` (the current cell diffs against itself).
pub(crate) fn plateau_around(inert: &[bool], i: usize) -> (usize, usize) {
    debug_assert!(inert[i]);
    let mut lo = i;
    while lo > 0 && inert[lo - 1] {
        lo -= 1;
    }
    let mut hi = i;
    while hi + 1 < inert.len() && inert[hi + 1] {
        hi += 1;
    }
    (lo, hi)
}

pub(crate) struct CollateralFlips {
    casualties: Vec<String>,
    rehabilitated: Vec<String>,
}

pub(crate) struct CandidateReport {
    value: f32,
    gains: Vec<String>,
    losses: Vec<String>,
    /// None when no collateral fixture was provided — a different claim
    /// from "checked, no flips", and printed differently.
    collateral: Option<CollateralFlips>,
}

/// The per-knob verdict, as a string so the rules are testable. The
/// three rules from the module header live here and nowhere else.
pub(crate) fn suggestion(
    knob: Knob,
    axis: &[f32],
    inert: &[bool],
    cur: usize,
    candidates: &[CandidateReport],
) -> String {
    if candidates.is_empty() {
        let current = fv(axis[cur]);
        if inert.iter().all(|&b| b) {
            return format!(
                "keep {current} — flat across the whole axis; the knob is inert \
                 on these cases at this base"
            );
        }
        let (lo, hi) = plateau_around(inert, cur);
        let below = if lo == 0 {
            "axis edge below (unmeasured further)".to_string()
        } else {
            format!("cliff below at {}", fv(axis[lo - 1]))
        };
        let above = if hi + 1 == axis.len() {
            "axis edge above (unmeasured further)".to_string()
        } else {
            format!("cliff above at {}", fv(axis[hi + 1]))
        };
        return format!(
            "keep {current} (plateau {}–{}; {below}, {above})",
            fv(axis[lo]),
            fv(axis[hi]),
        );
    }

    let mut s = String::new();
    for (i, c) in candidates.iter().enumerate() {
        if i > 0 {
            s.push('\n');
        }
        s.push_str(&format!(
            "REVIEW {} = {}: net {:+} recall@1 on gate\n",
            knob.name(),
            fv(c.value),
            c.gains.len() as i64 - c.losses.len() as i64,
        ));
        s.push_str(&format!("    gains:  {}\n", name_list(&c.gains)));
        s.push_str(&format!("    losses: {}\n", name_list(&c.losses)));
        match &c.collateral {
            Some(coll) => {
                s.push_str(&format!(
                    "    collateral: casualties {}; rehabilitated {}\n",
                    name_list(&coll.casualties),
                    name_list(&coll.rehabilitated),
                ));
                if !coll.rehabilitated.is_empty() {
                    s.push_str(
                        "    (a rehabilitation counts only once its mechanism is verified)\n",
                    );
                }
            }
            None => s.push_str("    collateral: NOT CHECKED — no known-failures fixture given\n"),
        }
        s.push_str(
            "    explain each flip before pinning; this tool nominates, promotion is a human act",
        );
        if knob.config_key().is_none() {
            s.push_str(&format!(
                "\n    note: {} has no config key — adopting a value is a \
                 format-boundary conversation, not a config edit",
                knob.name()
            ));
        }
        // min_cos is the one knob whose config key means something other
        // than "the value you measured": absent derives the floor from
        // dimensionality, present freezes it. current() reads a plain f32
        // off the engine, so nothing upstream of here can tell the two
        // apart — a REVIEW that says `config key min_cos` in the same
        // voice as min_gap would invite converting a derived floor into a
        // fixed one without saying so, and it would stay fixed across a
        // later --dims change where the derived floor would have moved.
        if knob == Knob::MinCos {
            s.push_str(
                "\n    note: min_cos in chops-search.toml is an override, not a \
                 default — absent derives the floor from dims, present freezes \
                 it across a future --dims change. Pinning converts a derived \
                 floor into a fixed one; say so in the commit.",
            );
        }
    }
    s
}

fn name_list(names: &[String]) -> String {
    if names.is_empty() {
        "(none)".into()
    } else {
        names.join(", ")
    }
}

fn label(o: &CaseOutcome) -> String {
    format!("[{}] {:?}", o.kind, o.q)
}

/// Render the walk table. Flip names print under their row — the table
/// is the per-case diff made legible, not a leaderboard.
fn knob_table(cells: &[Cell], cur: usize, n: usize, names: &[CaseOutcome]) -> String {
    use std::fmt::Write as _;
    let mut s = String::new();
    let _ = writeln!(
        s,
        "{:>2}{:>8} {:>9} {:>9}   flips",
        "", "value", "recall@1", "recall@3"
    );
    for (i, c) in cells.iter().enumerate() {
        let mark = if i == cur { ">" } else { " " };
        let _ = writeln!(
            s,
            "{mark:>2}{:>8} {:>8.0}% {:>8.0}%   {}",
            fv(c.value),
            pct(c.hit1, n),
            pct(c.hit3, n),
            c.flips.summary()
        );
        // Both depths by name. A case can legitimately appear twice —
        // rank 1 → rank 5 is a `−` line and a `−3` line — because those
        // are two facts. Only the @1 lines feed nomination (net1()).
        for (mark, idxs) in [
            ("+", &c.flips.gained1),
            ("−", &c.flips.lost1),
            ("+3", &c.flips.gained3),
            ("−3", &c.flips.lost3),
        ] {
            for &i in idxs {
                let _ = writeln!(s, "{:>14}{mark:<2} {}", "", label(&names[i]));
            }
        }
    }
    s
}

/// Trim a value for display: 0.120 → 0.12, 60.0 → 60, 0.05 stays 0.05.
fn fv(v: f32) -> String {
    let s = format!("{v:.3}");
    let s = s.trim_end_matches('0').trim_end_matches('.');
    if s.is_empty() || s == "-" {
        "0".to_string()
    } else {
        s.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oc(q: &str, hit1: bool, hit3: bool) -> CaseOutcome {
        CaseOutcome {
            kind: "exact".into(),
            q: q.into(),
            hit1,
            hit3,
        }
    }

    #[test]
    fn a_wash_stays_visible_as_names() {
        // The reason totals are forbidden: +1/−1 at identical recall.
        let base = vec![oc("a", true, true), oc("b", false, false)];
        let cand = vec![oc("a", false, false), oc("b", true, true)];
        let f = diff_outcomes(&base, &cand);
        assert_eq!(f.gained1, vec![1]);
        assert_eq!(f.lost1, vec![0]);
        assert_eq!(f.net1(), 0);
        assert!(!f.inert());
    }

    #[test]
    fn identical_passes_are_inert() {
        let base = vec![oc("a", true, true), oc("b", false, true)];
        let f = diff_outcomes(&base, &base.clone());
        assert!(f.inert());
        assert_eq!(f.summary(), "·");
    }

    #[test]
    #[should_panic(expected = "different case lists")]
    fn mismatched_case_lists_are_a_caller_bug() {
        let base = vec![oc("a", true, true)];
        let cand = vec![oc("b", true, true)];
        diff_outcomes(&base, &cand);
    }

    #[test]
    fn current_value_splices_into_the_axis() {
        let (axis, i) = insert_current(vec![0.0, 0.1, 0.2], 0.15);
        assert_eq!(axis, vec![0.0, 0.1, 0.15, 0.2]);
        assert_eq!(i, 2);
    }

    #[test]
    fn current_value_already_on_the_axis_dedups() {
        let (axis, i) = insert_current(vec![0.0, 0.1, 0.2], 0.1);
        assert_eq!(axis, vec![0.0, 0.1, 0.2]);
        assert_eq!(i, 1);
    }

    #[test]
    fn plateau_is_the_contiguous_inert_run() {
        //            0      1     2      3     4
        let inert = [false, true, true, true, false];
        assert_eq!(plateau_around(&inert, 2), (1, 3));
        assert_eq!(plateau_around(&inert, 1), (1, 3));
    }

    #[test]
    fn keep_states_plateau_and_cliffs() {
        let axis = [0.0, 0.04, 0.08, 0.12, 0.16];
        let inert = [false, true, true, true, false];
        let s = suggestion(Knob::MinGap, &axis, &inert, 2, &[]);
        assert!(s.starts_with("keep 0.08"), "{s}");
        assert!(s.contains("plateau 0.04–0.12"), "{s}");
        assert!(s.contains("cliff below at 0"), "{s}");
        assert!(s.contains("cliff above at 0.16"), "{s}");
    }

    #[test]
    fn an_axis_edge_is_not_called_a_cliff() {
        // Beyond the last measured cell is unmeasured, and the wording
        // must not promise a margin nobody measured.
        let axis = [0.1, 0.2, 0.3];
        let inert = [true, true, true];
        let s = suggestion(Knob::WTag, &axis, &inert, 1, &[]);
        assert!(s.contains("flat across the whole axis"), "{s}");
        let inert2 = [false, true, true];
        let s2 = suggestion(Knob::WTag, &axis, &inert2, 1, &[]);
        assert!(s2.contains("axis edge above (unmeasured"), "{s2}");
        assert!(!s2.contains("cliff above"), "{s2}");
    }

    #[test]
    fn candidates_are_nominated_never_adopted() {
        let cand = CandidateReport {
            value: 0.12,
            gains: vec![
                "[paraphrase] \"hide a page\"".into(),
                "[exact] \"prefix_rows\"".into(),
            ],
            losses: vec![],
            collateral: Some(CollateralFlips {
                casualties: vec!["[paraphrase] \"make the download smaller\"".into()],
                rehabilitated: vec![],
            }),
        };
        let s = suggestion(Knob::MinGap, &[0.08, 0.12], &[true, false], 0, &[cand]);
        assert!(s.contains("REVIEW min_gap = 0.12"), "{s}");
        assert!(s.contains("net +2"), "{s}");
        assert!(s.contains("casualties [paraphrase]"), "{s}");
        assert!(s.contains("explain each flip before pinning"), "{s}");
        assert!(!s.to_lowercase().contains("keep "), "{s}");
    }

    #[test]
    fn unchecked_collateral_is_loud() {
        let cand = CandidateReport {
            value: 8.0,
            gains: vec!["a".into(), "b".into()],
            losses: vec![],
            collateral: None,
        };
        let s = suggestion(Knob::RrfK, &[8.0, 60.0], &[false, true], 1, &[cand]);
        assert!(s.contains("NOT CHECKED"), "{s}");
        // Rule 3: no config key means a format-boundary note.
        assert!(s.contains("format-boundary conversation"), "{s}");
    }

    #[test]
    fn display_values_trim_honestly() {
        assert_eq!(fv(0.120), "0.12");
        assert_eq!(fv(0.05), "0.05");
        assert_eq!(fv(60.0), "60");
        assert_eq!(fv(0.0), "0");
    }

    #[test]
    fn a_min_cos_candidate_warns_about_override_semantics() {
        // min_cos is the only knob whose config key changes the KIND of
        // value (derived → fixed), not just its magnitude.
        let cand = CandidateReport {
            value: 0.32,
            gains: vec!["a".into(), "b".into()],
            losses: vec![],
            collateral: Some(CollateralFlips {
                casualties: vec![],
                rehabilitated: vec![],
            }),
        };
        let s = suggestion(Knob::MinCos, &[0.28, 0.32], &[true, false], 0, &[cand]);
        assert!(s.contains("override, not a default"), "{s}");
        assert!(s.contains("--dims"), "{s}");
        // Not the format-boundary note: min_cos HAS a config key.
        assert!(!s.contains("format-boundary"), "{s}");

        // And the note is min_cos-only: min_gap's key means what it says.
        let cand2 = CandidateReport {
            value: 0.12,
            gains: vec!["a".into(), "b".into()],
            losses: vec![],
            collateral: None,
        };
        let s2 = suggestion(Knob::MinGap, &[0.08, 0.12], &[true, false], 0, &[cand2]);
        assert!(!s2.contains("override, not a default"), "{s2}");
    }
    #[test]
    fn top3_only_movement_is_a_flip_not_inertia() {
        // The answer slid from rank 4 to rank 2: recall@1 is unmoved but
        // the cell must not read as "nothing happened".
        let base = vec![oc("a", false, false)];
        let cand = vec![oc("a", false, true)];
        let f = diff_outcomes(&base, &cand);
        assert!(!f.inert());
        assert_eq!(f.net1(), 0);
        assert_eq!(f.summary(), "+1/−0@3");

        // ...and the table must NAME it. A plateau edge decided by a
        // flip the table cannot name is the totals-not-names failure in
        // one row.
        let cell = Cell {
            value: 0.1,
            hit1: 0,
            hit3: 1,
            flips: f,
        };
        let t = knob_table(&[cell], 0, 1, &base);
        assert!(t.contains("+3 [exact] \"a\""), "{t}");
        assert!(
            !t.contains("+  [exact]"),
            "no @1 line for a @3-only move: {t}"
        );
    }

    #[test]
    fn a_case_that_falls_off_the_podium_prints_two_lines() {
        // rank 1 → rank 5: lost @1 AND lost @3. Two facts, two lines,
        // and it is a candidate LOSS at @1 (net −1), not just a @3 note.
        let base = vec![oc("a", true, true)];
        let cand = vec![oc("a", false, false)];
        let f = diff_outcomes(&base, &cand);
        assert_eq!(f.net1(), -1);
        let cell = Cell {
            value: 0.2,
            hit1: 0,
            hit3: 0,
            flips: f,
        };
        let t = knob_table(&[cell], 0, 1, &base);
        assert!(t.contains("−  [exact] \"a\""), "{t}");
        assert!(t.contains("−3 [exact] \"a\""), "{t}");
    }
}
