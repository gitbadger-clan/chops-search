//! `chops-search plan` — the network-cost instrument.
//!
//! Every scoring knob has been through calibrate: pre-registered,
//! per-case diffed, plateau-bounded. The two NETWORK knobs — the eager
//! prefix size and the range-coalescing gap — never had an instrument.
//! This is it: for one query or a fixture, what the browser's first
//! query would fetch from model.rows.i8, and how that changes as the
//! two knobs move.
//!
//! Simulation instead of rebuilds: rows are frequency-ordered, so "a
//! prefix of N rows" is exactly "row ids below N", and any prefix size
//! is a filter over token ids. Any gap is an argument to the same
//! `coalesce` the engine calls. The simulator is checked against
//! `Engine::plan` at the shipped cell over every case before any
//! simulated cell prints — a wrong simulator here would be eval and
//! explain disagreeing, in a new denomination.
//!
//! `--curl` emits one command per range against the hashed rows file,
//! each printing the status and byte count the server returned — the
//! "don't trust the table" check, and the article's demo. It is refused
//! at simulated cells: curling bytes the browser would not fetch is the
//! dishonest demo this command exists to replace.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

use anyhow::{Context, Result, bail};
use chops_search_core::engine::{Engine, MAX_GAP_ROWS};
use chops_search_core::format::ModelMeta;
use chops_search_core::plan::{ByteRange, coalesce};
use chops_search_core::wordpiece::Vocab;

use crate::eval::{Case, load_cases};

pub struct PlanArgs<'a> {
    /// One query → detail mode. None → aggregate over `queries`.
    pub query: Option<&'a str>,
    /// Fixture for aggregate mode.
    pub queries: &'a Path,
    pub kind: Option<&'a str>,
    /// Prefix sizes to simulate. Empty = the artifact's value.
    pub prefix_rows: Vec<u32>,
    /// Gap sizes to simulate. Empty = MAX_GAP_ROWS.
    pub max_gap: Vec<u32>,
    /// Emit curl against this base URL (the directory serving the
    /// artifacts). Only at the shipped cell — clap enforces.
    pub curl: Option<&'a str>,
    /// Rows in the missing-tokens table.
    pub top: usize,
}

/// HTTP `Range` is inclusive on both ends; `plan()` ranges are
/// half-open. The one place the conventions meet, so a function with a
/// test rather than an inline `- 1`.
pub(crate) fn range_header(r: &ByteRange) -> String {
    format!("{}-{}", r.start, r.end - 1)
}

/// Rows covered by a row-aligned byte range, inclusive.
pub(crate) fn rows_of(r: &ByteRange, dim: u32) -> (u32, u32) {
    debug_assert!(
        r.start.is_multiple_of(dim) && r.end.is_multiple_of(dim),
        "range not row-aligned"
    );
    (r.start / dim, r.end / dim - 1)
}

fn human(bytes: u64) -> String {
    if bytes >= 1 << 20 {
        format!("{:.2} MB", bytes as f64 / (1u64 << 20) as f64)
    } else if bytes >= 1 << 10 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{bytes} B")
    }
}

/// The planner as arithmetic over row ids, at one (prefix, gap) cell.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Sim {
    prefix_rows: u32,
    max_gap: u32,
    dim: u32,
}

/// One query's network cost at one cell.
#[derive(Default, Clone, Copy)]
struct Cost {
    /// Unique rows the query needs; OOV words are already gone.
    needed: usize,
    /// Of those, rows the prefix already holds.
    hit: usize,
    requests: usize,
    bytes: u64,
}

impl Cost {
    /// Bytes fetched for rows nobody asked for — the price of merging
    /// across a gap. What --max-gap trades against request count.
    fn dead(&self, dim: u32) -> u64 {
        self.bytes - (self.needed - self.hit) as u64 * u64::from(dim)
    }
}

impl Sim {
    /// Rows the browser would need from the network: not in the prefix,
    /// sorted, deduplicated — the same contract RowStore::missing gives
    /// coalesce.
    fn missing(&self, ids: &[u32]) -> Vec<u32> {
        let mut m: Vec<u32> = ids
            .iter()
            .copied()
            .filter(|&id| id >= self.prefix_rows)
            .collect();
        m.sort_unstable();
        m.dedup();
        m
    }

    fn plan(&self, ids: &[u32]) -> Vec<ByteRange> {
        coalesce(&self.missing(ids), self.dim, self.max_gap)
    }

    fn cost(&self, ids: &[u32]) -> Cost {
        let mut uniq = ids.to_vec();
        uniq.sort_unstable();
        uniq.dedup();
        let needed = uniq.len();
        let missing = self.missing(&uniq);
        let ranges = coalesce(&missing, self.dim, self.max_gap);
        Cost {
            needed,
            hit: needed - missing.len(),
            requests: ranges.len(),
            bytes: ranges.iter().map(|r| u64::from(r.end - r.start)).sum(),
        }
    }
}

/// Aggregate over the fixture at one cell.
#[derive(Default)]
struct Cell {
    n: usize,
    needed: usize,
    hit: usize,
    zero_fetch: usize,
    req_sum: usize,
    req_max: usize,
    bytes_sum: u64,
    bytes_max: u64,
    dead_sum: u64,
}

impl Cell {
    fn measure(sim: Sim, cases: &[(String, Vec<u32>)]) -> Cell {
        let mut c = Cell::default();
        for (_, ids) in cases {
            let k = sim.cost(ids);
            c.n += 1;
            c.needed += k.needed;
            c.hit += k.hit;
            if k.requests == 0 {
                c.zero_fetch += 1;
            }
            c.req_sum += k.requests;
            c.req_max = c.req_max.max(k.requests);
            c.bytes_sum += k.bytes;
            c.bytes_max = c.bytes_max.max(k.bytes);
            c.dead_sum += k.dead(sim.dim);
        }
        c
    }
    fn hit_pct(&self) -> f64 {
        if self.needed == 0 {
            100.0
        } else {
            100.0 * self.hit as f64 / self.needed as f64
        }
    }
    fn dead_pct(&self) -> f64 {
        if self.bytes_sum == 0 {
            0.0
        } else {
            100.0 * self.dead_sum as f64 / self.bytes_sum as f64
        }
    }
}

/// The simulator must reproduce Engine::plan exactly at the shipped
/// cell, on the engine the browser would run. Runs before any
/// simulated cell prints; a mismatch is a bug in this file, never in
/// the engine, and is reported as such.
fn parity_check(engine: &Engine, sim: Sim, cases: &[(String, Vec<u32>)]) -> Result<()> {
    for (q, ids) in cases {
        let real = engine.plan(q);
        let ours = sim.plan(ids);
        if real != ours {
            bail!(
                "planner parity failed on {q:?}: engine {real:?} vs simulator {ours:?} \
                 (prefix {}, gap {}) — bug in plan.rs, not in the engine",
                sim.prefix_rows,
                sim.max_gap
            );
        }
    }
    Ok(())
}

/// Prefix sizes that would cover the given percentages of needed row
/// occurrences across the fixture. THE number prefix_rows should be
/// argued from: "2048 covers 91% of what the gate fixture needs" is a
/// measurement; "2048 felt right" is not.
fn coverage(cases: &[(String, Vec<u32>)], pcts: &[u32]) -> Vec<(u32, u32)> {
    let mut all: Vec<u32> = Vec::new();
    for (_, ids) in cases {
        let mut u = ids.clone();
        u.sort_unstable();
        u.dedup();
        all.extend(u);
    }
    all.sort_unstable();
    if all.is_empty() {
        return pcts.iter().map(|&p| (p, 0)).collect();
    }
    pcts.iter()
        .map(|&p| {
            let idx = ((all.len() * p as usize).div_ceil(100)).max(1) - 1;
            (p, all[idx] + 1) // row id → prefix size that includes it
        })
        .collect()
}

pub fn plan(artifacts: &Path, args: &PlanArgs) -> Result<()> {
    // ---- Artifacts, as the worker fetches them --------------------------
    let a = crate::artifacts::resolve(artifacts)?;
    let meta_bytes = fs::read(&a.meta).with_context(|| format!("{}", a.meta.display()))?;
    let index_bytes = fs::read(&a.index).with_context(|| format!("{}", a.index.display()))?;
    let prefix_bytes = fs::read(&a.prefix).with_context(|| format!("{}", a.prefix.display()))?;
    let rows_len = fs::metadata(&a.rows)
        .with_context(|| format!("{}", a.rows.display()))?
        .len();
    let rows_name = a
        .rows
        .file_name()
        .and_then(|s| s.to_str())
        .context("rows artifact has no filename")?;

    let mut engine = Engine::new(&meta_bytes, &index_bytes).map_err(|e| anyhow::anyhow!("{e}"))?;
    engine
        .ingest(0, &prefix_bytes)
        .map_err(|e| anyhow::anyhow!("{e}"))?;
    let dim = engine.dim() as u32;
    let n_rows = engine.n_rows() as u32;

    // Display vocab, same as explain: token id == matrix row.
    let meta = ModelMeta::read(&meta_bytes).context("parsing model.meta.bin")?;
    let vocab = Vocab::from_tokens(&meta.tokens);
    let piece = |id: u32| meta.tokens[id as usize].as_str();

    // ---- Cases ---------------------------------------------------------
    let cases: Vec<Case> = match args.query {
        Some(q) => vec![Case {
            q: q.to_string(),
            expect: Vec::new(),
            kind: "single".to_string(),
        }],
        None => load_cases(args.queries, args.kind)?,
    };
    if cases.is_empty() {
        bail!("no cases matched (check --kind)");
    }
    let tokenized: Vec<(String, Vec<u32>)> = cases
        .iter()
        .map(|c| (c.q.clone(), vocab.tokenize(&c.q)))
        .collect();

    // ---- Shipped cell + parity ------------------------------------------
    let shipped = Sim {
        prefix_rows: engine.prefix_rows(),
        max_gap: MAX_GAP_ROWS,
        dim,
    };
    parity_check(&engine, shipped, &tokenized)?;

    let prefix_axis: Vec<u32> = if args.prefix_rows.is_empty() {
        vec![shipped.prefix_rows]
    } else {
        args.prefix_rows.clone()
    };
    for &p in &prefix_axis {
        if p > n_rows {
            bail!("--prefix-rows {p} exceeds the model's {n_rows} rows");
        }
    }
    let gap_axis: Vec<u32> = if args.max_gap.is_empty() {
        vec![shipped.max_gap]
    } else {
        args.max_gap.clone()
    };

    // With --curl, human output goes to stderr so stdout is a script.
    let to_err = args.curl.is_some();
    let diag = |s: String| {
        if to_err {
            eprintln!("{s}")
        } else {
            println!("{s}")
        }
    };

    diag(format!(
        "rows file: {rows_name} ({}, {n_rows} rows × {dim} B); shipped prefix {} rows \
         ({}), gap {}",
        human(rows_len),
        shipped.prefix_rows,
        human(u64::from(shipped.prefix_rows) * u64::from(dim)),
        shipped.max_gap
    ));

    // ---- Single-query detail ---------------------------------------------
    if let Some(q) = args.query {
        if prefix_axis.len() > 1 || gap_axis.len() > 1 {
            bail!("knob lists need a fixture: drop QUERY or use --queries");
        }
        let sim = Sim {
            prefix_rows: prefix_axis[0],
            max_gap: gap_axis[0],
            dim,
        };
        let ids = &tokenized[0].1;
        let ranges = sim.plan(ids);
        let cost = sim.cost(ids);

        diag(format!("query:     {q:?}"));
        if sim != shipped {
            diag(format!(
                "state:     SIMULATED prefix {} rows, gap {} (not what ships)",
                sim.prefix_rows, sim.max_gap
            ));
        } else {
            diag(format!(
                "state:     prefix rows 0..{} loaded, row cache cold",
                sim.prefix_rows
            ));
        }
        if ids.is_empty() {
            diag(
                "tokens:    none — every word is out of vocabulary; the browser \
                  searches keyword-only and fetches nothing"
                    .into(),
            );
        } else {
            diag(format!(
                "{:<10} {:>7}  {:<16} where",
                "tokens:", "row", "piece"
            ));
            for &id in ids {
                let byte = id * dim;
                let where_ = if id < sim.prefix_rows {
                    "prefix (already loaded)".to_string()
                } else {
                    match ranges.iter().position(|r| byte >= r.start && byte < r.end) {
                        Some(i) => format!("range #{}", i + 1),
                        None => unreachable!("missing row not covered by any range"),
                    }
                };
                diag(format!("{:<10} {id:>7}  {:<16} {where_}", "", piece(id)));
            }
        }
        if ranges.is_empty() {
            diag("plan:      nothing to fetch".into());
            if args.curl.is_some() {
                println!("# nothing to fetch for {q:?}");
            }
            return Ok(());
        }
        diag(format!(
            "{:<10} {:<20} {:<14} {:>8}  tokens",
            "ranges:", "bytes", "rows", "size"
        ));
        for (i, r) in ranges.iter().enumerate() {
            let (ra, rb) = rows_of(r, dim);
            let inside: Vec<&str> = ids
                .iter()
                .filter(|&&id| id >= ra && id <= rb)
                .map(|&id| piece(id))
                .collect();
            diag(format!(
                "{:<10} {:<20} {:<14} {:>8}  {}",
                format!("#{}", i + 1),
                range_header(r),
                if ra == rb {
                    ra.to_string()
                } else {
                    format!("{ra}-{rb}")
                },
                human(u64::from(r.end - r.start)),
                inside.join(" ")
            ));
        }
        diag(format!(
            "plan:      {} request(s), {} of {} ({:.4}%), {} dead",
            ranges.len(),
            human(cost.bytes),
            human(rows_len),
            100.0 * cost.bytes as f64 / rows_len as f64,
            human(cost.dead(dim))
        ));
        if let Some(base) = args.curl {
            emit_curl(base, rows_name, q, &ranges);
        }
        return Ok(());
    }

    // ---- Aggregate: the instrument -----------------------------------------
    diag(format!(
        "fixture:   {} ({} case(s){})",
        args.queries.display(),
        tokenized.len(),
        args.kind.map(|k| format!(", kind {k}")).unwrap_or_default()
    ));
    let cov = coverage(&tokenized, &[50, 90, 99, 100]);
    diag(format!(
        "coverage:  prefix rows needed for {}",
        cov.iter()
            .map(|(p, n)| format!("{p}% = {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    ));
    diag(String::new());
    diag(format!(
        "{:>7} {:>4} {:>9} {:>6} {:>7} {:>10} {:>10} {:>5} {:>4} {:>6}  ",
        "prefix", "gap", "eager", "hit%", "0fetch%", "bytes/q", "max", "req/q", "max", "dead%"
    ));
    for &p in &prefix_axis {
        for &g in &gap_axis {
            let sim = Sim {
                prefix_rows: p,
                max_gap: g,
                dim,
            };
            let c = Cell::measure(sim, &tokenized);
            diag(format!(
                "{p:>7} {g:>4} {:>9} {:>6.1} {:>7.1} {:>10} {:>10} {:>5.2} {:>4} {:>6.1}  {}",
                human(u64::from(p) * u64::from(dim)),
                c.hit_pct(),
                100.0 * c.zero_fetch as f64 / c.n as f64,
                human(c.bytes_sum / c.n as u64),
                human(c.bytes_max),
                c.req_sum as f64 / c.n as f64,
                c.req_max,
                c.dead_pct(),
                if sim == shipped { "← shipped" } else { "" }
            ));
        }
    }

    // Missing tokens at the FIRST prefix on the axis: with no flag that
    // is the shipped prefix; with a list, the first value is the one the
    // reader is asking about. Row id doubles as distance from the
    // boundary — a token at row 2100 under prefix 2048 is a different
    // argument from one at row 19000.
    let at = prefix_axis[0];
    let mut missing: BTreeMap<u32, usize> = BTreeMap::new();
    for (_, ids) in &tokenized {
        let mut u = ids.clone();
        u.sort_unstable();
        u.dedup();
        for id in u.into_iter().filter(|&id| id >= at) {
            *missing.entry(id).or_default() += 1;
        }
    }
    let mut ranked: Vec<(u32, usize)> = missing.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    diag(String::new());
    if ranked.is_empty() {
        diag(format!(
            "missing:   none — prefix {at} covers every row the fixture needs"
        ));
    } else {
        diag(format!(
            "missing:   {} distinct rows outside prefix {at}; top {} by case count",
            ranked.len(),
            args.top.min(ranked.len())
        ));
        diag(format!("{:<10} {:>7}  {:<20} cases", "", "row", "piece"));
        for (id, n) in ranked.iter().take(args.top) {
            diag(format!("{:<10} {id:>7}  {:<20} {n}", "", piece(*id)));
        }
    }

    // Costliest cases at the first cell. The max columns above are
    // per-case numbers; the instrument names the case rather than
    // leaving the reader to guess — and the top entry is the query
    // that shows range fetching best.
    let first = Sim {
        prefix_rows: prefix_axis[0],
        max_gap: gap_axis[0],
        dim,
    };
    let mut costly: Vec<(&str, Cost)> = tokenized
        .iter()
        .map(|(q, ids)| (q.as_str(), first.cost(ids)))
        .filter(|(_, c)| c.requests > 0)
        .collect();
    costly.sort_by(|a, b| {
        b.1.bytes
            .cmp(&a.1.bytes)
            .then(b.1.requests.cmp(&a.1.requests))
    });
    if !costly.is_empty() {
        diag(String::new());
        diag(format!(
            "costliest: top {} of {} fetching case(s) at prefix {}, gap {}",
            args.top.min(costly.len()),
            costly.len(),
            first.prefix_rows,
            first.max_gap
        ));
        diag(format!("{:<10} {:>4} {:>8}  query", "", "req", "bytes"));
        for (q, c) in costly.iter().take(args.top) {
            diag(format!(
                "{:<10} {:>4} {:>8}  {q:?}",
                "",
                c.requests,
                human(c.bytes)
            ));
        }
    }

    if let Some(base) = args.curl {
        // clap keeps us at the shipped cell here.
        for (q, ids) in &tokenized {
            let ranges = shipped.plan(ids);
            if ranges.is_empty() {
                println!("# {q:?}: nothing to fetch");
            } else {
                emit_curl(base, rows_name, q, &ranges);
            }
        }
    }
    Ok(())
}

/// One curl per range. `-w` prints what the SERVER returned, so the
/// arithmetic is verified over HTTP, not by this tool: status must be
/// 206 and the byte count must equal end - start + 1. A 200 with a
/// multi-megabyte count is a range-hostile host — the condition
/// search-worker.js detects with its rangeHostile flag.
fn emit_curl(base: &str, rows_name: &str, q: &str, ranges: &[ByteRange]) {
    let url = format!("{}/{rows_name}", base.trim_end_matches('/'));
    let total: u64 = ranges.iter().map(|r| u64::from(r.end - r.start)).sum();
    println!(
        "# {q:?}: {} request(s), {total} bytes — expect 206 per line",
        ranges.len()
    );
    for r in ranges {
        let hdr = range_header(r);
        println!(
            "curl -sS -r {hdr} '{url}' -o /dev/null \
             -w '{hdr}  %{{http_code}}  %{{size_download}} bytes\\n'"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sim(prefix_rows: u32, max_gap: u32) -> Sim {
        Sim {
            prefix_rows,
            max_gap,
            dim: 128,
        }
    }

    #[test]
    fn http_range_is_inclusive_where_plan_is_half_open() {
        let r = ByteRange {
            start: 640,
            end: 768,
        };
        assert_eq!(range_header(&r), "640-767");
        assert_eq!(767 - 640 + 1, r.end - r.start);
    }

    #[test]
    fn rows_of_single_and_run() {
        assert_eq!(
            rows_of(
                &ByteRange {
                    start: 640,
                    end: 768
                },
                128
            ),
            (5, 5)
        );
        assert_eq!(
            rows_of(
                &ByteRange {
                    start: 0,
                    end: 10 * 128
                },
                128
            ),
            (0, 9)
        );
    }

    #[test]
    fn prefix_filters_by_id_and_dedups() {
        // 3 and 5 sit inside prefix 10; 12 twice collapses to one row.
        assert_eq!(sim(10, 8).missing(&[12, 3, 12, 5, 40]), vec![12, 40]);
    }

    #[test]
    fn cost_counts_hits_and_dead_bytes() {
        // needed {3, 12, 40}; prefix covers 3; 12 and 40 are 28 rows
        // apart, so at gap 8 they split (2 requests, 0 dead), at gap 30
        // they merge into rows 12..=40 (1 request, 27 dead rows).
        let split = sim(10, 8).cost(&[3, 12, 40]);
        assert_eq!((split.needed, split.hit, split.requests), (3, 1, 2));
        assert_eq!(split.dead(128), 0);
        let merged = sim(10, 30).cost(&[3, 12, 40]);
        assert_eq!(merged.requests, 1);
        assert_eq!(merged.dead(128), 27 * 128);
    }

    #[test]
    fn coverage_returns_prefix_that_includes_the_row() {
        let cases = vec![
            ("a".into(), vec![1, 5]),
            ("b".into(), vec![5, 200]),
            ("c".into(), vec![9]),
        ];
        // occurrences sorted: 1,5,5,9,200 → 100% needs prefix 201, 50%
        // (ceil(2.5)=3rd → 5) needs 6.
        let cov = coverage(&cases, &[50, 100]);
        assert_eq!(cov, vec![(50, 6), (100, 201)]);
    }

    #[test]
    fn empty_query_costs_nothing() {
        let c = sim(2048, 8).cost(&[]);
        assert_eq!((c.needed, c.hit, c.requests, c.bytes), (0, 0, 0, 0));
        assert!(sim(0, 8).plan(&[]).is_empty());
    }
}
