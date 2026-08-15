//! `chops-search init` — scaffold a site's integration.
//!
//! The gap between "installed the binary" and "search works" was five
//! manual steps: write a config, create a page with the right element
//! ids, add gitignore lines, link two assets, and know that the runtime
//! lands in `out`. Every one of those is a place to get it subtly wrong
//! and see an empty results list with no error.
//!
//! What this does NOT do is edit templates. Where a search box belongs is
//! theme-specific — tabi wants `templates/tabi/extend_head.html`, another
//! theme wants `base.html`, a bare site wants neither — and a tool that
//! rewrites someone's layout on their behalf is a tool people stop
//! trusting. It prints the line and lets you place it.
//!
//! Nothing is ever overwritten. Re-running after editing the scaffold is
//! safe and reports what it skipped.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

const CONFIG: &str = r#"# chops-search — https://github.com/gitbadger-clan/chops-search
#
# Paths resolve relative to THIS file, so `chops-search build` works from
# anywhere inside the site.
content = "content"
out     = "static/search"
model   = ".chops-search/model"
# PCA target. The model's native size is 256; 128 halves the eager prefix
# and every per-query range fetch. Re-run `chops-search eval` after
# changing it — dimensionality reduction is real information loss.
dims = 128
# BM25F field weights: what a length-normalised occurrence in each field
# is worth against one in the body. Each field's term frequency is divided
# by that field's OWN average length before these apply, and saturation
# happens once on the combined value, so a weight biases without
# inflating. The useful range is therefore much smaller than a plain
# multiplier's would be. 0 ignores a field entirely.
#
# Tags are the author's own statement of what a page is about, so they
# outweigh the title, which outweighs body text. `description` sits at
# parity with body: it is indexed because it is written in the register
# searchers phrase questions in, but it measured no better weighted up.
#
# These are query-time knobs, so sweep them against a built index rather
# than guessing: `chops-search eval --w-title 1 --w-desc 0` needs no
# rebuild. Only the value committed here reaches the browser.
# title_weight = 2
# tag_weight   = 4
# desc_weight  = 1
# chunk_chars = 600
# prefix_rows = 2048

# Scoring calibration. Unlike the weights above, these are baked into
# index.bin at build time and read by the engine at construction, so
# the value committed here is the value every visitor's browser runs
# — and a bare `chops-search eval` measures the same configuration.
#
# They ship commented out because they are CALIBRATED, not chosen: a
# value that helps one corpus hurts another, and uncommenting an example
# without measuring is shipping someone else's calibration. The loop is:
# sweep with `chops-search eval` against a labeled fixture set, verify
# the mechanism with the explain output, then pin the winning value
# here with a dated comment saying what was measured, and rebuild.
#
# min_gap: the corroboration gate. When a query has no keyword evidence
# and no document stands out from the pack by at least this much (best
# cosine minus corpus median), the semantic ranking is suppressed rather
# than served — the flat-field signature is noise on a single-topic
# corpus. 0 disarms the gate, which is the compiled default.
# Sweep: chops-search eval --min-gap
# min_gap = 0.08
#
# rrf_alpha: confidence-weighted fusion. The keyword list's RRF vote
# scales by 1 + rrf_alpha × the fraction of the query's idf mass it
# matched, so an exact rare-term hit can outvote a merely-adjacent
# semantic first place, while stopword-heavy queries fuse as plain RRF.
# 0 is plain RRF, the compiled default.
# Sweep: chops-search eval --rrf-alpha
# rrf_alpha = 1.0
#
# chunk_penalty: expected-max correction. A document's semantic score is
# its BEST chunk's cosine, so many-chunk documents hold more lottery
# tickets; this subtracts coeff × sqrt(2 ln chunks) to offset that bias.
# Matters only when chunk counts vary widely across the corpus. The
# compiled default is 0.02.
# Sweep: chops-search eval --chunk-penalty
# chunk_penalty = 0.05
#
# min_cos: the semantic relevance floor, as an OVERRIDE. Leave this
# commented and the engine derives the floor from `dims`, which is right
# for almost every corpus and tracks dims changes automatically. Pin it
# only if a sweep said so — a value calibrated at one dims is wrong at
# another — and note 0 is itself an override, meaning "floor off".
# min_cos = 0.34
"#;

const SEARCH_PAGE: &str = r#"+++
title = "Search"
# Zola silently drops pages with no sort key from a sorted section, so a
# search page without one renders nowhere and 404s. Both keys are here
# because a section may sort by either, and neither value is meaningful
# for a page that is not content.
date = 1970-01-01
weight = 999
# The search page has nothing to find; indexing it makes it a result for
# every query typed into it.
in_search_index = false
+++

<div class="chops">
  <div class="chops-bar">
    <span class="chops-icon"></span>
    <input id="chops-input" type="search" placeholder="Search…"
           autocomplete="off" spellcheck="false"
           role="combobox" aria-expanded="false" aria-controls="chops-results"
           aria-autocomplete="list">
    <button type="button" class="chops-clear" data-chops-clear></button>
  </div>
  <span id="chops-mode" class="chops-mode" aria-live="polite"></span>
  <ul id="chops-results" class="chops-results" role="listbox"></ul>
</div>

<link rel="stylesheet" href="/search/chops-search.css">
<script defer src="/search/chops-search.js"></script>
"#;

const GITIGNORE_LINES: &[(&str, &str)] = &[
    ("/static/search/", "generated by `chops-search build`"),
    (
        "/.chops-search/model/",
        "~30 MB; the lockfile beside it is what to commit",
    ),
];

/// Scaffold into `root`. `with_page` controls whether a dedicated search
/// page is created — sites using only the site-wide overlay don't need
/// one.
pub fn init(root: &Path, with_page: bool) -> Result<()> {
    let mut created = Vec::new();
    let mut skipped = Vec::new();

    write_new(
        &root.join("chops-search.toml"),
        CONFIG,
        &mut created,
        &mut skipped,
    )?;

    if with_page {
        let content = root.join("content");
        if content.is_dir() {
            write_new(
                &content.join("search.md"),
                SEARCH_PAGE,
                &mut created,
                &mut skipped,
            )?;
        } else {
            skipped.push(format!(
                "content/search.md (no content/ directory — is {} a Zola site?)",
                root.display()
            ));
        }
    }

    // Appended rather than written: a site's .gitignore is not ours to
    // replace, and duplicate entries are noise in every future diff.
    let gitignore = root.join(".gitignore");
    let existing = fs::read_to_string(&gitignore).unwrap_or_default();
    // Compare the pattern only. A previously written line may carry a
    // trailing comment, so a whole-line match would never fire and every
    // run would append again.
    let existing_patterns: Vec<&str> = existing
        .lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|p| !p.is_empty())
        .collect();
    let missing: Vec<&(&str, &str)> = GITIGNORE_LINES
        .iter()
        .filter(|(line, _)| !existing_patterns.contains(line))
        .collect();
    if missing.is_empty() {
        skipped.push(".gitignore (already covered)".to_string());
    } else {
        let mut out = existing;
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str("\n# chops-search\n");
        for (line, why) in &missing {
            out.push_str(&format!("# {why}\n{line}\n"));
        }
        fs::write(&gitignore, out).with_context(|| format!("writing {}", gitignore.display()))?;
        created.push(format!(".gitignore (+{} lines)", missing.len()));
    }

    for c in &created {
        println!("  created  {c}");
    }
    for s in &skipped {
        println!("  exists   {s}");
    }

    println!(
        r#"
Next:

  1. chops-search model fetch      download the embedding model (~30 MB, once)
  2. chops-search build            artifacts + runtime -> static/search/
  3. zola serve                    then visit /search/

For a search box on EVERY page, add this to your base template
(tabi: templates/tabi/extend_head.html). The script mounts its own overlay
on pages with no #chops-input, opened with Ctrl/Cmd-K or /:

  <link rel="stylesheet" href="{{{{ get_url(path='search/chops-search.css') }}}}">
  <script defer src="{{{{ get_url(path='search/chops-search.js') }}}}"></script>

Also set `build_search_index = false` in config.toml. Zola's own index is
dead weight alongside this one.
"#
    );
    Ok(())
}

fn write_new(
    path: &Path,
    body: &str,
    created: &mut Vec<String>,
    skipped: &mut Vec<String>,
) -> Result<()> {
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    if path.exists() {
        skipped.push(name);
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, body).with_context(|| format!("writing {}", path.display()))?;
    created.push(name);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::CONFIG;
    use super::*;
    use crate::config::Config;
    use std::path::PathBuf;

    fn tmp(name: &str) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("chops-init-{name}-{}", std::process::id()));
        fs::create_dir_all(p.join("content")).unwrap();
        p
    }
    fn root() -> PathBuf {
        PathBuf::from("/site")
    }

    /// Strip the comment marker from lines shaped like `# key = value`,
    /// leaving prose comments alone. This is what a user's editor does
    /// when they adopt an example, so the test exercises exactly the
    /// text they will end up with.
    fn uncommented(template: &str) -> String {
        template
            .lines()
            .map(|l| {
                let Some(rest) = l.strip_prefix("# ") else {
                    return l.to_string();
                };
                let key_len = rest
                    .chars()
                    .take_while(|c| c.is_ascii_lowercase() || *c == '_')
                    .count();
                if key_len > 0 && rest[key_len..].trim_start().starts_with('=') {
                    rest.to_string()
                } else {
                    l.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    #[test]
    fn scaffold_parses_and_ships_inert_scoring() {
        // Two claims at once. Parsing at all proves every live key is in
        // KNOWN_KEYS — the unknown-key check turns template drift into a
        // test failure. The field asserts prove the scaffold ships the
        // inert configuration: examples in comments, no values smuggled.
        let cfg = Config::parse(CONFIG, root()).expect("scaffold must parse");
        assert_eq!(cfg.min_gap, 0.0, "scaffold must not arm the gate");
        assert_eq!(cfg.rrf_alpha, 0.0, "scaffold must fuse as plain RRF");
        assert_eq!(cfg.min_cos, None, "scaffold must derive the floor");
        assert_eq!(
            cfg.chunk_penalty,
            chops_search_core::score::ScoreOpts::default().chunk_penalty,
            "scaffold must ship the compiled penalty"
        );
        assert_eq!(cfg.dims, Some(128));
    }

    #[test]
    fn scaffold_examples_are_valid_when_uncommented() {
        // Every `# key = value` example is text a user will uncomment
        // verbatim, so each must be a known key with an in-range value.
        // A future rename or rail change that orphans an example fails
        // here instead of in some user's build.
        let live = uncommented(CONFIG);
        let cfg = Config::parse(&live, root()).expect("uncommented scaffold must parse");
        assert_eq!(cfg.min_gap, 0.08);
        assert_eq!(cfg.rrf_alpha, 1.0);
        assert_eq!(cfg.chunk_penalty, 0.05);
        assert_eq!(cfg.min_cos, Some(0.34));
        assert_eq!(cfg.title_weight, 2.0);
        assert_eq!(cfg.chunk_chars, 600);
    }

    #[test]
    fn creates_config_and_page() {
        let root = tmp("create");
        init(&root, true).unwrap();
        assert!(root.join("chops-search.toml").is_file());
        assert!(root.join("content/search.md").is_file());
        assert!(
            fs::read_to_string(root.join(".gitignore"))
                .unwrap()
                .contains("/static/search/")
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn scaffolded_config_is_loadable_and_matches_the_defaults() {
        // The template is commented-out config, which is exactly the kind
        // of text that rots: it described pre-multiplied tf weights for a
        // while after BM25F landed. Parsing it uncommented proves every
        // key is real, spelled right, and set to the value the binary
        // would have used anyway.
        let root = tmp("template");
        init(&root, false).unwrap();
        let path = root.join("chops-search.toml");
        let raw = fs::read_to_string(&path).unwrap();
        let uncommented: String = raw
            .lines()
            .map(|l| {
                let t = l.trim_start();
                match t.strip_prefix("# ") {
                    // Only revive lines that look like `key = value`;
                    // prose comments stay comments.
                    Some(rest) if rest.contains(" = ") => rest.to_string(),
                    _ => l.to_string(),
                }
            })
            .collect::<Vec<_>>()
            .join("\n");
        fs::write(&path, &uncommented).unwrap();

        let cfg = crate::config::Config::load(&path).unwrap();
        assert_eq!(cfg.title_weight, chops_search_core::keyword::W_TITLE);
        assert_eq!(cfg.tag_weight, chops_search_core::keyword::W_TAG);
        assert_eq!(cfg.desc_weight, chops_search_core::keyword::W_DESC);
        assert_eq!(cfg.chunk_chars, 600);
        assert_eq!(cfg.prefix_rows, 2048);
        assert_eq!(cfg.dims, Some(128));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn never_overwrites() {
        let root = tmp("idempotent");
        fs::write(root.join("chops-search.toml"), "content = \"mine\"\n").unwrap();
        init(&root, true).unwrap();
        assert_eq!(
            fs::read_to_string(root.join("chops-search.toml")).unwrap(),
            "content = \"mine\"\n",
            "an existing config must survive"
        );
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn gitignore_lines_are_not_duplicated() {
        let root = tmp("gitignore");
        init(&root, false).unwrap();
        init(&root, false).unwrap();
        let text = fs::read_to_string(root.join(".gitignore")).unwrap();
        assert_eq!(text.matches("/static/search/").count(), 1);
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn existing_gitignore_is_preserved() {
        let root = tmp("append");
        fs::write(root.join(".gitignore"), "/public/\n/target/\n").unwrap();
        init(&root, false).unwrap();
        let text = fs::read_to_string(root.join(".gitignore")).unwrap();
        assert!(text.contains("/public/"));
        assert!(text.contains("/target/"));
        assert!(text.contains("/static/search/"));
        fs::remove_dir_all(&root).ok();
    }
}
