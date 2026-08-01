//! TOML front matter, parsed for real (fix #1).
//!
//! The hand-rolled `title = "..."` scan in chops-search-core::chunk was fine for
//! title extraction and wrong for everything else: it couldn't see
//! `draft`, `in_search_index`, `slug`, `path`, or tags, so drafts got
//! indexed, opted-out pages got indexed, override'd URLs 404'd, and the
//! user's hand-curated tags — the strongest relevance signal on the page —
//! were thrown away with the rest of the front matter.
//!
//! This module is CLI-only. The wasm engine never sees front matter; it
//! sees artifacts. Parsing TOML properly therefore costs the browser
//! nothing.

use anyhow::{Context, Result};
use unicode_normalization::char::is_combining_mark;
use unicode_normalization::UnicodeNormalization;

#[derive(Debug)]
pub struct FrontMatter {
    pub title: Option<String>,
    pub slug: Option<String>,
    pub path: Option<String>,
    pub draft: bool,
    /// Zola's own opt-out convention; defaults to true like Zola's.
    pub in_search_index: bool,
    /// From `[taxonomies] tags = [...]`.
    pub tags: Vec<String>,
}

impl Default for FrontMatter {
    fn default() -> Self {
        FrontMatter {
            title: None,
            slug: None,
            path: None,
            draft: false,
            in_search_index: true,
            tags: Vec::new(),
        }
    }
}

/// Split `+++ ... +++` front matter off `src`, returning the parsed front
/// matter and the body that follows. Input without front matter yields
/// defaults and the whole input as body. Malformed TOML is an error —
/// Zola itself would refuse to build such a page, so silently indexing it
/// with defaults would diverge from the site.
pub fn split(src: &str) -> Result<(FrontMatter, &str)> {
    let Some(rest) = src.strip_prefix("+++") else {
        return Ok((FrontMatter::default(), src));
    };
    let Some(end) = rest.find("\n+++") else {
        return Ok((FrontMatter::default(), src));
    };
    let fm_src = &rest[..end];
    let body = rest[end + 4..].trim_start_matches('\n');

    let value: toml::Table = fm_src.parse().context("front matter is not valid TOML")?;
    let get_str = |k: &str| value.get(k).and_then(|v| v.as_str()).map(str::to_owned);
    let get_bool = |k: &str| value.get(k).and_then(|v| v.as_bool());

    let mut fm = FrontMatter {
        title: get_str("title"),
        slug: get_str("slug"),
        path: get_str("path"),
        draft: get_bool("draft").unwrap_or(false),
        in_search_index: get_bool("in_search_index").unwrap_or(true),
        ..FrontMatter::default()
    };
    if let Some(tags) = value
        .get("taxonomies")
        .and_then(|t| t.get("tags"))
        .and_then(|v| v.as_array())
    {
        fm.tags = tags
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect();
    }
    Ok((fm, body))
}

/// Zola-style path slugification (`slugify.paths = "on"`, the default):
/// NFD-decompose, drop combining marks, lowercase, and collapse every
/// non-alphanumeric run into a single `-`. "My Post" → "my-post",
/// "Café_Crème" → "cafe-creme".
///
/// Known divergence: Zola's `slug` crate transliterates non-Latin scripts
/// to ASCII; this keeps Unicode alphanumerics as-is. Latin-script content
/// (this site) is byte-identical; revisit before productizing for
/// non-Latin filenames.
pub fn slugify(segment: &str) -> String {
    let mut out = String::with_capacity(segment.len());
    let mut pending_dash = false;
    for c in segment.nfd() {
        if is_combining_mark(c) {
            continue;
        }
        for lc in c.to_lowercase() {
            if lc.is_alphanumeric() {
                if pending_dash && !out.is_empty() {
                    out.push('-');
                }
                pending_dash = false;
                out.push(lc);
            } else {
                pending_dash = true;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_front_matter_parses() {
        let src = "+++\ntitle = \"Baseline: Where Do the Tasks Go?\"\ndate = 2026-07-01\ndraft = false\n\n[taxonomies]\ntags = [\"celery\", \"redis\", \"chaos-engineering\"]\n+++\nBody text here.\n";
        let (fm, body) = split(src).unwrap();
        assert_eq!(
            fm.title.as_deref(),
            Some("Baseline: Where Do the Tasks Go?")
        );
        assert_eq!(fm.tags, vec!["celery", "redis", "chaos-engineering"]);
        assert!(!fm.draft);
        assert!(fm.in_search_index);
        assert_eq!(body.trim(), "Body text here.");
    }

    #[test]
    fn opt_outs_respected() {
        let (fm, _) = split("+++\ndraft = true\n+++\nx\n").unwrap();
        assert!(fm.draft);
        let (fm, _) = split("+++\nin_search_index = false\n+++\nx\n").unwrap();
        assert!(!fm.in_search_index);
    }

    #[test]
    fn slug_and_path_overrides() {
        let (fm, _) =
            split("+++\nslug = \"custom-slug\"\npath = \"/elsewhere/entirely\"\n+++\nx\n").unwrap();
        assert_eq!(fm.slug.as_deref(), Some("custom-slug"));
        assert_eq!(fm.path.as_deref(), Some("/elsewhere/entirely"));
    }

    #[test]
    fn no_front_matter_is_all_defaults() {
        let (fm, body) = split("Just a body.\n").unwrap();
        assert!(fm.title.is_none());
        assert!(fm.in_search_index);
        assert_eq!(body, "Just a body.\n");
    }

    #[test]
    fn malformed_toml_is_an_error() {
        assert!(split("+++\ntitle = unclosed\n+++\nx\n").is_err());
    }

    #[test]
    fn slugify_matches_zola_defaults() {
        assert_eq!(slugify("My Post"), "my-post");
        assert_eq!(slugify("Café_Crème"), "cafe-creme");
        assert_eq!(slugify("already-fine"), "already-fine");
        assert_eq!(slugify("  Spaces  Around  "), "spaces-around");
        assert_eq!(slugify("01-baseline"), "01-baseline");
    }
}
