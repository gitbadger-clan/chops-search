//! Resolve built artifacts by reading the manifest.
//!
//! Filenames carry a content hash so everything under /search/ can be
//! served `immutable`, which means nothing can hardcode
//! "model.meta.bin" any more — `eval`, `query`, and the browser all go
//! through the manifest instead.
//!
//! One hash covers the whole set rather than one per file: the artifacts
//! are only meaningful together (chunk vectors index into the same vocab
//! the rows file provides), so per-file hashes would allow a stale meta
//! to pair with a fresh index — exactly the skew the hashing exists to
//! prevent.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

#[derive(Debug)]
pub struct Artifacts {
    pub hash: String,
    pub meta: PathBuf,
    pub prefix: PathBuf,
    pub rows: PathBuf,
    pub index: PathBuf,
}

/// Read `manifest.json` from an artifacts directory. Falls back to the
/// pre-hashing filenames when no manifest exists, so an old build
/// directory still works with `chops-search query` without a rebuild.
pub fn resolve(dir: &Path) -> Result<Artifacts> {
    let manifest_path = dir.join("manifest.json");
    if !manifest_path.exists() {
        let legacy = Artifacts {
            hash: "unhashed".to_string(),
            meta: dir.join("model.meta.bin"),
            prefix: dir.join("model.prefix.i8"),
            rows: dir.join("model.rows.i8"),
            index: dir.join("index.bin"),
        };
        if legacy.meta.exists() {
            eprintln!("note: no manifest.json; using pre-hashing filenames");
            return Ok(legacy);
        }
        anyhow::bail!(
            "no manifest.json and no legacy artifacts in {} — run `chops-search build` first",
            dir.display()
        );
    }

    let text = fs::read_to_string(&manifest_path)
        .with_context(|| format!("reading {}", manifest_path.display()))?;
    let v: serde_json::Value = serde_json::from_str(&text).context("parsing manifest.json")?;

    let version = v.get("version").and_then(|x| x.as_u64()).unwrap_or(0);
    if version != 1 {
        anyhow::bail!("manifest version {version} is not supported (expected 1); rebuild");
    }
    let hash = v
        .get("hash")
        .and_then(|x| x.as_str())
        .context("manifest has no `hash`")?
        .to_string();

    let file = |key: &str| -> Result<PathBuf> {
        let name = v
            .get("files")
            .and_then(|f| f.get(key))
            .and_then(|x| x.as_str())
            .with_context(|| format!("manifest has no files.{key}"))?;
        Ok(dir.join(name))
    };

    Ok(Artifacts {
        hash,
        meta: file("meta")?,
        prefix: file("prefix")?,
        rows: file("rows")?,
        index: file("index")?,
    })
}
