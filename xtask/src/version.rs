//! Version bumping, as one command.
//!
//! A release touches two places in the root manifest: `[workspace.package]
//! version`, and the `version` field of the internal `chops-search-core`
//! dependency. Missing the second is silent locally (the path dependency
//! still resolves) and fatal in CI, where cargo resolves against
//! crates.io and finds no such version. That failure has burned several
//! version numbers, and burned numbers are permanent.
//!
//! `--check` asserts the two agree, which makes the invariant a CI gate
//! rather than something to remember.
//!
//! Tasks:
//!   version <major|minor|patch|X.Y.Z>   bump, update Cargo.lock
//!   version                             assert consistency, change nothing

use std::fs;
use std::path::Path;

use toml_edit::{DocumentMut, value};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Internal dependencies whose `version` must track the workspace
/// version. Path dependencies need one to be publishable, and it has to
/// match or the published crate points at a version that does not exist.
const INTERNAL_DEPS: &[&str] = &["chops-search-core"];

pub fn version(root: &Path, arg: Option<&str>) -> Result<()> {
    let manifest_path = root.join("Cargo.toml");
    let text = fs::read_to_string(&manifest_path)?;
    let mut doc: DocumentMut = text.parse()?;

    let current = doc["workspace"]["package"]["version"]
        .as_str()
        .ok_or("no [workspace.package] version")?
        .to_string();

    let Some(arg) = arg else {
        return check(&doc, &current);
    };

    let next = resolve(&current, arg)?;
    if next == current {
        return Err(format!("already at {current}").into());
    }
    println!("{current} → {next}");

    doc["workspace"]["package"]["version"] = value(&next);
    for dep in INTERNAL_DEPS {
        // get_mut rather than indexing: toml_edit's Index impl panics on
        // a missing key, so a manifest without the table aborts instead
        // of reaching the check below.
        let deps = doc
            .get_mut("workspace")
            .and_then(|w| w.get_mut("dependencies"))
            .ok_or("no [workspace.dependencies] table")?;
        let entry = deps.get_mut(dep).ok_or_else(|| {
            format!(
                "`{dep}` is not in [workspace.dependencies]. Internal deps must \
                 live there so one bump updates every consumer."
            )
        })?;
        if entry.get("version").is_none() {
            return Err(format!(
                "`{dep}` has no `version` field. A path-only dependency cannot \
                 be published: cargo needs a version to record in the registry."
            )
            .into());
        }
        entry["version"] = value(&next);
        println!("  {dep} = {next}");
    }

    fs::write(&manifest_path, doc.to_string())?;

    // A missing changelog section is a warning rather than an error: the
    // release workflow degrades to generated notes, which is worse but
    // not broken.
    let changelog = root.join("CHANGELOG.md");
    if changelog.is_file() {
        let body = fs::read_to_string(&changelog)?;
        if !body.contains(&format!("## [{next}]")) && !body.contains(&format!("## {next}")) {
            println!("\nwarning: CHANGELOG.md has no section for {next}");
            println!("         the GitHub release will fall back to commit subjects");
        }
    }

    println!(
        "\nnext:\n  \
         git commit -am \"release {next}\"\n  \
         git tag v{next}\n  \
         git push origin main && git push origin --tags"
    );
    Ok(())
}

/// Assert the workspace version and every internal dependency agree.
fn check(doc: &DocumentMut, current: &str) -> Result<()> {
    // Navigate with `get` rather than indexing: toml_edit's Index impl
    // panics on a missing key, so `doc["workspace"]["dependencies"]`
    // aborts on a manifest that simply has no such table.
    let deps = doc
        .get("workspace")
        .and_then(|w| w.get("dependencies"))
        .ok_or("no [workspace.dependencies] table")?;

    let mut bad = Vec::new();
    for dep in INTERNAL_DEPS {
        match deps
            .get(dep)
            .and_then(|d| d.get("version"))
            .and_then(|v| v.as_str())
        {
            Some(v) if v == current => println!("  ok  {dep} = {v}"),
            Some(v) => bad.push(format!("{dep} = {v}, workspace = {current}")),
            None => bad.push(format!(
                "{dep} is missing from [workspace.dependencies], or has no \
                 version field"
            )),
        }
    }
    if !bad.is_empty() {
        for b in &bad {
            eprintln!("  bad {b}");
        }
        return Err("internal dependency versions disagree with the workspace \
                    version; run `cargo xtask version <version>`"
            .into());
    }
    println!("versions consistent at {current}");
    Ok(())
}

/// `major` / `minor` / `patch` relative to `current`, or an explicit
/// version. Explicit versions must move forward: crates.io will not
/// accept a number that already exists, and finding that out after the
/// tag is pushed means burning another one.
fn resolve(current: &str, arg: &str) -> Result<String> {
    let (maj, min, pat) = parse(current)?;
    let next = match arg {
        "major" => format!("{}.0.0", maj + 1),
        "minor" => format!("{maj}.{}.0", min + 1),
        "patch" => format!("{maj}.{min}.{}", pat + 1),
        explicit => {
            let (a, b, c) = parse(explicit)?;
            if (a, b, c) <= (maj, min, pat) {
                return Err(format!("{explicit} is not ahead of {current}").into());
            }
            explicit.to_string()
        }
    };
    Ok(next)
}

fn parse(v: &str) -> Result<(u64, u64, u64)> {
    // Deliberately strict: no pre-release or build metadata. Publishing a
    // 0.3.0-rc1 would need thought about yanking and version ordering
    // that this task should not silently make for you.
    let parts: Vec<&str> = v.split('.').collect();
    if parts.len() != 3 {
        return Err(format!("`{v}` is not MAJOR.MINOR.PATCH").into());
    }
    let mut out = [0u64; 3];
    for (i, p) in parts.iter().enumerate() {
        out[i] = p
            .parse()
            .map_err(|_| format!("`{v}`: `{p}` is not a number"))?;
    }
    Ok((out[0], out[1], out[2]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bumps_each_component() {
        assert_eq!(resolve("0.2.3", "patch").unwrap(), "0.2.4");
        assert_eq!(resolve("0.2.3", "minor").unwrap(), "0.3.0");
        assert_eq!(resolve("0.2.3", "major").unwrap(), "1.0.0");
    }

    #[test]
    fn explicit_must_move_forward() {
        assert_eq!(resolve("0.2.3", "0.4.0").unwrap(), "0.4.0");
        assert!(resolve("0.2.3", "0.2.3").is_err());
        assert!(resolve("0.2.3", "0.2.2").is_err());
    }

    #[test]
    fn rejects_malformed_versions() {
        assert!(parse("0.2").is_err());
        assert!(parse("0.2.3-rc1").is_err());
        assert!(parse("v0.2.3").is_err());
    }

    #[test]
    fn check_catches_the_mismatch_that_burns_versions() {
        let doc: DocumentMut = r#"
[workspace.package]
version = "0.2.3"

[workspace.dependencies]
chops-search-core = { path = "crates/chops-search-core", version = "0.2.1" }
"#
        .parse()
        .unwrap();
        assert!(check(&doc, "0.2.3").is_err());
    }

    #[test]
    fn check_passes_when_aligned() {
        let doc: DocumentMut = r#"
[workspace.package]
version = "0.2.3"

[workspace.dependencies]
chops-search-core = { path = "crates/chops-search-core", version = "0.2.3" }
"#
        .parse()
        .unwrap();
        assert!(check(&doc, "0.2.3").is_ok());
    }
}
