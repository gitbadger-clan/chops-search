//! Version bumping, as one command.
//!
//! A release touches two places in the root manifest: `[workspace.package]
//! version`, and the `version` field of the internal `chops-search-core`
//! dependency. Missing the second is silent locally (the path dependency
//! still resolves) and fatal in CI, where cargo resolves against
//! crates.io and finds no such version. That failure has burned several
//! version numbers, and burned numbers are permanent.
//!
//! Member manifests are walked too. A crate that spells its own version
//! as a literal instead of `version.workspace = true` drifts silently:
//! nothing resolves against it locally, and `chops-search-wasm` is never
//! published, so no registry check would catch it either. The only
//! symptom is `cargo metadata` reporting a version nobody chose, which
//! surfaces in a bug report as "which build is this?".
//!
//! Members that inherit (`version.workspace = true`, or
//! `chops-search-core.workspace = true`) are reported as inheriting and
//! left alone. That is the state to prefer: inheritance cannot drift, so
//! the right fix for a literal is usually to replace it with inheritance
//! rather than to keep bumping it here.
//!
//! `--check` asserts all of it, which makes the invariant a CI gate
//! rather than something to remember.
//!
//! Tasks:
//!   version <major|minor|patch|X.Y.Z>   bump, update Cargo.lock
//!   version                             assert consistency, change nothing

use std::fs;
use std::path::{Path, PathBuf};

use toml_edit::{DocumentMut, Item, value};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Internal dependencies whose `version` must track the workspace
/// version. Path dependencies need one to be publishable, and it has to
/// match or the published crate points at a version that does not exist.
const INTERNAL_DEPS: &[&str] = &["chops-search-core"];

/// Dependency tables a member manifest may declare an internal crate in.
const DEP_TABLES: &[&str] = &["dependencies", "dev-dependencies", "build-dependencies"];

pub fn version(root: &Path, arg: Option<&str>) -> Result<()> {
    let manifest_path = root.join("Cargo.toml");
    let text = fs::read_to_string(&manifest_path)?;
    let mut doc: DocumentMut = text.parse()?;

    let current = doc["workspace"]["package"]["version"]
        .as_str()
        .ok_or("no [workspace.package] version")?
        .to_string();

    let members = member_manifests(root, &doc)?;

    let Some(arg) = arg else {
        return check(root, &doc, &current, &members);
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

    // Members second, so a failure here leaves the root already written
    // and the command re-runnable: `version <same>` would refuse as "not
    // ahead", but the check will name exactly what is still stale.
    for m in &members {
        for line in bump_member(root, m, &next)? {
            println!("  {line}");
        }
    }

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

/// Every member manifest named by `[workspace] members`, with `dir/*`
/// globs expanded. Cargo's glob support is wider than this (`**`,
/// character classes), but a workspace that needs those is a workspace
/// this task should be taught about explicitly rather than guess at.
fn member_manifests(root: &Path, doc: &DocumentMut) -> Result<Vec<PathBuf>> {
    let Some(members) = doc
        .get("workspace")
        .and_then(|w| w.get("members"))
        .and_then(|m| m.as_array())
    else {
        return Ok(Vec::new());
    };

    let mut out = Vec::new();
    for entry in members.iter() {
        let Some(pat) = entry.as_str() else { continue };
        if let Some(dir) = pat.strip_suffix("/*") {
            let base = root.join(dir);
            let Ok(read) = fs::read_dir(&base) else {
                continue;
            };
            for e in read.flatten() {
                let manifest = e.path().join("Cargo.toml");
                if manifest.is_file() {
                    out.push(manifest);
                }
            }
        } else {
            let manifest = root.join(pat).join("Cargo.toml");
            if manifest.is_file() {
                out.push(manifest);
            }
        }
    }
    // Deterministic order: the printed report is read by humans and
    // diffed by nobody, but a shuffling order makes it look like
    // something changed when nothing did.
    out.sort();
    out.dedup();
    Ok(out)
}

/// Whether an item is `key.workspace = true` in any of its spellings.
fn inherits(item: &Item) -> bool {
    item.get("workspace")
        .and_then(|w| w.as_bool())
        .unwrap_or(false)
}

/// Rewrite literal versions in one member manifest. Returns a line per
/// change, empty when the crate inherits everything (the good case).
fn bump_member(root: &Path, path: &Path, next: &str) -> Result<Vec<String>> {
    let text = fs::read_to_string(path)?;
    let mut doc: DocumentMut = text.parse()?;
    let name = rel(root, path);
    let mut changed = Vec::new();

    if let Some(v) = doc.get("package").and_then(|p| p.get("version")) {
        if v.as_str().is_some() {
            doc["package"]["version"] = value(next);
            changed.push(format!("{name}: version = {next}"));
        } else if !inherits(v) {
            return Err(format!(
                "{name}: `version` is neither a string nor `version.workspace = true`"
            )
            .into());
        }
    }

    for table in DEP_TABLES {
        for dep in INTERNAL_DEPS {
            let Some(entry) = doc.get_mut(table).and_then(|t| t.get_mut(dep)) else {
                continue;
            };
            // A path-only dep on an unpublished crate is legitimate and
            // needs no version, so absence is not an error here. It is
            // one in [workspace.dependencies], where publishing depends
            // on it.
            if entry.get("version").and_then(|v| v.as_str()).is_some() {
                entry["version"] = value(next);
                changed.push(format!("{name}: {table}.{dep} = {next}"));
            }
        }
    }

    if !changed.is_empty() {
        fs::write(path, doc.to_string())?;
    }
    Ok(changed)
}

/// Assert the workspace version and every internal dependency agree.
fn check(root: &Path, doc: &DocumentMut, current: &str, members: &[PathBuf]) -> Result<()> {
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

    for m in members {
        bad.extend(check_member(root, m, current)?);
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

/// Mismatches in one member manifest. Inheritance is reported and never
/// a mismatch: it is the state that cannot drift.
fn check_member(root: &Path, path: &Path, current: &str) -> Result<Vec<String>> {
    let doc: DocumentMut = fs::read_to_string(path)?.parse()?;
    let name = rel(root, path);
    let mut bad = Vec::new();

    match doc.get("package").and_then(|p| p.get("version")) {
        Some(v) if inherits(v) => println!("  ok  {name} inherits version"),
        Some(v) => match v.as_str() {
            Some(s) if s == current => println!("  ok  {name} = {s}"),
            Some(s) => bad.push(format!("{name}: version = {s}, workspace = {current}")),
            None => bad.push(format!(
                "{name}: `version` is neither a string nor `version.workspace = true`"
            )),
        },
        None => {}
    }

    for table in DEP_TABLES {
        for dep in INTERNAL_DEPS {
            let Some(entry) = doc.get(table).and_then(|t| t.get(dep)) else {
                continue;
            };
            match entry.get("version").and_then(|v| v.as_str()) {
                Some(v) if v == current => println!("  ok  {name}: {table}.{dep} = {v}"),
                Some(v) => bad.push(format!(
                    "{name}: {table}.{dep} = {v}, workspace = {current}"
                )),
                // No version key: either inherited from
                // [workspace.dependencies], or a path-only dep on a crate
                // that is never published. Both are fine.
                None => {}
            }
        }
    }
    Ok(bad)
}

fn rel(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .unwrap_or(path)
        .to_string_lossy()
        .into_owned()
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

    fn tmp(name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("chops-xtask-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn member(root: &Path, dir: &str, body: &str) -> PathBuf {
        let d = root.join(dir);
        fs::create_dir_all(&d).unwrap();
        let p = d.join("Cargo.toml");
        fs::write(&p, body).unwrap();
        p
    }

    const ROOT: &str = r#"
[workspace]
members = ["crates/*", "xtask"]

[workspace.package]
version = "0.2.3"

[workspace.dependencies]
chops-search-core = { path = "crates/chops-search-core", version = "0.2.3" }
"#;

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
        assert!(check(Path::new("/nowhere"), &doc, "0.2.3", &[]).is_err());
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
        assert!(check(Path::new("/nowhere"), &doc, "0.2.3", &[]).is_ok());
    }

    #[test]
    fn globs_and_plain_members_both_resolve() {
        let root = tmp("members");
        fs::write(root.join("Cargo.toml"), ROOT).unwrap();
        member(&root, "crates/a", "[package]\nname = \"a\"\n");
        member(&root, "crates/b", "[package]\nname = \"b\"\n");
        member(&root, "xtask", "[package]\nname = \"xtask\"\n");
        // A directory under crates/ with no manifest is not a member.
        fs::create_dir_all(root.join("crates/not-a-crate")).unwrap();

        let doc: DocumentMut = ROOT.parse().unwrap();
        let found: Vec<String> = member_manifests(&root, &doc)
            .unwrap()
            .iter()
            .map(|p| rel(&root, p))
            .collect();
        assert_eq!(found.len(), 3, "{found:?}");
        assert!(found.iter().any(|f| f.contains("crates/a")));
        assert!(found.iter().any(|f| f.contains("xtask")));
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn inherited_member_versions_are_left_alone() {
        // The wasm crate's shape, and the one to prefer: nothing to bump,
        // nothing that can drift.
        let root = tmp("inherit");
        let p = member(
            &root,
            "crates/chops-search-wasm",
            "[package]\nname = \"chops-search-wasm\"\nversion.workspace = true\n\n\
             [dependencies]\nchops-search-core.workspace = true\n",
        );
        let before = fs::read_to_string(&p).unwrap();
        assert!(bump_member(&root, &p, "0.3.0").unwrap().is_empty());
        assert_eq!(
            fs::read_to_string(&p).unwrap(),
            before,
            "file was rewritten"
        );
        assert!(check_member(&root, &p, "0.2.3").unwrap().is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn literal_member_versions_are_bumped_and_checked() {
        // The drift this change exists to close: a member spelling its
        // own version, plus a path dep pinning a stale one.
        let root = tmp("literal");
        let p = member(
            &root,
            "crates/chops-search-wasm",
            "[package]\nname = \"chops-search-wasm\"\nversion = \"0.2.1\"\n\n\
             [dependencies]\nchops-search-core = { path = \"../chops-search-core\", version = \"0.2.1\" }\n",
        );
        let stale = check_member(&root, &p, "0.2.3").unwrap();
        assert_eq!(stale.len(), 2, "{stale:?}");

        let changed = bump_member(&root, &p, "0.3.0").unwrap();
        assert_eq!(changed.len(), 2, "{changed:?}");
        let after = fs::read_to_string(&p).unwrap();
        assert!(after.contains("version = \"0.3.0\""));
        assert!(!after.contains("0.2.1"));
        assert!(check_member(&root, &p, "0.3.0").unwrap().is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn path_only_internal_dep_in_a_member_is_not_a_mismatch() {
        // An unpublished crate may depend on core by path alone. That is
        // legitimate and must not be reported, or the gate cries wolf.
        let root = tmp("pathonly");
        let p = member(
            &root,
            "crates/chops-search-wasm",
            "[package]\nname = \"w\"\nversion.workspace = true\n\n\
             [dependencies]\nchops-search-core = { path = \"../chops-search-core\" }\n",
        );
        assert!(check_member(&root, &p, "0.2.3").unwrap().is_empty());
        assert!(bump_member(&root, &p, "0.3.0").unwrap().is_empty());
        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn dev_dependencies_are_checked_too() {
        let root = tmp("devdeps");
        let p = member(
            &root,
            "crates/x",
            "[package]\nname = \"x\"\nversion.workspace = true\n\n\
             [dev-dependencies]\nchops-search-core = { path = \"../c\", version = \"0.1.0\" }\n",
        );
        assert_eq!(check_member(&root, &p, "0.2.3").unwrap().len(), 1);
        fs::remove_dir_all(&root).ok();
    }
}
