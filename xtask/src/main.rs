//! Repo automation. `cargo xtask <task>`.
//!
//! The one job that matters: regenerate
//! `crates/chops-search/assets/`, the files the CLI embeds with
//! `include_bytes!` and writes into a site's `out` directory.
//!
//! Committed build output is normally a smell, so it earns its place by
//! being mechanically verifiable — `cargo xtask assets --check` fails if
//! the committed bytes differ from a fresh build, and CI runs it. The
//! alternative (making users install wasm-pack and copy JS by hand) is
//! four steps and two toolchains before anything works, which is the
//! difference between a tool people try and one they don't.
//!
//! It also closes the drift hazard for good: the binary that writes the
//! artifacts carries the runtime that reads them, so a format change
//! can't ship a mismatched pair.
//!
//! Tasks:
//!   assets [--check]   build wasm + copy web/ → crates/chops-search/assets/
//!   dist               assets, then a release CLI build
//!
//! Set up with `.cargo/config.toml`:
//!
//! ```toml
//! [alias]
//! xtask = "run --package xtask --"
//! ```

use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Sources copied verbatim from web/. The wasm pair is generated and
/// handled separately.
const WEB_FILES: &[&str] = &["search-worker.js", "chops-search.js", "chops-search.css"];

/// wasm-pack output we keep. It also emits .d.ts, package.json, and a
/// README; none of those help a static site.
const WASM_FILES: &[&str] = &["chops_search_wasm_bg.wasm", "chops_search_wasm.js"];

fn main() {
    let mut args = std::env::args().skip(1);
    let task = args.next().unwrap_or_default();
    // Collect BEFORE inspecting: `any` consumes the iterator, which
    // silently emptied `demo`'s arguments.
    let rest: Vec<String> = args.collect();

    let result = match task.as_str() {
        "assets" => assets(rest.iter().any(|a| a == "--check")),
        "dist" => assets(false).and_then(|_| dist()),
        "demo" => demo(&rest),
        _ => {
            eprintln!("usage: cargo xtask <assets [--check] | dist | demo <args…>>");
            std::process::exit(2);
        }
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn root() -> PathBuf {
    // xtask/ sits beside crates/, so the workspace root is one level up
    // from this crate's manifest. Deriving it from CARGO_MANIFEST_DIR
    // rather than the cwd means `cargo xtask` works from anywhere.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has no parent directory")
        .to_path_buf()
}

fn assets(check: bool) -> Result<()> {
    let root = root();
    let dest = root.join("crates/chops-search/assets");
    let staging = root.join("target/xtask-wasm");

    // Stamp comparison first: --check must not need wasm-pack, and
    // rebuilding to compare bytes was never reliable across machines
    // (absolute paths in panic metadata, differing wasm-opt versions).
    if check {
        let want = input_hash(&root)?;
        let got = fs::read_to_string(dest.join(".stamp")).unwrap_or_default();
        if got.trim() != want {
            return Err(format!(
                "assets are stale: sources hash {want}, stamp says {}\n\
                 run `cargo xtask assets` and commit",
                if got.trim().is_empty() {
                    "(missing)"
                } else {
                    got.trim()
                }
            )
            .into());
        }
        println!("assets are current (stamp {want})");
        return Ok(());
    }

    // Build into a staging directory rather than straight into assets/,
    // so a failed or partial wasm-pack run can't leave the committed
    // tree half-updated — and so --check has something to diff against.
    println!("building wasm → {}", staging.display());
    let status = Command::new("wasm-pack")
        .current_dir(&root)
        .args([
            "build",
            "crates/chops-search-wasm",
            "--target",
            "web",
            "--release",
            "--out-dir",
        ])
        .arg(&staging)
        .env("RUSTFLAGS", "-Zunstable-options -Cpanic=immediate-abort")
        .args(["--", "-Z", "build-std=std,panic_abort"])
        .status()
        .map_err(|e| format!("wasm-pack not found ({e}) — cargo install wasm-pack --locked"))?;
    if !status.success() {
        return Err("wasm-pack build failed".into());
    }

    let mut planned: Vec<(PathBuf, PathBuf)> = Vec::new();
    for name in WASM_FILES {
        planned.push((staging.join(name), dest.join(name)));
    }
    for name in WEB_FILES {
        planned.push((root.join("web").join(name), dest.join(name)));
    }

    println!("building wasm → {}", staging.display());

    fs::create_dir_all(&dest)?;
    let mut total = 0usize;
    for (src, dst) in &planned {
        let bytes = fs::read(src).map_err(|e| format!("{}: {e}", src.display()))?;
        // Skip the write when bytes already match, so mtimes stay put and
        // cargo doesn't rebuild the CLI on a no-op sync.
        let unchanged = fs::read(dst).map(|c| c == bytes).unwrap_or(false);
        if !unchanged {
            fs::write(dst, &bytes)?;
        }
        total += bytes.len();
        println!(
            "  {:<32} {:>6} KB{}",
            rel(&root, dst),
            bytes.len() / 1024,
            if unchanged { "  (unchanged)" } else { "" }
        );
    }
    println!(
        "{} files, {} KB embedded into the CLI binary",
        planned.len(),
        total / 1024
    );

    // The wasm blob dominates, and it only grows by accident — a
    // dependency pulled into chops-search-core reaches the browser. Warn
    // rather than fail: the right response is usually to look at what
    // changed, not to block the build.
    let wasm = fs::metadata(dest.join("chops_search_wasm_bg.wasm"))?.len();
    fs::write(dest.join(".stamp"), format!("{}\n", input_hash(&root)?))?;
    if wasm > 300 * 1024 {
        eprintln!(
            "warning: wasm blob is {} KB (>300 KB) — check what entered \
             chops-search-core's dependency tree",
            wasm / 1024
        );
    }
    Ok(())
}

fn dist() -> Result<()> {
    let status = Command::new(env!("CARGO"))
        .current_dir(root())
        .args(["build", "--release", "-p", "chops-search"])
        .status()?;
    if !status.success() {
        return Err("release build failed".into());
    }
    println!("built target/release/chops-search");
    Ok(())
}

fn rel(root: &Path, p: &Path) -> String {
    p.strip_prefix(root).unwrap_or(p).display().to_string()
}

/// Run the CLI against examples/demo-site from anywhere in the repo.
/// `cargo xtask demo eval`, `cargo xtask demo build`, `cargo xtask demo
/// docs -- ...` — the site path is the only thing this adds.
fn demo(rest: &[String]) -> Result<()> {
    let root = root();
    let status = Command::new(env!("CARGO"))
        .current_dir(&root)
        .args(["run", "--release", "-p", "chops-search", "--", "--site"])
        .arg(root.join("examples/demo-site"))
        .args(rest)
        .status()?;
    if !status.success() {
        return Err("chops-search exited non-zero".into());
    }
    Ok(())
}

/// Recursively read every file under `dir` into `out`, keyed by path
/// relative to `root`. Skips nothing: a new file in web/ or src/ changes
/// the hash, which is the point.
fn collect(
    dir: &Path,
    root: &Path,
    out: &mut std::collections::BTreeMap<String, Vec<u8>>,
) -> Result<()> {
    if !dir.is_dir() {
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<std::io::Result<Vec<_>>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect(&path, root, out)?;
        } else {
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/"); // stable across platforms
            out.insert(rel, fs::read(&path)?);
        }
    }
    Ok(())
}
/// Files whose contents determine the embedded runtime. A change to any
/// of them means `cargo xtask assets` must be re-run.
fn input_hash(root: &Path) -> Result<String> {
    use std::collections::BTreeMap;

    let mut files: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for dir in [
        "crates/chops-search-core/src",
        "crates/chops-search-wasm/src",
        "web",
    ] {
        collect(&root.join(dir), root, &mut files)?;
    }
    for f in [
        "Cargo.lock",
        "crates/chops-search-wasm/Cargo.toml",
        "rust-toolchain.toml",
    ] {
        let p = root.join(f);
        if p.is_file() {
            files.insert(f.to_string(), fs::read(&p)?);
        }
    }

    // BTreeMap gives a stable order; each entry is length-prefixed so two
    // different splits can't hash alike.
    let mut h = Sha256::new();
    for (name, bytes) in &files {
        h.update((name.len() as u64).to_le_bytes());
        h.update(name.as_bytes());
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
    }
    Ok(h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}
