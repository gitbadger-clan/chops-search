//! Runtime assets embedded in the binary.
//!
//! The alternative — telling users to install wasm-pack, build the wasm
//! crate, and copy three JS files into the right place — is four steps
//! and two toolchains before anything works. Embedding them means
//! `cargo install` then `chops-search build`, and the runtime is
//! guaranteed to match the artifact format the same binary just wrote.
//! That last part matters more than the convenience: a separately
//! distributed frontend can drift from the builder, and format v1 glue
//! meeting format v2 artifacts is exactly the class of bug the manifest
//! hash exists to prevent.
//!
//! The files come from `web/` and from `wasm-pack` output, refreshed by
//! `cargo xtask assets` and committed. CI regenerates and fails if the
//! result differs, so a stale asset can't ship — see xtask/src/main.rs.
//!
//! Total embedded weight is around 200 KB, almost all of it the wasm
//! blob. That's paid once in the binary, not per site.

use std::fs;
use std::path::Path;

use anyhow::{Context, Result};

/// One emitted file: destination relative to the artifacts directory,
/// plus its bytes.
pub struct Asset {
    pub name: &'static str,
    pub bytes: &'static [u8],
}

/// wasm-bindgen emits the glue and the binary as a pair; both are
/// requested with `?v=<hash>` by the worker so they can be immutable.
pub const ASSETS: &[Asset] = &[
    Asset {
        name: "pkg/chops_search_wasm_bg.wasm",
        bytes: include_bytes!("../assets/chops_search_wasm_bg.wasm"),
    },
    Asset {
        name: "pkg/chops_search_wasm.js",
        bytes: include_bytes!("../assets/chops_search_wasm.js"),
    },
    Asset {
        name: "search-worker.js",
        bytes: include_bytes!("../assets/search-worker.js"),
    },
    Asset {
        name: "chops-search.js",
        bytes: include_bytes!("../assets/chops-search.js"),
    },
    Asset {
        name: "chops-search.css",
        bytes: include_bytes!("../assets/chops-search.css"),
    },
];

/// Write the runtime into the artifacts directory.
///
/// Unconditional overwrite: these are generated files under a directory
/// the tool owns, and a user who edited them in place has already lost
/// the next time they upgrade. Anyone customizing the UI should copy
/// chops-search.js elsewhere and load that instead — the worker protocol
/// is the supported extension point.
pub fn write_runtime(out: &Path) -> Result<()> {
    for asset in ASSETS {
        let path = out.join(asset.name);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, asset.bytes).with_context(|| format!("writing {}", path.display()))?;
    }
    eprintln!(
        "wrote runtime ({} files, {} KB) to {}",
        ASSETS.len(),
        ASSETS.iter().map(|a| a.bytes.len()).sum::<usize>() / 1024,
        out.display()
    );
    Ok(())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_asset_is_non_empty() {
        // include_bytes! of a missing file is a compile error, but an
        // empty one isn't — and an empty wasm blob would fail at boot in
        // the browser rather than here.
        for a in ASSETS {
            assert!(!a.bytes.is_empty(), "{} is empty", a.name);
        }
    }

    #[test]
    fn wasm_has_the_right_magic() {
        let wasm = ASSETS
            .iter()
            .find(|a| a.name.ends_with(".wasm"))
            .expect("no wasm asset");
        assert_eq!(&wasm.bytes[..4], b"\0asm", "not a wasm module");
    }

    #[test]
    fn glue_and_worker_look_like_modules() {
        let find = |n: &str| ASSETS.iter().find(|a| a.name.ends_with(n)).unwrap().bytes;
        let worker = std::str::from_utf8(find("search-worker.js")).unwrap();
        // The worker is imported as a module and must reach the glue.
        assert!(
            worker.contains("chops_search_wasm.js"),
            "worker lost its glue import"
        );
        assert!(
            worker.contains("manifest.json"),
            "worker lost manifest resolution"
        );
    }

    #[test]
    fn runtime_writes_into_nested_paths() {
        let tmp = std::env::temp_dir().join(format!("chops-assets-{}", std::process::id()));
        write_runtime(&tmp).unwrap();
        for a in ASSETS {
            assert!(tmp.join(a.name).is_file(), "{} not written", a.name);
        }
        std::fs::remove_dir_all(&tmp).ok();
    }
}
