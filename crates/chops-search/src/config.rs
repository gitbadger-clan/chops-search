//! `chops-search.toml` — per-site configuration.
//!
//! Once this is a tool other people install rather than a subdirectory of
//! one site, passing `--content ../content --model model --out
//! ../static/search` on every invocation stops being reasonable: the
//! paths are properties of the site, not of the command. They belong in a
//! file that lives beside the site and shows up in a diff when someone
//! changes them.
//!
//! Discovery walks up from the working directory the way cargo finds
//! Cargo.toml, so `chops-search build` works from anywhere inside the
//! site. Paths inside the file resolve relative to the FILE, not the
//! working directory — otherwise the same config would mean different
//! things depending on where you ran it from.
//!
//! Precedence is flags > file > defaults. Flags win because that's what
//! makes one-off experiments (`--dims 128`, a scratch `--out`) possible
//! without editing tracked config.
//!
//! Example, at the site root next to config.toml:
//!
//! ```toml
//! content = "content"
//! out     = "static/search"
//! model   = ".chops-search/model"
//!
//! dims        = 128     # PCA target; omit for the model's native size
//! chunk_chars = 600
//! prefix_rows = 2048
//!
//! title_weight = 2      # BM25F: a title occurrence against a body one
//! tag_weight   = 4      # tags are the author's own relevance signal
//! desc_weight  = 1      # front-matter description, at body parity
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

pub const FILE_NAME: &str = "chops-search.toml";

#[derive(Debug, Clone)]
pub struct Config {
    /// Directory the config was found in; every relative path resolves
    /// against this.
    pub root: PathBuf,
    pub content: PathBuf,
    pub out: PathBuf,
    pub model: PathBuf,
    pub dims: Option<usize>,
    pub chunk_chars: usize,
    pub prefix_rows: u32,
    /// BM25F field weights, baked into index.bin so the browser scores
    /// with what the site configured.
    ///
    /// These are NOT the pre-multiplied term-frequency multipliers they
    /// replaced. The old scheme counted a title mention as N body
    /// mentions before saturation, which let a weighted title tf run
    /// straight past k1. BM25F normalizes each field by its own length,
    /// combines the normalized tfs under these weights, and saturates
    /// once — so a weight biases without inflating, and the useful range
    /// is small. 0.0 is legal and means "ignore this field".
    ///
    /// `desc_weight` defaults to 1.0, at parity with body, because
    /// descriptions were previously counted as body terms and the field
    /// was introduced to fix dl_body pollution rather than to change
    /// ranking. Sweep it before raising it: measured on the docs corpus,
    /// description evidence is not weight-sensitive.
    pub title_weight: f32,
    pub tag_weight: f32,
    pub desc_weight: f32,
}

impl Config {
    /// Defaults matching a stock Zola layout, rooted at `root`.
    pub fn defaults_at(root: PathBuf) -> Self {
        Config {
            content: root.join("content"),
            out: root.join("static/search"),
            model: root.join(".chops-search/model"),
            root,
            dims: None,
            chunk_chars: 600,
            prefix_rows: 2048,
            title_weight: chops_search_core::keyword::W_TITLE,
            tag_weight: chops_search_core::keyword::W_TAG,
            desc_weight: chops_search_core::keyword::W_DESC,
        }
    }

    /// Find and load config, walking up from `start`. Returns defaults
    /// rooted at `start` when no file exists — a site with a stock layout
    /// needs no config at all.
    pub fn discover(start: &Path) -> Result<Self> {
        let start = start.canonicalize().unwrap_or_else(|_| start.to_path_buf());
        let mut dir = start.as_path();
        loop {
            let candidate = dir.join(FILE_NAME);
            if candidate.is_file() {
                return Self::load(&candidate);
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => return Ok(Self::defaults_at(start)),
            }
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let root = path
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let t: toml::Table = text
            .parse()
            .with_context(|| format!("{} is not valid TOML", path.display()))?;

        let mut cfg = Config::defaults_at(root.clone());
        let path_of = |key: &str| t.get(key).and_then(|v| v.as_str()).map(|s| root.join(s));
        if let Some(p) = path_of("content") {
            cfg.content = p;
        }
        if let Some(p) = path_of("out") {
            cfg.out = p;
        }
        if let Some(p) = path_of("model") {
            cfg.model = p;
        }

        // Integers are read through i64 and range-checked rather than
        // cast: a typo like prefix_rows = -1 should be an error at parse
        // time, not a panic or a wrapped value deep in the build.
        let int = |key: &str| -> Result<Option<i64>> {
            match t.get(key) {
                None => Ok(None),
                Some(v) => v
                    .as_integer()
                    .map(Some)
                    .with_context(|| format!("{key} must be an integer")),
            }
        };
        // Field weights are f32 but must still accept `title_weight = 2`:
        // TOML types that as an integer, and every config written before
        // BM25F landed (including the one `chops-search init` scaffolds)
        // spells them without a decimal point. Rejecting integers here
        // would break those sites on upgrade for no reason.
        let num = |key: &str| -> Result<Option<f64>> {
            match t.get(key) {
                None => Ok(None),
                Some(v) => v
                    .as_float()
                    .or_else(|| v.as_integer().map(|i| i as f64))
                    .map(Some)
                    .with_context(|| format!("{key} must be a number")),
            }
        };
        if let Some(v) = int("dims")? {
            if v <= 0 {
                anyhow::bail!("dims must be positive");
            }
            cfg.dims = Some(v as usize);
        }
        if let Some(v) = int("chunk_chars")? {
            if v < 100 {
                anyhow::bail!("chunk_chars below 100 defeats chunking");
            }
            cfg.chunk_chars = v as usize;
        }
        if let Some(v) = int("prefix_rows")? {
            if !(0..=u32::MAX as i64).contains(&v) {
                anyhow::bail!("prefix_rows out of range");
            }
            cfg.prefix_rows = v as u32;
        }
        // 0.0 is meaningful ("ignore titles"), so the floor is zero, not
        // one. The ceiling is a sanity rail: past ~100 the combined tf
        // saturates on the field alone and the other fields stop
        // mattering, which is a config mistake rather than a strategy.
        let weight = |key: &str, v: f64| -> Result<f32> {
            if !v.is_finite() || !(0.0..=100.0).contains(&v) {
                anyhow::bail!("{key} must be between 0 and 100");
            }
            Ok(v as f32)
        };
        if let Some(v) = num("title_weight")? {
            cfg.title_weight = weight("title_weight", v)?;
        }
        if let Some(v) = num("tag_weight")? {
            cfg.tag_weight = weight("tag_weight", v)?;
        }
        if let Some(v) = num("desc_weight")? {
            cfg.desc_weight = weight("desc_weight", v)?;
        }

        // Unknown keys are an error, not a shrug: a misspelled `chunk_size`
        // that silently does nothing is worse than a failed build.
        const KNOWN: &[&str] = &[
            "content",
            "out",
            "model",
            "dims",
            "chunk_chars",
            "prefix_rows",
            "title_weight",
            "tag_weight",
            "desc_weight",
        ];
        for key in t.keys() {
            if !KNOWN.contains(&key.as_str()) {
                anyhow::bail!(
                    "unknown key `{key}` in {} (known: {})",
                    path.display(),
                    KNOWN.join(", ")
                );
            }
        }
        Ok(cfg)
    }

    /// Apply CLI overrides. Flags win so one-off experiments don't need a
    /// config edit.
    ///
    /// The field weights are deliberately absent: they're a query-time
    /// knob, swept with `eval --w-title` / `--w-tag` / `--w-desc` against
    /// an index that's already built. Only the value baked into
    /// index.bin — what the browser will use — comes from here.
    pub fn with_overrides(
        mut self,
        content: Option<PathBuf>,
        out: Option<PathBuf>,
        model: Option<PathBuf>,
        dims: Option<usize>,
        chunk_chars: Option<usize>,
        prefix_rows: Option<u32>,
    ) -> Self {
        if let Some(p) = content {
            self.content = p;
        }
        if let Some(p) = out {
            self.out = p;
        }
        if let Some(p) = model {
            self.model = p;
        }
        if let Some(v) = dims {
            self.dims = Some(v);
        }
        if let Some(v) = chunk_chars {
            self.chunk_chars = v;
        }
        if let Some(v) = prefix_rows {
            self.prefix_rows = v;
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, body: &str) -> PathBuf {
        let p = dir.join(FILE_NAME);
        std::fs::write(&p, body).unwrap();
        p
    }

    #[test]
    fn paths_resolve_against_the_config_file() {
        let tmp = std::env::temp_dir().join(format!("chops-cfg-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = write(&tmp, "content = \"src/pages\"\nout = \"public/s\"\n");
        let cfg = Config::load(&p).unwrap();
        assert_eq!(cfg.content, tmp.join("src/pages"));
        assert_eq!(cfg.out, tmp.join("public/s"));
        // Untouched keys keep their defaults, also rooted at the file.
        assert_eq!(cfg.model, tmp.join(".chops-search/model"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn unknown_key_is_an_error() {
        let tmp = std::env::temp_dir().join(format!("chops-cfg-unk-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let p = write(&tmp, "chunk_size = 600\n");
        let err = Config::load(&p).unwrap_err().to_string();
        assert!(err.contains("chunk_size"), "{err}");
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn bad_values_are_rejected() {
        let tmp = std::env::temp_dir().join(format!("chops-cfg-bad-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        assert!(Config::load(&write(&tmp, "dims = 0\n")).is_err());
        assert!(Config::load(&write(&tmp, "dims = \"128\"\n")).is_err());
        assert!(Config::load(&write(&tmp, "chunk_chars = 10\n")).is_err());
        assert!(Config::load(&write(&tmp, "title_weight = -1\n")).is_err());
        assert!(Config::load(&write(&tmp, "tag_weight = 1000\n")).is_err());
        assert!(Config::load(&write(&tmp, "desc_weight = -0.5\n")).is_err());
        assert!(Config::load(&write(&tmp, "title_weight = \"2\"\n")).is_err());
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn weights_accept_integers_and_floats() {
        // Every config written before BM25F spells these without a
        // decimal point, and TOML types those as integers. Both forms
        // must load, or the change breaks sites on upgrade.
        let tmp = std::env::temp_dir().join(format!("chops-cfg-w-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = Config::load(&write(
            &tmp,
            "title_weight = 2\ntag_weight = 4\ndesc_weight = 1\n",
        ))
        .unwrap();
        assert_eq!(cfg.title_weight, 2.0);
        assert_eq!(cfg.tag_weight, 4.0);
        assert_eq!(cfg.desc_weight, 1.0);
        let cfg = Config::load(&write(
            &tmp,
            "title_weight = 1.5\ntag_weight = 0.0\ndesc_weight = 0.25\n",
        ))
        .unwrap();
        assert_eq!(cfg.title_weight, 1.5);
        assert_eq!(cfg.tag_weight, 0.0, "zero means ignore the field");
        assert_eq!(cfg.desc_weight, 0.25);
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn flags_beat_file() {
        let tmp = std::env::temp_dir().join(format!("chops-cfg-ovr-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let cfg = Config::load(&write(&tmp, "dims = 256\n"))
            .unwrap()
            .with_overrides(
                None,
                Some(PathBuf::from("/tmp/scratch")),
                None,
                Some(128),
                None,
                None,
            );
        assert_eq!(cfg.dims, Some(128));
        assert_eq!(cfg.out, PathBuf::from("/tmp/scratch"));
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn missing_config_yields_stock_zola_defaults() {
        let cfg = Config::defaults_at(PathBuf::from("/site"));
        assert_eq!(cfg.content, PathBuf::from("/site/content"));
        assert_eq!(cfg.out, PathBuf::from("/site/static/search"));
        assert_eq!(cfg.chunk_chars, 600);
        // Defaults track core's constants rather than restating them.
        assert_eq!(cfg.title_weight, chops_search_core::keyword::W_TITLE);
        assert_eq!(cfg.tag_weight, chops_search_core::keyword::W_TAG);
        assert_eq!(cfg.desc_weight, chops_search_core::keyword::W_DESC);
    }

    #[test]
    fn each_weight_key_sets_only_its_own_field() {
        // Three near-identical parse blocks reading three near-identical
        // key names: a copy-paste that assigns tag_weight twice compiles
        // and silently ignores one key.
        let tmp = std::env::temp_dir().join(format!("chops-cfg-each-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let d = Config::defaults_at(tmp.clone());
        let only_title = Config::load(&write(&tmp, "title_weight = 9\n")).unwrap();
        assert_eq!(
            (
                only_title.title_weight,
                only_title.tag_weight,
                only_title.desc_weight
            ),
            (9.0, d.tag_weight, d.desc_weight)
        );
        let only_tag = Config::load(&write(&tmp, "tag_weight = 9\n")).unwrap();
        assert_eq!(
            (
                only_tag.title_weight,
                only_tag.tag_weight,
                only_tag.desc_weight
            ),
            (d.title_weight, 9.0, d.desc_weight)
        );
        let only_desc = Config::load(&write(&tmp, "desc_weight = 9\n")).unwrap();
        assert_eq!(
            (
                only_desc.title_weight,
                only_desc.tag_weight,
                only_desc.desc_weight
            ),
            (d.title_weight, d.tag_weight, 9.0)
        );
        std::fs::remove_dir_all(&tmp).ok();
    }
}
