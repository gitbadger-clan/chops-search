//! chops-search.toml discovery and parsing.
//!
//! Discovered by walking up from the working directory the way cargo
//! finds Cargo.toml, so the CLI works from anywhere inside the site.
//! Every key is optional: an empty file (or no file) is a valid
//! configuration and means the compiled defaults. Paths are resolved
//! relative to the config file's directory, never the working directory,
//! so `chops-search build` behaves identically from the repo root and
//! from three directories down.
//!
//! Unknown keys are an error, not a shrug. A misspelled `chunk_size`
//! that silently does nothing is worse than a failed build — and after
//! v5 the stakes are higher: a typo'd `min_gp = 0.08` would silently
//! ship the gate disarmed, which is the exact honesty gap the scoring
//! keys exist to close.
//!
//! Hand-navigated TOML rather than serde-derived, same reasoning as the
//! fixture loader: the schema is a dozen keys, the error messages can
//! name the exact key and expectation, and serde stays out of the
//! dependency tree.
//!
//! SCORING CALIBRATION (`min_gap`, `rrf_alpha`, `min_cos`). These are
//! per-corpus calibrated values, and they follow the field weights'
//! provenance rule: a value calibrated against a corpus travels with the
//! corpus. `build` writes them into index.bin, the engine reads them at
//! construction, and the browser, a bare `eval`, and CI all score the
//! same configuration from the same bytes. The eval/query flags still
//! override per run for sweeping — the config states what ships, the
//! flags state deviations from it.
//!
//! `min_cos` is an OVERRIDE, not a default: absent means the engine
//! derives the floor from dimensionality (min_cos_for), and an explicit
//! 0.0 is the different, meaningful claim "floor off". The distinction
//! survives the wire — see format.rs. The other two write
//! unconditionally: their compiled defaults (0.0, both inert) are
//! legitimate shipped values, so absent and zero coinciding is harmless.

use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chops_search_core::keyword::FieldWeights;
use chops_search_core::keyword::{W_DESC, W_TAG, W_TITLE};

/// Compiled path/shape defaults. Scoring defaults come from
/// chops-search-core so this file cannot drift from the engine.
const DEFAULT_CONTENT: &str = "content";
const DEFAULT_OUT: &str = "static/search";
const DEFAULT_MODEL: &str = ".chops-search/model";
const DEFAULT_PREFIX_ROWS: u32 = 2048;
const DEFAULT_CHUNK_CHARS: usize = 600;

/// Every key `parse` understands, checked before anything else so a
/// typo'd key fails even when the value happens to be well-formed.
const KNOWN_KEYS: &[&str] = &[
    "content",
    "out",
    "model",
    "dims",
    "chunk_chars",
    "prefix_rows",
    "title_weight",
    "tag_weight",
    "desc_weight",
    "min_gap",
    "rrf_alpha",
    "min_cos",
    "chunk_penalty",
];

#[derive(Debug, Clone)]
pub struct Config {
    /// Directory containing chops-search.toml (or the discovery start
    /// when no file exists). Relative paths resolve against this, and
    /// eval finds fixtures/ under it.
    pub root: PathBuf,
    pub content: PathBuf,
    pub out: PathBuf,
    pub model: PathBuf,
    /// PCA target dimensionality. None means the model's native size.
    pub dims: Option<usize>,
    pub chunk_chars: usize,
    pub prefix_rows: u32,
    /// BM25F field weights, baked into index.bin at build time.
    pub title_weight: f32,
    pub tag_weight: f32,
    pub desc_weight: f32,
    /// Corroboration gate threshold, baked into index.bin. 0.0 (the
    /// default) ships the gate disarmed. Calibrate against the corpus
    /// with `chops-search eval --min-gap` before setting it.
    pub min_gap: f32,
    /// Confidence-weighted fusion coefficient, baked into index.bin.
    /// 0.0 (the default) is plain RRF. Same calibration discipline.
    pub rrf_alpha: f32,
    /// Relevance-floor override, baked into index.bin. None derives the
    /// floor from dimensionality at engine construction, which is the
    /// right answer for almost every corpus — an explicit value pins
    /// the floor across dims changes, and an explicit 0.0 disables it.
    pub min_cos: Option<f32>,

    /// Per-chunk expected-max correction, baked into index.bin. A doc's
    /// best-of-n chunk cosine is biased upward by having more lottery
    /// tickets; this subtracts coeff × sqrt(2 ln n) in cosine units.
    /// Default is the compiled constant. Calibrate before changing:
    /// `chops-search calibrate` or `eval --chunk-penalty`.
    pub chunk_penalty: f32,
}

impl Config {
    fn defaults(root: PathBuf) -> Self {
        Config {
            content: root.join(DEFAULT_CONTENT),
            out: root.join(DEFAULT_OUT),
            model: root.join(DEFAULT_MODEL),
            // None = the model's native size. Sites that reduce (both
            // current deploys run dims = 128) say so in their config,
            // where the choice is diffable — a compiled 128 here would
            // make "native" unspellable and turn a per-site calibration
            // into a silent product default.
            dims: None,
            chunk_chars: DEFAULT_CHUNK_CHARS,
            prefix_rows: DEFAULT_PREFIX_ROWS,
            title_weight: W_TITLE,
            tag_weight: W_TAG,
            desc_weight: W_DESC,
            min_gap: 0.0,
            rrf_alpha: 0.0,
            min_cos: None,
            chunk_penalty: chops_search_core::score::ScoreOpts::default().chunk_penalty,
            root,
        }
    }

    /// Walk up from `start` looking for chops-search.toml; parse it when
    /// found, defaults rooted at `start` when not. A present-but-broken
    /// file is an error, never a silent fall-through to defaults — a
    /// typo'd key must not quietly rebuild with different weights.
    pub fn discover(start: &Path) -> Result<Self> {
        let start = start
            .canonicalize()
            .with_context(|| format!("resolving {}", start.display()))?;
        let mut dir: &Path = &start;
        loop {
            let candidate = dir.join("chops-search.toml");
            if candidate.is_file() {
                let text = fs::read_to_string(&candidate)
                    .with_context(|| format!("reading {}", candidate.display()))?;
                return Self::parse(&text, dir.to_path_buf())
                    .with_context(|| format!("in {}", candidate.display()));
            }
            match dir.parent() {
                Some(p) => dir = p,
                None => return Ok(Self::defaults(start)),
            }
        }
    }

    /// Parse config text against defaults rooted at `root`. Public for
    /// tests; discover() is the entry point.
    pub fn parse(text: &str, root: PathBuf) -> Result<Self> {
        let doc: toml::Table = text
            .parse()
            .context("chops-search.toml is not valid TOML")?;

        // Reject unknown keys before reading any known one, so the error
        // for a half-right file names the typo rather than a downstream
        // symptom.
        for key in doc.keys() {
            if !KNOWN_KEYS.contains(&key.as_str()) {
                bail!("unknown key `{key}` (known: {})", KNOWN_KEYS.join(", "));
            }
        }

        let mut cfg = Self::defaults(root);

        if let Some(v) = doc.get("content") {
            cfg.content = cfg.root.join(as_str(v, "content")?);
        }
        if let Some(v) = doc.get("out") {
            cfg.out = cfg.root.join(as_str(v, "out")?);
        }
        if let Some(v) = doc.get("model") {
            cfg.model = cfg.root.join(as_str(v, "model")?);
        }
        if let Some(v) = doc.get("dims") {
            let d = as_usize(v, "dims")?;
            if d == 0 {
                bail!("dims must be positive");
            }
            cfg.dims = Some(d);
        }
        if let Some(v) = doc.get("chunk_chars") {
            let c = as_usize(v, "chunk_chars")?;
            // Chunking exists to keep the embedding mean sharp; a chunk
            // this small carries no context and a typo'd 60-for-600
            // should fail here, not surface as mysterious recall loss.
            if c < 100 {
                bail!("chunk_chars below 100 defeats chunking, got {c}");
            }
            cfg.chunk_chars = c;
        }
        if let Some(v) = doc.get("prefix_rows") {
            cfg.prefix_rows =
                u32::try_from(as_usize(v, "prefix_rows")?).context("prefix_rows exceeds u32")?;
        }

        if let Some(v) = doc.get("title_weight") {
            cfg.title_weight = as_f32(v, "title_weight")?;
        }
        if let Some(v) = doc.get("tag_weight") {
            cfg.tag_weight = as_f32(v, "tag_weight")?;
        }
        if let Some(v) = doc.get("desc_weight") {
            cfg.desc_weight = as_f32(v, "desc_weight")?;
        }
        let weights = FieldWeights {
            title: cfg.title_weight,
            tag: cfg.tag_weight,
            desc: cfg.desc_weight,
        };
        if !weights.is_sane() {
            bail!(
                "field weights out of range (finite, 0..=100): title {}, tag {}, desc {}",
                cfg.title_weight,
                cfg.tag_weight,
                cfg.desc_weight
            );
        }

        // Scoring calibration. Ranges are rails, not taste: min_gap and
        // min_cos are cosine-space quantities, so anything outside 0..=1
        // is a unit error; rrf_alpha shares the weights' sanity ceiling.
        // The checks live in helpers because the build FLAGS must reject
        // exactly what the file keys reject — two validators drift.
        if let Some(v) = doc.get("min_gap") {
            cfg.min_gap = check_cosine("min_gap", "cosine-space value", as_f32(v, "min_gap")?)?;
        }
        if let Some(v) = doc.get("rrf_alpha") {
            cfg.rrf_alpha = check_alpha(as_f32(v, "rrf_alpha")?)?;
        }
        if let Some(v) = doc.get("min_cos") {
            // 0.0 is a real override ("floor off"), which is exactly why
            // this field is Option and why absence must stay None.
            cfg.min_cos = Some(check_cosine(
                "min_cos",
                "min cosine",
                as_f32(v, "min_cos")?,
            )?);
        }

        if let Some(v) = doc.get("chunk_penalty") {
            cfg.chunk_penalty = check_cosine(
                "chunk_penalty",
                "coefficient in cosine units",
                as_f32(v, "chunk_penalty")?,
            )?;
        }

        Ok(cfg)
    }

    /// Parse a specific config file. `discover` is the CLI entry point;
    /// this is for callers that already know the path — init uses it to
    /// verify that the file it just scaffolded actually parses. Paths
    /// resolve against the file's own directory, same as discover.
    pub fn load(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let root = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Self::parse(&text, root).with_context(|| format!("in {}", path.display()))
    }

    /// Build-shape overrides: flags win over the file so one-off
    /// experiments (`--dims 128`, a scratch `--out`) don't need an edit
    /// to tracked config. Shape only — the scoring calibration has its
    /// own layer in `with_scoring`, with rails, because unlike these a
    /// scoring flag can bake a value the browser will run forever.
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
        if let Some(d) = dims {
            self.dims = Some(d);
        }
        if let Some(c) = chunk_chars {
            self.chunk_chars = c;
        }
        if let Some(r) = prefix_rows {
            self.prefix_rows = r;
        }
        self
    }

    /// Build-flag layer for the scoring calibration, highest precedence:
    /// flag > chops-search.toml key > compiled default. Validated
    /// through the same rails as the file keys, so a flag cannot bake a
    /// value the config parser would have rejected.
    pub fn with_scoring(mut self, flags: ScoringFlags) -> Result<Self> {
        if let Some(g) = flags.min_gap {
            self.min_gap = check_cosine("--min-gap", "cosine-space value", g)?;
        }
        if let Some(a) = flags.rrf_alpha {
            self.rrf_alpha = check_alpha(a)?;
        }
        if let Some(c) = flags.min_cos {
            // A flag of 0.0 is the explicit "floor off" override, same
            // as the file key; there is no flag spelling for "un-set an
            // override the file made" because nothing needs it — omit
            // both and the floor derives.
            self.min_cos = Some(check_cosine("--min-cos", "min cosine value", c)?);
        }

        if let Some(c) = flags.chunk_penalty {
            self.chunk_penalty = check_cosine("--chunk-penalty", "coefficient in cosine units", c)?;
        }
        Ok(self)
    }
}

/// The three scoring flags as named fields rather than three positional
/// `Option<f32>`s — the same transposition argument as ScoreArgs and
/// FieldWeights: consecutive same-typed parameters compile transposed
/// and surface only as "ranking got quietly worse".
#[derive(Debug, Default, Clone, Copy)]
pub struct ScoringFlags {
    pub min_gap: Option<f32>,
    pub rrf_alpha: Option<f32>,
    pub min_cos: Option<f32>,
    pub chunk_penalty: Option<f32>,
}

/// Range rail shared by file keys and build flags — one validator so a
/// flag can never bake what a key would refuse. `key` and `what` are
/// only for the message: min_gap/min_cos are cosine-space values,
/// chunk_penalty is a coefficient in cosine units; same range, different
/// noun, and the message shouldn't misdescribe the quantity.
fn check_cosine(key: &str, what: &str, v: f32) -> Result<f32> {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        bail!("{key} must be a {what} in 0..=1, got {v}");
    }
    Ok(v)
}

fn check_alpha(v: f32) -> Result<f32> {
    if !v.is_finite() || !(0.0..=100.0).contains(&v) {
        bail!("rrf_alpha must be finite and in 0..=100, got {v}");
    }
    Ok(v)
}

fn as_str<'a>(v: &'a toml::Value, key: &str) -> Result<&'a str> {
    v.as_str()
        .with_context(|| format!("`{key}` must be a string"))
}

fn as_usize(v: &toml::Value, key: &str) -> Result<usize> {
    let n = v
        .as_integer()
        .with_context(|| format!("`{key}` must be an integer"))?;
    usize::try_from(n).with_context(|| format!("`{key}` must be non-negative"))
}

fn as_f32(v: &toml::Value, key: &str) -> Result<f32> {
    match v {
        toml::Value::Float(f) => Ok(*f as f32),
        toml::Value::Integer(i) => Ok(*i as f32),
        _ => bail!("`{key}` must be a number"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn root() -> PathBuf {
        PathBuf::from("/site")
    }

    #[test]
    fn empty_config_is_the_compiled_defaults() {
        let cfg = Config::parse("", root()).unwrap();
        assert_eq!(cfg.content, PathBuf::from("/site/content"));
        assert_eq!(cfg.out, PathBuf::from("/site/static/search"));
        assert_eq!(cfg.model, PathBuf::from("/site/.chops-search/model"));
        assert_eq!(cfg.dims, None, "absent dims must mean the native size");
        assert_eq!(cfg.prefix_rows, DEFAULT_PREFIX_ROWS);
        assert_eq!(cfg.title_weight, W_TITLE);
        // The scoring knobs ship inert unless the corpus calibrated them.
        assert_eq!(cfg.min_gap, 0.0);
        assert_eq!(cfg.rrf_alpha, 0.0);
        assert_eq!(cfg.min_cos, None, "absent min_cos must mean derive");
    }

    #[test]
    fn unknown_key_is_an_error() {
        let err = Config::parse("chunk_size = 600\n", root())
            .unwrap_err()
            .to_string();
        assert!(err.contains("chunk_size"), "{err}");
        // The motivating case for keeping this check through the v5
        // batch: a typo'd gate key must fail the build, not silently
        // ship the gate disarmed.
        let err = Config::parse("min_gp = 0.08\n", root())
            .unwrap_err()
            .to_string();
        assert!(err.contains("min_gp"), "{err}");
    }

    #[test]
    fn calibrated_scoring_keys_parse() {
        let cfg = Config::parse("min_gap = 0.08\nrrf_alpha = 1.0\n", root()).unwrap();
        assert_eq!(cfg.min_gap, 0.08);
        assert_eq!(cfg.rrf_alpha, 1.0);
        assert_eq!(
            cfg.min_cos, None,
            "unrelated keys must not arm the override"
        );
    }

    #[test]
    fn min_cos_zero_is_an_override_not_an_absence() {
        // Same claim as the ScoreArgs test of the same name: "floor off"
        // and "derive the floor" are different engines, and the config
        // layer is the first place the distinction can be lost.
        let cfg = Config::parse("min_cos = 0.0", root()).unwrap();
        assert_eq!(cfg.min_cos, Some(0.0));
        let cfg = Config::parse("min_cos = 0.34", root()).unwrap();
        assert_eq!(cfg.min_cos, Some(0.34));
    }

    #[test]
    fn integer_literals_are_accepted_for_scoring_floats() {
        // `rrf_alpha = 1` is how a human writes 1.0; rejecting it over
        // TOML's int/float distinction would be a paper cut.
        let cfg = Config::parse("rrf_alpha = 1", root()).unwrap();
        assert_eq!(cfg.rrf_alpha, 1.0);
    }

    #[test]
    fn weights_accept_integers_and_floats() {
        // Every config written before BM25F spells these without a
        // decimal point, and TOML types those as integers. Both forms
        // must load, or the change breaks sites on upgrade.
        let cfg = Config::parse(
            "title_weight = 2\ntag_weight = 4\ndesc_weight = 1\n",
            root(),
        )
        .unwrap();
        assert_eq!(cfg.title_weight, 2.0);
        assert_eq!(cfg.tag_weight, 4.0);
        assert_eq!(cfg.desc_weight, 1.0);
        let cfg = Config::parse("tag_weight = 0.0\n", root()).unwrap();
        assert_eq!(cfg.tag_weight, 0.0, "zero means ignore the field");
    }

    #[test]
    fn bad_shape_values_are_rejected() {
        for bad in [
            "dims = 0",
            "dims = \"128\"",
            "dims = -1",
            "chunk_chars = 10",
            "prefix_rows = -1",
            "title_weight = \"2\"",
        ] {
            assert!(
                Config::parse(bad, root()).is_err(),
                "{bad:?} must be rejected"
            );
        }
    }

    #[test]
    fn out_of_range_scoring_values_fail_loudly() {
        for bad in [
            "min_gap = 1.5",
            "min_gap = -0.1",
            "min_cos = 2.0",
            "rrf_alpha = -1",
        ] {
            assert!(
                Config::parse(bad, root()).is_err(),
                "{bad:?} must be rejected, not clamped or ignored"
            );
        }
    }

    #[test]
    fn insane_weights_are_rejected() {
        assert!(Config::parse("tag_weight = -3.0", root()).is_err());
        assert!(Config::parse("title_weight = 1e7", root()).is_err());
    }

    #[test]
    fn each_weight_key_sets_only_its_own_field() {
        // Three near-identical parse blocks reading three near-identical
        // key names: a copy-paste that assigns tag_weight twice compiles
        // and silently ignores one key.
        let d = Config::parse("", root()).unwrap();
        let only_title = Config::parse("title_weight = 9\n", root()).unwrap();
        assert_eq!(
            (
                only_title.title_weight,
                only_title.tag_weight,
                only_title.desc_weight
            ),
            (9.0, d.tag_weight, d.desc_weight)
        );
        let only_tag = Config::parse("tag_weight = 9\n", root()).unwrap();
        assert_eq!(
            (
                only_tag.title_weight,
                only_tag.tag_weight,
                only_tag.desc_weight
            ),
            (d.title_weight, 9.0, d.desc_weight)
        );
        let only_desc = Config::parse("desc_weight = 9\n", root()).unwrap();
        assert_eq!(
            (
                only_desc.title_weight,
                only_desc.tag_weight,
                only_desc.desc_weight
            ),
            (d.title_weight, d.tag_weight, 9.0)
        );
    }

    #[test]
    fn each_scoring_key_sets_only_its_own_field() {
        // Same copy-paste tripwire for the v5 keys — min_gap and min_cos
        // even share a validator, so a pasted block assigning the wrong
        // field would pass every range check.
        let only_gap = Config::parse("min_gap = 0.9\n", root()).unwrap();
        assert_eq!(
            (only_gap.min_gap, only_gap.rrf_alpha, only_gap.min_cos),
            (0.9, 0.0, None)
        );
        let only_alpha = Config::parse("rrf_alpha = 9\n", root()).unwrap();
        assert_eq!(
            (only_alpha.min_gap, only_alpha.rrf_alpha, only_alpha.min_cos),
            (0.0, 9.0, None)
        );
        let only_cos = Config::parse("min_cos = 0.9\n", root()).unwrap();
        assert_eq!(
            (only_cos.min_gap, only_cos.rrf_alpha, only_cos.min_cos),
            (0.0, 0.0, Some(0.9))
        );
    }

    #[test]
    fn shape_flags_beat_file() {
        let cfg = Config::parse("dims = 256\n", root())
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
        // Untouched values keep the file's / defaults' word.
        assert_eq!(cfg.content, PathBuf::from("/site/content"));
    }

    #[test]
    fn build_flags_outrank_file_keys() {
        // Precedence: flag > file > default, and a flag on one knob must
        // leave the file's other values standing.
        let cfg = Config::parse("min_gap = 0.05\nrrf_alpha = 0.5\n", root())
            .unwrap()
            .with_scoring(ScoringFlags {
                min_gap: Some(0.08),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(cfg.min_gap, 0.08, "flag must win");
        assert_eq!(cfg.rrf_alpha, 0.5, "untouched file key must survive");
        assert_eq!(cfg.min_cos, None, "untouched override must stay unset");
    }

    #[test]
    fn empty_flags_are_the_identity() {
        let base = Config::parse("min_gap = 0.08\nmin_cos = 0.0", root()).unwrap();
        let same = base.clone().with_scoring(ScoringFlags::default()).unwrap();
        assert_eq!(same.min_gap, base.min_gap);
        assert_eq!(same.min_cos, base.min_cos);
    }

    #[test]
    fn flags_hit_the_same_rails_as_file_keys() {
        // The reason the validators are shared: a flag must not bake a
        // value the parser would have rejected.
        let base = Config::parse("", root()).unwrap();
        for flags in [
            ScoringFlags {
                min_gap: Some(1.5),
                ..Default::default()
            },
            ScoringFlags {
                rrf_alpha: Some(-1.0),
                ..Default::default()
            },
            ScoringFlags {
                min_cos: Some(f32::NAN),
                ..Default::default()
            },
        ] {
            assert!(
                base.clone().with_scoring(flags).is_err(),
                "{flags:?} must be rejected"
            );
        }
    }

    #[test]
    fn paths_resolve_against_the_config_root() {
        let cfg = Config::parse("out = \"public/find\"", root()).unwrap();
        assert_eq!(cfg.out, PathBuf::from("/site/public/find"));
    }
}
