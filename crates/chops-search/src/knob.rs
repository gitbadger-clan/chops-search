//! Sweepable scoring knobs as data.
//!
//! One `apply`/`current` pair instead of nine near-identical sweep loops
//! that can each transpose a field: `calibrate` iterates `Knob::ALL`,
//! and a knob added here is automatically walked, tabled, and judged
//! without touching the walk itself. The transposition hazard that
//! justifies `FieldWeights` and `ScoreArgs` elsewhere is the same one
//! this enum removes from sweeping.

use chops_search_core::score::ScoreOpts;

/// Every knob `calibrate` walks.
///
/// `strong_cos` is deliberately absent: it disables at infinity, so a
/// linear axis has no honest cell for "off", and every measured session
/// so far has kept it off. When a hypothesis actually names it, sweep it
/// by hand with `eval --strong-cos`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Knob {
    MinGap,
    RrfAlpha,
    /// In the walk even though a 1-D axis undersells it: k and alpha are
    /// coupled (see ScoreOpts::rrf_alpha), so a candidate here is a
    /// pointer at the joint `eval --sweep-rrf-k --sweep-rrf-alpha` grid,
    /// not a value to pin from this table alone.
    RrfK,
    KwFloor,
    ChunkPenalty,
    WTitle,
    WTag,
    WDesc,
    /// Walked despite being engine-derived by default, because it is the
    /// knob with the recorded contamination history: the 0.28–0.34 band
    /// holds noise and signal simultaneously, so this is exactly where
    /// per-case flip lists and the collateral check earn their keep over
    /// a summary total.
    MinCos,
}

impl Knob {
    pub const ALL: [Knob; 9] = [
        Knob::MinGap,
        Knob::RrfAlpha,
        Knob::RrfK,
        Knob::KwFloor,
        Knob::ChunkPenalty,
        Knob::WTitle,
        Knob::WTag,
        Knob::WDesc,
        Knob::MinCos,
    ];

    /// The flag/header spelling, which is also what `--knob` parses.
    pub fn name(self) -> &'static str {
        match self {
            Knob::MinGap => "min_gap",
            Knob::RrfAlpha => "rrf_alpha",
            Knob::RrfK => "rrf_k",
            Knob::KwFloor => "kw_floor",
            Knob::ChunkPenalty => "chunk_penalty",
            Knob::WTitle => "w_title",
            Knob::WTag => "w_tag",
            Knob::WDesc => "w_desc",
            Knob::MinCos => "min_cos",
        }
    }

    pub fn from_name(s: &str) -> Option<Knob> {
        Knob::ALL.iter().copied().find(|k| k.name() == s)
    }

    /// The chops-search.toml key a suggestion would name, or None for
    /// compiled-constant knobs. The suggestion printer uses this to
    /// distinguish a config edit from "a winning value here is a
    /// format-boundary conversation, not a flag to carry around".
    pub fn config_key(self) -> Option<&'static str> {
        match self {
            Knob::MinGap => Some("min_gap"),
            Knob::RrfAlpha => Some("rrf_alpha"),
            Knob::MinCos => Some("min_cos"),
            Knob::WTitle => Some("title_weight"),
            Knob::WTag => Some("tag_weight"),
            Knob::WDesc => Some("desc_weight"),
            // v6: chunk_penalty rides in chops-search.toml and index.bin
            // alongside the other calibration keys.
            Knob::ChunkPenalty => Some("chunk_penalty"),
            Knob::RrfK | Knob::KwFloor => None,
        }
    }

    pub fn apply(self, o: &mut ScoreOpts, v: f32) {
        match self {
            Knob::MinGap => o.min_gap = v,
            Knob::RrfAlpha => o.rrf_alpha = v,
            Knob::RrfK => o.rrf_k = v,
            Knob::KwFloor => o.kw_confidence = v,
            Knob::ChunkPenalty => o.chunk_penalty = v,
            Knob::WTitle => o.weights.title = v,
            Knob::WTag => o.weights.tag = v,
            Knob::WDesc => o.weights.desc = v,
            Knob::MinCos => o.min_cos = v,
        }
    }

    pub fn current(self, o: &ScoreOpts) -> f32 {
        match self {
            Knob::MinGap => o.min_gap,
            Knob::RrfAlpha => o.rrf_alpha,
            Knob::RrfK => o.rrf_k,
            Knob::KwFloor => o.kw_confidence,
            Knob::ChunkPenalty => o.chunk_penalty,
            Knob::WTitle => o.weights.title,
            Knob::WTag => o.weights.tag,
            Knob::WDesc => o.weights.desc,
            Knob::MinCos => o.min_cos,
        }
    }

    /// Default walk axis, bracketing the measured history rather than a
    /// generic range: kw_floor's cliff sits in 0.10–0.20 with a plateau
    /// through 0.50, min_gap shipped at 0.08 out of a 0.02–0.16 walk,
    /// chunk_penalty's live candidates came from {0.02..0.16}, rrf_alpha
    /// is inert below ~1 at rrf_k 60 (see ScoreOpts::rrf_alpha), and the
    /// min_cos axis straddles the 0.28–0.34 noise/signal band on both
    /// sides. Cells go where the curve can actually bend.
    pub fn default_axis(self) -> Vec<f32> {
        match self {
            Knob::MinGap => vec![0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16],
            Knob::RrfAlpha => vec![0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
            Knob::RrfK => vec![4.0, 8.0, 16.0, 32.0, 60.0],
            Knob::KwFloor => vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            Knob::ChunkPenalty => vec![0.0, 0.02, 0.05, 0.08, 0.12, 0.16],
            Knob::WTitle => vec![0.0, 0.5, 1.0, 2.0, 3.0, 4.0],
            Knob::WTag => vec![0.0, 1.0, 2.0, 4.0, 6.0, 8.0],
            Knob::WDesc => vec![0.0, 0.5, 1.0, 2.0, 3.0],
            Knob::MinCos => vec![0.24, 0.26, 0.28, 0.30, 0.32, 0.34, 0.36, 0.38, 0.40],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chops_search_core::keyword::FieldWeights;

    /// Nothing at a compiled-in default, so a `current` reading the
    /// wrong field shows up.
    fn base() -> ScoreOpts {
        ScoreOpts {
            min_cos: 0.34,
            chunk_penalty: 0.05,
            kw_confidence: 0.25,
            min_gap: 0.11,
            strong_cos: 0.55,
            weights: FieldWeights {
                title: 3.0,
                tag: 7.0,
                desc: 0.5,
            },
            rrf_alpha: 0.75,
            rrf_k: 12.0,
        }
    }

    #[test]
    fn names_round_trip() {
        for k in Knob::ALL {
            assert_eq!(Knob::from_name(k.name()), Some(k));
        }
        assert_eq!(Knob::from_name("strong_cos"), None);
        assert_eq!(Knob::from_name("min-gap"), None, "underscore spelling only");
    }

    #[test]
    fn apply_touches_only_its_own_field() {
        // Nine same-typed knobs landing on nine f32 slots is exactly
        // where a transposition compiles, type-checks, and surfaces as
        // "calibrate says the wrong knob moved".
        const SENTINEL: f32 = 9.25;
        for k in Knob::ALL {
            let mut o = base();
            k.apply(&mut o, SENTINEL);
            assert_eq!(k.current(&o), SENTINEL, "{} did not apply", k.name());
            for other in Knob::ALL {
                if other != k {
                    assert_eq!(
                        other.current(&o),
                        other.current(&base()),
                        "{} disturbed {}",
                        k.name(),
                        other.name()
                    );
                }
            }
        }
    }

    #[test]
    fn axes_are_sorted_finite_and_contain_a_disabling_or_shipping_cell() {
        for k in Knob::ALL {
            let axis = k.default_axis();
            assert!(
                axis.len() >= 3,
                "{} axis too small to show a shape",
                k.name()
            );
            for w in axis.windows(2) {
                assert!(w[0] < w[1], "{} axis not strictly increasing", k.name());
            }
            assert!(axis.iter().all(|v| v.is_finite()));
        }
        // Floor-style knobs must include their off cell, so every table
        // states what disabling costs.
        for k in [
            Knob::MinGap,
            Knob::RrfAlpha,
            Knob::KwFloor,
            Knob::ChunkPenalty,
        ] {
            assert_eq!(
                k.default_axis()[0],
                0.0,
                "{} axis must start at off",
                k.name()
            );
        }
        // rrf_k has no disabling value; its axis must not pretend one.
        assert!(Knob::RrfK.default_axis()[0] > 0.0);
    }
}
