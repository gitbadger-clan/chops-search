+++
title = "Configuration"
description = "Every chops-search.toml key with its default, how config discovery works, and which values get baked into index.bin."
weight = 5
[taxonomies]
tags = ["configuration", "toml", "defaults", "tuning", "calibration"]
+++

Configuration lives in `chops-search.toml` at the site root. Discovery walks
up from the current directory the way cargo finds `Cargo.toml`, so commands
work from anywhere inside the site. **Paths resolve relative to the config
file**, not the working directory.

Every key is optional; an empty file (or no file) means the compiled
defaults. An **unknown key is an error**, not a shrug: a misspelled
`min_gp = 0.08` would otherwise silently ship the corroboration gate
disarmed, which is exactly the honesty gap the scoring keys exist to close.

{% code(title="chops-search.toml") %}
```toml
content = "content"            # the Zola content tree to index
out     = "static/search"      # where artifacts + runtime land
model   = ".chops-search/model"

# dims = 128        # PCA target; unset means the model's native size (256)
# chunk_chars = 600
# prefix_rows = 2048

# BM25F field weights (body is fixed at 1.0)
# title_weight = 2
# tag_weight   = 4
# desc_weight  = 1

# Scoring calibration: baked into index.bin at build time
# min_gap   = 0.08   # corroboration gate; 0 (the default) disarms it
# rrf_alpha = 1.0    # confidence-weighted fusion; 0 (the default) is plain RRF
# min_cos   = 0.34   # relevance-floor OVERRIDE; unset derives from dims
```
{% end %}

## Shape keys

| Key | Default | Notes |
| --- | --- | --- |
| `content` | `content` | Walked recursively for `.md` files |
| `out` | `static/search` | Inside `static/` so your SSG copies it verbatim |
| `model` | `.chops-search/model` | The lockfile lives beside this directory |
| `dims` | unset (native size) | Real PCA, not truncation; `init` scaffolds `128`. [Re-eval after changing](/how-to/manage-the-model/) |
| `chunk_chars` | `600` | Smaller sharpens rare-word signal, costs more vectors. Values below 100 are rejected |
| `prefix_rows` | `2048` | Larger trades eager payload for fewer range requests |

## BM25F field weights

What a length-normalised occurrence in each field is worth against one in
the body (body is fixed at 1.0, so three knobs, not four). Each field's term
frequency is normalised by that field's *own* average length before the
weight applies, and saturation happens once on the combined value, so a
weight biases without inflating. The useful range is therefore much smaller
than a plain multiplier's would be; 0 ignores a field entirely.

| Key | Default | Notes |
| --- | --- | --- |
| `title_weight` | `2` | |
| `tag_weight` | `4` | Tags outweigh titles: they're the author's own summary |
| `desc_weight` | `1` | Front-matter `description`, indexed as its own field. Parity with body is deliberate; it measured no better weighted up |

## Scoring calibration

`min_gap`, `rrf_alpha`, and `min_cos` are per-corpus calibrated values, and
they follow the same provenance rule as the field weights: a value
calibrated against a corpus travels with the corpus. **`build` writes all of
them into `index.bin`, and the engine reads them at construction**, so the
browser, a bare `chops-search eval`, and CI all score the same configuration
from the same bytes. The `eval` and `query` flags still override per run for
sweeping: the config states what ships, the flags state deviations from it.

| Key | Default | Notes |
| --- | --- | --- |
| `min_gap` | `0` (gate disarmed) | The corroboration gate threshold. When a query has no keyword evidence and no document stands out from the corpus median by at least this much, the semantic ranking is suppressed rather than served |
| `rrf_alpha` | `0` (plain RRF) | Confidence-weighted fusion: the keyword list's vote scales by 1 + alpha × keyword confidence |
| `min_cos` | unset (derived from dims) | Relevance-floor **override**. Unset means the engine derives the floor from dimensionality, which tracks `dims` changes automatically and is right for almost every corpus. An explicit `0.0` is itself an override, meaning "floor off" |

These ship commented out in the scaffolded config because they are
calibrated, not chosen: a value that helps one corpus hurts another. The
loop is sweep with `chops-search eval`, verify the mechanism with
`chops-search query`, then pin the winning value here and rebuild. See
[the evaluation tutorial](/tutorials/evaluate-your-search/) and
[how ranking works](/explanations/hybrid-ranking/).

## Validation

Out-of-range values fail the build loudly rather than being clamped or
ignored: `min_gap` and `min_cos` must be cosine-space values in 0..=1,
`rrf_alpha` and the field weights must be finite and in 0..=100, and the
`build` flags reject exactly what the file keys reject, so a flag cannot
bake a value the config parser would have refused.

## Precedence

Flag > file key > compiled default. The `build` flags (`--dims`,
`--chunk-chars`, `--prefix-rows`, `--content`, `--model`, `--out`,
`--min-gap`, `--rrf-alpha`, `--min-cos`) override the file for a single
run; the scoring flags on `eval` and `query` sweep values against a built
index without rebuilding anything. Only what `build` bakes reaches the
browser, and the build log's `scoring:` line states what shipped.
