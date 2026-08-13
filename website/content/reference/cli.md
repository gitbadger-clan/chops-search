+++
title = "CLI"
description = "Every chops-search subcommand and flag. Defaults are stated inline; --help says the same things."
weight = 1
[taxonomies]
tags = ["commands", "flags", "subcommands"]
+++

All commands resolve `chops-search.toml` by walking up from the current
directory, and per-command flags override its values for that run. A global
`--site <DIR>` starts the walk somewhere else, useful from a repo root or in
CI.

## `chops-search init`

Scaffold a site's integration: `chops-search.toml`, `content/search.md`, and
`.gitignore` entries, then print the template snippet for a site-wide search
box. Nothing is ever overwritten; re-running reports what it skipped.

| Flag | Effect |
| --- | --- |
| `--no-page` | Skip `content/search.md`, for sites using only the overlay |

## `chops-search model fetch [REPO]`

Download the embedding model and record what landed in
`.chops-search/model.lock.json`. The only command that touches the network.
`REPO` is a HuggingFace repo, default `minishlab/potion-base-8M`; any
model2vec model works, the listed ones are tested.

| Flag | Effect |
| --- | --- |
| `--revision <SHA>` | Fetch a specific upstream revision (use the lockfile's in CI). Default: the repo's default branch, resolved to a commit so the lockfile pins something immutable |
| `--dir <DIR>` | Destination. Default: `model` from chops-search.toml |

## `chops-search model verify`

Re-hash the model directory against the lockfile; non-zero exit on mismatch.
No network.

| Flag | Effect |
| --- | --- |
| `--dir <DIR>` | Model directory. Default: `model` from chops-search.toml |

## `chops-search build`

Read the content tree and the model, write hashed artifacts plus the wasm
engine, worker, page script, and stylesheet into the output directory. The
BM25F field weights and the scoring calibration from `chops-search.toml` are
written into `index.bin`, so the browser scores with what the site
configured; the build log's `scoring:` line states what shipped.

| Flag | Default | Effect |
| --- | --- | --- |
| `--content <DIR>` | `content` from config | Content directory |
| `--model <DIR>` | `model` from config | Model directory |
| `--out <DIR>` | `out` from config | Output directory |
| `--prefix-rows <N>` | 2048 | Rows bundled eagerly; larger means bigger eager payload, fewer range requests |
| `--chunk-chars <N>` | 600 | Target chunk size; smaller sharpens rare-word signal, costs more vectors |
| `--dims <N>` | `dims` from config, or the model's native size | PCA target; re-run `eval` after changing |
| `--min-gap <GAP>` | `min_gap` from config (0, disarmed) | Corroboration gate threshold to bake into index.bin. Calibrate with `eval --min-gap` first; this flag ships the value, it doesn't sweep it |
| `--rrf-alpha <ALPHA>` | `rrf_alpha` from config (0, plain RRF) | Fusion weighting to bake into index.bin |
| `--min-cos <COS>` | `min_cos` from config (derive from dims) | Relevance-floor override to bake into index.bin; 0 disables the floor |
| `--no-runtime` | off | Artifacts only, skip the wasm/JS runtime |

## `chops-search docs`

Print every indexed document with its URL and chunk count. Run it after
adding a post; these URLs are what `eval` expectations must match.

| Flag | Default | Effect |
| --- | --- | --- |
| `--artifacts <DIR>` | `out` from config | Artifacts directory to read |

## `chops-search query <QUERY>`

Explain a ranking: how the query tokenized on both sides, per-term keyword
scores with document frequencies and per-field term frequencies, best-chunk
cosine per document, and each engine's contribution to the fused order.
Calls the same scoring code as the ranker, so it cannot drift. The scoring
flags exist for diagnosis: they override what `index.bin` baked, for this
run only.

| Flag | Default | Effect |
| --- | --- | --- |
| `--artifacts <DIR>` | `out` from config | Artifacts directory to read |
| `--limit <N>` | 20 | Rows to print |
| `--kw-floor <FRACTION>` | 0.30 | Keyword evidence gate: below this confidence ratio the keyword list is suppressed from fusion. 0 disables |
| `--w-title <WEIGHT>` | from index.bin | BM25F title weight; 0 asks whether a result still wins without its title |
| `--w-tag <WEIGHT>` | from index.bin | BM25F tag weight |
| `--w-desc <WEIGHT>` | from index.bin | BM25F description weight; 0 asks whether a result is riding on its description |
| `--min-gap <GAP>` | from index.bin | Corroboration gate threshold |
| `--strong-cos <COS>` | off | Best-chunk cosine at or above which the gate never fires. Disables at infinity, not at 0: 0 would disable the gate, not the hatch |
| `--rrf-alpha <ALPHA>` | from index.bin | Fusion weighting; at the default `rrf_k` of 60 the keyword list needs to reach about 2 before it can overturn a semantic first place |
| `--rrf-k <K>` | 60 | RRF rank discount; smaller sharpens the top of the curve |

## `chops-search eval`

Run a labelled query set through the real engine over the real byte path
(plan, range-fetch, ingest, search) and report recall@1 and recall@3 by
kind, plus real bytes-per-query. Baked values from `index.bin` are the
defaults; every scoring flag overrides for the run, which is how sweeping
works without a rebuild.

| Flag | Default | Effect |
| --- | --- | --- |
| `--queries <FILE>` | `fixtures/queries.toml` beside the config | Query set |
| `--kind <KIND>` | all | Only run cases of this kind |
| `--fail-under <FRACTION>` | 0.0 | Exit non-zero below this overall recall@1. Ignored in sweep mode |
| `--min-cos <COS>` | index.bin override, or derived from dims | Sweep the relevance floor |
| `--chunk-penalty <COEFF>` | 0.02 | Sweep the chunk-count correction; 0 disables |
| `--kw-floor <FRACTION>` | 0.30 | Sweep the keyword evidence gate; 0 disables |
| `--w-title <WEIGHT>` | from index.bin | Sweep the BM25F title weight |
| `--w-tag <WEIGHT>` | from index.bin | Sweep the BM25F tag weight |
| `--w-desc <WEIGHT>` | from index.bin | Sweep the BM25F description weight; 0 answers whether your descriptions earn their keep |
| `--min-gap <GAP>` | from index.bin | Sweep the corroboration gate |
| `--strong-cos <COS>` | off | Sweep the gate's escape hatch; disables at infinity |
| `--rrf-alpha <ALPHA>` | from index.bin | Sweep the fusion weighting; values below ~1 are inert at the default `rrf_k`, so sweep the two jointly |
| `--rrf-k <K>` | 60 | Pin a swept rank discount |
| `--sweep-rrf-k <LIST>` | | Comma-separated `rrf_k` values, e.g. `2,4,8,16,32,60`. Enables sweep mode |
| `--sweep-rrf-alpha <LIST>` | | Comma-separated `rrf_alpha` values, e.g. `0,0.5,1,2`. Enables sweep mode |

**Sweep mode.** With either sweep flag the case set runs once per point of
the k × alpha grid and prints recall per cell instead of per-case lines;
every other scoring flag acts as the fixed base for every cell, and the
best cell is re-run for its per-kind breakdown and failure list.
`--fail-under` is ignored, since half the grid is supposed to be worse than
the baseline; that's what a sweep is.

## `chops-search completions <SHELL>`

Emit a conventional static completion script. Prefer the
[dynamic path](/how-to/shell-completion/), which computes candidates at
completion time.
