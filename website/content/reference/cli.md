+++
title = "CLI"
description = "Every chops-search subcommand and flag. Defaults are stated inline; --help says the same things."
weight = 1
[taxonomies]
tags = ["commands", "flags", "subcommands"]
+++

All commands resolve `chops-search.toml` by walking up from the current
directory, and per-command flags override its values for that run.

## `chops-search init`

Scaffold a site's integration: `chops-search.toml`, `content/search.md`, and
`.gitignore` entries, then print the template snippet for a site-wide search
box. Nothing is ever overwritten; re-running reports what it skipped.

| Flag | Effect |
| --- | --- |
| `--no-page` | Skip `content/search.md`, for sites using only the overlay |

## `chops-search model fetch`

Download the embedding model and record what landed in
`.chops-search/model.lock.json`. The only command that touches the network.

| Flag | Effect |
| --- | --- |
| `--revision <REV>` | Fetch a specific upstream revision (use the lockfile's in CI) |

## `chops-search model verify`

Re-hash the model directory against the lockfile; non-zero exit on mismatch.

## `chops-search build`

Read the content tree and the model, write hashed artifacts plus the wasm
engine, worker, page script, and stylesheet into the output directory.
Content changes touch only `index.bin` and `snippets.bin`.

| Flag | Default | Effect |
| --- | --- | --- |
| `--content <DIR>` | `content` from config | Content directory |
| `--model <DIR>` | `model` from config | Model directory |
| `--out <DIR>` | `out` from config | Output directory |
| `--prefix-rows <N>` | 2048 | Rows bundled eagerly; larger means bigger eager payload, fewer range requests |
| `--chunk-chars <N>` | 600 | Target chunk size; smaller sharpens rare-word signal, costs more vectors |
| `--dims <N>` | 128 | PCA target (native 256); re-run `eval` after changing |
| `--no-runtime` | off | Artifacts only, skip the wasm/JS runtime |

## `chops-search docs`

Print every indexed document with its URL and chunk count. Run it after
adding a post; these URLs are what `eval` expectations must match.

| Flag | Default | Effect |
| --- | --- | --- |
| `--artifacts <DIR>` | `out` from config | Artifacts directory to read |

## `chops-search query <QUERY>`

Explain a ranking: how the query tokenized on both sides, per-term keyword
scores with document frequencies, best-chunk cosine per document, and each
engine's contribution to the fused order. Calls the same scoring code as the
ranker, so it cannot drift.

| Flag | Default | Effect |
| --- | --- | --- |
| `--artifacts <DIR>` | `out` from config | Artifacts directory to read |
| `--limit <N>` | 20 | Rows to print |

## `chops-search eval`

Run a labelled query set through the real engine over the real byte path and
report recall@1 and recall@3 by kind, plus real bytes-per-query.

| Flag | Default | Effect |
| --- | --- | --- |
| `--queries <FILE>` | `fixtures/queries.toml` beside the config | Query set |
| `--kind <KIND>` | all | Only run cases of this kind |
| `--fail-under <FRACTION>` | 0.0 | Exit non-zero below this overall recall@1 |
| `--min-cos <COS>` | derived from dims | Sweep the relevance floor |
| `--chunk-penalty <COEFF>` | 0.02 | Sweep the chunk-count correction; 0 disables |
| `--artifacts <DIR>` | `out` from config | Artifacts directory to read |

## `chops-search completions <SHELL>`

Emit a conventional static completion script. Prefer the
[dynamic path](/how-to/shell-completion/), which computes candidates at
completion time.
