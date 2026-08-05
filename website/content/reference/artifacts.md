+++
title = "Artifacts"
description = "Every file build emits, its loading discipline, its cache policy, and what changes when."
weight = 10
+++

`chops-search build` writes two kinds of file into `out` (default
`static/search/`): content-hashed index artifacts, and the unhashed browser
runtime.

## Index artifacts

| File | Loading | Contents |
| --- | --- | --- |
| `model.meta.<hash>.bin` | Eager, gzipped | Complete vocab + per-row quantization scales. Never partial: a truncated vocab tokenizes silently wrong |
| `model.prefix.<hash>.i8` | Eager | Top ~2048 frequency-ordered rows, covering most queries outright |
| `index.<hash>.bin` | Eager, gzipped | Chunk vectors, document URLs and titles, keyword postings |
| `model.rows.<hash>.i8` | Range-fetched per query | Full matrix, headerless raw i8; row *i* at byte *i × dim* |
| `snippets.<hash>.bin` | Range-fetched after ranking | Per-chunk display text |
| `manifest.json` | Eager, revalidating | Names every hashed file; the only unhashed artifact |

Filenames carry a build hash so everything above `manifest.json` can be
served immutable. Artifacts are byte-stable given the same content and model:
sorted walks, sorted postings, deterministic tie-breaks, so an unchanged site
rebuilds to identical hashes and browser caches stay warm.

## What changes when

| You changed | Files that get new hashes |
| --- | --- |
| Site content | `index`, `snippets` |
| The model or `dims` | `model.meta`, `model.prefix`, `model.rows`, plus `index` and `snippets` (chunk vectors re-embed) |
| Nothing | Nothing; the build is reproducible |

## The runtime

| File | Role |
| --- | --- |
| `chops-search.js` | Page script: overlay/inline UI, worker bootstrap |
| `chops-search.css` | Overlay styles, themed via `currentColor` and `.chops` custom properties |
| `search-worker.js` | The Web Worker that plans, fetches, and queries |
| `pkg/` | The wasm engine and its glue, requested with `?v=<build hash>` |

The runtime files have stable names (they're referenced from your templates),
so they must not be cached immutable; the wasm under `pkg/` is version-queried
and safe to pin. The [deployment guide](/how-to/deploy-with-caching/) has the
exact `_headers` policy.
