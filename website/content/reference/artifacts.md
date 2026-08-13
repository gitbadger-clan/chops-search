+++
title = "Artifacts"
description = "Every file build emits, its loading discipline, its cache policy, and what changes when."
weight = 10
[taxonomies]
tags = ["artifacts", "build-output", "caching", "hashing"]
+++

`chops-search build` writes two kinds of file into `out` (default
`static/search/`): content-hashed index artifacts, and the unhashed browser
runtime.

## Index artifacts

| File | Loading | Contents |
| --- | --- | --- |
| `model.meta.<hash>.bin` | Eager, gzipped | Complete vocab + per-row quantization scales. Never partial: a truncated vocab tokenizes silently wrong |
| `model.prefix.<hash>.i8` | Eager | Top ~2048 frequency-ordered rows, covering most queries outright |
| `index.<hash>.bin` | Eager, gzipped | Chunk vectors, document URLs and titles, keyword postings, plus the BM25F field weights and the scoring calibration (`min_gap`, `rrf_alpha`, `min_cos` override) baked at build time, so the browser scores with what the site configured |
| `model.rows.<hash>.i8` | Range-fetched per query | Full matrix, headerless raw i8; row *i* at byte *i × dim* |
| `snippets.<hash>.bin` | Range-fetched | Offset table fetched once at boot as a single small range; per-chunk display text fetched after ranking |
| `manifest.json` | Eager, revalidating | Names every hashed file; the only unhashed artifact |

**Gzip siblings.** "Eager, gzipped" means the build writes a `.gz` sibling
next to the raw file (`index.<hash>.bin.gz`) and the worker fetches it,
decompressing with `DecompressionStream`. This is deliberate: hosts compress
a fixed list of content types and `application/octet-stream` isn't on it,
so relying on host compression would ship the eager payload raw. Compressing
at build time keeps the saving host-agnostic. Range-served files are never
compressed, since a byte offset into a gzip stream is meaningless.

**One hash, shared.** Every hashed filename carries the same build hash,
computed over the whole artifact set. The model rows are frequency-ordered
against *your* corpus (so the eager prefix covers the queries your content
attracts), which makes the model artifacts corpus-dependent: a content
change re-derives the frequency permutation, so all five files legitimately
travel together under one hash. Artifacts are byte-stable given the same
content and model: sorted walks, sorted postings, deterministic tie-breaks,
so an unchanged site rebuilds to identical hashes and browser caches stay
warm.

## What changes when

| You changed | Effect |
| --- | --- |
| Site content | New build hash; all five artifacts get new names (the row matrix is frequency-permuted against the corpus, so it changes with the content) |
| The model, `dims`, field weights, or scoring calibration | New build hash, same as above; weights and calibration live in `index.bin` |
| Nothing | Nothing; the build is reproducible and every cache stays warm |

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
