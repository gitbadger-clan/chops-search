+++
title = "The model is a file you read offsets from"
description = "Why a model2vec lookup table can be range-fetched row by row, and the four loading disciplines that keep a query under a kilobyte."
weight = 1
[taxonomies]
tags = ["embeddings", "range-requests", "bandwidth", "performance", "http"]
+++

Every existing answer to "semantic search on a static site" is some flavour of
the same compromise: run a server, pay a SaaS, or ship the entire embedding
model to the browser and ask visitors to download tens of megabytes before
they can type. chops-search takes none of those, and this page explains the
one architectural decision that makes it possible.

## Why it's even possible

A model2vec/potion model isn't a transformer at inference time. It's a lookup
table: one static vector per vocabulary token, and a sentence embedding is
the mean of its token rows. No attention, no layers, no runtime beyond "look
up rows, average them".

A lookup table has a property transformers don't: **you can read one row
without the others**. If a query tokenizes to six tokens, you need six rows.
At int8 quantization a row is `dim` bytes, and row `i` lives at byte
`i × dim`. That's an HTTP range request. The model stops being a download and
becomes an address space.

## Four loading disciplines

Each artifact answers "when does the browser fetch this" differently:

| Artifact | Discipline |
| --- | --- |
| `model.meta.bin` | Eager, gzipped, **never partial**. The complete vocab plus per-row scales. A truncated vocab doesn't fail loudly; it tokenizes silently wrong. |
| `model.prefix.i8` | Eager. The top ~2048 frequency-ordered rows, covering most real queries outright. |
| `index.bin` | Eager, gzipped. Chunk vectors, document table, keyword postings, and the scoring configuration the build baked in. |
| `model.rows.i8` | Range-fetched per query. The full matrix as headerless raw i8; no framing, because the offset arithmetic *is* the format. |
| `snippets.bin` | Offset table range-fetched once at boot; display text range-fetched after ranking. |

The rows are frequency-ordered against *your corpus*, so the eager prefix
covers the vocabulary your content actually attracts rather than a generic
top-2048. That decision has a cache consequence covered below.

At query time a Web Worker asks the wasm engine which byte ranges it's
missing, fetches them, feeds them back, and renders ranked results. Fetched
rows persist in a Cache API row cache keyed to the build, and the artifacts
ship content-hashed under immutable headers, so the second time anyone
searches for anything vaguely similar the network doesn't get involved at
all.

## What this buys

After initial load, most queries need no network at all (a prefix hit or a
warm row cache), and the rest fetch a handful of 128-byte rows. That claim
is checkable rather than asserted: `chops-search plan "your query"` prints
each token's row and whether the eager prefix already holds it, and
`--curl` emits the range requests so the server can confirm the byte
counts. Over a whole labelled set it reports prefix hit rate, share of
queries fetching nothing, and mean bytes per query. On this site's gate
fixture (36 cases, measured 2026-08-19 at `40805d4`, dims 128, the shipped
2048-row prefix, gap 8) 26 of 36 queries fetch nothing, the prefix holds
89% of the rows the fixture needs, and the mean cost is 71 bytes per query
against a 3.6 MB rows file, 512 bytes worst case. `eval`, warm across the
run, lands on the same numbers. Halving the prefix to 1024 rows saves
every visitor 128 KB once and costs 60 bytes more per query on average,
but drops the zero-fetch share from 72% to 53%, and a first-keystroke
round trip is what the prefix exists to remove; going the other way, 90%
coverage needs 3306 rows, another 160 KB for every visitor to shave the
last few range fetches off the tail. 2048 stays. The fixture is an upper
bound on hit rate, since visitors type words its author did not think of,
so read these as the calibration numbers, not a promise.

The honest fine print: because the row matrix is frequency-permuted against
the corpus, a content rebuild changes every artifact's hash, not just the
index. What survives a deploy is the byte-stable build (an unchanged site
rebuilds to identical hashes, so caches stay warm through no-op deploys) and
the immutable headers within one. Compare tools that embed the index in the
wasm binary, where every content change recompiles and re-ships the engine
itself; here the engine binary is content-independent and only its cache key
rotates.

The [artifact reference](/reference/artifacts/) has the exact file table;
[Designed degradation](/explanations/designed-degradation/) covers what
happens when a needed row can't be loaded.
