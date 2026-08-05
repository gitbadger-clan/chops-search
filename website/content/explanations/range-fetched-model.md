+++
title = "The model is a file you read offsets from"
description = "Why a model2vec lookup table can be range-fetched row by row, and the four loading disciplines that make a query cost 0.1 KB."
weight = 1
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
| `index.bin` | Eager, gzipped. Chunk vectors, document table, keyword postings. |
| `model.rows.i8` | Range-fetched per query. The full matrix as headerless raw i8; no framing, because the offset arithmetic *is* the format. |
| `snippets.bin` | Range-fetched after ranking, for display text only. |

At query time a Web Worker asks the wasm engine which byte ranges it's
missing, fetches them, feeds them back, and renders ranked results. Fetched
rows persist in a Cache API row cache, and the artifacts ship content-hashed
under immutable headers, so the second time anyone searches for anything
vaguely similar the network doesn't get involved at all.

## What this buys

After initial load, most queries need no network at all (a prefix hit or a
warm row cache), and the rest average around 0.1 KB range-fetched. The eager
payload is a few hundred kilobytes, most of it gzip-friendly, and the wasm is
content-independent: a content rebuild touches `index.bin` and
`snippets.bin`, the model files change only when the model does, and the
engine caches across every deploy. Compare tools that embed the index in the
wasm binary, where every content change recompiles and re-ships the engine.

The [artifact reference](/reference/artifacts/) has the exact file table;
[Designed degradation](/explanations/designed-degradation/) covers what
happens when a needed row can't be loaded.
