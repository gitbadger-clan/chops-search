+++
title = "One tokenizer, enforced structurally"
description = "Why the engine is one Rust core compiled twice, and the parity tests that pin the tokenizer and embeddings to the reference implementation."
weight = 20
[taxonomies]
tags = ["tokenizer", "wordpiece", "parity", "unicode", "wasm"]
+++

The classic failure mode of client-side search is two tokenizers: one in the
build tool that indexed your content, one in the browser that reads queries,
and a promise that someone keeps them in sync. The promise always breaks, and
it breaks silently, as queries that should match and don't.

chops-search removes the promise. The core crate, `chops-search-core`, is
pure Rust with no I/O: WordPiece tokenizer, int8 row store, scoring, fusion,
artifact formats. It compiles unchanged to native for the build CLI and to
wasm for the browser. The tokenizer that indexed your content is bit-for-bit
the tokenizer that handles queries, because it's the same compiled code. The
guarantee is structural, not aspirational.

## Pinned to the reference, per codepoint

Being consistent with yourself isn't enough if you're consistently wrong, so
the tokenizer is also pinned to the official implementations:

- **Codepoint sweeps against HuggingFace.** Around 1,500 codepoints across
  eight Unicode blocks, each probed standalone, between Latin letters, and
  between ideographs, comparing exact token ids. Fixtures only prove what
  someone thought to write down; the sweep found three real divergences (CJK
  spacing, symbol-vs-punctuation, control-character deletion) that survived a
  fixture suite that looked thorough.
- **Embedding parity with model2vec-rs.** Fixture sentences run through both
  chops-search and MinishLab's official implementation, asserting cosine
  above 0.9999 per input. If tokenization or quantization drifts from the
  reference, CI fails.

## The details that silently poison queries

Each of these is a test, because each produces wrong embeddings with no
error when handled differently than the model expects:

- No `[CLS]`/`[SEP]` tokens: model2vec means over content tokens only.
- Unknown words are deleted, not mapped to `[UNK]`.
- Accents strip (café → cafe), matching the model's tokenizer config.
- `embed()` returns nothing rather than a shrunken mean when a needed row
  isn't loaded, so a known-but-unloaded token stays distinguishable from an
  out-of-vocabulary one. What happens then is the subject of
  [Designed degradation](/explanations/designed-degradation/).
