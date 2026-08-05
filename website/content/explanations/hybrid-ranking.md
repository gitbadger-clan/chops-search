+++
title = "How ranking works"
description = "BM25 over pre-WordPiece tokens, cosine over embedded chunks, reciprocal rank fusion, and the relevance floor that makes empty results possible."
weight = 10
+++

Two engines, fused. They fail differently, which is the whole point: keyword
nails exact rare terms and scores zero on paraphrases; semantic does the
reverse.

## The keyword engine

BM25 with length normalisation, over word-level tokens taken **before**
WordPiece. That ordering matters: an out-of-vocabulary term like
`chromiumoxide` stays a first-class searchable term even though the vector
side shatters it into subword confetti. The trailing query term also matches
by prefix, which keeps results alive mid-word while you type.

Term weights reflect where a word appears. Tags are the author's own
statement of what a page is about, so they outweigh the title, which
outweighs body text (`tag_weight = 4`, `title_weight = 2` by default).

## The semantic engine

Content is split into chunks of roughly 600 characters, each embedded as the
mean of its token vectors. A query embeds the same way and scores against
chunks by cosine; a document's semantic score is its best chunk's, with a
small correction for chunk count, since a longer document gets more chances
at a high max-pooled score (`chunk_penalty`, default 0.02).

## The relevance floor

A query about nothing in your corpus should return nothing, not a
confident-looking ranking of noise. Below a minimum best-chunk cosine, a
document counts as unrelated; if neither engine is confident, the result list
is empty and the UI says so.

The floor scales with dimensionality as √(256/dim), because PCA raises the
cosines of unrelated vectors: at the default `dims = 128` the effective floor
is 0.28. This is applied at query time, so changing thresholds or upgrading
the binary can shift ranking without a rebuild, which is exactly why the
[eval harness](/tutorials/evaluate-your-search/) exists.

## Fusion

The two ranked lists merge with reciprocal rank fusion, which needs no score
calibration between engines that measure incompatible things. The trade RRF
makes is discarding score magnitude, which at small corpus sizes can produce
exact ties broken by document id; a normalised convex combination is on the
roadmap for corpora large enough to tell the difference.

Every claim on this page is inspectable: `chops-search query "anything"`
prints per-term keyword scores, best-chunk cosines, the chunk penalty, and
each engine's contribution to the fused order, using the same scoring code as
the ranker, so the explanation cannot drift from the behaviour.
