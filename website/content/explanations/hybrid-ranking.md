+++
title = "How ranking works"
description = "BM25F over pre-WordPiece tokens, cosine over embedded chunks, confidence-weighted rank fusion, and the two gates that make empty results possible."
weight = 10
[taxonomies]
tags = ["ranking", "bm25f", "fusion", "relevance-floor", "tuning", "calibration"]
+++

Two engines, fused. They fail differently, which is the whole point: keyword
nails exact rare terms and scores zero on paraphrases; semantic does the
reverse.

## The keyword engine

BM25F over four fields (title, tags, description, body), on word-level
tokens taken **before** WordPiece. That ordering matters: an
out-of-vocabulary term like `chromiumoxide` stays a first-class searchable
term even though the vector side shatters it into subword confetti.

BM25F rather than pre-multiplied field weights: each field's term frequency
is normalised by that field's *own* average length (a term in a 5-word
title scores like a term in a 5-word field, not averaged against 2,000
words of body), the normalised frequencies combine under the field weights,
and saturation applies once to the combined value. Weights bias, they don't
inflate; the old scheme let a weighted title mention blow past k1 and
behave like keyword stuffing.

Term weights reflect where a word appears. Tags are the author's own
statement of what a page is about, so they outweigh the title, which
outweighs body text (`tag_weight = 4`, `title_weight = 2` by default). The
front-matter description is its own field at parity with body
(`desc_weight = 1`): counting it as body inflated the body length, so a
fuller description quietly discounted every other term on the page.

The trailing query term also matches by prefix, which keeps results alive
mid-word while you type, under three constraints: a minimum prefix length,
a cap on expansions taking the lowest-document-frequency matches, and a
damping factor so an expansion is worth strictly less than the fully typed
term. Expansions are competing hypotheses about what the user is typing, so
per document they score as a max, never a sum: a page containing several
completions of "cli" has not multiplied its evidence that the user means
any of them.

Two judgments sit on top of the raw scores:

- **Title-cover tier.** A query whose every typed word appears in a
  document's title is, with high probability, someone typing toward that
  page: a navigational act, not a topical one. Such documents rank ahead of
  documents whose titles don't cover the query, with BM25F ordering within
  each tier. The trailing word may be covered through an expansion, so
  mid-typing still counts.
- **Keyword evidence gate** (`kw_floor`, default 0.30). Below a minimum
  confidence ratio (the fraction of the query's idf mass the corpus
  actually matched), the keyword engine submits nothing to fusion: a
  ranking assembled from stopword coincidences or a lone prefix expansion
  is worse than no ranking.

## The semantic engine

Content is split into chunks of roughly 600 characters, each embedded as the
mean of its token vectors. A query embeds the same way and scores against
chunks by cosine; a document's semantic score is its best chunk's, with a
correction for chunk count, since the max over n chunks grows as √(2 ln n)
on sampling alone (`chunk_penalty`, compiled default 0.02, calibrated per corpus and baked into index.bin, anchored to that
extreme-value bound rather than an arbitrary length fudge).

## The relevance floor

A query about nothing in your corpus should return nothing, not a
confident-looking ranking of noise. Below a minimum best-chunk cosine, a
document counts as unrelated; if neither engine is confident, the result
list is empty and the UI says so.

The floor is calibrated at the model's native dimensionality (0.20) and
scales as √(256/dim), because PCA raises the cosines of unrelated vectors:
at `dims = 128` the derived floor is about 0.28. A `min_cos` key in
`chops-search.toml` pins it explicitly instead, and an explicit 0 turns it
off. The floor filters the semantic list only: BM25F already requires a
document to literally contain a query term, and an exact rare-term hit must
not be vetoed by vectors that never learned the word, which is the whole
reason the hybrid exists.

## The corroboration gate

The floor alone can't catch every noise ranking on a small, single-topic
corpus, where everything is somewhat similar to everything. The gate
(`min_gap`) covers the remaining case: when the keyword side contributed
nothing and no document stands out from the corpus median cosine by at
least the threshold, the flat field is noise rather than an answer, and the
result list is empty. A `strong_cos` escape hatch exempts queries whose
best document is strongly relevant on its own: a flat field with a real
winner is broad, not empty. The gate ships disarmed (`min_gap = 0`) and is
armed per corpus by calibration.

## Fusion

The two ranked lists merge with reciprocal rank fusion, which needs no score
calibration between engines that measure incompatible things. Plain RRF has
one observable failure: it treats both engines as equally credible on every
query, so an exact rare-term hit ranked first by the keyword engine can
lose to a merely-adjacent semantic first place on rank arithmetic alone.
`rrf_alpha` fixes that by scaling the keyword list's vote by
1 + alpha × keyword confidence, so only queries with real keyword evidence
get the louder vote; 0 (the default) is plain RRF.

The rank discount `rrf_k` defaults to the conventional 60, which comes from
TREC-scale runs fusing thousand-deep lists. At corpus scale that curve is
nearly flat (rank 1 vs rank 2 is 1/61 vs 1/62), so fusion degenerates
toward best-average-rank and the two knobs interact: at k = 60 the keyword
weight must reach about 2 before it can overturn a semantic first place two
ranks up. Sweep them jointly with `eval --sweep-rrf-k --sweep-rrf-alpha`.

## Where the numbers live

The field weights and the calibration (`min_gap`, `rrf_alpha`, `chunk_penalty`, `min_cos`)
are baked into `index.bin` at build time and read by the engine at
construction, so the browser, a bare `chops-search eval`, and CI all score
the same configuration from the same bytes. The `eval` and `query` flags
override per run, which is how sweeping works without a rebuild; only a
rebuild changes what visitors run. `chops-search calibrate` walks each of them against the labelled set and
reports whether the baked value sits on a plateau or next to a cliff, with
every moved case named; a candidate is nominated with its casualties, never
adopted.This is exactly why the [eval harness](/tutorials/evaluate-your-search/) exists.

Every claim on this page is inspectable: `chops-search query "anything"`
prints per-term keyword scores with per-field frequencies, best-chunk
cosines, the chunk penalty, both gates' verdicts, and each engine's
contribution to the fused order, using the same scoring code as the ranker,
so the explanation cannot drift from the behaviour.
