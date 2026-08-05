+++
title = "Evaluate Your Search"
description = "Write a labelled query set for your own corpus, measure recall, diagnose misses with the query command, and gate regressions in CI."
weight = 2
+++

{% aside(kind="note", title="Tutorial Overview") %}
Ranking changes need a number rather than a vibe. This tutorial builds a
labelled query set for your site, runs it, reads the output, and turns it into
a CI gate. Even eight cases catch a chunking regression.
{% end %}

## 1. List what's actually indexed

```sh
chops-search docs
```

This prints every indexed document with its URL and chunk count. The URLs are
what your expectations must match, so start here: a mistyped expectation reads
as a ranking failure rather than a typo, which is a bad afternoon. If a page
you expected is missing, check the [indexing rules](/reference/indexing-rules/)
before blaming the ranker.

## 2. Write the query set

Create `fixtures/queries.toml` beside your `chops-search.toml`. Four kinds of
case, each testing a different failure mode:

{% code(title="fixtures/queries.toml") %}
```toml
# Rare literal terms. The keyword engine must carry this.
[[query]]
q = "acks_late visibility timeout"
expect = ["/labs/celery-task-loss/"]
kind = "exact"

# Shares no useful words with the answer. Semantic must carry it.
[[query]]
q = "what happens to queued jobs when the process dies"
expect = ["/labs/celery-task-loss/"]
kind = "paraphrase"

# Typing toward a known page.
[[query]]
q = "about"
expect = ["/about/"]
kind = "navigational"

# Nothing answers this. The engine must return empty, not noise.
[[query]]
q = "sourdough starter hydration"
expect = []
kind = "negative"
```
{% end %}

`expect` lists URLs, any of which counts as a correct top-1. When two pages
legitimately answer the same query (a project page and the blog post about
it), list both rather than forcing an arbitrary winner. The
[queries.toml reference](/reference/queries-toml/) has the full format.

{% aside(kind="tip", title="Choosing negative controls") %}
Pick queries with zero corpus words. BM25 will happily latch onto an innocent
shared word like "plan" and surface a keyword hit, which is correct behaviour
ruining a bad test case, not a bug.
{% end %}

## 3. Run it

```sh
chops-search build   # eval scores the artifacts on disk, so build first
chops-search eval
```

You get a PASS/FAIL line per case, recall@1 and recall@3 by kind, real
bytes-per-query numbers, and for each failure the top 3 it returned instead.
The eval drives the actual engine over the actual byte path (plan,
range-fetch, ingest, search), so a bug in the loading logic shows up as a
recall drop instead of hiding.

## 4. Diagnose a miss

For any failing case:

```sh
chops-search query "the failing query"
```

This prints the evidence behind the ranking: how the query tokenized on both
sides, per-term keyword scores with document frequencies, best-chunk cosine
per document, and each engine's contribution to the fused order. When a result
looks wrong the answer is almost always visible in the chunk count or the term
frequencies. Three patterns account for most misses:

- **The expectation is wrong.** The "wrong" winner genuinely answers the
  query. Fix the fixture, not the engine.
- **The page is content-thin.** Two chunks against a rival's thirty means the
  semantic side barely has surface to match. Fix the content.
- **A stopword dragged in a rival.** Look for a discriminating word that
  matches no documents while a common one scores. That's a ranking issue
  worth filing.

## 5. Gate it in CI

```sh
chops-search eval --fail-under 0.85
```

Exit is non-zero below the threshold. Set it just under your measured
baseline: high enough to catch a regression, low enough that one flipped case
in a small set doesn't fail the build. Then wire it into your
[deploy workflow](/how-to/reindex-in-ci/) so a new post that quietly wrecks
ranking becomes a red build instead of a discovery three weeks later.

{% aside(kind="caution", title="Upgrades can shift ranking without a rebuild") %}
Scoring thresholds are applied at query time, so upgrading the binary can
change live ranking with no content change. The eval gate is what turns that
from a surprise into a checked assertion: re-run it before trusting a new
version, and when sweeping thresholds, `eval --min-cos` and
`--chunk-penalty` let you try values without editing anything.
{% end %}
