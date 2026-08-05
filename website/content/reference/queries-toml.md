+++
title = "queries.toml"
description = "The labelled query set format that eval reads: fields, kinds, and how expectations are scored."
weight = 20
+++

`chops-search eval` reads `fixtures/queries.toml` beside your
`chops-search.toml` (override with `--queries <FILE>`).

## Format

{% code(title="fixtures/queries.toml") %}
```toml
[[query]]
q = "reciprocal rank fusion"
expect = ["/blog/how-search-works/", "/projects/search/"]
kind = "exact"

[[query]]
q = "sourdough starter hydration"
expect = []
kind = "negative"
```
{% end %}

| Field | Meaning |
| --- | --- |
| `q` | The query string, exactly as a user would type it |
| `expect` | URLs, **any** of which counts as a correct top-1. Empty means the query must return no results |
| `kind` | A free-form label; recall is reported per kind |

URLs must match what `chops-search docs` prints, trailing slash included.

## Kinds

Kinds are labels, not behaviour: the engine treats every case identically,
and the per-kind breakdown exists so you can see *which* capability
regressed. The conventional set:

| Kind | Tests |
| --- | --- |
| `exact` | Rare literal terms; the keyword engine must carry it |
| `paraphrase` | No useful shared words; the semantic engine must carry it |
| `navigational` | Typing toward a known page |
| `negative` | Nothing answers this; the relevance floor must return empty |

`eval --kind paraphrase` runs one kind in isolation, and with dynamic
[shell completion](/how-to/shell-completion/) enabled, `--kind <TAB>` lists
whatever kinds your file actually uses.

## Scoring

A case passes at recall@1 if any expected URL ranks first (for `expect = []`,
if the result list is empty). recall@3 is reported alongside, which is the
number to watch for near-duplicate corpora where two pages legitimately
compete for the same queries. `--fail-under <FRACTION>` gates on overall
recall@1.

## Writing a good set

Small and honest beats large and aspirational: even eight cases catch a
chunking regression. List every legitimately correct URL in `expect` rather
than forcing arbitrary winners, keep negative controls free of any corpus
word so BM25 can't legitimately match them, and when a case fails, run
`chops-search query` before editing anything; half of all "failures" are
fixture bugs. The [evaluation tutorial](/tutorials/evaluate-your-search/)
walks the full loop.
