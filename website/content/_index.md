+++
title = "chops-search Documentation"
description = "Official chops-search documentation"
template = "index.html"
sort_by = "weight"

[extra.hero]
title = "chops-search"
tagline = "Semantic search for static sites. No server, no SaaS, no megabyte model download."
image = "assets/logo.svg"   # drop a chops-search logo at static/assets/logo.svg
[[extra.hero.actions]]
text = "Get Started"
link = "/tutorials/getting-started/"
icon = "right-arrow"
variant = "primary"
[[extra.hero.actions]]
text = "Install"
link = "/how-to/install/"
variant = "minimal"
+++

## Quick Start

{{ linkcard(title="Getting Started", href="/tutorials/getting-started/", description="From `cargo install` to a working search box on a Zola site.") }}
{{ linkcard(title="Site-wide search", href="/how-to/site-wide-search/", description="Two template lines give every page a Cmd-K overlay.") }}
{{ linkcard(title="Deploy with caching", href="/how-to/deploy-with-caching/", description="The headers file, the CSP directives, and the three gotchas.") }}
{{ linkcard(title="Evaluate your search", href="/tutorials/evaluate-your-search/", description="Turn ranking quality into a number and gate it in CI.") }}

`chops-search` is hybrid keyword + semantic search for static sites, running
entirely in the browser. The "model" is a model2vec/potion int8 lookup table
streamed over HTTP range requests, so a query costs about 0.2 KB on average
rather than the tens of megabytes a transformer would. Try it right now: the
search on this site is chops-search, indexing these docs. Press `Cmd-K` (or
`/`) and phrase a question in words the pages don't use.

{% cardgrid() %}
{% card(title="Hybrid ranking", icon="setting") %}
BM25 keyword scores fused with semantic cosine similarity. The two engines
fail differently, which is the whole point.
{% end %}
{% card(title="Range-fetched model", icon="external") %}
The embedding model is a static file. The browser fetches individual rows out
of it, a kilobyte at a time.
{% end %}
{% card(title="One tokenizer", icon="document") %}
One Rust core compiled to native and wasm, so the tokenizer that indexed your
content is bit-for-bit the one that reads queries.
{% end %}
{% card(title="Measured, not vibed", icon="add-document") %}
A labelled-query eval harness turns every ranking change into a recall number,
and `--fail-under` makes CI enforce it.
{% end %}
{% end %}

## Key Features

- **Runs entirely in the browser**: no search server, no API key, no request leaves the page except range fetches against static files.
- **Understands paraphrase**: semantic matching finds "context packing" when you searched "fitting a repo into a prompt".
- **Returns nothing for nonsense**: a relevance floor suppresses confident-looking garbage when nothing in your corpus answers the query.
- **[Zola conventions honoured](/reference/indexing-rules/)**: drafts, `in_search_index`, slug and path overrides, page bundles, date-prefixed filenames.
- **Cache-friendly by construction**: content-hashed artifacts under immutable headers; a content change touches two files and the model stays cache-hot.
- **[Designed degradation](/explanations/designed-degradation/)**: offline or range-hostile hosts degrade to keyword-only or eager loading, never to wrong results.
