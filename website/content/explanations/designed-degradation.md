+++
title = "When it breaks, it says so"
description = "The failure modes partial loading creates, and why chops-search degrades to keyword-only or eager loading instead of returning plausible garbage."
weight = 30
[taxonomies]
tags = ["degradation", "keyword-only", "offline", "csp", "troubleshooting"]
+++

Partial loading creates a failure mode most search libraries don't have: what
if a row you need isn't loaded? Offline, a strict CSP, a host that ignores
range requests. The design principle throughout is that **degraded and
honest beats complete and wrong**.

## The wrong answer, and the right one

The tempting implementation averages the rows you have. That produces a
shrunken, wrong embedding that returns plausible-looking garbage, which is
worse than returning nothing, because nobody notices. So `embed()` returns
nothing instead, search degrades to keyword-only, and it **reports that it
did**: the status line tells the user semantic matching is off rather than
letting search silently get dumber.

## The degradation ladder

| Condition | Behaviour |
| --- | --- |
| Row fetch fails mid-query | Keyword-only for that query, reported in the status line |
| Host ignores range requests (200 instead of 206) | The whole file is ingested once; eager loading, slower but correct |
| Wasm blocked (CSP without `'wasm-unsafe-eval'`, fetch failure) | Search reports unavailable rather than rendering a dead input |
| Stale page script with fresh artifacts | "Search unavailable", never wrong results; the runtime cache policy bounds this to minutes |

The 200-tolerance is why `zola serve` works at all: dev servers commonly
don't implement ranges, and a range-hostile host should cost bandwidth, not
correctness. It's also why you should test range behaviour against a real
deploy; locally you're always on the eager path.

## Trust, but verify the trusted parts

The same suspicion applies to inputs the browser can't control. A
`Content-Range` header is checked against what was asked for rather than
believed outright, and artifacts are addressed by content hash from a
revalidating manifest, so a half-updated CDN can't pair a new index with old
rows.

Degradation paths that are designed rather than discovered in an issue
tracker have a second benefit: they're testable. The
[eval harness](/tutorials/evaluate-your-search/) drives the real plan,
range-fetch, ingest, and search path, so a bug in the loading logic shows up
as a recall drop instead of hiding behind a fallback.
