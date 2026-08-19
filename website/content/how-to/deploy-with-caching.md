+++
title = "Deploy with correct caching"
description = "The _headers file for Cloudflare and Netlify, the CSP directives wasm needs, and the three deployment gotchas that cost afternoons."
weight = 20
[taxonomies]
tags = ["deployment", "caching", "csp", "headers", "cloudflare", "netlify"]
+++

Search works on any static host with no configuration at all. This guide is
about not leaving performance on the table: without cache headers you pay
revalidation on artifacts that could have been cached for a year, and with a
strict CSP you can silently lose the semantic half of the engine.

## The headers file

Everything chops-search emits under `/search/` carries a content hash
**except** `manifest.json` and the three runtime files. Copy this to
`static/_headers` (Cloudflare Workers static assets and Pages read it
natively; so does Netlify):

{% code(title="static/_headers") %}
```text
/search/model.*
  Cache-Control: public, max-age=31536000, immutable

/search/index.*
  Cache-Control: public, max-age=31536000, immutable

/search/snippets.*
  Cache-Control: public, max-age=31536000, immutable

/search/pkg/*
  Cache-Control: public, max-age=31536000, immutable

# The manifest names every hashed file, so it must never go stale.
/search/manifest.json
  Cache-Control: public, max-age=0, must-revalidate

# Unhashed runtime, loaded on every page. Five minutes bounds upgrade
# staleness without a conditional request per navigation.
/search/chops-search.js
  Cache-Control: public, max-age=300

/search/chops-search.css
  Cache-Control: public, max-age=300

/search/search-worker.js
  Cache-Control: public, max-age=300
```
{% end %}

The globs also cover the `.gz` siblings the build writes next to the eager
artifacts. Those exist because most hosts only compress a fixed list of
content types that doesn't include `application/octet-stream`; the worker
fetches the `.gz` directly and decompresses in the browser, so the eager
payload ships small without any host-side compression rule.

The wasm under `/search/pkg/` is unhashed on disk but always requested with
`?v=<build hash>` by the worker, so pinning it is safe: a rebuild changes the
query string and therefore the browser's cache key. And a stale page script
paired with fresh artifacts degrades to "search unavailable", never to wrong
results, which is why five minutes of runtime staleness is acceptable.

Verify after deploying:

```sh
curl -sI https://your.site/search/manifest.json | grep -i cache-control
```

## The three afternoon-eaters

1. **`Content-Type: application/wasm`.** Streaming wasm instantiation fails
   without it. Most hosts get this right; verify only if you front yours with
   something unusual.
2. **CSP.** Wasm compilation needs `'wasm-unsafe-eval'` in `script-src`, the
   worker needs `worker-src 'self'`, and range fetches need
   `connect-src 'self'`. Missing any of them shows as "search unavailable"
   rather than an obvious error. With tabi's `enable_csp = true`:
   ```toml
   allowed_domains = [
       { directive = "script-src",  domains = ["'self'", "'wasm-unsafe-eval'"] },
       { directive = "worker-src",  domains = ["'self'"] },
       { directive = "connect-src", domains = ["'self'"] },
   ]
   ```
   If a directive is already defined for another purpose (an analytics
   endpoint in `connect-src`, say), make sure `'self'` is in its list; a
   defined directive that omits it silently blocks the range fetches.
3. **Range requests.** Cloudflare, Netlify, and S3 honour them; some dev
   servers don't (`zola serve` included). The worker tolerates a
   200-instead-of-206 by ingesting the whole file once, so a range-hostile
   host degrades to eager loading rather than breaking; it also remembers the
   hostility for the session, so later queries skip ranges instead of paying
   the fallback again. Test range behaviour against a real preview deploy,
   and read the network tab: per-query requests should be partial responses
   of a kilobyte or less.

   `chops-search plan "some query" --curl https://your.site/search | sh`
   runs the exact range requests the browser would make and prints each status;
   every line should be `206`. A `200` with a multi-megabyte count is the range-hostile case.

{% aside(kind="tip", title="What a healthy deploy looks like") %}
First visit: the eager artifacts and wasm load once. Every query after that:
either no network at all (prefix hit or warm row cache) or one or two range
requests totalling a few hundred bytes. If you see the full row matrix downloading per query,
range requests are being rejected somewhere in front of your files.
{% end %}
