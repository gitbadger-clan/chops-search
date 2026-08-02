# chops-search

Hybrid keyword + semantic search for static sites, running entirely in the
browser. No API keys and no server.

The "model" is a model2vec/potion int8 lookup table streamed over HTTP
range requests, so a query costs about 0.2 KB on average rather than the
23 MB a transformer would. The engine is one Rust core compiled twice:
natively for the build tool, to wasm for the browser. That means the
tokenizer indexing your content is the same code that tokenizes queries.

On the demo corpus (9 posts, 24 labelled queries):

| query kind | recall@1 | recall@3 |
|---|---:|---:|
| exact (`chromedp iframes`) | 100% | 100% |
| paraphrase (`how long will this project take`) | 82% | 100% |
| navigational (`about`) | 100% | 100% |
| **overall** | **88%** | **100%** |

Paraphrase is the row that matters. Those queries share no words with the
documents that answer them, so a keyword engine scores 0% there.

Based on Bart de Goede's *Client-side semantic search for your static
site*, restructured around lazy row loading (Pagefind-style) instead of
one eager multi-megabyte blob.

## Install

```fish
cargo install --git https://github.com/gitbadger-clan/chops-search --locked chops-search-cli
```

The wasm engine and browser runtime are embedded in the binary. That is
the only install step, with no wasm toolchain, no npm, and nothing to copy
by hand.

## Zola

```fish
cd my-site
chops-search init          # config, search page, .gitignore entries
chops-search model fetch   # ~30 MB, once
chops-search build         # artifacts + runtime → static/search/
zola serve                 # then visit /search/
```

`init` never overwrites. Run it again after editing and it reports what it
skipped.

### A search box on every page

`build` writes `chops-search.js`, which mounts its own dialog on any page
that doesn't already contain a search box. Two lines in your base template
(tabi: `templates/tabi/extend_head.html`) give you site-wide search opened
with `Ctrl/Cmd-K` or `/`:

```html
<link rel="stylesheet" href="{{ get_url(path='search/chops-search.css') }}">
<script defer src="{{ get_url(path='search/chops-search.js') }}"></script>
```

Any element with `data-chops-open` opens it on click. Worth wiring to your
header's search icon, since a keyboard-only entry point stays invisible to
most visitors.

Also set `build_search_index = false` in `config.toml`. Zola's elasticlunr
index is dead weight alongside this one.

### What gets indexed

Zola's own conventions, honoured rather than approximated:

- `draft = true` and `in_search_index = false` pages are skipped
- `slug` and `path` front-matter overrides shape URLs
- `YYYY-MM-DD-` filename prefixes are stripped from slugs, as Zola does
- page bundles collapse (`foo/index.md` → `/foo/`)
- tags are indexed as high-weight terms, since they carry more signal per
  byte than anything in the body

Not handled: multilingual suffixes (`foo.fr.md`) and per-section `path`
overrides in ancestor `_index.md` files.

### Configuration

`chops-search.toml` at the site root. Every key is optional, and these are
the defaults.

```toml
content = "content"
out     = "static/search"
model   = ".chops-search/model"

dims         = 128    # PCA target; the model's native size is 256
chunk_chars  = 600
prefix_rows  = 2048
title_weight = 2      # a title mention counts like N body mentions
tag_weight   = 4
```

## Hugo, Astro, Next.js, and everything else

Not yet. Worth being clear-eyed about before you plan around it.

The engine, the artifact formats, and the browser runtime are
generator-agnostic. Nothing in `chops-search-core` knows what Zola is. The
indexer is the exception: it reads TOML `+++` front matter and
reconstructs URLs using Zola's slug rules. Point it at a Hugo site with
YAML front matter and it will either error or produce URLs that 404.

The fix is to index the **rendered HTML** in the output directory instead
of the source markdown. URLs then come from each file's location, titles
from `<title>`, and body text from a configurable selector. Pagefind works
this way. It is correct by construction for every generator at once, and
it also picks up shortcodes and templating that a markdown reader cannot
see. A per-generator adapter would mean chasing every SSG's front-matter
dialect and routing rules forever.

Tracked as the next significant piece of work. If you want it for a
specific generator, an issue describing your output layout is genuinely
useful, because the selector defaults are where the guesswork is.

## How it works

Two engines, fused with Reciprocal Rank Fusion.

**Keyword.** BM25 over word-level tokens taken *before* WordPiece, so an
out-of-vocabulary term like `chromiumoxide` stays a first-class term even
though the vector side shatters it into subword confetti. The trailing
query term also matches by prefix, which keeps results alive mid-word.

**Semantic.** Chunks of roughly 600 characters embedded as the mean of
their token vectors, scored by cosine against the query. A relevance floor
filters noise, so a query about nothing in your corpus returns nothing
instead of a confident-looking ranking of garbage.

The two fail differently, which is the whole point. Keyword nails exact
rare terms and scores zero on paraphrases. Semantic does the reverse.

### Artifacts

| File | Loading |
|---|---|
| `model.meta.<hash>.bin` | eager, gzipped. Vocab + per-row scales. Must be complete, since a partial vocab tokenizes silently wrong |
| `model.prefix.<hash>.i8` | eager. Top ~2048 frequency-ordered rows, covering most queries outright |
| `index.<hash>.bin` | eager, gzipped. Chunk vectors, doc URLs and titles, keyword postings |
| `model.rows.<hash>.i8` | range-fetched per query. Full matrix, headerless raw i8; row *i* at byte *i×dim* |
| `snippets.<hash>.bin` | range-fetched after ranking. Per-chunk display text |
| `manifest.json` | the only unhashed artifact, and the only one that revalidates |

Filenames carry a build hash so everything can be served `immutable`. The
eager payload is around 800 KB at `dims = 128`. Everything else arrives a
kilobyte at a time, only when a query needs it.

The wasm is content-independent by design. tinysearch and ternlight embed
the index in the binary, which means recompiling on every content change.
Here a content rebuild touches `index.bin` and `snippets.bin`, the model
files change only when the model does, and the engine caches across every
deploy.

## Deployment

Copy [`examples/demo-site/static/_headers`](examples/demo-site/static/_headers)
to your site's `static/_headers` for Cloudflare Pages/Workers or Netlify.
Without it search still works, but you pay revalidation on artifacts that
could have been cached for a year.

On other hosts the policy is `immutable` for `model.*`, `index.*`,
`snippets.*`, and `pkg/*`, then short or revalidating for `manifest.json`
and the three runtime files.

Three things that will otherwise cost you an afternoon:

1. **`Content-Type: application/wasm`.** `instantiateStreaming` fails
   without it. Most hosts get this right, so verify only if you front
   yours with something unusual.
2. **CSP.** wasm needs `'wasm-unsafe-eval'` in `script-src`, the worker
   needs `worker-src 'self'`, and range fetches need `connect-src 'self'`.
   Missing any of them shows as "search unavailable" rather than an
   obvious error.
3. **Range requests.** Cloudflare, Netlify, and S3 honour them; some dev
   servers don't. The worker tolerates a 200-instead-of-206 by ingesting
   the whole file, so a range-hostile host degrades to eager loading
   rather than breaking. `zola serve` is one of those, so test range
   behaviour against a real preview deploy.

## Evaluating your own site

Ranking changes need a number rather than a vibe.

```fish
chops-search docs                     # indexed URLs, for writing expectations
chops-search query "some query"       # why a result ranked where it did
chops-search eval --fail-under 0.85   # recall@1 by query kind
```

`eval` reads `fixtures/queries.toml` beside your config: labelled queries
split into exact, paraphrase, and navigational kinds, plus negative
controls that must return nothing. It drives the real engine over the real
byte path (plan, range-fetch, ingest, search), so a bug in the loading
logic shows up as a recall drop instead of hiding. Copy
[the demo set](examples/demo-site/fixtures/queries.toml) as a starting
shape. Even eight cases catch a chunking regression.

`--fail-under` in CI turns ranking regressions into red builds.

## What the tests pin down

- **Tokenizer parity, per codepoint.** `tokenizer_parity.rs` sweeps around
  1,500 codepoints across eight Unicode blocks, probing each one
  standalone, between Latin letters, and between ideographs, comparing
  exact token ids against HuggingFace's tokenizer. Fixtures only prove
  what someone thought to write down. Three real divergences (CJK spacing,
  symbol-vs-punctuation, control-character deletion) survived a fixture
  suite that looked thorough. Two known gaps are listed explicitly rather
  than hidden.
- **Embedding parity with the official implementation.**
  `model2vec_parity.rs` runs fixture sentences through both chops-search
  and MinishLab's `model2vec-rs`, asserting cosine > 0.9999. model2vec-rs
  is a dev-dependency only, because it pulls the HF `tokenizers` crate,
  which is exactly what `chops-search-core` exists to avoid shipping to
  wasm.
- **No `[CLS]`/`[SEP]`, unknown words deleted rather than `[UNK]`-ed,
  accents stripped** (café → cafe). Each is a test, because each silently
  poisons the query vector when wrong.
- **`embed()` returns `None` rather than a shrunken mean** if a needed row
  is unloaded. A known token with a missing row stays distinguishable from
  an out-of-vocabulary token precisely because the vocab is always
  complete.
- **Artifacts are byte-stable** given the same content and model: sorted
  walks, sorted postings, tie-broken permutations, deterministic
  best-chunk selection.
- **The embedded runtime matches its source.** `cargo xtask assets
  --check` rebuilds the wasm and diffs it against what's committed, so the
  binary that writes artifacts always carries the runtime that reads them.

Both parity oracles need model files:

```fish
CHOPS_SEARCH_MODEL_DIR=.chops-search/model cargo test --workspace -- --ignored
```

## Not done yet (deliberately)

- **Rendered-HTML indexing.** The unlock for every non-Zola generator, as
  described above.
- **Score fusion.** RRF discards magnitude, which at small corpus sizes
  produces exact ties broken by document id. A normalised convex
  combination would fix that, at the cost of the calibration sensitivity
  RRF exists to avoid. Worth revisiting on a corpus large enough to tell
  the difference.
- **wasm SIMD dot products.** The scalar f32 loop is already
  sub-millisecond at this scale. When it matters, accumulate i32 rather
  than i8 (`i16x8_extmul_low_i8x16` plus pairwise adds).
- **Zero-copy ingest.** wasm-bindgen copies `&[u8]` once on the way in,
  which is irrelevant at about 1 KB per query. The
  pointer-into-linear-memory path is sketched in
  `chops-search-wasm/src/lib.rs`.
- **Multilingual models.** `potion-multilingual-128M` needs the
  tokenizer's Indic and Hangul gaps closed first. They're documented in
  `wordpiece.rs`.

## Licence

Dual-licensed under [Apache-2.0](LICENSE-APACHE) or [MIT](LICENSE-MIT), at
your option.

Embedding models by [MinishLab](https://github.com/MinishLab/model2vec)
(MIT). Prior art and the idea come from Bart de Goede and Ken Hawkins.

Unless you state otherwise, any contribution you intentionally submit for
inclusion in this work, as defined in the Apache-2.0 licence, shall be
dual licensed as above, without additional terms or conditions.
