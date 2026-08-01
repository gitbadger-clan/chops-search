# chops-search

Hybrid (keyword + semantic) search for a static site, running entirely in
the browser. The "model" is a model2vec/potion int8 lookup table streamed
via HTTP range requests; the engine is one Rust core compiled twice —
natively for the build tool, to wasm for the browser. Based on Bart de
Goede's "Client-side semantic search for your static site", restructured
around lazy row loading (Pagefind-style) instead of one eager 4 MB blob.

## Layout

```
crates/chops-search-core    pure engine: WordPiece, int8 row store, scoring,
                      RRF, artifact formats. No I/O. Compiles to both
                      targets unchanged — the single-tokenizer guarantee
                      is structural, not aspirational.
crates/chops-search-cli     `chops-search build`: model dir + content dir → artifacts
crates/chops-search-wasm    wasm-bindgen wrapper: plan / ingest / search
web/                  worker (byte pump) + page glue
```

## Artifacts and their loading discipline

| File              | Size (potion-base-8M, dims 256)* | Loading |
|-------------------|----------------------------------|---------|
| `model.meta.bin`  | vocab + scales, ~500 KB, gzips hard | eager, must be COMPLETE (partial vocab ⇒ silently wrong tokenization) |
| `model.prefix.i8` | top ~2048 frequency-ordered rows | eager |
| `model.rows.i8`   | full matrix, headerless raw i8   | range-fetched per query; row i at byte i×dim |
| `index.bin`       | chunk vectors + docs + keyword postings | eager |

\* Native potion-base-8M is 256-dim (~7.4 MB matrix). Pass `--dims 128` to
match the article's ~3.8 MB: chops-search re-runs PCA on the token matrix at
build time (naive column truncation would be wrong — potion models are
trained after model2vec's distillation-time PCA, so stored coordinates
are no longer variance-ordered). Note the streaming design makes this
less critical than in the article's eager-load setup: users download the
meta + prefix + index eagerly, and the matrix only ever arrives a
kilobyte at a time — the main effect of `--dims` is halving the prefix
and per-query range sizes.

## Build

```fish
# once: fetch the model (the build tool is deliberately offline).
# config.json is only needed by the model2vec-rs parity test.
huggingface-cli download minishlab/potion-base-8M \
    tokenizer.json model.safetensors config.json --local-dir model/

cargo run -p chops-search-cli --release -- build \
    --content ../site/content \
    --model model/ \
    --out ../site/static/search \
    --prefix-rows 2048 \
    --dims 128   # optional; omit to keep the model's native dims

# wasm blob (once per engine change, NOT per content change)
cargo install wasm-pack   # or wasm-bindgen-cli + manual
wasm-pack build crates/chops-search-wasm --target web --release \
    --out-dir ../../site/static/search/pkg
# then: wasm-opt -Oz on the emitted .wasm if wasm-pack didn't already

cp web/search-worker.js ../site/static/search/
cp web/search.js        ../site/static/js/
```

The engine wasm is content-independent by design — the index is fetched,
never embedded. tinysearch/ternlight embed the index in the binary, which
means recompiling (or WASM byte-patching) on every content change; here a
content rebuild touches only `index.bin`, `model.*` never change until the
model does, and the wasm caches across every deploy.

## Deployment (Cloudflare Pages)

`static/_headers`:

```
/search/model.rows.i8
  Cache-Control: public, max-age=31536000, immutable
/search/model.prefix.i8
  Cache-Control: public, max-age=31536000, immutable
/search/model.meta.bin
  Cache-Control: public, max-age=31536000, immutable
/search/pkg/*
  Cache-Control: public, max-age=31536000, immutable
/search/index.bin
  Cache-Control: public, max-age=0, must-revalidate
```

The model files are content-addressed in spirit; if you ever swap models,
rename them (hash suffix) rather than relying on revalidation.

Three gotchas that will otherwise eat an afternoon:

1. **`Content-Type: application/wasm`** — `instantiateStreaming` silently
   falls back (or fails) without it. Pages gets this right by default;
   verify if you front it with anything.
2. **CSP** — if you set a Content-Security-Policy, WebAssembly needs the
   `wasm-unsafe-eval` (or `unsafe-eval`) script-src directive or it is
   blocked from loading at all.
3. **Range requests** — Pages/Netlify/S3 honor them; some dev servers
   don't. The worker tolerates a 200-instead-of-206 by ingesting the whole
   file, so a range-hostile host degrades to "eager everything", not
   breakage. `zola serve` proxies its own static handler — test range
   behavior against a production preview, not just localhost.

## CI (GitHub Actions sketch)

```yaml
- uses: dtolnay/rust-toolchain@stable
  with: { targets: wasm32-unknown-unknown }
- uses: actions/cache@v4
  with: { path: model/, key: potion-base-8M-v1 }
- run: cargo test --workspace
- run: cargo run -p chops-search-cli --release -- build
       --content content --model model --out static/search
# wasm build only when crates/ changed, or unconditionally if you prefer
```

## Invariants the tests pin down

- **Parity with the official implementation.** `crates/chops-search-cli/tests/
  model2vec_parity.rs` pushes fixture sentences through chops-search-core's
  tokenizer + embedding AND MinishLab's official `model2vec-rs` crate,
  asserting cosine > 0.9999 per input (plus agreement on which inputs
  embed to nothing). model2vec-rs is a dev-dependency only — it pulls the
  HF `tokenizers` crate, which is exactly what chops-search-core exists to avoid
  shipping to wasm. The browser must run our tokenizer, so build time
  runs it too; the oracle proves ours matches theirs. Needs model files:

  ```fish
  CHOPS_SEARCH_MODEL_DIR=model cargo test -p chops-search-cli -- --ignored
  ```

- No `[CLS]`/`[SEP]`; unknown words are **deleted**, never `[UNK]`-ed;
  accents stripped (café → cafe). Each is a test in `wordpiece.rs`.
- `embed()` returns `None` — never a shrunken mean — if any needed row is
  unloaded. A known token with a missing row is distinguishable from an
  out-of-vocabulary token precisely because the vocab is always complete.
- Keyword-only results are valid RRF (single-list); the semantic list is
  purely additive when it arrives. The worker's `gen` guard applies it on
  the next keystroke instead of reordering under the cursor.
- Artifacts are byte-stable given the same content + model: sorted
  directory walks, sorted postings, tie-broken permutations.

## Not done yet (deliberately)

- **wasm SIMD dot products** — the f32 scalar loop is already ~µs at this
  scale; revisit if the corpus grows 100×. When you do: accumulate i32,
  never i8 (`i16x8_extmul_low_i8x16` + pairwise adds), and ship
  feature-detected dual builds or accept the Safari 16.4 floor.
- **Zero-copy ingest** — wasm-bindgen copies `&[u8]` once on the way in;
  fine at 1.3 KB/query. The pointer-into-linear-memory path is sketched in
  `chops-search-wasm/src/lib.rs` comments if it ever matters.
- **Snippets in results** — `index.bin` has room; add a `str16` per chunk
  and a best-chunk id per result.
- **Eval harness** — port the thirty labeled queries + recall@1 idea
  before trusting any ranking change. The metric lesson from the article
  stands: recall@3 on a small corpus cannot see anything.
