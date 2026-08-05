+++
title = "Reindex on every push"
description = "A GitHub Actions recipe that rebuilds the search index whenever content changes, with the model cached against its lockfile and recall gated before deploy."
weight = 30
+++

The index should be built by CI, not committed: gitignore `static/search/`,
make the pipeline the single producer, and every content push reindexes
automatically. On a content-only push the steady-state cost is a few seconds.

## The shape

Three cached inputs, one build, one gate, then your existing deploy:

1. The **chops-search binary**, cached against the pinned version.
2. The **model**, cached against the committed `model.lock.json`.
3. `chops-search build` into `static/search/`, **before** your SSG build so
   the output gets copied into the deployable site.
4. `chops-search eval --fail-under` as the quality gate.

## GitHub Actions

{% code(title=".github/workflows/deploy.yml (excerpt)") %}
```yaml
env:
  CHOPS_SEARCH_VERSION: "0.2.10"

steps:
  # ---- chops-search binary, cached on the pinned version ----
  - name: Restore chops-search
    id: chops-bin
    uses: actions/cache/restore@v4
    with:
      path: ~/.cargo/bin/chops-search
      key: chops-search-${{ runner.os }}-${{ env.CHOPS_SEARCH_VERSION }}

  - name: Install chops-search
    if: steps.chops-bin.outputs.cache-hit != 'true'
    run: cargo install chops-search --version "$CHOPS_SEARCH_VERSION" --locked

  - name: Save chops-search
    if: always() && steps.chops-bin.outputs.cache-hit != 'true'
    uses: actions/cache/save@v4
    with:
      path: ~/.cargo/bin/chops-search
      key: chops-search-${{ runner.os }}-${{ env.CHOPS_SEARCH_VERSION }}

  # ---- model, cached on the committed lockfile ----
  - name: Restore model
    id: model
    uses: actions/cache/restore@v4
    with:
      path: .chops-search/model
      key: model-${{ hashFiles('.chops-search/model.lock.json') }}

  - name: Fetch model at locked revision
    if: steps.model.outputs.cache-hit != 'true'
    run: |
      REV=$(jq -r .revision .chops-search/model.lock.json)
      chops-search model fetch --revision "$REV"

  - name: Verify model against lockfile
    run: chops-search model verify

  - name: Save model
    if: always() && steps.model.outputs.cache-hit != 'true'
    uses: actions/cache/save@v4
    with:
      path: .chops-search/model
      key: model-${{ hashFiles('.chops-search/model.lock.json') }}

  # ---- index + runtime, then gate ----
  - name: Build search index
    run: chops-search build

  - name: Eval search quality
    if: hashFiles('fixtures/queries.toml') != ''
    run: chops-search eval --fail-under 0.85

  # ---- then your existing SSG build and deploy, unchanged ----
```
{% end %}

## Why each piece is shaped that way

**The restore/save split with `if: always()`.** The combined `actions/cache`
saves in a post step that never runs if the job fails first, so a red build
after the 30 MB model fetch would throw the download away. Splitting restore
and save keeps failed builds from costing the next run.

**`--revision` from the lockfile, then `verify`.** `model fetch` on its own
re-resolves upstream and rewrites the lock, which CI must never do. Passing
the locked revision keeps the build reproducible, and running `verify`
unconditionally (even on cache hit) means a corrupted or evicted-and-restored
cache can't feed the build silently wrong bytes.

**The full `build`, not `--no-runtime`.** Since `static/search/` is
gitignored, every CI run must produce the runtime too. The runtime is
embedded in the pinned binary, so runtime and artifact format can't drift.

**The gate is conditional on the fixture existing.** The workflow is
paste-able before you've written a query set, and the gate switches itself on
the moment `fixtures/queries.toml` merges. Writing one is
[worth an hour](/tutorials/evaluate-your-search/).

{% aside(kind="note", title="Previews need nothing special") %}
The index stores root-relative URLs, so results resolve against whatever
origin a preview deploy serves from. Your SSG's `--base-url` flag affects its
own HTML, not the search artifacts.
{% end %}
