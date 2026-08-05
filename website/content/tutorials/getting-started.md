+++
title = "Getting Started"
description = "Install chops-search, scaffold a Zola site's integration, fetch the model, build the index, and see results locally."
weight = 1
+++

{% aside(kind="note", title="Tutorial Overview") %}
This tutorial takes you from nothing to a working search box on a Zola site:
install, scaffold, model, build, browse. About ten minutes, most of it a
one-time 30 MB model download.
{% end %}

## Prerequisites

A Zola site with a `content/` directory, and Rust with cargo installed. No
wasm toolchain and no npm: the browser runtime ships embedded in the binary.

## 1. Install

{% code(title="bash") %}
```bash
cargo install chops-search --locked
```
{% end %}

See the [installation guide](/how-to/install/) for the bleeding-edge git build.

## 2. Scaffold

From your site root:

```sh
chops-search init
```

This writes three things, and never overwrites anything that exists:

| Created | Purpose |
| --- | --- |
| `chops-search.toml` | Configuration, with every default stated in comments |
| `content/search.md` | A `/search/` page wired to the runtime |
| `.gitignore` entries | Ignores the model directory and the generated output |

Re-running `init` after you've edited the scaffold is safe: it reports what it
skipped. Sites that only want the site-wide overlay can pass `--no-page` to
skip the dedicated search page.

{% aside(kind="tip", title="Why the search page has a date") %}
Zola silently drops pages without a sort key from a sorted section, so a
search page without one renders nowhere and 404s with only a build warning to
explain it. The scaffolded page carries both `date` and `weight` for this
reason. Leave them in.
{% end %}

## 3. Fetch the model

```sh
chops-search model fetch
```

This downloads the embedding model (about 30 MB, once) and records exactly
what landed in `.chops-search/model.lock.json`. **Commit the lockfile**; the
model directory itself is already gitignored. This is the only chops-search
command that ever touches the network: `build` reads a directory and nothing
else, so a build can never fail because an upstream repo moved.

## 4. Build

```sh
chops-search build
```

The content tree is walked, chunked, and embedded, and everything lands in
`static/search/`: content-hashed index artifacts plus the wasm engine, worker,
page script, and stylesheet. Zola copies `static/` verbatim, so the next
`zola build` ships all of it.

Also set this in your `config.toml`:

```toml
build_search_index = false
```

Zola's own elasticlunr index is dead weight alongside this one.

## 5. See it

```sh
zola serve
```

Visit `/search/` and type. Two things to know about local preview:

- Results appear as you type, with the trailing word matched by prefix.
- `zola serve` ignores HTTP range requests, so the engine falls back to eager
  loading. That's a [designed degradation](/explanations/designed-degradation/),
  not a bug, but it means you should test range behaviour against a real
  deploy, not the dev server.

## Next steps

Add a search box to [every page](/how-to/site-wide-search/), set up
[deployment caching](/how-to/deploy-with-caching/), and then, the part that
separates search you trust from search you hope works, [measure
it](/tutorials/evaluate-your-search/).
