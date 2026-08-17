+++
title = "Manage the embedding model"
description = "Fetch, lock, and verify the model2vec model, and what to re-check when you change dims."
weight = 40
[taxonomies]
tags = ["model", "lockfile", "reproducibility", "pca", "dimensions"]
+++

The embedding model is an input to your build, and chops-search treats it the
way cargo treats dependencies: fetched explicitly, pinned in a lockfile,
verified against it.

## Fetch and lock

```sh
chops-search model fetch
```

This resolves the model repo's default branch to a concrete commit, downloads
the model files into `.chops-search/model/`, and writes
`.chops-search/model.lock.json` beside it, recording the revision and a hash
of every file. The default repo is `minishlab/potion-base-8M`; any model2vec
repo works as a positional argument
(`chops-search model fetch minishlab/potion-base-4M`), the listed default is
the tested one.

**Commit the lockfile. Ignore the directory.** `chops-search init` writes
exactly those gitignore entries. The lockfile is what makes a model fetch
reproducible on another machine or in CI:

```sh
REV=$(jq -r .revision .chops-search/model.lock.json)
chops-search model fetch --revision "$REV"
```

## Verify

```sh
chops-search model verify
```

Re-hashes what's on disk against the lockfile and fails on any mismatch. Run
it in CI even when the model came from cache: a corrupted cache should be a
red build, not a silently different index.

`fetch` is the only chops-search command that touches the network. `build`
reads the model directory and nothing else, so builds keep working when
upstream is down.

## Changing `dims`

`dims` in `chops-search.toml` sets the PCA target dimensionality. Unset, it
means the model's native size (256 for potion-base-8M); the scaffolded
config pins `dims = 128`, which halves the eager prefix and every per-query
range fetch, at some cost in recall. Leaving it unset keeps native
dimensionality, and that being spellable is the point of the default: a
compiled 128 would make "native" unsayable and turn a per-site calibration
into a silent product default.

Two things to know before touching it:

- **It's real PCA, not truncation.** Potion models are trained after
  model2vec's distillation-time PCA, so the columns aren't variance-ordered
  and naive column truncation would be silently wrong. chops-search re-runs
  PCA on the token matrix at build time.
- **The relevance floor scales with it.** The floor that lets empty results
  happen is calibrated at native dimensionality (0.20) and scaled by
  √(256/dim), so at `dims = 128` the derived floor is about 0.28. Changing
  dims changes what counts as "related enough"; dimensionality reduction is
  real information loss. A `min_cos` key in the config pins the floor
  explicitly instead, and precisely because a value calibrated at one dims
  is wrong at another, leave it unset unless `calibrate` said otherwise.

So the procedure is: change the value, `chops-search build`, then
`chops-search eval` against your labelled set before shipping. If you don't
have a query set yet, that's the [tutorial to do
first](/tutorials/evaluate-your-search/).
