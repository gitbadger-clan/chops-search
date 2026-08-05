+++
title = "Configuration"
description = "Every chops-search.toml key with its default, and how config discovery and path resolution work."
weight = 5
+++

Configuration lives in `chops-search.toml` at the site root. Discovery walks
up from the current directory the way cargo finds `Cargo.toml`, so commands
work from anywhere inside the site. **Paths resolve relative to the config
file**, not the working directory.

Every key is optional. These are the defaults:

{% code(title="chops-search.toml") %}
```toml
content = "content"            # the Zola content tree to index
out     = "static/search"      # where artifacts + runtime land
model   = ".chops-search/model"

dims         = 128    # PCA target; the model's native size is 256
chunk_chars  = 600    # target chunk size in characters
prefix_rows  = 2048   # rows bundled eagerly
title_weight = 2      # a title mention counts like N body mentions
tag_weight   = 4      # tags outweigh titles: they're the author's own summary
```
{% end %}

| Key | Default | Notes |
| --- | --- | --- |
| `content` | `content` | Walked recursively for `.md` files |
| `out` | `static/search` | Inside `static/` so your SSG copies it verbatim |
| `model` | `.chops-search/model` | The lockfile lives beside this directory |
| `dims` | `128` | Real PCA, not truncation; [re-eval after changing](/how-to/manage-the-model/) |
| `chunk_chars` | `600` | Smaller sharpens rare-word signal, costs more vectors |
| `prefix_rows` | `2048` | Larger trades eager payload for fewer range requests |
| `title_weight` | `2` | Keyword weight multiplier for title terms |
| `tag_weight` | `4` | Keyword weight multiplier for tag terms |

The corresponding `build` flags (`--dims`, `--chunk-chars`, `--prefix-rows`,
`--content`, `--model`, `--out`) override the file for a single run; the
scoring flags on `eval` (`--min-cos`, `--chunk-penalty`) exist for sweeping
thresholds without editing anything.
