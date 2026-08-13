+++
title = "Indexing rules"
description = "Exactly which pages get indexed, how URLs are reconstructed from Zola's conventions, and the documented gaps."
weight = 15
[taxonomies]
tags = ["indexing", "front-matter", "urls", "drafts", "zola"]
+++

The indexer reads Zola's TOML `+++` front matter properly and honours Zola's
own conventions rather than approximating them. This page is the exact
contract; `chops-search docs` shows you its output for your site.

## What gets indexed

Every `.md` file under `content`, except:

- `_index.md` files (section pages) are skipped entirely
- `draft = true` pages
- `in_search_index = false` pages (Zola's own opt-out convention)

Malformed front-matter TOML is a build error, not a silent default: Zola
itself would refuse to build such a page, so indexing it anyway would diverge
from the site.

## What gets weighted

Keyword scoring is BM25F over four fields. Tags from
`[taxonomies] tags = [...]` are the highest-weighted field (default 4): they
carry more signal per byte than anything in the body, being the author's own
statement of what a page is about. Titles weight 2, the front-matter
`description` sits at parity with body (1), and each field is normalised by
its own length before the weights apply, so a weight biases rather than
inflates. [How ranking works](/explanations/hybrid-ranking/) has the
mechanics; the weights are
[configurable](/reference/configuration/) and baked into the index at build
time.

## URL reconstruction

Each page's URL is rebuilt the way Zola would build it:

| Rule | Example |
| --- | --- |
| `path` front-matter override wins outright | `path = "/elsewhere"` → `/elsewhere/` |
| `slug` replaces the final segment | `slug = "custom"` → `/blog/custom/` |
| Page bundles collapse | `foo/index.md` → `/foo/` |
| `YYYY-MM-DD-` filename prefixes strip | `2024-06-04-my-post.md` → `/blog/my-post/` (a front-matter `date` overrides the date but not the stripping) |
| Every segment is slugified (`slugify.paths = "on"`, Zola's default) | `My Post.md` → `/my-post/`, `Café_Crème` → `/cafe-creme/` |

Numeric prefixes that aren't full dates are kept, as Zola keeps them:
`01-baseline` stays `01-baseline`.

## Documented gaps

Stated rather than guessed at:

- **`_index.md` content is invisible to search.** If your substantive prose
  lives in a section landing page (a series intro, say), it won't be
  findable. Move it into a page, or accept the gap.
- **Multilingual suffixes** (`foo.fr.md`) are not handled.
- **Per-section `path` overrides in ancestor `_index.md` files** are not
  applied to descendant URLs.
- **Non-Latin slug transliteration** diverges from Zola's: Unicode
  alphanumerics are kept as-is where Zola transliterates to ASCII.
  Latin-script content is byte-identical.

If `eval` expectations 404, run `chops-search docs` and diff its URLs against
your live site before assuming a ranking bug.
