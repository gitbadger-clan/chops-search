# Changelog

Notable changes per release. The release workflow reads the section
matching the tag and puts it in the GitHub release, with the commit list
appended underneath.

## [0.2.13]

### Added
- A keyword confidence gate. The keyword engine now submits nothing to
  fusion when the matched idf mass falls below a fraction of the query's
  potential mass (`kw_floor`, default 0.30). A ranking assembled from
  stopword coincidences or a single prefix expansion is worse than no
  ranking, because RRF consumes ranks and launders away how weak the
  evidence was. The trailing word is exempt from the denominator when it
  produced expansions, so type-ahead still works mid-word.
- Identifier compounds are indexed whole alongside their parts, so
  `data-chops-open`, `prefix_rows`, and `v0.2.10` are single rare terms
  rather than decomposing into pieces that a docs corpus makes common.
  Both the builder and the query path call the same function, so emission
  is symmetric by construction.
- A corroboration gate for the semantic side, shipping inert. When the
  keyword list is empty and nothing in the corpus stands out from the
  pack, a semantic-only ranking is noise rather than an answer. Two knobs
  control it: `min_gap` (minimum top-median cosine contrast, 0 disables)
  and `strong_cos` (a top cosine at or above this is never suppressed,
  however flat the field; disables at infinity, not at 0). Both default
  to off, so ranking is unchanged unless you set them.
- `--min-gap` and `--strong-cos` on `eval` and `query`, and `--kw-floor`
  on both, for sweeping.
- `query` prints the top raw cosine and the top-median gap, and says which
  gate suppressed a list when one did.

### Changed
- `Engine::search_detailed` returns the full evidence behind a search:
  per-term keyword scoring, the gates' verdicts, pre-floor cosines, and
  the fused order with per-engine contributions. `search()` is a view of
  it, and `chops-search query` prints it. The explain path previously
  reimplemented keyword scoring, chunk scanning, and RRF arithmetic
  against the same artifacts; it had already drifted once, when BM25
  landed in core a commit earlier. It now restates nothing.
- Semantic scoring splits into `score_docs` (max-pooling, the
  measurement) and `rank_scored` (floor, penalty, sort — the judgment).
  Statistics about the whole field, like the top-median gap, read the
  measurement, so the relevance floor cannot distort them by filtering
  before they are computed.

### Fixed
- `chops-search query` printed its keyword summary line twice.
## [0.2.10]

### Fixed
- The relevance floor was calibrated at the model's native 256 dimensions
  but applied unchanged at any `dims`. PCA raises the cosines of unrelated
  vectors, so at the default `dims = 128` the floor was too permissive: a
  query about nothing in the corpus returned a confident-looking
  irrelevant result instead of nothing. It now scales as √(256/dim), which
  is 0.28 at the default. On the demo corpus this took overall recall@1
  from 88% to 92%, with the negative control going from 0% to 100% and no
  other case changing.

  This changes ranking behaviour on any site using non-native `dims`.
  Rebuilding is not required — the floor is applied at query time — but
  results will differ after upgrading the binary.

### Changed
- `Engine::score_opts()` exposes the thresholds actually in effect, so a
  caller overriding one of them starts from the derived floor rather than
  from a default that does not know the index's dimensionality.
- The `end_to_end` plumbing test opts out of scoring thresholds with
  `ScoreOpts::raw()`. Its synthetic corpus has arbitrary cosines, so
  asserting on them was testing the fixture rather than the byte path.

## [0.2.9]

### Added
- Search bar chrome in the overlay: a magnifier icon and a button that
  clears when there is text and closes when empty.
- Keyboard bindings beyond the arrow keys. `Ctrl-N` and `Ctrl-P` move
  through results, `Ctrl-[` acts as Escape. These are the terminal and
  readline equivalents, for whom they are muscle memory.
- `[data-chops-clear]` for inline mode, so a page or theme can supply its
  own clear button and have it wired up. An empty one gets the icon
  injected.
- `[data-chops-open]` now responds to Enter and Space as well as click.
  Themes routinely use `<div role="button">`, which gets no free keyboard
  activation, so the trigger was mouse-only.

### Fixed
- Closing the overlay left the query text in the input while dropping the
  results, so reopening showed a query with nothing under it. Closing now
  clears the input, the results, and any in-flight query.
- `init` scaffolded a search page with no sort key. Zola silently drops
  such pages from a sorted section, so the page 404'd with only a build
  warning to explain it.
- A duplicate click listener on `[data-chops-open]` referred to
  `window.open`, so clicking a trigger opened a popup alongside the
  dialog.
- Arrow keys from no selection went to the first result in both
  directions; `↑` now goes to the last.

- The relevance floor was calibrated at the model's native 256 dimensions
  but applied unchanged at any `dims`. PCA raises noise cosines, so at the
  default 128 the floor was too permissive and a query about nothing in
  the corpus returned a confident-looking irrelevant result. It now scales
  as √(256/dim).

### Changed
- `cargo xtask assets --check` compares a hash of the wasm crate's
  sources, the browser sources, and the lockfile against a committed
  stamp, instead of rebuilding the wasm and diffing bytes. Byte
  comparison never held across machines: panic metadata embeds absolute
  paths, and wasm-opt versions differ. The weaker claim is the one that
  can actually be true, and CI no longer needs a wasm toolchain.

## [0.2.0]

### Added
- Dynamic shell completions via `COMPLETE=<shell> chops-search`.
  Candidates are computed at completion time, so `--kind` lists the kinds
  present in your query set rather than a list frozen when a script was
  generated.
- `chops-search completions <shell>` for a conventional static script.
- Expanded `--help`: every subcommand explains what it is for, and every
  default is stated where you would look for it.

### Changed
- `clap_complete` is pinned exactly, because `unstable-dynamic` is an
  unstable API and a break would leave published versions uninstallable.

## [0.1.0]

Initial release.
