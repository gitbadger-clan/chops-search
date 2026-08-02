# Changelog

## [0.2.0]

### Added
- Dynamic shell completions via `COMPLETE=<shell> chops-search`. Candidates
  are computed at completion time, so `--kind` lists the kinds present in
  your query set rather than a frozen list.
- `chops-search completions <shell>` for a conventional static script.
- Expanded `--help`: every subcommand explains what it is for, and every
  default is stated where you would look for it.

### Changed
- `clap_complete` is pinned exactly, because `unstable-dynamic` is an
  unstable API and a break would leave published versions uninstallable.

## [0.1.0]

Initial release.
