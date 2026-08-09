+++
title = "Add search to every page"
description = "Two template lines mount a Cmd-K search overlay on your whole site, plus how to wire your header's search icon to open it."
weight = 10
[taxonomies]
tags = ["templates", "overlay", "keyboard", "shortcuts", "zola"]
+++

`build` writes `chops-search.js`, which mounts its own search dialog on any
page that doesn't already contain a search box. Two lines in your base
template give you site-wide search.

## The two lines

Add to your base template's `<head>` (tabi: `templates/tabi/extend_head.html`):

{% code(title="templates/tabi/extend_head.html") %}
```html
<link rel="stylesheet" href="/search/chops-search.css">
<script defer src="/search/chops-search.js"></script>
```
{% end %}

On pages with no `#chops-input`, the script mounts an overlay dialog as a
direct child of `<body>`, opened with `Ctrl/Cmd-K` or `/`.

## Give it a visible entry point

A keyboard-only entry point stays invisible to most visitors, so wire the
overlay to something clickable. Any element with `data-chops-open` opens it on
click, Enter, or Space:

```html
<button class="search-button" data-chops-open aria-label="Search">…</button>
```

Themes routinely use `<div role="button">` for header icons; that works too,
since the trigger responds to keyboard activation, not just click.

## Keyboard

| Key | Action |
| --- | --- |
| `Ctrl/Cmd-K` or `/` | Open the search dialog |
| `↓` `↑` or `Ctrl-N` `Ctrl-P` | Move through results |
| `Enter` | Open the selected result |
| `Esc` or `Ctrl-[` | Clear the query; close the dialog if already empty |

`/` only opens search when you're not already typing in a field, so it stays
usable as a character everywhere else. Escape clears before it closes,
deliberately: one keystroke that both wiped a long query and dismissed the
dialog would be a keystroke too destructive. Press it twice to do both.

{% aside(kind="caution", title="Content Security Policy") %}
If your site sets a CSP (tabi's `enable_csp = true`, a `_headers` rule, or a
meta tag), three directives are required: `'wasm-unsafe-eval'` in
`script-src`, `'self'` in `worker-src`, and `'self'` in `connect-src`.
Missing any of them shows as "search unavailable" rather than an obvious
error. [Deploy with caching](/how-to/deploy-with-caching/) covers this in
detail.
{% end %}

## Styling

The overlay is unopinionated by design: every colour derives from
`currentColor`, and the custom properties on `.chops` are the theming
surface. Restyling it is usually less work than adopting your theme's own
modal, but if you'd rather keep your theme's dialog, see
[Drive your theme's search UI](/how-to/use-your-themes-search-ui/).
