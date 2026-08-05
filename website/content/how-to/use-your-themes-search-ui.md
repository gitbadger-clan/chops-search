+++
title = "Drive your theme's search UI"
description = "Run chops-search in inline mode against your theme's existing search markup instead of the self-mounted overlay."
weight = 15
+++

If your theme already has a search dialog you'd rather keep, chops-search can
drive it instead of building its own. This is inline mode, and it activates
automatically when the page already contains the expected element ids.

## The markup contract

| Element | Requirement |
| --- | --- |
| `id="chops-input"` | The query input |
| `id="chops-results"` | Must be a `<ul>`; the script appends `<li>` elements |
| `id="chops-mode"` | A `<span>` for the status line ("keyword-only", result counts) |

When `chops-search.js` finds `#chops-input` on load, it skips mounting the
overlay and drives this markup instead.

## What becomes your responsibility

Three things were your theme's script's job, and chops-search does not touch
DOM it did not create:

- **Showing and hiding.** chops-search has no opinion about a container it
  didn't build. Your open/close logic stays yours.
- **Clearing.** Add `data-chops-clear` to a button and it will be wired up;
  the script fills an empty one with a clear icon. If you clear the input
  programmatically, note that setting `.value` fires nothing; dispatch the
  event yourself:
  ```js
  input.value = "";
  input.dispatchEvent(new Event("input"));
  ```
- **Stacking context.** If the modal markup sits inside a container with
  `transform`, `filter`, or its own `z-index`, no `z-index` on the modal can
  lift it above the page. The self-mounted overlay avoids this by being a
  direct child of `<body>`; your theme's markup may not.

{% aside(kind="tip", title="Consider restyling the overlay instead") %}
Inline mode exists for themes with search UI worth keeping, but restyling the
built-in overlay is usually less work: every colour derives from
`currentColor` and the custom properties on `.chops` are the theming surface.
Weigh the three responsibilities above against a few lines of CSS.
{% end %}
