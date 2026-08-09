+++
title = "Enable shell completion"
description = "Turn on dynamic tab-completion for chops-search in fish, zsh, or bash."
weight = 50
[taxonomies]
tags = ["shell", "completion", "fish", "zsh", "bash"]
+++

chops-search ships dynamic completion: the binary computes candidates when you
press Tab, so there's no generated script to go stale. `eval --kind <TAB>`
lists the kinds your query set actually uses, not a list frozen when a script
was written.

## Turn it on

{% tabs() %}
=== fish
```fish
echo 'COMPLETE=fish chops-search | source' >> ~/.config/fish/config.fish
```
=== zsh
```zsh
echo 'source <(COMPLETE=zsh chops-search)' >> ~/.zshrc
```
=== bash
```bash
echo 'source <(COMPLETE=bash chops-search)' >> ~/.bashrc
```
{% end %}

Open a new shell and try `chops-search eval --kind <TAB>`.

## Static script fallback

For environments where the dynamic hook doesn't fit (system packages,
restricted shells):

```sh
chops-search completions fish   # or zsh, bash, and friends
```

writes a conventional completion script to stdout. Prefer the dynamic path
where your shell supports it; the static script's candidates are frozen at
generation time.

{% aside(kind="note", title="After upgrading") %}
The dynamic hook re-invokes the installed binary per completion request, so
it always matches what's installed, but re-source it (open a new shell) after
upgrading in case the completion protocol changed between releases.
{% end %}
