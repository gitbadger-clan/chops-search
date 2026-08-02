# Demo site

A minimal Zola site used as chops-search's end-to-end test: CI builds a
real index here and gates on recall, which catches the class of bug unit
tests can't — a chunking or URL change that compiles fine and quietly
wrecks ranking.

## Running it

    cargo run -p chops-search --release -- model fetch   # once
    cargo run -p chops-search --release -- build
    cargo run -p chops-search --release -- eval
    zola serve                                               # then /search/

## About the content

The posts are adapted from unicow.dev and are © Artur Daschevici,
included here as test fixtures. They're deliberately uneven: `til.md` is
one page covering thirty unrelated topics, which is the worst case for
chunk-based ranking and therefore the most useful page in the corpus.
`about.md` has no date, and one post is `draft = true` — both exercise
paths that a tidy demo corpus wouldn't.

`fixtures/queries.toml` is the labeled query set: 24 cases split into
exact / paraphrase / navigational, plus one negative control that must
return nothing. `chops-search docs` lists the URLs to write `expect`
entries against.
