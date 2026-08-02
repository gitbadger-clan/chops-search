Generated. Do not edit.

Regenerate with `cargo xtask assets`; CI runs `cargo xtask assets --check`
and fails if these differ from a fresh build. They're committed because
`src/assets.rs` embeds them with `include_bytes!`, which resolves at
compile time — without them, `cargo install --git` would require wasm-pack
and a build step before the crate compiles at all.

Sources: `web/` (worker, page script, CSS) and `wasm-pack` output for
`crates/chops-search-wasm`.
