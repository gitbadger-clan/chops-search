//! The wasm boundary. Rust decides, JS moves bytes.
//!
//! Everything here is a mechanical delegation to chops_core::engine —
//! no logic, so there is nothing here that can drift from the build-time
//! path. Note that wasm-bindgen copies &[u8] arguments into wasm memory
//! on the way in; that's one copy per ingest, which at ~1.3 KB per query
//! is irrelevant. If it ever isn't, the next step is exposing the row
//! buffer pointer and writing fetched chunks straight into linear memory
//! from JS — reconstructing the Uint8Array view on every write, because
//! views detach whenever wasm memory grows.

use chops_core::engine::Engine as CoreEngine;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Engine {
    inner: CoreEngine,
}

#[wasm_bindgen]
impl Engine {
    /// meta = model.meta.bin, index = index.bin. Allocates the full row
    /// matrix up front so wasm memory doesn't grow (and JS views don't
    /// detach) mid-session.
    #[wasm_bindgen(constructor)]
    pub fn new(meta: &[u8], index: &[u8]) -> Result<Engine, JsError> {
        let inner = CoreEngine::new(meta, index).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Engine { inner })
    }

    pub fn dim(&self) -> u32 {
        self.inner.dim() as u32
    }

    /// Rows duplicated into model.prefix.i8; ingest that file at offset 0.
    pub fn prefix_rows(&self) -> u32 {
        self.inner.prefix_rows()
    }

    /// Byte ranges of model.rows.i8 needed before this query can run
    /// semantically, as a flat [start, end, start, end, ...] array
    /// (half-open). Empty array = nothing to fetch; search now.
    pub fn plan(&self, query: &str) -> Box<[u32]> {
        self.inner
            .plan(query)
            .into_iter()
            .flat_map(|r| [r.start, r.end])
            .collect()
    }

    /// Hand back bytes fetched from model.rows.i8 at byte_start.
    pub fn ingest(&mut self, byte_start: u32, bytes: &[u8]) -> Result<(), JsError> {
        self.inner
            .ingest(byte_start, bytes)
            .map_err(|e| JsError::new(&e.to_string()))
    }

    /// Ranked doc ids. Never fails: degrades to keyword-only when rows
    /// are missing; check used_semantic() afterward.
    pub fn search(&mut self, query: &str, limit: u32) -> Box<[u16]> {
        self.inner.search(query, limit as usize).into_boxed_slice()
    }

    pub fn used_semantic(&self) -> bool {
        self.inner.used_semantic()
    }

    pub fn doc_url(&self, id: u16) -> Option<String> {
        self.inner.doc_url(id).map(str::to_owned)
    }

    pub fn doc_title(&self, id: u16) -> Option<String> {
        self.inner.doc_title(id).map(str::to_owned)
    }
}
