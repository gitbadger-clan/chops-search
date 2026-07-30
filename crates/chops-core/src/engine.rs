//! The plan / ingest / search surface. chops-wasm wraps this 1:1;
//! integration tests drive it natively with the exact bytes the browser
//! would see.
//!
//! Contract with the JS pump:
//!   1. construct with meta + index bytes (vocab must be complete)
//!   2. ingest(0, prefix_bytes) for the eager prefix
//!   3. per query: plan() → range-fetch → ingest() each → search()
//!
//! search() never fails: if any needed row is still unloaded (or the fetch
//! never happened — offline, CSP, whatever), it degrades to keyword-only
//! and reports that via used_semantic(). Degrading loudly-but-gracefully
//! beats a quietly shrunken mean.

use crate::format::{Index, ModelMeta};
use crate::keyword::KeywordIndex;
use crate::plan::{coalesce, ByteRange};
use crate::store::RowStore;
use crate::wordpiece::Vocab;
use crate::{rrf, score, FormatError, StoreError};

/// Gap (in rows) below which two needed rows share one range request.
const MAX_GAP_ROWS: u32 = 8;

pub struct Engine {
    vocab: Vocab,
    store: RowStore,
    index: Index,
    kw: KeywordIndex,
    dim: usize,
    prefix_rows: u32,
    used_semantic: bool,
    opts: crate::score::ScoreOpts,
}

impl Engine {
    pub fn new(meta_bytes: &[u8], index_bytes: &[u8]) -> Result<Self, FormatError> {
        let meta = ModelMeta::read(meta_bytes)?;
        let index = Index::read(index_bytes)?;
        if index.dim != meta.dim {
            return Err(FormatError::Inconsistent("index dim != model dim"));
        }
        let dim = meta.dim as usize;
        let vocab = Vocab::from_tokens(&meta.tokens);
        // Full matrix buffer reserved here, once, before anything else in
        // the session allocates — wasm memory never grows because of us.
        let store = RowStore::new(dim, meta.tokens.len(), meta.scales.clone());
        let kw = index.keyword_index();
        Ok(Engine {
            vocab,
            store,
            index,
            kw,
            dim,
            prefix_rows: meta.prefix_rows,
            used_semantic: false,
            opts: crate::score::ScoreOpts::default(),
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn prefix_rows(&self) -> u32 {
        self.prefix_rows
    }

    pub fn n_rows(&self) -> usize {
        self.store.n_rows()
    }

    pub fn doc_count(&self) -> usize {
        self.index.docs.len()
    }

    pub fn doc_url(&self, id: u16) -> Option<&str> {
        self.index.docs.get(id as usize).map(|d| d.url.as_str())
    }

    pub fn doc_title(&self, id: u16) -> Option<&str> {
        self.index.docs.get(id as usize).map(|d| d.title.as_str())
    }

    /// Byte ranges of `model.rows.i8` needed before this query can be
    /// answered semantically. Empty when everything is already loaded —
    /// including the all-out-of-vocabulary case, where no rows exist to
    /// fetch and search() will correctly go keyword-only.
    pub fn plan(&self, query: &str) -> Vec<ByteRange> {
        let ids = self.vocab.tokenize(query);
        let missing = self.store.missing(&ids);
        coalesce(&missing, self.dim as u32, MAX_GAP_ROWS)
    }

    /// Feed back bytes fetched from `model.rows.i8` at `byte_start`
    /// (the prefix file is byte_start 0).
    pub fn ingest(&mut self, byte_start: u32, bytes: &[u8]) -> Result<(), StoreError> {
        self.store.ingest(byte_start as usize, bytes)
    }

    /// Override scoring thresholds (eval sweeps these; the browser uses
    /// the defaults).
    pub fn set_score_opts(&mut self, opts: crate::score::ScoreOpts) {
        self.opts = opts;
    }

    /// Hybrid search: keyword tf-idf and semantic ranked lists fused with
    /// RRF. Returns ranked doc ids, truncated to `limit`.
    pub fn search(&mut self, query: &str, limit: usize) -> Vec<u16> {
        // Keyword side works on word-level tokens (pre-WordPiece) so
        // out-of-vocabulary terms are first-class here.
        let norm = Vocab::normalize(query);
        let words: Vec<&str> = Vocab::words(&norm)
            .into_iter()
            .filter(|w| w.chars().any(|c| c.is_alphanumeric()))
            .collect();
        let kw_ranked = self.kw.rank(&words);

        let ids = self.vocab.tokenize(query);
        let fused = match self.store.embed(&ids) {
            Some(q) => {
                self.used_semantic = true;
                let sem_ranked = score::rank_docs(
                    &q,
                    &self.index.chunk_vecs,
                    self.dim,
                    self.index.global_scale,
                    &self.index.chunk_doc,
                    self.index.docs.len(),
                    self.opts,
                );
                rrf::fuse(&[&kw_ranked, &sem_ranked], rrf::K)
            }
            None => {
                self.used_semantic = false;
                kw_ranked
            }
        };
        fused.into_iter().take(limit).collect()
    }

    /// Whether the last search() actually used the vector side. False
    /// means keyword-only: all-OOV query, or rows not yet loaded.
    pub fn used_semantic(&self) -> bool {
        self.used_semantic
    }
}
