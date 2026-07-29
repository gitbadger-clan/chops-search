//! chops-core: the entire search engine as pure logic.
//!
//! No filesystem, no network, no async. Compiles unchanged for the native
//! build tool (chops-cli) and the browser blob (chops-wasm). Every
//! decision that could silently produce a wrong vector — tokenization,
//! quantization, unloaded-row handling — lives here, once.

pub mod bytes;
pub mod wordpiece;
pub mod store;
pub mod plan;
pub mod score;
pub mod rrf;
pub mod keyword;
pub mod format;
pub mod chunk;
pub mod builder;
pub mod engine;

/// Errors while parsing the binary artifacts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormatError {
    /// Buffer ended before the declared content did.
    Truncated,
    /// Magic bytes or version didn't match.
    BadHeader,
    /// A length or count field is inconsistent with the rest of the file.
    Inconsistent(&'static str),
    /// A string field wasn't valid UTF-8.
    BadUtf8,
}

impl core::fmt::Display for FormatError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            FormatError::Truncated => write!(f, "artifact truncated"),
            FormatError::BadHeader => write!(f, "bad magic or unsupported version"),
            FormatError::Inconsistent(what) => write!(f, "inconsistent artifact: {what}"),
            FormatError::BadUtf8 => write!(f, "invalid utf-8 in artifact string"),
        }
    }
}

impl std::error::Error for FormatError {}

/// Errors while ingesting matrix bytes into the row store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// Byte offset not aligned to a row boundary.
    Unaligned,
    /// Ingest would write past the end of the matrix.
    OutOfBounds,
    /// Payload length is not a whole number of rows.
    PartialRow,
}

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            StoreError::Unaligned => write!(f, "ingest offset not row-aligned"),
            StoreError::OutOfBounds => write!(f, "ingest past end of matrix"),
            StoreError::PartialRow => write!(f, "ingest length not a whole number of rows"),
        }
    }
}

impl std::error::Error for StoreError {}
