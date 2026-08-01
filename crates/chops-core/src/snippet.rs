//! `snippets.bin` — per-chunk display text, range-fetched on demand.
//!
//! Snippets answer the question the ranking can't: *why* did this result
//! match? That matters most exactly where the ranker is weakest — when
//! two sibling posts are both plausible and the model picks the wrong
//! one, a reader resolves it in half a second from the text.
//!
//! Deliberately NOT part of index.bin. That file is eager, and snippet
//! text is the largest thing in the corpus after the model itself
//! (~40 KB gzipped here, ~120 KB for a hundred-post site) — inlining it
//! would undo the payload work. Its own file, range-fetched after the
//! ranking is known, costs a few hundred bytes of offset table up front
//! and ~600 bytes per displayed result.
//!
//! Because it's lazy, the FULL chunk text is stored rather than a
//! truncated build-time guess: the browser windows the snippet around
//! whichever query term actually matched, which no build-time truncation
//! can anticipate.
//!
//! Layout (all little-endian):
//!
//!   0..4                 magic b"CHSN"
//!   4..6                 version u16
//!   6..8                 reserved u16 (0)
//!   8..12                n_chunks u32
//!   12..12+4*(n+1)       offsets u32 — text-relative, ascending,
//!                        offsets[i]..offsets[i+1] is chunk i
//!   text_start..         UTF-8 text blob
//!
//! The header is a known size once n_chunks is known (the engine has it
//! from index.bin), so a client fetches bytes 0..header_len as one small
//! range at boot and every snippet thereafter as a direct range.

pub const MAGIC: &[u8; 4] = b"CHSN";
pub const VERSION: u16 = 1;

/// Byte length of magic + version + reserved + n_chunks + offset table.
pub const fn header_len(n_chunks: usize) -> usize {
    12 + 4 * (n_chunks + 1)
}

#[derive(Debug, PartialEq, Eq)]
pub enum SnippetError {
    BadMagic,
    UnsupportedVersion(u16),
    Truncated,
    /// Offsets must be ascending and within the blob.
    CorruptOffsets,
    OutOfRange,
    /// A range that doesn't land on UTF-8 character boundaries.
    NotUtf8,
}

impl core::fmt::Display for SnippetError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            SnippetError::BadMagic => write!(f, "not a snippets.bin (bad magic)"),
            SnippetError::UnsupportedVersion(v) => {
                write!(
                    f,
                    "snippets.bin version {v} unsupported (expected {VERSION}); rebuild"
                )
            }
            SnippetError::Truncated => write!(f, "snippets.bin truncated"),
            SnippetError::CorruptOffsets => write!(f, "snippets.bin offset table corrupt"),
            SnippetError::OutOfRange => write!(f, "chunk id out of range"),
            SnippetError::NotUtf8 => write!(f, "snippet is not valid UTF-8"),
        }
    }
}

/// Serialize chunk texts. Order must match `Index::chunk_doc`, since chunk
/// ids are indices into both.
pub fn write(texts: &[String]) -> Vec<u8> {
    let n = texts.len();
    let mut out = Vec::with_capacity(header_len(n) + texts.iter().map(String::len).sum::<usize>());
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&(n as u32).to_le_bytes());

    let mut acc = 0u32;
    out.extend_from_slice(&acc.to_le_bytes());
    for t in texts {
        acc += t.len() as u32;
        out.extend_from_slice(&acc.to_le_bytes());
    }
    for t in texts {
        out.extend_from_slice(t.as_bytes());
    }
    out
}

/// The header alone — enough to compute every snippet's byte range without
/// holding any text. This is what a client keeps resident.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SnippetIndex {
    offsets: Vec<u32>,
    text_start: u32,
}

impl SnippetIndex {
    /// Parse from at least `header_len(n_chunks)` bytes. Passing the whole
    /// file works too.
    pub fn read(bytes: &[u8]) -> Result<Self, SnippetError> {
        if bytes.len() < 12 {
            return Err(SnippetError::Truncated);
        }
        if &bytes[0..4] != MAGIC {
            return Err(SnippetError::BadMagic);
        }
        let version = u16::from_le_bytes([bytes[4], bytes[5]]);
        if version != VERSION {
            return Err(SnippetError::UnsupportedVersion(version));
        }
        let n = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        let need = header_len(n);
        if bytes.len() < need {
            return Err(SnippetError::Truncated);
        }
        let mut offsets = Vec::with_capacity(n + 1);
        let mut prev = 0u32;
        for i in 0..=n {
            let at = 12 + i * 4;
            let v = u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
            if i > 0 && v < prev {
                return Err(SnippetError::CorruptOffsets);
            }
            prev = v;
            offsets.push(v);
        }
        Ok(SnippetIndex {
            offsets,
            text_start: need as u32,
        })
    }

    pub fn n_chunks(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Absolute byte range of chunk `i` within the file, half-open —
    /// exactly what goes into a Range header.
    pub fn range(&self, i: usize) -> Result<(u32, u32), SnippetError> {
        if i + 1 >= self.offsets.len() {
            return Err(SnippetError::OutOfRange);
        }
        Ok((
            self.text_start + self.offsets[i],
            self.text_start + self.offsets[i + 1],
        ))
    }

    /// Read chunk `i` out of a complete in-memory file. Used by the CLI;
    /// the browser slices ranges instead and never holds the whole blob.
    pub fn text<'a>(&self, bytes: &'a [u8], i: usize) -> Result<&'a str, SnippetError> {
        let (lo, hi) = self.range(i)?;
        let (lo, hi) = (lo as usize, hi as usize);
        if hi > bytes.len() {
            return Err(SnippetError::Truncated);
        }
        core::str::from_utf8(&bytes[lo..hi]).map_err(|_| SnippetError::NotUtf8)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Vec<String> {
        vec![
            "Scraping with rust and headless chrome".to_string(),
            "The chromiumoxide api is higher level.".to_string(),
            String::new(), // empty chunks must round-trip
            "café ☕ unicode — multi-byte".to_string(),
        ]
    }

    #[test]
    fn round_trips() {
        let texts = sample();
        let bytes = write(&texts);
        let idx = SnippetIndex::read(&bytes).unwrap();
        assert_eq!(idx.n_chunks(), texts.len());
        for (i, t) in texts.iter().enumerate() {
            assert_eq!(idx.text(&bytes, i).unwrap(), t);
        }
    }

    #[test]
    fn header_alone_is_enough_for_ranges() {
        // The client fetches only the header; ranges must still be exact.
        let texts = sample();
        let bytes = write(&texts);
        let header = &bytes[..header_len(texts.len())];
        let idx = SnippetIndex::read(header).unwrap();
        for (i, t) in texts.iter().enumerate() {
            let (lo, hi) = idx.range(i).unwrap();
            assert_eq!((hi - lo) as usize, t.len());
            assert_eq!(
                core::str::from_utf8(&bytes[lo as usize..hi as usize]).unwrap(),
                t
            );
        }
    }

    #[test]
    fn multibyte_ranges_land_on_char_boundaries() {
        let texts = sample();
        let bytes = write(&texts);
        let idx = SnippetIndex::read(&bytes).unwrap();
        let s = idx.text(&bytes, 3).unwrap();
        assert!(s.contains('☕'));
        assert!(s.contains('—'));
    }

    #[test]
    fn empty_corpus() {
        let bytes = write(&[]);
        let idx = SnippetIndex::read(&bytes).unwrap();
        assert_eq!(idx.n_chunks(), 0);
        assert_eq!(idx.range(0), Err(SnippetError::OutOfRange));
    }

    #[test]
    fn rejects_bad_magic_and_version() {
        let mut bytes = write(&sample());
        bytes[0] = b'X';
        assert_eq!(SnippetIndex::read(&bytes), Err(SnippetError::BadMagic));

        let mut bytes = write(&sample());
        bytes[4] = 9;
        assert_eq!(
            SnippetIndex::read(&bytes),
            Err(SnippetError::UnsupportedVersion(9))
        );
    }

    #[test]
    fn rejects_truncation_and_descending_offsets() {
        let bytes = write(&sample());
        assert_eq!(
            SnippetIndex::read(&bytes[..10]),
            Err(SnippetError::Truncated)
        );
        assert_eq!(
            SnippetIndex::read(&bytes[..header_len(4) - 1]),
            Err(SnippetError::Truncated)
        );

        let mut bytes = write(&sample());
        // Make offsets[2] smaller than offsets[1].
        bytes[12 + 8..12 + 12].copy_from_slice(&0u32.to_le_bytes());
        assert_eq!(
            SnippetIndex::read(&bytes),
            Err(SnippetError::CorruptOffsets)
        );
    }

    #[test]
    fn out_of_range_chunk_is_an_error_not_a_panic() {
        let bytes = write(&sample());
        let idx = SnippetIndex::read(&bytes).unwrap();
        assert_eq!(idx.range(4), Err(SnippetError::OutOfRange));
        assert_eq!(idx.range(999), Err(SnippetError::OutOfRange));
    }
}
