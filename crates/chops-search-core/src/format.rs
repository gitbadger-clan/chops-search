//! Wire formats for the two eager artifacts: model.meta.bin (vocab and
//! per-row scales) and index.bin (chunk vectors, docs, keyword postings,
//! and the corpus's scoring configuration).
//!
//! Everything is little-endian, length-prefixed, and written in one
//! deterministic pass — two builds of the same inputs must be
//! byte-identical, because the artifact filenames are content hashes and
//! a nondeterministic byte forces every visitor to re-download.
//!
//! READS VALIDATE, NOT JUST PARSE. The engine trusts what comes out of
//! here: chunk_doc entries and posting doc ids index into per-doc arrays
//! unguarded, and in wasm an out-of-bounds panic is an aborted module,
//! not a caught error. Likewise a NaN field weight NaNs every score it
//! touches, a NaN min_gap silently disarms the corroboration gate
//! (`gap < NaN` is false), and a NaN rrf_alpha NaNs every fused score —
//! all quieter and worse than rejecting the artifact. So the readers
//! reject anything the builder cannot legitimately produce, and the
//! scoring rails here mirror the config parser's rails exactly: an
//! artifact must not be able to carry a value chops-search.toml would
//! have refused.
//!
//! VERSIONING. One version constant per artifact, checked strictly on
//! read: a mismatch is a rebuild instruction, never a defaulted parse.
//! The manifest hash already guarantees a browser can never pair a new
//! runtime with an old index (the filenames change together), so the
//! only party who can hit the version error is a developer running a
//! stale out/ against a newer binary, and that developer wants a loud
//! message, not a silently disarmed gate. Breaking changes batch into a
//! single bump so users rebuild once, not once per knob.
//!
//! The v5 batch also renamed the magics (CHPM/CHPI → CSMM/CSIX) and
//! widened the version field to u32, which would have made the rebuild
//! message unreachable by its one intended audience — a stale v4 out/
//! fails the MAGIC check, not the version check. The readers therefore
//! recognize the legacy magics specifically, to say "rebuild" rather
//! than the confusing "not an index.bin" (it is one; it's just old).
//!
//! Index version history:
//!   v3  compound terms join the postings (df semantics change)
//!   v4  BM25F: per-field tfs in postings, field weights in the header
//!   v5  scoring calibration: min_gap, rrf_alpha, optional min_cos
//!       override ride next to the weights, same provenance argument —
//!       a value calibrated against a corpus travels with the corpus,
//!       and index.bin is the only way the browser learns it
//!
//! model.meta.bin is at version 2: the v5 batch rewrote its header too
//! (new magic, u32 version field, header field order), so it bumped in
//! the same batch even though it carries no scoring — the two artifacts
//! are rebuilt together and their version discipline moves together.
//!
//! THE min_cos ENCODING. "Absent" and "0.0" are different claims: absent
//! means "derive the floor from dimensionality at construction" and 0.0
//! means "floor off" (floors disable at zero, per the knob convention).
//! The wire keeps them distinct with a fixed-width presence flag — one
//! u8 followed by an f32 that is written as 0.0 when absent and ignored
//! on read. Fixed width rather than conditional so the layout is
//! trivially seekable and the absent case has exactly one byte
//! representation, which byte-stability requires.

use std::collections::HashMap;

use crate::FormatError;
use crate::bytes::{Reader, Writer};
use crate::keyword::{FieldWeights, KeywordIndex};

/// index.bin magic + version.
pub const INDEX_MAGIC: &[u8; 4] = b"CSIX";
pub const INDEX_VERSION: u32 = 5;

/// model.meta.bin magic + version.
pub const META_MAGIC: &[u8; 4] = b"CSMM";
pub const META_VERSION: u32 = 2;

/// Pre-v5 magics, recognized only to emit the rebuild message. Never
/// parsed: the old layouts had a u16 version where the u32 now sits.
const LEGACY_INDEX_MAGIC: &[u8; 4] = b"CHPI";
const LEGACY_META_MAGIC: &[u8; 4] = b"CHPM";

/// One indexed document, in doc-id order.
#[derive(Debug, Clone, PartialEq)]
pub struct Doc {
    pub url: String,
    pub title: String,
}

/// One term's occurrence counts in one document, per BM25F field. Raw
/// counts — the weights apply at query time, after per-field length
/// normalization (see keyword.rs for why pre-multiplying here was a bug).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posting {
    pub doc: u16,
    pub title: u16,
    pub tag: u16,
    pub desc: u16,
    pub body: u16,
}

/// The vocab-and-scales artifact. Complete and eager: the engine cannot
/// tokenize without the full vocabulary, and the per-row scales are what
/// make range-fetched int8 rows meaningful.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMeta {
    pub dim: u16,
    pub prefix_rows: u32,
    /// One dequantization scale per row, row order.
    pub scales: Vec<f32>,
    /// One token per row, row (frequency) order.
    pub tokens: Vec<String>,
}

/// The corpus artifact: everything the engine needs at construction.
#[derive(Debug, Clone, PartialEq)]
pub struct Index {
    pub dim: u16,
    /// Global dequantization scale for chunk_vecs.
    pub global_scale: f32,
    /// BM25F field weights the corpus was built with (v4).
    pub weights: FieldWeights,
    /// Corroboration gate threshold (v5). 0.0 ships the gate disarmed,
    /// which is also the compiled default, so absent-in-config and
    /// written-as-zero coincide harmlessly.
    pub min_gap: f32,
    /// Confidence-weighted fusion coefficient (v5). 0.0 is plain RRF.
    pub rrf_alpha: f32,
    /// Relevance-floor override (v5). None: derive from dim at engine
    /// construction. Some(0.0): floor off. Different engines — see the
    /// module header for the wire encoding that keeps them distinct.
    pub min_cos: Option<f32>,
    pub docs: Vec<Doc>,
    /// chunk index → doc id, contiguous per doc in doc order.
    pub chunk_doc: Vec<u16>,
    /// n_chunks × dim int8, chunk order.
    pub chunk_vecs: Vec<i8>,
    /// term → postings, terms sorted, postings by ascending doc id.
    /// A Vec of pairs rather than a map so the artifact is byte-stable.
    pub terms: Vec<(String, Vec<Posting>)>,
}

// ---------------------------------------------------------------------
// Scoring rails, shared shape with config.rs
// ---------------------------------------------------------------------

/// min_gap and min_cos are cosine-space quantities; outside 0..=1 is a
/// unit error, and NaN silently changes engine behavior (see the module
/// header). Same range the config parser enforces.
fn check_cosine(v: f32, what: &'static str) -> Result<(), FormatError> {
    if !v.is_finite() || !(0.0..=1.0).contains(&v) {
        return Err(FormatError::Inconsistent(what));
    }
    Ok(())
}

fn check_alpha(v: f32) -> Result<(), FormatError> {
    if !v.is_finite() || !(0.0..=100.0).contains(&v) {
        return Err(FormatError::Inconsistent(
            "rrf_alpha out of range (0..=100)",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------
// ModelMeta
// ---------------------------------------------------------------------

impl ModelMeta {
    pub fn write(&self) -> Vec<u8> {
        debug_assert_eq!(self.scales.len(), self.tokens.len(), "one scale per row");
        let mut w = Writer::new();
        w.buf.extend_from_slice(META_MAGIC);
        w.u32(META_VERSION);
        w.u16(self.dim);
        w.u32(self.prefix_rows);
        w.u32(self.scales.len() as u32);
        for &s in &self.scales {
            w.f32(s);
        }
        for t in &self.tokens {
            w.str16(t);
        }
        w.buf
    }

    pub fn read(bytes: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader::new(bytes);
        let magic = r.take(4)?;
        if magic == LEGACY_META_MAGIC {
            return Err(FormatError::Inconsistent(
                "model.meta.bin was built by an older chops-search version; \
                 run `chops-search build` to regenerate",
            ));
        }
        if magic != META_MAGIC {
            return Err(FormatError::Inconsistent("not a model.meta.bin"));
        }
        if r.u32()? != META_VERSION {
            return Err(FormatError::Inconsistent(
                "model.meta.bin was built by a different chops-search version; \
                 run `chops-search build` to regenerate",
            ));
        }
        let dim = r.u16()?;
        let prefix_rows = r.u32()?;
        let n_rows = r.u32()? as usize;
        if dim == 0 || n_rows == 0 {
            return Err(FormatError::Inconsistent("zero dim or rows"));
        }
        if prefix_rows as usize > n_rows {
            return Err(FormatError::Inconsistent("prefix larger than matrix"));
        }
        let scales = r.f32s(n_rows)?;
        let mut tokens = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            tokens.push(r.str16()?.to_owned());
        }
        if r.remaining() != 0 {
            return Err(FormatError::Inconsistent("trailing bytes after artifact"));
        }
        Ok(ModelMeta {
            dim,
            prefix_rows,
            scales,
            tokens,
        })
    }
}

// ---------------------------------------------------------------------
// Index
// ---------------------------------------------------------------------

impl Index {
    pub fn write(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.buf.extend_from_slice(INDEX_MAGIC);
        w.u32(INDEX_VERSION);
        w.u16(self.dim);
        w.f32(self.global_scale);

        // v4: the field weights the corpus was built with.
        w.f32(self.weights.title);
        w.f32(self.weights.tag);
        w.f32(self.weights.desc);

        // v5: scoring calibration, same provenance as the weights.
        w.f32(self.min_gap);
        w.f32(self.rrf_alpha);
        // Presence flag + fixed-width value; the absent case writes 0.0
        // so there is exactly one byte representation of "derive".
        match self.min_cos {
            Some(v) => {
                w.u8(1);
                w.f32(v);
            }
            None => {
                w.u8(0);
                w.f32(0.0);
            }
        }

        w.u16(self.docs.len() as u16);
        for d in &self.docs {
            w.str16(&d.url);
            w.str16(&d.title);
        }

        w.u32(self.chunk_doc.len() as u32);
        for &c in &self.chunk_doc {
            w.u16(c);
        }
        debug_assert_eq!(
            self.chunk_vecs.len(),
            self.chunk_doc.len() * self.dim as usize
        );
        w.i8s(&self.chunk_vecs);

        w.u32(self.terms.len() as u32);
        for (term, postings) in &self.terms {
            w.str16(term);
            w.u32(postings.len() as u32);
            for p in postings {
                w.u16(p.doc);
                w.u16(p.title);
                w.u16(p.tag);
                w.u16(p.desc);
                w.u16(p.body);
            }
        }
        w.buf
    }

    pub fn read(bytes: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader::new(bytes);
        let magic = r.take(4)?;
        if magic == LEGACY_INDEX_MAGIC {
            return Err(FormatError::Inconsistent(
                "index.bin was built by an older chops-search version; \
                 run `chops-search build` to regenerate",
            ));
        }
        if magic != INDEX_MAGIC {
            return Err(FormatError::Inconsistent("not an index.bin"));
        }
        if r.u32()? != INDEX_VERSION {
            // Strict on purpose: the manifest hash means browsers never
            // see a mismatched pair, so the only reader who can land
            // here is a developer with a stale out/, and a defaulted
            // parse would hand them a silently different engine (gate
            // disarmed, fusion unweighted) instead of this sentence.
            return Err(FormatError::Inconsistent(
                "index.bin was built by a different chops-search version; \
                 run `chops-search build` to regenerate",
            ));
        }
        let dim = r.u16()?;
        let global_scale = r.f32()?;
        let weights = FieldWeights {
            title: r.f32()?,
            tag: r.f32()?,
            desc: r.f32()?,
        };
        // A NaN weight would silently NaN every score it touched, and a
        // negative one would make a field's presence count against the
        // document. Neither is a state the builder can produce, so reject
        // the artifact rather than ranking on it.
        if !weights.is_sane() {
            return Err(FormatError::Inconsistent("field weight out of range"));
        }

        let min_gap = r.f32()?;
        check_cosine(min_gap, "min_gap out of range (0..=1)")?;
        let rrf_alpha = r.f32()?;
        check_alpha(rrf_alpha)?;
        let min_cos = match r.u8()? {
            0 => {
                // The value slot is fixed-width; consume and discard.
                let _ = r.f32()?;
                None
            }
            1 => {
                let v = r.f32()?;
                check_cosine(v, "min_cos override out of range (0..=1)")?;
                Some(v)
            }
            _ => return Err(FormatError::Inconsistent("bad min_cos presence flag")),
        };

        let n_docs = r.u16()? as usize;
        let mut docs = Vec::with_capacity(n_docs);
        for _ in 0..n_docs {
            let url = r.str16()?.to_owned();
            let title = r.str16()?.to_owned();
            docs.push(Doc { url, title });
        }

        let n_chunks = r.u32()? as usize;
        let mut chunk_doc = Vec::with_capacity(n_chunks);
        for _ in 0..n_chunks {
            // Bounds-checked HERE because the engine indexes per-doc
            // arrays with these unguarded — in wasm that panic is an
            // aborted module, not a caught error.
            let d = r.u16()?;
            if d as usize >= n_docs {
                return Err(FormatError::Inconsistent("chunk points past docs"));
            }
            chunk_doc.push(d);
        }
        let chunk_vecs = r.i8s(n_chunks * dim as usize)?;

        let n_terms = r.u32()? as usize;
        let mut terms = Vec::with_capacity(n_terms);
        for _ in 0..n_terms {
            let term = r.str16()?.to_owned();
            let n_post = r.u32()? as usize;
            let mut postings = Vec::with_capacity(n_post);
            for _ in 0..n_post {
                let p = Posting {
                    doc: r.u16()?,
                    title: r.u16()?,
                    tag: r.u16()?,
                    desc: r.u16()?,
                    body: r.u16()?,
                };
                if p.doc as usize >= n_docs {
                    return Err(FormatError::Inconsistent("posting points past docs"));
                }
                postings.push(p);
            }
            terms.push((term, postings));
        }
        if r.remaining() != 0 {
            return Err(FormatError::Inconsistent("trailing bytes after artifact"));
        }

        Ok(Index {
            dim,
            global_scale,
            weights,
            min_gap,
            rrf_alpha,
            min_cos,
            docs,
            chunk_doc,
            chunk_vecs,
            terms,
        })
    }

    /// Build the query-time keyword structure from the serialized
    /// postings. The map is rebuilt at load rather than serialized: the
    /// artifact stays a sorted list (byte-stable), and per-field lengths
    /// are derived in KeywordIndex::new.
    pub fn keyword_index(&self) -> KeywordIndex {
        let mut map: HashMap<Box<str>, Vec<Posting>> = HashMap::with_capacity(self.terms.len());
        for (term, postings) in &self.terms {
            map.insert(Box::from(term.as_str()), postings.clone());
        }
        KeywordIndex::new(self.docs.len() as u16, map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_index(min_cos: Option<f32>) -> Index {
        Index {
            dim: 2,
            global_scale: 0.01,
            weights: FieldWeights {
                title: 2.0,
                tag: 4.0,
                desc: 1.0,
            },
            min_gap: 0.08,
            rrf_alpha: 1.0,
            min_cos,
            docs: vec![
                Doc {
                    url: "/a/".into(),
                    title: "A".into(),
                },
                Doc {
                    url: "/b/".into(),
                    title: "B page".into(),
                },
            ],
            chunk_doc: vec![0, 0, 1],
            chunk_vecs: vec![1, -2, 3, -4, 5, -6],
            terms: vec![
                (
                    "alpha".into(),
                    vec![Posting {
                        doc: 0,
                        title: 1,
                        tag: 0,
                        desc: 0,
                        body: 2,
                    }],
                ),
                (
                    "beta".into(),
                    vec![
                        Posting {
                            doc: 0,
                            title: 0,
                            tag: 1,
                            desc: 0,
                            body: 0,
                        },
                        Posting {
                            doc: 1,
                            title: 0,
                            tag: 0,
                            desc: 2,
                            body: 3,
                        },
                    ],
                ),
            ],
        }
    }

    #[test]
    fn index_round_trips_including_v5_fields() {
        let idx = sample_index(None);
        let got = Index::read(&idx.write()).unwrap();
        assert_eq!(got, idx);
        assert_eq!(got.min_gap, 0.08);
        assert_eq!(got.rrf_alpha, 1.0);
    }

    #[test]
    fn min_cos_three_states_are_distinguishable() {
        // The encoding's entire job: absent ("derive from dims"),
        // explicit 0.0 ("floor off"), and explicit 0.28 are three
        // different engines, and the wire must never conflate the first
        // two even though their value bytes are identical.
        for state in [None, Some(0.0), Some(0.28)] {
            let idx = sample_index(state);
            let got = Index::read(&idx.write()).unwrap();
            assert_eq!(got.min_cos, state, "state {state:?} did not survive");
        }
        // And the absent/zero pair specifically differ on the wire.
        assert_ne!(
            sample_index(None).write(),
            sample_index(Some(0.0)).write(),
            "absent and zero must have different byte representations"
        );
    }

    #[test]
    fn writes_are_byte_stable() {
        // The filenames are content hashes; a nondeterministic byte is a
        // forced re-download for every visitor.
        let idx = sample_index(Some(0.28));
        assert_eq!(idx.write(), idx.write());
    }

    #[test]
    fn wrong_version_is_rejected_with_a_rebuild_message() {
        let mut bytes = sample_index(None).write();
        // Corrupt the version field (bytes 4..8).
        bytes[4] = bytes[4].wrapping_add(1);
        let err = Index::read(&bytes).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("chops-search build"),
            "version error must tell the reader how to fix it: {msg}"
        );
    }

    #[test]
    fn legacy_magics_get_the_rebuild_message_too() {
        // The stale-out/ developer the version check was designed for
        // never reaches it: a v4 file fails at the MAGIC check, because
        // the magics were renamed in the same batch. "Not an index.bin"
        // would be actively confusing (it is one; it's just old), so the
        // legacy magics are recognized specifically.
        let err = Index::read(b"CHPI").unwrap_err();
        assert!(format!("{err}").contains("chops-search build"));
        let err = ModelMeta::read(b"CHPM").unwrap_err();
        assert!(format!("{err}").contains("chops-search build"));
        // Genuinely foreign bytes still get the blunt answer.
        let err = Index::read(b"XXXXxxxx").unwrap_err();
        assert!(format!("{err}").contains("not an index.bin"));
    }

    #[test]
    fn insane_weights_are_rejected_on_read() {
        // Each field checked, not just the first: a validator that
        // forgets one is exactly the bug this catches.
        for bad in [f32::NAN, -1.0, f32::INFINITY, 1e6] {
            let mut idx = sample_index(None);
            idx.weights.title = bad;
            assert!(Index::read(&idx.write()).is_err(), "title {bad} accepted");
            let mut idx = sample_index(None);
            idx.weights.tag = bad;
            assert!(Index::read(&idx.write()).is_err(), "tag {bad} accepted");
            let mut idx = sample_index(None);
            idx.weights.desc = bad;
            assert!(Index::read(&idx.write()).is_err(), "desc {bad} accepted");
        }
    }

    #[test]
    fn corrupt_scoring_calibration_is_rejected() {
        // The quiet failures these rails exist for: a NaN min_gap never
        // gates (gap < NaN is false), a NaN alpha NaNs every fused
        // score. Both must die at read, not at ranking.
        for bad in [f32::NAN, -0.1, 1.5] {
            let mut idx = sample_index(None);
            idx.min_gap = bad;
            assert!(Index::read(&idx.write()).is_err(), "min_gap {bad} accepted");
            let mut idx = sample_index(None);
            idx.min_cos = Some(bad);
            assert!(Index::read(&idx.write()).is_err(), "min_cos {bad} accepted");
        }
        for bad in [f32::NAN, -1.0, 1e6] {
            let mut idx = sample_index(None);
            idx.rrf_alpha = bad;
            assert!(
                Index::read(&idx.write()).is_err(),
                "rrf_alpha {bad} accepted"
            );
        }
        // The legal edges survive: 0.0 and 1.0 are meaningful values
        // (disarmed / whole range), not out-of-range near-misses.
        for edge in [0.0f32, 1.0] {
            let mut idx = sample_index(Some(edge));
            idx.min_gap = edge;
            assert_eq!(Index::read(&idx.write()).unwrap(), idx);
        }
    }

    #[test]
    fn chunk_pointing_past_docs_is_rejected() {
        // The engine indexes per-doc arrays with chunk_doc unguarded; in
        // wasm that panic aborts the module. This is the read-side
        // license for that trust.
        let mut idx = sample_index(None);
        idx.chunk_doc[2] = 99;
        assert!(Index::read(&idx.write()).is_err());
    }

    #[test]
    fn posting_pointing_past_docs_is_rejected() {
        let mut idx = sample_index(None);
        idx.terms[0].1[0].doc = 99;
        assert!(Index::read(&idx.write()).is_err());
    }

    #[test]
    fn truncated_index_fails_loudly() {
        let bytes = sample_index(None).write();
        assert!(Index::read(&bytes[..bytes.len() - 3]).is_err());
        assert!(Index::read(&bytes[..10]).is_err());
    }

    #[test]
    fn trailing_bytes_are_rejected() {
        let mut bytes = sample_index(None).write();
        bytes.push(0);
        assert!(Index::read(&bytes).is_err());
    }

    #[test]
    fn model_meta_round_trips() {
        let meta = ModelMeta {
            dim: 2,
            prefix_rows: 1,
            scales: vec![0.5, 0.25, 0.125],
            tokens: vec!["a".into(), "##b".into(), "long-token".into()],
        };
        assert_eq!(ModelMeta::read(&meta.write()).unwrap(), meta);
    }

    #[test]
    fn meta_rejects_impossible_shapes() {
        // Zero dim or rows: nothing downstream can do anything with the
        // matrix, and RowStore would allocate a zero-size buffer and
        // then divide by dim.
        let meta = ModelMeta {
            dim: 0,
            prefix_rows: 0,
            scales: vec![],
            tokens: vec![],
        };
        assert!(ModelMeta::read(&meta.write()).is_err());
        // A prefix claiming more rows than the matrix has: the JS pump
        // would ingest past the end.
        let meta = ModelMeta {
            dim: 2,
            prefix_rows: 5,
            scales: vec![1.0],
            tokens: vec!["a".into()],
        };
        assert!(ModelMeta::read(&meta.write()).is_err());
    }

    #[test]
    fn keyword_index_carries_the_postings() {
        let idx = sample_index(None);
        let kw = idx.keyword_index();
        assert_eq!(kw.n_docs, 2);
        assert_eq!(kw.terms.len(), 2);
        assert_eq!(kw.terms["beta"].len(), 2);
    }
}
