//! The two eager binary artifacts. Layouts are versioned and hand-rolled;
//! JS never parses these — it pumps the bytes straight into the engine.
//!
//! `model.meta.bin` — must be COMPLETE before any query (WordPiece does
//! longest-match over the full vocab; a partial vocab silently tokenizes
//! differently and produces a wrong vector with no error):
//!
//!   magic "CHPM" | version u16 | dim u16 | n_rows u32 | prefix_rows u32
//!   scales:  n_rows × f32                       (per-row dequant scales)
//!   vocab:   n_rows × str16                     (index == id == matrix row)
//!
//! `index.bin` — chunk vectors + doc metadata + keyword postings. Small
//! (tens of KB for a blog), loads with the meta:
//!
//!   magic "CHPI" | version u16 | dim u16 | gscale f32
//!   w_title f32 | w_tag f32                     (BM25F field weights)
//!   n_docs u16 | docs: n_docs × { url str16, title str16 }
//!   n_chunks u32 | chunk_doc: n_chunks × u16
//!   chunk_vecs: n_chunks × dim × i8
//!   n_terms u32 | terms: n_terms × { term str16, n_post u16,
//!                                    postings n_post × { doc u16,
//!                                      title u16, tag u16, body u16 } }
//!
//! The field weights are stored rather than compiled in because the right
//! values are corpus-dependent — the same reason `min_cos` is per-index —
//! and because the browser has no other way to learn what
//! `chops-search.toml` said. The CLI can still override them per run for
//! sweeps.
//!
//! The row matrix itself (`model.rows.i8`) is deliberately headerless raw
//! bytes so that row i sits at byte offset exactly i × dim — that identity
//! is what makes HTTP range requests trivial. `model.prefix.i8` is a
//! verbatim copy of its first prefix_rows × dim bytes, published as a
//! separate file so the eager part is a plain cacheable GET instead of a
//! range request.

use crate::FormatError;
use crate::bytes::{Reader, Writer};
use crate::keyword::KeywordIndex;
use std::collections::HashMap;

const MAGIC_MODEL: &[u8; 4] = b"CHPM";
const MAGIC_INDEX: &[u8; 4] = b"CHPI";
/// Bumped to 3 for BM25F: postings grew from 4 to 8 bytes and the header
/// gained the field weights. A v2 reader would parse a v3 file into
/// plausible garbage rather than failing, so the version check is what
/// makes the layout change safe. Both artifacts share the constant and
/// are rebuilt together; the meta layout itself is unchanged.
const VERSION: u16 = 3;

pub struct ModelMeta {
    pub dim: u16,
    pub prefix_rows: u32,
    pub scales: Vec<f32>,
    /// Ordered by id; id == matrix row (frequency order is baked in at
    /// build time by renumbering, so no remap table exists at runtime).
    pub tokens: Vec<String>,
}

impl ModelMeta {
    pub fn n_rows(&self) -> u32 {
        self.tokens.len() as u32
    }

    pub fn write(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.buf.extend_from_slice(MAGIC_MODEL);
        w.u16(VERSION);
        w.u16(self.dim);
        w.u32(self.tokens.len() as u32);
        w.u32(self.prefix_rows);
        for &s in &self.scales {
            w.f32(s);
        }
        for t in &self.tokens {
            w.str16(t);
        }
        w.buf
    }

    pub fn read(buf: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader::new(buf);
        if r.take(4)? != MAGIC_MODEL {
            return Err(FormatError::BadHeader);
        }
        if r.u16()? != VERSION {
            return Err(FormatError::BadHeader);
        }
        let dim = r.u16()?;
        let n_rows = r.u32()? as usize;
        let prefix_rows = r.u32()?;
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
        Ok(ModelMeta {
            dim,
            prefix_rows,
            scales,
            tokens,
        })
    }
}

pub struct Doc {
    pub url: String,
    pub title: String,
}

/// Per-field term frequencies for one document. Fields are separate
/// rather than pre-multiplied because BM25F normalizes each field by
/// its own length: a term in a 5-word title should score like a term
/// in a 5-word field, not get averaged against 2,000 words of body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posting {
    pub doc: u16,
    pub title: u16,
    pub tag: u16,
    pub body: u16,
}

pub struct Index {
    pub dim: u16,
    pub global_scale: f32,
    /// BM25F field weights this index was built to be scored with. Body
    /// is implicitly 1.0, so these two are the whole knob set.
    pub w_title: f32,
    pub w_tag: f32,
    pub docs: Vec<Doc>,
    /// chunk index → doc id
    pub chunk_doc: Vec<u16>,
    /// n_chunks × dim
    pub chunk_vecs: Vec<i8>,
    /// term → postings, each carrying per-field term frequencies
    pub terms: Vec<(String, Vec<Posting>)>,
}

impl Index {
    pub fn write(&self) -> Vec<u8> {
        let mut w = Writer::new();
        w.buf.extend_from_slice(MAGIC_INDEX);
        w.u16(VERSION);
        w.u16(self.dim);
        w.f32(self.global_scale);
        w.f32(self.w_title);
        w.f32(self.w_tag);
        w.u16(self.docs.len() as u16);
        for d in &self.docs {
            w.str16(&d.url);
            w.str16(&d.title);
        }
        w.u32(self.chunk_doc.len() as u32);
        for &cd in &self.chunk_doc {
            w.u16(cd);
        }
        w.i8s(&self.chunk_vecs);
        w.u32(self.terms.len() as u32);
        for (term, postings) in &self.terms {
            w.str16(term);
            w.u16(postings.len() as u16);
            for p in postings {
                w.u16(p.doc);
                w.u16(p.title);
                w.u16(p.tag);
                w.u16(p.body);
            }
        }
        w.buf
    }

    pub fn read(buf: &[u8]) -> Result<Self, FormatError> {
        let mut r = Reader::new(buf);
        if r.take(4)? != MAGIC_INDEX {
            return Err(FormatError::BadHeader);
        }
        if r.u16()? != VERSION {
            return Err(FormatError::BadHeader);
        }
        let dim = r.u16()?;
        let global_scale = r.f32()?;
        let w_title = r.f32()?;
        let w_tag = r.f32()?;
        // A NaN weight would silently NaN every score it touched, and a
        // negative one would make a field's presence count against the
        // document. Neither is a state the builder can produce, so reject
        // the artifact rather than ranking on it.
        if !w_title.is_finite() || !w_tag.is_finite() || w_title < 0.0 || w_tag < 0.0 {
            return Err(FormatError::Inconsistent("field weight out of range"));
        }
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
            let n_post = r.u16()? as usize;
            let mut postings: Vec<Posting> = Vec::with_capacity(n_post);
            for _ in 0..n_post {
                let doc = r.u16()?;
                let title = r.u16()?;
                let tag = r.u16()?;
                let body = r.u16()?;
                if doc as usize >= n_docs {
                    return Err(FormatError::Inconsistent("posting points past docs"));
                }
                postings.push(Posting {
                    doc,
                    title,
                    tag,
                    body,
                });
            }
            terms.push((term, postings));
        }
        Ok(Index {
            dim,
            global_scale,
            w_title,
            w_tag,
            docs,
            chunk_doc,
            chunk_vecs,
            terms,
        })
    }

    /// The keyword half of this index. Weights are deliberately NOT baked
    /// in here: they live in `ScoreOpts`, seeded from `w_title`/`w_tag` at
    /// engine construction, so eval can sweep them without a rebuild.
    pub fn keyword_index(&self) -> KeywordIndex {
        let mut map: HashMap<Box<str>, Vec<Posting>> = HashMap::new();
        for (term, postings) in &self.terms {
            map.insert(Box::from(term.as_str()), postings.clone());
        }
        KeywordIndex::new(self.docs.len() as u16, map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn post(doc: u16, title: u16, tag: u16, body: u16) -> Posting {
        Posting {
            doc,
            title,
            tag,
            body,
        }
    }

    #[test]
    fn model_meta_roundtrip() {
        let m = ModelMeta {
            dim: 4,
            prefix_rows: 1,
            scales: vec![0.5, 0.25],
            tokens: vec!["the".into(), "##s".into()],
        };
        let bytes = m.write();
        let back = ModelMeta::read(&bytes).unwrap();
        assert_eq!(back.dim, 4);
        assert_eq!(back.prefix_rows, 1);
        assert_eq!(back.scales, vec![0.5, 0.25]);
        assert_eq!(back.tokens, vec!["the".to_string(), "##s".to_string()]);
    }

    #[test]
    fn index_roundtrip() {
        let idx = Index {
            dim: 2,
            global_scale: 0.01,
            w_title: 2.0,
            w_tag: 4.0,
            docs: vec![Doc {
                url: "/a/".into(),
                title: "A".into(),
            }],
            chunk_doc: vec![0, 0],
            chunk_vecs: vec![1, -2, 3, -4],
            terms: vec![("pydub".into(), vec![post(0, 1, 2, 3)])],
        };
        let bytes = idx.write();
        let back = Index::read(&bytes).unwrap();
        assert_eq!(back.docs.len(), 1);
        assert_eq!(back.chunk_vecs, vec![1, -2, 3, -4]);
        assert_eq!(back.terms[0].0, "pydub");
        // Every field survives independently: a transposed pair here
        // would silently swap title and tag weighting at query time.
        assert_eq!(back.terms[0].1, vec![post(0, 1, 2, 3)]);
        assert_eq!(back.w_title, 2.0);
        assert_eq!(back.w_tag, 4.0);
    }

    #[test]
    fn zero_weights_are_legal() {
        // w_title = 0 is a meaningful sweep point ("ignore titles"), not
        // a corrupt artifact.
        let idx = Index {
            dim: 1,
            global_scale: 0.01,
            w_title: 0.0,
            w_tag: 0.0,
            docs: vec![Doc {
                url: "/a/".into(),
                title: "A".into(),
            }],
            chunk_doc: vec![0],
            chunk_vecs: vec![1],
            terms: vec![("a".into(), vec![post(0, 0, 0, 1)])],
        };
        let back = Index::read(&idx.write()).unwrap();
        assert_eq!(back.w_title, 0.0);
        assert_eq!(back.w_tag, 0.0);
    }

    #[test]
    fn bad_magic_rejected() {
        assert_eq!(
            ModelMeta::read(b"XXXX0000").err(),
            Some(FormatError::BadHeader)
        );
    }

    #[test]
    fn stale_version_rejected() {
        // The v2 posting record was half the width of v3's. Reading one
        // as the other must fail at the header, not at the postings.
        let mut bytes = Index {
            dim: 1,
            global_scale: 0.01,
            w_title: 2.0,
            w_tag: 4.0,
            docs: vec![Doc {
                url: "/a/".into(),
                title: "A".into(),
            }],
            chunk_doc: vec![0],
            chunk_vecs: vec![1],
            terms: vec![("a".into(), vec![post(0, 0, 0, 1)])],
        }
        .write();
        bytes[4..6].copy_from_slice(&2u16.to_le_bytes());
        assert_eq!(Index::read(&bytes).err(), Some(FormatError::BadHeader));
    }

    #[test]
    fn nonsense_weights_rejected() {
        let mut bytes = Index {
            dim: 1,
            global_scale: 0.01,
            w_title: 2.0,
            w_tag: 4.0,
            docs: vec![Doc {
                url: "/a/".into(),
                title: "A".into(),
            }],
            chunk_doc: vec![0],
            chunk_vecs: vec![1],
            terms: vec![("a".into(), vec![post(0, 0, 0, 1)])],
        }
        .write();
        // w_title sits at magic(4) + version(2) + dim(2) + gscale(4) = 12.
        bytes[12..16].copy_from_slice(&f32::NAN.to_le_bytes());
        assert_eq!(
            Index::read(&bytes).err(),
            Some(FormatError::Inconsistent("field weight out of range"))
        );

        bytes[12..16].copy_from_slice(&(-1.0f32).to_le_bytes());
        assert_eq!(
            Index::read(&bytes).err(),
            Some(FormatError::Inconsistent("field weight out of range"))
        );
    }
}
