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
//!   w_title f32 | w_tag f32 | w_desc f32        (BM25F field weights)
//!   n_docs u16 | docs: n_docs × { url str16, title str16 }
//!   n_chunks u32 | chunk_doc: n_chunks × u16
//!   chunk_vecs: n_chunks × dim × i8
//!   n_terms u32 | terms: n_terms × { term str16, n_post u16,
//!                                    postings n_post × { doc u16, title u16,
//!                                      tag u16, desc u16, body u16 } }
//!
//! The field weights are stored rather than compiled in because the right
//! values are corpus-dependent — the same reason `min_cos` will be — and
//! because the browser has no other way to learn what
//! `chops-search.toml` said. The CLI can still override them per run for
//! sweeps, which is the whole point: whether a field earns its weight is
//! a question about a corpus, answerable with a flag rather than a
//! rebuild.
//!
//! The row matrix itself (`model.rows.i8`) is deliberately headerless raw
//! bytes so that row i sits at byte offset exactly i × dim — that identity
//! is what makes HTTP range requests trivial. `model.prefix.i8` is a
//! verbatim copy of its first prefix_rows × dim bytes, published as a
//! separate file so the eager part is a plain cacheable GET instead of a
//! range request.

use crate::FormatError;
use crate::bytes::{Reader, Writer};
use crate::keyword::{FieldWeights, KeywordIndex};
use std::collections::HashMap;

const MAGIC_MODEL: &[u8; 4] = b"CHPM";
const MAGIC_INDEX: &[u8; 4] = b"CHPI";
/// 3 introduced BM25F (postings 4 → 8 bytes, field weights in the header).
/// 4 added the description field (postings 8 → 10 bytes). Every bump so
/// far has widened the posting record, which is exactly why the version
/// check matters: every field is a u16 and nothing in the byte stream
/// announces its own shape, so an old reader would parse a new file into
/// plausible garbage rather than failing. Both artifacts share the
/// constant and are rebuilt together; the meta layout is unchanged since 2.
const VERSION: u16 = 4;

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
///
/// `desc` is Zola's front-matter description. It has its own field rather
/// than being folded into body for two reasons, neither of which is
/// recall: counting it as body inflated `dl_body`, so a longer
/// description quietly discounted every other term on the page; and a
/// field is the only way to make "should descriptions count here?" a
/// query-time flag instead of a rebuild.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Posting {
    pub doc: u16,
    pub title: u16,
    pub tag: u16,
    pub desc: u16,
    pub body: u16,
}

pub struct Index {
    pub dim: u16,
    pub global_scale: f32,
    /// BM25F field weights this index was built to be scored with. Body
    /// is implicitly 1.0, so these three are the whole knob set.
    pub weights: FieldWeights,
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
        w.f32(self.weights.title);
        w.f32(self.weights.tag);
        w.f32(self.weights.desc);
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
                w.u16(p.desc);
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
                let desc = r.u16()?;
                let body = r.u16()?;
                if doc as usize >= n_docs {
                    return Err(FormatError::Inconsistent("posting points past docs"));
                }
                postings.push(Posting {
                    doc,
                    title,
                    tag,
                    desc,
                    body,
                });
            }
            terms.push((term, postings));
        }
        Ok(Index {
            dim,
            global_scale,
            weights,
            docs,
            chunk_doc,
            chunk_vecs,
            terms,
        })
    }

    /// The keyword half of this index. Weights are deliberately NOT baked
    /// in here: they live in `ScoreOpts`, seeded from `weights` at engine
    /// construction, so eval can sweep them without a rebuild.
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

    fn post(doc: u16, title: u16, tag: u16, desc: u16, body: u16) -> Posting {
        Posting {
            doc,
            title,
            tag,
            desc,
            body,
        }
    }

    fn index_with(weights: FieldWeights, postings: Vec<Posting>) -> Index {
        Index {
            dim: 1,
            global_scale: 0.01,
            weights,
            docs: vec![Doc {
                url: "/a/".into(),
                title: "A".into(),
            }],
            chunk_doc: vec![0],
            chunk_vecs: vec![1],
            terms: vec![("a".into(), postings)],
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
            weights: FieldWeights {
                title: 2.0,
                tag: 4.0,
                desc: 1.0,
            },
            docs: vec![Doc {
                url: "/a/".into(),
                title: "A".into(),
            }],
            chunk_doc: vec![0, 0],
            chunk_vecs: vec![1, -2, 3, -4],
            terms: vec![("pydub".into(), vec![post(0, 1, 2, 3, 4)])],
        };
        let bytes = idx.write();
        let back = Index::read(&bytes).unwrap();
        assert_eq!(back.docs.len(), 1);
        assert_eq!(back.chunk_vecs, vec![1, -2, 3, -4]);
        assert_eq!(back.terms[0].0, "pydub");
        // Every field survives independently and in order: the four tfs
        // are distinct values precisely so a transposed pair in write or
        // read cannot pass this.
        assert_eq!(back.terms[0].1, vec![post(0, 1, 2, 3, 4)]);
        assert_eq!(back.weights.title, 2.0);
        assert_eq!(back.weights.tag, 4.0);
        assert_eq!(back.weights.desc, 1.0);
    }

    #[test]
    fn zero_weights_are_legal() {
        // w_desc = 0 is the "does this field earn its keep" sweep point,
        // and the reason the field exists as a field at all.
        let idx = index_with(
            FieldWeights {
                title: 0.0,
                tag: 0.0,
                desc: 0.0,
            },
            vec![post(0, 0, 0, 1, 1)],
        );
        let back = Index::read(&idx.write()).unwrap();
        assert_eq!(back.weights.title, 0.0);
        assert_eq!(back.weights.tag, 0.0);
        assert_eq!(back.weights.desc, 0.0);
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
        // The v3 posting record was 8 bytes against v4's 10. Reading one
        // as the other must fail at the header, not at the postings.
        let mut bytes = index_with(FieldWeights::default(), vec![post(0, 0, 0, 0, 1)]).write();
        bytes[4..6].copy_from_slice(&3u16.to_le_bytes());
        assert_eq!(Index::read(&bytes).err(), Some(FormatError::BadHeader));
    }

    #[test]
    fn nonsense_weights_rejected() {
        let bytes = index_with(FieldWeights::default(), vec![post(0, 0, 0, 0, 1)]).write();
        // w_title sits at magic(4) + version(2) + dim(2) + gscale(4) = 12,
        // then w_tag at 16 and w_desc at 20.
        for at in [12usize, 16, 20] {
            let mut b = bytes.clone();
            b[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            assert_eq!(
                Index::read(&b).err(),
                Some(FormatError::Inconsistent("field weight out of range")),
                "NaN at byte {at} was accepted"
            );
        }
        let mut b = bytes.clone();
        b[20..24].copy_from_slice(&(-1.0f32).to_le_bytes());
        assert_eq!(
            Index::read(&b).err(),
            Some(FormatError::Inconsistent("field weight out of range"))
        );
    }
}
