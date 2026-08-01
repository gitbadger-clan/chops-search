//! Loading a model2vec model directory (tokenizer.json + model.safetensors).
//!
//! Deliberately offline: fetching the model is a one-time
//! `huggingface-cli download`, not something the build tool does silently.

use std::fs;
use std::path::Path;

use anyhow::{bail, Context, Result};

/// Read tokenizer.json + model.safetensors from a model2vec model dir.
/// Returns (tokens ordered by id, f32 rows, dim).
pub fn load_model2vec(dir: &Path) -> Result<(Vec<String>, Vec<f32>, usize)> {
    // Vocab from tokenizer.json → model.vocab: { token: id }.
    let tok_path = dir.join("tokenizer.json");
    let tok_raw = fs::read_to_string(&tok_path)
        .with_context(|| format!("reading {}", tok_path.display()))?;
    let tok_json: serde_json::Value = serde_json::from_str(&tok_raw)?;
    let vocab_obj = tok_json
        .pointer("/model/vocab")
        .and_then(|v| v.as_object())
        .context("tokenizer.json has no model.vocab object (expected WordPiece)")?;

    let mut tokens: Vec<Option<String>> = vec![None; vocab_obj.len()];
    for (tok, id) in vocab_obj {
        let id = id.as_u64().context("non-integer id in vocab")? as usize;
        if id >= tokens.len() {
            tokens.resize(id + 1, None);
        }
        tokens[id] = Some(tok.clone());
    }
    let tokens: Vec<String> = tokens
        .into_iter()
        .enumerate()
        .map(|(i, t)| t.with_context(|| format!("vocab id {i} missing")))
        .collect::<Result<_>>()?;

    // Embedding matrix from safetensors.
    let st_path = dir.join("model.safetensors");
    let st_raw =
        fs::read(&st_path).with_context(|| format!("reading {}", st_path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&st_raw)?;
    // model2vec names the tensor "embeddings"; fall back to the sole tensor.
    let names = st.names();
    let name = if names.iter().any(|n| *n == "embeddings") {
        "embeddings"
    } else if names.len() == 1 {
        names[0]
    } else {
        bail!("expected an 'embeddings' tensor, found: {names:?}");
    };
    let t = st.tensor(name)?;
    let shape = t.shape();
    if shape.len() != 2 {
        bail!("embeddings tensor is not 2-D: {shape:?}");
    }
    let (n, dim) = (shape[0], shape[1]);
    if n != tokens.len() {
        bail!("vocab has {} tokens but matrix has {n} rows", tokens.len());
    }
    let rows: Vec<f32> = match t.dtype() {
        safetensors::Dtype::F32 => t
            .data()
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        other => bail!(
            "embeddings dtype is {other:?}; convert to f32 first \
             (model2vec exports f32 by default)"
        ),
    };
    Ok((tokens, rows, dim))
}
