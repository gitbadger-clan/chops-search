//! chops-search build — turn a model2vec model + a Zola content tree into
//! the five static artifacts the browser engine consumes:
//!
//!   model.meta.<hash>.bin    vocab + per-row scales (eager, complete)
//!   model.prefix.<hash>.i8   top-frequency rows (eager)
//!   model.rows.<hash>.i8     full matrix, frequency-ordered (range-fetched)
//!   index.<hash>.bin         chunk vectors + docs + keyword postings (eager)
//!   snippets.<hash>.bin      per-chunk display text (range-fetched)
//!
//! plus manifest.json (the only unhashed artifact, and the only one that
//! revalidates) and the runtime — wasm, glue, worker, page script, CSS —
//! written from bytes embedded in this binary unless --no-runtime.
//!
//! Paths and tuning come from chops-search.toml, discovered by walking up
//! from the working directory; flags override it. See config.rs.
//!
//! Expects a local model directory containing `tokenizer.json` and
//! `model.safetensors` (e.g. `hf download minishlab/potion-base-8M
//! --local-dir .chops-search/model`). No network access here; fetching the
//! model is deliberately a separate step, so a build can never fail
//! because an upstream repo moved.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};
use sha2::{Digest, Sha256};

use chops_search_cli::assets;
use chops_search_cli::config::Config;
use chops_search_cli::frontmatter::{self, FrontMatter};
use chops_search_cli::model_loader::load_model2vec;
use chops_search_cli::pca::pca_reduce;
use chops_search_core::builder::{
    embed_f32, frequency_permutation, permute_rows_f32, quantize_global, quantize_rows,
};
use chops_search_core::chunk::{chunk_prose, prepare_markdown};
use chops_search_core::format::{Doc, Index, ModelMeta};
use chops_search_core::keyword::keyword_words;
use chops_search_core::wordpiece::Vocab;

#[derive(Parser)]
#[command(
    name = "chops-search",
    about = "Static-site hybrid search index builder"
)]
struct Cli {
    /// Directory to resolve chops-search.toml from. Defaults to the
    /// working directory, walking up as cargo does for Cargo.toml.
    #[arg(long, global = true)]
    site: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum ModelCmd {
    /// Download a model2vec model and write the lockfile.
    Fetch {
        /// HuggingFace repo, e.g. minishlab/potion-base-8M.
        #[arg(default_value = "minishlab/potion-base-8M")]
        repo: String,
        /// Exact revision to pin. Defaults to the current default branch,
        /// resolved to a commit so the lock stays reproducible.
        #[arg(long)]
        revision: Option<String>,
        /// Destination. Defaults to `model` from chops-search.toml.
        #[arg(long)]
        dir: Option<PathBuf>,
    },
    /// Re-hash the model against its lockfile. No network.
    Verify {
        #[arg(long)]
        dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum Cmd {
    /// Build all search artifacts from a content tree and a model2vec model.
    Build {
        /// Zola content directory (walked recursively for .md files).
        #[arg(long)]
        content: Option<PathBuf>,
        /// Directory containing tokenizer.json and model.safetensors.
        #[arg(long)]
        model: Option<PathBuf>,
        /// Output directory (e.g. static/search).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Rows bundled eagerly in model.prefix.i8.
        #[arg(long)]
        prefix_rows: Option<u32>,
        /// Target chunk size in characters.
        #[arg(long)]
        chunk_chars: Option<usize>,
        /// Reduce embedding dimensionality via PCA before quantization
        /// (e.g. 128 halves the shipped matrix). Defaults to the model's
        /// native dimensionality.
        #[arg(long)]
        dims: Option<usize>,

        /// Artifacts only — skip the wasm + JS runtime. For CI jobs that
        /// rebuild the index without shipping a new frontend.
        #[arg(long)]
        no_runtime: bool,
    },
    /// Explain a query against built artifacts: keyword scores, best-chunk
    /// cosines, chunk counts, and RRF contributions per document.
    Query {
        /// Directory containing the built artifacts (the --out of `build`).
        #[arg(long)]
        artifacts: Option<PathBuf>,
        /// Max rows to print.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// The query string.
        query: String,
    },
    /// Score the engine against a labeled query set (recall@1 by kind).
    Eval {
        /// Directory containing the built artifacts.
        #[arg(long)]
        artifacts: Option<PathBuf>,
        /// Labeled query set.
        #[arg(long)]
        queries: Option<PathBuf>,
        /// Only run cases of this kind (exact, paraphrase, navigational, negative).
        #[arg(long)]
        kind: Option<String>,
        /// Exit non-zero if overall recall@1 falls below this fraction (0.0–1.0).
        #[arg(long, default_value_t = 0.0)]
        fail_under: f32,
        /// Minimum raw best-chunk similarity (semantic floor).
        #[arg(long)]
        min_cos: Option<f32>,
        /// Coefficient on the √(2 ln n) chunk-count correction.
        #[arg(long)]
        chunk_penalty: Option<f32>,
    },

    /// Fetch or verify the embedding model.
    Model {
        #[command(subcommand)]
        action: ModelCmd,
    },

    /// List indexed documents and their URLs — what you need to write
    /// `expect` entries after adding a post.
    Docs {
        #[arg(long)]
        artifacts: Option<PathBuf>,
    },
}

fn load_config(site: &Option<PathBuf>) -> Result<Config> {
    let start = match site {
        Some(p) => p.clone(),
        None => std::env::current_dir()?,
    };
    Config::discover(&start)
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let site = cli.site;
    match cli.cmd {
        Cmd::Build {
            content,
            model,
            out,
            prefix_rows,
            chunk_chars,
            dims,
            no_runtime,
        } => {
            let cfg = Config::discover(&std::env::current_dir()?)?.with_overrides(
                content,
                out,
                model,
                dims,
                chunk_chars,
                prefix_rows,
            );
            eprintln!(
                "config: content {}, out {}",
                cfg.content.display(),
                cfg.out.display()
            );
            build(&cfg, !no_runtime)
        }
        Cmd::Query {
            artifacts,
            limit,
            query,
        } => {
            let cfg = load_config(&site)?;
            let dir = artifacts.unwrap_or(cfg.out);
            chops_search_cli::explain::explain(&dir, &query, limit)
        }
        Cmd::Eval {
            artifacts,
            queries,
            kind,
            fail_under,
            min_cos,
            chunk_penalty,
        } => {
            let cfg = load_config(&site)?;
            let dir = artifacts.unwrap_or_else(|| cfg.out.clone());
            let queries = queries.unwrap_or_else(|| cfg.root.join("fixtures/queries.toml"));
            chops_search_cli::eval::eval(
                &dir,
                &queries,
                kind.as_deref(),
                fail_under,
                min_cos,
                chunk_penalty,
            )
        }
        Cmd::Model { action } => {
            let cfg = load_config(&site)?;
            match action {
                ModelCmd::Fetch {
                    repo,
                    revision,
                    dir,
                } => chops_search_cli::model::fetch(
                    &repo,
                    revision.as_deref(),
                    &dir.unwrap_or(cfg.model),
                ),
                ModelCmd::Verify { dir } => {
                    chops_search_cli::model::verify(&dir.unwrap_or(cfg.model))
                }
            }
        }
        Cmd::Docs { artifacts } => {
            let cfg = load_config(&site)?;
            chops_search_cli::explain::list_docs(&artifacts.unwrap_or(cfg.out))
        }
    }
}

/// Short content hash over the whole artifact set. Each part is
/// length-prefixed so two different splits can't hash identically.
fn build_hash(parts: &[&[u8]]) -> String {
    let mut h = Sha256::new();
    for p in parts {
        h.update((p.len() as u64).to_le_bytes());
        h.update(p);
    }
    h.finalize()[..8]
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// Remove artifacts from previous builds. Hashed names would otherwise
/// accumulate in static/ and get copied into the site by `zola build` —
/// invisible dead weight that grows with every rebuild.
fn clean_artifacts(out: &Path) -> Result<()> {
    if !out.exists() {
        return Ok(());
    }
    for entry in fs::read_dir(out)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let stale = name.starts_with("model.")
            || name.starts_with("index.")
            || name.starts_with("snippets.")
            || name == "manifest.json";
        if stale && entry.path().is_file() {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}

fn build(cfg: &Config, runtime: bool) -> Result<()> {
    // ---- 1. Load model -------------------------------------------------
    let (tokens, mut rows, mut dim) = load_model2vec(&cfg.model)?;
    let n_rows = tokens.len();
    eprintln!("model: {n_rows} tokens × {dim} dims");

    // Optional PCA reduction. Must happen before anything downstream so
    // chunk vectors and browser-side query vectors live in the same space.
    if let Some(k) = cfg.dims {
        if k == 0 || k > dim {
            bail!("--dims {k} out of range (model is {dim}-dimensional)");
        }
        if k < dim {
            eprintln!("pca: {dim} → {k} dims");
            rows = pca_reduce(&rows, dim, k);
            dim = k;
        }
    }
    let vocab = Vocab::from_tokens(&tokens);

    // ---- 2. Load + chunk content ---------------------------------------
    let posts = collect_posts(&cfg.content)?;
    if posts.is_empty() {
        bail!("no .md files found under {}", cfg.content.display());
    }

    struct ChunkRec {
        doc: u16,
        ids: Vec<u32>, // token ids under the ORIGINAL vocab order
        text: String,  // display text for snippets.bin
    }
    let mut chunks: Vec<ChunkRec> = Vec::new();
    let mut docs: Vec<Doc> = Vec::new();
    let mut doc_words: Vec<HashMap<String, u16>> = Vec::new();

    for post in &posts {
        let (fm, body) = frontmatter::split(&post.markdown)
            .with_context(|| format!("front matter in {}", post.rel.display()))?;
        if fm.draft {
            eprintln!("skip (draft): {}", post.rel.display());
            continue;
        }
        if !fm.in_search_index {
            eprintln!("skip (in_search_index = false): {}", post.rel.display());
            continue;
        }

        // Doc ids are assigned to INDEXED docs only — skips must not leave
        // holes, so the id is docs.len() at push time, never a loop index.
        let doc_id = docs.len();
        if doc_id > u16::MAX as usize {
            bail!("more than {} docs; widen doc ids to u32 first", u16::MAX);
        }
        let doc_id = doc_id as u16;

        // Body already has front matter split off; prepare_markdown's own
        // front-matter path is inert on it and its title is superseded.
        let (fallback_title, prose) = prepare_markdown(body);
        let title = fm
            .title
            .clone()
            .or(fallback_title)
            .unwrap_or_else(|| post.slug_title());
        let url = url_for(&post.rel, &fm);

        // Keyword terms: word-level tokens (post-normalization,
        // pre-WordPiece). Tags weigh hardest, then title, then body.
        // keyword_words only emits alphanumeric runs, so no filter here.
        let mut tf: HashMap<String, u16> = HashMap::new();
        let mut count_words = |text: &str, weight: u16| {
            let norm = Vocab::normalize(text);
            for w in keyword_words(&norm) {
                let e = tf.entry(w.to_string()).or_insert(0);
                *e = e.saturating_add(weight);
            }
        };
        count_words(&title, cfg.title_weight);
        for tag in &fm.tags {
            count_words(tag, cfg.tag_weight);
        }
        count_words(&prose, 1);
        doc_words.push(tf);

        // Synthetic chunk 0: title + tags. Body chunks never contain the
        // title, so without this the semantic side scores every doc as if
        // it were untitled.
        let mut head = title.clone();
        for tag in &fm.tags {
            head.push_str(". ");
            head.push_str(tag);
        }
        let mut texts = vec![head];
        texts.extend(chunk_prose(&prose, cfg.chunk_chars));

        for text in &texts {
            let ids = vocab.tokenize(text);
            if !ids.is_empty() {
                chunks.push(ChunkRec {
                    doc: doc_id,
                    ids,
                    text: text.trim().to_string(),
                });
            }
        }
        docs.push(Doc { url, title });
    }
    if docs.is_empty() {
        bail!("every page under {} was skipped", cfg.content.display());
    }
    eprintln!(
        "content: {} docs indexed, {} chunks",
        docs.len(),
        chunks.len()
    );

    // ---- 3. Frequency ordering ----------------------------------------
    let mut counts = vec![0u64; n_rows];
    for c in &chunks {
        for &id in &c.ids {
            counts[id as usize] += 1;
        }
    }
    let new_id_of_old = frequency_permutation(&counts);
    let rows = permute_rows_f32(&rows, dim, &new_id_of_old);
    let mut new_tokens = vec![String::new(); n_rows];
    for (old, tok) in tokens.into_iter().enumerate() {
        new_tokens[new_id_of_old[old] as usize] = tok;
    }
    for c in &mut chunks {
        for id in &mut c.ids {
            *id = new_id_of_old[*id as usize];
        }
    }

    // ---- 4. Embed chunks against the f32 table ------------------------
    let mut chunk_vecs_f32 = Vec::with_capacity(chunks.len() * dim);
    let mut chunk_doc = Vec::with_capacity(chunks.len());
    let mut snip_texts: Vec<String> = Vec::with_capacity(chunks.len());
    for c in &chunks {
        // ids are non-empty by construction → embed only returns None for
        // a zero-norm mean, which we skip rather than ship. Snippets are
        // appended in the SAME branch so chunk ids index both arrays.
        if let Some(v) = embed_f32(&c.ids, &rows, dim) {
            chunk_vecs_f32.extend(v);
            chunk_doc.push(c.doc);
            snip_texts.push(c.text.clone());
        }
    }
    let (chunk_vecs, global_scale) = quantize_global(&chunk_vecs_f32);

    // ---- 5. Quantize the table ----------------------------------------
    let (data_i8, scales) = quantize_rows(&rows, dim);

    // ---- 6. Keyword postings ------------------------------------------
    let mut postings: HashMap<String, Vec<(u16, u16)>> = HashMap::new();
    for (doc_id, tf) in doc_words.iter().enumerate() {
        for (term, &f) in tf {
            postings
                .entry(term.clone())
                .or_default()
                .push((doc_id as u16, f));
        }
    }
    let mut terms: Vec<(String, Vec<(u16, u16)>)> = postings.into_iter().collect();
    terms.sort_by(|a, b| a.0.cmp(&b.0)); // byte-stable output
    for (_, p) in &mut terms {
        p.sort_unstable();
    }

    // ---- 7. Serialize --------------------------------------------------
    // Clamped: a config asking for more prefix rows than the model has
    // would slice past the end of the row buffer below.
    let prefix_rows = cfg.prefix_rows.min(n_rows as u32);
    let meta = ModelMeta {
        dim: u16::try_from(dim).context("dim exceeds u16")?,
        prefix_rows,
        scales,
        tokens: new_tokens,
    };
    let index = Index {
        dim: meta.dim,
        global_scale,
        docs,
        chunk_doc,
        chunk_vecs,
        terms,
    };

    fs::create_dir_all(&cfg.out).with_context(|| format!("creating {}", cfg.out.display()))?;
    let rows_bytes: Vec<u8> = data_i8.iter().map(|&v| v as u8).collect();
    let prefix_bytes = &rows_bytes[..prefix_rows as usize * dim];
    let meta_out = meta.write();
    let index_out = index.write();
    let snips_out = chops_search_core::snippet::write(&snip_texts);

    let hash = build_hash(&[&meta_out, prefix_bytes, &rows_bytes, &index_out, &snips_out]);
    clean_artifacts(&cfg.out)?;

    let f_meta = format!("model.meta.{hash}.bin");
    let f_prefix = format!("model.prefix.{hash}.i8");
    let f_rows = format!("model.rows.{hash}.i8");
    let f_index = format!("index.{hash}.bin");
    let f_snips = format!("snippets.{hash}.bin");

    write_artifact(&cfg.out, &f_meta, &meta_out, true)?;
    write_artifact(&cfg.out, &f_prefix, prefix_bytes, false)?;
    write_artifact(&cfg.out, &f_rows, &rows_bytes, false)?;
    write_artifact(&cfg.out, &f_index, &index_out, true)?;
    // Range-served, so never gzipped — same reason as the rows file.
    write_artifact(&cfg.out, &f_snips, &snips_out, false)?;

    let manifest = serde_json::json!({
        "version": 1,
        "hash": hash,
        "files": {
            "meta": f_meta, "prefix": f_prefix, "rows": f_rows,
            "index": f_index, "snippets": f_snips,
        },
    });
    fs::write(
        cfg.out.join("manifest.json"),
        serde_json::to_vec_pretty(&manifest)?,
    )?;
    eprintln!("wrote {}/manifest.json (hash {hash})", cfg.out.display());

    // After the manifest, so a half-written build never leaves a runtime
    // pointing at artifacts that don't exist.
    if runtime {
        assets::write_runtime(&cfg.out)?;
    }
    Ok(())
}

/// Write an artifact, plus a gzip sibling when `compress` is set.
///
/// Cloudflare only compresses a fixed list of content types, and
/// `application/octet-stream` (what Wrangler assigns .bin/.i8) isn't on
/// it — so the eager payload ships raw unless we compress it ourselves.
/// Doing it at build time keeps this host-agnostic instead of depending
/// on a zone-level Compression Rule that doesn't travel with the repo.
///
/// Range-served files are never compressed: a byte offset into a gzip
/// stream is meaningless. The int8 row blobs wouldn't gain much anyway.
///
/// flate2 writes mtime 0 by default, so the .gz is byte-stable like
/// everything else here — worth confirming with two builds and a diff.
fn write_artifact(out: &Path, name: &str, bytes: &[u8], compress: bool) -> Result<()> {
    let path = out.join(name);
    fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    if !compress {
        eprintln!("wrote {} ({} KB)", path.display(), bytes.len() / 1024);
        return Ok(());
    }
    let gz_path = out.join(format!("{name}.gz"));
    let mut enc = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::best());
    enc.write_all(bytes).context("gzip encode")?;
    let gz = enc.finish().context("gzip finish")?;
    fs::write(&gz_path, &gz).with_context(|| format!("writing {}", gz_path.display()))?;
    eprintln!(
        "wrote {} ({} KB → {} KB gzip, {:.0}% saved)",
        path.display(),
        bytes.len() / 1024,
        gz.len() / 1024,
        100.0 - (gz.len() as f64 * 100.0 / bytes.len() as f64)
    );
    Ok(())
}

// ---------------------------------------------------------------------
// URL mapping
// ---------------------------------------------------------------------

/// Strip a leading `YYYY-MM-DD` followed by `-` or `_`, as Zola does when
/// deriving a slug from a filename or bundle directory. A name that is
/// only a date is returned unchanged (Zola doesn't strip those), as is
/// anything whose digits aren't a plausible date.
///
/// Not handled: the RFC3339 form (`2002-10-02T15:00:00Z-slug`). Zola
/// accepts it, but it's Windows-hostile and unused here.
fn strip_date_prefix(name: &str) -> &str {
    let b = name.as_bytes();
    if b.len() < 12 {
        return name;
    }
    let digits_at = |r: std::ops::Range<usize>| b[r].iter().all(u8::is_ascii_digit);
    if !(digits_at(0..4) && b[4] == b'-' && digits_at(5..7) && b[7] == b'-' && digits_at(8..10)) {
        return name;
    }
    if !matches!(b[10], b'-' | b'_') {
        return name;
    }
    let month: u32 = name[5..7].parse().unwrap_or(0);
    let day: u32 = name[8..10].parse().unwrap_or(0);
    if !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return name;
    }
    let rest = &name[11..];
    if rest.is_empty() {
        name
    } else {
        rest
    }
}
/// Map a content-relative path + front matter to the URL Zola will give
/// the page, replicating Zola's defaults: `path` override wins outright;
/// `slug` replaces the final segment; page bundles collapse (`foo/index.md`
/// → `/foo/`); every segment is slugified (`slugify.paths = "on"`).
///
/// Known limitations, documented rather than guessed at: multilingual
/// suffixes (`foo.fr.md`) and per-section path overrides in ancestor
/// `_index.md` files are not handled — neither occurs on this site.
fn url_for(rel: &Path, fm: &FrontMatter) -> String {
    if let Some(p) = &fm.path {
        let p = p.trim_matches('/');
        return if p.is_empty() {
            "/".to_string()
        } else {
            format!("/{p}/")
        };
    }
    let mut segs: Vec<String> = rel
        .with_extension("")
        .components()
        .map(|c| c.as_os_str().to_string_lossy().into_owned())
        .collect();
    if segs.last().is_some_and(|s| s == "index") {
        segs.pop();
    }
    if let Some(slug) = &fm.slug {
        match segs.last_mut() {
            Some(last) => *last = slug.clone(),
            None => segs.push(slug.clone()),
        }
    } else if let Some(last) = segs.last_mut() {
        // Zola parses a YYYY-MM-DD prefix as the page date and drops it
        // from the slug. Front-matter `date` overrides the date but NOT
        // the stripping — 2024-06-04-using-go-chromedp with
        // date = 2024-06-06 still lives at /blog/using-go-chromedp/.
        *last = strip_date_prefix(last).to_string();
    }
    let segs: Vec<String> = segs
        .iter()
        .map(|s| frontmatter::slugify(s))
        .filter(|s| !s.is_empty())
        .collect();
    if segs.is_empty() {
        "/".to_string()
    } else {
        format!("/{}/", segs.join("/"))
    }
}

// ---------------------------------------------------------------------
// Content walking
// ---------------------------------------------------------------------

struct Post {
    markdown: String,
    rel: PathBuf,
}

impl Post {
    fn slug_title(&self) -> String {
        self.rel
            .file_stem()
            .map(|s| s.to_string_lossy().replace(['-', '_'], " "))
            .unwrap_or_else(|| "untitled".into())
    }
}

/// Recursively collect .md files under content/. Skips _index.md (section
/// pages). Draft and in_search_index filtering happens at build time,
/// where the front matter is parsed.
fn collect_posts(root: &Path) -> Result<Vec<Post>> {
    let mut out = Vec::new();
    walk(root, root, &mut out)?;
    // Deterministic order → deterministic doc ids → byte-stable artifacts.
    out.sort_by(|a, b| a.rel.cmp(&b.rel));
    Ok(out)
}

fn walk(root: &Path, dir: &Path, out: &mut Vec<Post>) -> Result<()> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .with_context(|| format!("reading {}", dir.display()))?
        .collect::<std::io::Result<_>>()?;
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            walk(root, &path, out)?;
        } else if path.extension().is_some_and(|e| e == "md") {
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if name == "_index.md" {
                continue;
            }
            let rel = path.strip_prefix(root)?.to_path_buf();
            let markdown =
                fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))?;
            out.push(Post { markdown, rel });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fm() -> FrontMatter {
        FrontMatter::default()
    }

    #[test]
    fn plain_page_and_bundle_urls() {
        assert_eq!(url_for(Path::new("blog/foo.md"), &fm()), "/blog/foo/");
        assert_eq!(url_for(Path::new("blog/foo/index.md"), &fm()), "/blog/foo/");
    }

    #[test]
    fn filenames_are_slugified_like_zola() {
        assert_eq!(
            url_for(Path::new("blog/My Post.md"), &fm()),
            "/blog/my-post/"
        );
        assert_eq!(
            url_for(Path::new("blog/Café Crème/index.md"), &fm()),
            "/blog/cafe-creme/"
        );
    }

    #[test]
    fn slug_replaces_last_segment_path_wins_outright() {
        let mut f = fm();
        f.slug = Some("custom".into());
        assert_eq!(url_for(Path::new("blog/foo/index.md"), &f), "/blog/custom/");
        f.path = Some("/elsewhere/entirely".into());
        assert_eq!(
            url_for(Path::new("blog/foo/index.md"), &f),
            "/elsewhere/entirely/"
        );
    }

    #[test]
    fn nested_bundle_collapses_only_the_index() {
        assert_eq!(
            url_for(
                Path::new("labs/lost-tasks-on-server-crash/01-baseline/index.md"),
                &fm()
            ),
            "/labs/lost-tasks-on-server-crash/01-baseline/"
        );
    }

    #[test]
    fn date_prefixes_are_stripped_like_zola() {
        assert_eq!(
            url_for(
                Path::new("blog/2024-06-04-using-go-chromedp/index.md"),
                &fm()
            ),
            "/blog/using-go-chromedp/"
        );
        assert_eq!(
            url_for(Path::new("blog/2022-07-21_topic/index.md"), &fm()),
            "/blog/topic/"
        );
        // Not a date → not stripped.
        assert_eq!(
            url_for(Path::new("blog/2024-tbd-ai-landscape/index.md"), &fm()),
            "/blog/2024-tbd-ai-landscape/"
        );
        // slug wins outright, no date logic.
        let mut f = fm();
        f.slug = Some("custom".into());
        assert_eq!(
            url_for(Path::new("blog/2024-06-04-x/index.md"), &f),
            "/blog/custom/"
        );
    }
}
