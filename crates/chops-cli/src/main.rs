//! chops build — turn a model2vec model + a Zola content tree into the
//! four static artifacts the browser engine consumes:
//!
//!   search/model.meta.bin   vocab + per-row scales (eager, complete)
//!   search/model.prefix.i8  top-frequency rows (eager)
//!   search/model.rows.i8    full matrix, frequency-ordered (range-fetched)
//!   search/index.bin        chunk vectors + docs + keyword postings (eager)
//!
//! Expects a local model directory containing `tokenizer.json` and
//! `model.safetensors` (e.g. `hf download minishlab/potion-base-8M
//! --local-dir model/`). No network access here; fetching the model is
//! your job, deliberately.
//!
//! Front matter is parsed for real (see chops_cli::frontmatter): drafts
//! and `in_search_index = false` pages are skipped, `slug`/`path`
//! overrides and Zola's path slugification shape URLs, and tags feed both
//! ranking sides — as high-weight keyword terms, and inside a synthetic
//! title+tags chunk so the semantic side finally sees titles at all
//! (body chunks alone never contain them).

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use clap::{Parser, Subcommand};

use chops_cli::frontmatter::{self, FrontMatter};
use chops_cli::model_loader::load_model2vec;
use chops_cli::pca::pca_reduce;
use chops_core::builder::{
    embed_f32, frequency_permutation, permute_rows_f32, quantize_global, quantize_rows,
};
use chops_core::chunk::{chunk_prose, prepare_markdown};
use chops_core::format::{Doc, Index, ModelMeta};
use chops_core::wordpiece::Vocab;

/// Extra term-frequency weight for title words (a title mention counts
/// like this many body mentions) and for tag words. Tags are weighted
/// hardest: they're the author's hand-curated statement of what the page
/// is about, and a 4-doc corpus already showed the cost of ignoring them
/// ("chaos" lived only in a tag and matched nothing).
const TITLE_WEIGHT: u16 = 2;
const TAG_WEIGHT: u16 = 4;

#[derive(Parser)]
#[command(name = "chops", about = "Static-site hybrid search index builder")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Build all search artifacts from a content tree and a model2vec model.
    Build {
        /// Zola content directory (walked recursively for .md files).
        #[arg(long)]
        content: PathBuf,
        /// Directory containing tokenizer.json and model.safetensors.
        #[arg(long)]
        model: PathBuf,
        /// Output directory (e.g. static/search).
        #[arg(long)]
        out: PathBuf,
        /// Rows bundled eagerly in model.prefix.i8.
        #[arg(long, default_value_t = 2048)]
        prefix_rows: u32,
        /// Target chunk size in characters.
        #[arg(long, default_value_t = 600)]
        chunk_chars: usize,
        /// Reduce embedding dimensionality via PCA before quantization
        /// (e.g. 128 halves the shipped matrix). Defaults to the model's
        /// native dimensionality.
        #[arg(long)]
        dims: Option<usize>,
    },
    /// Explain a query against built artifacts: keyword scores, best-chunk
    /// cosines, chunk counts, and RRF contributions per document.
    Query {
        /// Directory containing the built artifacts (the --out of `build`).
        #[arg(long, default_value = "../static/search")]
        artifacts: PathBuf,
        /// Max rows to print.
        #[arg(long, default_value_t = 20)]
        limit: usize,
        /// The query string.
        query: String,
    },
    /// Score the engine against a labeled query set (recall@1 by kind).
    Eval {
        /// Directory containing the built artifacts.
        #[arg(long, default_value = "../static/search")]
        artifacts: PathBuf,
        /// Labeled query set.
        #[arg(long, default_value = "fixtures/queries.toml")]
        queries: PathBuf,
        /// Only run cases of this kind (exact, paraphrase, navigational, negative).
        #[arg(long)]
        kind: Option<String>,
        /// Exit non-zero if overall recall@1 falls below this fraction (0.0–1.0).
        #[arg(long, default_value_t = 0.0)]
        fail_under: f32,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Cmd::Build {
            content,
            model,
            out,
            prefix_rows,
            chunk_chars,
            dims,
        } => build(&content, &model, &out, prefix_rows, chunk_chars, dims),
        Cmd::Query {
            artifacts,
            limit,
            query,
        } => chops_cli::explain::explain(&artifacts, &query, limit),
        Cmd::Eval {
            artifacts,
            queries,
            kind,
            fail_under,
        } => chops_cli::eval::eval(&artifacts, &queries, kind.as_deref(), fail_under),
    }
}

fn build(
    content: &Path,
    model_dir: &Path,
    out: &Path,
    prefix_rows: u32,
    chunk_chars: usize,
    dims: Option<usize>,
) -> Result<()> {
    // ---- 1. Load model -------------------------------------------------
    let (tokens, mut rows, mut dim) = load_model2vec(model_dir)?;
    let n_rows = tokens.len();
    eprintln!("model: {n_rows} tokens × {dim} dims");

    // Optional PCA reduction. Must happen before anything downstream so
    // chunk vectors and browser-side query vectors live in the same space.
    if let Some(k) = dims {
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
    let posts = collect_posts(content)?;
    if posts.is_empty() {
        bail!("no .md files found under {}", content.display());
    }

    struct ChunkRec {
        doc: u16,
        ids: Vec<u32>, // token ids under the ORIGINAL vocab order
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
        let mut tf: HashMap<String, u16> = HashMap::new();
        let mut count_words = |text: &str, weight: u16| {
            let norm = Vocab::normalize(text);
            for w in Vocab::words(&norm) {
                if w.chars().any(|c| c.is_alphanumeric()) {
                    let e = tf.entry(w.to_string()).or_insert(0);
                    *e = e.saturating_add(weight);
                }
            }
        };
        count_words(&title, TITLE_WEIGHT);
        for tag in &fm.tags {
            count_words(tag, TAG_WEIGHT);
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
        texts.extend(chunk_prose(&prose, chunk_chars));

        for text in &texts {
            let ids = vocab.tokenize(text);
            if !ids.is_empty() {
                chunks.push(ChunkRec { doc: doc_id, ids });
            }
        }
        docs.push(Doc { url, title });
    }
    if docs.is_empty() {
        bail!("every page under {} was skipped", content.display());
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
    for c in &chunks {
        // ids are non-empty by construction → embed only returns None for
        // a zero-norm mean, which we skip rather than ship.
        if let Some(v) = embed_f32(&c.ids, &rows, dim) {
            chunk_vecs_f32.extend(v);
            chunk_doc.push(c.doc);
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
    let prefix_rows = prefix_rows.min(n_rows as u32);
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

    fs::create_dir_all(out)?;
    let rows_bytes: Vec<u8> = data_i8.iter().map(|&v| v as u8).collect();
    let prefix_bytes = &rows_bytes[..prefix_rows as usize * dim];

    write_report(out, "model.meta.bin", &meta.write())?;
    write_report(out, "model.prefix.i8", prefix_bytes)?;
    write_report(out, "model.rows.i8", &rows_bytes)?;
    write_report(out, "index.bin", &index.write())?;
    Ok(())
}

fn write_report(out: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let path = out.join(name);
    fs::write(&path, bytes).with_context(|| format!("writing {}", path.display()))?;
    eprintln!("wrote {} ({} KB)", path.display(), bytes.len() / 1024);
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
