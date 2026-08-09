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
//! `model.safetensors`. `chops-search model fetch` puts one there. No
//! network access in `build`, deliberately, so a build can never fail
//! because an upstream repo moved.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::{CommandFactory, Parser, Subcommand};
use sha2::{Digest, Sha256};

use chops_search::assets;
use chops_search::completion;
use chops_search::config::Config;
use chops_search::frontmatter::{self, FrontMatter};
use chops_search::model_loader::load_model2vec;
use chops_search::pca::pca_reduce;
use chops_search_core::builder::{
    embed_f32, frequency_permutation, permute_rows_f32, quantize_global, quantize_rows,
};
use chops_search_core::chunk::{chunk_prose, prepare_markdown};
use chops_search_core::format::{Doc, Index, ModelMeta, Posting};
use chops_search_core::keyword::{FieldWeights, keyword_words};
use chops_search_core::wordpiece::Vocab;

const AFTER_HELP: &str = "\
EXAMPLES
  chops-search init                     scaffold a Zola site
  chops-search model fetch              download the embedding model
  chops-search build                    artifacts + runtime -> static/search/

  chops-search docs                     list indexed URLs
  chops-search query \"why this rank\"     explain a ranking
  chops-search eval --fail-under 0.85   gate on recall@1

  COMPLETE=fish chops-search | source   completions for this session

Docs: https://github.com/gitbadger-clan/chops-search";

#[derive(Parser)]
#[command(
    name = "chops-search",
    version,
    about = "Hybrid keyword + semantic search for static sites",
    long_about = "\
Builds a browser-side search index for a static site: BM25F over keyword \
postings (title, tags, description, and body scored as separate fields) \
fused with cosine similarity over model2vec embeddings, served as static \
files and queried without a server.

Run `chops-search init` in a Zola site to get started.",
    after_help = AFTER_HELP
)]
struct Cli {
    /// Site directory to resolve chops-search.toml from.
    ///
    /// Defaults to the working directory, searching upward the way cargo
    /// finds Cargo.toml. Useful from a repo root, or in CI.
    #[arg(long, global = true, value_name = "DIR", add = completion::site_candidates())]
    site: Option<PathBuf>,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum ModelCmd {
    /// Download a model2vec model and write the lockfile.
    #[command(long_about = "\
Downloads a model2vec model and records exactly what landed in a lockfile \
beside the model directory.

Commit the lockfile and gitignore the model itself: the lock is what makes \
a build reproducible, and the weights are ~30 MB.")]
    Fetch {
        /// HuggingFace repo.
        ///
        /// Any model2vec model works; the listed ones are tested.
        #[arg(
            default_value = "minishlab/potion-base-8M",
            value_name = "REPO",
            add = completion::model_candidates()
        )]
        repo: String,
        /// Exact revision to pin.
        ///
        /// Defaults to the repo's current default branch, resolved to a
        /// commit so the lockfile pins something immutable.
        #[arg(long, value_name = "SHA")]
        revision: Option<String>,
        /// Destination. Default: `model` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
    },
    /// Re-hash the model against its lockfile. No network.
    #[command(long_about = "\
Verifies the model on disk matches the committed lockfile.

Catches a partial download, a corrupted blob, or a model swapped without \
updating the lock. Worth running in CI before a build, since the failure \
it prevents (building against different weights than the lock claims) \
surfaces days later as a confusing recall change.")]
    Verify {
        /// Model directory. Default: `model` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        dir: Option<PathBuf>,
    },
}

#[derive(Subcommand)]
enum Cmd {
    /// Scaffold a site: config, search page, .gitignore entries.
    #[command(long_about = "\
Writes chops-search.toml, a /search/ page, and .gitignore entries, then \
prints the template snippet for a site-wide search box.

Nothing is ever overwritten. Re-running after editing the scaffold is \
safe and reports what it skipped.")]
    Init {
        /// Skip content/search.md.
        ///
        /// For sites using only the site-wide overlay, where a dedicated
        /// search page would be redundant.
        #[arg(long)]
        no_page: bool,
    },

    /// Download or verify the embedding model.
    #[command(long_about = "\
Fetches a model2vec model and records what landed in a lockfile.

This is the only command that touches the network. `build` reads a \
directory and nothing else, so a build can never fail because an upstream \
repo moved or went down.")]
    Model {
        #[command(subcommand)]
        action: ModelCmd,
    },

    /// Build search artifacts and the browser runtime.
    #[command(long_about = "\
Reads the content tree and the model, writes hashed artifacts plus the \
wasm engine, worker, page script, and stylesheet into the output \
directory.

Content changes touch only index.bin and snippets.bin. The model files \
change when the model does, and the wasm caches across every deploy.

The BM25F field weights from chops-search.toml are written into \
index.bin, so the browser scores with what the site configured. Sweep \
them with `eval --w-title` / `--w-tag` against a built index; only the \
committed value needs a rebuild.")]
    Build {
        /// Content directory. Default: `content` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        content: Option<PathBuf>,
        /// Model directory. Default: `model` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        model: Option<PathBuf>,
        /// Output directory. Default: `out` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        out: Option<PathBuf>,
        /// Rows bundled eagerly, covering most queries without a fetch.
        ///
        /// Default 2048. Larger means a bigger eager payload and fewer
        /// range requests.
        #[arg(long, value_name = "N")]
        prefix_rows: Option<u32>,
        /// Target chunk size in characters. Default 600.
        ///
        /// Smaller chunks sharpen rare-word signal and cost more vectors.
        #[arg(long, value_name = "N")]
        chunk_chars: Option<usize>,
        /// PCA target dimensionality. Default 128; native size is 256.
        ///
        /// Halves the eager prefix and every range fetch, at some cost in
        /// recall. Re-run `eval` after changing it.
        #[arg(long, value_name = "N")]
        dims: Option<usize>,
        /// Artifacts only; skip the wasm and JS runtime.
        ///
        /// For CI jobs that rebuild an index without shipping a new
        /// frontend.
        #[arg(long)]
        no_runtime: bool,
    },

    /// List indexed documents and their URLs.
    #[command(long_about = "\
Prints every indexed document with its URL and chunk count.

The URLs are what `eval` expectations must match, so this is the command \
to run after adding a post. A mistyped expectation reads as a ranking \
failure rather than a typo, which is a bad afternoon.")]
    Docs {
        /// Artifacts directory. Default: `out` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        artifacts: Option<PathBuf>,
    },

    /// Explain why a query ranked the way it did.
    #[command(long_about = "\
Prints the evidence behind a ranking: how the query tokenized on both \
sides, per-term keyword scores with document frequencies and per-field \
term frequencies, best-chunk cosine per document, and each engine's \
contribution to the fused order.

This is the diagnostic tool. When a result looks wrong, the answer is \
almost always visible in the chunk count or in which field the term \
turned up in.")]
    Query {
        /// The query string.
        #[arg(value_name = "QUERY")]
        query: String,
        /// Artifacts directory. Default: `out` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        artifacts: Option<PathBuf>,
        /// Rows to print. Default 20.
        #[arg(long, value_name = "N", default_value_t = 20)]
        limit: usize,

        /// Minimum keyword-confidence ratio. Default 0.30.
        ///
        /// For diagnosing the keyword evidence gate: below it the
        /// keyword list is suppressed from fusion. 0 disables.
        #[arg(long, value_name = "FRACTION")]
        kw_floor: Option<f32>,
        /// BM25F weight on title matches. Default: whatever index.bin
        /// was built with.
        ///
        /// For asking whether a result still wins without its title:
        /// 0 ignores the field entirely.
        #[arg(long, value_name = "WEIGHT")]
        w_title: Option<f32>,
        /// BM25F weight on tag matches. Default: whatever index.bin was
        /// built with.
        ///
        /// Tags are the author's own statement of what a page is about,
        /// so they normally outweigh the title.
        #[arg(long, value_name = "WEIGHT")]
        w_tag: Option<f32>,
        /// BM25F weight on front-matter description matches. Default:
        /// whatever index.bin was built with.
        ///
        /// For asking whether a result is riding on its description
        /// rather than its prose: 0 ignores the field entirely.
        #[arg(long, value_name = "WEIGHT")]
        w_desc: Option<f32>,
        /// Minimum top-median cosine contrast for an uncorroborated
        /// semantic list. Default 0 (disabled).
        ///
        /// The corroboration gate: when the keyword side contributed
        /// nothing and no document stands out from the corpus, the
        /// ranking is noise rather than an answer.
        #[arg(long, value_name = "GAP")]
        min_gap: Option<f32>,
        /// Best-chunk cosine at or above which the gate never fires.
        /// Default off.
        ///
        /// Escape hatch for broad-but-real queries. Disables at
        /// infinity, not at 0 — 0 would disable the gate, not the hatch.
        #[arg(long, value_name = "COS")]
        strong_cos: Option<f32>,
        /// How much a confident keyword list outvotes the semantic one
        /// in fusion. Default: whatever the engine ships (0, plain RRF).
        ///
        /// For diagnosing a result that both engines ranked differently:
        /// the keyword list fuses at 1 + alpha × confidence. At the
        /// default rrf_k of 60 it needs to reach about 2 before it can
        /// overturn a semantic first place; a smaller --rrf-k moves that
        /// crossover down. 0 is plain RRF.
        #[arg(long, value_name = "ALPHA")]
        rrf_alpha: Option<f32>,
        /// RRF rank discount k. Default: whatever the engine ships (60).
        ///
        /// For diagnosing fusion resolution: the conventional 60 is
        /// nearly flat across a dozen-deep list, so at corpus scale a
        /// decisive #1 in one engine can lose to mediocre presence in
        /// both. A shape parameter with no disabling value — smaller
        /// sharpens the top of the curve.
        #[arg(long, value_name = "K")]
        rrf_k: Option<f32>,
    },

    /// Score the engine against a labelled query set.
    #[command(long_about = "\
Runs a labelled query set through the real engine over the real byte path \
(plan, range-fetch, ingest, search) and reports recall@1 and recall@3 by \
query kind.

Because it drives the actual loading logic, a bug in range planning shows \
up as a recall drop rather than hiding. The reported bytes-per-query are \
the real ones.

With --sweep-rrf-k / --sweep-rrf-alpha the case set runs once per point \
of the k × alpha grid and prints recall per cell instead of per-case \
lines; every other scoring flag acts as the fixed base for every cell, \
and the best cell is re-run for its per-kind breakdown and failures.")]
    Eval {
        /// Artifacts directory. Default: `out` from chops-search.toml.
        #[arg(long, value_name = "DIR")]
        artifacts: Option<PathBuf>,
        /// Query set. Default: fixtures/queries.toml beside the config.
        #[arg(long, value_name = "FILE")]
        queries: Option<PathBuf>,
        /// Only run cases of this kind.
        ///
        /// Candidates are read from the query set, so they reflect
        /// whatever kinds it actually uses.
        #[arg(long, value_name = "KIND", add = completion::kind_candidates())]
        kind: Option<String>,
        /// Exit non-zero below this recall@1 fraction. Default 0.0.
        ///
        /// Set it just under your current baseline in CI: high enough to
        /// catch a regression, low enough that one flipped case in a
        /// small set does not fail the build. Ignored in sweep mode.
        #[arg(long, value_name = "FRACTION", default_value_t = 0.0)]
        fail_under: f32,
        /// Minimum best-chunk cosine for semantic relevance. Default 0.20.
        ///
        /// For sweeping the relevance floor. Below it a document counts
        /// as unrelated, which is what makes empty results possible.
        #[arg(long, value_name = "COS")]
        min_cos: Option<f32>,
        /// Minimum keyword-confidence ratio. Default 0.30.
        ///
        /// For sweeping the keyword evidence gate. Below it the keyword
        /// engine submits nothing to fusion: a ranking assembled from
        /// stopword coincidences or a lone prefix expansion is worse
        /// than no ranking. 0 disables the gate.
        #[arg(long, value_name = "FRACTION")]
        kw_floor: Option<f32>,
        /// BM25F weight on title matches. Default: whatever index.bin
        /// was built with.
        ///
        /// For sweeping. Under BM25F a title hit is already normalized
        /// by the title's own length, so the useful range is small and
        /// well under the old pre-multiplied weight. 0 ignores titles.
        #[arg(long, value_name = "WEIGHT")]
        w_title: Option<f32>,
        /// BM25F weight on tag matches. Default: whatever index.bin was
        /// built with.
        ///
        /// For sweeping. Tags are short and hand-picked, so they carry
        /// more per occurrence than anything else on the page.
        #[arg(long, value_name = "WEIGHT")]
        w_tag: Option<f32>,
        /// BM25F weight on front-matter description matches. Default:
        /// whatever index.bin was built with.
        ///
        /// For sweeping. Descriptions are author-written summaries in
        /// the register searchers phrase questions in, but they are also
        /// low-variance across a corpus; 0 answers whether yours earn
        /// their keep.
        #[arg(long, value_name = "WEIGHT")]
        w_desc: Option<f32>,
        /// Coefficient on the chunk-count correction. Default 0.02.
        ///
        /// For sweeping. Longer documents get more chances at a high
        /// max-pooled score; this corrects the bias. 0 disables it.
        #[arg(long, value_name = "COEFF")]
        chunk_penalty: Option<f32>,
        /// Minimum top-median cosine contrast for an uncorroborated
        /// semantic list. Default 0 (disabled).
        ///
        /// The corroboration gate: when the keyword side contributed
        /// nothing and no document stands out from the corpus, the
        /// ranking is noise rather than an answer.
        #[arg(long, value_name = "GAP")]
        min_gap: Option<f32>,
        /// Best-chunk cosine at or above which the gate never fires.
        /// Default off.
        ///
        /// Escape hatch for broad-but-real queries: a flat field whose
        /// best document is strongly relevant is not noise. Disables at
        /// infinity, not at 0 — 0 would disable the gate, not the hatch.
        #[arg(long, value_name = "COS")]
        strong_cos: Option<f32>,
        /// How much a confident keyword list outvotes the semantic one
        /// in fusion. Default 0, which is plain RRF.
        ///
        /// For sweeping. Plain RRF resolves a keyword first place
        /// against a semantic first place by rank arithmetic alone,
        /// which is backwards for exact-match queries; this scales the
        /// keyword list by 1 + alpha × confidence, so only queries with
        /// real keyword evidence get the louder vote. At the default
        /// rrf_k of 60 values below ~1 are inert — the curve is that
        /// flat — and a smaller k moves the crossover down, so sweep
        /// the two jointly.
        #[arg(long, value_name = "ALPHA")]
        rrf_alpha: Option<f32>,
        /// RRF rank discount k. Default 60, the deep-list convention.
        ///
        /// For pinning a swept value. A shape parameter with no
        /// disabling value: at corpus scale the conventional 60 is
        /// nearly flat across a dozen-deep list (rank 1 vs rank 2 is
        /// 1/61 vs 1/62), so fusion degenerates toward best-average-rank.
        /// Smaller sharpens the top of the curve.
        #[arg(long, value_name = "K")]
        rrf_k: Option<f32>,
        /// Comma-separated rrf_k values to sweep, e.g. 2,4,8,16,32,60.
        ///
        /// Enables sweep mode: the case set runs once per grid point.
        /// Combine with --sweep-rrf-alpha for the full k × alpha grid;
        /// alone, alpha stays at its base value.
        #[arg(long, value_name = "LIST", value_delimiter = ',')]
        sweep_rrf_k: Vec<f32>,
        /// Comma-separated rrf_alpha values to sweep, e.g. 0,0.5,1,2.
        ///
        /// Enables sweep mode: the case set runs once per grid point.
        /// Combine with --sweep-rrf-k for the full k × alpha grid;
        /// alone, k stays at its base value.
        #[arg(long, value_name = "LIST", value_delimiter = ',')]
        sweep_rrf_alpha: Vec<f32>,
    },

    /// Emit a shell completion script.
    #[command(long_about = "\
Writes a conventional completion script to stdout.

Prefer the dynamic path where your shell supports it, since it computes \
candidates at completion time (so `--kind` lists the kinds your query set \
actually uses):

  fish:  echo 'COMPLETE=fish chops-search | source' >> ~/.config/fish/config.fish
  zsh:   echo 'source <(COMPLETE=zsh chops-search)' >> ~/.zshrc
  bash:  echo 'source <(COMPLETE=bash chops-search)' >> ~/.bashrc")]
    Completions {
        /// Shell to generate for.
        #[arg(value_name = "SHELL")]
        shell: clap_complete::Shell,
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
    // Must run before Cli::parse. When the shell invokes us for
    // completions this exits early, so a half-typed command line never
    // reaches clap as a parse error.
    clap_complete::CompleteEnv::with_factory(Cli::command).complete();

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
            let cfg = load_config(&site)?.with_overrides(
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
            kw_floor,
            w_title,
            w_tag,
            w_desc,
            min_gap,
            strong_cos,
            rrf_alpha,
            rrf_k,
        } => {
            let cfg = load_config(&site)?;
            let dir = artifacts.unwrap_or(cfg.out);
            let args = chops_search::eval::ScoreArgs {
                kw_floor,
                w_title,
                w_tag,
                w_desc,
                min_gap,
                strong_cos,
                rrf_alpha,
                rrf_k,
                ..Default::default()
            };
            chops_search::explain::explain(&dir, &query, limit, args)
        }
        Cmd::Eval {
            artifacts,
            queries,
            kind,
            fail_under,
            min_cos,
            chunk_penalty,
            kw_floor,
            w_title,
            w_tag,
            w_desc,
            min_gap,
            strong_cos,
            rrf_alpha,
            rrf_k,
            sweep_rrf_k,
            sweep_rrf_alpha,
        } => {
            let cfg = load_config(&site)?;
            let dir = artifacts.unwrap_or_else(|| cfg.out.clone());
            let queries = queries.unwrap_or_else(|| cfg.root.join("fixtures/queries.toml"));
            let args = chops_search::eval::ScoreArgs {
                min_cos,
                chunk_penalty,
                kw_floor,
                min_gap,
                strong_cos,
                w_title,
                w_tag,
                w_desc,
                rrf_alpha,
                rrf_k,
            };
            chops_search::eval::eval(
                &dir,
                &queries,
                kind.as_deref(),
                fail_under,
                args,
                &sweep_rrf_k,
                &sweep_rrf_alpha,
            )
        }
        Cmd::Model { action } => {
            let cfg = load_config(&site)?;
            match action {
                ModelCmd::Fetch {
                    repo,
                    revision,
                    dir,
                } => chops_search::model::fetch(
                    &repo,
                    revision.as_deref(),
                    &dir.unwrap_or(cfg.model),
                ),
                ModelCmd::Verify { dir } => chops_search::model::verify(&dir.unwrap_or(cfg.model)),
            }
        }
        Cmd::Docs { artifacts } => {
            let cfg = load_config(&site)?;
            chops_search::explain::list_docs(&artifacts.unwrap_or(cfg.out))
        }
        Cmd::Init { no_page } => {
            let root = match &site {
                Some(p) => p.clone(),
                None => std::env::current_dir()?,
            };
            chops_search::init::init(&root, !no_page)
        }
        Cmd::Completions { shell } => {
            completion::generate(shell, &mut Cli::command());
            eprintln!("{}", completion::install_hint(shell));
            Ok(())
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

/// Which BM25F field a run of words belongs to.
#[derive(Debug, Clone, Copy)]
enum Field {
    Title,
    Tag,
    /// Zola's front-matter `description`. Its own field rather than body
    /// text: counting it as body inflated dl_body, so a fuller
    /// description quietly discounted every other term on the page.
    Desc,
    Body,
}

/// Per-field term frequencies for one term in one document. Raw counts,
/// deliberately unweighted: the field weights apply at query time, after
/// each field has been normalized by its own length. Pre-multiplying here
/// (what this used to do) pushed a weighted title tf past k1's saturation
/// point, so a title mention behaved like keyword stuffing.
#[derive(Debug, Default, Clone, Copy)]
struct FieldTf {
    title: u16,
    tag: u16,
    desc: u16,
    body: u16,
}

impl FieldTf {
    fn add(&mut self, field: Field) {
        let slot = match field {
            Field::Title => &mut self.title,
            Field::Tag => &mut self.tag,
            Field::Desc => &mut self.desc,
            Field::Body => &mut self.body,
        };
        // A term appearing 65k times in one body is absurd but not worth
        // failing a build over.
        *slot = slot.saturating_add(1);
    }
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
    let mut doc_words: Vec<HashMap<String, FieldTf>> = Vec::new();

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
        // pre-WordPiece), counted per FIELD at raw frequency. Which field
        // matters more is a query-time question — BM25F normalizes each
        // field by its own length before the weights apply, and neither
        // the lengths nor the weights are known here.
        // keyword_words only emits alphanumeric runs, so no filter here.
        let mut tf: HashMap<String, FieldTf> = HashMap::new();
        let mut count_words = |text: &str, field: Field| {
            let norm = Vocab::normalize(text);
            for w in keyword_words(&norm) {
                tf.entry(w.to_string()).or_default().add(field);
            }
        };
        count_words(&title, Field::Title);
        if let Some(desc) = &fm.description {
            count_words(desc, Field::Desc);
        }
        for tag in &fm.tags {
            count_words(tag, Field::Tag);
        }
        count_words(&prose, Field::Body);
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
    let mut postings: HashMap<String, Vec<Posting>> = HashMap::new();
    for (doc_id, tf) in doc_words.iter().enumerate() {
        for (term, f) in tf {
            postings.entry(term.clone()).or_default().push(Posting {
                doc: doc_id as u16,
                title: f.title,
                tag: f.tag,
                desc: f.desc,
                body: f.body,
            });
        }
    }
    let mut terms: Vec<(String, Vec<Posting>)> = postings.into_iter().collect();
    terms.sort_by(|a, b| a.0.cmp(&b.0)); // byte-stable output
    for (_, p) in &mut terms {
        // By doc id: Posting has no Ord (the four tf fields have no
        // meaningful order between them), and doc id is the only key the
        // reader cares about anyway.
        p.sort_unstable_by_key(|p| p.doc);
    }
    eprintln!(
        "keyword: {} terms, BM25F weights title {:.2}, tag {:.2}, desc {:.2}",
        terms.len(),
        cfg.title_weight,
        cfg.tag_weight,
        cfg.desc_weight
    );

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
        // Baked in so the browser scores with the site's configuration;
        // eval and query can still override per run without a rebuild.
        weights: FieldWeights {
            title: cfg.title_weight,
            tag: cfg.tag_weight,
            desc: cfg.desc_weight,
        },
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
    if rest.is_empty() { name } else { rest }
}

/// Map a content-relative path + front matter to the URL Zola will give
/// the page, replicating Zola's defaults: `path` override wins outright;
/// `slug` replaces the final segment; page bundles collapse (`foo/index.md`
/// → `/foo/`); every segment is slugified (`slugify.paths = "on"`).
///
/// Known limitations, documented rather than guessed at: multilingual
/// suffixes (`foo.fr.md`) and per-section path overrides in ancestor
/// `_index.md` files are not handled.
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

    /// Catches conflicting arg names, bad defaults, and missing value
    /// names. Cheap insurance around the completion work, which touches
    /// every argument declaration.
    #[test]
    fn cli_definition_is_valid() {
        Cli::command().debug_assert();
    }

    #[test]
    fn field_counts_land_in_their_own_slots() {
        // The builder's whole job on the keyword side: a term in the
        // title and the body must arrive as two separate tfs, not one
        // pre-multiplied number.
        let mut tf = FieldTf::default();
        tf.add(Field::Title);
        tf.add(Field::Desc);
        tf.add(Field::Body);
        tf.add(Field::Body);
        assert_eq!(tf.title, 1);
        assert_eq!(tf.tag, 0);
        assert_eq!(tf.desc, 1);
        assert_eq!(tf.body, 2);
    }

    #[test]
    fn every_field_has_its_own_slot() {
        // Four same-typed counters behind one `add`: a mismatched match
        // arm would be invisible except here.
        for (field, pick) in [
            (Field::Title, 0usize),
            (Field::Tag, 1),
            (Field::Desc, 2),
            (Field::Body, 3),
        ] {
            let mut tf = FieldTf::default();
            tf.add(field);
            let got = [tf.title, tf.tag, tf.desc, tf.body];
            for (i, v) in got.iter().enumerate() {
                assert_eq!(*v, u16::from(i == pick), "{field:?} landed in slot {i}");
            }
        }
    }

    #[test]
    fn field_counts_saturate() {
        let mut tf = FieldTf {
            body: u16::MAX,
            ..Default::default()
        };
        tf.add(Field::Body);
        assert_eq!(tf.body, u16::MAX, "must not wrap to zero");
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
