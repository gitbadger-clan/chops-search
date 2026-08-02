//! `chops-search model` — fetch and verify the embedding model.
//!
//! A CLI subcommand rather than a shell script because two audiences need
//! this and a script serves neither well: repo developers and CI need a
//! model for the parity tests and the demo build, and users of an
//! installed binary need one in their site. A script can't be shipped to
//! the second group at all, and portability across BSD/GNU `stat`,
//! `shasum`/`sha256sum`, and `jq` availability already caused problems
//! when it was one.
//!
//! `build` remains network-free — that was always the real invariant. It
//! reads a directory and nothing else, so a build cannot fail because an
//! upstream repo moved, went down, or changed a default branch. Fetching
//! is a separate, explicit act.
//!
//! The lockfile lives BESIDE the model directory (`model` →
//! `model.lock.json`), not inside it, so a site can gitignore the 30 MB
//! of weights while committing the 400 bytes that describe them. That
//! pairing — ignored payload, committed manifest — is what makes a build
//! reproducible without bloating anyone's repo.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sha2::{Digest, Sha256};

/// Files chops-search reads. `config.json` is needed only by the
/// model2vec-rs parity oracle, but it's ~1 KB and leaving it out makes
/// the ignored test fail confusingly.
const FILES: &[&str] = &["tokenizer.json", "model.safetensors", "config.json"];

/// `<dir>` → `<dir>.lock.json`, alongside rather than within.
pub fn lock_path(model_dir: &Path) -> PathBuf {
    let mut s = model_dir.as_os_str().to_os_string();
    s.push(".lock.json");
    PathBuf::from(s)
}

#[derive(Debug)]
pub struct Lock {
    pub repo: String,
    pub revision: String,
    pub files: Vec<(String, String, u64)>, // name, sha256, bytes
}

impl Lock {
    pub fn read(path: &Path) -> Result<Self> {
        let text =
            fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let v: serde_json::Value = serde_json::from_str(&text).context("parsing lockfile")?;
        let repo = v["repo"]
            .as_str()
            .context("lock has no `repo`")?
            .to_string();
        let revision = v["revision"]
            .as_str()
            .context("lock has no `revision`")?
            .to_string();
        let files = v["files"]
            .as_array()
            .context("lock has no `files` array")?
            .iter()
            .map(|f| {
                Ok((
                    f["name"]
                        .as_str()
                        .context("file entry has no name")?
                        .to_string(),
                    f["sha256"]
                        .as_str()
                        .context("file entry has no sha256")?
                        .to_string(),
                    f["bytes"].as_u64().context("file entry has no bytes")?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Lock {
            repo,
            revision,
            files,
        })
    }

    fn write(&self, path: &Path) -> Result<()> {
        let files: Vec<serde_json::Value> = self
            .files
            .iter()
            .map(|(n, s, b)| serde_json::json!({ "name": n, "sha256": s, "bytes": b }))
            .collect();
        let doc = serde_json::json!({
            "repo": self.repo,
            "revision": self.revision,
            "files": files,
        });
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(path, serde_json::to_vec_pretty(&doc)?)
            .with_context(|| format!("writing {}", path.display()))?;
        Ok(())
    }
}

/// Re-hash the files on disk against the lock. No network — this is what
/// CI runs to prove the model it's about to build with is the one the
/// lockfile describes.
pub fn verify(model_dir: &Path) -> Result<()> {
    let lock_file = lock_path(model_dir);
    let lock = Lock::read(&lock_file)?;
    println!("{} @ {}", lock.repo, lock.revision);

    let mut bad = Vec::new();
    for (name, want_sha, want_bytes) in &lock.files {
        let path = model_dir.join(name);
        let Ok(meta) = fs::metadata(&path) else {
            bad.push(format!("{name}: missing"));
            continue;
        };
        if meta.len() != *want_bytes {
            bad.push(format!(
                "{name}: {} bytes, expected {want_bytes}",
                meta.len()
            ));
            continue;
        }
        let got = hash_file(&path)?;
        if &got != want_sha {
            bad.push(format!("{name}: sha256 {got}, expected {want_sha}"));
        } else {
            println!("  ok  {name} ({} KB)", meta.len() / 1024);
        }
    }
    if !bad.is_empty() {
        for b in &bad {
            eprintln!("  bad {b}");
        }
        bail!(
            "model does not match {} — re-run `model fetch`",
            lock_file.display()
        );
    }
    Ok(())
}

/// Download a model and write the lockfile.
///
/// `revision` of None resolves the repo's default branch to a concrete
/// commit, so the lock always pins something immutable — "main" is not a
/// reproducible reference.
pub fn fetch(repo: &str, revision: Option<&str>, model_dir: &Path) -> Result<()> {
    let revision = match revision {
        Some(r) => r.to_string(),
        None => resolve_default_revision(repo)?,
    };
    println!("fetching {repo} @ {revision}");

    fs::create_dir_all(model_dir).with_context(|| format!("creating {}", model_dir.display()))?;

    let mut entries = Vec::new();
    for name in FILES {
        // The canonical raw-file endpoint; redirects to a CDN, which ureq
        // follows. Pinned by revision, so this URL is stable forever.
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{name}");
        let dest = model_dir.join(name);
        // Stream to a temp file rather than buffering: model.safetensors
        // is ~30 MB, and a partial download must never be mistaken for a
        // complete one by a later `verify`.
        let tmp = model_dir.join(format!(".{name}.part"));
        let bytes =
            download(&url, &tmp).with_context(|| format!("downloading {name} from {url}"))?;
        fs::rename(&tmp, &dest)?;
        let sha = hash_file(&dest)?;
        println!("  {name}: {} KB", bytes / 1024);
        entries.push((name.to_string(), sha, bytes));
    }

    let lock = Lock {
        repo: repo.to_string(),
        revision,
        files: entries,
    };
    let lock_file = lock_path(model_dir);
    lock.write(&lock_file)?;
    println!("wrote {}", lock_file.display());
    println!(
        "\ncommit the lockfile; the model directory itself is large and \
         belongs in .gitignore."
    );
    Ok(())
}

/// Ask HuggingFace which commit the default branch points at.
fn resolve_default_revision(repo: &str) -> Result<String> {
    let url = format!("https://huggingface.co/api/models/{repo}");
    let body = ureq::get(&url)
        .call()
        .with_context(|| format!("querying {url}"))?
        .into_body()
        .read_to_string()
        .context("reading model info")?;
    let v: serde_json::Value = serde_json::from_str(&body).context("parsing model info")?;
    // If HuggingFace changes this response shape, this is the line to fix:
    // the field is the commit sha of the repo's default branch.
    v["sha"]
        .as_str()
        .map(str::to_string)
        .with_context(|| format!("no `sha` in the response for {repo} — does it exist?"))
}

/// Stream a URL to `dest`, returning the byte count.
fn download(url: &str, dest: &Path) -> Result<u64> {
    let resp = ureq::get(url).call()?;
    let mut reader = resp.into_body().into_reader();
    let mut file =
        fs::File::create(dest).with_context(|| format!("creating {}", dest.display()))?;
    let n = io::copy(&mut reader, &mut file)?;
    Ok(n)
}

fn hash_file(path: &Path) -> Result<String> {
    use std::io::Read;

    let mut file = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut hasher = Sha256::new();
    // 64 KB at a time rather than io::copy: Digest's io::Write bridge
    // lives behind sha2's `std` feature and moved between 0.10 and 0.11,
    // and a 30 MB safetensors must not be buffered whole to hash it.
    let mut buf = vec![0u8; 64 * 1024];
    loop {
        let n = file.read(&mut buf)?;
        if n == 0 {
            break;
        }
        hasher.update(&buf[..n]);
    }
    Ok(hasher
        .finalize()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lock_sits_beside_the_model_dir() {
        assert_eq!(
            lock_path(Path::new(".chops-search/model")),
            PathBuf::from(".chops-search/model.lock.json")
        );
        assert_eq!(
            lock_path(Path::new("/abs/path/m")),
            PathBuf::from("/abs/path/m.lock.json")
        );
    }

    #[test]
    fn lock_round_trips() {
        let tmp = std::env::temp_dir().join(format!("chops-lock-{}", std::process::id()));
        fs::create_dir_all(&tmp).unwrap();
        let path = tmp.join("m.lock.json");
        let lock = Lock {
            repo: "minishlab/potion-base-8M".into(),
            revision: "abc123".into(),
            files: vec![("tokenizer.json".into(), "deadbeef".into(), 42)],
        };
        lock.write(&path).unwrap();
        let back = Lock::read(&path).unwrap();
        assert_eq!(back.repo, lock.repo);
        assert_eq!(back.revision, lock.revision);
        assert_eq!(back.files, lock.files);
        fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn verify_reports_a_mismatch_rather_than_panicking() {
        let tmp = std::env::temp_dir().join(format!("chops-ver-{}", std::process::id()));
        let model = tmp.join("model");
        fs::create_dir_all(&model).unwrap();
        fs::write(model.join("tokenizer.json"), b"actual contents").unwrap();
        Lock {
            repo: "x/y".into(),
            revision: "r".into(),
            // Right size, wrong hash — the case a length check alone misses.
            files: vec![("tokenizer.json".into(), "0".repeat(64), 15)],
        }
        .write(&lock_path(&model))
        .unwrap();
        assert!(verify(&model).is_err());
        fs::remove_dir_all(&tmp).ok();
    }
}
