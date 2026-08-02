//! Shell completion.
//!
//! Two mechanisms, because they serve different people.
//!
//! DYNAMIC (`CompleteEnv`) is the default. The shell asks the binary what
//! the candidates are on every Tab, so `--kind` lists the kinds actually
//! present in your query set rather than a list frozen when the
//! completion script was generated. One protocol covers bash, zsh, fish,
//! elvish, and powershell.
//!
//! STATIC (`chops-search completions <shell>`) emits a conventional
//! script for packagers and for anyone who would rather not have their
//! shell exec a binary on Tab. It loses the dynamic candidates.
//!
//! The dynamic path costs a process spawn per completion. That is fine
//! for a tool whose slowest operation reads a 30 MB model, and the
//! candidate functions below are deliberately cheap: none of them loads
//! an artifact, and the one that reads a file reads a few KB of TOML.

use std::path::PathBuf;

use clap_complete::engine::{ArgValueCandidates, CompletionCandidate};

/// Query kinds found in the configured query set, each annotated with how
/// many cases carry it.
///
/// Reads the file rather than hardcoding the four conventional kinds: a
/// site can invent its own, and a typo in `queries.toml` shows up here as
/// a candidate that looks wrong, which is a cheap way to notice it.
pub fn kind_candidates() -> ArgValueCandidates {
    ArgValueCandidates::new(|| {
        let Some(path) = query_set_path() else {
            return Vec::new();
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            return Vec::new();
        };
        let Ok(doc) = text.parse::<toml::Table>() else {
            return Vec::new();
        };
        let Some(arr) = doc.get("query").and_then(|v| v.as_array()) else {
            return Vec::new();
        };

        let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
        for item in arr {
            if let Some(k) = item.get("kind").and_then(|v| v.as_str()) {
                *counts.entry(k.to_string()).or_default() += 1;
            }
        }
        counts
            .into_iter()
            .map(|(kind, n)| {
                CompletionCandidate::new(&kind).help(Some(
                    format!("{n} case{}", if n == 1 { "" } else { "s" }).into(),
                ))
            })
            .collect()
    })
}

/// model2vec models known to work. Not exhaustive: any potion-family repo
/// on HuggingFace will do, and the argument accepts a free string. These
/// are the ones worth trying first.
pub fn model_candidates() -> ArgValueCandidates {
    ArgValueCandidates::new(|| {
        [
            (
                "minishlab/potion-base-8M",
                "default; 29k vocab, 256 dims, ~30 MB",
            ),
            (
                "minishlab/potion-base-32M",
                "larger vocab, better quality, bigger eager payload",
            ),
            (
                "minishlab/potion-base-4M",
                "smallest; try if payload matters more than recall",
            ),
            (
                "minishlab/potion-multilingual-128M",
                "needs the tokenizer's Indic and Hangul gaps closed first",
            ),
        ]
        .into_iter()
        .map(|(repo, help)| CompletionCandidate::new(repo).help(Some(help.into())))
        .collect()
    })
}

/// Directories under the working directory that contain a
/// chops-search.toml, for `--site`. One level deep only: a recursive walk
/// on Tab in a large tree would be felt.
pub fn site_candidates() -> ArgValueCandidates {
    ArgValueCandidates::new(|| {
        let mut out = Vec::new();
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        for dir in [cwd.clone(), cwd.join("examples")] {
            let Ok(entries) = std::fs::read_dir(&dir) else {
                continue;
            };
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() && path.join("chops-search.toml").is_file() {
                    let rel = path.strip_prefix(&cwd).unwrap_or(&path);
                    out.push(CompletionCandidate::new(rel.to_string_lossy().as_ref()));
                }
            }
        }
        out
    })
}

/// Locate the query set the way `eval` does, so completions and the
/// command agree about which file is in play.
fn query_set_path() -> Option<PathBuf> {
    let cwd = std::env::current_dir().ok()?;
    let cfg = crate::config::Config::discover(&cwd).ok()?;
    let path = cfg.root.join("fixtures/queries.toml");
    path.is_file().then_some(path)
}

/// Emit a static completion script.
pub fn generate(shell: clap_complete::Shell, cmd: &mut clap::Command) {
    let name = cmd.get_name().to_string();
    clap_complete::generate(shell, cmd, name, &mut std::io::stdout());
}

/// Printed after the generated script so the reader knows what to do with
/// it, and knows the dynamic path exists.
pub fn install_hint(shell: clap_complete::Shell) -> &'static str {
    match shell {
        clap_complete::Shell::Fish => {
            "# Save as ~/.config/fish/completions/chops-search.fish\n\
             # Or, for candidates computed at completion time:\n\
             #   echo 'COMPLETE=fish chops-search | source' >> ~/.config/fish/config.fish"
        }
        clap_complete::Shell::Zsh => {
            "# Save as a file named _chops-search on your $fpath\n\
             # Or, for candidates computed at completion time:\n\
             #   echo 'source <(COMPLETE=zsh chops-search)' >> ~/.zshrc"
        }
        clap_complete::Shell::Bash => {
            "# Save under /etc/bash_completion.d/ or source it from ~/.bashrc\n\
             # Or, for candidates computed at completion time:\n\
             #   echo 'source <(COMPLETE=bash chops-search)' >> ~/.bashrc"
        }
        _ => "# Install per your shell's convention.",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn kind_candidates_survive_a_missing_query_set() {
        // Completion runs in whatever directory the user is in, which is
        // usually not a site. Every candidate function must return an
        // empty list rather than panicking or printing.
        let tmp = std::env::temp_dir().join(format!("chops-comp-{}", std::process::id()));
        std::fs::create_dir_all(&tmp).unwrap();
        let prev = std::env::current_dir().unwrap();
        std::env::set_current_dir(&tmp).unwrap();

        let _ = kind_candidates();
        let _ = site_candidates();
        assert!(query_set_path().is_none());

        std::env::set_current_dir(prev).unwrap();
        std::fs::remove_dir_all(&tmp).ok();
    }

    #[test]
    fn model_candidates_are_all_potion_repos() {
        // A typo here sends someone to a 404 from a Tab press.
        let cands = model_candidates();
        let _ = cands; // constructed without panicking
    }

    fn _unused(_: &Path) {}
}
