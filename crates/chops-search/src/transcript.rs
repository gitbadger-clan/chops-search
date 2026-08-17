//! A transcript tee: everything a command prints also lands in a buffer
//! that can be saved to a file (`-O`) or pushed to the system clipboard
//! (`--clipboard`) when the run ends.
//!
//! Exists because a calibrate or sweep transcript is the deliverable —
//! the thing that gets diffed against the last run and quoted in a
//! commit — and `| tee` loses the exit status while `> file` loses the
//! live view of a walk that takes a while. The saved copy is prefixed
//! with the exact command line, so a transcript found in a directory
//! six weeks later states its own provenance: which fixture, which
//! base flags, which knob.
//!
//! Terminal output is unchanged: this writes to stdout exactly what it
//! captures. Streaming is preserved — nothing is held back until the
//! end except the file write and the clipboard push themselves.
//!
//! What it does NOT capture: anything printed by code that bypasses it.
//! In calibrate that is the `--explain` blocks, which go through
//! `explain::print_report` straight to stdout. Those stream to the
//! terminal only; the saved transcript notes where each would have
//! been so the gap is visible rather than silent.
//!
//! Clipboard is the platform tool, not a crate: pbcopy on macOS,
//! wl-copy or xclip or xsel on Linux, clip.exe on Windows. A GUI
//! clipboard crate would drag X11/Wayland client libraries into a CLI
//! that otherwise links nothing, and a build machine without a display
//! server should still be able to `cargo build`. Missing tool is an
//! error, not a silent no-op: `--clipboard` was an explicit request.

use anstyle::{AnsiColor, Effects, Style};
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use anyhow::{Context, Result, bail};

// The one palette, one meaning per colour, for every transcript this
// crate prints: green is a pass, a gain, or a keep; yellow is attention
// (top3, REVIEW) rather than failure; red is a fail, a loss, a casualty,
// or an instrument warning; bold is structure; dim is a note. Rendered
// only on a tty (anstream strips otherwise) and stripped from the -O /
// --clipboard copy, so saved transcripts stay byte-plain.
pub(crate) const GREEN: Style = AnsiColor::Green.on_default();
pub(crate) const YELLOW: Style = AnsiColor::Yellow.on_default();
pub(crate) const RED: Style = AnsiColor::Red.on_default().bold();
pub(crate) const HEADING: Style = Style::new().bold();
pub(crate) const NOTE: Style = Style::new().effects(Effects::DIMMED);

pub struct Transcript {
    /// None when neither sink is requested: then `line` is just
    /// println! and nothing is buffered.
    capture: Option<String>,
    output: Option<PathBuf>,
    clipboard: bool,
}

impl Transcript {
    pub fn new(output: Option<&Path>, clipboard: bool) -> Self {
        let capture = if output.is_some() || clipboard {
            // First line of the saved copy: the invocation, so the file
            // is self-describing without a shell history to consult.
            let argv: Vec<String> = std::env::args().collect();
            Some(format!("command: {}\n\n", argv.join(" ")))
        } else {
            None
        };
        Transcript {
            capture,
            output: output.map(Path::to_path_buf),
            clipboard,
        }
    }

    /// One line, newline appended — the println! replacement.
    pub fn line(&mut self, s: impl AsRef<str>) {
        let s = s.as_ref();
        anstream::println!("{s}");
        if let Some(buf) = &mut self.capture {
            buf.push_str(&anstream::adapter::strip_str(s).to_string());
            buf.push('\n');
        }
    }

    /// Raw text, no newline appended — the print! replacement, for
    /// pre-rendered blocks that carry their own line breaks.
    pub fn raw(&mut self, s: impl AsRef<str>) {
        let s = s.as_ref();
        anstream::print!("{s}");
        if let Some(buf) = &mut self.capture {
            buf.push_str(&anstream::adapter::strip_str(s).to_string());
        }
    }

    /// Record, in the saved copy only, that something printed here that
    /// this transcript did not capture. Terminal readers saw it; file
    /// readers get a placeholder instead of a silent gap.
    pub fn note_uncaptured(&mut self, what: &str) {
        if let Some(buf) = &mut self.capture {
            buf.push_str(&format!("[{what}: streamed to terminal, not captured]\n"));
        }
    }

    /// Flush the sinks. On an error the partial transcript is still
    /// saved, with a trailer naming the failure, so six knobs of
    /// measurement do not vanish because the seventh hit a bad row —
    /// and the file cannot be mistaken for a complete run.
    pub fn finish(mut self, outcome: &Result<()>) -> Result<()> {
        if let (Some(buf), Err(e)) = (&mut self.capture, outcome) {
            buf.push_str(&format!("\n[run aborted: {e:#}]\n"));
        }
        let Some(buf) = self.capture else {
            return Ok(());
        };
        if let Some(path) = &self.output {
            fs::write(path, &buf).with_context(|| format!("writing {}", path.display()))?;
            eprintln!("transcript: {} ({} bytes)", path.display(), buf.len());
        }
        if self.clipboard {
            copy_to_clipboard(&buf)?;
            eprintln!("transcript: copied to clipboard ({} bytes)", buf.len());
        }
        Ok(())
    }
}

/// Push text to the system clipboard through the platform's tool.
fn copy_to_clipboard(text: &str) -> Result<()> {
    let candidates: Vec<(&str, &[&str])> = if cfg!(target_os = "macos") {
        vec![("pbcopy", &[])]
    } else if cfg!(target_os = "windows") {
        vec![("clip.exe", &[])]
    } else {
        // Wayland first when a compositor is present; X tools otherwise.
        let mut v: Vec<(&str, &[&str])> = Vec::new();
        if std::env::var_os("WAYLAND_DISPLAY").is_some() {
            v.push(("wl-copy", &[]));
        }
        v.push(("xclip", &["-selection", "clipboard"]));
        v.push(("xsel", &["--clipboard", "--input"]));
        v
    };

    let mut tried = Vec::new();
    for (tool, argv) in &candidates {
        tried.push(*tool);
        let child = Command::new(tool)
            .args(*argv)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn();
        let mut child = match child {
            Ok(c) => c,
            // Not installed: try the next one.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => continue,
            Err(e) => return Err(e).with_context(|| format!("spawning {tool}")),
        };
        child
            .stdin
            .take()
            .context("clipboard tool has no stdin")?
            .write_all(text.as_bytes())
            .with_context(|| format!("writing to {tool}"))?;
        let status = child
            .wait()
            .with_context(|| format!("waiting for {tool}"))?;
        if status.success() {
            return Ok(());
        }
        bail!("{tool} exited with {status}");
    }
    bail!(
        "no clipboard tool found (tried {}); use -O FILE instead",
        tried.join(", ")
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nothing_requested_captures_nothing() {
        let mut t = Transcript::new(None, false);
        t.line("hello");
        assert!(t.capture.is_none());
    }

    #[test]
    fn capture_mirrors_lines_and_raw_exactly() {
        let mut t = Transcript::new(Some(Path::new("/dev/null")), false);
        t.line("a");
        t.raw("b\nc");
        t.line("");
        let buf = t.capture.as_ref().unwrap();
        assert!(buf.starts_with("command: "), "{buf}");
        assert!(buf.ends_with("\n\na\nb\nc\n"), "{buf:?}");
    }

    #[test]
    fn uncaptured_marker_lands_only_in_the_copy() {
        let mut t = Transcript::new(Some(Path::new("/dev/null")), false);
        t.note_uncaptured("explain \"x\"");
        assert!(
            t.capture
                .as_ref()
                .unwrap()
                .contains("[explain \"x\": streamed")
        );
        // And is a no-op when nothing is being captured.
        let mut n = Transcript::new(None, false);
        n.note_uncaptured("explain \"x\"");
        assert!(n.capture.is_none());
    }

    #[test]
    fn finish_writes_the_file() {
        let dir = std::env::temp_dir().join(format!("chops-transcript-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.txt");
        let mut t = Transcript::new(Some(&path), false);
        t.line("row 1");
        t.finish(&Ok(())).unwrap();
        let s = fs::read_to_string(&path).unwrap();
        assert!(s.contains("row 1"));
        assert!(s.starts_with("command: "));
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn the_copy_is_stripped_of_styles() {
        let mut t = Transcript::new(Some(Path::new("/dev/null")), false);
        t.line("\u{1b}[32mkeep 0.08\u{1b}[0m (plateau)");
        assert!(
            t.capture
                .as_ref()
                .unwrap()
                .ends_with("keep 0.08 (plateau)\n")
        );
    }

    #[test]
    fn an_aborted_run_still_saves_with_a_trailer() {
        let dir =
            std::env::temp_dir().join(format!("chops-transcript-abort-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.txt");
        let mut t = Transcript::new(Some(&path), false);
        t.line("knob one done");
        t.finish(&Err(anyhow::anyhow!("bad row"))).unwrap();
        let s = fs::read_to_string(&path).unwrap();
        assert!(s.contains("knob one done"));
        assert!(s.ends_with("[run aborted: bad row]\n"), "{s:?}");
        fs::remove_dir_all(&dir).unwrap();
    }
}
