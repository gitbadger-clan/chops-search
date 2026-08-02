//! Build-time content preparation and chunking, kept pure so it's
//! trivially testable.
//!
//! The chunking matters more than you'd expect with a static model: the
//! embedding is the MEAN of token vectors, and averaging an entire post
//! into one vector yields "generic English prose about software". Rare,
//! distinctive words drown under hundreds of ordinary ones. ~600-char
//! chunks keep those signals sharp.
//!
//! Preparation removes everything that is markup rather than content —
//! each of these was observed polluting rankings, not hypothesized:
//!
//! - fenced code blocks (```/~~~): code identifiers belong to keyword
//!   search via inline code, not to the embedding mean
//! - Zola shortcode invocations {{ ... }} and {% ... %} (block shortcode
//!   BODIES are kept — they're content; only the delimiters go)
//! - markdown links/images: keep the link text and alt text, drop URLs
//!   ("https example com" as indexed terms is pure noise)
//! - inline HTML: tags and their attributes are dropped, element text is
//!   kept, and <style>/<script> contents are dropped wholesale. An
//!   `<input placeholder="...">` must contribute NOTHING — attribute
//!   text once made a page rank first for its own placeholder.
//!
//! Splitting guarantees no chunk exceeds ~target regardless of paragraph
//! structure: a single wall-of-text paragraph is split on sentence
//! boundaries (hard-cut only if it has none), because one oversized chunk
//! quietly reintroduces the diluted-mean problem chunking exists to solve.

/// Strip Zola TOML front matter (+++ ... +++) and everything non-content,
/// returning prose plus a best-effort title from `title = "..."`.
pub fn prepare_markdown(src: &str) -> (Option<String>, String) {
    let mut title = None;
    let mut body = src;

    // Front matter: must start the file.
    if let Some(rest) = body.strip_prefix("+++")
        && let Some(end) = rest.find("\n+++")
    {
        let fm = &rest[..end];
        for line in fm.lines() {
            let line = line.trim();
            if let Some(v) = line.strip_prefix("title") {
                let v = v.trim_start().strip_prefix('=').unwrap_or("").trim();
                let v = v.trim_matches('"').trim_matches('\'');
                if !v.is_empty() {
                    title = Some(v.to_string());
                }
            }
        }
        body = rest[end + 4..].trim_start_matches('\n');
    }

    // Drop fenced code blocks; strip heading markers but keep heading
    // text — headings are distinctive signal.
    let mut prose = String::with_capacity(body.len());
    let mut in_fence = false;
    for line in body.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with("```") || trimmed.starts_with("~~~") {
            in_fence = !in_fence;
            continue;
        }
        if in_fence {
            continue;
        }
        let line = trimmed.trim_start_matches('#').trim_start();
        prose.push_str(line);
        prose.push('\n');
    }

    // Order matters: style/script bodies first (so their contents never
    // reach the tag stripper as "text"), then shortcodes (so their
    // arguments never look like links), then links, then remaining tags.
    let prose = strip_spans(&prose, "<style", "</style>");
    let prose = strip_spans(&prose, "<script", "</script>");
    let prose = strip_spans(&prose, "{{", "}}");
    let prose = strip_spans(&prose, "{%", "%}");
    let prose = rewrite_links(&prose);
    let prose = strip_tags(&prose);

    (title, prose)
}

/// Remove every span from `open` through `close`, inclusive. An unclosed
/// span is removed to end of input. Matching is case-sensitive; lowercase
/// covers markdown-authored HTML in practice — an uppercase <STYLE> would
/// only leak its CSS text, not break anything.
fn strip_spans(text: &str, open: &str, close: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find(open) {
        out.push_str(&rest[..i]);
        match rest[i + open.len()..].find(close) {
            Some(j) => rest = &rest[i + open.len() + j + close.len()..],
            None => return out,
        }
    }
    out.push_str(rest);
    out
}

/// `[text](url)` → `text`, `![alt](url)` → `alt`. URLs never reach the
/// index. Bare brackets without a `](...)` are left alone.
fn rewrite_links(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    loop {
        let Some(i) = rest.find('[') else {
            out.push_str(rest);
            return out;
        };
        let is_image = rest[..i].ends_with('!');
        let Some(j) = rest[i..].find("](") else {
            // No link syntax after this bracket anywhere: emit through it.
            out.push_str(&rest[..=i]);
            rest = &rest[i + 1..];
            continue;
        };
        let j = i + j;
        let Some(k) = rest[j..].find(')') else {
            out.push_str(&rest[..=i]);
            rest = &rest[i + 1..];
            continue;
        };
        let k = j + k;
        out.push_str(&rest[..i - usize::from(is_image)]);
        out.push_str(&rest[i + 1..j]); // link text / alt text
        out.push(' ');
        rest = &rest[k + 1..];
    }
}

/// Drop `<...>` spans that look like tags (next char is a letter, `/`, or
/// `!`), replacing each with a space so words don't fuse. `a < b` and
/// `x <5` survive untouched.
fn strip_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('<') {
        let after = &rest[i + 1..];
        let looks_like_tag = after
            .chars()
            .next()
            .is_some_and(|c| c.is_ascii_alphabetic() || c == '/' || c == '!');
        if !looks_like_tag {
            out.push_str(&rest[..=i]);
            rest = after;
            continue;
        }
        match after.find('>') {
            Some(j) => {
                out.push_str(&rest[..i]);
                out.push(' ');
                rest = &after[j + 1..];
            }
            None => {
                // Dangling open tag: drop the remainder.
                out.push_str(&rest[..i]);
                return out;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Split prose into overlapping ~target_chars chunks on paragraph
/// boundaries. Overlap = the last piece of the previous chunk leads the
/// next one, so a sentence split across a boundary still lands whole in
/// at least one chunk. Paragraphs longer than target are pre-split on
/// sentence boundaries so no single paragraph can produce an oversized,
/// mean-diluted chunk.
pub fn chunk_prose(prose: &str, target_chars: usize) -> Vec<String> {
    let pieces: Vec<String> = prose
        .split("\n\n")
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .flat_map(|p| split_long_paragraph(p, target_chars))
        .collect();

    let mut chunks = Vec::new();
    let mut cur = String::new();
    let mut last_piece: Option<&str> = None;

    for p in &pieces {
        if !cur.is_empty() && cur.len() + p.len() > target_chars {
            chunks.push(core::mem::take(&mut cur));
            // overlap: seed next chunk with the previous piece
            if let Some(lp) = last_piece
                && lp.len() < target_chars
            {
                cur.push_str(lp);
                cur.push('\n');
            }
        }
        cur.push_str(p);
        cur.push('\n');
        last_piece = Some(p);
    }
    if !cur.trim().is_empty() {
        chunks.push(cur);
    }
    // Drop stub chunks that are pure boilerplate.
    chunks.retain(|c| c.trim().len() >= 40);
    chunks
}

/// A paragraph within target passes through unchanged. A longer one is
/// packed sentence-by-sentence into ~target pieces; a "sentence" that is
/// itself absurdly long (no boundaries — minified junk, giant URLs that
/// survived, non-Latin prose without ASCII stops) is hard-cut on char
/// boundaries rather than shipped whole.
fn split_long_paragraph(p: &str, target: usize) -> Vec<String> {
    if p.len() <= target {
        return vec![p.to_string()];
    }

    // Sentence boundaries: . ! ? followed by whitespace.
    let mut sentences: Vec<&str> = Vec::new();
    let mut last = 0;
    let mut iter = p.char_indices().peekable();
    while let Some((i, c)) = iter.next() {
        if matches!(c, '.' | '!' | '?')
            && let Some(&(_, next)) = iter.peek()
            && next.is_whitespace()
        {
            sentences.push(&p[last..i + c.len_utf8()]);
            last = i + c.len_utf8();
        }
    }
    if last < p.len() {
        sentences.push(&p[last..]);
    }

    let mut out = Vec::new();
    let mut cur = String::new();
    for s in sentences {
        let s = s.trim();
        if s.is_empty() {
            continue;
        }
        if !cur.is_empty() && cur.len() + s.len() > target {
            out.push(core::mem::take(&mut cur));
        }
        if s.len() > target {
            // Boundary-free run: hard cut, respecting char boundaries.
            if !cur.is_empty() {
                out.push(core::mem::take(&mut cur));
            }
            let mut piece = String::new();
            for ch in s.chars() {
                piece.push(ch);
                if piece.len() >= target {
                    out.push(core::mem::take(&mut piece));
                }
            }
            if !piece.is_empty() {
                out.push(piece);
            }
        } else {
            cur.push_str(s);
            cur.push(' ');
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn front_matter_and_fences_stripped() {
        let src = "+++\ntitle = \"Bloom Filters\"\ndate = 2026-01-01\n+++\n\nIntro paragraph about bloom filters and why they matter for search.\n\n```rust\nfn secret() {}\n```\n\nMore prose here that should definitely survive the stripping pass.\n";
        let (title, prose) = prepare_markdown(src);
        assert_eq!(title.as_deref(), Some("Bloom Filters"));
        assert!(prose.contains("Intro paragraph"));
        assert!(!prose.contains("fn secret"));
    }

    #[test]
    fn html_attributes_never_reach_the_index() {
        // The bug observed live: a placeholder made the search page rank
        // first for its own placeholder text.
        let src = "Real prose before.\n\n<div class=\"chops-lab\">\n<input id=\"x\" placeholder=\"losing tasks when the server dies\">\nvisible text survives\n</div>\n";
        let (_, prose) = prepare_markdown(src);
        assert!(!prose.contains("losing tasks"));
        assert!(!prose.contains("placeholder"));
        assert!(!prose.contains("chops-lab"));
        assert!(prose.contains("visible text survives"));
        assert!(prose.contains("Real prose before."));
    }

    #[test]
    fn style_and_script_bodies_dropped() {
        let src = "Before.\n<style>\n.x { color: red; }\n</style>\n<script src=\"/js/app.js\"></script>\nAfter.\n";
        let (_, prose) = prepare_markdown(src);
        assert!(!prose.contains("color"));
        assert!(!prose.contains("app.js"));
        assert!(prose.contains("Before."));
        assert!(prose.contains("After."));
    }

    #[test]
    fn shortcodes_removed_bodies_kept() {
        let src = "Watch {{ youtube(id=\"abc123\") }} for context.\n\n{% note() %}\nThe body of a block shortcode is content.\n{% end %}\n";
        let (_, prose) = prepare_markdown(src);
        assert!(!prose.contains("abc123"));
        assert!(!prose.contains("youtube"));
        assert!(prose.contains("Watch"));
        assert!(prose.contains("for context"));
        assert!(prose.contains("body of a block shortcode"));
    }

    #[test]
    fn links_keep_text_drop_urls_images_keep_alt() {
        let src = "Try [Pagefind](https://pagefind.app) today.\n\n![diagram of the queue](/img/queue.png)\n";
        let (_, prose) = prepare_markdown(src);
        assert!(prose.contains("Pagefind"));
        assert!(!prose.contains("pagefind.app"));
        assert!(!prose.contains("https"));
        assert!(prose.contains("diagram of the queue"));
        assert!(!prose.contains("queue.png"));
    }

    #[test]
    fn math_comparisons_survive_tag_stripping() {
        let src = "We need a < b here, and x <5 there, but <em>emphasis</em> is markup.\n";
        let (_, prose) = prepare_markdown(src);
        assert!(prose.contains("a < b"));
        assert!(prose.contains("x <5"));
        assert!(prose.contains("emphasis"));
        assert!(!prose.contains("<em>"));
    }

    #[test]
    fn chunks_respect_target_and_overlap() {
        let para = "word ".repeat(30); // ~150 chars
        let prose = format!("{p}\n\n{p}\n\n{p}\n\n{p}", p = para.trim());
        let chunks = chunk_prose(&prose, 320);
        assert!(chunks.len() >= 2);
        for c in &chunks {
            assert!(c.trim().len() >= 40);
        }
    }

    #[test]
    fn oversized_paragraph_is_split_on_sentences() {
        // One paragraph, ~5 sentences of ~120 chars: previously a single
        // 600-char chunk at target 300; now must split.
        let sentence = format!("{}.", "alpha beta gamma delta ".repeat(5).trim());
        let para = format!("{s} {s} {s} {s} {s}", s = sentence);
        let chunks = chunk_prose(&para, 300);
        assert!(chunks.len() >= 2, "expected a split, got {chunks:?}");
        for c in &chunks {
            // target + one overlap piece is the worst case; nothing near
            // the unsplit 600.
            assert!(c.len() <= 300 * 2, "oversized chunk: {} chars", c.len());
        }
    }

    #[test]
    fn boundary_free_run_is_hard_cut() {
        let para = "x".repeat(1000);
        let pieces = split_long_paragraph(&para, 300);
        assert!(pieces.len() >= 3);
        for p in &pieces {
            assert!(p.len() <= 300);
        }
    }

    #[test]
    fn empty_input_no_chunks() {
        assert!(chunk_prose("", 600).is_empty());
    }
}
