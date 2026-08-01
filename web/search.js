// Page-side glue for chops. Changes from v2: renders snippets.
//
// Snippets arrive as a SECOND message after the results, so the list
// renders immediately and then fills in — same progressive pattern as
// keyword-then-semantic. Result rows are kept as DOM references so the
// snippet pass mutates them in place instead of re-rendering (which would
// flash and would drop focus if the user were arrowing through results).
//
// Highlighting builds text nodes and <mark> elements directly. Never
// innerHTML: snippet text is site content, but it reaches here as bytes
// off the network, and one XSS in a search box is worse than every
// ranking bug in this repo combined.

let worker = null;
let ready = false;
let failed = false;
let gen = 0;
let debounceTimer = 0;
let rows = [];        // { li, snipEl } per rendered result
let lastWords = [];   // query words for the currently rendered results

const DEBOUNCE_MS = 120;
/// Characters of context shown around the first matched term.
const SNIPPET_CHARS = 200;
/// How far before the match the window starts, so the term isn't flush left.
const SNIPPET_LEAD = 50;

const input = document.querySelector('#chops-input');
const resultsEl = document.querySelector('#chops-results');
const modeEl = document.querySelector('#chops-mode');

input?.addEventListener('focus', boot, { once: true });

input?.addEventListener('input', () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fire, DEBOUNCE_MS);
});

input?.addEventListener('keydown', (ev) => {
  if (ev.key === 'Enter') {
    clearTimeout(debounceTimer);
    fire();
  }
});

function fire() {
  if (failed) return;
  const q = input.value.trim();
  if (!ready || q.length < 2) {
    render([], true, q);
    return;
  }
  gen += 1;
  worker.postMessage({ type: 'query', q, gen, limit: 8 });
}

function boot() {
  if (worker) return;
  worker = new Worker('/search/search-worker.js', { type: 'module' });
  worker.addEventListener('error', () => fail('worker failed to load'));

  worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === 'ready') {
      ready = true;
      if (input.value.trim().length >= 2) fire();
    } else if (msg.type === 'results') {
      if (msg.gen !== gen) return;
      render(msg.results, msg.semantic, input.value.trim());
    } else if (msg.type === 'snippets') {
      if (msg.gen !== gen) return;
      applySnippets(msg.snippets);
    } else if (msg.type === 'error') {
      if (!ready) fail(msg.message);
      else console.warn('chops:', msg.message);
    }
  };
  worker.postMessage({ type: 'init', base: '/search' });
}

function fail(message) {
  failed = true;
  console.warn('chops: search unavailable —', message);
  if (modeEl) modeEl.textContent = 'search unavailable';
  if (resultsEl) resultsEl.replaceChildren();
  rows = [];
  worker?.terminate();
  worker = null;
}

function render(results, semantic, queryText) {
  if (!resultsEl) return;
  lastWords = queryWords(queryText);
  rows = results.map(({ url, title }) => {
    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = url;
    a.textContent = title;
    li.appendChild(a);
    // Placeholder for the snippet pass; stays empty if snippets never
    // arrive, so nothing shifts when they don't.
    const snipEl = document.createElement('p');
    snipEl.className = 'chops-snippet';
    li.appendChild(snipEl);
    return { li, snipEl };
  });
  resultsEl.replaceChildren(...rows.map((r) => r.li));
  resultsEl.dataset.mode = semantic ? 'hybrid' : 'keyword';
  if (modeEl) {
    modeEl.textContent = results.length
      ? semantic
        ? 'hybrid (keyword + semantic)'
        : 'keyword only'
      : '';
  }
}

function applySnippets(snippets) {
  snippets.forEach((text, i) => {
    const row = rows[i];
    if (!row || !text) return;
    row.snipEl.replaceChildren(...highlight(window_(text, lastWords), lastWords));
  });
}

/// Alphanumeric runs, lowercased — the same split the keyword index uses,
/// so what gets highlighted is what actually matched. Accents are NOT
/// stripped here while the index does strip them, so "café" highlights
/// only on an exact-accent match; a cosmetic gap, not a ranking one.
function queryWords(q) {
  return (q.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []).filter((w) => w.length > 1);
}

/// Trim the chunk to a readable window around the first matched term,
/// snapping to word boundaries so it doesn't start mid-word.
function window_(text, words) {
  const flat = text.replace(/\s+/g, ' ').trim();
  if (flat.length <= SNIPPET_CHARS) return flat;

  const hay = flat.toLowerCase();
  let at = -1;
  for (const w of words) {
    const i = hay.indexOf(w);
    if (i !== -1 && (at === -1 || i < at)) at = i;
  }
  if (at === -1) return snapEnd(flat.slice(0, SNIPPET_CHARS)) + '…';

  let start = Math.max(0, at - SNIPPET_LEAD);
  if (start > 0) {
    const sp = flat.indexOf(' ', start);
    if (sp !== -1 && sp < start + 20) start = sp + 1;
  }
  const slice = flat.slice(start, start + SNIPPET_CHARS);
  return (start > 0 ? '…' : '') + snapEnd(slice) + (start + SNIPPET_CHARS < flat.length ? '…' : '');
}

function snapEnd(s) {
  const sp = s.lastIndexOf(' ');
  return sp > s.length - 20 ? s.slice(0, sp) : s;
}

/// Build alternating text nodes and <mark> elements. Returns nodes, never
/// a string — the caller inserts them with replaceChildren.
function highlight(text, words) {
  if (words.length === 0) return [document.createTextNode(text)];
  const hay = text.toLowerCase();

  // Collect non-overlapping match spans, longest word first so "chromium"
  // doesn't pre-empt "chromiumoxide".
  const spans = [];
  for (const w of [...words].sort((a, b) => b.length - a.length)) {
    let from = 0;
    for (; ;) {
      const i = hay.indexOf(w, from);
      if (i === -1) break;
      const end = i + w.length;
      if (!spans.some((s) => i < s.end && end > s.start)) spans.push({ start: i, end });
      from = end;
    }
  }
  spans.sort((a, b) => a.start - b.start);

  const nodes = [];
  let cursor = 0;
  for (const s of spans) {
    if (s.start > cursor) nodes.push(document.createTextNode(text.slice(cursor, s.start)));
    const mark = document.createElement('mark');
    mark.textContent = text.slice(s.start, s.end);
    nodes.push(mark);
    cursor = s.end;
  }
  if (cursor < text.length) nodes.push(document.createTextNode(text.slice(cursor)));
  return nodes;
}
