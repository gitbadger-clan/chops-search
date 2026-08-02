// Page-side glue for chops-search. Two modes, chosen automatically:
//
// INLINE — the page already has #chops-input (a dedicated /search/ page).
// Used as-is.
//
// OVERLAY — no #chops-input on the page. The script builds its own dialog
// and opens it on Ctrl/Cmd-K, `/`, or a click on any [data-chops-open]
// element. This is what makes site-wide search one <script> tag in a base
// template instead of markup repeated on every page, and it's why the
// mode is detected rather than configured: a site with both a search page
// and a header shortcut gets the right behavior on each without a flag.
//
// The overlay DOM is built on first open, not at load. Most visits never
// search, and a script in <head> on every page should cost nothing until
// it's used — same reasoning as booting the worker on first focus rather
// than on load.
//
// Keyboard: ArrowUp/Down move the selection, Enter follows it, Escape
// closes and restores focus to whatever had it before. That's the ARIA
// combobox pattern; without it the overlay is a mouse-only feature, which
// for a keyboard-opened dialog is an odd thing to be.

let worker = null;
let ready = false;
let failed = false;
let gen = 0;
let debounceTimer = 0;
let rows = [];
let lastWords = [];
let selected = -1;

let input = null;
let resultsEl = null;
let modeEl = null;
let overlay = null;      // null in inline mode
let lastFocused = null;

const DEBOUNCE_MS = 120;
const SNIPPET_CHARS = 200;
const SNIPPET_LEAD = 50;
const BASE = '/search';

// ---- mode detection -------------------------------------------------

const existing = document.querySelector('#chops-input');
if (existing) {
  bindInline(existing);
} else {
  bindOverlayTriggers();
}

function bindInline(el) {
  input = el;
  resultsEl = document.querySelector('#chops-results');
  modeEl = document.querySelector('#chops-mode');
  wireInput();
  input.addEventListener('focus', boot, { once: true });
}

function bindOverlayTriggers() {
  document.addEventListener('keydown', (ev) => {
    const k = ev.key;
    const mod = ev.metaKey || ev.ctrlKey;
    if ((mod && (k === 'k' || k === 'K')) || (k === '/' && !mod && !isTyping(ev.target))) {
      ev.preventDefault();
      openOverlay();
    }
  });
  document.querySelectorAll('[data-chops-open]').forEach((el) => {
    el.addEventListener('click', (ev) => {
      ev.preventDefault();
      openOverlay();
    });
  });
}

/// `/` is a shortcut only when it isn't a character someone is typing.
function isTyping(el) {
  if (!el) return false;
  const tag = el.tagName;
  return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
}

// ---- overlay --------------------------------------------------------

function buildOverlay() {
  overlay = document.createElement('div');
  overlay.className = 'chops-overlay';
  overlay.setAttribute('role', 'dialog');
  overlay.setAttribute('aria-modal', 'true');
  overlay.setAttribute('aria-label', 'Search');
  overlay.hidden = true;

  const panel = document.createElement('div');
  panel.className = 'chops chops-panel';

  input = document.createElement('input');
  input.type = 'search';
  input.id = 'chops-input';
  input.placeholder = 'Search…';
  input.autocomplete = 'off';
  input.spellcheck = false;
  input.setAttribute('role', 'combobox');
  input.setAttribute('aria-expanded', 'false');
  input.setAttribute('aria-controls', 'chops-results');
  input.setAttribute('aria-autocomplete', 'list');

  modeEl = document.createElement('span');
  modeEl.id = 'chops-mode';
  modeEl.className = 'chops-mode';
  modeEl.setAttribute('aria-live', 'polite');

  resultsEl = document.createElement('ul');
  resultsEl.id = 'chops-results';
  resultsEl.className = 'chops-results';
  resultsEl.setAttribute('role', 'listbox');

  panel.append(input, modeEl, resultsEl);
  overlay.append(panel);
  document.body.appendChild(overlay);

  // Clicking the backdrop closes; clicking the panel must not.
  overlay.addEventListener('click', (ev) => {
    if (ev.target === overlay) closeOverlay();
  });
  wireInput();
}

function openOverlay() {
  if (failed) return;
  if (!overlay) buildOverlay();
  if (!overlay.hidden) return;
  lastFocused = document.activeElement;
  overlay.hidden = false;
  document.documentElement.classList.add('chops-open');
  input.focus();
  input.select();
  boot();
}

function closeOverlay() {
  if (!overlay || overlay.hidden) return;
  overlay.hidden = true;
  document.documentElement.classList.remove('chops-open');
  render([], true, '');
  // Returning focus is the part people notice only when it's missing:
  // without it, Escape drops the caret at the top of the document.
  lastFocused?.focus?.();
}

// ---- input + keyboard ------------------------------------------------

function wireInput() {
  input.addEventListener('input', () => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(fire, DEBOUNCE_MS);
  });

  input.addEventListener('keydown', (ev) => {
    switch (ev.key) {
      case 'Enter':
        if (selected >= 0 && rows[selected]) {
          ev.preventDefault();
          rows[selected].li.querySelector('a')?.click();
          return;
        }
        clearTimeout(debounceTimer);
        fire();
        break;
      case 'ArrowDown':
        ev.preventDefault();
        move(1);
        break;
      case 'ArrowUp':
        ev.preventDefault();
        move(-1);
        break;
      case 'Escape':
        if (overlay) {
          ev.preventDefault();
          closeOverlay();
        }
        break;
      case 'Tab':
        // A modal dialog keeps focus inside it. With one focusable
        // element that means holding focus on the input.
        if (overlay) ev.preventDefault();
        break;
    }
  });
}

function move(delta) {
  if (rows.length === 0) return;
  if (selected < 0) {
    select(delta > 0 ? 0 : rows.length - 1);
    return;
  }
  select((selected + delta + rows.length) % rows.length);
}

function select(i) {
  rows.forEach((r, n) => r.li.setAttribute('aria-selected', String(n === i)));
  selected = i;
  rows[i]?.li.scrollIntoView({ block: 'nearest' });
  // aria-activedescendant keeps the input as the focused element while
  // announcing the highlighted option — moving real focus into the list
  // would break typing.
  input.setAttribute('aria-activedescendant', rows[i] ? rows[i].li.id : '');
}

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

// ---- worker ----------------------------------------------------------

function boot() {
  if (worker) return;
  worker = new Worker(`${BASE}/search-worker.js`, { type: 'module' });
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
      else console.warn('chops-search:', msg.message);
    }
  };
  worker.postMessage({ type: 'init', base: BASE });
}

function fail(message) {
  failed = true;
  console.warn('chops-search: unavailable —', message);
  if (modeEl) modeEl.textContent = 'search unavailable';
  if (resultsEl) resultsEl.replaceChildren();
  rows = [];
  worker?.terminate();
  worker = null;
}

// ---- rendering -------------------------------------------------------

function render(results, semantic, queryText) {
  if (!resultsEl) return;
  lastWords = queryWords(queryText);
  selected = -1;
  rows = results.map(({ url, title }, i) => {
    const li = document.createElement('li');
    li.id = `chops-opt-${i}`;
    li.setAttribute('role', 'option');
    li.setAttribute('aria-selected', 'false');

    const a = document.createElement('a');
    a.href = url;
    a.textContent = title;
    li.appendChild(a);

    const snipEl = document.createElement('p');
    snipEl.className = 'chops-snippet';
    li.appendChild(snipEl);

    li.addEventListener('mousemove', () => select(i));
    return { li, snipEl };
  });
  resultsEl.replaceChildren(...rows.map((r) => r.li));
  resultsEl.dataset.mode = semantic ? 'hybrid' : 'keyword';
  input.setAttribute('aria-expanded', String(results.length > 0));
  input.setAttribute('aria-activedescendant', '');
  if (modeEl) {
    modeEl.textContent = results.length
      ? `${results.length} result${results.length === 1 ? '' : 's'} · ${semantic ? 'hybrid' : 'keyword only'
      }`
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
/// so what gets highlighted is what actually matched.
function queryWords(q) {
  return (q.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []).filter((w) => w.length > 1);
}

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

/// Text nodes and <mark> elements, never innerHTML: snippet text arrives
/// as bytes off the network, and one XSS in a search box outweighs every
/// ranking bug in this project.
function highlight(text, words) {
  if (words.length === 0) return [document.createTextNode(text)];
  const hay = text.toLowerCase();

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
