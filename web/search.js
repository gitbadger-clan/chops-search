// Page-side glue for chops. Changes from the phase-0 baseline:
//
// Fix #2 — debounce: input events are coalesced (~120 ms) so a fast typist
// doesn't fire a query (and its range fetches) per keystroke. Enter always
// fires immediately.
//
// Fix #6 — boot failure is visible: if the worker script fails to load or
// boot() throws (missing artifacts, CSP blocking wasm, offline), the UI
// says so instead of being a silently dead input. Errors after a
// successful boot are non-fatal: keyword-only degradation already covers
// them, so they only warn on the console.

let worker = null;
let ready = false;
let failed = false;
let gen = 0;
let debounceTimer = 0;

const DEBOUNCE_MS = 120;

const input = document.querySelector('#chops-input');
const resultsEl = document.querySelector('#chops-results');
const modeEl = document.querySelector('#chops-mode');

input?.addEventListener('focus', boot, { once: true });

input?.addEventListener('input', () => {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(fire, DEBOUNCE_MS);
});

// Enter = "I'm done typing": skip the debounce window.
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
    render([], true);
    return;
  }
  gen += 1;
  worker.postMessage({ type: 'query', q, gen, limit: 8 });
}

function boot() {
  if (worker) return;
  worker = new Worker('/search/search-worker.js', { type: 'module' });

  // Fires when the worker script itself fails to load or parse — the
  // onmessage protocol never gets a chance to report this one.
  worker.addEventListener('error', () => fail('worker failed to load'));

  worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === 'ready') {
      ready = true;
      if (input.value.trim().length >= 2) fire();
    } else if (msg.type === 'results') {
      if (msg.gen !== gen) return; // stale
      render(msg.results, msg.semantic);
    } else if (msg.type === 'error') {
      if (!ready) {
        // Boot never completed: artifacts missing, wasm blocked by CSP,
        // offline first visit. Fatal for this session.
        fail(msg.message);
      } else {
        // Post-boot errors degrade to keyword-only inside the engine;
        // nothing to do here but note it.
        console.warn('chops:', msg.message);
      }
    }
  };
  worker.postMessage({ type: 'init', base: '/search' });
}

function fail(message) {
  failed = true;
  console.warn('chops: search unavailable —', message);
  if (modeEl) modeEl.textContent = 'search unavailable';
  if (resultsEl) resultsEl.replaceChildren();
  worker?.terminate();
  worker = null;
}

function render(results, semantic) {
  if (!resultsEl) return;
  resultsEl.replaceChildren(
    ...results.map(({ url, title }) => {
      const li = document.createElement('li');
      const a = document.createElement('a');
      a.href = url;
      a.textContent = title;
      li.appendChild(a);
      return li;
    })
  );
  resultsEl.dataset.mode = semantic ? 'hybrid' : 'keyword';
  if (modeEl) {
    modeEl.textContent = results.length
      ? (semantic ? 'hybrid (keyword + semantic)' : 'keyword only')
      : '';
  }
}
