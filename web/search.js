// Page-side glue: lazy-boot the worker on first focus of the search box
// so the page itself pays nothing, debounce input, render results.
// Adapt the selectors/markup to the tabi theme.

let worker = null;
let ready = false;
let gen = 0;

const input = document.querySelector('#search-input');
const resultsEl = document.querySelector('#search-results');

input?.addEventListener('focus', boot, { once: true });

input?.addEventListener('input', () => {
  const q = input.value.trim();
  if (!ready || q.length < 2) {
    render([], true);
    return;
  }
  gen += 1;
  worker.postMessage({ type: 'query', q, gen, limit: 8 });
});

function boot() {
  if (worker) return;
  worker = new Worker('/search/search-worker.js', { type: 'module' });
  worker.onmessage = (ev) => {
    const msg = ev.data;
    if (msg.type === 'ready') {
      ready = true;
      // If the user already typed while we were booting, run it now.
      if (input.value.trim().length >= 2) {
        gen += 1;
        worker.postMessage({ type: 'query', q: input.value.trim(), gen, limit: 8 });
      }
    } else if (msg.type === 'results') {
      if (msg.gen !== gen) return; // stale
      render(msg.results, msg.semantic);
    } else if (msg.type === 'error') {
      console.warn('search:', msg.message);
    }
  };
  worker.postMessage({ type: 'init', base: '/search' });
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
}
