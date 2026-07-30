// Web Worker: the dumb byte pump. All decisions live in the wasm engine;
// this file fetches, ingests, renders nothing, and never touches vectors.
//
// Change from upstream (fix #7): on a 206, ingest at the offset the server
// SAYS it sent (Content-Range) rather than the offset we asked for. They
// agree on every sane server; when a misbehaving proxy rewrites the range,
// trusting the response header turns silent corruption into either correct
// ingestion or a loud row-alignment error from the engine.
//
// Protocol (postMessage):
//   in : { type: 'init', base }            base = '/search'
//   in : { type: 'query', q, gen, limit }  gen = monotonically increasing
//   out: { type: 'ready' }
//   out: { type: 'results', gen, semantic, results: [{ url, title }] }
//   out: { type: 'error', message }

import init, { Engine } from './pkg/chops_wasm.js';

let engine = null;
let base = '/search';
let latestGen = 0;

self.onmessage = async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      base = msg.base ?? base;
      await boot();
      self.postMessage({ type: 'ready' });
    } else if (msg.type === 'query') {
      latestGen = msg.gen;
      await query(msg.q, msg.gen, msg.limit ?? 8);
    }
  } catch (e) {
    self.postMessage({ type: 'error', message: String(e) });
  }
};

async function boot() {
  // wasm-bindgen glue + module. instantiateStreaming inside init() needs
  // Content-Type: application/wasm from the host, and CSP must allow
  // 'wasm-unsafe-eval' — both are deployment config, see README.
  await init();

  const [meta, index, prefix] = await Promise.all([
    fetchBytes(`${base}/model.meta.bin`),
    fetchBytes(`${base}/index.bin`),
    fetchBytes(`${base}/model.prefix.i8`),
  ]);
  engine = new Engine(meta, index);
  engine.ingest(0, prefix);
}

// "Content-Range: bytes 512-1023/9469952" → 512. Null when absent or in a
// shape we don't understand (e.g. "bytes */N" for unsatisfiable ranges).
function contentRangeStart(response) {
  const m = /^bytes (\d+)-\d+\/(?:\d+|\*)$/.exec(
    response.headers.get('Content-Range') ?? ''
  );
  return m ? Number(m[1]) : null;
}

async function query(q, gen, limit) {
  if (!engine) return;

  // 1. Ask Rust which bytes it needs. Flat [start, end, start, end, ...].
  const plan = engine.plan(q);

  // 2. Parallel single-range requests over HTTP/2. Deliberately not one
  //    multi-range request: multipart/byteranges handling in fetch() is
  //    inconsistent across servers and browsers.
  if (plan.length > 0) {
    const jobs = [];
    for (let i = 0; i < plan.length; i += 2) {
      const start = plan[i];
      const end = plan[i + 1]; // half-open; Range header is inclusive
      jobs.push(
        fetch(`${base}/model.rows.i8`, {
          headers: { Range: `bytes=${start}-${end - 1}` },
        }).then(async (r) => {
          if (r.status === 206) {
            // Ingest at the offset the server declares. Fall back to the
            // requested start only if Content-Range is missing/unparseable.
            const declared = contentRangeStart(r);
            engine.ingest(declared ?? start, new Uint8Array(await r.arrayBuffer()));
          } else if (r.ok) {
            // 200 = server ignored Range and sent the whole file; ingest it
            // all at offset 0 — wasteful once, then everything is loaded.
            engine.ingest(0, new Uint8Array(await r.arrayBuffer()));
          } else {
            throw new Error(`range fetch failed: ${r.status}`);
          }
        })
      );
    }
    try {
      await Promise.all(jobs);
    } catch {
      // Offline / range-hostile host: fall through. search() degrades to
      // keyword-only on its own and reports it via `semantic: false`.
    }
  }

  // 3. Stale-query guard: if the user kept typing while we fetched, drop
  //    this result instead of reordering under their cursor. The rows we
  //    ingested stay warm for the next keystroke.
  if (gen !== latestGen) return;

  const ids = engine.search(q, limit);
  const results = [];
  for (const id of ids) {
    results.push({ url: engine.doc_url(id), title: engine.doc_title(id) });
  }
  self.postMessage({
    type: 'results',
    gen,
    semantic: engine.used_semantic(),
    results,
  });
}

async function fetchBytes(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return new Uint8Array(await r.arrayBuffer());
}
