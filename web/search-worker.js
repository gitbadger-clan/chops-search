// Web Worker: the dumb byte pump. All decisions live in the wasm engine;
// this file fetches, ingests, renders nothing, and never touches vectors.
//
// Changes from v3:
//
// HASHED ARTIFACTS. Filenames now carry a build hash, read from
// manifest.json, so every byte under /search/ can be served immutable.
// That includes the wasm: the glue is dynamically imported with ?v=<hash>
// and the binary URL is passed to init() explicitly, which closes the
// version-skew hole where a cached old engine meets fresh artifacts.
// manifest.json and this file are the only things that revalidate.
//
// Protocol (postMessage):
//   in : { type: 'init', base }            base = '/search'
//   in : { type: 'query', q, gen, limit }  gen = monotonically increasing
//   out: { type: 'ready' }
//   out: { type: 'results', gen, semantic, results: [{ url, title }] }
//   out: { type: 'error', message }

let engine = null;
let base = '/search';
let latestGen = 0;
let rowCache = null;

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
  // The manifest is the only unhashed fetch, so it's the only one that
  // costs a round trip on a warm visit. Everything after it is immutable
  // and can come straight from disk cache.
  const manifest = await (await fetchOk(`${base}/manifest.json`)).json();
  if (manifest.version !== 1) {
    throw new Error(`manifest version ${manifest.version} unsupported`);
  }
  const hash = manifest.hash;
  files = manifest.files;

  // Dynamic import so the glue URL can carry the build hash — a static
  // top-level import is fetched before any of our code runs, which would
  // leave the engine on a revalidate-only cache policy.
  const glue = await import(`./pkg/chops_wasm.js?v=${hash}`);
  const initWasm = glue.default;
  const Engine = glue.Engine;

  const [, meta, index, prefix] = await Promise.all([
    // Newer wasm-bindgen wants the object form; older versions took a
    // bare URL here. Passing it explicitly is what lets the .wasm be
    // versioned rather than fetched by its unhashed default name.
    initWasm({ module_or_path: `${base}/pkg/chops_wasm_bg.wasm?v=${hash}` }),
    fetchEager(`${base}/${files.meta}`),
    fetchEager(`${base}/${files.index}`),
    // prefix rows are int8 and near-incompressible; no .gz is emitted.
    fetchBytes(`${base}/${files.prefix}`),
  ]);

  engine = new Engine(meta, index);
  engine.ingest(0, prefix);
}

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r;
}

async function fetchBytes(url) {
  return new Uint8Array(await (await fetchOk(url)).arrayBuffer());
}

/// Fetch an eager artifact, preferring the gzip sibling.
async function fetchEager(url) {
  if (!CAN_DECOMPRESS) return fetchBytes(url);
  const r = await fetch(`${url}.gz`);
  if (!r.ok) return fetchBytes(url); // no .gz deployed; don't fail to boot
  return gunzip(new Uint8Array(await r.arrayBuffer()));
}

/// Decompress if the bytes actually start with the gzip magic. Sniffing
/// rather than assuming matters because some hosts serve .gz files with
/// `Content-Encoding: gzip`, in which case the browser has already
/// decompressed the body and handing it to DecompressionStream would
/// throw on valid data.
async function gunzip(buf) {
  if (buf.length < 2 || buf[0] !== 0x1f || buf[1] !== 0x8b) return buf;
  const stream = new Blob([buf])
    .stream()
    .pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

// "Content-Range: bytes 512-1023/9469952" → 512. Null when absent or in a
// shape we don't understand (e.g. "bytes */N" for unsatisfiable ranges).
function contentRangeStart(response) {
  const m = /^bytes (\d+)-\d+\/(?:\d+|\*)$/.exec(
    response.headers.get('Content-Range') ?? ''
  );
  return m ? Number(m[1]) : null;
}

/// Fetch one planned range, preferring the persistent cache. Deliberately
/// not one multi-range request: multipart/byteranges handling in fetch()
/// is inconsistent across servers and browsers.
async function fetchRange(start, end) {
  const rowsUrl = `${base}/${files.rows}`;
  // Synthetic key: the Cache API keys on URL, and the real request
  // differs only by a header it doesn't consider.
  const key = `${rowsUrl}?r=${start}-${end}`;

  if (rowCache) {
    try {
      const hit = await rowCache.match(key);
      if (hit) {
        engine.ingest(start, new Uint8Array(await hit.arrayBuffer()));
        return;
      }
    } catch {
      // Cache read failed; fall through to the network.
    }
  }

  const r = await fetch(rowsUrl, {
    headers: { Range: `bytes=${start}-${end - 1}` },
  });

  if (r.status === 206) {
    // Ingest at the offset the server declares, not the one we asked for.
    const declared = contentRangeStart(r);
    const bytes = new Uint8Array(await r.arrayBuffer());
    const at = declared ?? start;
    engine.ingest(at, bytes);
    // Only cache when the server sent what we asked for; otherwise the
    // key would lie. Cache API rejects 206 responses outright, so store a
    // fresh 200 carrying the same bytes.
    if (rowCache && at === start) {
      try {
        await rowCache.put(key, new Response(bytes));
      } catch {
        // Quota exceeded or storage evicted — warmth is an optimization.
      }
    }
  } else if (r.ok) {
    // 200 = server ignored Range and sent the whole file; ingest it all at
    // offset 0. Not cached under a range key: it isn't one.
    engine.ingest(0, new Uint8Array(await r.arrayBuffer()));
  } else {
    throw new Error(`range fetch failed: ${r.status}`);
  }
}

async function query(q, gen, limit) {
  if (!engine) return;

  // 1. Ask Rust which bytes it needs. Flat [start, end, start, end, ...].
  const plan = engine.plan(q);

  // 2. Parallel fetches over HTTP/2, cache-first.
  if (plan.length > 0) {
    const jobs = [];
    for (let i = 0; i < plan.length; i += 2) {
      jobs.push(fetchRange(plan[i], plan[i + 1])); // end is half-open
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
