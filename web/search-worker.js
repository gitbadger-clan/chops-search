// Web Worker: the dumb byte pump. All decisions live in the wasm engine;
// this file fetches, ingests, renders nothing, and never touches vectors.
//
// Changes from v4: SNIPPETS. After a ranking is known, the engine names
// the best-matching chunk per result and this worker range-fetches those
// chunks' text from snippets.bin. Two messages per query rather than one:
// results render immediately, snippets fill in when they land — the same
// progressive pattern as keyword-then-semantic, for the same reason.
//
// The snippet OFFSET TABLE is fetched once at boot as a single small
// range (a few hundred bytes); the text blob itself is never fetched
// whole. That's what keeps snippets off the eager payload.
//
// Protocol (postMessage):
//   in : { type: 'init', base }
//   in : { type: 'query', q, gen, limit }
//   out: { type: 'ready' }
//   out: { type: 'results', gen, semantic, results: [{ url, title }] }
//   out: { type: 'snippets', gen, snippets: [string|null] }  // by index
//   out: { type: 'error', message }

let engine = null;
let base = '/search';
let files = null;
let latestGen = 0;
let rowCache = null;
let snips = null; // { offsets: Uint32Array, textStart: number }

const CAN_DECOMPRESS = typeof DecompressionStream !== 'undefined';

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
    initWasm({ module_or_path: `${base}/pkg/chops_wasm_bg.wasm?v=${hash}` }),
    fetchEager(`${base}/${files.meta}`),
    fetchEager(`${base}/${files.index}`),
    fetchBytes(`${base}/${files.prefix}`),
  ]);

  engine = new Engine(meta, index);
  engine.ingest(0, prefix);

  rowCache = await openRowCache(hash);
  // Non-fatal: without offsets, results simply render without snippets.
  snips = files.snippets ? await loadSnippetHeader(files.snippets) : null;
}

/// Fetch just the snippet header: magic + version + reserved + n_chunks +
/// (n+1) u32 offsets. n_chunks equals the engine's chunk count, so the
/// exact size is known before the request.
async function loadSnippetHeader(name) {
  try {
    const n = engine.chunk_count();
    const headerLen = 12 + 4 * (n + 1);
    const r = await fetch(`${base}/${name}`, {
      headers: { Range: `bytes=0-${headerLen - 1}` },
    });
    if (!r.ok) return null;
    const buf = new Uint8Array(await r.arrayBuffer());
    if (buf.length < headerLen) return null; // 200 with a short body: give up
    const dv = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
    if (dv.getUint32(0, false) !== 0x4348534e) return null; // "CHSN"
    if (dv.getUint16(4, true) !== 1) return null;
    if (dv.getUint32(8, true) !== n) return null; // header/index disagree
    const offsets = new Uint32Array(n + 1);
    for (let i = 0; i <= n; i++) offsets[i] = dv.getUint32(12 + i * 4, true);
    return { offsets, textStart: headerLen };
  } catch {
    return null;
  }
}

async function openRowCache(hash) {
  if (typeof caches === 'undefined') return null;
  const name = `chops-rows-${hash}`;
  try {
    const cache = await caches.open(name);
    for (const k of await caches.keys()) {
      if (k.startsWith('chops-rows-') && k !== name) await caches.delete(k);
    }
    return cache;
  } catch {
    return null;
  }
}

async function fetchOk(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`fetch ${url}: ${r.status}`);
  return r;
}

async function fetchBytes(url) {
  return new Uint8Array(await (await fetchOk(url)).arrayBuffer());
}

async function fetchEager(url) {
  if (!CAN_DECOMPRESS) return fetchBytes(url);
  const r = await fetch(`${url}.gz`);
  if (!r.ok) return fetchBytes(url);
  return gunzip(new Uint8Array(await r.arrayBuffer()));
}

async function gunzip(buf) {
  if (buf.length < 2 || buf[0] !== 0x1f || buf[1] !== 0x8b) return buf;
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

function contentRangeStart(response) {
  const m = /^bytes (\d+)-\d+\/(?:\d+|\*)$/.exec(
    response.headers.get('Content-Range') ?? ''
  );
  return m ? Number(m[1]) : null;
}

async function fetchRange(start, end) {
  const rowsUrl = `${base}/${files.rows}`;
  const key = `${rowsUrl}?r=${start}-${end}`;

  if (rowCache) {
    try {
      const hit = await rowCache.match(key);
      if (hit) {
        engine.ingest(start, new Uint8Array(await hit.arrayBuffer()));
        return;
      }
    } catch {
      /* fall through to network */
    }
  }

  const r = await fetch(rowsUrl, { headers: { Range: `bytes=${start}-${end - 1}` } });
  if (r.status === 206) {
    const declared = contentRangeStart(r);
    const bytes = new Uint8Array(await r.arrayBuffer());
    const at = declared ?? start;
    engine.ingest(at, bytes);
    if (rowCache && at === start) {
      try {
        await rowCache.put(key, new Response(bytes));
      } catch {
        /* quota — warmth is an optimization */
      }
    }
  } else if (r.ok) {
    engine.ingest(0, new Uint8Array(await r.arrayBuffer()));
  } else {
    throw new Error(`range fetch failed: ${r.status}`);
  }
}

/// Fetch the text of one chunk. Small enough (~600 bytes) that these go
/// out in parallel and land well inside one RTT of each other.
async function fetchSnippet(chunk) {
  if (!snips || chunk < 0 || chunk + 1 >= snips.offsets.length) return null;
  const lo = snips.textStart + snips.offsets[chunk];
  const hi = snips.textStart + snips.offsets[chunk + 1];
  if (hi <= lo) return null; // empty chunk
  try {
    const r = await fetch(`${base}/${files.snippets}`, {
      headers: { Range: `bytes=${lo}-${hi - 1}` },
    });
    if (!r.ok) return null;
    if (r.status === 206) return await r.text();
    // 200: server ignored Range and sent the whole blob — slice it.
    const buf = new Uint8Array(await r.arrayBuffer());
    return new TextDecoder().decode(buf.subarray(lo, hi));
  } catch {
    return null;
  }
}

async function query(q, gen, limit) {
  if (!engine) return;

  const plan = engine.plan(q);
  if (plan.length > 0) {
    const jobs = [];
    for (let i = 0; i < plan.length; i += 2) {
      jobs.push(fetchRange(plan[i], plan[i + 1]));
    }
    try {
      await Promise.all(jobs);
    } catch {
      // Offline / range-hostile host: search() degrades to keyword-only
      // on its own and reports it via `semantic: false`.
    }
  }

  if (gen !== latestGen) return;

  const ids = engine.search(q, limit);
  const results = [];
  const chunks = [];
  for (const id of ids) {
    results.push({ url: engine.doc_url(id), title: engine.doc_title(id) });
    chunks.push(engine.best_chunk(id));
  }
  self.postMessage({
    type: 'results',
    gen,
    semantic: engine.used_semantic(),
    results,
  });

  // Second pass: snippets. Never blocks the ranking, and a failure here
  // leaves the results list exactly as it already rendered.
  if (!snips || results.length === 0) return;
  const texts = await Promise.all(chunks.map(fetchSnippet));
  if (gen !== latestGen) return;
  self.postMessage({ type: 'snippets', gen, snippets: texts });
}
