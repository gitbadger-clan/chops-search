// worker.js — range-slicing shim for Workers static assets.
// Only invoked for run_worker_first paths (model.rows.*, snippets.*).

const inflight = new Map(); // url -> Promise<{buf, type, cc}>

async function loadFull(url, env) {
  let p = inflight.get(url);
  if (!p) {
    p = (async () => {
      const res = await env.ASSETS.fetch(new Request(url));
      if (!res.ok) throw res;
      return {
        buf: await res.arrayBuffer(),
        type: res.headers.get("Content-Type") ?? "application/octet-stream",
        cc: res.headers.get("Cache-Control") ?? "public, max-age=31536000, immutable",
      };
    })();
    inflight.set(url, p);
    p.finally(() => inflight.delete(url));
  }
  return p;
}

export default {
  async fetch(request, env, ctx) {
    const range = request.headers.get("Range");
    if (!range || request.method !== "GET") return env.ASSETS.fetch(request);

    const url = new URL(request.url).toString();
    const cache = caches.default;

    const cached = await cache.match(request); // Range-aware: returns a 206 slice
    if (cached) {
      const out = new Response(cached.body, cached);
      out.headers.set("x-range-source", "edge-cache");
      return out;
    }

    let full;
    try { full = await loadFull(url, env); }
    catch (res) { return res instanceof Response ? res : new Response(null, { status: 502 }); }
    const { buf, type, cc } = full;

    ctx.waitUntil(cache.put(url, new Response(buf, {
      headers: {
        "Content-Type": type,
        "Content-Length": String(buf.byteLength),
        "Cache-Control": cc,
        "Accept-Ranges": "bytes",
      },
    })));

    const m = /^bytes=(\d+)-(\d*)$/.exec(range);
    if (!m) {
      // Multipart or malformed: ignoring Range is spec-legal; client handles 200s.
      return new Response(buf, {
        status: 200, headers: {
          "Content-Type": type, "Cache-Control": cc,
          "Accept-Ranges": "bytes", "x-range-source": "cold",
        }
      });
    }
    const start = Number(m[1]);
    if (start >= buf.byteLength) return new Response(null, {
      status: 416,
      headers: { "Content-Range": `bytes */${buf.byteLength}` }
    });
    const end = m[2] ? Math.min(Number(m[2]), buf.byteLength - 1) : buf.byteLength - 1;

    return new Response(new Uint8Array(buf, start, end - start + 1), {
      status: 206,
      headers: {
        "Content-Type": type,
        "Content-Range": `bytes ${start}-${end}/${buf.byteLength}`,
        "Content-Length": String(end - start + 1),
        "Accept-Ranges": "bytes",
        "Cache-Control": cc,
        "x-range-source": "cold",
      },
    });
  },
};
