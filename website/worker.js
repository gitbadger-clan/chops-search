// worker.js — range-slicing shim for Workers static assets.
// Only invoked for run_worker_first paths (model.rows.*, snippets.*).

const inflight = new Map(); // url -> Promise<{buf, type, cc}>

async function loadFull(url, env) {
  let p = inflight.get(url);
  if (!p) {
    p = (async () => {
      const res = await env.ASSETS.fetch(new Request(url));
      if (!res.ok) {
        // Don't throw the Response itself: this promise is shared by every
        // coalesced caller, and a single Response object can only be
        // returned from one fetch handler. Throw the status; each caller
        // mints its own Response.
        res.body?.cancel();
        throw Object.assign(new Error(`asset fetch failed: ${res.status}`), {
          status: res.status,
        });
      }
      return {
        buf: await res.arrayBuffer(),
        type: res.headers.get("Content-Type") ?? "application/octet-stream",
        cc: res.headers.get("Cache-Control") ?? "public, max-age=31536000, immutable",
      };
    })();
    inflight.set(url, p);
    // .catch first: .finally alone returns a derived promise that mirrors
    // the rejection with no handler — an unhandled rejection in the logs
    // on every origin failure. Callers still see the rejection via `p`.
    p.catch(() => { }).finally(() => inflight.delete(url));
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
    try {
      full = await loadFull(url, env);
    } catch (e) {
      return new Response(null, { status: e?.status ?? 502 });
    }
    const { buf, type, cc } = full;

    ctx.waitUntil(
      cache
        .put(url, new Response(buf, {
          headers: {
            "Content-Type": type,
            "Content-Length": String(buf.byteLength),
            "Cache-Control": cc,
            "Accept-Ranges": "bytes",
          },
        }))
        .catch(() => { }) // caching is an optimization; don't log quota/uncacheable rejections
    );

    const full200 = () =>
      new Response(buf, {
        status: 200,
        headers: {
          "Content-Type": type,
          "Cache-Control": cc,
          "Accept-Ranges": "bytes",
          "x-range-source": "cold",
        },
      });

    const m = /^bytes=(\d+)-(\d*)$/.exec(range);
    // Multipart, suffix (`bytes=-N`), or malformed: ignoring Range is
    // spec-legal; the client handles 200s.
    if (!m) return full200();

    const start = Number(m[1]);
    if (start >= buf.byteLength) {
      return new Response(null, {
        status: 416,
        headers: { "Content-Range": `bytes */${buf.byteLength}` },
      });
    }
    const end = m[2] ? Math.min(Number(m[2]), buf.byteLength - 1) : buf.byteLength - 1;
    // `bytes=5-2`: negative slice length would throw. Treat as malformed.
    if (end < start) return full200();

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
