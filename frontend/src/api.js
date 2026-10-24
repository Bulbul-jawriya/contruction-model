// Backend base URL, fixed at build time. Empty = same origin (local dev uses the Vite proxy).
const BASE = (import.meta.env.VITE_API_URL || "").trim().replace(/\/+$/, "");
const BASE_IS_PLAIN_URL = !BASE || /^https?:\/\/[^\s[\]()<>]+$/.test(BASE);   // rejects "[text](url)" Markdown pasted by mistake
const TIMEOUT_MS = 150000;           // free hosting can be slow to wake and AI calls can take a while
const GATEWAY = new Set([502, 503, 504]);
const WAKE_RETRY_MS = 4000;

const wait = (ms) => new Promise((r) => setTimeout(r, ms));

async function once(path, init) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  try {
    const res = await fetch(BASE + path, { ...init, signal: ctl.signal });
    let body = null;
    try { body = await res.json(); } catch { /* non-JSON error page (proxy/gateway) */ }
    return { res, body };
  } finally {
    clearTimeout(timer);
  }
}

/** Calls the backend and returns the response envelope. Throws Error with a readable message (never a secret). */
export async function api(path, { json, form } = {}) {
  if (!BASE_IS_PLAIN_URL) {
    throw new Error("The app is misconfigured: VITE_API_URL must be a plain URL such as https://your-service.onrender.com.");
  }
  // For multipart uploads we deliberately set NO Content-Type: the browser adds it together with the boundary.
  const init = json
    ? { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(json) }
    : form ? { method: "POST", body: form } : {};
  const isGet = !json && !form;

  let out;
  try {
    out = await once(path, init);
    // A sleeping free-tier server often answers its first request with a gateway page. Retrying a GET is harmless.
    if (isGet && GATEWAY.has(out.res.status) && !out.body) { await wait(WAKE_RETRY_MS); out = await once(path, init); }
  } catch (e) {
    if (e?.name === "AbortError") throw new Error("The request took too long. Please try again.");
    throw new Error(BASE
      ? "Cannot reach the server. If it is on free hosting it may be waking up (this can take about a minute). Try again shortly."
      : "Cannot reach the backend. Start it with: uvicorn app:app --port 8000");
  }
  const { res, body } = out;
  if (!res.ok || !body?.success) {
    if (body?.error) throw new Error(body.error);
    if (GATEWAY.has(res.status)) throw new Error(`The server is temporarily unavailable or still waking up (HTTP ${res.status}). Wait a minute and try again.`);
    if (res.status === 404) throw new Error("The API route was not found (HTTP 404). Check that VITE_API_URL points at the backend.");
    throw new Error(`Request failed (${res.status}).`);
  }
  return body;
}
