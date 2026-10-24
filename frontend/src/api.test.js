import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const reply = (body, status = 200) => Promise.resolve({ ok: status < 400, status, json: () => (body === undefined ? Promise.reject(new Error("not json")) : Promise.resolve(body)) });
const OK = { success: true, mode: "ai", data: { hello: 1 } };

let fetchMock;
async function loadApi(url) {
  vi.resetModules();
  vi.stubEnv("VITE_API_URL", url);
  return (await import("./api")).api;
}
beforeEach(() => { fetchMock = vi.fn(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => { vi.unstubAllGlobals(); vi.unstubAllEnvs(); vi.useRealTimers(); });

describe("shared API utility", () => {
  it("calls the Render backend URL (trailing slash ignored) with the exact FastAPI path", async () => {
    const api = await loadApi("https://construction-intelligence-1.onrender.com/");
    fetchMock.mockReturnValue(reply(OK));
    await api("/api/weather/advisory", { json: { location: "Indore" } });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("https://construction-intelligence-1.onrender.com/api/weather/advisory");
    expect(init.method).toBe("POST");
    expect(init.headers).toEqual({ "Content-Type": "application/json" });
    expect(JSON.parse(init.body)).toEqual({ location: "Indore" });
  });

  it("sends multipart uploads without any Content-Type header", async () => {
    const api = await loadApi("https://x.onrender.com");
    fetchMock.mockReturnValue(reply(OK));
    const form = new FormData(); form.append("file", new File(["x"], "a.jpg", { type: "image/jpeg" }));
    await api("/api/cracks/analyze", { form });
    const [, init] = fetchMock.mock.calls[0];
    expect(init.body).toBe(form);
    expect(init.headers).toBeUndefined();
  });

  it("refuses a Markdown-style VITE_API_URL instead of building a broken request", async () => {
    const api = await loadApi("[https://x.onrender.com](https://x.onrender.com)");
    await expect(api("/api/health")).rejects.toThrow(/VITE_API_URL must be a plain URL/);
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("passes the server's own readable error through (e.g. an upstream 503)", async () => {
    const api = await loadApi("https://x.onrender.com");
    fetchMock.mockReturnValue(reply({ success: false, mode: "error", error: "The AI provider has retired the configured model 'm'." }, 503));
    await expect(api("/api/cracks/analyze", { form: new FormData() })).rejects.toThrow("The AI provider has retired the configured model 'm'.");
  });

  it("explains a gateway page (non-JSON 502) instead of a bare status", async () => {
    const api = await loadApi("https://x.onrender.com");
    fetchMock.mockReturnValue(reply(undefined, 502));
    await expect(api("/api/valuation", { json: { a: 1 } })).rejects.toThrow(/temporarily unavailable or still waking up \(HTTP 502\)/);
    expect(fetchMock).toHaveBeenCalledTimes(1);          // POSTs are never auto-retried (AI calls cost money)
  });

  it("retries a GET once while a free-tier server wakes up", async () => {
    vi.useFakeTimers();
    const api = await loadApi("https://x.onrender.com");
    fetchMock.mockReturnValueOnce(reply(undefined, 502)).mockReturnValueOnce(reply(OK));
    const p = api("/api/health");
    await vi.advanceTimersByTimeAsync(4000);
    await expect(p).resolves.toMatchObject({ success: true });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it("points at VITE_API_URL when the route is not found", async () => {
    const api = await loadApi("https://x.onrender.com");
    fetchMock.mockReturnValue(reply(undefined, 404));
    await expect(api("/api/health")).rejects.toThrow(/VITE_API_URL/);
  });
});
