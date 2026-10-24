
import mimetypes
import os
import sys

import httpx

BASE = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")


def run(kind: str, path: str) -> dict:
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    with open(path, "rb") as fh:
        r = httpx.post(f"{BASE}/api/{kind}/analyze", files={"file": (os.path.basename(path), fh, mime)},
                       data={"description": "live check"}, timeout=180)
    body = r.json()
    if not body.get("success"):
        sys.exit(f"{path}: FAILED ({r.status_code}): {body.get('error')}")
    return body["data"]


def selftest():
    """One command that exercises each real dependency and prints PASS/FAIL per check. Uses a generated wall image with a drawn crack."""
    import cv2
    import numpy as np
    img = np.full((600, 800, 3), 190, np.uint8)
    cv2.polylines(img, [np.array([[40, 60], [240, 240], [440, 400], [760, 560]], np.int32)], False, (25, 25, 25), 5)
    png = cv2.imencode(".png", img)[1].tobytes()
    failures = 0

    def check(name, fn):
        nonlocal failures
        try:
            detail = fn()
            print(f"PASS  {name}: {detail}")
        except Exception as e:                       # report and continue with the other checks
            failures += 1
            print(f"FAIL  {name}: {e}")

    def call(method, path, **kw):
        r = httpx.request(method, BASE + path, timeout=180, **kw)
        try:
            b = r.json()
        except ValueError:
            raise RuntimeError(f"HTTP {r.status_code}, not JSON (is the server up / awake?)")
        if not b.get("success"):
            raise RuntimeError(f"HTTP {r.status_code}: {b.get('error')}")
        return b

    def models():
        d = call("GET", "/api/ai/check")["data"]
        for k in ("text_model", "vision_model"):
            if not d[k]["listed"]:
                raise RuntimeError(f"{k} {d[k]['name']!r} is NOT in the provider's model list. Provider offers: {d['available_model_ids']}")
        return f"text={d['text_model']['name']} vision={d['vision_model']['name']} (both listed)"

    def weather():
        d = call("POST", "/api/weather/advisory", json={"location": "Indore", "days": 2})["data"]
        if d.get("weather_source_type") != "live_forecast":
            raise RuntimeError(f"not live: {d.get('weather_source')}")
        return d["weather_source"]

    def image(kind, fields):
        d = call("POST", f"/api/{kind}/analyze", files={"file": ("selftest.png", png, "image/png")}, data={"description": "generated test image"})["data"]
        return f"model={d.get('model')} " + " ".join(f"{k}={d.get(k)}" for k in fields)

    check("health", lambda: call("GET", "/api/health")["data"]["ai"]["detail"])
    check("provider model list", models)
    check("live weather (Open-Meteo)", weather)
    check("text model (knowledge chat)", lambda: call("POST", "/api/knowledge/chat", json={"message": "What is curing of concrete?"})["data"]["answer"][:60] + "...")
    check("vision: safety/analyze", lambda: image("safety", ["overall_status", "safety_score"]))
    check("vision: cracks/analyze", lambda: image("cracks", ["crack_type", "severity"]))
    print("\nALL CHECKS PASSED" if not failures else f"\n{failures} CHECK(S) FAILED. Read the FAIL lines above.")
    sys.exit(1 if failures else 0)


def main():
    if len(sys.argv) == 2 and sys.argv[1] == "selftest":
        selftest()
    if len(sys.argv) != 4 or sys.argv[1] not in ("safety", "cracks"):
        sys.exit(__doc__)
    kind, a, b = sys.argv[1:]
    da, db = run(kind, a), run(kind, b)
    keys = ["scene_summary", "overall_status", "safety_score"] if kind == "safety" else ["crack_type", "severity", "urgency", "estimated_width_mm"]
    for name, d in ((a, da), (b, db)):
        print(f"\n== {name} (model: {d.get('model')})")
        for k in keys:
            print(f"  {k}: {d.get(k)}")
        items = d.get("hazards") if kind == "safety" else d.get("visual_observations")
        for it in (items or [])[:3]:
            print("  -", it.get("description") if isinstance(it, dict) else it)
    same = all(da.get(k) == db.get(k) for k in keys)
    print("\nRESULT:", "IDENTICAL answers for two different photos. Check the photos, model and prompt." if same
          else "Answers differ between the two photos (now check that each matches its photo).")


if __name__ == "__main__":
    main()
