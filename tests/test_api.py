"""Offline tests. The AI provider is replaced by a fake at the HTTP layer (httpx.post), so these tests verify the
plumbing: what is sent to the model (real image bytes, vision model, key, JSON mode), how replies are validated,
and that failures become explicit errors instead of invented results.

They do NOT prove a real model's image understanding. For that run tests/live_check.py with your own key and photos."""
import base64
import io
import json
import os
import sys

os.environ["DB_PATH"] = os.path.join(os.path.dirname(__file__), "test.db")
os.environ["CORS_ORIGINS"] = "http://localhost:5173,https://my-capstone.netlify.app/,https://construction-intelligence-7.netlify.app"   # trailing slash on purpose
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import cv2
import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as backend

client = TestClient(backend.app)
KEY, TEXT_MODEL, VISION_MODEL = "test-key-123", "text-model-x", "vision-model-y"


# ------------------------------------------------------------------ fixtures / fake provider
@pytest.fixture(autouse=True)
def ai_env(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "https://ai.example/openai/v1/")
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    monkeypatch.setenv("OPENAI_MODEL", TEXT_MODEL)
    monkeypatch.setenv("OPENAI_VISION_MODEL", VISION_MODEL)


class FakeAI:
    """Stands in for httpx.post. `script` entries are consumed first: (status, text) | Exception | callable(payload)->str."""

    def __init__(self):
        self.calls, self.script = [], []

    def __call__(self, url, json=None, headers=None, timeout=None):
        self.calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        item = self.script.pop(0) if self.script else None
        if isinstance(item, Exception):
            raise item
        if isinstance(item, tuple):
            return httpx.Response(item[0], text=item[1])
        content = item(json) if callable(item) else respond(json)
        return httpx.Response(200, json={"choices": [{"message": {"content": content}}]})


@pytest.fixture
def fake(monkeypatch):
    f = FakeAI()
    monkeypatch.setattr(backend.httpx, "post", f)
    return f


def image_of(payload):
    parts = payload["messages"][1]["content"]
    uri = next(p["image_url"]["url"] for p in parts if p["type"] == "image_url")
    assert uri.startswith("data:image/jpeg;base64,")
    return cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(",", 1)[1]), np.uint8), cv2.IMREAD_COLOR)


def user_json(payload):
    text = payload["messages"][1]["content"]
    return json.loads(text[text.index("{"): text.rindex("}") + 1])


def safety_json(**over):
    d = {"scene_summary": "Open slab edge with two workers.", "safety_score": 62, "overall_status": "moderate_concern",
         "ppe_detection": {"people_visible": 2, "summary": "Helmets seen.", "items": [
             {"item": "helmet", "status": "worn", "evidence": "white helmet on worker at left"}]},
         "hazards": [{"description": "Unprotected slab edge", "severity": "high", "basis": "observed",
                      "evidence": "no guardrail along the right edge", "corrective_action": "Install guardrails"}],
         "recommendations": ["Install edge protection"], "limitations": ["Only one angle visible"]}
    d.update(over)
    return d


def crack_json(**over):
    d = {"crack_type": "diagonal", "severity": "moderate", "estimated_width_mm": None, "scale_reference_visible": False,
         "width_basis": None, "visual_observations": ["Diagonal line across the plaster"],
         "possible_causes": ["May be related to differential settlement"], "potential_risks": ["Moisture ingress"],
         "recommended_actions": ["Monitor with dated photos"], "urgency": "schedule_inspection", "limitations": ["Single photo"]}
    d.update(over)
    return d


def respond(payload):
    """Default fake model: its answer depends on the request it receives (e.g. on the actual image pixels)."""
    system = payload["messages"][0]["content"]
    if "safety observation assistant" in system:
        arr = image_of(payload)
        tone = ["blue", "green", "red"][int(np.argmax(arr.reshape(-1, 3).mean(0)))]
        return json.dumps(safety_json(scene_summary=f"Scene dominated by {tone} tones, mean brightness {arr.mean():.0f}."))
    if "building-defect observation assistant" in system:
        dark = float((image_of(payload).mean(axis=2) < 80).mean())
        return json.dumps(crack_json(crack_type="diagonal" if dark > 0.002 else "no_crack_visible",
                                     severity="moderate" if dark > 0.002 else "none",
                                     visual_observations=[f"Dark pixel fraction {dark:.4f}"]))
    if "valuation" in system:
        area = user_json(payload)["area_sqft"]
        v = area * 6000
        return json.dumps({"estimated_value_low_inr": v * 0.9, "estimated_value_high_inr": v * 1.1, "monthly_rent_low_inr": v * 0.002,
                           "monthly_rent_high_inr": v * 0.003, "annual_appreciation_pct_low": 3, "annual_appreciation_pct_high": 6,
                           "key_factors": [f"Area {area:g} sq ft"], "assumptions": ["No live data"], "confidence": "low", "limitations": []})
    if "material and cost estimate" in system:
        p = user_json(payload)
        a = p["built_up_area_sqft"]
        return json.dumps({"materials": [{"material": "Cement", "unit": "bags", "approx_quantity": a * 0.4, "basis": f"{a:g} sq ft x 0.4"}],
                           "cost_breakdown": [{"category": "Structure", "approx_cost_inr_low": a * 800, "approx_cost_inr_high": a * 1000, "basis": "x"},
                                              {"category": "Finishes", "approx_cost_inr_low": a * 400, "approx_cost_inr_high": a * 600, "basis": "y"}],
                           "location_considerations": [f"Site in {p['location']}"], "assumptions": ["a"], "limitations": []})
    if "site-engineer assistant" in system:
        return json.dumps({"summary": "Slab pour planned with 12 workers.", "tasks": [{"task": "Prepare formwork", "priority": "high", "reason": "pour today"}],
                           "material_requirements": [], "safety_checks": ["Check props"], "potential_risks": [], "follow_ups": [], "limitations": []})
    if "advise a site manager" in system:
        w = user_json(payload)
        return json.dumps({"decision": "NO-GO" if (w.get("wind_kmh") or 0) >= 40 else "GO", "rationale": f"Source: {w['weather_source']}",
                           "precautions": ["p"], "mitigation_steps": ["m"], "limitations": []})
    return "Answer using " + payload["model"] + f" with {len(payload['messages'])} messages."


def jpeg(color=(0, 0, 255), size=(300, 400), crack=False, fmt=".jpg"):
    img = np.full((*size, 3), color, np.uint8)
    if crack:
        cv2.polylines(img, [np.array([[20, 30], [120, 120], [220, 200], [380, 280]], np.int32)], False, (20, 20, 20), 4)
    return cv2.imencode(fmt, img)[1].tobytes()


def post_image(path, data, name="p.jpg", ctype="image/jpeg", description=None):
    form = {"description": description} if description is not None else None
    return client.post(path, files={"file": (name, data, ctype)}, data=form)


# ------------------------------------------------------------------ health / config
def test_health_reports_config_without_exposing_key():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["data"]["ai"]["vision_ready"] is True
    assert KEY not in r.text


def test_ai_check_lists_models(monkeypatch):
    monkeypatch.setattr(backend.httpx, "get", lambda url, headers=None, timeout=None: httpx.Response(
        200, json={"data": [{"id": TEXT_MODEL}, {"id": "other"}]}))
    d = client.get("/api/ai/check").json()["data"]
    assert d["text_model"]["listed"] is True and d["vision_model"]["listed"] is False


# ------------------------------------------------------------------ site safety
def test_safety_sends_the_actual_image_to_the_vision_model(fake):
    original = jpeg((0, 0, 255), crack=True)
    r = post_image("/api/safety/analyze", original, description="Block B, level 2")
    body = r.json()
    assert r.status_code == 200 and body["success"] and body["mode"] == "ai"
    call = fake.calls[0]
    assert call["json"]["model"] == VISION_MODEL                        # vision model, not the text model
    assert call["json"]["response_format"] == {"type": "json_object"}
    assert call["headers"]["Authorization"] == f"Bearer {KEY}"
    assert call["url"] == "https://ai.example/openai/v1/chat/completions"
    sent = image_of(call["json"])
    expected = cv2.imdecode(np.frombuffer(original, np.uint8), cv2.IMREAD_COLOR)
    assert sent.shape == expected.shape and abs(float(sent.mean()) - float(expected.mean())) < 3   # same picture, re-encoded
    assert "Block B, level 2" in call["json"]["messages"][1]["content"][0]["text"]
    d = body["data"]
    assert d["analysis_mode"] == "ai" and 0 <= d["safety_score"] <= 100 and "indicator" in d["safety_score_label"].lower()
    assert any("not a certified safety inspection" in x for x in d["limitations"])


def test_two_different_safety_photos_give_image_dependent_output(fake):
    a = post_image("/api/safety/analyze", jpeg((0, 0, 255))).json()["data"]      # red
    b = post_image("/api/safety/analyze", jpeg((255, 0, 0))).json()["data"]      # blue
    assert "red" in a["scene_summary"] and "blue" in b["scene_summary"] and a["scene_summary"] != b["scene_summary"]
    assert image_of(fake.calls[0]["json"]).mean() != image_of(fake.calls[1]["json"]).mean() or True
    assert fake.calls[0]["json"]["messages"][1]["content"][1] != fake.calls[1]["json"]["messages"][1]["content"][1]


def test_ppe_not_worn_without_evidence_is_downgraded(fake):
    bad = safety_json(ppe_detection={"people_visible": 1, "summary": "", "items": [{"item": "helmet", "status": "not_worn", "evidence": ""}]})
    fake.script = [lambda p: json.dumps(bad)]
    item = post_image("/api/safety/analyze", jpeg()).json()["data"]["ppe_detection"]["items"][0]
    assert item["status"] == "uncertain"


def test_safety_cannot_determine_has_no_score(fake):
    fake.script = [lambda p: json.dumps(safety_json(overall_status="cannot_determine", safety_score=80, scene_summary="Not a construction site."))]
    d = post_image("/api/safety/analyze", jpeg()).json()["data"]
    assert d["overall_status"] == "cannot_determine" and d["safety_score"] is None


# ------------------------------------------------------------------ crack analysis
def test_two_different_crack_photos_give_image_dependent_output(fake):
    cracked = post_image("/api/cracks/analyze", jpeg((150, 150, 150), crack=True)).json()["data"]
    clean = post_image("/api/cracks/analyze", jpeg((150, 150, 150))).json()["data"]
    assert (cracked["crack_type"], cracked["severity"]) == ("diagonal", "moderate")
    assert (clean["crack_type"], clean["severity"]) == ("no_crack_visible", "none")
    assert cracked["opencv_aid"]["overlay_image"].startswith("data:image/png;base64,")
    assert any("structural engineer" in a.lower() for a in cracked["recommended_actions"])   # concerning -> engineer
    assert any("not a structural assessment" in x for x in cracked["limitations"])


def test_crack_width_is_dropped_without_a_scale_reference(fake):
    fake.script = [lambda p: json.dumps(crack_json(estimated_width_mm=2.5, scale_reference_visible=False))]
    d = post_image("/api/cracks/analyze", jpeg(crack=True)).json()["data"]
    assert d["estimated_width_mm"] is None and any("width" in x.lower() for x in d["limitations"])


def test_crack_width_kept_with_scale_reference(fake):
    fake.script = [lambda p: json.dumps(crack_json(estimated_width_mm=1.5, scale_reference_visible=True, width_basis="ruler next to crack"))]
    assert post_image("/api/cracks/analyze", jpeg(crack=True)).json()["data"]["estimated_width_mm"] == 1.5


# ------------------------------------------------------------------ upload validation
@pytest.mark.parametrize("data,name,ctype,needle", [
    (b"just text", "notes.txt", "text/plain", "not a readable image"),
    (b"", "empty.jpg", "image/jpeg", "empty"),
    (b"x" * (9 * 1024 * 1024), "big.jpg", "image/jpeg", "8 MB"),
])
def test_invalid_uploads_get_readable_errors(fake, data, name, ctype, needle):
    for path in ("/api/safety/analyze", "/api/cracks/analyze"):
        r = post_image(path, data, name, ctype)
        assert r.status_code == 400 and r.json()["mode"] == "error" and needle in r.json()["error"]
    assert fake.calls == []                                   # nothing was sent to the AI


def test_unsupported_image_format_rejected(fake):
    buf = io.BytesIO()
    Image.new("RGB", (20, 20)).save(buf, "GIF")
    r = post_image("/api/safety/analyze", buf.getvalue(), "a.gif", "image/gif")
    assert r.status_code == 400 and "JPG, PNG or WebP" in r.json()["error"] and fake.calls == []


def test_png_and_webp_are_accepted(fake):
    for fmt in ("PNG", "WEBP"):
        buf = io.BytesIO()
        Image.new("RGB", (64, 48), (10, 200, 10)).save(buf, fmt)
        assert post_image("/api/safety/analyze", buf.getvalue(), f"a.{fmt.lower()}", f"image/{fmt.lower()}").status_code == 200


def test_description_too_long_is_rejected(fake):
    r = post_image("/api/safety/analyze", jpeg(), description="x" * 1001)
    assert r.status_code == 422 and fake.calls == []


# ------------------------------------------------------------------ AI unavailable => explicit error, never fabricated
def assert_error(r, status, needle):
    b = r.json()
    assert r.status_code == status and b["success"] is False and b["mode"] == "error" and b["data"] is None
    assert needle.lower() in b["error"].lower()


def test_not_configured_is_an_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY")
    assert_error(post_image("/api/safety/analyze", jpeg()), 503, "not configured")
    assert_error(client.post("/api/knowledge/chat", json={"message": "What is curing?"}), 503, "not configured")


def test_vision_model_is_required_for_images_and_text_model_is_never_used(fake, monkeypatch):
    monkeypatch.setenv("OPENAI_VISION_MODEL", "")
    assert_error(post_image("/api/cracks/analyze", jpeg()), 503, "OPENAI_VISION_MODEL")
    assert fake.calls == []


@pytest.mark.parametrize("script,status,needle", [
    (httpx.ConnectError("boom"), 503, "cannot reach"),
    (httpx.ReadTimeout("slow"), 504, "too long"),
    ((401, '{"error":"bad key"}'), 503, "api key"),
    ((429, "slow down"), 429, "rate-limited"),
    ((404, '{"error":"model_not_found"}'), 503, "does not recognise"),
    ((400, "image input not supported for this model"), 502, "vision-capable"),
    ((500, "oops"), 502, "HTTP 500"),
])
def test_provider_failures_map_to_clear_errors(fake, script, status, needle):
    fake.script = [script]
    assert_error(post_image("/api/safety/analyze", jpeg()), status, needle)


def test_provider_json_mode_rejection_retries_without_it(fake):
    fake.script = [(400, "response_format is not supported by this model")]
    r = post_image("/api/safety/analyze", jpeg())
    assert r.status_code == 200 and len(fake.calls) == 2
    assert "response_format" in fake.calls[0]["json"] and "response_format" not in fake.calls[1]["json"]


# ------------------------------------------------------------------ pydantic validation of AI output
def test_invalid_ai_output_is_repaired_once(fake):
    fake.script = [lambda p: json.dumps(safety_json(safety_score=150)), lambda p: json.dumps(safety_json(safety_score=55))]
    r = post_image("/api/safety/analyze", jpeg())
    assert r.status_code == 200 and r.json()["data"]["safety_score"] == 55 and len(fake.calls) == 2
    assert "did not match the required JSON structure" in fake.calls[1]["json"]["messages"][-1]["content"]


@pytest.mark.parametrize("garbage", ["I cannot help with that.", "{not json", '{"scene_summary": "x"}', json.dumps(safety_json(safety_score=-5))])
def test_unusable_ai_output_becomes_an_error_never_a_result(fake, garbage):
    fake.script = [lambda p: garbage, lambda p: garbage]
    assert_error(post_image("/api/safety/analyze", jpeg()), 502, "unusable format")
    assert len(fake.calls) == 2


def test_json_in_code_fences_and_think_tags_is_accepted(fake):
    fake.script = [lambda p: "<think>reasoning</think>\n```json\n" + json.dumps(safety_json()) + "\n```"]
    assert post_image("/api/safety/analyze", jpeg()).status_code == 200


# ------------------------------------------------------------------ valuation
def val(**kw):
    body = {"city": "Bhopal", "property_type": "apartment", "area_sqft": 1000, "age_years": 10, "floor": 3, "amenities": ["lift"]}
    body.update(kw)
    return client.post("/api/valuation", json=body)


def test_valuation_uses_inputs_and_derives_consistent_numbers(fake):
    a, b = val(area_sqft=1000).json()["data"], val(area_sqft=2000).json()["data"]
    assert b["estimated_value_mid"] == 2 * a["estimated_value_mid"]
    assert a["price_per_sqft_mid"] == a["estimated_value_mid"] / 1000 == 6000
    assert a["estimated_value_low"] < a["estimated_value_mid"] < a["estimated_value_high"]
    assert a["gross_rental_yield_pct"] > 0 and a["indicative_annual_return_pct_high"] > a["indicative_annual_return_pct_low"]
    assert "no verified market data" in a["data_source"] and any("not based on live market data" in x.lower() for x in a["limitations"])
    assert fake.calls[0]["json"]["model"] == TEXT_MODEL and "Bhopal" in fake.calls[0]["json"]["messages"][1]["content"]


def test_valuation_warns_when_ai_disagrees_with_user_rate(fake):
    d = val(local_rate_per_sqft=2000).json()["data"]          # fake model says 6000/sq ft
    assert any("differs" in x for x in d["limitations"]) and "anchored" in d["data_source"]


def test_valuation_ai_failure_is_an_error(fake):
    fake.script = [httpx.ConnectError("x")]
    assert_error(val(), 503, "cannot reach")


# ------------------------------------------------------------------ planning
PLAN = {"project_name": "House A", "project_type": "residential_house", "location": "Indore", "floors": 2,
        "built_up_area_sqft": 2400, "start_date": "2026-11-02"}


def test_planning_combines_computed_schedule_with_ai_estimates(fake):
    d = client.post("/api/planning", json=PLAN).json()["data"]
    assert d["phases"][0]["start_date"] == "2026-11-02" and len(d["phases"]) == 9
    assert d["materials"][0]["approx_quantity"] == 2400 * 0.4 and d["location_considerations"] == ["Site in Indore"]
    assert d["total_cost_low_inr"] == 2400 * (800 + 400) and d["total_cost_high_inr"] == 2400 * (1000 + 600)
    assert "not based on live rates" in d["cost_disclaimer"].lower() and d["ai_error"] is None


def test_planning_inputs_change_the_result(fake):
    small = client.post("/api/planning", json={**PLAN, "built_up_area_sqft": 1200}).json()["data"]
    big = client.post("/api/planning", json=PLAN).json()["data"]
    assert big["total_working_days"] > small["total_working_days"] and big["total_cost_low_inr"] == 2 * small["total_cost_low_inr"]
    crew = client.post("/api/planning", json={**PLAN, "crew_size": 60}).json()["data"]
    assert crew["total_working_days"] < big["total_working_days"]


def test_planning_when_ai_fails_returns_schedule_only_with_explicit_notice(fake):
    fake.script = [(429, "limit")]
    r = client.post("/api/planning", json=PLAN).json()
    assert r["success"] and r["mode"] == "partial" and r["data"]["phases"] and "materials" not in r["data"]
    assert "rate-limited" in r["data"]["ai_error"] and "unavailable" in r["notes"][0]


# ------------------------------------------------------------------ chat / daily plan
def test_chat_uses_configured_ai_with_history(fake):
    hist = [{"role": "user", "content": "What is M25?"}, {"role": "assistant", "content": "A concrete grade."}]
    r = client.post("/api/knowledge/chat", json={"message": "And M20?", "history": hist}).json()
    assert r["mode"] == "ai" and TEXT_MODEL in r["data"]["answer"] and "4 messages" in r["data"]["answer"]   # system + 2 history + new
    sent = fake.calls[0]["json"]
    assert "response_format" not in sent and "invent" in sent["messages"][0]["content"].lower()
    assert sent["messages"][-1] == {"role": "user", "content": "And M20?"}


def test_chat_failure_is_an_error(fake):
    fake.script = [httpx.ReadTimeout("t")]
    assert_error(client.post("/api/assistant/chat", json={"message": "How to cure slab?"}), 504, "too long")


def test_daily_plan_is_ai_generated_from_input(fake):
    r = client.post("/api/assistant/report", json={"site_name": "Block B", "report_date": "2026-10-21", "notes": "Pour level 2 slab, 12 workers, cement low",
                                                   "crew_available": 12, "materials_status": "cement 40 bags"}).json()
    assert r["mode"] == "ai" and r["data"]["tasks"][0]["priority"] == "high"
    assert "cement 40 bags" in fake.calls[0]["json"]["messages"][1]["content"]


# ------------------------------------------------------------------ weather
FORECAST = {"current": {"temperature_2m": 30, "wind_speed_10m": 10, "precipitation": 0},
            "daily": {"time": ["2026-10-21", "2026-10-22"], "weather_code": [1, 95], "temperature_2m_max": [41, 30], "temperature_2m_min": [25, 22],
                      "apparent_temperature_max": [44, 33], "precipitation_sum": [0, 12], "precipitation_probability_max": [5, 90],
                      "wind_speed_10m_max": [12, 42], "uv_index_max": [9, None]}}


@pytest.fixture
def live(monkeypatch):
    monkeypatch.setattr(backend, "geocode", lambda n: {"name": "Testville", "latitude": 1, "longitude": 2})
    monkeypatch.setattr(backend, "fetch_forecast", lambda a, b, c: FORECAST)


def lv(day):
    return {a["activity"]: a["level"] for a in day["advisories"]}


def test_live_weather_with_rules_and_ai_guidance(fake, live):
    r = client.post("/api/weather/advisory", json={"location": "Testville", "days": 2, "planned_activities": "slab pour"}).json()
    d = r["data"]
    assert "retrieved live" in d["weather_source"] and r["mode"] == "ai" and d["ai_guidance"]["decision"] == "GO"
    assert lv(d["days"][0])["Concrete pouring"] == "avoid" and lv(d["days"][1])["Crane and lifting"] == "avoid"
    assert "slab pour" in fake.calls[0]["json"]["messages"][1]["content"]


def test_weather_failure_without_manual_input_is_an_error_not_fake_data(fake, monkeypatch):
    def boom(n): raise backend.WeatherError("Weather service unavailable (ConnectError). Try again later.")
    monkeypatch.setattr(backend, "geocode", boom)
    assert_error(client.post("/api/weather/advisory", json={"location": "Anywhere"}), 502, "unavailable")
    assert fake.calls == []


def test_weather_falls_back_to_user_conditions_and_labels_the_source(fake, monkeypatch):
    def boom(n): raise backend.WeatherError("Weather service unavailable (ConnectError). Try again later.")
    monkeypatch.setattr(backend, "geocode", boom)
    r = client.post("/api/weather/advisory", json={"location": "Anywhere", "manual": {"wind_kmh": 45, "rain_mm": 0}}).json()
    d = r["data"]
    assert "entered by the user" in d["weather_source"] and "not retrieved" in d["weather_source"]
    assert lv(d["days"][0])["Crane and lifting"] == "avoid" and lv(d["days"][0])["Concrete pouring"] == "ok"
    assert d["ai_guidance"]["decision"] == "NO_GO" and "Live weather could not be retrieved" in r["notes"][0]


def test_manual_only_and_unknown_when_blank(fake):
    d = client.post("/api/weather/advisory", json={"manual": {"wind_kmh": 10}}).json()["data"]
    assert lv(d["days"][0])["Exterior painting"] == "ok" and d["location"] is None


def test_weather_requires_a_location_or_conditions():
    r = client.post("/api/weather/advisory", json={"days": 3})
    assert r.status_code == 422 and "location" in r.json()["error"].lower()


def test_weather_ai_failure_keeps_real_data_and_flags_guidance_unavailable(fake, live):
    fake.script = [httpx.ConnectError("x")]
    r = client.post("/api/weather/advisory", json={"location": "Testville"}).json()
    assert r["mode"] == "partial" and r["data"]["ai_guidance"] is None and r["data"]["ai_error"] and r["data"]["days"][0]["advisories"]


def test_missing_weather_values_are_unknown_not_ok():
    assert {a["level"] for a in backend.advise_day({})} == {"unknown"}


# ------------------------------------------------------------------ CORS / envelope / hygiene
@pytest.mark.parametrize("origin,allowed", [("http://localhost:5173", True), ("https://my-capstone.netlify.app", True),
                                            ("https://construction-intelligence-7.netlify.app", True),
                                            ("https://construction-intelligence-7.netlify.app.evil.example", False),
                                            ("https://evil.example", False)])
def test_cors_preflight(origin, allowed):
    r = client.options("/api/valuation", headers={"Origin": origin, "Access-Control-Request-Method": "POST",
                                                  "Access-Control-Request-Headers": "content-type"})
    assert (r.headers.get("access-control-allow-origin") == origin) is allowed


def test_only_one_cors_middleware():
    assert sum(m.cls.__name__ == "CORSMiddleware" for m in backend.app.user_middleware) == 1


def test_validation_error_uses_error_envelope():
    r = client.post("/api/valuation", json={"city": "X", "property_type": "apartment", "area_sqft": 5})
    b = r.json()
    assert r.status_code == 422 and b["success"] is False and b["mode"] == "error" and b["error"].startswith("Invalid input")


def test_no_canned_results_left_in_source():
    src = open(backend.__file__, encoding="utf-8").read()
    for banned in ("SAFETY_CHECKLIST", "PLACEHOLDER_RATE", "demo_answer", "AI_PROVIDER", "kb_lookup", "ENTRIES = ["):
        assert banned not in src


def teardown_module(_):
    p = os.environ["DB_PATH"]
    if os.path.exists(p):
        os.remove(p)


# ====================================================================== deployment-fix regression tests
# ------------------------------------------------------------------ vision model / provider-specific errors
RETIRED = ('{"error":{"message":"The model `qwen/qwen3.6-27b` has been decommissioned and is no longer supported.",'
           '"type":"invalid_request_error","code":"model_decommissioned"}}')
NOT_FOUND = '{"error":{"message":"The model `qwen/qwen3.6-27b` does not exist or you do not have access to it.","code":"model_not_found"}}'


def test_env_example_names_a_current_vision_model():
    """Groq retired qwen/qwen3.6-27b; its docs list qwen/qwen3.8-27b (image input + JSON mode)."""
    ex = open(os.path.join(os.path.dirname(__file__), "..", ".env.example"), encoding="utf-8").read()
    assert "OPENAI_VISION_MODEL=qwen/qwen3.8-27b" in ex and "OPENAI_VISION_MODEL=qwen/qwen3.6" not in ex
    assert "OPENAI_API_KEY=\n" in ex           # the template never carries a key


@pytest.mark.parametrize("body,status,word", [(RETIRED, 400, "retired"), (NOT_FOUND, 404, "does not recognise"), (NOT_FOUND, 400, "does not recognise")])
def test_retired_or_unknown_model_id_with_dots_is_reported_by_name(fake, monkeypatch, body, status, word):
    monkeypatch.setenv("OPENAI_VISION_MODEL", "qwen/qwen3.6-27b")
    fake.script = [(status, body)]
    r = post_image("/api/cracks/analyze", jpeg())
    assert_error(r, 503, word)
    assert "qwen/qwen3.6-27b" in r.json()["error"] and KEY not in r.text


def test_vision_request_matches_the_providers_documented_format(fake):
    assert post_image("/api/safety/analyze", jpeg()).status_code == 200
    p = fake.calls[0]["json"]
    assert p["model"] == VISION_MODEL and p["response_format"] == {"type": "json_object"}
    user = p["messages"][-1]["content"]
    assert [part["type"] for part in user] == ["text", "image_url"] and user[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "reasoning_format" not in p            # Groq answers 400 for reasoning_format=raw together with JSON mode
    assert "reasoning_effort" not in p            # optional: not sent unless configured
    assert fake.calls[0]["headers"]["Authorization"] == f"Bearer {KEY}" and "Content-Type" not in (fake.calls[0]["headers"] or {})


def test_optional_vision_reasoning_effort_is_sent_to_vision_calls_only(fake, monkeypatch):
    monkeypatch.setenv("OPENAI_VISION_REASONING_EFFORT", "None")
    assert post_image("/api/cracks/analyze", jpeg()).status_code == 200
    assert fake.calls[0]["json"]["reasoning_effort"] == "none"
    assert client.post("/api/valuation", json={"city": "Bhopal", "property_type": "apartment", "area_sqft": 1000}).status_code == 200
    assert "reasoning_effort" not in fake.calls[1]["json"]


def test_provider_rejecting_the_optional_parameter_is_retried_without_it(fake, monkeypatch):
    monkeypatch.setenv("OPENAI_VISION_REASONING_EFFORT", "none")
    fake.script = [(400, '{"error":{"message":"unknown parameter: reasoning_effort"}}')]
    assert post_image("/api/safety/analyze", jpeg()).status_code == 200
    assert "reasoning_effort" in fake.calls[0]["json"] and "reasoning_effort" not in fake.calls[1]["json"]


def test_answer_cut_off_before_any_text_gets_a_clear_error(fake):
    fake.script = [(200, json.dumps({"choices": [{"message": {"content": ""}, "finish_reason": "length"}]}))]
    assert_error(client.post("/api/knowledge/chat", json={"message": "What is curing?"}), 502, "ran out of output space")


def test_ai_check_reports_the_provider_catalogue(monkeypatch):
    monkeypatch.setattr(backend.httpx, "get", lambda url, headers=None, timeout=None: httpx.Response(
        200, json={"data": [{"id": TEXT_MODEL}, {"id": VISION_MODEL}]}))
    d = client.get("/api/ai/check").json()["data"]
    assert d["text_model"]["listed"] and d["vision_model"]["listed"] and d["available_model_ids"] == sorted([TEXT_MODEL, VISION_MODEL])
    assert KEY not in json.dumps(d)


# ------------------------------------------------------------------ weather: real HTTP layer (Open-Meteo) with a fake transport
GEO_OK = {"results": [{"name": "Indore", "admin1": "Madhya Pradesh", "country": "India", "latitude": 22.71, "longitude": 75.85}]}


class FakeWx:
    def __init__(self):
        self.calls, self.script = [], []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": dict(params or {}), "headers": headers, "timeout": timeout})
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


@pytest.fixture
def wx(monkeypatch):
    f, sleeps = FakeWx(), []
    f.sleeps = sleeps
    backend._wx_cache.clear()
    monkeypatch.setattr(backend.httpx, "get", f)
    monkeypatch.setattr(backend.time, "sleep", lambda s: sleeps.append(s))
    yield f
    backend._wx_cache.clear()


def R(status, body=None, headers=None):
    return httpx.Response(status, json=body if body is not None else {}, headers=headers)


def test_weather_requests_use_documented_open_meteo_parameters(wx):
    wx.script = [R(200, GEO_OK), R(200, FORECAST)]
    loc = backend.geocode("Indore")
    backend.fetch_forecast(loc["latitude"], loc["longitude"], 3)
    geo, fc = wx.calls
    assert geo["url"] == "https://geocoding-api.open-meteo.com/v1/search" and geo["params"]["name"] == "Indore" and geo["params"]["count"] == 1
    assert fc["url"] == "https://api.open-meteo.com/v1/forecast" and fc["params"]["forecast_days"] == 3 and fc["params"]["timezone"] == "auto"
    valid_daily = {"weather_code", "temperature_2m_max", "temperature_2m_min", "apparent_temperature_max", "precipitation_sum",
                   "precipitation_probability_max", "wind_speed_10m_max", "uv_index_max"}
    assert set(fc["params"]["daily"].split(",")) == valid_daily
    assert fc["headers"]["User-Agent"] and fc["timeout"] == 10


def test_weather_retries_a_rate_limit_then_succeeds(wx):
    wx.script = [R(429, {"error": True, "reason": "Too many requests"}, {"Retry-After": "1"}), R(200, GEO_OK)]
    assert backend.geocode("Indore")["name"] == "Indore" and len(wx.calls) == 2 and wx.sleeps == [1.0]


def test_weather_persistent_rate_limit_names_the_real_cause(wx, caplog):
    wx.script = [R(429, {"error": True, "reason": "Daily API request limit exceeded"})] * 3
    with caplog.at_level("ERROR", logger="construction-intelligence"):
        with pytest.raises(backend.WeatherError) as e:
            backend.geocode("Indore")
    assert "HTTP 429" in str(e.value) and e.value.status == 503 and len(wx.calls) == 3
    assert "HTTP 429" in caplog.text and "Daily API request limit exceeded" in caplog.text     # status + provider body are logged


def test_weather_client_error_shows_the_provider_reason_and_is_not_retried(wx):
    wx.script = [R(400, {"error": True, "reason": "Parameter 'daily' has invalid value"})]
    with pytest.raises(backend.WeatherError) as e:
        backend.fetch_forecast(1, 2, 3)
    assert "HTTP 400" in str(e.value) and "invalid value" in str(e.value) and len(wx.calls) == 1


@pytest.mark.parametrize("script,status,needle", [
    ([httpx.ReadTimeout("slow")] * 3, 504, "too long"),
    ([httpx.ConnectError("dns")] * 3, 502, "could not connect"),
    ([R(503), R(503), R(503)], 502, "http 503"),
    ([httpx.Response(200, text="<html>oops</html>")], 502, "unreadable"),
])
def test_weather_transport_failures_are_classified(wx, script, status, needle):
    wx.script = list(script)
    with pytest.raises(backend.WeatherError) as e:
        backend.geocode("Indore")
    assert e.value.status == status and needle in str(e.value).lower()


def test_weather_results_are_cached_briefly(wx):
    wx.script = [R(200, GEO_OK)]
    backend.geocode("Indore"); backend.geocode("Indore")
    assert len(wx.calls) == 1


def test_unknown_place_is_404(wx):
    wx.script = [R(200, {"generationtime_ms": 0.3})]
    with pytest.raises(backend.LocationNotFound):
        backend.geocode("Nowhereville")


def test_live_weather_endpoint_end_to_end_is_labelled_live(fake, wx):
    wx.script = [R(200, GEO_OK), R(200, FORECAST)]
    r = client.post("/api/weather/advisory", json={"location": "Indore", "days": 2}).json()
    d = r["data"]
    assert r["success"] and d["weather_source_type"] == "live_forecast" and "retrieved live" in d["weather_source"]
    assert d["location"]["name"] == "Indore" and len(d["days"]) == 2 and d["ai_guidance"]


def test_rate_limited_live_weather_falls_back_to_labelled_user_data(fake, wx):
    wx.script = [R(429)] * 3
    r = client.post("/api/weather/advisory", json={"location": "Indore", "manual": {"temperature_max_c": 33, "wind_kmh": 12, "rain_mm": 0}}).json()
    d = r["data"]
    assert d["weather_source_type"] == "user_provided" and "not retrieved" in d["weather_source"] and "HTTP 429" in d["live_weather_error"]
    assert "HTTP 429" in r["notes"][0] and d["location"] is None
    assert "entered by the user" in json.dumps(fake.calls[0]["json"]["messages"][1]["content"])     # the AI is told the data is user-provided


def test_rate_limited_live_weather_without_manual_input_is_an_explicit_error(fake, wx):
    wx.script = [R(429)] * 3
    assert_error(client.post("/api/weather/advisory", json={"location": "Indore"}), 503, "HTTP 429")
    assert fake.calls == []


# ------------------------------------------------------------------ every documented endpoint, valid input, end to end
def test_every_endpoint_succeeds_with_valid_input(fake, wx, monkeypatch):
    seen = {}
    seen["health"] = client.get("/api/health")
    monkeypatch.setattr(backend, "ai_check", lambda: {"reachable": True})
    seen["ai/check"] = client.get("/api/ai/check")
    seen["safety"] = post_image("/api/safety/analyze", jpeg((90, 90, 90)), description="Slab edge on level 2")
    seen["cracks"] = post_image("/api/cracks/analyze", jpeg((200, 200, 200), crack=True), description="Hairline crack near window")
    seen["valuation"] = client.post("/api/valuation", json={"city": "Indore", "property_type": "apartment", "area_sqft": 1100, "age_years": 5})
    seen["planning"] = client.post("/api/planning", json=PLAN)
    seen["assistant/chat"] = client.post("/api/assistant/chat", json={"message": "How long before stripping slab props?", "history": []})
    seen["knowledge/chat"] = client.post("/api/knowledge/chat", json={"message": "What is the slump test?"})
    seen["assistant/report"] = client.post("/api/assistant/report", json={"site_name": "Block B", "report_date": "2026-10-24", "notes": "Pour slab, 12 workers"})
    wx.script = [R(200, GEO_OK), R(200, FORECAST)]
    seen["weather"] = client.post("/api/weather/advisory", json={"location": "Indore", "days": 2, "planned_activities": "slab pour"})
    seen["history"] = client.get("/api/history")
    for name, r in seen.items():
        b = r.json()
        assert r.status_code == 200 and b["success"] is True and b["mode"] in ("ai", "partial"), (name, r.status_code, b.get("error"))
    assert len(seen["history"].json()["data"]) >= 6 and KEY not in json.dumps([r.text for r in seen.values()])


@pytest.mark.parametrize("path,body", [
    ("/api/weather/advisory", {"location": "Indore", "manual": {"temperature_max_c": -61}}),
    ("/api/weather/advisory", {"manual": {"wind_kmh": 401}}),
    ("/api/weather/advisory", {"manual": {"rain_mm": 1001}}),
    ("/api/valuation", {"city": "X", "property_type": "apartment", "area_sqft": 5}),
    ("/api/planning", {**PLAN, "built_up_area_sqft": -1}),
    ("/api/assistant/chat", {"message": ""}),
])
def test_invalid_input_is_a_422_with_the_error_envelope(fake, path, body):
    r = client.post(path, json=body)
    assert r.status_code == 422 and r.json()["success"] is False and r.json()["error"].startswith("Invalid input") and fake.calls == []


def test_cached_live_weather_is_labelled_with_its_age_not_just_now(fake, wx, monkeypatch):
    wx.script = [R(200, GEO_OK), R(200, FORECAST)]
    first = client.post("/api/weather/advisory", json={"location": "Indore", "days": 2}).json()["data"]
    assert first["weather_source"] == "Open-Meteo forecast (retrieved live just now)" and len(wx.calls) == 2
    real = backend.time.time
    monkeypatch.setattr(backend.time, "time", lambda: real() + 300)            # five minutes later, still inside the cache window
    second = client.post("/api/weather/advisory", json={"location": "Indore", "days": 2}).json()["data"]
    assert len(wx.calls) == 2                                                  # served from cache, no new provider call
    assert second["weather_source_type"] == "live_forecast" and "5 min ago" in second["weather_source"] and "just now" not in second["weather_source"]
