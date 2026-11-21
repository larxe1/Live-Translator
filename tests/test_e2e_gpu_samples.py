from __future__ import annotations

import base64
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import httpx
import pytest

from server.translator.translator import is_refusal

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_URL = "http://127.0.0.1:8765"


def _box_iou(a: dict[str, int], b: dict[str, int]) -> float:
    ix1 = max(a["x1"], b["x1"])
    iy1 = max(a["y1"], b["y1"])
    ix2 = min(a["x2"], b["x2"])
    iy2 = min(a["y2"], b["y2"])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    area_a = max(1, (a["x2"] - a["x1"]) * (a["y2"] - a["y1"]))
    area_b = max(1, (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))
    return inter / float(area_a + area_b - inter)


def _is_server_up() -> bool:
    try:
        r = httpx.get(f"{BASE_URL}/health", timeout=1.5)
        return r.status_code == 200
    except Exception:
        return False


def _wait_until_ready(timeout_s: float = 150.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    last_payload: dict[str, Any] = {}
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{BASE_URL}/health", timeout=3.0)
            if r.status_code == 200:
                last_payload = r.json()
                if last_payload.get("ready") is True:
                    return last_payload
                if str(last_payload.get("status", "")).lower() == "error":
                    raise RuntimeError(f"Server startup error: {last_payload}")
        except httpx.HTTPError:
            pass
        time.sleep(1.5)
    raise TimeoutError(f"Server did not become ready within {timeout_s}s (last={last_payload})")


@pytest.fixture(scope="module")
def live_server():
    started_here = False
    proc: subprocess.Popen[str] | None = None
    log_file = None

    if not _is_server_up():
        env = os.environ.copy()
        cache_root = PROJECT_ROOT / "cache"
        cache_root.mkdir(parents=True, exist_ok=True)
        env["HF_HOME"] = str(cache_root / "huggingface")
        env["TORCH_HOME"] = str(cache_root / "torch")
        env["EASYOCR_MODULE_PATH"] = str(cache_root / "easyocr")
        env["PIP_CACHE_DIR"] = str(cache_root / "pip")
        env["XDG_CACHE_HOME"] = str(cache_root / "xdg")
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(cache_root / "playwright")

        log_path = cache_root / "e2e_server.log"
        log_file = open(log_path, "w", encoding="utf-8", buffering=1)

        proc = subprocess.Popen(
            [sys.executable, "-m", "server.app"],
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        started_here = True

    try:
        health = _wait_until_ready(timeout_s=180.0)
        yield health
    finally:
        if started_here and proc is not None:
            try:
                httpx.post(f"{BASE_URL}/shutdown", timeout=3.0)
                proc.wait(timeout=15.0)
            except Exception:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except Exception:
                    proc.kill()
        if log_file is not None:
            log_file.close()


@pytest.mark.gpu
def test_e2e_gpu_health_jp_centipede_and_ko_webtoon(live_server: dict[str, Any]) -> None:
    assert live_server["ready"] is True
    assert live_server["llm_ready"] is True
    vram_used = int(live_server.get("vram_used_mb", 0))
    assert vram_used < 7200, f"VRAM usage too high on 8GB RTX 3070: {vram_used} MB"

    jp_path = PROJECT_ROOT / "assets" / "samples" / "jp_centipede.jpg"
    ko_path = PROJECT_ROOT / "assets" / "samples" / "ko_webtoon_strip.png"
    assert jp_path.exists(), f"Missing sample: {jp_path}"
    assert ko_path.exists(), f"Missing sample: {ko_path}"

    with httpx.Client(base_url=BASE_URL, timeout=60.0) as client:
        # Clear cache so we measure real inference on cold + warm pass
        assert client.post("/cache/clear").status_code == 200
        assert client.post("/memory/clear", json={"chapter_id": "e2e_jp"}).status_code == 200

        jp_b64 = base64.b64encode(jp_path.read_bytes()).decode("ascii")
        # First pass (warms CUDA kernels if needed)
        r_cold = client.post(
            "/translate",
            json={
                "image_base64": jp_b64,
                "lang": "auto",
                "chapter_id": "e2e_jp",
                "priority": 0,
                "glossary": {"藍星": "Ranshin"},
            },
        )
        assert r_cold.status_code == 200, r_cold.text

        # Second uncached pass measures warm GPU latency
        assert client.post("/cache/clear").status_code == 200
        assert client.post("/memory/clear", json={"chapter_id": "e2e_jp"}).status_code == 200
        r_jp = client.post(
            "/translate",
            json={
                "image_base64": jp_b64,
                "lang": "auto",
                "chapter_id": "e2e_jp",
                "priority": 0,
                "glossary": {"藍星": "Ranshin"},
            },
        )
        assert r_jp.status_code == 200, r_jp.text
        jp_data = r_jp.json()
        print(f"\n[E2E JP] elapsed_ms={jp_data['elapsed_ms']}, bubbles={len(jp_data['bubbles'])}")
        for b in jp_data["bubbles"]:
            print(f"  JP #{b['id']} dark={b['is_dark']} {b['original']!r} -> {b['translated']!r}")

        assert jp_data["lang"] == "ja"
        assert jp_data["elapsed_ms"] < 4500, f"Warm JP latency too slow: {jp_data['elapsed_ms']} ms"
        assert len(jp_data["bubbles"]) >= 4

        joined_en = " ".join(b["translated"] for b in jp_data["bubbles"]).lower()
        assert "ranshin" in joined_en, f"Expected 'Ranshin' in translations: {joined_en}"
        assert any(w in joined_en for w in ("help", "save")), (
            f"Expected 'help'/'save' in translations: {joined_en}"
        )
        # Note: rare hyougai kanji 蜈/蚣 (蜈蚣) are OOV in kha-white/manga-ocr-base vocab.txt,
        # whereas in-vocab 蟲 on the same page translates cleanly to 'Insect'/'bug'.
        assert any(w in joined_en for w in ("centipede", "insect", "bug")), (
            f"Expected 'centipede'/'insect'/'bug' in translations: {joined_en}"
        )

        dark_bubbles = [b for b in jp_data["bubbles"] if b["is_dark"] is True]
        assert dark_bubbles, f"Expected at least one dark screaming bubble: {jp_data['bubbles']}"
        assert all(b["text_color"] == "#FFFFFF" for b in dark_bubbles)
        dark_en = " ".join(b["translated"] for b in dark_bubbles).lower()
        assert any(tok in dark_en for tok in ("gya", "aah")), (
            f"Expected scream translation in dark bubble: {dark_bubbles}"
        )

        img_w = jp_data["width"]
        img_h = jp_data["height"]
        for b in jp_data["bubbles"]:
            assert b.get("region_type") in ("bubble", "narration", "floating")
            assert not is_refusal(b["translated"]), f"Refusal leaked into bubble: {b}"
            if b.get("region_type") != "floating":
                assert "カサ" not in b["original"], f"Floating SFX leaked into non-floating bubble: {b}"
                cx = (b["bbox"]["x1"] + b["bbox"]["x2"]) / 2.0
                cy = (b["bbox"]["y1"] + b["bbox"]["y2"]) / 2.0
                in_sfx_region = (0.30 * img_w <= cx <= 0.58 * img_w) and (0.0 <= cy <= 0.29 * img_h)
                assert not in_sfx_region, f"Top-left floating SFX region covered by non-floating bubble: {b}"

        # With include_floating=False, floating artwork SFX MUST be excluded completely
        r_jp_no_float = client.post(
            "/translate",
            json={
                "image_base64": jp_b64,
                "lang": "auto",
                "chapter_id": "e2e_jp_no_float",
                "priority": 0,
                "glossary": {"藍星": "Ranshin"},
                "include_floating": False,
            },
        )
        assert r_jp_no_float.status_code == 200, r_jp_no_float.text
        for b in r_jp_no_float.json()["bubbles"]:
            assert b.get("region_type") != "floating", f"Unexpected floating bubble: {b}"
            assert "カサ" not in b["original"], f"Floating SFX leaked when include_floating=False: {b}"
            cx = (b["bbox"]["x1"] + b["bbox"]["x2"]) / 2.0
            cy = (b["bbox"]["y1"] + b["bbox"]["y2"]) / 2.0
            in_sfx_region = (0.30 * img_w <= cx <= 0.58 * img_w) and (0.0 <= cy <= 0.29 * img_h)
            assert not in_sfx_region, f"Top-left SFX region covered when include_floating=False: {b}"

        # Third call must be an instant cache hit
        r_jp_cached = client.post(
            "/translate",
            json={
                "image_base64": jp_b64,
                "lang": "auto",
                "chapter_id": "e2e_jp",
                "priority": 0,
                "glossary": {"藍星": "Ranshin"},
            },
        )
        assert r_jp_cached.status_code == 200
        assert r_jp_cached.json()["cached"] is True

        # Korean tall webtoon strip test
        assert client.post("/memory/clear", json={"chapter_id": "e2e_ko"}).status_code == 200
        ko_b64 = base64.b64encode(ko_path.read_bytes()).decode("ascii")
        r_ko = client.post(
            "/translate",
            json={
                "image_base64": ko_b64,
                "lang": "auto",
                "chapter_id": "e2e_ko",
                "priority": 0,
            },
        )
        assert r_ko.status_code == 200, r_ko.text
        ko_data = r_ko.json()
        print(f"[E2E KO] elapsed_ms={ko_data['elapsed_ms']}, bubbles={len(ko_data['bubbles'])}")
        for b in ko_data["bubbles"]:
            print(f"  KO #{b['id']} dark={b['is_dark']} {b['original']!r} -> {b['translated']!r}")

        assert ko_data["lang"] == "ko"
        assert len(ko_data["bubbles"]) >= 3
        for b in ko_data["bubbles"]:
            assert b["translated"].strip()
            assert not is_refusal(b["translated"]), f"Refusal leaked into KO bubble: {b}"

        # Verify seam bubbles were not duplicated
        for i in range(len(ko_data["bubbles"])):
            for j in range(i + 1, len(ko_data["bubbles"])):
                iou = _box_iou(ko_data["bubbles"][i]["bbox"], ko_data["bubbles"][j]["bbox"])
                assert iou < 0.5, f"Duplicate overlapping seam bubbles detected: {iou}"

        # Verify dashboard & /test pages serve cleanly
        assert client.get("/").status_code == 200
        assert client.get("/test").status_code == 200


def _launch_playwright_browser(p: Any):
    """Launch system Edge first (zero C: Playwright download), falling back to Chrome or bundled Chromium."""
    last_exc: Exception | None = None
    for kwargs in (
        {"channel": "msedge", "headless": True},
        {"channel": "chrome", "headless": True},
        {"headless": True},
    ):
        try:
            return p.chromium.launch(**kwargs)
        except Exception as exc:
            last_exc = exc
    raise RuntimeError(f"Unable to launch Playwright browser: {last_exc}")


@pytest.mark.gpu
def test_e2e_test_page_browser_renders_cleanly(live_server: dict[str, Any]) -> None:
    pw = pytest.importorskip("playwright.sync_api")
    console_errors: list[str] = []

    with pw.sync_playwright() as p:
        browser = _launch_playwright_browser(p)
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 1000})
            page.on("console", lambda msg: console_errors.append(msg.text) if msg.type == "error" else None)
            page.on("pageerror", lambda exc: console_errors.append(str(exc)))

            page.goto(f"{BASE_URL}/test", wait_until="networkidle")
            page.click("#translate-btn")
            page.wait_for_selector(".mlt-overlay .mlt-bubble", timeout=20000)

            bubbles = page.locator(".mlt-overlay .mlt-bubble")
            assert bubbles.count() >= 4

            shot_path = PROJECT_ROOT / "cache" / "e2e_test_screenshot.png"
            shot_path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(shot_path), full_page=True)
            assert console_errors == [], f"Console errors on /test: {console_errors}"
        finally:
            browser.close()
