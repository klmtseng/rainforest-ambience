"""
tests/verify_panel.py — 面板 smoke test
啟動 server.py → curl GET / → POST /render → GET /reference → 關 server
exit 0 = 全過
"""
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import time
import urllib.request
import urllib.error

import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
PANEL_SERVER = ROOT / "panel" / "server.py"
PYTHON = sys.executable
CLI = ROOT / "render_ambience.py"
PORT = 18765  # 非預設 port，避免衝突

FAIL = 0


def check(label: str, cond: bool, msg: str = ""):
    global FAIL
    if cond:
        print(f"  [PASS] {label}")
    else:
        print(f"  [FAIL] {label}  {msg}")
        FAIL += 1


def get(url: str, timeout: int = 10):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {}
    except Exception as e:
        return None, None, {"error": str(e)}


def post_json(url: str, payload: dict, timeout: int = 90):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), {}
    except Exception as e:
        return None, None, {"error": str(e)}


def render_wav_direct(extra_args: list, seed: int = 77) -> np.ndarray:
    """呼叫 render_ambience.py CLI 渲染 6s 預覽,回傳 float64 audio array。"""
    from scipy.io import wavfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_path = tf.name
    try:
        cmd = [
            PYTHON, str(CLI),
            "--duration", "6",
            "--seed", str(seed),
            "--thunder-rate", "0",
            "--no-rain-gush",
            "--out", out_path,
        ] + extra_args
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode != 0:
            raise RuntimeError(f"render failed:\n{r.stderr[-300:]}")
        sr, data = wavfile.read(out_path)
        return data.astype(np.float64)
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass


def test_knob_effect():
    """四拉桿各自兩極端:渲染兩次輸出必須不同且無 clip(|peak| < 1.0)。"""
    print("[E] 四拉桿效果驗證(CLI 直接) …")

    knobs = [
        ("drop_rate_scale",  ["--drop-rate-scale", "0.3"],  ["--drop-rate-scale", "3.0"]),
        ("drop_amp_sigma",   ["--drop-amp-sigma",  "0.3"],   ["--drop-amp-sigma",  "2.5"]),
        ("spectral_tilt_hi", ["--spectral-tilt-hi", "-14.0"], ["--spectral-tilt-hi", "-2.0"]),
        ("bed_mix",          ["--bed-mix", "0.0"],            ["--bed-mix", "1.0"]),
    ]

    for name, lo_args, hi_args in knobs:
        try:
            a_lo = render_wav_direct(lo_args)
            a_hi = render_wav_direct(hi_args)
        except Exception as e:
            check(f"E.{name}.render", False, str(e))
            continue

        arrays_differ = not np.array_equal(a_lo, a_hi)
        check(f"E.{name}.differs", arrays_differ,
              "兩極端輸出相同(拉桿無效)" if not arrays_differ else "")

        peak_lo = np.max(np.abs(a_lo))
        peak_hi = np.max(np.abs(a_hi))
        no_clip = peak_lo < 1.0 and peak_hi < 1.0
        check(f"E.{name}.no_clip", no_clip,
              f"peak_lo={peak_lo:.4f} peak_hi={peak_hi:.4f}" if not no_clip else
              f"peak_lo={peak_lo:.4f} peak_hi={peak_hi:.4f}")


def main():
    print("=== verify_panel.py ===")

    # 啟動伺服器
    proc = subprocess.Popen(
        [PYTHON, str(PANEL_SERVER), "--port", str(PORT)],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=str(ROOT),
    )
    base = f"http://127.0.0.1:{PORT}"

    # 等待啟動（最多 10s）
    for _ in range(20):
        try:
            urllib.request.urlopen(f"{base}/", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.terminate()
        print("[FAIL] 伺服器未在 10s 內啟動")
        sys.exit(1)

    print("[INFO] 伺服器已啟動")

    try:
        # 1. GET / → 200 + HTML
        status, body, _ = get(f"{base}/")
        check("GET / → 200", status == 200, f"status={status}")
        check("GET / → HTML 含拉桿", body and b"rain_intensity" in body, "body missing rain_intensity")

        # 2. POST /render → audio/mpeg bytes > 50KB
        params = {
            "rain_intensity": 0.55,
            "rain_variability": 0.08,
            "drop_rate_scale": 1.0,
            "drop_amp_sigma": 1.3,
            "spectral_tilt_hi": -7.0,
            "spectral_tilt_lo": -4.5,
            "bed_mix": 0.75,
            "thunder_rate": 0.0,   # 不渲染雷，加快測試
            "thunder_dist_min": 2.0,
            "thunder_dist_max": 8.0,
            "rain_gush": True,
            "critters": False,
            "duration": 8,
            "seed": 77,
        }

        t0 = time.time()
        status, body, headers = post_json(f"{base}/render", params, timeout=90)
        elapsed = time.time() - t0

        print(f"[INFO] /render 耗時 {elapsed:.1f}s，回應 {len(body) if body else 0} bytes")
        check("POST /render → 200", status == 200, f"status={status}, body[:200]={body[:200] if body else ''}")
        check("POST /render → audio/mpeg", "audio/mpeg" in headers.get("Content-Type", ""), headers.get("Content-Type"))
        check("POST /render → >50KB", body is not None and len(body) > 50_000, f"{len(body) if body else 0} bytes")
        check("POST /render → 耗時記錄", elapsed > 0, f"{elapsed:.1f}s")

        # 3. GET /reference/user → 200 或 404（取決於檔案存在）
        status_ref, body_ref, hdrs_ref = get(f"{base}/reference/user")
        ref_exists = (ROOT / "assets" / "reference" / "user_rain_reference.wav").exists()
        if ref_exists:
            check("GET /reference/user → 200 + audio", status_ref == 200, f"status={status_ref}")
        else:
            check("GET /reference/user → 404 (file not present)", status_ref == 404, f"status={status_ref}")

        # 4. GET /reference/cc0
        status_cc0, _, _ = get(f"{base}/reference/cc0")
        cc0_exists = (ROOT / "assets" / "curated" / "dark_rainy_night.wav").exists()
        if cc0_exists:
            check("GET /reference/cc0 → 200", status_cc0 == 200, f"status={status_cc0}")
        else:
            check("GET /reference/cc0 → 404 (file not present)", status_cc0 == 404, f"status={status_cc0}")

        # 5. 四拉桿效果驗證（CLI 直呼叫，不需 server）
        test_knob_effect()

    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    print(f"\n渲染耗時: {elapsed:.1f}s（預覽 {params['duration']}s）")
    if FAIL == 0:
        print("=== ALL PASS ===")
        sys.exit(0)
    else:
        print(f"=== {FAIL} FAIL(s) ===")
        sys.exit(1)


if __name__ == "__main__":
    main()
