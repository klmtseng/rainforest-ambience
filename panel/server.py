"""
panel/server.py — 雨夜音景微調面板 HTTP 伺服器
用法: python panel/server.py [--port 8765]
只綁 127.0.0.1，不對外。
"""

import argparse
import json
import os
import shutil
import pathlib
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

ROOT = pathlib.Path(__file__).parent.parent
PANEL_DIR = pathlib.Path(__file__).parent
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
RENDER_SCRIPT = ROOT / "render_ambience.py"
PYTHON = sys.executable

# 參考錄音路徑
REFERENCE_DIR = ROOT / "assets" / "reference"
USER_REFERENCE = REFERENCE_DIR / "user_rain_reference.wav"
CC0_REFERENCE = ROOT / "assets" / "curated" / "dark_rainy_night.wav"

# 全域快取：參考音頻 mp3 bytes
_reference_cache: dict[str, bytes] = {}
_reference_lock = threading.Lock()


def wav_to_mp3_bytes(wav_path: pathlib.Path, bitrate: str = "128k") -> bytes:
    """用 ffmpeg 將 WAV 轉成 MP3 bytes，回傳 bytes。"""
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tf:
        tmp_mp3 = tf.name
    try:
        subprocess.run(
            [FFMPEG, "-y", "-i", str(wav_path),
             "-codec:a", "libmp3lame", "-b:a", bitrate,
             "-ac", "1",   # mono
             tmp_mp3],
            capture_output=True, check=True,
        )
        with open(tmp_mp3, "rb") as f:
            return f.read()
    finally:
        try:
            os.unlink(tmp_mp3)
        except OSError:
            pass


def render_preview(params: dict) -> tuple[bytes, dict]:
    """
    呼叫 render_ambience.py subprocess，產生預覽 WAV，轉 MP3 bytes 回傳。
    回傳 (mp3_bytes, stats_dict)。
    """
    duration = float(params.get("duration", 10))
    seed = int(params.get("seed", 77))
    rain_intensity = float(params.get("rain_intensity", 0.55))
    rain_variability = float(params.get("rain_variability", 0.08))
    thunder_rate = float(params.get("thunder_rate", 0.0))
    thunder_dist_min = float(params.get("thunder_dist_min", 2.0))
    thunder_dist_max = float(params.get("thunder_dist_max", 8.0))
    rain_gush = bool(params.get("rain_gush", True))
    critters = bool(params.get("critters", False))
    # 四拉桿（None 表示面板未傳值,使用 preset 預設）
    drop_rate_scale = params.get("drop_rate_scale")
    drop_amp_sigma = params.get("drop_amp_sigma")
    spectral_tilt_hi = params.get("spectral_tilt_hi")
    spectral_tilt_lo = params.get("spectral_tilt_lo")
    bed_mix = params.get("bed_mix")

    # 限制預覽長度（面板最多 30 秒）
    duration = max(4.0, min(30.0, duration))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        out_wav = tf.name

    try:
        t0 = time.time()
        cmd = [
            PYTHON, str(RENDER_SCRIPT),
            "--duration", str(duration),
            "--seed", str(seed),
            "--rain-intensity", str(rain_intensity),
            "--rain-variability", str(rain_variability),
            "--thunder-rate", str(thunder_rate),
            "--thunder-distance-range", str(thunder_dist_min), str(thunder_dist_max),
            "--out", out_wav,
        ]
        if not rain_gush:
            cmd.append("--no-rain-gush")
        if critters:
            cmd.append("--critters")
        if drop_rate_scale is not None:
            cmd += ["--drop-rate-scale", str(float(drop_rate_scale))]
        if drop_amp_sigma is not None:
            cmd += ["--drop-amp-sigma", str(float(drop_amp_sigma))]
        if spectral_tilt_hi is not None:
            cmd += ["--spectral-tilt-hi", str(float(spectral_tilt_hi))]
        if spectral_tilt_lo is not None:
            cmd += ["--spectral-tilt-lo", str(float(spectral_tilt_lo))]
        if bed_mix is not None:
            cmd += ["--bed-mix", str(float(bed_mix))]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - t0

        if result.returncode != 0:
            raise RuntimeError(f"render failed:\n{result.stderr[-500:]}")

        # 轉 MP3
        mp3_bytes = wav_to_mp3_bytes(pathlib.Path(out_wav))

        stats = {
            "render_time_s": round(elapsed, 2),
            "duration_s": duration,
            "file_size_bytes": len(mp3_bytes),
        }
        return mp3_bytes, stats

    finally:
        try:
            os.unlink(out_wav)
        except OSError:
            pass


def get_reference_mp3(kind: str) -> bytes | None:
    """取得參考錄音 MP3 bytes（有快取）。kind='user' 或 'cc0'。"""
    with _reference_lock:
        if kind in _reference_cache:
            return _reference_cache[kind]

    src = USER_REFERENCE if kind == "user" else CC0_REFERENCE
    if not src.exists():
        return None

    data = wav_to_mp3_bytes(src, bitrate="128k")
    with _reference_lock:
        _reference_cache[kind] = data
    return data


class PanelHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        # 簡化日誌
        ts = time.strftime("%H:%M:%S")
        print(f"[{ts}] {self.command} {self.path} -> {args[1] if len(args) > 1 else ''}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            html_path = PANEL_DIR / "index.html"
            self._serve_file(html_path, "text/html; charset=utf-8")

        elif path == "/reference/user":
            self._serve_reference("user")

        elif path == "/reference/cc0":
            self._serve_reference("cc0")

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/render":
            self._handle_render()
        elif path == "/save-preset":
            self._handle_save_preset()
        else:
            self._send_json({"error": "not found"}, 404)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        return json.loads(body) if body else {}

    def _handle_render(self):
        try:
            params = self._read_json_body()
            mp3_bytes, stats = render_preview(params)
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(len(mp3_bytes)))
            self.send_header("X-Render-Stats", json.dumps(stats))
            self.send_header("Access-Control-Expose-Headers", "X-Render-Stats")
            self.end_headers()
            self.wfile.write(mp3_bytes)
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _handle_save_preset(self):
        try:
            data = self._read_json_body()
            preset_dir = ROOT / "config"
            preset_dir.mkdir(exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            name = data.get("name", f"panel_{ts}").replace(" ", "_")
            out_path = preset_dir / f"preset_panel_{name}.json"
            with open(out_path, "w") as f:
                json.dump(data.get("params", data), f, indent=2, ensure_ascii=False)
            self._send_json({"saved": str(out_path)})
        except Exception as e:
            self._send_json({"error": str(e)}, 500)

    def _serve_reference(self, kind: str):
        data = get_reference_mp3(kind)
        if data is None:
            self._send_json({"error": f"reference '{kind}' not found"}, 404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/mpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _serve_file(self, path: pathlib.Path, content_type: str):
        if not path.exists():
            self._send_json({"error": "file not found"}, 404)
            return
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, obj: dict, status: int = 200):
        body = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description="雨夜音景微調面板伺服器")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), PanelHandler)
    print(f"[panel] 啟動  http://{args.host}:{args.port}/")
    print(f"[panel] 用 Firefox 開啟  http://localhost:{args.port}/")
    print(f"[panel] Ctrl+C 停止")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[panel] 停止")


if __name__ == "__main__":
    main()
