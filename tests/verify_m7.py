#!/usr/bin/env python3
"""
verify_m7.py — M7 Blender 線驗收腳本

驗收條件(計畫文件釘死,builder 不得改):
  V7.1  240 幀 PNG 齊全
  V7.2  MP4 幀數正確(240 幀 ±2)且可播(ffprobe)
  V7.3  單幀耗時中位數記錄在 PROGRESS.md
  SKIP  若 Blender 被標 BLOCKED(見 blender_blocked.txt),exit 0 並附診斷路徑

Exit code:
  0 = ALL PASS 或 正當 SKIP
  1 = FAIL
"""

import os
import shutil
import sys
import subprocess
import re
import pathlib
import json

PROJ = pathlib.Path(__file__).parent.parent
FRAMES_DIR = PROJ / "out/video/frames_blender"
OUT_MP4 = PROJ / "out/video/blender_loop_10s.mp4"
PROGRESS_MD = PROJ / "PROGRESS.md"
BLOCKED_FILE = PROJ / "out/video/blender_blocked.txt"
TIMING_FILE = FRAMES_DIR / "timing_summary.txt"
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"

results = []
all_pass = True
is_skip = False

def check(name, cond, detail=""):
    global all_pass
    status = "PASS" if cond else "FAIL"
    if not cond:
        all_pass = False
    results.append(f"  [{status}] {name}" + (f": {detail}" if detail else ""))
    return cond

# ── SKIP path ─────────────────────────────────────────────────────────────────
if BLOCKED_FILE.exists():
    is_skip = True
    diag = BLOCKED_FILE.read_text().strip()
    print("[verify_m7] SKIP — Blender line BLOCKED")
    print(f"  Diagnosis: {diag}")
    print(f"  Blocked file: {BLOCKED_FILE}")
    sys.exit(0)

# ── V7.1 : 240 PNG frames exist ───────────────────────────────────────────────
frames = sorted(FRAMES_DIR.glob("frame_*.png"))
n_frames = len(frames)
check("V7.1 >= 240 frames rendered", n_frames >= 240,
      f"found {n_frames} PNGs in {FRAMES_DIR}")

# ── V7.2 : MP4 playable with correct frame count ──────────────────────────────
if OUT_MP4.exists():
    try:
        # Use ffprobe to count frames
        ffprobe_path = os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe"
        if not os.path.exists(ffprobe_path):
            ffprobe_path = FFMPEG.replace("ffmpeg", "ffprobe")
        cmd = [
            ffprobe_path, "-v", "error",
            "-select_streams", "v:0",
            "-count_frames",
            "-show_entries", "stream=nb_read_frames,r_frame_rate",
            "-of", "json",
            str(OUT_MP4)
        ]
        out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, timeout=60)
        info = json.loads(out.decode())
        stream = info.get("streams", [{}])[0]
        nb_frames = int(stream.get("nb_read_frames", 0))
        fps_str = stream.get("r_frame_rate", "0/1")
        fps_num, fps_den = map(int, fps_str.split("/"))
        fps = fps_num / fps_den if fps_den else 0

        check("V7.2a MP4 frame count 238-242", 238 <= nb_frames <= 242,
              f"got {nb_frames} frames")
        check("V7.2b MP4 fps=24", abs(fps - 24) < 0.1, f"got {fps:.2f}fps")
    except subprocess.CalledProcessError as e:
        check("V7.2 MP4 playable", False, f"ffprobe error: {e.output.decode()[:200]}")
    except Exception as e:
        check("V7.2 MP4 playable", False, str(e))
else:
    check("V7.2 MP4 exists", False, f"not found: {OUT_MP4}")

# ── V7.3 : timing recorded ────────────────────────────────────────────────────
median_seconds = None
if TIMING_FILE.exists():
    txt = TIMING_FILE.read_text()
    m = re.search(r"median_seconds=([0-9.]+)", txt)
    if m:
        median_seconds = float(m.group(1))
        check("V7.3 median frame time recorded", True, f"{median_seconds:.1f}s/frame")
        check("V7.3a median < 60s", median_seconds < 60.0, f"{median_seconds:.1f}s")
    else:
        check("V7.3 timing file parseable", False, "no median_seconds line")
else:
    # Timing file missing but frames exist — compute from file timestamps roughly
    if n_frames >= 5:
        check("V7.3 timing file present", False, f"missing: {TIMING_FILE}")

# ── Print results ─────────────────────────────────────────────────────────────
print("\n[verify_m7] M7 BLENDER LINE — RESULTS")
print(f"  Frames dir: {FRAMES_DIR}")
print(f"  MP4: {OUT_MP4}")
for r in results:
    print(r)

if all_pass:
    print("\n[verify_m7] ALL PASS")

    # ── V7.3 write timing to PROGRESS.md ──────────────────────────────────────
    if median_seconds is not None and PROGRESS_MD.exists():
        prog_text = PROGRESS_MD.read_text()
        timing_line = f"\n## M7 記錄（verify_m7.py ALL PASS）\n\n" \
                      f"- 引擎：Cycles CPU 8 samples，無 denoiser\n" \
                      f"- 解析度：960×540 @ 24fps，{n_frames} 幀渲完\n" \
                      f"- 單幀中位耗時：**{median_seconds:.1f}s**（目標 <60s）\n" \
                      f"- 輸出：`out/video/frames_blender/` ({n_frames} PNG) + `out/video/blender_loop_10s.mp4`\n" \
                      f"- Blender crash 診斷：BLENDER_WORKBENCH 需 GLX/EGL context → abort (exit 134)\n" \
                      f"  退路鏈：EEVEE→Workbench(同問題)→**Cycles CPU** ✓（無需 GL context）\n"

        # Only append if M7 section not already present
        if "M7 記錄" not in prog_text:
            with open(PROGRESS_MD, "a") as f:
                f.write(timing_line)
            print(f"[verify_m7] Timing written to PROGRESS.md")

    sys.exit(0)
else:
    print("\n[verify_m7] FAIL")
    sys.exit(1)
