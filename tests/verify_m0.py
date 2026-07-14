"""
verify_m0.py — M0 骨架冒煙驗收
exit 0 = PASS;exit 1 = FAIL
驗收項目:
  1. 目錄結構全到位
  2. 用 scipy.io.wavfile 寫 5s 白噪 WAV → 讀回驗時長誤差 < 0.1s
  3. ffmpeg 產 5s 測試片 WAV → 驗檔案存在且時長 > 4.9s
  4. blender -b 渲 1 幀 PNG → 存在(若起不來記錄錯誤並警告,不阻止整體 PASS)
"""

import sys
import os
import shutil
import subprocess
import pathlib
import tempfile
import time

import numpy as np
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).parent.parent
FFMPEG = pathlib.Path(os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg")
SR = 48000

PASS = []
WARN = []
FAIL = []


def check(name, cond, msg=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {msg}")


def warn(name, msg):
    WARN.append(name)
    print(f"  WARN  {name}  {msg}")


# ---- 1. 目錄結構 ----
print("\n[M0] 目錄結構")
required_dirs = [
    "config", "synth", "assets/fetch", "assets/curated",
    "mix", "video/ffmpeg_line", "video/blender_line",
    "out/stems", "out/loops", "out/long", "out/video", "tests",
]
required_files = [
    "CLAUDE.md", "PROGRESS.md",
    "config/preset_default.json",
    "synth/rain.py", "synth/thunder.py", "synth/render.py",
]
for d in required_dirs:
    check(f"dir:{d}", (ROOT / d).is_dir(), f"missing {ROOT/d}")
for fp in required_files:
    check(f"file:{fp}", (ROOT / fp).is_file(), f"missing {ROOT/fp}")


# ---- 2. scipy WAV 5s 白噪 ----
print("\n[M0] scipy WAV 寫 5s 白噪")
wav_path = ROOT / "out/stems/smoke_whitenoise.wav"
rng = np.random.default_rng(0)
noise = rng.standard_normal(5 * SR).astype(np.float32) * 0.1
wavfile.write(str(wav_path), SR, noise)

sr_read, data_read = wavfile.read(str(wav_path))
dur_actual = len(data_read) / sr_read
check("wav_exists", wav_path.is_file(), str(wav_path))
check("wav_sr", sr_read == SR, f"got {sr_read}")
check("wav_duration", abs(dur_actual - 5.0) < 0.1, f"dur={dur_actual:.4f}s")


# ---- 3. ffmpeg 5s 測試片 ----
print("\n[M0] ffmpeg 5s 測試片")
ffmpeg_out = ROOT / "out/stems/smoke_ffmpeg.wav"
if not FFMPEG.is_file():
    warn("ffmpeg_binary", f"ffmpeg not found at {FFMPEG}")
else:
    cmd = [
        str(FFMPEG), "-y",
        "-f", "lavfi",
        "-i", "anoisesrc=color=white:sample_rate=48000:duration=5",
        "-t", "5",
        str(ffmpeg_out)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    check("ffmpeg_exit0", result.returncode == 0,
          f"stderr: {result.stderr[-200:]}")
    if result.returncode == 0:
        check("ffmpeg_out_exists", ffmpeg_out.is_file())
        if ffmpeg_out.is_file():
            sr_ff, data_ff = wavfile.read(str(ffmpeg_out))
            dur_ff = len(data_ff) / sr_ff
            check("ffmpeg_duration", dur_ff > 4.9, f"dur={dur_ff:.4f}s")


# ---- 4. blender -b 渲 1 幀 ----
print("\n[M0] blender 渲 1 幀 PNG")
blender_out = ROOT / "out/video/smoke_frame.png"
blender_out.parent.mkdir(parents=True, exist_ok=True)

# 最小 blender python 腳本:渲染 1 幀白色場景
blend_script = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
blend_script.write(f"""
import bpy
bpy.context.scene.render.engine = 'BLENDER_WORKBENCH'
bpy.context.scene.render.filepath = '{str(blender_out)}'
bpy.context.scene.render.image_settings.file_format = 'PNG'
bpy.context.scene.render.resolution_x = 64
bpy.context.scene.render.resolution_y = 64
bpy.context.scene.frame_start = 1
bpy.context.scene.frame_end = 1
bpy.ops.render.render(write_still=True)
""")
blend_script.close()

blender_cmd = ["blender", "-b", "--python", blend_script.name]
try:
    t0 = time.time()
    result_b = subprocess.run(blender_cmd, capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0
    os.unlink(blend_script.name)

    if result_b.returncode == 0 and blender_out.is_file():
        check("blender_frame", True)
        print(f"        blender render: {elapsed:.1f}s")
    else:
        # blender 起不來不阻止 M0 PASS,記錄警告
        warn("blender_frame",
             f"returncode={result_b.returncode} "
             f"stderr_tail={result_b.stderr[-300:]}")
except subprocess.TimeoutExpired:
    os.unlink(blend_script.name)
    warn("blender_frame", "timeout 120s — blender headless too slow on this machine")
except FileNotFoundError:
    warn("blender_frame", "blender not found in PATH")


# ---- 結果 ----
print(f"\n[M0] PASS={len(PASS)}  WARN={len(WARN)}  FAIL={len(FAIL)}")
if WARN:
    print("  Warnings:", WARN)
if FAIL:
    print("  Failures:", FAIL)
    sys.exit(1)

print("[M0] ALL PASS")
sys.exit(0)
