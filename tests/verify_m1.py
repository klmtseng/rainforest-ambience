"""
verify_m1.py — M1 雨聲驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收清單(閘門凍結,不准修改數字):
  V1.1  render.py --dur 60 --seed 1 → 生成 rain_1.wav 時長 60±0.1s
  V1.2  RMS ∈ [-30, -12] dBFS
  V1.3  峰值 < -1 dBFS
  V1.4  1k-8kHz 能量占比 > 30%
  V1.5  同 seed 兩次 bit-identical (比較 int32 bytes)
  V1.6  渲染耗時 < 20s (即時率 ≥ 3×)
"""

import sys
import subprocess
import pathlib
import time
import hashlib

import numpy as np
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).parent.parent
RENDER = ROOT / "synth/render.py"
SR = 48000

PASS = []
FAIL = []


def check(name, cond, msg=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {msg}")


print("\n[M1] 雨聲驗收")

# ---- V1.6 渲染時間 + V1.1 時長 ----
out_dir = ROOT / "out/stems"
rain_path = out_dir / "rain_1.wav"

t0 = time.time()
result = subprocess.run(
    [sys.executable, str(RENDER), "--dur", "60", "--seed", "1",
     "--out-dir", str(out_dir)],
    capture_output=True, text=True, cwd=str(ROOT)
)
elapsed = time.time() - t0

check("V1.6_render_time", elapsed < 20.0, f"elapsed={elapsed:.2f}s (limit 20s)")

if result.returncode != 0:
    print(f"  render.py FAILED (exit {result.returncode})")
    print(result.stderr[-500:])
    sys.exit(1)

print(f"        render output:\n{result.stdout.strip()}")

# ---- 讀 WAV ----
sr_r, data_raw = wavfile.read(str(rain_path))
# 支援 float32（新格式）或 int32 24-bit（舊格式）
if data_raw.dtype in (np.float32, np.float64):
    data = data_raw.astype(np.float64)
elif data_raw.dtype == np.int32:
    data = data_raw.astype(np.float64) / (2 ** 23)
else:
    data = data_raw.astype(np.float64) / 32768.0

dur = len(data) / sr_r
check("V1.1_duration", abs(dur - 60.0) < 0.1, f"dur={dur:.4f}s")

# ---- V1.2 RMS ----
rms = np.sqrt(np.mean(data ** 2))
rms_db = 20 * np.log10(rms + 1e-12)
check("V1.2_rms", -30.0 <= rms_db <= -12.0, f"RMS={rms_db:.2f}dBFS (want -30 to -12)")

# ---- V1.3 峰值 ----
peak = np.max(np.abs(data))
peak_db = 20 * np.log10(peak + 1e-12)
check("V1.3_peak", peak_db < -1.0, f"peak={peak_db:.2f}dBFS (want < -1)")

# ---- V1.4 頻率能量占比 ----
from scipy.signal import welch
freqs, psd = welch(data, fs=sr_r, nperseg=4096)
mask_band = (freqs >= 1000) & (freqs <= 8000)
mask_total = freqs > 0
energy_band = np.trapezoid(psd[mask_band], freqs[mask_band])
energy_total = np.trapezoid(psd[mask_total], freqs[mask_total])
ratio = energy_band / (energy_total + 1e-12)
check("V1.4_freq_ratio", ratio > 0.30,
      f"1k-8kHz ratio={ratio*100:.1f}% (want >30%)")

# ---- V1.5 bit-identical ----
# 第二次渲染到不同路徑
rain_path2 = out_dir / "rain_1_dup.wav"
result2 = subprocess.run(
    [sys.executable, str(RENDER), "--dur", "60", "--seed", "1",
     "--out-dir", str(out_dir / "dup_check")],
    capture_output=True, text=True, cwd=str(ROOT)
)
dup_dir = ROOT / "out/stems/dup_check"
rain_path2 = dup_dir / "rain_1.wav"

if result2.returncode == 0 and rain_path2.is_file():
    h1 = hashlib.md5(rain_path.read_bytes()).hexdigest()
    h2 = hashlib.md5(rain_path2.read_bytes()).hexdigest()
    check("V1.5_bit_identical", h1 == h2,
          f"MD5 mismatch: {h1} vs {h2}")
else:
    FAIL.append("V1.5_bit_identical")
    print(f"  FAIL  V1.5_bit_identical  second render failed: {result2.stderr[-200:]}")

# ---- 結果 ----
print(f"\n[M1] PASS={len(PASS)}  FAIL={len(FAIL)}")
if FAIL:
    print("  Failures:", FAIL)
    sys.exit(1)

print("[M1] ALL PASS")
sys.exit(0)
