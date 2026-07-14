"""
verify_m4.py — M4 90s 無縫 loop 驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收清單(閘門凍結,不准修改數字):
  V4.1  首尾拼接 50ms 窗 RMS 差 < 1dB
  V4.2  拼接點一階差分最大值 < 全檔 99.9 百分位
  V4.3  ffmpeg concat 循環播 3 遍無 clip(峰值 < 0dBFS)
"""

import os
import shutil
import sys
import subprocess
import pathlib
import tempfile
import numpy as np
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).parent.parent
LOOPER = ROOT / "mix/looper.py"
LOOP_PATH = ROOT / "out/loops/rain_loop_90s.wav"
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
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


print("\n[M4] 90s loop 驗收")

# 產生 loop
result = subprocess.run(
    [sys.executable, str(LOOPER), "--seed", "42"],
    capture_output=True, text=True, cwd=str(ROOT)
)
if result.returncode != 0:
    print(f"  looper.py FAILED:\n{result.stderr[-500:]}")
    sys.exit(1)
print(f"  looper output:\n{result.stdout.strip()}")

if not LOOP_PATH.exists():
    print(f"  FAIL: output file not found: {LOOP_PATH}")
    sys.exit(1)

# 讀 WAV
sr_r, data_raw = wavfile.read(str(LOOP_PATH))
if data_raw.dtype in (np.float32, np.float64):
    data = data_raw.astype(np.float64)
elif data_raw.dtype == np.int32:
    data = data_raw.astype(np.float64) / (2 ** 23)
else:
    data = data_raw.astype(np.float64) / 32768.0
n = len(data)
dur = n / sr_r
print(f"  file: {LOOP_PATH.name}  dur={dur:.3f}s  samples={n}")

# V4.1 首尾拼接 50ms 窗 RMS 差 < 1dB
win_n = int(0.050 * sr_r)   # 50ms
tail_win = data[-win_n:]      # loop 最後 50ms（接縫前段）
head_win = data[:win_n]       # loop 最前 50ms（接縫後段）

rms_tail = np.sqrt(np.mean(tail_win ** 2) + 1e-12)
rms_head = np.sqrt(np.mean(head_win ** 2) + 1e-12)
rms_diff_db = abs(20 * np.log10(rms_tail / rms_head))
check("V4.1_seam_rms", rms_diff_db < 1.0,
      f"首尾 RMS 差={rms_diff_db:.3f}dB (want <1dB)  tail={20*np.log10(rms_tail):.2f}dBFS  head={20*np.log10(rms_head):.2f}dBFS")

# V4.2 拼接點一階差分最大值 < 全檔 99.9 百分位
# 拼接點 = 首樣本到尾樣本的跳躍
# 取最後一樣本和第一樣本的差，與全檔一階差分的 99.9 百分位比較
diff_all = np.abs(np.diff(data))
p999 = np.percentile(diff_all, 99.9)
seam_jump = abs(float(data[0]) - float(data[-1]))
check("V4.2_seam_jump", seam_jump < p999,
      f"拼接跳躍={seam_jump:.6f}  全檔99.9%={p999:.6f}")

# V4.3 ffmpeg concat 循環播 3 遍無 clip
with tempfile.TemporaryDirectory() as tmpdir:
    concat_txt = pathlib.Path(tmpdir) / "concat.txt"
    out_concat = pathlib.Path(tmpdir) / "concat3.wav"
    with open(concat_txt, "w") as f:
        for _ in range(3):
            f.write(f"file '{LOOP_PATH}'\n")

    cmd = [
        FFMPEG, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_txt),
        "-c", "copy",
        str(out_concat)
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        check("V4.3_no_clip", False, f"ffmpeg concat failed: {r.stderr[-300:]}")
    else:
        sr_c, data_c_raw = wavfile.read(str(out_concat))
        if data_c_raw.dtype in (np.float32, np.float64):
            data_c = data_c_raw.astype(np.float64)
        elif data_c_raw.dtype == np.int32:
            data_c = data_c_raw.astype(np.float64) / (2 ** 23)
        else:
            data_c = data_c_raw.astype(np.float64) / 32768.0
        peak_concat = np.max(np.abs(data_c))
        peak_db_concat = 20 * np.log10(peak_concat + 1e-12)
        check("V4.3_no_clip", peak_concat < 1.0,
              f"3× concat peak={peak_db_concat:.2f}dBFS (want <0dBFS)")

# 結果
print(f"\n[M4] PASS={len(PASS)}  FAIL={len(FAIL)}")
if FAIL:
    print("  Failures:", FAIL)
    sys.exit(1)

print("[M4] ALL PASS")
sys.exit(0)
