"""
verify_playback_level.py — 跨工具響度閘門
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收條件(閘門凍結,不准修改數字):
  VPL.1  每個 loop WAV 的 ffmpeg volumedetect max_volume 在 -12dB ~ -0.1dB 之間
          （外部工具解讀下確實有聲且不爆）
  VPL.2  scipy 讀出的 peak dBFS 與 ffmpeg max_volume 相差 ≤ 0.5dB

對象：out/loops/ 的三個 WAV
  - rain_loop_90s.wav
  - rain_loop_90s_real.wav
  - rain_synth_v2_90s.wav

用法:
    python tests/verify_playback_level.py
"""

import os
import shutil
import sys
import subprocess
import pathlib
import re
import numpy as np
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).parent.parent
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
LOOPS_DIR = ROOT / "out/loops"

LOOP_FILES = [
    "rain_loop_90s.wav",
    "rain_loop_90s_real.wav",
    "rain_synth_v2_90s.wav",
]

MAX_VOL_LO = -12.0   # dB（下限：確實有聲）
MAX_VOL_HI = -0.1    # dB（上限：不爆）
DELTA_LIMIT = 0.5    # dB（scipy vs ffmpeg 最大容差）

PASS_LIST = []
FAIL_LIST = []


def check(name: str, cond: bool, msg: str = "") -> None:
    if cond:
        PASS_LIST.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL_LIST.append(name)
        print(f"  FAIL  {name}  {msg}")


def ffmpeg_max_volume(wav_path: pathlib.Path) -> float:
    """
    呼叫 ffmpeg volumedetect，解析並回傳 max_volume（dBFS）。
    """
    cmd = [
        FFMPEG, "-y", "-i", str(wav_path),
        "-af", "volumedetect",
        "-f", "null", "/dev/null"
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    # volumedetect 輸出在 stderr
    m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", r.stderr)
    if m is None:
        raise RuntimeError(
            f"ffmpeg volumedetect 未找到 max_volume\nstderr: {r.stderr[-500:]}"
        )
    return float(m.group(1))


def scipy_peak_dbfs(wav_path: pathlib.Path) -> float:
    """
    用 scipy.io.wavfile 讀取，計算 peak dBFS（支援 float32 和 int32/int16）。
    """
    sr, x = wavfile.read(str(wav_path))
    if x.dtype in (np.float32, np.float64):
        data = x.astype(np.float64)
    elif x.dtype == np.int32:
        peak_raw = np.max(np.abs(x))
        if peak_raw <= (2 ** 23):
            data = x.astype(np.float64) / (2 ** 23)
        else:
            data = x.astype(np.float64) / 2147483648.0
    elif x.dtype == np.int16:
        data = x.astype(np.float64) / 32768.0
    else:
        data = x.astype(np.float64)

    if data.ndim > 1:
        data = data.mean(axis=1)

    peak = float(np.max(np.abs(data)))
    return 20.0 * np.log10(peak + 1e-12)


print("\n[VPL] 跨工具響度閘門驗收")
print(f"  ffmpeg: {FFMPEG}")
print(f"  loops:  {LOOPS_DIR}")
print()

for fname in LOOP_FILES:
    wav_path = LOOPS_DIR / fname
    print(f"  [{fname}]")

    if not wav_path.exists():
        check(f"VPL.1_{fname}_exists", False, f"檔案不存在: {wav_path}")
        check(f"VPL.2_{fname}_delta", False, "跳過（檔案不存在）")
        continue

    try:
        ff_max = ffmpeg_max_volume(wav_path)
    except RuntimeError as e:
        check(f"VPL.1_{fname}_range", False, str(e))
        check(f"VPL.2_{fname}_delta", False, "ffmpeg 失敗")
        continue

    scipy_peak = scipy_peak_dbfs(wav_path)

    print(f"    ffmpeg max_volume = {ff_max:.2f} dB")
    print(f"    scipy  peak dBFS  = {scipy_peak:.2f} dBFS")

    ok_range = MAX_VOL_LO <= ff_max <= MAX_VOL_HI
    check(
        f"VPL.1_{fname}_range",
        ok_range,
        f"max_volume={ff_max:.2f}dB (want {MAX_VOL_LO}~{MAX_VOL_HI}dB)"
    )

    delta = abs(ff_max - scipy_peak)
    ok_delta = delta <= DELTA_LIMIT
    check(
        f"VPL.2_{fname}_delta",
        ok_delta,
        f"|ffmpeg-scipy|={delta:.3f}dB (want ≤{DELTA_LIMIT}dB)"
    )

print()
print(f"[VPL] PASS={len(PASS_LIST)}  FAIL={len(FAIL_LIST)}")
if FAIL_LIST:
    print("  Failures:", FAIL_LIST)
    sys.exit(1)

print("[VPL] ALL PASS")
sys.exit(0)
