"""
verify_rain_v3.py — v3 CC0 真實頻譜匹配雨聲驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收清單(閘門凍結,不准修改數字):
  V3.1  63Hz~4kHz 各八度頻帶與新目標差 ≤3dB
        目標(量自 out/loops/rain_loop_90s_real.wav):
          63:+12.0  125:+11.0  250:+8.7  500:+7.9  1k:+9.0  2k:+3.7  4k:-1.1
  V3.2  env-kurtosis 在 [15, 70]
  V3.3  2-8kHz crest 在 [18, 28]dB
  V3.4  無 clip(任何樣本絕對值 ≤ 1.0)
  V3.5  同 seed 兩次渲染 bit-identical
  V3.6  ffmpeg volumedetect max_volume 在 -12 ~ -0.1dB

說明:
  - 八度頻帶分析使用 Welch PSD(nperseg=8192),與 analyze_rain_stats.py 一致
  - 目標數字為 CC0 真實錄音(rain_loop_90s_real.wav)量測值,非手機錄音
  - v3 合成 95s,套 LFO,crossfade 成 90s loop,輸出 out/loops/rain_synth_v3_90s.wav
"""

import os
import shutil
import sys
import pathlib
import subprocess
import re
import numpy as np
from scipy import signal

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from wavio import write_wav, load_wav as _load_wav

SR = 48000
LOOP_DUR_S = 90.0
TAIL_DUR_S = 5.0
XFADE_DUR_S = 5.0
TOTAL_DUR_S = LOOP_DUR_S + TAIL_DUR_S  # 95s
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"

PASS_LIST = []
FAIL_LIST = []

# 目標頻帶(CC0 真實錄音量測值)
TARGETS = {
    "63":  12.0,
    "125": 11.0,
    "250":  8.7,
    "500":  7.9,
    "1k":   9.0,
    "2k":   3.7,
    "4k":  -1.1,
}
SPEC_TOL_DB = 3.0
CHECK_BANDS = ["63", "125", "250", "500", "1k", "2k", "4k"]


def check(name: str, cond: bool, msg: str = "") -> None:
    if cond:
        PASS_LIST.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL_LIST.append(name)
        print(f"  FAIL  {name}  {msg}")


def load_wav(path: pathlib.Path, max_s: float | None = None) -> np.ndarray:
    sr_r, x = _load_wav(path, max_s=max_s, target_sr=SR)
    return x


def octave_band_rel(x: np.ndarray) -> dict:
    """返回各八度頻帶相對 dB(相對於 20Hz 以上全頻帶均值)。"""
    f, pxx = signal.welch(x, SR, nperseg=8192)
    edges = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000]
    labels = ["31", "63", "125", "250", "500", "1k", "2k", "4k"]
    bands_db = []
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        bands_db.append(10.0 * np.log10(pxx[m].mean() + 1e-20))
    total_db = 10.0 * np.log10(pxx[f > 20].mean() + 1e-20)
    rel = np.array(bands_db) - total_db
    return dict(zip(labels, rel))


def temporal_stats(x: np.ndarray):
    """2-8kHz 帶通 → 包絡 → crest / env-kurtosis。"""
    sos = signal.butter(4, [2000, 8000], "bandpass", fs=SR, output="sos")
    b = signal.sosfilt(sos, x)
    env = np.abs(signal.hilbert(b))
    win = int(SR * 0.005)  # 5ms 平滑
    env_s = np.convolve(env, np.ones(win) / win, mode="valid")
    crest = 20.0 * np.log10(np.max(np.abs(b)) / (np.std(b) + 1e-12))
    kurt = float(((env_s - env_s.mean()) ** 4).mean() / (env_s.std() ** 4 + 1e-20))
    return crest, kurt


def make_lfo(n_samples: int, period_s: float, depth: float) -> np.ndarray:
    t = np.arange(n_samples) / SR
    return 1.0 + depth * np.sin(2.0 * np.pi * t / period_s)


def render_v3_loop(seed: int) -> np.ndarray:
    """
    渲染 95s v3 雨聲,套 LFO,crossfade 成 90s loop,回傳 float64 array。
    """
    import json
    from rain_v3 import synthesize_rain_v3

    preset_path = ROOT / "config/preset_realistic_v3.json"
    with open(preset_path) as fp:
        cfg = json.load(fp)
    preset = cfg.get("rain_v3", {})

    rng = np.random.default_rng(seed)
    rain_raw = synthesize_rain_v3(TOTAL_DUR_S, SR, rng, preset)

    n_total = len(rain_raw)
    n_loop = int(LOOP_DUR_S * SR)
    n_xfade = int(XFADE_DUR_S * SR)

    # LFO period = LOOP_DUR_S / 3 = 30s (首尾相位相同)
    LFO_PERIOD_S = LOOP_DUR_S / 3.0
    lfo = make_lfo(n_total, LFO_PERIOD_S, depth=0.08)
    rain_lfo = rain_raw * lfo

    loop_body = rain_lfo[:n_loop].copy()
    tail = rain_lfo[n_loop:n_loop + n_xfade]

    t_xf = np.linspace(0.0, 1.0, n_xfade, dtype=np.float64)
    fade_in = np.sqrt(t_xf)
    fade_out = np.sqrt(1.0 - t_xf)
    loop_body[:n_xfade] = loop_body[:n_xfade] * fade_out + tail * fade_in

    # 限峰 -1dBFS
    peak = np.max(np.abs(loop_body))
    if peak > 0.891:
        loop_body = loop_body * (0.891 / peak)

    return loop_body.astype(np.float64)


def ffmpeg_max_volume(wav_path: pathlib.Path) -> float:
    cmd = [FFMPEG, "-y", "-i", str(wav_path),
           "-af", "volumedetect", "-f", "null", "/dev/null"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    m = re.search(r"max_volume:\s*([-\d.]+)\s*dB", r.stderr)
    if m is None:
        raise RuntimeError(f"ffmpeg volumedetect 未找到 max_volume\nstderr: {r.stderr[-500:]}")
    return float(m.group(1))


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

print("\n[v3] CC0 真實頻譜匹配雨聲驗收")

# ---- 渲染 v3 loop ----
print("  渲染 90s v3 loop (seed=42)...")
SEED = 42
loop_audio = render_v3_loop(SEED)

# 儲存 WAV
out_dir = ROOT / "out/loops"
out_dir.mkdir(parents=True, exist_ok=True)
out_wav = out_dir / "rain_synth_v3_90s.wav"
write_wav(out_wav, SR, loop_audio)
print(f"  -> {out_wav}  ({len(loop_audio)/SR:.1f}s)")

# ---- V3.1 頻譜比對 ----
v3_bands = octave_band_rel(loop_audio)
print(f"  [v3]    octave rel dB: " +
      "  ".join(f"{k}:{v:+.1f}" for k, v in v3_bands.items()))
print(f"  [target]               " +
      "  ".join(f"{k}:{v:+.1f}" for k, v in TARGETS.items()))

all_spec_ok = True
for band in CHECK_BANDS:
    diff = v3_bands[band] - TARGETS[band]
    ok = abs(diff) <= SPEC_TOL_DB
    if not ok:
        all_spec_ok = False
    print(f"    {band}: v3={v3_bands[band]:+.1f}  target={TARGETS[band]:+.1f}  "
          f"diff={diff:+.2f}  {'OK' if ok else 'FAIL'}")
check("V3.1_spectral_63Hz_4kHz", all_spec_ok,
      "至少一個頻帶差>3dB (見上方詳情)")

# ---- V3.2 env-kurtosis & V3.3 crest ----
crest, kurt = temporal_stats(loop_audio)
check("V3.2_env_kurtosis_15_70", 15.0 <= kurt <= 70.0,
      f"kurtosis={kurt:.1f} (need 15~70)")
check("V3.3_crest_18_28", 18.0 <= crest <= 28.0,
      f"crest={crest:.1f}dB (need 18~28dB)")
print(f"  crest={crest:.1f}dB  env-kurtosis={kurt:.1f}")

# ---- V3.4 無 clip ----
peak_val = float(np.max(np.abs(loop_audio)))
check("V3.4_no_clip", peak_val <= 1.0,
      f"peak={peak_val:.6f}")

# ---- V3.5 bit-identical ----
loop_audio_2 = render_v3_loop(SEED)
data_a = np.clip(loop_audio, -1.0, 1.0).astype(np.float32)
data_b = np.clip(loop_audio_2, -1.0, 1.0).astype(np.float32)
check("V3.5_bit_identical", np.array_equal(data_a, data_b),
      "第二次渲染與第一次 float32 不完全相同")

# ---- V3.6 ffmpeg volumedetect ----
try:
    ff_max = ffmpeg_max_volume(out_wav)
    print(f"  ffmpeg max_volume={ff_max:.2f}dB")
    check("V3.6_ffmpeg_max_volume",
          -12.0 <= ff_max <= -0.1,
          f"max_volume={ff_max:.2f}dB (want -12~-0.1dB)")
except RuntimeError as e:
    check("V3.6_ffmpeg_max_volume", False, str(e))

# 同時輸出 MP3
mp3_out = pathlib.Path("/tmp/rain_synth_v3_90s.mp3")
cmd_mp3 = [
    FFMPEG, "-y", "-i", str(out_wav),
    "-codec:a", "libmp3lame", "-b:a", "192k",
    str(mp3_out)
]
r_mp3 = subprocess.run(cmd_mp3, capture_output=True, text=True)
if mp3_out.exists():
    print(f"  -> {mp3_out}  (MP3 192k)")
else:
    print(f"  WARN: MP3 輸出失敗 (stderr: {r_mp3.stderr[-200:]})")

# ---- 結果 ----
print(f"\n[v3] PASS={len(PASS_LIST)}  FAIL={len(FAIL_LIST)}")
if FAIL_LIST:
    print("  Failures:", FAIL_LIST)
    sys.exit(1)

print("[v3] ALL PASS")
sys.exit(0)
