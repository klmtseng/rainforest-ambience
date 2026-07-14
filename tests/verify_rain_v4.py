"""
verify_rain_v4.py — v4 頻帶時間波動匹配雨聲驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收清單(閘門凍結,不准修改數字):

  === 繼承自 v3 的閘門(共用真實 CC0 目標) ===
  V4.1  63Hz~4kHz 各八度頻帶與目標差 ≤3dB
        目標(量自 out/loops/rain_loop_90s_real.wav):
          63:+12.0  125:+11.0  250:+8.7  500:+7.9  1k:+9.0  2k:+3.7  4k:-1.1
  V4.2  env-kurtosis 在 [15, 70]
  V4.3  2-8kHz crest 在 [18, 28]dB
  V4.4  無 clip(任何樣本絕對值 ≤ 1.0)
  V4.5  同 seed 兩次渲染 bit-identical
  V4.6  ffmpeg volumedetect max_volume 在 -12 ~ -0.1dB

  === v4 新增閘門(統計時間波動匹配) ===
  V4.7  六帶包絡 CV 各與目標差 ≤±30%(相對)
        目標(CC0 真實參考 0.5s 平滑包絡):
          125-250Hz:0.254  250-500Hz:0.173  500-1kHz:0.168
          1k-2kHz:0.139   2k-4kHz:0.158    4k-8kHz:0.154
        量測方法:butter 4 階帶通 → np.abs → uniform_filter1d(0.5s) → 去頭尾 1s → std/mean
  V4.8  相鄰帶包絡 Pearson 相關與真實矩陣對應值差 ≤±0.25
        相鄰帶真實目標:
          125-250 vs 250-500: 0.790
          250-500 vs 500-1kHz: 0.821
          500-1kHz vs 1k-2kHz: 0.875
          1k-2kHz vs 2k-4kHz: 0.929
          2k-4kHz vs 4k-8kHz: 0.879
  V4.9  loop 首尾 RMS 差 <1dB
        (量首 0.5s 與尾 0.5s 的 RMS;crossfade 段已使兩端連續)

說明:
  - 使用 config/preset_realistic_v4.json
  - v4 合成 95s,套 LFO(30s 週期,±8% 深度),crossfade 成 90s loop
  - crossfade 使調變器首尾相位吻合(periodic lowpass noise 保證)
"""

import os
import shutil
import json
import pathlib
import re
import subprocess
import sys

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from wavio import write_wav, load_wav as _load_wav

SR = 48000
LOOP_DUR_S = 90.0
TAIL_DUR_S = 5.0
XFADE_DUR_S = 5.0
TOTAL_DUR_S = LOOP_DUR_S + TAIL_DUR_S   # 95s
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"

PASS_LIST = []
FAIL_LIST = []

# V4.1 目標頻帶(同 v3,CC0 真實錄音量測值)
SPECTRAL_TARGETS = {
    "63":  12.0, "125": 11.0, "250": 8.7, "500": 7.9,
    "1k":   9.0, "2k":  3.7, "4k": -1.1,
}
SPEC_TOL_DB = 3.0
CHECK_BANDS = ["63", "125", "250", "500", "1k", "2k", "4k"]

# V4.7 六帶 CV 目標
CV_BANDS = [(125, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 4000), (4000, 8000)]
CV_BAND_NAMES = ["125-250Hz", "250-500Hz", "500-1kHz", "1k-2kHz", "2k-4kHz", "4k-8kHz"]
CV_TARGETS = [0.254, 0.173, 0.168, 0.139, 0.158, 0.154]
CV_TOL_REL = 0.30   # ±30% relative

# V4.8 相鄰帶相關目標
ADJ_CORR_TARGETS = [0.790, 0.821, 0.875, 0.929, 0.879]
ADJ_CORR_TOL = 0.25

# V4.9 首尾 RMS 差容忍
LOOP_SEAM_TOL_DB = 1.0
SEAM_WIN_S = 0.5


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
    win = int(SR * 0.005)
    env_s = np.convolve(env, np.ones(win) / win, mode="valid")
    crest = 20.0 * np.log10(np.max(np.abs(b)) / (np.std(b) + 1e-12))
    kurt = float(((env_s - env_s.mean()) ** 4).mean() / (env_s.std() ** 4 + 1e-20))
    return crest, kurt


def band_envelope_cv(x: np.ndarray, lo: int, hi: int) -> tuple[float, np.ndarray]:
    """帶通 → 整流 → 0.5s uniform 平滑 → 去頭尾 1s → (CV, envelope)。"""
    sos = signal.butter(4, [lo, hi], btype="bandpass", fs=SR, output="sos")
    filt = signal.sosfilt(sos, x)
    env = np.abs(filt)
    win = int(0.5 * SR)
    env_smooth = uniform_filter1d(env, size=win)
    trim = SR   # 去頭尾 1s
    env_trimmed = env_smooth[trim:-trim]
    cv = float(env_trimmed.std() / (env_trimmed.mean() + 1e-12))
    return cv, env_trimmed


def make_lfo(n_samples: int, period_s: float, depth: float) -> np.ndarray:
    t = np.arange(n_samples) / SR
    return 1.0 + depth * np.sin(2.0 * np.pi * t / period_s)


def render_v4_loop(seed: int) -> np.ndarray:
    """
    渲染 95s v4 雨聲,套 LFO,crossfade 成 90s loop,回傳 float64 array。
    """
    from rain_v4 import synthesize_rain_v4

    preset_path = ROOT / "config/preset_realistic_v4.json"
    with open(preset_path) as fp:
        cfg = json.load(fp)
    preset = cfg.get("rain_v4", {})

    rng = np.random.default_rng(seed)
    rain_raw = synthesize_rain_v4(TOTAL_DUR_S, SR, rng, preset)

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

print("\n[v4] 頻帶時間波動匹配雨聲驗收")

SEED = 42
print(f"  渲染 90s v4 loop (seed={SEED})...")
loop_audio = render_v4_loop(SEED)

out_dir = ROOT / "out/loops"
out_dir.mkdir(parents=True, exist_ok=True)
out_wav = out_dir / "rain_synth_v4_90s.wav"
write_wav(out_wav, SR, loop_audio)
print(f"  -> {out_wav}  ({len(loop_audio)/SR:.1f}s)")

# ---- V4.1 頻譜比對 ----
v4_bands = octave_band_rel(loop_audio)
print(f"  [v4]    octave rel dB: " +
      "  ".join(f"{k}:{v:+.1f}" for k, v in v4_bands.items()))
print(f"  [target]               " +
      "  ".join(f"{k}:{v:+.1f}" for k, v in SPECTRAL_TARGETS.items()))

all_spec_ok = True
for band in CHECK_BANDS:
    diff = v4_bands[band] - SPECTRAL_TARGETS[band]
    ok = abs(diff) <= SPEC_TOL_DB
    if not ok:
        all_spec_ok = False
    print(f"    {band}: v4={v4_bands[band]:+.1f}  target={SPECTRAL_TARGETS[band]:+.1f}  "
          f"diff={diff:+.2f}  {'OK' if ok else 'FAIL'}")
check("V4.1_spectral_63Hz_4kHz", all_spec_ok, "至少一個頻帶差>3dB (見上方詳情)")

# ---- V4.2 env-kurtosis & V4.3 crest ----
crest, kurt = temporal_stats(loop_audio)
check("V4.2_env_kurtosis_15_70", 15.0 <= kurt <= 70.0,
      f"kurtosis={kurt:.1f} (need 15~70)")
check("V4.3_crest_18_28", 18.0 <= crest <= 28.0,
      f"crest={crest:.1f}dB (need 18~28dB)")
print(f"  crest={crest:.1f}dB  env-kurtosis={kurt:.1f}")

# ---- V4.4 無 clip ----
peak_val = float(np.max(np.abs(loop_audio)))
check("V4.4_no_clip", peak_val <= 1.0, f"peak={peak_val:.6f}")

# ---- V4.5 bit-identical ----
loop_audio_2 = render_v4_loop(SEED)
data_a = np.clip(loop_audio, -1.0, 1.0).astype(np.float32)
data_b = np.clip(loop_audio_2, -1.0, 1.0).astype(np.float32)
check("V4.5_bit_identical", np.array_equal(data_a, data_b),
      "第二次渲染與第一次 float32 不完全相同")

# ---- V4.6 ffmpeg volumedetect ----
try:
    ff_max = ffmpeg_max_volume(out_wav)
    print(f"  ffmpeg max_volume={ff_max:.2f}dB")
    check("V4.6_ffmpeg_max_volume",
          -12.0 <= ff_max <= -0.1,
          f"max_volume={ff_max:.2f}dB (want -12~-0.1dB)")
except RuntimeError as e:
    check("V4.6_ffmpeg_max_volume", False, str(e))

# 輸出 MP3
mp3_out = pathlib.Path("/tmp/rain_synth_v4_90s.mp3")
cmd_mp3 = [
    FFMPEG, "-y", "-i", str(out_wav),
    "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_out),
]
r_mp3 = subprocess.run(cmd_mp3, capture_output=True, text=True)
if mp3_out.exists():
    print(f"  -> {mp3_out}  (MP3 192k)")
else:
    print(f"  WARN: MP3 輸出失敗 (stderr: {r_mp3.stderr[-200:]})")

# ---- V4.7 六帶包絡 CV ----
print(f"\n  [V4.7] 六帶包絡 CV 比對...")
print(f"  {'Band':14}  {'Target':>7}  {'V4':>7}  {'Ratio':>6}  {'OK?':>5}")
all_cv_ok = True
envs_for_corr = []
for (lo, hi), name, cv_tgt in zip(CV_BANDS, CV_BAND_NAMES, CV_TARGETS):
    cv, env_arr = band_envelope_cv(loop_audio, lo, hi)
    ratio = cv / cv_tgt
    ok = abs(ratio - 1.0) <= CV_TOL_REL
    if not ok:
        all_cv_ok = False
    print(f"  {name:14}  {cv_tgt:7.3f}  {cv:7.3f}  {ratio:6.2f}  {'OK' if ok else 'FAIL'}")
    envs_for_corr.append(env_arr)
check("V4.7_six_band_CV", all_cv_ok,
      "至少一個頻帶 CV 與目標相對差>30% (見上方詳情)")

# ---- V4.8 相鄰帶包絡相關 ----
print(f"\n  [V4.8] 相鄰帶包絡 Pearson 相關...")
print(f"  {'Pair':34}  {'Target':>7}  {'V4':>7}  {'Diff':>7}  {'OK?':>5}")
all_corr_ok = True
min_len = min(len(e) for e in envs_for_corr)
envs_trim = [e[:min_len] for e in envs_for_corr]
for k in range(5):
    c = float(np.corrcoef(envs_trim[k], envs_trim[k + 1])[0, 1])
    tgt = ADJ_CORR_TARGETS[k]
    ok = abs(c - tgt) <= ADJ_CORR_TOL
    if not ok:
        all_corr_ok = False
    pair_name = f"{CV_BAND_NAMES[k]} vs {CV_BAND_NAMES[k+1]}"
    print(f"  {pair_name:34}  {tgt:7.3f}  {c:7.3f}  {c-tgt:+7.3f}  {'OK' if ok else 'FAIL'}")
check("V4.8_adj_band_corr", all_corr_ok,
      "至少一對相鄰帶相關與目標差>0.25 (見上方詳情)")

# ---- V4.9 loop 首尾 RMS 差 ----
seam_n = int(SEAM_WIN_S * SR)
head_seg = loop_audio[:seam_n]
tail_seg = loop_audio[-seam_n:]
head_rms_db = 20.0 * np.log10(np.sqrt(np.mean(head_seg ** 2)) + 1e-12)
tail_rms_db = 20.0 * np.log10(np.sqrt(np.mean(tail_seg ** 2)) + 1e-12)
seam_diff = abs(head_rms_db - tail_rms_db)
print(f"\n  [V4.9] loop 首尾 RMS: head={head_rms_db:.2f}dB  tail={tail_rms_db:.2f}dB  diff={seam_diff:.3f}dB")
check("V4.9_loop_seam_rms", seam_diff < LOOP_SEAM_TOL_DB,
      f"首尾 RMS 差={seam_diff:.3f}dB (need <1dB)")

# ---- 結果 ----
print(f"\n[v4] PASS={len(PASS_LIST)}  FAIL={len(FAIL_LIST)}")
if FAIL_LIST:
    print("  Failures:", FAIL_LIST)
    sys.exit(1)

print("[v4] ALL PASS")
sys.exit(0)
