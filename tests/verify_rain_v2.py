"""
verify_rain_v2.py — v2 統計匹配雨聲驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收清單(閘門凍結,不准修改數字):
  V2.1  125Hz~6kHz 八度頻帶相對曲線與參考錄音各頻帶差 ≤4dB
  V2.2  2-8kHz env-kurtosis ≥ 20
  V2.3  2-8kHz crest ≥ 18dB
  V2.4  無 clip(任何樣本絕對值 ≤ 1.0)
  V2.5  同 seed 兩次渲染 bit-identical

說明:
  - 八度頻帶分析使用 Welch PSD,相同於 analyze_rain_stats.py
  - 參考錄音:assets/reference/user_rain_reference.wav(取前 60s)
  - 比對頻帶:125Hz、250Hz、500Hz、1kHz、2kHz、4kHz
    (4-8kHz 八度帶涵蓋 4~8kHz,代表 6kHz 附近;8kHz 以上跳過,為 HE-AAC 編碼截斷假象)
  - v2 合成用 synthesize_rain_v2(95s) + looper crossfade 邏輯 → 90s loop
"""

import sys
import pathlib
import numpy as np
from scipy.io import wavfile
from scipy import signal

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from wavio import write_wav, load_wav as _load_wav
SR = 48000
LOOP_DUR_S = 90.0
TAIL_DUR_S = 5.0
XFADE_DUR_S = 5.0
TOTAL_DUR_S = LOOP_DUR_S + TAIL_DUR_S  # 95s

PASS_LIST = []
FAIL_LIST = []


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
    """
    返回各八度頻帶相對 dB(相對於 20Hz 以上的全頻帶均值)。
    keys: '125', '250', '500', '1k', '2k', '4k'
    """
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


def render_v2_loop(seed: int) -> np.ndarray:
    """
    渲染 95s v2 雨聲,套 LFO,crossfade 成 90s loop,回傳 float64 array。
    邏輯與 mix/looper.py 一致,但呼叫 synthesize_rain_v2。
    """
    import json
    sys.path.insert(0, str(ROOT / "synth"))
    from rain_v2 import synthesize_rain_v2

    preset_path = ROOT / "config/preset_realistic_v2.json"
    with open(preset_path) as fp:
        cfg = json.load(fp)
    preset = cfg.get("rain_v2", {})

    rng = np.random.default_rng(seed)
    rain_raw = synthesize_rain_v2(TOTAL_DUR_S, SR, rng, preset)

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


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

print("\n[v2] 雨聲統計匹配驗收")

# ---- 參考錄音 ----
ref_path = ROOT / "assets/reference/user_rain_reference.wav"
if not ref_path.is_file():
    print(f"  ERROR: 找不到參考錄音 {ref_path}")
    sys.exit(1)

ref_audio = load_wav(ref_path, max_s=60.0)
ref_bands = octave_band_rel(ref_audio)
print(f"  [ref] octave rel dB: " +
      "  ".join(f"{k}:{v:+.1f}" for k, v in ref_bands.items()
                if k in ["125", "250", "500", "1k", "2k", "4k"]))

# ---- 渲染 v2 loop ----
print("  渲染 90s v2 loop (seed=42)...")
SEED = 42
loop_audio = render_v2_loop(SEED)

# 儲存 WAV
out_dir = ROOT / "out/loops"
out_dir.mkdir(parents=True, exist_ok=True)
out_wav = out_dir / "rain_synth_v2_90s.wav"
write_wav(out_wav, SR, loop_audio)
print(f"  -> {out_wav}  ({len(loop_audio)/SR:.1f}s)")

# ---- V2.1 頻譜比對 ----
v2_bands = octave_band_rel(loop_audio)
print(f"  [v2]  octave rel dB: " +
      "  ".join(f"{k}:{v:+.1f}" for k, v in v2_bands.items()
                if k in ["125", "250", "500", "1k", "2k", "4k"]))

CHECK_BANDS = ["125", "250", "500", "1k", "2k", "4k"]
all_spec_ok = True
for band in CHECK_BANDS:
    diff = v2_bands[band] - ref_bands[band]
    ok = abs(diff) <= 4.0
    if not ok:
        all_spec_ok = False
        print(f"    {band}: v2={v2_bands[band]:+.1f}  ref={ref_bands[band]:+.1f}  diff={diff:+.2f}  FAIL")
check("V2.1_spectral_match_125Hz_6kHz", all_spec_ok,
      "至少一個頻帶差>4dB (見上方詳情)")

# ---- V2.2 env-kurtosis & V2.3 crest ----
crest, kurt = temporal_stats(loop_audio)
check("V2.2_env_kurtosis", kurt >= 20.0,
      f"kurtosis={kurt:.1f} (need ≥20)")
check("V2.3_crest", crest >= 18.0,
      f"crest={crest:.1f}dB (need ≥18dB)")

print(f"  crest={crest:.1f}dB  env-kurtosis={kurt:.1f}")

# ---- V2.4 無 clip ----
check("V2.4_no_clip", np.max(np.abs(loop_audio)) <= 1.0,
      f"peak={np.max(np.abs(loop_audio)):.6f}")

# ---- V2.5 bit-identical ----
loop_audio_2 = render_v2_loop(SEED)
data_a = np.clip(loop_audio, -1.0, 1.0).astype(np.float32)
data_b = np.clip(loop_audio_2, -1.0, 1.0).astype(np.float32)
check("V2.5_bit_identical", np.array_equal(data_a, data_b),
      "第二次渲染與第一次 float32 不完全相同")

# ---- 結果 ----
print(f"\n[v2] PASS={len(PASS_LIST)}  FAIL={len(FAIL_LIST)}")
if FAIL_LIST:
    print("  Failures:", FAIL_LIST)
    sys.exit(1)

print("[v2] ALL PASS")
sys.exit(0)
