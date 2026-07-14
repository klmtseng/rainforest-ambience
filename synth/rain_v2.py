"""
rain_v2.py — 雨聲程序合成 v2「統計匹配」版
改進目標:
  1. 頻譜傾斜:FFT 域振幅整形直接施於白噪,無帶通預濾波衝突。
     125Hz 以下平坦,125~2kHz: -8dB/oct,2kHz 以上: -5dB/oct。
     讓 125~6kHz 八度曲線與參考錄音各頻帶差 ≤4dB。
  2. 滴粒層重設計:高密度 Poisson + lognormal 幅度重尾分布,
     把 2-8kHz env-kurtosis 拉到 ≥20、crest ≥18dB。
  3. 獨立程式路徑(不動 rain.py 公開 API);舊 preset 行為不變。

公開 API:
    synthesize_rain_v2(duration_s, sr, rng, preset) -> np.ndarray
    preset dict 欄位見 config/preset_realistic_v2.json
"""

import numpy as np
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# FFT 域頻譜整形
# ---------------------------------------------------------------------------

def _spectral_shape_fft(
    x: np.ndarray,
    sr: int,
    slope1_db_oct: float,
    ref_f1: float,
    slope2_db_oct: float,
    ref_f2: float,
) -> np.ndarray:
    """
    FFT 域振幅整形:
      - DC ~ ref_f1:平坦(amp = 1.0)
      - ref_f1 ~ ref_f2:slope1_db_oct (dB/octave)
      - ref_f2 以上:slope2_db_oct,連續接 ref_f2 處振幅
    使用 rfft/irfft 保持實值輸出。
    """
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    X = np.fft.rfft(x)

    amp = np.ones(len(freqs), dtype=np.float64)

    mask1 = (freqs > ref_f1) & (freqs <= ref_f2)
    if mask1.any():
        amp[mask1] = 10.0 ** (slope1_db_oct * np.log2(freqs[mask1] / ref_f1) / 20.0)

    amp_at_ref_f2 = 10.0 ** (slope1_db_oct * np.log2(ref_f2 / ref_f1) / 20.0)
    mask2 = freqs > ref_f2
    if mask2.any():
        amp[mask2] = amp_at_ref_f2 * 10.0 ** (
            slope2_db_oct * np.log2(freqs[mask2] / ref_f2) / 20.0
        )

    return np.fft.irfft(X * amp, n=n)


# ---------------------------------------------------------------------------
# 床層:FFT 整形白噪
# ---------------------------------------------------------------------------

def _colored_bed(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    slope1_db_oct: float,
    ref_f1: float,
    slope2_db_oct: float,
    ref_f2: float,
) -> np.ndarray:
    """
    生成白噪並直接做 FFT 頻譜整形。
    不做帶通預濾波——避免帶通與 FFT 整形相互抵消。
    """
    n = int(duration_s * sr)
    white = rng.standard_normal(n)
    return _spectral_shape_fft(white, sr, slope1_db_oct, ref_f1, slope2_db_oct, ref_f2)


# ---------------------------------------------------------------------------
# 滴粒模板 v2
# ---------------------------------------------------------------------------

def _make_drop_template_v2(
    sr: int,
    rng: np.random.Generator,
    fc_range_hz: tuple,
    decay_range_ms: tuple,
) -> np.ndarray:
    """
    單顆雨滴:帶通衝擊(1~5kHz)+ 指數衰減(5~30ms)。
    模板長度 = 4× decay_ms;正規化 peak = 1.0。
    """
    fc = rng.uniform(*fc_range_hz)
    bw = rng.uniform(fc * 0.4, fc * 1.2)
    lo = max(fc - bw / 2, 50.0)
    hi = min(fc + bw / 2, sr / 2 - 1.0)

    decay_ms = rng.uniform(*decay_range_ms)
    n = max(int(decay_ms * sr / 1000 * 4), 16)

    noise = rng.standard_normal(n)
    sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
    filtered = sosfilt(sos, noise)

    tau = decay_ms / 1000.0
    t = np.arange(n) / sr
    envelope = np.exp(-t / tau)

    tmpl = (filtered * envelope).astype(np.float64)
    pk = np.max(np.abs(tmpl))
    if pk > 0:
        tmpl = tmpl / pk
    return tmpl


def _build_template_pool_v2(
    n: int, sr: int, rng: np.random.Generator,
    fc_range_hz: tuple, decay_range_ms: tuple,
) -> list:
    return [_make_drop_template_v2(sr, rng, fc_range_hz, decay_range_ms)
            for _ in range(n)]


# ---------------------------------------------------------------------------
# 滴粒層:Poisson + lognormal 幅度重尾
# ---------------------------------------------------------------------------

def _poisson_drops_v2(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    rate_hz: float,
    templates: list,
    amp_mu: float,
    amp_sigma: float,
    intensity: float,
) -> np.ndarray:
    """
    Poisson 到達時刻 + lognormal 幅度。
    大多數滴粒安靜(exp(-1.5)≈0.22),偶有響亮瞬態 → 高 env-kurtosis。
    """
    n_total = int(duration_s * sr)
    out = np.zeros(n_total, dtype=np.float64)

    mean_interval = 1.0 / max(rate_hz, 1.0)
    t = 0.0
    n_tmpl = len(templates)
    while t < duration_s:
        t += rng.exponential(mean_interval)
        if t >= duration_s:
            break
        idx_t = int(t * sr)
        tmpl = templates[rng.integers(0, n_tmpl)]
        amp = rng.lognormal(mean=amp_mu, sigma=amp_sigma) * intensity
        end_t = min(idx_t + len(tmpl), n_total)
        out[idx_t:end_t] += amp * tmpl[:end_t - idx_t]
    return out


# ---------------------------------------------------------------------------
# 主合成函式
# ---------------------------------------------------------------------------

def synthesize_rain_v2(
    duration_s: float,
    sr: int = 48000,
    rng: np.random.Generator | None = None,
    preset: dict | None = None,
) -> np.ndarray:
    """
    v2 雨聲合成:統計匹配版。
    回傳 float64 array shape (N,),N = int(duration_s * sr)。
    同 rng(seed) 多次呼叫 bit-identical。
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if preset is None:
        preset = {}

    intensity = float(preset.get("intensity", 0.6))

    # --- 床層頻譜參數 ---
    slope1 = float(preset.get("spectral_slope1_db_oct", -8.0))
    ref_f1 = float(preset.get("spectral_ref_f1_hz", 125.0))
    slope2 = float(preset.get("spectral_slope2_db_oct", -5.0))
    ref_f2 = float(preset.get("spectral_ref_f2_hz", 2000.0))

    # --- 滴粒參數 ---
    drop_rate_hz = float(preset.get("drop_rate_hz", 600.0))
    drop_template_count = int(preset.get("drop_template_count", 80))
    fc_range = tuple(preset.get("drop_fc_range_hz", [1000.0, 5000.0]))
    decay_range = tuple(preset.get("drop_decay_range_ms", [5.0, 30.0]))
    amp_mu = float(preset.get("drop_amp_lognormal_mu", -1.5))
    amp_sigma = float(preset.get("drop_amp_lognormal_sigma", 1.2))

    # --- 混合比例 ---
    mix = preset.get("bed_drop_mix", [0.55, 0.45])
    bed_w, drop_w = float(mix[0]), float(mix[1])

    # --- 合成 ---
    bed = _colored_bed(duration_s, sr, rng, slope1, ref_f1, slope2, ref_f2)

    templates = _build_template_pool_v2(
        drop_template_count, sr, rng, fc_range, decay_range
    )
    drops = _poisson_drops_v2(
        duration_s, sr, rng, drop_rate_hz, templates,
        amp_mu, amp_sigma, intensity,
    )

    # 正規化 bed 與 drops 到相同 RMS,再按 mix 比例加權
    # 這樣 bed_w/drop_w 才真正控制頻譜貢獻比例
    bed_rms = np.sqrt(np.mean(bed ** 2))
    if bed_rms > 0:
        bed = bed / bed_rms
    drop_rms = np.sqrt(np.mean(drops ** 2))
    if drop_rms > 0:
        drops = drops / drop_rms

    rain = bed * bed_w + drops * drop_w

    # 正規化到 -20 dBFS RMS
    rms = np.sqrt(np.mean(rain ** 2))
    if rms > 0:
        rain = rain * (10.0 ** (-20.0 / 20.0) / rms)

    # 限峰 -1.5 dBFS
    peak_limit = 10.0 ** (-1.5 / 20.0)
    peak = np.max(np.abs(rain))
    if peak > peak_limit:
        rain = rain * (peak_limit / peak)

    return rain.astype(np.float64)
