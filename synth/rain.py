"""
rain.py — 雨聲程序合成
架構:
  - 帶通噪音 bed:4-5 個並聯帶通濾波器(400/1k/2.5k/6k/10kHz)
  - 滴粒層:Poisson 排程 + 預生成 drop_template_count 個模板,稀疏疊加
輸出:float64 array,shape (N,)

公開 API:
    synthesize_rain(duration_s, sr, rng, intensity, band_freqs_hz, band_widths_hz,
                    drop_rate_hz, drop_template_count) -> np.ndarray
"""

import numpy as np
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# 帶通床層
# ---------------------------------------------------------------------------

def _bandpass_noise(duration_s: float, sr: int, rng: np.random.Generator,
                    freqs_hz: list[float], widths_hz: list[float],
                    weights: list[float]) -> np.ndarray:
    """白噪通過多個帶通濾波器後加權疊合。"""
    n = int(duration_s * sr)
    white = rng.standard_normal(n)
    out = np.zeros(n, dtype=np.float64)
    for f, w, wt in zip(freqs_hz, widths_hz, weights):
        lo = max(f - w / 2, 20.0)
        hi = min(f + w / 2, sr / 2 - 1.0)
        sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
        out += wt * sosfilt(sos, white)
    return out


# ---------------------------------------------------------------------------
# 滴粒模板
# ---------------------------------------------------------------------------

def _make_drop_template(sr: int, rng: np.random.Generator,
                        dur_ms_range=(4, 18)) -> np.ndarray:
    """
    單顆雨滴聲模板:短帶通脈衝 + 指數衰減包絡。
    持續 4-18ms,隨機中心頻率 600-6000Hz。
    """
    dur_ms = rng.uniform(*dur_ms_range)
    n = int(dur_ms * sr / 1000)
    fc = rng.uniform(600, 6000)
    bw = rng.uniform(fc * 0.3, fc * 0.8)
    lo = max(fc - bw / 2, 20.0)
    hi = min(fc + bw / 2, sr / 2 - 1.0)
    noise = rng.standard_normal(n)
    sos = butter(3, [lo, hi], btype='bandpass', fs=sr, output='sos')
    filtered = sosfilt(sos, noise)
    tau = rng.uniform(0.002, 0.008)   # 衰減時間常數 (秒)
    t = np.arange(n) / sr
    envelope = np.exp(-t / tau)
    return (filtered * envelope).astype(np.float64)


def _build_template_pool(n: int, sr: int, rng: np.random.Generator) -> list[np.ndarray]:
    return [_make_drop_template(sr, rng) for _ in range(n)]


# ---------------------------------------------------------------------------
# 滴粒層疊加
# ---------------------------------------------------------------------------

def _poisson_drops(duration_s: float, sr: int, rng: np.random.Generator,
                   rate_hz: float, templates: list[np.ndarray],
                   intensity: float) -> np.ndarray:
    """Poisson 排程,從模板池隨機選模板疊加到輸出緩衝。"""
    n_total = int(duration_s * sr)
    out = np.zeros(n_total, dtype=np.float64)

    # 生成 Poisson 到達時刻(秒)
    mean_interval = 1.0 / max(rate_hz, 1.0)
    t = 0.0
    n_tmpl = len(templates)
    while t < duration_s:
        interval = rng.exponential(mean_interval)
        t += interval
        if t >= duration_s:
            break
        idx_t = int(t * sr)
        tmpl = templates[rng.integers(0, n_tmpl)]
        amp = rng.uniform(0.3, 1.0) * intensity
        end_t = min(idx_t + len(tmpl), n_total)
        out[idx_t:end_t] += amp * tmpl[:end_t - idx_t]
    return out


# ---------------------------------------------------------------------------
# 主合成函式
# ---------------------------------------------------------------------------

def synthesize_rain(
    duration_s: float,
    sr: int = 48000,
    rng: np.random.Generator | None = None,
    intensity: float = 0.6,
    band_freqs_hz: list[float] | None = None,
    band_widths_hz: list[float] | None = None,
    drop_rate_hz: float = 80.0,
    drop_template_count: int = 40,
) -> np.ndarray:
    """
    合成雨聲,回傳 float64 array shape (N,),N = int(duration_s * sr)。
    同 rng(seed) 多次呼叫保證 bit-identical 輸出。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    if band_freqs_hz is None:
        band_freqs_hz = [400.0, 1000.0, 2500.0, 6000.0, 10000.0]
    if band_widths_hz is None:
        band_widths_hz = [200.0, 500.0, 800.0, 2000.0, 3000.0]

    # 帶通床:高頻段 weight 隨 intensity 增加(大雨高頻成分多)
    base_weights = [0.4, 0.7, 1.0, 0.8, 0.5]
    hi_boost = 0.3 * intensity
    weights = [w * (1.0 + hi_boost * (i / (len(base_weights) - 1)))
               for i, w in enumerate(base_weights)]

    bed = _bandpass_noise(duration_s, sr, rng,
                          band_freqs_hz, band_widths_hz, weights)

    # 滴粒層
    templates = _build_template_pool(drop_template_count, sr, rng)
    drops = _poisson_drops(duration_s, sr, rng, drop_rate_hz, templates, intensity)

    # 合成
    rain = bed * 0.7 + drops * 0.3

    # 正規化讓目標 RMS 落在 -20dBFS 附近
    rms = np.sqrt(np.mean(rain ** 2))
    if rms > 0:
        target_rms = 10 ** (-20 / 20)   # -20 dBFS
        rain = rain * (target_rms / rms)

    # 限峰 -1.5dBFS
    peak_limit = 10 ** (-1.5 / 20)
    peak = np.max(np.abs(rain))
    if peak > peak_limit:
        rain = rain * (peak_limit / peak)

    return rain.astype(np.float64)
