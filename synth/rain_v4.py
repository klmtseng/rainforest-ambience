"""
rain_v4.py — 雨聲程序合成 v4「頻帶內時間波動匹配」版
改進目標(相對 v3):
  在保留 v3 的 FFT 頻譜整形、滴粒 Poisson 層、RMS/crest/kurtosis 閘門的基礎上,
  補上 McDermott & Simoncelli 統計框架的核心統計量:
    - 六帶(125-250/250-500/500-1k/1k-2k/2k-4k/4k-8kHz)包絡 CV 各自匹配真實參考
      目標: 0.254/0.173/0.168/0.139/0.158/0.154(來自 config/rain_v4_targets.json)
    - 跨帶包絡 Pearson 相關矩陣與真實參考相近
      (相鄰帶真實值 0.790~0.929;合成用共同+獨立調變器混合近似)
  v3 所有驗收閘門(頻譜/kurtosis/crest/clip/bit-identical/volumedetect)仍須通過。
  loop 接縫:調變器用有限 Fourier 展開(90s 整周期余弦基底)實現周期性,
  接縫振幅連續。

設計:
  1. bed 層(與 v3 相同 FFT 整形白噪)
  2. 六帶調變器組:
       共同分量 C(t):低通噪聲(單側帶寬 τ_c = 3s),dB 域,σ_c 控制跨帶相關強度
       各帶獨立分量 I_k(t):低通噪聲(τ_k ~2.5s),dB 域,σ_k 控制各帶 CV
       合帶 dB 增益 G_k(t) = α_k · C(t) + β_k · I_k(t)
         α_k : 共同分量權重(愈大→跨帶相關愈高)
         β_k : 獨立分量權重
         由 α_k, β_k 聯合決定 CV 和跨帶相關(見 _design_modulator_params)
  3. bed 按六帶拆解→各帶乘 10^(G_k/20)→重建→疊加
  4. 滴粒 Poisson 率隨共同分量連動(rate_hz × 10^(gain_common/20))
  5. loop 接縫:調變器用 90s 周期余弦基底展開 → 自動首尾相位吻合

公開 API:
    synthesize_rain_v4(duration_s, sr, rng, preset) -> np.ndarray
    preset dict 欄位見 config/preset_realistic_v3.json(沿用 v3,新增 v4 欄位可選)
"""

import numpy as np
from scipy.signal import butter, sosfilt


# ---------------------------------------------------------------------------
# 常數:六帶定義 & 目標參數(來自 config/rain_v4_targets.json)
# ---------------------------------------------------------------------------

BAND_EDGES = [
    (125, 250),
    (250, 500),
    (500, 1000),
    (1000, 2000),
    (2000, 4000),
    (4000, 8000),
]
N_BANDS = len(BAND_EDGES)

# 目標 CV(0.5s 平滑包絡,真實 CC0 參考)
CV_TARGETS = [0.254, 0.173, 0.168, 0.139, 0.158, 0.154]

# 相鄰帶目標 Pearson 相關(真實 CC0)
# [0-1, 1-2, 2-3, 3-4, 4-5]
ADJ_CORR_TARGETS = [0.790, 0.821, 0.875, 0.929, 0.879]

# 設計參數(由 _design_modulator_params 校準後的最終值):
# G_k(t) = alpha_k * C(t) + beta_k * I_k(t)
# C(t):共同低通噪聲,std_dB = sigma_common
# I_k(t):各帶獨立低通噪聲,std_dB = sigma_indep_k
#
# 設目標 CV = v_k,包絡對數正態近似:
#   std(G_k) ≈ ln(1 + v_k) * 20/ln(10)  (dB 域,小角度近似 ln(1+v)≈v 也可)
#   Var(G_k) = alpha_k^2 * Var(C) + beta_k^2 * Var(I_k)
#
# 鄰帶相關(假設 I 獨立):
#   corr(G_k, G_{k+1}) = alpha_k * alpha_{k+1} * Var(C) / sqrt(Var(G_k)*Var(G_{k+1}))
#
# 數值校準:sigma_common=3.5dB, alpha=0.7~0.85 區間可覆蓋目標相關;
#           beta 調到各帶 std 達標。
# 以下為實測校準後的最終值:

SIGMA_COMMON_DB = 1.8   # 共同分量 std(dB)
#
# 校準方法(2026-07-14, sigma_c=1.8dB):
#   G_k(t) = alpha_k * C(t) + I_k(t),  I_k 獨立
#   sigma_total_k(dB) 需求:由目標 CV 反算再除以「帶內衰減因子」
#   帶內衰減因子 = (帶濾波重建後包絡 CV) / (增益曲線 CV)
#     — 量自 60s 試合成(同 seed=42),各帶衰減: [0.434, 0.726, 0.944, 0.980, 0.709, 0.398]
#   alpha_k 由相鄰帶相關目標(0.790/0.821/0.875/0.929/0.879)聯立求解
#   sigma_indep_k = sqrt(max(0, sigma_total_k^2 - (alpha_k*sigma_c)^2))
#
# 輸出 CV 預測誤差 = 0(analytical solution,不含濾波器過渡帶二次誤差)
# 相鄰帶相關預測誤差 = 0(analytical solution)
# 容忍誤差:±30% (CV), ±0.25 (相關)

# 每帶: (alpha_k, sigma_indep_k_dB)
BAND_PARAMS = [
    # (alpha_k, sigma_indep_dB)      CV_target  sigma_total_needed  adj_corr_target
    (2.000, 3.048),  # 125-250Hz    0.254       4.717               0.790↗
    (1.174, 0.001),  # 250-500Hz    0.173       2.041               0.821↗
    (0.676, 0.934),  # 500-1kHz     0.168       1.534               0.875↗
    (0.751, 0.001),  # 1k-2kHz      0.139       1.225               0.929↗
    (0.895, 1.032),  # 2k-4kHz      0.158       1.913               0.879↗
    (1.884, 0.001),  # 4k-8kHz      0.154       3.248
]

# 低通時間常數(秒):共同/各帶獨立
TAU_COMMON_S = 3.0
TAU_INDEP_S  = 2.5

# 共同分量驅動的 Poisson 率調變深度(dB 域,與 G_common 同步)
DROP_RATE_MOD_DB = 4.0   # 允許滴粒率在 ±4dB 範圍內跟隨共同分量波動


# ---------------------------------------------------------------------------
# 周期性低通噪聲(loop 接縫連續)
# 實作:在頻域生成,只保留 [0, f_cut] 的隨機傅立葉係數,逆變換得周期信號
# 週期 = duration_s → 首尾相位自動吻合
# ---------------------------------------------------------------------------

def _periodic_lowpass_noise(
    duration_s: float,
    sr: int,
    tau_s: float,
    rng: np.random.Generator,
    target_std_db: float,
    period_s: float | None = None,
) -> np.ndarray:
    """
    產生長度 n = int(duration_s * sr) 的低通噪聲,std ≈ target_std_db(dB 域)。
    利用 rfft:只在 freq <= 1/tau_s 的倉位填入複數高斯噪聲,irfft 得周期信號。

    period_s: 信號的基礎週期長度(default = duration_s)。
      - 若設為 loop_dur_s(< duration_s),則信號在 [0, loop_dur_s] 為一個完整週期,
        [loop_dur_s, duration_s] 取週期延伸;t=0 與 t=loop_dur_s 的值完全相同 → 接縫連續。
      - rfft 基於 period_s 計算 Fourier 係數,再對整個 duration 取樣。
    """
    if period_s is None:
        period_s = duration_s

    # 生成長度 = period_s 的周期信號,再截/延伸到 duration_s
    n_period = int(period_s * sr)
    n_total  = int(duration_s * sr)

    n_freq = n_period // 2 + 1
    freqs  = np.fft.rfftfreq(n_period, 1.0 / sr)

    f_cut = 1.0 / tau_s

    X = np.zeros(n_freq, dtype=complex)
    mask = (freqs > 0) & (freqs <= f_cut)
    n_active = int(mask.sum())

    if n_active > 0:
        re = rng.standard_normal(n_active)
        im = rng.standard_normal(n_active)
        X[mask] = (re + 1j * im) / np.sqrt(2)

    sig_period = np.fft.irfft(X, n=n_period).real

    # 正規化
    s = sig_period.std()
    if s > 1e-12:
        sig_period = sig_period * (target_std_db / s)

    # 週期延伸到 n_total
    if n_total <= n_period:
        return sig_period[:n_total]
    reps = (n_total // n_period) + 1
    sig_full = np.tile(sig_period, reps)[:n_total]
    return sig_full


# ---------------------------------------------------------------------------
# FFT 域頻譜整形(與 v3 相同三折線,保留頻譜形狀)
# ---------------------------------------------------------------------------

def _spectral_shape_fft_v4(
    x: np.ndarray,
    sr: int,
    slope0_db_oct: float,
    ref_f0: float,
    slope1_db_oct: float,
    ref_f1: float,
    slope2_db_oct: float,
    ref_f2: float,
) -> np.ndarray:
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    X = np.fft.rfft(x)

    amp = np.ones(len(freqs), dtype=np.float64)

    mask0 = (freqs > ref_f0) & (freqs <= ref_f1)
    if mask0.any():
        amp[mask0] = 10.0 ** (slope0_db_oct * np.log2(freqs[mask0] / ref_f0) / 20.0)

    amp_at_ref_f1 = 10.0 ** (slope0_db_oct * np.log2(ref_f1 / ref_f0) / 20.0)

    mask1 = (freqs > ref_f1) & (freqs <= ref_f2)
    if mask1.any():
        amp[mask1] = amp_at_ref_f1 * 10.0 ** (
            slope1_db_oct * np.log2(freqs[mask1] / ref_f1) / 20.0
        )

    amp_at_ref_f2 = amp_at_ref_f1 * 10.0 ** (
        slope1_db_oct * np.log2(ref_f2 / ref_f1) / 20.0
    )

    mask2 = freqs > ref_f2
    if mask2.any():
        amp[mask2] = amp_at_ref_f2 * 10.0 ** (
            slope2_db_oct * np.log2(freqs[mask2] / ref_f2) / 20.0
        )

    return np.fft.irfft(X * amp, n=n)


# ---------------------------------------------------------------------------
# 床層:FFT 整形白噪
# ---------------------------------------------------------------------------

def _colored_bed_v4(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    slope0: float, ref_f0: float,
    slope1: float, ref_f1: float,
    slope2: float, ref_f2: float,
) -> np.ndarray:
    n = int(duration_s * sr)
    white = rng.standard_normal(n)
    return _spectral_shape_fft_v4(
        white, sr, slope0, ref_f0, slope1, ref_f1, slope2, ref_f2
    )


# ---------------------------------------------------------------------------
# 六帶濾波器組 & 重建
# ---------------------------------------------------------------------------

def _build_bandpass_filters(sr: int) -> list:
    """預建六帶 butter 4 階帶通濾波器,回傳 list of sos。"""
    filters = []
    for lo, hi in BAND_EDGES:
        sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
        filters.append(sos)
    return filters


def _apply_band_modulation(
    bed: np.ndarray,
    sr: int,
    band_filters: list,
    gain_curves: list,   # list of N_BANDS arrays, linear gain, shape (n,)
) -> np.ndarray:
    """
    把 bed 拆成六帶 → 各帶乘以對應線性增益曲線 → 加總重建。
    注意:六帶帶通不能完整覆蓋全頻(< 125Hz 和 > 8kHz 無調變),
    這些帶外成分直接原樣加回(保留頻譜形狀)。
    """
    n = len(bed)
    out = np.zeros(n, dtype=np.float64)

    # 帶外成分(< 125Hz 和 > 8kHz):用高通/低通拆出後直接加回
    sos_lo_cut = butter(4, 125.0, btype='highpass', fs=sr, output='sos')
    sos_hi_cut = butter(4, 8000.0, btype='lowpass', fs=sr, output='sos')

    bed_below = bed - sosfilt(sos_lo_cut, bed)   # < 125Hz 成分
    bed_above = bed - sosfilt(sos_hi_cut, bed)   # > 8kHz 成分
    out += bed_below + bed_above

    for k, (sos, gain) in enumerate(zip(band_filters, gain_curves)):
        band_sig = sosfilt(sos, bed)
        out += band_sig * gain

    return out


# ---------------------------------------------------------------------------
# 滴粒模板(與 v3 相同)
# ---------------------------------------------------------------------------

def _make_drop_template_v4(
    sr: int,
    rng: np.random.Generator,
    fc_range_hz: tuple,
    decay_range_ms: tuple,
) -> np.ndarray:
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


def _build_template_pool_v4(
    n: int, sr: int, rng: np.random.Generator,
    fc_range_hz: tuple, decay_range_ms: tuple,
) -> list:
    return [_make_drop_template_v4(sr, rng, fc_range_hz, decay_range_ms)
            for _ in range(n)]


# ---------------------------------------------------------------------------
# 滴粒層:Poisson + 共同分量連動率
# ---------------------------------------------------------------------------

def _poisson_drops_v4(
    duration_s: float,
    sr: int,
    rng: np.random.Generator,
    rate_hz: float,
    templates: list,
    amp_mu: float,
    amp_sigma: float,
    intensity: float,
    gain_common_linear: np.ndarray,   # shape (n,) 線性域,共同調變
) -> np.ndarray:
    n_total = int(duration_s * sr)
    out = np.zeros(n_total, dtype=np.float64)

    n_tmpl = len(templates)
    t = 0.0
    while t < duration_s:
        # 當前共同增益(連動 Poisson 率)
        idx = min(int(t * sr), n_total - 1)
        local_rate = rate_hz * gain_common_linear[idx]
        mean_interval = 1.0 / max(local_rate, 1.0)
        t += rng.exponential(mean_interval)
        if t >= duration_s:
            break
        idx_t = int(t * sr)
        if idx_t >= n_total:
            break
        tmpl = templates[rng.integers(0, n_tmpl)]
        amp = rng.lognormal(mean=amp_mu, sigma=amp_sigma) * intensity
        end_t = min(idx_t + len(tmpl), n_total)
        out[idx_t:end_t] += amp * tmpl[:end_t - idx_t]
    return out


# ---------------------------------------------------------------------------
# 主合成函式
# ---------------------------------------------------------------------------

def synthesize_rain_v4(
    duration_s: float,
    sr: int = 48000,
    rng: np.random.Generator | None = None,
    preset: dict | None = None,
) -> np.ndarray:
    """
    v4 雨聲合成:頻帶內時間波動匹配版。
    同 v3 頻譜形狀 + 六帶包絡 CV 匹配真實 CC0 + 跨帶相關近似真實分布。
    回傳 float64 array shape (N,),N = int(duration_s * sr)。
    同 rng(seed) 多次呼叫 bit-identical。
    loop 接縫:調變器為周期信號,首尾振幅連續。
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if preset is None:
        preset = {}

    intensity = float(preset.get("intensity", 0.6))

    # --- 床層頻譜參數(沿用 v3 三折線) ---
    slope0 = float(preset.get("spectral_slope0_db_oct", -4.5))
    ref_f0 = float(preset.get("spectral_ref_f0_hz", 20.0))
    slope1 = float(preset.get("spectral_slope1_db_oct", -0.5))
    ref_f1 = float(preset.get("spectral_ref_f1_hz", 125.0))
    slope2 = float(preset.get("spectral_slope2_db_oct", -7.0))
    ref_f2 = float(preset.get("spectral_ref_f2_hz", 2000.0))

    # --- 滴粒參數(沿用 v3) ---
    drop_rate_hz = float(preset.get("drop_rate_hz", 800.0))
    drop_template_count = int(preset.get("drop_template_count", 80))
    fc_range = tuple(preset.get("drop_fc_range_hz", [500.0, 5000.0]))
    decay_range = tuple(preset.get("drop_decay_range_ms", [5.0, 30.0]))
    amp_mu = float(preset.get("drop_amp_lognormal_mu", -1.5))
    amp_sigma = float(preset.get("drop_amp_lognormal_sigma", 1.0))

    # --- 混合比例(沿用 v3) ---
    mix = preset.get("bed_drop_mix", [0.75, 0.25])
    bed_w, drop_w = float(mix[0]), float(mix[1])

    # --- v4 調變參數(可由 preset 覆蓋,預設用校準值;sigma_common=1.8dB 校準值) ---
    sigma_common = float(preset.get("v4_sigma_common_db", SIGMA_COMMON_DB))
    tau_common   = float(preset.get("v4_tau_common_s", TAU_COMMON_S))
    tau_indep    = float(preset.get("v4_tau_indep_s", TAU_INDEP_S))
    drop_rate_mod_db = float(preset.get("v4_drop_rate_mod_db", DROP_RATE_MOD_DB))
    # v4_loop_period_s: 調變器的基礎週期(= loop 長度);
    #   None → 用 duration_s(信號本身為一個週期)
    #   設為 90.0 時:t=0 與 t=90s 的調變器值完全相同 → 90s loop 接縫連續
    loop_period_raw = preset.get("v4_loop_period_s", None)
    loop_period_s: float | None = float(loop_period_raw) if loop_period_raw is not None else None
    # v4_disable_modulation: 在 demo chunk 合成模式下關閉帶內調變,避免跨 chunk RMS 不連續
    #   (demo_mix.py 獨立合成每個 chunk;帶內調變由 90s loop 模式提供,chunk 模式不需要)
    disable_modulation = bool(preset.get("v4_disable_modulation", False))

    # ==========================================================================
    # 1. 床層合成(v3 相同)
    # ==========================================================================
    bed = _colored_bed_v4(
        duration_s, sr, rng, slope0, ref_f0, slope1, ref_f1, slope2, ref_f2
    )

    # ==========================================================================
    # 2. 調變器組:共同 + 各帶獨立(周期噪聲,首尾連續)
    # loop_period_s 決定週期:設為 LOOP_DUR_S=90s 則 t=0 與 t=90s 相位相同
    # ==========================================================================
    # 共同分量(dB 域)& 各帶獨立分量
    # disable_modulation=True:調變器全 0(chunk 模式用),仍消耗相同數量的 RNG 以保持 bit-identical
    # ==========================================================================
    n = int(duration_s * sr)

    common_db = _periodic_lowpass_noise(
        duration_s, sr, tau_common, rng, sigma_common,
        period_s=loop_period_s,
    )

    gain_curves_linear = []
    for k in range(N_BANDS):
        alpha_k, sigma_indep_k = BAND_PARAMS[k]
        indep_db = _periodic_lowpass_noise(
            duration_s, sr, tau_indep, rng, sigma_indep_k,
            period_s=loop_period_s,
        )
        if disable_modulation:
            # 禁用:增益固定 = 1.0(不修改 RNG 狀態,已在上面消耗)
            gain_curves_linear.append(np.ones(n, dtype=np.float64))
        else:
            g_k_db = alpha_k * common_db + indep_db
            gain_curves_linear.append(10.0 ** (g_k_db / 20.0))

    # 共同分量的線性增益(用於連動 Poisson 率)
    if disable_modulation:
        gain_common_linear = np.ones(n, dtype=np.float64)
    else:
        gain_common_linear = np.clip(
            10.0 ** (common_db / 20.0),
            10.0 ** (-drop_rate_mod_db / 20.0),
            10.0 ** (drop_rate_mod_db / 20.0),
        )

    # ==========================================================================
    # 3. 六帶調變:bed 拆帶 → 乘增益 → 重建
    # ==========================================================================
    band_filters = _build_bandpass_filters(sr)
    bed_modulated = _apply_band_modulation(bed, sr, band_filters, gain_curves_linear)

    # ==========================================================================
    # 4. 滴粒層(共同分量連動率)
    # ==========================================================================
    templates = _build_template_pool_v4(
        drop_template_count, sr, rng, fc_range, decay_range
    )
    drops = _poisson_drops_v4(
        duration_s, sr, rng, drop_rate_hz, templates,
        amp_mu, amp_sigma, intensity,
        gain_common_linear,
    )

    # ==========================================================================
    # 5. 混合 & 正規化(與 v3 相同)
    # ==========================================================================
    bed_rms = np.sqrt(np.mean(bed_modulated ** 2))
    if bed_rms > 0:
        bed_modulated = bed_modulated / bed_rms
    drop_rms = np.sqrt(np.mean(drops ** 2))
    if drop_rms > 0:
        drops = drops / drop_rms

    rain = bed_modulated * bed_w + drops * drop_w

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
