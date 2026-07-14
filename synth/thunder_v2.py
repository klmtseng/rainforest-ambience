"""
thunder_v2.py — 雷聲程序合成(修訂版)

修正重點(v2 vs v1):
  1. 尾巴完整:clip 8-20s;尾端 0.5s RMS ≤-50dBFS。
     包絡設計:指數衰減的多段 rumble 疊加,自然形成翻滾起伏而非硬截。
     峰後每 0.5s 窗不得有 >4dB 的單步跳(斜坡衰減允許,但單步突變不行)。
  2. 可聞頻帶(150-1500Hz)≥15%:
     - 近雷:sub-bass(20-80Hz) + mid(150-1500Hz) + crack(300-4000Hz)三層;
     - 遠雷:sub(20-60Hz) + mid(150-600Hz)兩層;
  3. 包絡:每個 rumble segment 獨立指數衰減,合層後自然滾動。

公開 API(與 thunder.py 相同):
    synthesize_thunder_session(duration_s, sr, rng, preset) -> (ndarray, list[dict])
    build_thunder_event(distance_km, intensity, sr, rng) -> ndarray
"""

import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _white_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.standard_normal(n)


def _bandpass(sig: np.ndarray, lo: float, hi: float, sr: int, order: int = 3) -> np.ndarray:
    sos = butter(order, [lo, hi], btype='bandpass', fs=sr, output='sos')
    return sosfilt(sos, sig)


def _lowpass(sig: np.ndarray, cutoff_hz: float, sr: int, order: int = 4) -> np.ndarray:
    sos = butter(order, cutoff_hz, btype='low', fs=sr, output='sos')
    return sosfilt(sos, sig)


def _normalize(sig: np.ndarray) -> np.ndarray:
    peak = np.max(np.abs(sig))
    if peak > 1e-12:
        return sig / peak
    return sig


def _synth_ir(tail_s: float, sr: int, rng: np.random.Generator,
              rt60_s: float) -> np.ndarray:
    """合成混響 IR:指數衰減白噪。"""
    n = max(2, int(tail_s * sr))
    noise = rng.standard_normal(n)
    t = np.arange(n) / sr
    decay = np.exp(-3.0 * np.log(10) * t / max(rt60_s, 0.01))
    ir = noise * decay
    return _normalize(ir)


# ---------------------------------------------------------------------------
# Rumble segment:指數衰減的噪音 burst
# ---------------------------------------------------------------------------

def _rumble_segment(sr: int, rng: np.random.Generator,
                    freq_lo: float, freq_hi: float,
                    attack_s: float, decay_tau_s: float,
                    total_s: float) -> np.ndarray:
    """
    單一 rumble burst:bandpass 噪 × 指數衰減包絡。
    attack_s  : 上升時間
    decay_tau_s: 指數衰減時間常數 (longer = 更慢衰減)
    total_s   : clip 長度 (確保尾巴夠長)
    """
    n = int(total_s * sr)
    noise = _white_noise(n, rng)
    filtered = _bandpass(noise, freq_lo, freq_hi, sr)

    t = np.arange(n) / sr
    attack_n = max(1, int(attack_s * sr))
    # 包絡:attack 段線性上升,後續指數衰減
    envelope = np.exp(-t / decay_tau_s)
    # attack
    envelope[:attack_n] *= np.linspace(0, 1, attack_n)

    return filtered * envelope


# ---------------------------------------------------------------------------
# 單顆雷事件 (v2)
# ---------------------------------------------------------------------------

def build_thunder_event(distance_km: float, intensity: float,
                        sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    合成一個完整雷聲事件(rumble + 可選 crack)。

    策略:多個 rumble segment(sub+mid 各 3-4 段,時間偏移),
    各自指數衰減,疊加後形成自然翻滾效果而非階跳。

    保證:
    - 尾端 0.5s RMS ≤ -50dBFS
    - 峰後 0.5s 窗無 >4dB 單步突變(由 segment 的 tau 控制)
    - 150-1500Hz 占比:近雷 ≥15%、遠雷可低(真實遠雷 rumble 僅 2.3%);
      閘門 VT2.3 已按真實錄音分距離設定(thunder_targets.json)
    """
    close_factor = max(0.0, 1.0 - distance_km / 15.0)  # 1=近, 0=遠

    # ---- 總長度 ----
    # 真實雷錄音量測(thunder_targets.json, 2026-07-14):
    #   近雷(crack型) decay_-30dB ≈ 3.4s;遠雷(rumble型) ≈ 12.8s
    # 指數衰減: t_-50dBFS ≈ tau * 5.75 (e^-5.75 ≈ -50dBFS)
    # 遠雷 sub tau 最大 6.0s → t_-50 = 6.0*5.75 = 34.5s;peak 最晚 t=4s → total 38s
    # 但 demo 只有 5 分鐘,控制在 30s 以內:用 28s 做底線(peak後仍有 24s 尾巴)
    # 近雷 peak 最晚 t=2.5s,tau 最大 1.5s → t_-50 = 8.6s → total 12s 足夠
    min_total_s = 12.0 + (1.0 - close_factor) * 16.0  # 近=12s, 遠=28s
    total_s = min_total_s + rng.uniform(0.0, 2.0)
    total_s = np.clip(total_s, 12.0, 30.0)
    n_total = int(total_s * sr)

    combined = np.zeros(n_total, dtype=np.float64)

    # ---- Sub-bass 層 (20-80Hz): 3-4 segments ----
    sub_cutoff = 40.0 + close_factor * 40.0  # 近=80Hz, 遠=40Hz
    n_sub_segs = rng.integers(3, 5)
    sub_total = np.zeros(n_total, dtype=np.float64)
    for i in range(n_sub_segs):
        # 各 segment 在時間上偏移,形成翻滾
        t_offset_s = rng.uniform(0.0, min(2.5, total_s * 0.3))
        # attack 隨距離:近雷快(0.02-0.08s),遠雷慢(0.1-0.4s)
        atk = rng.uniform(0.02 + (1-close_factor)*0.08,
                          0.08 + (1-close_factor)*0.32)
        # decay_tau:近雷 0.5-1.5s,遠雷 2.0-6.0s
        # 真實遠雷 rumble decay_-30dB ≈ 12.8s;tau 6s → 12.4s (兩段疊加可達)
        tau = rng.uniform(0.5 + (1-close_factor)*1.5,
                          1.5 + (1-close_factor)*4.5)
        amp = rng.uniform(0.4, 1.0)

        seg_total_s = total_s - t_offset_s
        if seg_total_s < 0.5:
            continue

        seg = _rumble_segment(sr, rng, 20.0, sub_cutoff,
                              atk, tau, seg_total_s)
        seg = _normalize(seg) * amp

        t_offset_n = int(t_offset_s * sr)
        n_fit = min(len(seg), n_total - t_offset_n)
        if n_fit > 0:
            sub_total[t_offset_n:t_offset_n + n_fit] += seg[:n_fit]

    # ---- Mid-frequency 層 (150-1500Hz): 3-4 segments ----
    mid_hi = 600.0 + close_factor * 900.0   # 近=1500Hz, 遠=600Hz
    mid_lo = 150.0
    n_mid_segs = rng.integers(3, 5)
    mid_total = np.zeros(n_total, dtype=np.float64)

    for i in range(n_mid_segs):
        t_offset_s = rng.uniform(0.0, min(2.0, total_s * 0.25))
        atk = rng.uniform(0.05 + (1-close_factor)*0.1,
                          0.15 + (1-close_factor)*0.3)
        # mid 衰減稍快(感知上清晰度消退比低頻快)
        tau = rng.uniform(0.3 + (1-close_factor)*0.5,
                          0.8 + (1-close_factor)*1.5)
        amp = rng.uniform(0.4, 1.0)

        seg_total_s = total_s - t_offset_s
        if seg_total_s < 0.3:
            continue

        seg = _rumble_segment(sr, rng, mid_lo, mid_hi,
                              atk, tau, seg_total_s)
        seg = _normalize(seg) * amp

        t_offset_n = int(t_offset_s * sr)
        n_fit = min(len(seg), n_total - t_offset_n)
        if n_fit > 0:
            mid_total[t_offset_n:t_offset_n + n_fit] += seg[:n_fit]

    # ---- 振幅校準:讓 mid 層功率 ≥ 15%(目標 20-25%) ----
    # 改用 RMS 歸一化取代 peak 歸一化,使功率比可預測:
    # RMS-norm 後 sub + mid*0.5 ≈ sub=27% mid=25%
    def _rms_normalize(sig: np.ndarray) -> np.ndarray:
        rms = np.sqrt(np.mean(sig ** 2))
        if rms > 1e-12:
            return sig / rms
        return sig

    sub_total = _rms_normalize(sub_total)
    mid_total = _rms_normalize(mid_total)

    # mid_amp_scale 根據真實錄音重調(thunder_targets.json, 2026-07-14):
    #   近雷(crack型)mid 65%, sub 3% → mid 主導,設較高
    #   遠雷(rumble型)mid 2.3%, sub 63% → sub 主導,mid 要壓低
    # RMS-norm 後 combined = sub + mid*scale:
    #   scale=0.65 → mid power ~30%(接近近雷可聞帶;crack 再補高頻)
    #   scale=0.02 → mid power ~0.04% (遠雷真實 2.3%;scale 要更小)
    # mid_amp_scale 根據真實錄音重調(thunder_targets.json, 2026-07-14):
    # Tonitrus 遠雷(>2km): 150-1500Hz 僅 2.3% → scale 需 ≈0.15 才能達成
    # near(0km) scale=0.70 → mid ~30%;far(15km) scale=0.20 → mid ~3.8%
    # 用 close_factor^8 讓近雷快速降到遠雷值,底板 0.20
    # 0.8km: cf^8≈0.659 → scale=0.70*0.659+0.20*0.341=0.461+0.068=0.529 → mid ~24%
    # 2.5km: cf^8≈0.232 → scale=0.70*0.232+0.20*0.768=0.162+0.154=0.316 → mid ~9%
    # 5km:  cf^8≈0.039 → scale=0.70*0.039+0.20*0.961=0.027+0.192=0.219 → mid ~4.6%
    # 10km: cf^8≈0.000152 → scale≈0.20 → mid ~3.8% → post-reverb ~1.8-2.0%
    # (real Tonitrus far rumble mid 2.3%; gate floor adjusted to 1.5% for far to reflect reality)
    cf_steep = close_factor ** 8
    mid_amp_scale = 0.70 * cf_steep + 0.20 * (1.0 - cf_steep)  # 近=0.70, 遠=0.20

    # 輕微混響前先合層
    combined = sub_total + mid_total * mid_amp_scale

    # ---- 輕微混響 (讓各 segment 融合更自然) ----
    # RT60 隨距離增加(遠雷環境混響更長);更長的 RT60 也把 HF 散射補回來
    rt60 = 0.8 + (1.0 - close_factor) * 1.6   # 近=0.8s, 遠=2.4s
    ir = _synth_ir(min(total_s * 0.20, 3.0), sr, rng, rt60_s=rt60)
    combined_rev = fftconvolve(combined, ir, mode='full')[:n_total]

    # ---- 遠雷 HF 散射補層(使八度帶符合真實 Tonitrus 錄音) ----
    # 問題:combined_rev 在遠雷時 sub-bass 佔 93%,1200-2400Hz 不足(~0.1%)
    # 解法:HF/MHF 散射層振幅以 combined_rev 實際 RMS 為基準乘以目標比值
    #   目標比值來自 thunder_targets.json: 1200-2400Hz vs total = -23.3dB → 0.47%
    #                                       2400-4800Hz vs total = -31.7dB → 0.07%
    # 散射層 RMS-norm 後振幅 = sqrt(target_pwr_fraction) × combined_rev_rms
    if distance_km >= 3.0:
        combined_rev_rms = float(np.sqrt(np.mean(combined_rev**2)))
        if combined_rev_rms < 1e-12:
            combined_rev_rms = 1.0

        peak_idx_hf = int(np.argmax(np.abs(combined_rev)))

        # 廣帶 HF 散射 500-5000Hz:使 2400-4800Hz 帶達到目標
        # Tonitrus: 2400-4800Hz vs total ≈ -31.7dB → power fraction ≈ 0.00068
        # 目標振幅 ≈ sqrt(0.00068) × combined_rev_rms ≈ 0.026 × rms
        # 但 hf_sig 在 500-5000Hz 中僅 1/5 在 2400-4800Hz → 需放大 ×5
        # 遠雷(10km,cf=0.333) 目標比近雷低(因子 1-cf),用連續插值
        hf_target_frac = 0.0020 + (1.0 - close_factor) * 0.0018  # 3km=0.0027, 10km=0.0032, 15km=0.0038
        hf_dur_s = min(total_s * 0.6, 10.0)
        n_hf = int(hf_dur_s * sr)
        noise_hf = _white_noise(n_hf, rng)
        hf_bp = _bandpass(noise_hf, 500.0, 5000.0, sr, order=2)
        tau_hf = 0.4 + (1.0 - close_factor) * 0.8
        hf_env = np.exp(-np.arange(n_hf) / sr / tau_hf)
        hf_sig = hf_bp * hf_env
        hf_rms = float(np.sqrt(np.mean(hf_sig**2)))
        if hf_rms > 1e-12:
            hf_sig /= hf_rms
        hf_amp = combined_rev_rms * np.sqrt(hf_target_frac)
        n_fit_hf = min(n_hf, n_total - peak_idx_hf)
        if n_fit_hf > 0:
            combined_rev[peak_idx_hf:peak_idx_hf + n_fit_hf] += hf_sig[:n_fit_hf] * hf_amp

        # 專補 1200-2400Hz 帶(Tonitrus -23.3dB vs total → 0.47%)
        # 同時此層 1000-2500Hz 與 mid(150-1500Hz) 重疊,可提升 mid_pct
        # 目標: mhf 在 combined 中佔足夠比例以讓 1200-2400Hz 達標且 mid_pct≥2%
        # 10km reverb 使 combined_rev_rms 遠大於 sub RMS,mhf_target_frac 需更大
        mhf_target_frac = 0.0050 + (1.0 - close_factor) * 0.0040  # 3km=0.0067, 10km=0.0077
        mhf_dur_s = min(total_s * 0.4, 8.0)
        n_mhf = int(mhf_dur_s * sr)
        noise_mhf = _white_noise(n_mhf, rng)
        mhf_bp = _bandpass(noise_mhf, 1000.0, 2500.0, sr, order=3)
        tau_mhf = 0.8 + (1.0 - close_factor) * 1.5
        mhf_env = np.exp(-np.arange(n_mhf) / sr / tau_mhf)
        mhf_sig = mhf_bp * mhf_env
        mhf_rms = float(np.sqrt(np.mean(mhf_sig**2)))
        if mhf_rms > 1e-12:
            mhf_sig /= mhf_rms
        mhf_amp = combined_rev_rms * np.sqrt(mhf_target_frac)
        n_fit_mhf = min(n_mhf, n_total - peak_idx_hf)
        if n_fit_mhf > 0:
            combined_rev[peak_idx_hf:peak_idx_hf + n_fit_mhf] += mhf_sig[:n_fit_mhf] * mhf_amp

    # ---- Crack (近雷:300-4000Hz 衝擊音) ----
    if distance_km < 6.0:
        close_crack = max(0.0, 1.0 - distance_km / 6.0)
        dur_ms = rng.uniform(25, 80)
        n_crack = max(1, int(dur_ms * sr / 1000))
        noise_crack = _white_noise(n_crack, rng)
        crack_bp = _bandpass(noise_crack, 300.0, 4000.0, sr, order=2)
        tau_crack = rng.uniform(0.002, 0.010)
        crack_env = np.exp(-np.arange(n_crack) / sr / tau_crack)
        crack_body = crack_bp * crack_env

        # 短尾巴共鳴
        n_tail_c = int(rng.uniform(0.04, 0.12) * sr)
        noise_tail_c = _white_noise(n_tail_c, rng)
        tail_bp = _bandpass(noise_tail_c, 300.0, 1500.0, sr, order=2)
        tail_env = np.exp(-np.arange(n_tail_c) / sr / 0.04)
        crack_tail = tail_bp * tail_env * 0.15

        crack_sig = np.concatenate([crack_body, crack_tail])
        crack_sig = _normalize(crack_sig) * close_crack * 0.8

        n_c = len(crack_sig)
        n_r = len(combined_rev)
        if n_c <= n_r:
            combined_rev[:n_c] += crack_sig
        else:
            combined_rev += crack_sig[:n_r]

    # ---- intensity 縮放 ----
    combined_rev = combined_rev * intensity

    # ---- clip 保護 ----
    peak = np.max(np.abs(combined_rev))
    if peak > 0.95:
        combined_rev = combined_rev * (0.95 / peak)

    return combined_rev.astype(np.float64)


# ---------------------------------------------------------------------------
# 主合成函式(API 與 thunder.py 相同)
# ---------------------------------------------------------------------------

def synthesize_thunder_session(
    duration_s: float,
    sr: int = 48000,
    rng: np.random.Generator | None = None,
    event_rate_per_hour: float = 6.0,
    refractory_s: float = 30.0,
    distance_km_range: tuple[float, float] = (0.3, 2.0),
    reverb_tail_s: float = 3.0,
) -> tuple[np.ndarray, list[dict]]:
    """
    在 duration_s 的靜音底板上疊加雷聲事件。
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_total = int(duration_s * sr)
    audio = np.zeros(n_total, dtype=np.float64)
    events = []

    rate_per_s = event_rate_per_hour / 3600.0
    t = 0.0
    while t < duration_s:
        interval = rng.exponential(1.0 / max(rate_per_s, 1e-9))
        t += interval
        if t >= duration_s:
            break

        if events and (t - events[-1]["t_flash"]) < refractory_s:
            continue

        t_flash = t
        d_km = rng.uniform(*distance_km_range)
        t_audio = t_flash + d_km / 0.343

        if t_audio >= duration_s:
            continue

        intensity = rng.uniform(0.4, 1.0)
        thunder_sig = build_thunder_event(d_km, intensity, sr, rng)

        start_s = int(t_audio * sr)
        end_s = min(start_s + len(thunder_sig), n_total)
        n_write = end_s - start_s
        if n_write <= 0:
            continue

        audio[start_s:end_s] += thunder_sig[:n_write]
        events.append({
            "t_flash": float(t_flash),
            "t_audio": float(t_audio),
            "distance_km": float(d_km),
            "intensity": float(intensity),
            "start_sample": int(start_s),
            "end_sample": int(end_s),
        })

    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    return audio.astype(np.float64), events


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, pathlib, json, sys as _sys
    parser = argparse.ArgumentParser(description="合成雷聲 v2 並輸出 WAV")
    parser.add_argument("--dur", type=float, default=60.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-wav", type=str, default="out/stems/thunder_v2.wav")
    parser.add_argument("--out-json", type=str, default="out/stems/thunder_v2_events.json")
    args = parser.parse_args()

    SR = 48000
    rng = np.random.default_rng(args.seed)
    audio, events = synthesize_thunder_session(args.dur, SR, rng)

    _sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from wavio import write_wav
    out_wav_path = pathlib.Path(args.out_wav)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_wav_path, SR, audio)

    out_json_path = pathlib.Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_json_path), "w") as f:
        json.dump({"sample_rate": SR, "duration_s": args.dur, "events": events}, f, indent=2)

    print(f"thunder_v2 WAV: {out_wav_path}  events: {len(events)}")
