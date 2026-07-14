"""
thunder.py — 雷聲程序合成
架構:
  - rumble:布朗噪(粉紅化 1/f²)→ lowpass 60-150Hz + 多峰包絡(3-5 個 Gaussian peak)
  - crack:近雷(d_km < 5)才加;寬頻衝擊脈衝 + 快速衰減
  - 混響:fftconvolve 合成 IR(指數衰減噪)

輸出:
  synthesize_thunder_session() → (audio: np.ndarray, events: list[dict])
  每個 event:
    { t_flash, t_audio, distance_km, intensity,
      start_sample, end_sample }
  t_audio = t_flash + distance_km / 0.343   (光速∞,聲速 343m/s)

公開 API:
    synthesize_thunder_session(duration_s, sr, rng, preset) -> (ndarray, list[dict])
    build_thunder_event(distance_km, intensity, sr, rng) -> ndarray
"""

import json
import numpy as np
from scipy.signal import butter, sosfilt, fftconvolve


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def _brown_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """布朗噪:白噪累積和,然後去趨勢。"""
    w = rng.standard_normal(n)
    b = np.cumsum(w)
    b -= np.mean(b)
    b /= (np.std(b) + 1e-12)
    return b


def _lowpass(sig: np.ndarray, cutoff_hz: float, sr: int) -> np.ndarray:
    sos = butter(4, cutoff_hz, btype='low', fs=sr, output='sos')
    return sosfilt(sos, sig)


def _synth_ir(tail_s: float, sr: int, rng: np.random.Generator,
              rt60_s: float | None = None) -> np.ndarray:
    """合成混響 IR:指數衰減白噪。"""
    if rt60_s is None:
        rt60_s = tail_s * 0.8
    n = int(tail_s * sr)
    noise = rng.standard_normal(n)
    t = np.arange(n) / sr
    decay = np.exp(-3.0 * np.log(10) * t / rt60_s)
    ir = noise * decay
    ir /= (np.max(np.abs(ir)) + 1e-12)
    return ir


# ---------------------------------------------------------------------------
# rumble 合成
# ---------------------------------------------------------------------------

def _rumble(sr: int, rng: np.random.Generator, distance_km: float,
            duration_s: float | None = None) -> np.ndarray:
    """
    低頻隆隆聲。
    遠雷:截止頻率低、持續時間長。
    近雷:截止頻率高、持續時間短。
    """
    # 持續時間:近雷 2-4s,遠雷 4-8s
    if duration_s is None:
        close_factor = max(0.0, 1.0 - distance_km / 15.0)
        duration_s = 8.0 - close_factor * 4.0
        duration_s += rng.uniform(-0.5, 0.5)

    n = int(duration_s * sr)
    brown = _brown_noise(n, rng)

    # 截止頻率:近雷最高 150Hz,遠雷最低 60Hz
    close_factor = max(0.0, 1.0 - distance_km / 15.0)
    cutoff = 60.0 + close_factor * 90.0

    rumble = _lowpass(brown, cutoff, sr)

    # 多峰包絡:3-5 個 Gaussian peak
    n_peaks = rng.integers(3, 6)
    t = np.linspace(0, 1, n)
    envelope = np.zeros(n, dtype=np.float64)
    for _ in range(n_peaks):
        mu = rng.uniform(0.1, 0.85)
        sigma = rng.uniform(0.05, 0.25)
        amp = rng.uniform(0.4, 1.0)
        envelope += amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)
    envelope = envelope / (np.max(envelope) + 1e-12)

    # 緩慢起音(50ms attack)
    attack_n = min(int(0.05 * sr), n)
    envelope[:attack_n] *= np.linspace(0, 1, attack_n)

    rumble = rumble * envelope

    # 加混響
    ir = _synth_ir(min(duration_s * 0.6, 4.0), sr, rng)
    rumble_rev = fftconvolve(rumble, ir, mode='full')[:n]

    return rumble_rev.astype(np.float64)


# ---------------------------------------------------------------------------
# crack 合成 (近雷才有)
# ---------------------------------------------------------------------------

def _crack(sr: int, rng: np.random.Generator, distance_km: float) -> np.ndarray:
    """
    近雷爆裂聲。distance_km < 5 才顯著。
    寬頻衝擊 + 快速指數衰減。
    """
    dur_ms = rng.uniform(20, 80)
    n = int(dur_ms * sr / 1000)
    noise = rng.standard_normal(n)

    # 高通讓 crack 偏高頻(與 rumble 區分)
    sos = butter(2, 200.0, btype='high', fs=sr, output='sos')
    noise = sosfilt(sos, noise)

    tau = rng.uniform(0.003, 0.015)
    t = np.arange(n) / sr
    envelope = np.exp(-t / tau)

    # 近雷振幅更強
    close_factor = max(0.0, 1.0 - distance_km / 5.0)
    amp = close_factor * rng.uniform(0.6, 1.0)

    return (noise * envelope * amp).astype(np.float64)


# ---------------------------------------------------------------------------
# 單顆雷事件
# ---------------------------------------------------------------------------

def build_thunder_event(distance_km: float, intensity: float,
                        sr: int, rng: np.random.Generator) -> np.ndarray:
    """
    合成一個完整雷聲事件(rumble + 可選 crack)。
    回傳 float64 array。
    """
    rumble = _rumble(sr, rng, distance_km)

    # 近雷加 crack,crack 在 rumble 開頭附近
    if distance_km < 6.0:
        crack = _crack(sr, rng, distance_km)
        n_r = len(rumble)
        n_c = len(crack)
        combined_len = max(n_r, n_c)
        combined = np.zeros(combined_len, dtype=np.float64)
        combined[:n_r] += rumble
        combined[:n_c] += crack
    else:
        combined = rumble

    # intensity 縮放
    combined = combined * intensity

    # 歸一 clip 保護
    peak = np.max(np.abs(combined))
    if peak > 0.95:
        combined = combined * (0.95 / peak)

    return combined.astype(np.float64)


# ---------------------------------------------------------------------------
# 主合成函式
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
    回傳 (audio_float64, events_list)。

    events_list 每條:
        t_flash      : 閃電時刻 (秒)
        t_audio      : 雷聲到達時刻 = t_flash + distance_km / 0.343
        distance_km  : 雷暴距離
        intensity    : 0-1
        start_sample : audio 裡的起始 sample index
        end_sample   : audio 裡的結束 sample index (exclusive)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n_total = int(duration_s * sr)
    audio = np.zeros(n_total, dtype=np.float64)
    events = []

    # 泊松過程生成閃電時刻
    rate_per_s = event_rate_per_hour / 3600.0
    t = 0.0
    while t < duration_s:
        interval = rng.exponential(1.0 / max(rate_per_s, 1e-9))
        t += interval
        if t >= duration_s:
            break

        # 不應期:跳過太近的雷
        if events and (t - events[-1]["t_flash"]) < refractory_s:
            continue

        t_flash = t
        d_km = rng.uniform(*distance_km_range)
        t_audio = t_flash + d_km / 0.343   # 聲速 343 m/s

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

    # 最後 clip 保護
    peak = np.max(np.abs(audio)) if len(audio) > 0 else 0.0
    if peak > 0.95:
        audio = audio * (0.95 / peak)

    return audio.astype(np.float64), events


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse, pathlib
    parser = argparse.ArgumentParser(description="合成雷聲並輸出 WAV + events JSON")
    parser.add_argument("--dur", type=float, default=60.0, help="時長(秒)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-wav", type=str, default="out/stems/thunder.wav")
    parser.add_argument("--out-json", type=str, default="out/stems/thunder_events.json")
    args = parser.parse_args()

    SR = 48000
    rng = np.random.default_rng(args.seed)
    audio, events = synthesize_thunder_session(args.dur, SR, rng)

    # 寫 WAV
    import sys as _sys
    _sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from wavio import write_wav
    out_wav_path = pathlib.Path(args.out_wav)
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(out_wav_path, SR, audio)

    # 寫 JSON
    out_json_path = pathlib.Path(args.out_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_json_path), "w") as f:
        json.dump({"sample_rate": SR, "duration_s": args.dur, "events": events}, f, indent=2)

    print(f"thunder WAV: {out_wav_path}  events: {len(events)}")
    print(f"thunder JSON: {out_json_path}")
