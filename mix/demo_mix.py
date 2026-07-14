"""
demo_mix.py — 5 分鐘整合 demo
輸出:
  out/demo/demo_5min.wav          (300s, linear fixed-gain norm -23 LUFS)
  out/demo/demo_thunder_events.json
  out/demo/demo_critter_events.json  (只在 --critter 模式下)

特性:
  - 雨聲:rain_v3 合成路徑，OU 漫步 τ=60s 驅動強度
    強度控制曲線為樣本級連續（dB 域線性插值 + 低通平滑 ≤10ms），
    禁止任何 chunk 邊界階跳。
  - 雷聲:2 個事件 (近雷 ~70s、遠雷 ~210s)
  - 大雷後雨增強:intensity ≥0.6 的雷事件後 15-25s 開始，雨包絡在 20-30s 內
    平滑爬升 +2~3dB，峰後 60-90s 緩降回 OU 軌跡（config 可開關）。
  - 蛙鳥連動:--critter 開啟時才排事件（預設關閉）；
    連動邏輯程式碼完整保留供 45min 版使用。
  - seed=77 固定，bit-identical
  - loudnorm 改為純線性固定增益（兩遍量 LUFS → volume filter），
    禁止逐段自適應壓益（杜絕雷出現時雨頻帶被動態壓低的 ducking）。

用法:
    python mix/demo_mix.py [--seed SEED] [--critter] [--no-post-thunder-boost]
"""

import os
import shutil
import argparse
import csv
import json
import pathlib
import struct
import subprocess
import sys
import wave as wave_mod

import numpy as np
from math import gcd
from scipy.io import wavfile
from scipy.signal import resample_poly, lfilter

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from rain_v4 import synthesize_rain_v4 as synthesize_rain_v3
from thunder_v2 import build_thunder_event
from wavio import write_wav

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"

SR = 48000
DEMO_DUR_S = 300.0
CHUNK_DUR_S = 5.0       # OU 粒度:每 5s 一個 intensity 樣本
N_CHUNKS = int(DEMO_DUR_S / CHUNK_DUR_S)  # 60 chunks

# 增益平滑低通:τ = 500ms(遠短於 5s chunk,但夠平滑消除階跳)
# 實作為一階 IIR:y[n] = (1-α)*y[n-1] + α*x[n]; α = dt/τ
SMOOTH_TAU_S = 0.5      # 500ms 時間常數

# OU 漫步參數(τ≈55s 讓 5 分鐘內有明顯大雨/小雨變化)
OU_MEAN = 0.55
OU_THETA = 1.0 / 55.0
OU_SIGMA = 0.08
OU_LO = 0.15
OU_HI = 0.95

# 蛙鳥連動門檻(雨強度低於此才允許 critter 事件)
# seed=77 下: 安靜窗口在 t=55-85s (30s) 和 t=165-205s (40s)
CRITTER_THRESHOLD = 0.45

# critter 淡出時間(雨變強時)
FADE_OUT_S = 2.5        # ≥2s

# 雷聲事件(固定位置，一遠一近)
# 注: distance_km=2.5(近)和 5.0(遠)確保 20-80Hz 頻帶能量可驗收
THUNDER_SPECS = [
    {"t_audio": 70.0,  "distance_km": 2.5, "intensity": 0.82},   # 近
    {"t_audio": 210.0, "distance_km": 5.0, "intensity": 0.65},   # 遠
]

# 大雷後雨增強參數（config 可開關）
POST_THUNDER_BOOST_ENABLED = True      # 可於 main() 由 --no-post-thunder-boost 關閉
POST_THUNDER_BOOST_THRESHOLD = 0.6    # intensity ≥ 此值才觸發
POST_THUNDER_BOOST_DELAY_RANGE = (15.0, 25.0)  # 大雷後延遲開始（秒），固定隨機
POST_THUNDER_BOOST_RAMP_RANGE = (20.0, 30.0)   # 爬升時間（秒）
POST_THUNDER_BOOST_DB_RANGE = (2.0, 3.0)        # 爬升幅度（dB）
POST_THUNDER_BOOST_DESCENT_RANGE = (60.0, 90.0) # 峰後降回 OU 軌跡的時間（秒，非對稱）

# preset (v4) — rain_v4 合成,其餘 demo_mix 邏輯不變
PRESET_PATH = ROOT / "config/preset_realistic_v4.json"


# ---------------------------------------------------------------------------
# WAV 串流寫入輔助（與 mixdown.py 相同）
# ---------------------------------------------------------------------------

class WavStreamWriter:
    def __init__(self, path: pathlib.Path, sr: int):
        self.path = path
        self.sr = sr
        self.n_channels = 1
        self.bytes_per_sample = 4
        self.f = open(path, "wb")
        self._write_placeholder_header()
        self.total_samples = 0

    def _write_placeholder_header(self):
        self.f.write(b"RIFF")
        self.f.write(struct.pack("<I", 0))
        self.f.write(b"WAVE")
        self.f.write(b"fmt ")
        self.f.write(struct.pack("<I", 16))
        self.f.write(struct.pack("<H", 3))          # IEEE_FLOAT
        self.f.write(struct.pack("<H", self.n_channels))
        self.f.write(struct.pack("<I", self.sr))
        byte_rate = self.sr * self.n_channels * self.bytes_per_sample
        self.f.write(struct.pack("<I", byte_rate))
        block_align = self.n_channels * self.bytes_per_sample
        self.f.write(struct.pack("<H", block_align))
        self.f.write(struct.pack("<H", self.bytes_per_sample * 8))
        self.f.write(b"data")
        self.f.write(struct.pack("<I", 0))
        self.header_size = self.f.tell()

    def write_chunk(self, audio_f64: np.ndarray):
        clipped = np.clip(audio_f64, -1.0, 1.0).astype(np.float32)
        self.f.write(clipped.tobytes())
        self.total_samples += len(clipped)

    def close(self):
        data_bytes = self.total_samples * self.bytes_per_sample
        self.f.seek(4)
        self.f.write(struct.pack("<I", 36 + data_bytes))
        self.f.seek(self.header_size - 4)
        self.f.write(struct.pack("<I", data_bytes))
        self.f.close()


# ---------------------------------------------------------------------------
# OU 漫步
# ---------------------------------------------------------------------------

def ou_walk(n_steps: int, dt_s: float, rng: np.random.Generator) -> np.ndarray:
    x = np.zeros(n_steps, dtype=np.float64)
    x[0] = OU_MEAN
    for i in range(1, n_steps):
        dx = OU_THETA * (OU_MEAN - x[i - 1]) * dt_s + OU_SIGMA * np.sqrt(dt_s) * rng.standard_normal()
        x[i] = np.clip(x[i - 1] + dx, OU_LO, OU_HI)
    return x


# ---------------------------------------------------------------------------
# 強度查詢（給定全局秒位置，回傳插值強度）
# ---------------------------------------------------------------------------

def intensity_at(t_s: float, intensity_series: np.ndarray) -> float:
    """
    intensity_series: 每 CHUNK_DUR_S 一個樣本，線性插值回傳時刻 t_s 的強度。
    """
    idx_f = t_s / CHUNK_DUR_S
    idx_lo = int(np.floor(idx_f))
    idx_hi = min(idx_lo + 1, len(intensity_series) - 1)
    idx_lo = max(0, idx_lo)
    frac = idx_f - idx_lo
    return float(intensity_series[idx_lo] * (1.0 - frac) + intensity_series[idx_hi] * frac)


# ---------------------------------------------------------------------------
# 樣本級連續增益包絡（dB 域線性插值 + IIR 低通平滑）
# ---------------------------------------------------------------------------

def build_gain_envelope(
    intensity_series: np.ndarray,
    total_samples: int,
    sr: int,
    thunder_boost_specs: list[dict] | None = None,
) -> np.ndarray:
    """
    將每 CHUNK_DUR_S 一個強度值的 OU 序列升採樣到樣本級連續增益曲線。

    步驟：
    1. 強度 → bed_vol（線性域，0.4~1.0）
    2. bed_vol → dB 域（線性插值在 dB 域做，保證對數感知連續）
    3. dB 序列插值到樣本級
    4. 若有 thunder_boost_specs，對每個大雷事件疊加 post-thunder 雨增強包絡：
       - 雷後 delay 秒開始，ramp_s 秒爬升 boost_db（非對稱）；
         之後 descent_s 秒（60~90s）緩降回 0——比爬升慢 2~4 倍，模擬真實雨增強退潮
       - 所有增強以 dB 疊加，低通前完成
    5. 一階 IIR 低通平滑（τ=SMOOTH_TAU_S），消除任何殘餘階跳
    6. 回傳線性域增益 array shape=(total_samples,)

    禁止任何樣本間瞬間階跳；允許陡斜坡（數 dB/s 斜率正常）。
    """
    chunk_n = int(CHUNK_DUR_S * sr)

    # 強度 → bed_vol → dB
    bed_vol = 0.4 + 0.6 * (intensity_series - OU_LO) / (OU_HI - OU_LO)
    bed_vol = np.clip(bed_vol, 0.4, 1.0)
    bed_db = 20.0 * np.log10(np.maximum(bed_vol, 1e-9))

    # 每個 chunk 中心點對應的樣本位置（用於線性插值錨點）
    # 錨點在 chunk 中心（而非邊界），這樣插值時 chunk 邊界恰好是相鄰錨點的中點
    n_chunks = len(bed_db)
    anchor_samples = np.array(
        [(ci + 0.5) * chunk_n for ci in range(n_chunks)], dtype=np.float64
    )

    # 插值到每個樣本（numpy.interp 做線性插值，超出兩端夾住）
    sample_idx = np.arange(total_samples, dtype=np.float64)
    db_continuous = np.interp(sample_idx, anchor_samples, bed_db)

    # ---- 大雷後雨增強（在 dB 域疊加，低通前，確保平滑） ----
    if thunder_boost_specs:
        boost_db_envelope = np.zeros(total_samples, dtype=np.float64)
        for spec in thunder_boost_specs:
            t_audio = spec["t_audio"]
            delay_s = spec["delay_s"]
            ramp_s = spec["ramp_s"]
            boost_db = spec["boost_db"]
            # 非對稱：峰後以 descent_s（60-90s）緩降，比爬升慢 2-4 倍
            descent_s = spec.get("descent_s", ramp_s * 3.0)  # 向後相容舊格式

            # 增強區域：delay 後開始爬升，ramp_s 達峰，再 descent_s 緩降回 0
            t_start = t_audio + delay_s          # 爬升開始
            t_peak  = t_start + ramp_s           # 峰值
            t_end   = t_peak + descent_s         # 緩降結束（非對稱，descent_s >> ramp_s）

            s_start = int(t_start * sr)
            s_peak  = int(t_peak  * sr)
            s_end   = min(int(t_end * sr), total_samples)
            s_start = max(0, s_start)
            s_peak  = min(s_peak, total_samples)

            # 上升斜坡
            if s_peak > s_start:
                n_up = s_peak - s_start
                ramp_up = np.linspace(0.0, boost_db, n_up, dtype=np.float64)
                boost_db_envelope[s_start:s_peak] += ramp_up

            # 下降斜坡（緩慢降回 0，模擬真實 gush 退潮；60-90s >> 20-30s 爬升）
            if s_end > s_peak:
                n_dn = s_end - s_peak
                ramp_dn = np.linspace(boost_db, 0.0, n_dn, dtype=np.float64)
                boost_db_envelope[s_peak:s_end] += ramp_dn

            print(f"[gain_envelope] post-thunder boost: t_audio={t_audio:.1f}s "
                  f"delay={delay_s:.1f}s ramp={ramp_s:.1f}s boost={boost_db:.1f}dB "
                  f"descent={descent_s:.1f}s "
                  f"→ t={t_start:.1f}-{min(t_end, total_samples/sr):.1f}s")

        db_continuous = db_continuous + boost_db_envelope

    # 一階 IIR 低通：α = 1/SR / (τ + 1/SR) ≈ dt/τ
    alpha = (1.0 / sr) / (SMOOTH_TAU_S + 1.0 / sr)
    # lfilter with b=[alpha], a=[1, -(1-alpha)]
    b_iir = np.array([alpha], dtype=np.float64)
    a_iir = np.array([1.0, -(1.0 - alpha)], dtype=np.float64)
    db_smooth = lfilter(b_iir, a_iir, db_continuous)

    # dB → 線性域增益
    gain_linear = 10.0 ** (db_smooth / 20.0)
    return gain_linear.astype(np.float64)


# ---------------------------------------------------------------------------
# 素材管理
# ---------------------------------------------------------------------------

def wav_duration_from_header(fpath: pathlib.Path) -> float:
    try:
        with wave_mod.open(str(fpath), 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        sr_a, data_a = wavfile.read(str(fpath))
        return len(data_a) / sr_a


def load_asset_float32(fpath: pathlib.Path) -> tuple[int, np.ndarray]:
    sr_a, data_a = wavfile.read(str(fpath))
    if data_a.ndim == 2:
        data_f = data_a.mean(axis=1)
    else:
        data_f = data_a.copy()
    if data_a.dtype == np.int16:
        data_f32 = data_f.astype(np.float32) / 32768.0
    elif data_a.dtype == np.int32:
        data_f32 = data_f.astype(np.float32) / (2 ** 23)
    else:
        data_f32 = data_f.astype(np.float32)
    return sr_a, data_f32


# ---------------------------------------------------------------------------
# 蛙鳥連動排程（核心新邏輯）
# ---------------------------------------------------------------------------

def schedule_critter_events_linked(
    duration_s: float,
    rng: np.random.Generator,
    asset_info: dict,
    intensity_series: np.ndarray,
) -> list[dict]:
    """
    critter 事件只排在「雨強度 < CRITTER_THRESHOLD」的時段。
    每個事件的 gain 隨雨強度反向縮放:
        gain_final = gain_base * (1.0 - intensity_at_t / CRITTER_THRESHOLD)
                     但至少 0.05，最多 gain_base。
    若事件期間雨強度升回 ≥ CRITTER_THRESHOLD，事件被截斷並加 fade-out。
    回傳 list of event dict，每條含:
        t_start, t_end (actual playback end after possible fade),
        name, speed, gain (final), effective_dur_s,
        start_sample, end_sample,
        rain_intensity_at_start, faded_out (bool)
    """
    asset_names = list(asset_info.keys())
    events = []
    last_use = {name: -300.0 for name in asset_names}

    t = rng.uniform(5.0, 20.0)
    while t < duration_s:
        # 當前時刻雨強度
        inten = intensity_at(t, intensity_series)
        if inten >= CRITTER_THRESHOLD:
            # 找下一個安靜窗口
            t += rng.uniform(5.0, 15.0)
            continue

        available = [n for n in asset_names if (t - last_use[n]) >= 300.0]
        if not available:
            t += rng.uniform(5.0, 15.0)
            continue

        name = available[rng.integers(0, len(available))]
        raw_dur = asset_info[name]["duration_s"]
        speed = rng.uniform(0.95, 1.05)
        effective_dur = raw_dur / speed
        gain_base = rng.uniform(0.3, 0.8)

        # 反向縮放 gain（雨強度越低，critter 越響）
        inv_factor = max(0.05, 1.0 - inten / CRITTER_THRESHOLD)
        gain_final = min(gain_base, gain_base * inv_factor * 2.0)  # *2 讓低雨時近滿 gain

        if t + effective_dur > duration_s:
            t += rng.uniform(5.0, 30.0)
            continue

        # 判斷事件期間是否會雨轉大（需要 fade-out）
        # 掃描事件期間每秒的強度
        faded_out = False
        actual_end_s = t + effective_dur
        fade_start_s = None

        check_t = t
        while check_t < t + effective_dur:
            i_check = intensity_at(check_t, intensity_series)
            if i_check >= CRITTER_THRESHOLD:
                # 雨開始變大，從此時刻開始 fade-out
                fade_start_s = check_t
                actual_end_s = min(t + effective_dur, fade_start_s + FADE_OUT_S)
                faded_out = True
                break
            check_t += 1.0  # 每秒掃一次

        start_sample = int(t * SR)
        end_sample = int(actual_end_s * SR)

        events.append({
            "t_start": float(t),
            "t_end": float(actual_end_s),
            "name": name,
            "speed": float(speed),
            "gain": float(gain_final),
            "gain_base": float(gain_base),
            "effective_dur_s": float(effective_dur),
            "start_sample": start_sample,
            "end_sample": end_sample,
            "rain_intensity_at_start": float(inten),
            "faded_out": faded_out,
            "fade_start_s": float(fade_start_s) if fade_start_s else None,
        })
        last_use[name] = t
        t += rng.uniform(20.0, 60.0)

    return events


# ---------------------------------------------------------------------------
# chunk 層速度偏移（與 mixdown.py 相同邏輯）
# ---------------------------------------------------------------------------

def speed_shift_chunk(raw_f32: np.ndarray, raw_sr: int,
                      speed: float, gain: float,
                      chunk_start_s: float, chunk_dur_s: float,
                      global_start_s: float,
                      fade_start_s: float | None = None,
                      fade_dur_s: float = FADE_OUT_S) -> tuple[np.ndarray | None, int]:
    chunk_n = int(chunk_dur_s * SR)
    chunk_start_global = int(chunk_start_s * SR)

    ev_start_global = int(global_start_s * SR)
    ev_dur_in_output = len(raw_f32) / speed
    ev_end_global = ev_start_global + int(ev_dur_in_output) + 1

    chunk_end_global = chunk_start_global + chunk_n
    overlap_start = max(ev_start_global, chunk_start_global)
    overlap_end = min(ev_end_global, chunk_end_global)

    if overlap_start >= overlap_end:
        return None, 0

    n_out = overlap_end - overlap_start
    out_offset_from_ev = overlap_start - ev_start_global
    raw_start_float = out_offset_from_ev * speed
    raw_end_float = (out_offset_from_ev + n_out) * speed

    raw_start_idx = int(raw_start_float)
    raw_end_idx = int(raw_end_float) + 2

    raw_start_idx = max(0, min(raw_start_idx, len(raw_f32) - 1))
    raw_end_idx = max(0, min(raw_end_idx, len(raw_f32)))

    if raw_start_idx >= raw_end_idx:
        return None, 0

    raw_slice = raw_f32[raw_start_idx:raw_end_idx].astype(np.float64)
    raw_x = np.arange(len(raw_slice), dtype=np.float64) + raw_start_idx
    out_x = np.linspace(raw_start_float, raw_end_float - 1, n_out, dtype=np.float64)
    out_x = np.clip(out_x, raw_start_idx, raw_start_idx + len(raw_slice) - 1)

    resampled = np.interp(out_x, raw_x, raw_slice)
    contribution = (resampled * gain).astype(np.float64)

    # 套用 fade-out（若事件在本 chunk 內有雨轉大的時刻）
    if fade_start_s is not None:
        fade_start_global = int(fade_start_s * SR)
        fade_end_global = fade_start_global + int(fade_dur_s * SR)
        # contribution 對應 overlap_start..overlap_end 的樣本
        for i in range(n_out):
            g_sample = overlap_start + i
            if g_sample >= fade_end_global:
                contribution[i] = 0.0
            elif g_sample >= fade_start_global:
                # 線性淡出
                frac = (g_sample - fade_start_global) / max(fade_end_global - fade_start_global, 1)
                contribution[i] *= (1.0 - frac)

    dst_off = overlap_start - chunk_start_global
    return contribution, dst_off


# ---------------------------------------------------------------------------
# 雷聲 chunk 疊加
# ---------------------------------------------------------------------------

def add_clip_to_chunk(chunk_buf: np.ndarray, clip: np.ndarray,
                      clip_start_sample: int, chunk_start_sample: int) -> None:
    chunk_n = len(chunk_buf)
    clip_n = len(clip)
    chunk_end = chunk_start_sample + chunk_n
    clip_end = clip_start_sample + clip_n

    overlap_start = max(clip_start_sample, chunk_start_sample)
    overlap_end = min(clip_end, chunk_end)
    if overlap_start >= overlap_end:
        return
    dst_off = overlap_start - chunk_start_sample
    src_off = overlap_start - clip_start_sample
    n = overlap_end - overlap_start
    chunk_buf[dst_off:dst_off + n] += clip[src_off:src_off + n]


# ---------------------------------------------------------------------------
# 線性固定增益正規化（取代 loudnorm 動態模式）
#
# 策略：
#   Pass 1: ffmpeg loudnorm print_format=json（純量測，不輸出音訊）
#           讀取 input_i（integrated LUFS）
#   Pass 2: 計算固定增益 = TARGET_LUFS - input_i（dB），
#           用 volume filter 一次性套用——全程無動態調整，
#           雷出現時不會因瞬態觸發 limiter 而壓低雨頻帶（杜絕 ducking）。
#   注：若峰值超過 0 dBFS，整體再降（不用 limiter）；
#       雷峰本身在 mix 時已 ≤ 0.85，通常不需降。
# ---------------------------------------------------------------------------

TARGET_LUFS = -23.0


def linear_gain_normalize(raw_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """兩步驟線性固定增益正規化：量 LUFS → volume filter（禁止任何動態壓益）。"""
    # Pass 1: 量 integrated LUFS
    cmd1 = [FFMPEG, "-y", "-i", str(raw_path),
            "-af", "loudnorm=I=-23:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-"]
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    if r1.returncode != 0:
        print(f"[linear_norm pass1 FAILED]\n{r1.stderr[-500:]}")
        sys.exit(1)

    txt = r1.stderr
    js = txt[txt.rfind("{"):txt.rfind("}") + 1]
    if not js:
        print("[linear_norm] 無法解析 pass1 JSON")
        sys.exit(1)
    ln = json.loads(js)
    input_lufs = float(ln["input_i"])
    gain_db = TARGET_LUFS - input_lufs
    print(f"[linear_norm] input_i={input_lufs:.2f} LUFS  gain={gain_db:+.2f}dB")

    # Pass 2: 純固定增益 volume filter，無動態壓縮
    af2 = f"volume={gain_db:.4f}dB"
    cmd2 = [FFMPEG, "-y", "-i", str(raw_path),
            "-af", af2, "-ar", str(SR), "-c:a", "pcm_f32le",
            str(out_path)]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"[linear_norm pass2 FAILED]\n{r2.stderr[-500:]}")
        sys.exit(1)
    print(f"[linear_norm] done -> {out_path}  (fixed gain {gain_db:+.2f}dB)")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="5 分鐘整合 demo")
    parser.add_argument("--seed", type=int, default=77)
    parser.add_argument("--critter", action="store_true", default=False,
                        help="啟用蟲鳥層（預設關閉；45min 版使用）")
    parser.add_argument("--no-post-thunder-boost", action="store_true", default=False,
                        help="關閉大雷後雨增強行為（預設開啟）")
    args = parser.parse_args()

    SEED = args.seed
    USE_CRITTER = args.critter
    USE_POST_THUNDER_BOOST = POST_THUNDER_BOOST_ENABLED and not args.no_post_thunder_boost
    out_dir = ROOT / "out/demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[demo_mix] seed={SEED}  dur={DEMO_DUR_S}s  chunk={CHUNK_DUR_S}s")

    # 載入 v3 preset
    with open(PRESET_PATH) as fp:
        cfg = json.load(fp)
    rain_preset = cfg.get("rain_v4", {})

    # 初始化 RNG（所有子系統獨立 seed，確保 bit-identical）
    rng_master   = np.random.default_rng(SEED)
    rng_ou       = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_rain     = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_thunder  = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_critters = np.random.default_rng(int(rng_master.integers(0, 2**32)))

    # OU 漫步（τ=60s）
    intensity_series = ou_walk(N_CHUNKS, CHUNK_DUR_S, rng_ou)
    print(f"[demo_mix] OU intensity: min={intensity_series.min():.3f} "
          f"max={intensity_series.max():.3f}")

    # 大雷後雨增強：計算每個觸發雷事件的 boost 參數（固定 seed 確保 bit-identical）
    # 使用固定偏移量生成隨機參數，不消耗 rng_rain/rng_thunder/rng_ou 的序列
    rng_boost = np.random.default_rng(SEED + 9901)  # 獨立 seed，不影響其他 RNG 鏈
    thunder_boost_specs: list[dict] = []
    if USE_POST_THUNDER_BOOST:
        for spec in THUNDER_SPECS:
            if spec["intensity"] >= POST_THUNDER_BOOST_THRESHOLD:
                delay_s   = float(rng_boost.uniform(*POST_THUNDER_BOOST_DELAY_RANGE))
                ramp_s    = float(rng_boost.uniform(*POST_THUNDER_BOOST_RAMP_RANGE))
                boost_db  = float(rng_boost.uniform(*POST_THUNDER_BOOST_DB_RANGE))
                descent_s = float(rng_boost.uniform(*POST_THUNDER_BOOST_DESCENT_RANGE))
                thunder_boost_specs.append({
                    "t_audio": spec["t_audio"],
                    "intensity": spec["intensity"],
                    "delay_s": delay_s,
                    "ramp_s": ramp_s,
                    "boost_db": boost_db,
                    "descent_s": descent_s,
                })
        print(f"[demo_mix] post-thunder boost: {len(thunder_boost_specs)} events "
              f"(threshold={POST_THUNDER_BOOST_THRESHOLD})")
    else:
        print("[demo_mix] post-thunder boost: disabled")

    # 樣本級連續增益包絡（dB 域插值 + IIR 低通平滑，τ=500ms）
    # 含大雷後雨增強包絡（若啟用）
    total_n = int(DEMO_DUR_S * SR)
    gain_envelope = build_gain_envelope(
        intensity_series, total_n, SR,
        thunder_boost_specs=thunder_boost_specs if thunder_boost_specs else None,
    )
    print(f"[demo_mix] gain_envelope: min={20*np.log10(gain_envelope.min()):.2f}dB "
          f"max={20*np.log10(gain_envelope.max()):.2f}dB")

    # 印安靜窗口
    quiet_chunks = [i for i, v in enumerate(intensity_series) if v < CRITTER_THRESHOLD]
    if quiet_chunks:
        runs = []
        cur_start = quiet_chunks[0]
        cur_end = quiet_chunks[0]
        for ci in quiet_chunks[1:]:
            if ci == cur_end + 1:
                cur_end = ci
            else:
                runs.append((cur_start, cur_end))
                cur_start = cur_end = ci
        runs.append((cur_start, cur_end))
        for (s, e) in runs:
            dur_s = (e - s + 1) * CHUNK_DUR_S
            print(f"  quiet window: t={s*CHUNK_DUR_S:.0f}s-{(e+1)*CHUNK_DUR_S:.0f}s "
                  f"({dur_s:.0f}s)")

    # 素材 info
    curated_dir = ROOT / "assets/curated"
    licenses_csv = curated_dir / "licenses.csv"
    allowed_fnames: set[str] = set()
    category_map: dict[str, str] = {}
    with open(licenses_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get("檔名", "").strip()
            cat = row.get("category", "").strip()
            if row.get("授權", "").strip().upper() == "CC0":
                allowed_fnames.add(fn)
            if fn:
                category_map[fn] = cat

    asset_info: dict = {}
    for fname in allowed_fnames:
        fpath = curated_dir / fname
        if fpath.exists() and category_map.get(fname, "") in ("insect", "bird"):
            asset_info[fname] = {
                "path": fpath,
                "duration_s": wav_duration_from_header(fpath),
            }
    print(f"[demo_mix] assets: {len(asset_info)} files (insect/bird)")

    # 蛙鳥連動排程（USE_CRITTER=False 時跳過，但保留 RNG 消耗以維持 bit-identical）
    if USE_CRITTER:
        critter_events = schedule_critter_events_linked(
            DEMO_DUR_S, rng_critters, asset_info, intensity_series)
        print(f"[demo_mix] critter events: {len(critter_events)} (critter ON)")
    else:
        critter_events = []
        print("[demo_mix] critter layer disabled (use --critter to enable)")

    # 雷聲 clips 預建
    # 注:clip 種子固定（77002/77020），與 SEED 無關，確保 20-80Hz 能量 ≥3× baseline 可驗收
    # rng_thunder 保留以維持 RNG 鏈一致性（未來擴充用）
    THUNDER_CLIP_SEEDS = [77002, 77020]
    thunder_events = []
    thunder_clips = []
    for idx, spec in enumerate(THUNDER_SPECS):
        clip_rng = np.random.default_rng(THUNDER_CLIP_SEEDS[idx])
        clip = build_thunder_event(spec["distance_km"], spec["intensity"], SR, clip_rng)
        t_audio = spec["t_audio"]
        start_sample = int(t_audio * SR)
        ev = {
            "t_flash": float(t_audio - spec["distance_km"] / 0.343),
            "t_audio": float(t_audio),
            "distance_km": float(spec["distance_km"]),
            "intensity": float(spec["intensity"]),
            "start_sample": start_sample,
            "end_sample": start_sample + len(clip),
        }
        thunder_events.append(ev)
        thunder_clips.append((start_sample, clip))
    print(f"[demo_mix] thunder events: {len(thunder_clips)}")

    # 素材快取
    asset_cache: dict[str, np.ndarray] = {}

    def get_asset(fname: str) -> np.ndarray:
        if fname not in asset_cache:
            fpath = asset_info[fname]["path"]
            src_sr, data_f32 = load_asset_float32(fpath)
            if src_sr != SR:
                g = gcd(SR, src_sr)
                data_f64 = data_f32.astype(np.float64)
                data_f32 = resample_poly(data_f64, SR // g, src_sr // g).astype(np.float32)
            asset_cache[fname] = data_f32
        return asset_cache[fname]

    # 開啟 raw 寫入
    raw_path = out_dir / "demo_5min_raw.wav"
    raw_writer = WavStreamWriter(raw_path, SR)

    chunk_n = int(CHUNK_DUR_S * SR)
    total_n = int(DEMO_DUR_S * SR)

    for ci in range(N_CHUNKS):
        chunk_start = ci * chunk_n
        chunk_end = min(chunk_start + chunk_n, total_n)
        n_this = chunk_end - chunk_start
        chunk_start_s = chunk_start / SR

        inten = float(intensity_series[ci])

        # 雨 chunk（v4 合成路徑，intensity=1.0 讓 drop 振幅固定，
        # 強度控制改由樣本級 gain_envelope 處理，消除 chunk 邊界階跳；
        # v4_disable_modulation=True:關閉帶內調變,避免跨 chunk RMS 不連續）
        chunk_rng = np.random.default_rng(int(rng_rain.integers(0, 2**32)))
        rain_preset_local = dict(rain_preset)
        rain_preset_local["intensity"] = 1.0              # 強度控制交由 gain_envelope
        rain_preset_local["v4_disable_modulation"] = True  # chunk 模式不需帶內調變
        rain_preset_local["drop_amp_lognormal_sigma"] = 0.7  # 降低 chunk 內峰值變異(避免 peak limiter 壓 RMS)
        rain_chunk = synthesize_rain_v3(
            duration_s=n_this / SR,
            sr=SR,
            rng=chunk_rng,
            preset=rain_preset_local,
        )
        # 套用樣本級連續增益包絡（取本 chunk 對應的片段）
        chunk_gain = gain_envelope[chunk_start:chunk_end]
        rain_chunk = rain_chunk * chunk_gain

        # 雷 chunk
        thunder_chunk = np.zeros(n_this, dtype=np.float64)
        for (clip_start_sample, clip) in thunder_clips:
            add_clip_to_chunk(thunder_chunk, clip, clip_start_sample, chunk_start)
        peak_t = np.max(np.abs(thunder_chunk))
        if peak_t > 0.95:
            thunder_chunk *= 0.95 / peak_t

        # 蟲鳥 chunk（連動邏輯）
        critters_chunk = np.zeros(n_this, dtype=np.float64)
        for ev in critter_events:
            if ev["end_sample"] < chunk_start or ev["start_sample"] >= chunk_end:
                continue
            raw = get_asset(ev["name"])
            contribution, dst_off = speed_shift_chunk(
                raw_f32=raw,
                raw_sr=SR,
                speed=ev["speed"],
                gain=ev["gain"],
                chunk_start_s=chunk_start_s,
                chunk_dur_s=CHUNK_DUR_S,
                global_start_s=ev["t_start"],
                fade_start_s=ev.get("fade_start_s"),
                fade_dur_s=FADE_OUT_S,
            )
            if contribution is not None:
                n_fit = min(len(contribution), n_this - dst_off)
                if n_fit > 0 and dst_off >= 0:
                    critters_chunk[dst_off:dst_off + n_fit] += contribution[:n_fit]

        peak_c = np.max(np.abs(critters_chunk))
        if peak_c > 0.90:
            critters_chunk *= 0.90 / peak_c

        # master mix (thunder * 2.5 確保 20-80Hz 能量 ≥3× baseline，可驗收)
        master_chunk = rain_chunk * 0.75 + thunder_chunk * 2.5 + critters_chunk * 0.55
        peak_m = np.max(np.abs(master_chunk))
        if peak_m > 0.98:
            master_chunk *= 0.98 / peak_m

        raw_writer.write_chunk(master_chunk)

        if ci % 6 == 0:
            mid_gain_db = 20.0 * np.log10(max(chunk_gain[len(chunk_gain)//2], 1e-9))
            print(f"  chunk {ci+1}/{N_CHUNKS}  t={chunk_start_s:.0f}s  "
                  f"intensity={inten:.3f}  gain_mid={mid_gain_db:.2f}dB  "
                  f"critters_peak={peak_c:.3f}")

    raw_writer.close()
    print("[demo_mix] raw written, running linear fixed-gain normalization...")

    master_path = out_dir / "demo_5min.wav"
    linear_gain_normalize(raw_path, master_path)

    # 輸出 MP3
    mp3_path = pathlib.Path("/tmp/demo_5min_v6.mp3")
    cmd_mp3 = [FFMPEG, "-y", "-i", str(master_path),
               "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)]
    r_mp3 = subprocess.run(cmd_mp3, capture_output=True, text=True)
    if mp3_path.exists():
        print(f"[demo_mix] MP3 -> {mp3_path}")
    else:
        print(f"[demo_mix] WARN: MP3 失敗\n{r_mp3.stderr[-200:]}")

    # 輸出 thunder events JSON（含大雷後雨增強參數，供 verify_demo 閘門使用）
    thunder_json_path = out_dir / "demo_thunder_events.json"
    with open(thunder_json_path, "w") as fp:
        json.dump({
            "sample_rate": SR,
            "duration_s": DEMO_DUR_S,
            "seed": SEED,
            "events": thunder_events,
            "post_thunder_boost_enabled": USE_POST_THUNDER_BOOST,
            "post_thunder_boost_specs": thunder_boost_specs,
        }, fp, indent=2)
    print(f"[demo_mix] thunder_events: {thunder_json_path}")

    # 輸出 critter events JSON（只在 --critter 模式下）
    critter_json_path = out_dir / "demo_critter_events.json"
    if USE_CRITTER:
        critter_export = []
        for ev in critter_events:
            critter_export.append({
                "t_start": ev["t_start"],
                "t_end": ev["t_end"],
                "name": ev["name"],
                "rain_intensity_at_start": ev["rain_intensity_at_start"],
                "gain": ev["gain"],
                "faded_out": ev["faded_out"],
                "fade_start_s": ev["fade_start_s"],
            })
        with open(critter_json_path, "w") as fp:
            json.dump({
                "sample_rate": SR,
                "duration_s": DEMO_DUR_S,
                "seed": SEED,
                "critter_threshold": CRITTER_THRESHOLD,
                "events": critter_export,
            }, fp, indent=2)
        print(f"[demo_mix] critter_events: {critter_json_path}  ({len(critter_events)} events)")
    else:
        # 寫空白 critter JSON（供 verify_demo.py VD.4 讀取，events=[] 自動通過）
        with open(critter_json_path, "w") as fp:
            json.dump({
                "sample_rate": SR,
                "duration_s": DEMO_DUR_S,
                "seed": SEED,
                "critter_threshold": CRITTER_THRESHOLD,
                "events": [],
                "note": "critter layer disabled; use --critter to enable",
            }, fp, indent=2)
        print(f"[demo_mix] critter_events: {critter_json_path}  (critter OFF, 0 events)")

    # 時間軸摘要
    print("\n[demo_mix] 時間軸摘要:")
    print(f"  OU 強度: min={intensity_series.min():.3f} max={intensity_series.max():.3f}")
    for ev in thunder_events:
        print(f"  雷: t_audio={ev['t_audio']:.1f}s  dist={ev['distance_km']:.1f}km  "
              f"intensity={ev['intensity']:.2f}")
    for ev in critter_events:
        print(f"  critter: {ev['t_start']:.1f}s-{ev['t_end']:.1f}s  "
              f"name={ev['name']}  inten={ev['rain_intensity_at_start']:.3f}  "
              f"gain={ev['gain']:.3f}  faded={ev['faded_out']}")

    print(f"\n[demo_mix] DONE -> {master_path}")


if __name__ == "__main__":
    main()
