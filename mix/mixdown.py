"""
mixdown.py — 45 分鐘長檔製作
輸出:
  out/long/rain_stem.wav         (2700s, 48kHz 24-bit)
  out/long/thunder_stem.wav      (2700s, 48kHz 24-bit)
  out/long/critters_stem.wav     (2700s, 48kHz 24-bit)
  out/long/master_raw.wav        (2700s, 48kHz 24-bit)
  out/long/master.wav            (2700s, 48kHz 24-bit) ← loudnorm -23 LUFS
  out/long/thunder_events.json

記憶體控制方案:
  1. Assets 存為 float32（非 float64），降低一半記憶體（全部 ~340MB）
  2. 速度偏移在 chunk 層做（numpy interp 切片），不預建全 clip
  3. 事件排程時只讀 WAV header（wave 模組），不載資料
  4. 雷聲 clip 預建（≤30 條 × 8s×SR ≈ 90MB，可接受）

用法:
    python mix/mixdown.py [--seed SEED] [--preset PATH] [--dur DUR] [--out-dir PATH]
"""

import os
import shutil
import argparse
import json
import pathlib
import struct
import sys
import subprocess
import wave as wave_mod
import numpy as np
from scipy.io import wavfile
from math import gcd

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from rain import synthesize_rain
from thunder import build_thunder_event

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"

SR = 48000
TARGET_DUR_S = 2700.0
CHUNK_DUR_S = 60.0


# ---------------------------------------------------------------------------
# WAV 串流寫入輔助
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
        self.f.write(struct.pack("<H", 3))  # 3 = IEEE_FLOAT（float32 WAV）
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

def ou_walk(n_steps: int, dt_s: float, rng: np.random.Generator,
            mean: float = 0.6, theta: float = None, sigma: float = 0.05,
            lo: float = 0.3, hi: float = 0.9) -> np.ndarray:
    if theta is None:
        theta = 1.0 / 180.0
    x = np.zeros(n_steps, dtype=np.float64)
    x[0] = mean
    for i in range(1, n_steps):
        dx = theta * (mean - x[i - 1]) * dt_s + sigma * np.sqrt(dt_s) * rng.standard_normal()
        x[i] = np.clip(x[i - 1] + dx, lo, hi)
    return x


# ---------------------------------------------------------------------------
# 雷事件排程
# ---------------------------------------------------------------------------

def schedule_thunder_events(duration_s: float, rng: np.random.Generator,
                             mean_interval_s: float = 150.0,
                             refractory_s: float = 30.0,
                             distance_km_range=(0.3, 20.0)):
    events = []
    t = rng.exponential(mean_interval_s)
    while t < duration_s:
        if events and (t - events[-1]["t_flash"]) < refractory_s:
            t += rng.exponential(mean_interval_s)
            continue
        d_km = rng.uniform(*distance_km_range)
        t_audio = t + d_km / 0.343
        if t_audio >= duration_s:
            t += rng.exponential(mean_interval_s)
            continue
        intensity = rng.uniform(0.4, 1.0)
        events.append({
            "t_flash": float(t),
            "t_audio": float(t_audio),
            "distance_km": float(d_km),
            "intensity": float(intensity),
            "start_sample": int(t_audio * SR),
        })
        t += rng.exponential(mean_interval_s)
    return events


# ---------------------------------------------------------------------------
# 素材管理（header-only duration；全量 float32 載入）
# ---------------------------------------------------------------------------

def wav_duration_from_header(fpath: pathlib.Path) -> float:
    """用 wave 模組讀 WAV header，不載資料。"""
    try:
        with wave_mod.open(str(fpath), 'rb') as wf:
            return wf.getnframes() / wf.getframerate()
    except Exception:
        # fallback: load partially
        sr_a, data_a = wavfile.read(str(fpath))
        return len(data_a) / sr_a


def load_asset_float32(fpath: pathlib.Path) -> tuple[int, np.ndarray]:
    """載入 WAV 為 float32 mono，回傳 (sr, data_f32)。"""
    sr_a, data_a = wavfile.read(str(fpath))
    if data_a.ndim == 2:
        data_f = data_a.mean(axis=1)
    else:
        data_f = data_a.copy()
    # 轉 float32
    if data_a.dtype == np.int16:
        data_f32 = data_f.astype(np.float32) / 32768.0
    elif data_a.dtype == np.int32:
        data_f32 = data_f.astype(np.float32) / (2 ** 23)
    else:
        data_f32 = data_f.astype(np.float32)
    return sr_a, data_f32


# ---------------------------------------------------------------------------
# 蟲鳥事件排程（header-only duration）
# ---------------------------------------------------------------------------

def schedule_critter_events(duration_s: float, rng: np.random.Generator,
                             asset_info: dict):
    """
    asset_info: {fname: {"duration_s": float, "path": Path}}
    同素材重用間隔 ≥ 5min。
    """
    asset_names = list(asset_info.keys())
    events = []
    last_use = {name: -300.0 for name in asset_names}

    t = rng.uniform(5.0, 30.0)
    while t < duration_s:
        available = [n for n in asset_names if (t - last_use[n]) >= 300.0]
        if not available:
            t += rng.uniform(10.0, 30.0)
            continue

        name = available[rng.integers(0, len(available))]
        raw_dur = asset_info[name]["duration_s"]
        speed = rng.uniform(0.97, 1.03)
        effective_dur = raw_dur / speed

        if t + effective_dur > duration_s:
            t += rng.uniform(10.0, 60.0)
            continue

        gain = rng.uniform(0.2, 0.7)
        pan = rng.uniform(-0.6, 0.6)  # mono 輸出不用 pan，記錄供未來用

        start_sample = int(t * SR)
        # 保守估計 end_sample（+1s margin）
        end_sample = start_sample + int(effective_dur * SR) + SR

        events.append({
            "t_start": float(t),
            "name": name,
            "speed": float(speed),
            "gain": float(gain),
            "pan": float(pan),
            "effective_dur_s": float(effective_dur),
            "start_sample": start_sample,
            "end_sample": end_sample,
        })
        last_use[name] = t
        t += rng.uniform(15.0, 120.0)

    return events


# ---------------------------------------------------------------------------
# chunk 層速度偏移（numpy interp，不需全 clip）
# ---------------------------------------------------------------------------

def speed_shift_chunk(raw_f32: np.ndarray, raw_sr: int,
                      speed: float, gain: float,
                      clip_start_raw_sample: int,  # raw 中此 event 對應的起始 sample
                      chunk_start_s: float, chunk_dur_s: float,
                      global_start_s: float) -> tuple[np.ndarray, int]:
    """
    只取 raw 中與當前 chunk 有交集的段落，做線性插值速度偏移。
    global_start_s: 此 critter event 在全局 timeline 的起始秒數

    回傳 (chunk_contribution, dst_offset_samples)
      chunk_contribution: 本次貢獻的音訊 float64，長度為交集長度
      dst_offset_samples: 在 chunk buffer 中的偏移（samples）
    """
    chunk_n = int(chunk_dur_s * SR)
    chunk_start_global = int(chunk_start_s * SR)

    # event 在全局 timeline 的起始/結束 sample（以 SR 為基礎）
    ev_start_global = int(global_start_s * SR)
    # event 結束取決於 raw 長度和 speed
    ev_dur_raw_n = len(raw_f32) - clip_start_raw_sample  # raw 剩餘樣本數
    ev_dur_in_output = ev_dur_raw_n / speed               # 輸出持續樣本數
    ev_end_global = ev_start_global + int(ev_dur_in_output) + 1

    # 交集
    chunk_end_global = chunk_start_global + chunk_n
    overlap_start = max(ev_start_global, chunk_start_global)
    overlap_end = min(ev_end_global, chunk_end_global)

    if overlap_start >= overlap_end:
        return None, 0

    # 在輸出空間，overlap 的樣本數
    n_out = overlap_end - overlap_start

    # 這段輸出對應 raw 中的哪個範圍
    # output_sample i (相對 ev_start_global) → raw_sample i * speed
    out_offset_from_ev = overlap_start - ev_start_global
    raw_start_float = out_offset_from_ev * speed + clip_start_raw_sample
    raw_end_float = (out_offset_from_ev + n_out) * speed + clip_start_raw_sample

    raw_start_idx = int(raw_start_float)
    raw_end_idx = int(raw_end_float) + 2  # +2 margin for interp

    raw_start_idx = max(0, min(raw_start_idx, len(raw_f32) - 1))
    raw_end_idx = max(0, min(raw_end_idx, len(raw_f32)))

    if raw_start_idx >= raw_end_idx:
        return None, 0

    raw_slice = raw_f32[raw_start_idx:raw_end_idx].astype(np.float64)

    # 插值到 n_out 個樣本
    raw_x = np.arange(len(raw_slice), dtype=np.float64) + raw_start_idx
    out_x = np.linspace(raw_start_float, raw_end_float - 1, n_out, dtype=np.float64)
    out_x = np.clip(out_x, raw_start_idx, raw_start_idx + len(raw_slice) - 1)

    resampled = np.interp(out_x, raw_x, raw_slice)
    contribution = (resampled * gain).astype(np.float64)

    # dst_offset in chunk buffer
    dst_off = overlap_start - chunk_start_global

    return contribution, dst_off


# ---------------------------------------------------------------------------
# chunk buffer 加入 clip（預建 thunder clips 用）
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
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="45min 長檔製作")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dur", type=float, default=TARGET_DUR_S)
    parser.add_argument("--preset", type=str,
                        default=str(ROOT / "config/preset_default.json"))
    parser.add_argument("--out-dir", type=str,
                        default=str(ROOT / "out/long"))
    args = parser.parse_args()

    dur = args.dur
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preset_path = pathlib.Path(args.preset)
    preset = json.load(open(preset_path)) if preset_path.exists() else {}
    rain_cfg = preset.get("rain", {})

    print(f"[mixdown] seed={args.seed}  dur={dur}s  chunk={CHUNK_DUR_S}s")

    rng_master = np.random.default_rng(args.seed)
    rng_rain    = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_thunder = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_critters= np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_ou      = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_tsyn    = np.random.default_rng(int(rng_master.integers(0, 2**32)))

    n_chunks = int(np.ceil(dur / CHUNK_DUR_S))

    # OU 漫步
    intensity_series = ou_walk(n_chunks, CHUNK_DUR_S, rng_ou)
    print(f"[mixdown] OU intensity: min={intensity_series.min():.3f} max={intensity_series.max():.3f}")

    # 雷事件排程
    thunder_events = schedule_thunder_events(
        dur, rng_thunder, mean_interval_s=150.0,
        refractory_s=30.0, distance_km_range=(0.3, 20.0))
    print(f"[mixdown] thunder events: {len(thunder_events)}")

    # 素材 info（header-only，不載資料）
    curated_dir = ROOT / "assets/curated"
    licenses_csv = curated_dir / "licenses.csv"
    import csv
    allowed_fnames = set()
    with open(licenses_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("授權", "").strip().upper() == "CC0":
                allowed_fnames.add(row["檔名"].strip())

    # Build category map from licenses.csv (only insect/bird go into critters stem)
    category_map: dict = {}
    with open(licenses_csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname_c = row.get("檔名", "").strip()
            cat = row.get("category", "").strip()
            if fname_c:
                category_map[fname_c] = cat

    asset_info = {}
    for fname in allowed_fnames:
        fpath = curated_dir / fname
        if fpath.exists() and category_map.get(fname, "") in ("insect", "bird"):
            asset_info[fname] = {
                "path": fpath,
                "duration_s": wav_duration_from_header(fpath),
            }
    print(f"[mixdown] assets: {len(asset_info)} files (insect/bird only, header-only)")

    # 蟲鳥事件排程
    critter_events = schedule_critter_events(dur, rng_critters, asset_info)
    print(f"[mixdown] critter events: {len(critter_events)}")

    # 預建雷聲 clips（小量）
    thunder_clips = []
    for ev in thunder_events:
        clip_rng = np.random.default_rng(int(rng_tsyn.integers(0, 2**32)))
        clip = build_thunder_event(ev["distance_km"], ev["intensity"], SR, clip_rng)
        thunder_clips.append((ev["start_sample"], clip))
    print(f"[mixdown] thunder clips prebuilt: {len(thunder_clips)}")

    # 素材載入 cache（float32，懶惰載入）
    asset_cache = {}   # fname -> float32 array

    def get_asset(fname: str) -> np.ndarray:
        if fname not in asset_cache:
            fpath = asset_info[fname]["path"]
            src_sr, data_f32 = load_asset_float32(fpath)
            if src_sr != SR:
                # 重取樣
                g = gcd(SR, src_sr)
                from scipy.signal import resample_poly
                data_f64 = data_f32.astype(np.float64)
                data_f32 = resample_poly(data_f64, SR // g, src_sr // g).astype(np.float32)
            asset_cache[fname] = data_f32
        return asset_cache[fname]

    # 開啟 stem writers
    rain_writer    = WavStreamWriter(out_dir / "rain_stem.wav", SR)
    thunder_writer = WavStreamWriter(out_dir / "thunder_stem.wav", SR)
    critters_writer= WavStreamWriter(out_dir / "critters_stem.wav", SR)
    master_raw_wr  = WavStreamWriter(out_dir / "master_raw.wav", SR)

    chunk_n = int(CHUNK_DUR_S * SR)
    total_n = int(dur * SR)

    for ci in range(n_chunks):
        chunk_start = ci * chunk_n
        chunk_end = min(chunk_start + chunk_n, total_n)
        n_this = chunk_end - chunk_start
        chunk_start_s = chunk_start / SR

        # 雨 chunk
        chunk_rng = np.random.default_rng(int(rng_rain.integers(0, 2**32)))
        rain_chunk = synthesize_rain(
            duration_s=n_this / SR,
            sr=SR,
            rng=chunk_rng,
            intensity=float(intensity_series[ci]),
            band_freqs_hz=rain_cfg.get("band_freqs_hz"),
            band_widths_hz=rain_cfg.get("band_widths_hz"),
            drop_rate_hz=float(rain_cfg.get("drop_rate_hz", 80.0)),
            drop_template_count=int(rain_cfg.get("drop_template_count", 40)),
        )

        # 雷 chunk
        thunder_chunk = np.zeros(n_this, dtype=np.float64)
        for (clip_start_s_abs, clip) in thunder_clips:
            add_clip_to_chunk(thunder_chunk, clip, clip_start_s_abs, chunk_start)
        peak_t = np.max(np.abs(thunder_chunk)) if n_this > 0 else 0.0
        if peak_t > 0.95:
            thunder_chunk *= (0.95 / peak_t)

        # 蟲鳥 chunk（chunk-level speed shift，不預建全 clip）
        critters_chunk = np.zeros(n_this, dtype=np.float64)
        for ev in critter_events:
            # 快速跳過無交集
            if ev["end_sample"] < chunk_start or ev["start_sample"] >= chunk_end:
                continue
            raw = get_asset(ev["name"])
            contribution, dst_off = speed_shift_chunk(
                raw_f32=raw,
                raw_sr=SR,
                speed=ev["speed"],
                gain=ev["gain"],
                clip_start_raw_sample=0,
                chunk_start_s=chunk_start_s,
                chunk_dur_s=CHUNK_DUR_S,
                global_start_s=ev["t_start"],
            )
            if contribution is not None and dst_off + len(contribution) <= n_this:
                critters_chunk[dst_off:dst_off + len(contribution)] += contribution
            elif contribution is not None:
                n_fit = n_this - dst_off
                if n_fit > 0:
                    critters_chunk[dst_off:dst_off + n_fit] += contribution[:n_fit]

        peak_c = np.max(np.abs(critters_chunk)) if n_this > 0 else 0.0
        if peak_c > 0.95:
            critters_chunk *= (0.95 / peak_c)

        # master mix
        master_chunk = (rain_chunk * 0.707 +
                        thunder_chunk * 1.0 +
                        critters_chunk * 0.5)
        peak_m = np.max(np.abs(master_chunk)) if n_this > 0 else 0.0
        if peak_m > 0.98:
            master_chunk *= (0.98 / peak_m)

        rain_writer.write_chunk(rain_chunk)
        thunder_writer.write_chunk(thunder_chunk)
        critters_writer.write_chunk(critters_chunk)
        master_raw_wr.write_chunk(master_chunk)

        if ci % 10 == 0:
            n_cached = len(asset_cache)
            print(f"  chunk {ci+1}/{n_chunks}  t={chunk_start_s:.0f}s  "
                  f"intensity={intensity_series[ci]:.3f}  cached_assets={n_cached}")

    rain_writer.close()
    thunder_writer.close()
    critters_writer.close()
    master_raw_wr.close()

    print("[mixdown] stems written, running loudnorm...")

    raw_path = out_dir / "master_raw.wav"
    master_path = out_dir / "master.wav"

    cmd1 = [FFMPEG, "-y", "-i", str(raw_path),
            "-af", "loudnorm=I=-23:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-"]
    r1 = subprocess.run(cmd1, capture_output=True, text=True)
    if r1.returncode != 0:
        print(f"[loudnorm pass1 FAILED]\n{r1.stderr[-500:]}")
        sys.exit(1)

    stderr_text = r1.stderr
    json_start = stderr_text.rfind("{")
    json_end = stderr_text.rfind("}") + 1
    if json_start == -1 or json_end == 0:
        print("[loudnorm] Could not parse pass1 JSON")
        print(stderr_text[-1000:])
        sys.exit(1)
    ln_data = json.loads(stderr_text[json_start:json_end])
    print(f"[loudnorm pass1] {ln_data}")

    measured_I      = ln_data.get("input_i", "-23")
    measured_LRA    = ln_data.get("input_lra", "7")
    measured_TP     = ln_data.get("input_tp", "-2")
    measured_thresh = ln_data.get("input_thresh", "-33")
    offset          = ln_data.get("target_offset", "0")

    af2 = (f"loudnorm=I=-23:TP=-1.5:LRA=11:"
           f"measured_I={measured_I}:measured_LRA={measured_LRA}:"
           f"measured_TP={measured_TP}:measured_thresh={measured_thresh}:"
           f"offset={offset}:linear=true:print_format=json")
    cmd2 = [FFMPEG, "-y", "-i", str(raw_path),
            "-af", af2, "-ar", str(SR), "-c:a", "pcm_s24le",
            str(master_path)]
    r2 = subprocess.run(cmd2, capture_output=True, text=True)
    if r2.returncode != 0:
        print(f"[loudnorm pass2 FAILED]\n{r2.stderr[-500:]}")
        sys.exit(1)
    print(f"[loudnorm pass2] done -> {master_path}")

    events_path = out_dir / "thunder_events.json"
    with open(events_path, "w") as f:
        json.dump({"sample_rate": SR, "duration_s": dur,
                   "seed": args.seed, "events": thunder_events}, f, indent=2)
    print(f"[mixdown] thunder_events.json: {len(thunder_events)} events")

    print(f"\n[mixdown] DONE  → {out_dir}")


if __name__ == "__main__":
    main()
