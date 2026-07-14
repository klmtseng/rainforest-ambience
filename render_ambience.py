"""
render_ambience.py — 深山雨夜音景 單一入口 CLI
重用 mix/demo_mix.py 的管線函式,不重複實作 DSP 邏輯。

用法範例:
    # 細雨 30min,無雷,無蟲鳥
    python render_ambience.py --duration 1800 --thunder-rate 0 --out /tmp/quiet_rain.wav

    # 雷雨夜 45min + mp3
    python render_ambience.py --duration 2700 --seed 42 --thunder-rate 2 --mp3 --out /tmp/rainy.wav

    # 只要 60s 帶雷測試 + 事件 JSON
    python render_ambience.py --duration 60 --seed 77 --thunder-rate 2 --events-json /tmp/ev.json --out /tmp/test.wav
"""

import os
import shutil
import argparse
import csv
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
from math import gcd
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT / "synth"))
sys.path.insert(0, str(ROOT / "mix"))

# 從 demo_mix import 所有可重用的 DSP 工具
from demo_mix import (
    WavStreamWriter,
    ou_walk,
    build_gain_envelope,
    wav_duration_from_header,
    load_asset_float32,
    schedule_critter_events_linked,
    speed_shift_chunk,
    add_clip_to_chunk,
    linear_gain_normalize,
    CRITTER_THRESHOLD,
    FADE_OUT_S,
    OU_MEAN,
    OU_THETA,
    OU_LO,
    OU_HI,
    SMOOTH_TAU_S,
    TARGET_LUFS,
    SR as DEMO_SR,
)
from rain_v4 import synthesize_rain_v4 as synthesize_rain
from thunder_v2 import build_thunder_event
from wavio import write_wav

FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
SR = 48000
CHUNK_DUR_S = 5.0      # 保持與 demo_mix 一致(OU 粒度)
PRESET_PATH = ROOT / "config/preset_realistic_v4.json"

# post-thunder boost 參數(沿用 demo_mix 定義)
POST_THUNDER_BOOST_THRESHOLD = 0.6
POST_THUNDER_BOOST_DELAY_RANGE = (15.0, 25.0)
POST_THUNDER_BOOST_RAMP_RANGE = (20.0, 30.0)
POST_THUNDER_BOOST_DB_RANGE = (2.0, 3.0)
POST_THUNDER_BOOST_DESCENT_RANGE = (60.0, 90.0)


# ---------------------------------------------------------------------------
# 指數分布雷事件排程
# ---------------------------------------------------------------------------

def schedule_thunder_events(
    duration_s: float,
    rate_per_min: float,
    distance_range_km: tuple[float, float],
    rng: np.random.Generator,
) -> list[dict]:
    """
    指數分布排程雷事件。
    rate_per_min=0 回傳空列表。
    每事件 distance_km ∈ distance_range_km 均勻分布,intensity 隨距離反向縮放。
    """
    if rate_per_min <= 0:
        return []
    lam = rate_per_min / 60.0  # 事件/秒
    events = []
    t = rng.exponential(1.0 / lam)  # 第一個事件時間
    while t < duration_s - 2.0:
        dist_km = float(rng.uniform(distance_range_km[0], distance_range_km[1]))
        # intensity 隨距離反向縮放：2km→0.9, 10km→0.4
        dist_lo, dist_hi = distance_range_km
        span = max(dist_hi - dist_lo, 0.01)
        intensity = float(np.clip(0.9 - 0.5 * (dist_km - dist_lo) / span, 0.35, 0.95))
        events.append({
            "t_audio": float(t),
            "distance_km": dist_km,
            "intensity": intensity,
        })
        gap = rng.exponential(1.0 / lam)
        t += gap
    return events


# ---------------------------------------------------------------------------
# 主渲染管線
# ---------------------------------------------------------------------------

def render(
    duration_s: float,
    seed: int,
    rain_intensity: float,
    rain_variability: float,
    thunder_rate: float,
    thunder_distance_range: tuple[float, float],
    rain_gush: bool,
    critters: bool,
    out_wav: pathlib.Path,
    mp3: bool = False,
    stems: bool = False,
    events_json: pathlib.Path | None = None,
    drop_rate_scale: float | None = None,
    drop_amp_sigma: float | None = None,
    spectral_tilt_hi: float | None = None,
    spectral_tilt_lo: float | None = None,
    bed_mix: float | None = None,
) -> None:
    """核心渲染管線。所有參數化控制從這裡進入,管線邏輯重用 demo_mix。
    drop_rate_scale: drop_rate_hz 縮放倍數(None=用 preset 值)。
    drop_amp_sigma: drop_amp_lognormal_sigma(None=用 preset 值)。
    spectral_tilt_hi: spectral_slope2_db_oct(None=用 preset 值)。
    spectral_tilt_lo: spectral_slope0_db_oct(None=用 preset 值)。
    bed_mix: bed_drop_mix[0] 床聲占比(None=用 preset 值)。
    """

    n_chunks = math.ceil(duration_s / CHUNK_DUR_S)
    total_n = int(duration_s * SR)

    out_wav.parent.mkdir(parents=True, exist_ok=True)

    print(f"[render] seed={seed}  dur={duration_s:.0f}s  "
          f"thunder_rate={thunder_rate}/min  gush={rain_gush}  critters={critters}")

    # 載入 preset
    with open(PRESET_PATH) as fp:
        cfg = json.load(fp)
    rain_preset = cfg.get("rain_v4", {})

    # RNG 分叉(與 demo_mix 相同模式,保持各子系統獨立)
    rng_master   = np.random.default_rng(seed)
    rng_ou       = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_rain     = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_thunder  = np.random.default_rng(int(rng_master.integers(0, 2**32)))
    rng_critters = np.random.default_rng(int(rng_master.integers(0, 2**32)))

    # OU 漫步(以 rain_intensity 為均值, rain_variability 為 sigma 縮放)
    ou_sigma = 0.08 * (rain_variability / 0.08) if rain_variability > 0 else 0.001
    ou_mean_clamped = float(np.clip(rain_intensity, OU_LO + 0.05, OU_HI - 0.05))
    x = np.zeros(n_chunks, dtype=np.float64)
    x[0] = ou_mean_clamped
    theta = OU_THETA
    dt = CHUNK_DUR_S
    for i in range(1, n_chunks):
        dx = theta * (ou_mean_clamped - x[i - 1]) * dt + ou_sigma * np.sqrt(dt) * rng_ou.standard_normal()
        x[i] = np.clip(x[i - 1] + dx, OU_LO, OU_HI)
    intensity_series = x
    print(f"[render] OU intensity: min={intensity_series.min():.3f}  max={intensity_series.max():.3f}")

    # 雷事件排程
    thunder_specs = schedule_thunder_events(
        duration_s, thunder_rate, thunder_distance_range, rng_thunder
    )
    print(f"[render] thunder events scheduled: {len(thunder_specs)}")

    # post-thunder boost specs
    rng_boost = np.random.default_rng(seed + 9901)
    thunder_boost_specs: list[dict] = []
    if rain_gush:
        for spec in thunder_specs:
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
        print(f"[render] post-thunder boost: {len(thunder_boost_specs)} events")
    else:
        print("[render] post-thunder boost: disabled")

    # 樣本級增益包絡
    gain_envelope = build_gain_envelope(
        intensity_series, total_n, SR,
        thunder_boost_specs=thunder_boost_specs if thunder_boost_specs else None,
    )
    print(f"[render] gain_envelope: "
          f"min={20*np.log10(max(gain_envelope.min(),1e-9)):.2f}dB  "
          f"max={20*np.log10(max(gain_envelope.max(),1e-9)):.2f}dB")

    # 素材載入
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

    # 蟲鳥排程
    if critters and asset_info:
        critter_events = schedule_critter_events_linked(
            duration_s, rng_critters, asset_info, intensity_series)
        print(f"[render] critter events: {len(critter_events)}")
    else:
        critter_events = []
        if critters and not asset_info:
            print("[render] WARN: --critters enabled but no insect/bird assets found")
        else:
            print("[render] critter layer: disabled")

    # 雷 clips 預建
    thunder_events_out = []
    thunder_clips = []
    for idx, spec in enumerate(thunder_specs):
        clip_seed = seed * 1000 + idx + 1
        clip_rng = np.random.default_rng(clip_seed)
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
        thunder_events_out.append(ev)
        thunder_clips.append((start_sample, clip))

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

    # stems writers
    rain_writer = thunder_writer = critters_writer = None
    stems_dir = out_wav.parent / (out_wav.stem + "_stems")
    if stems:
        stems_dir.mkdir(parents=True, exist_ok=True)
        rain_writer     = WavStreamWriter(stems_dir / "rain_stem.wav", SR)
        thunder_writer  = WavStreamWriter(stems_dir / "thunder_stem.wav", SR)
        critters_writer = WavStreamWriter(stems_dir / "critters_stem.wav", SR)

    # raw 輸出
    raw_path = out_wav.parent / (out_wav.stem + "_raw.wav")
    raw_writer = WavStreamWriter(raw_path, SR)

    chunk_n = int(CHUNK_DUR_S * SR)

    for ci in range(n_chunks):
        chunk_start = ci * chunk_n
        chunk_end = min(chunk_start + chunk_n, total_n)
        n_this = chunk_end - chunk_start
        chunk_start_s = chunk_start / SR

        inten = float(intensity_series[ci])

        # 雨 chunk
        chunk_rng = np.random.default_rng(int(rng_rain.integers(0, 2**32)))
        rain_preset_local = dict(rain_preset)
        rain_preset_local["intensity"] = 1.0
        rain_preset_local["v4_disable_modulation"] = True
        rain_preset_local["drop_amp_lognormal_sigma"] = 0.7
        # 四拉桿 override（None 表示沿用 preset 值）
        if drop_rate_scale is not None:
            base_rate = float(rain_preset.get("drop_rate_hz", 800.0))
            rain_preset_local["drop_rate_hz"] = base_rate * drop_rate_scale
        if drop_amp_sigma is not None:
            rain_preset_local["drop_amp_lognormal_sigma"] = drop_amp_sigma
        if spectral_tilt_hi is not None:
            rain_preset_local["spectral_slope2_db_oct"] = spectral_tilt_hi
        if spectral_tilt_lo is not None:
            rain_preset_local["spectral_slope0_db_oct"] = spectral_tilt_lo
        if bed_mix is not None:
            drop_w_local = 1.0 - float(bed_mix)
            rain_preset_local["bed_drop_mix"] = [float(bed_mix), drop_w_local]
        rain_chunk = synthesize_rain(
            duration_s=n_this / SR,
            sr=SR,
            rng=chunk_rng,
            preset=rain_preset_local,
        )
        chunk_gain = gain_envelope[chunk_start:chunk_end]
        rain_chunk = rain_chunk * chunk_gain

        # 雷 chunk
        thunder_chunk = np.zeros(n_this, dtype=np.float64)
        for (clip_start_sample, clip) in thunder_clips:
            add_clip_to_chunk(thunder_chunk, clip, clip_start_sample, chunk_start)
        peak_t = np.max(np.abs(thunder_chunk))
        if peak_t > 0.95:
            thunder_chunk *= 0.95 / peak_t

        # 蟲鳥 chunk
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

        # master mix
        master_chunk = rain_chunk * 0.75 + thunder_chunk * 2.5 + critters_chunk * 0.55
        peak_m = np.max(np.abs(master_chunk))
        if peak_m > 0.98:
            master_chunk *= 0.98 / peak_m

        raw_writer.write_chunk(master_chunk)

        if stems:
            rain_writer.write_chunk(rain_chunk)
            thunder_writer.write_chunk(thunder_chunk)
            critters_writer.write_chunk(critters_chunk)

        if ci % 12 == 0:
            mid_gain_db = 20.0 * np.log10(max(chunk_gain[len(chunk_gain)//2], 1e-9))
            print(f"  chunk {ci+1}/{n_chunks}  t={chunk_start_s:.0f}s  "
                  f"intensity={inten:.3f}  gain_mid={mid_gain_db:.2f}dB")

    raw_writer.close()
    if stems:
        for w in (rain_writer, thunder_writer, critters_writer):
            w.close()

    print(f"[render] raw written ({raw_path}), normalizing...")
    linear_gain_normalize(raw_path, out_wav)
    raw_path.unlink(missing_ok=True)

    # stems も正規化
    if stems:
        for stem_name in ("rain_stem", "thunder_stem", "critters_stem"):
            sp = stems_dir / f"{stem_name}.wav"
            if sp.exists():
                print(f"[render] stems: {sp}")

    # MP3
    if mp3:
        mp3_path = out_wav.with_suffix(".mp3")
        cmd_mp3 = [FFMPEG, "-y", "-i", str(out_wav),
                   "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3_path)]
        r = subprocess.run(cmd_mp3, capture_output=True, text=True)
        if mp3_path.exists():
            print(f"[render] MP3 -> {mp3_path}")
        else:
            print(f"[render] WARN: MP3 failed\n{r.stderr[-200:]}")

    # events JSON
    if events_json is not None:
        events_json = pathlib.Path(events_json)
        events_json.parent.mkdir(parents=True, exist_ok=True)
        with open(events_json, "w") as fp:
            json.dump({
                "sample_rate": SR,
                "duration_s": duration_s,
                "seed": seed,
                "events": thunder_events_out,
                "post_thunder_boost_enabled": rain_gush,
                "post_thunder_boost_specs": thunder_boost_specs,
            }, fp, indent=2)
        print(f"[render] events_json -> {events_json}")

    n_t = len(thunder_events_out)
    n_c = len(critter_events)
    print(f"\n[render] DONE -> {out_wav}")
    print(f"  thunder={n_t}  critters={n_c}  gush={rain_gush}  mp3={mp3}  stems={stems}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="深山雨夜環境音景生成器（參數化 CLI）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--duration", type=float, default=300.0,
        help="輸出時長（秒）；1800=30min, 2700=45min",
    )
    parser.add_argument(
        "--seed", type=int, default=77,
        help="隨機種子；相同 seed + 相同參數保證 bit-identical",
    )
    parser.add_argument(
        "--rain-intensity", type=float, default=0.55,
        help="OU 漫步基準強度 [0~1]；影響整體雨量/響度",
    )
    parser.add_argument(
        "--rain-variability", type=float, default=0.08,
        help="OU 漫步振幅（sigma）；0=幾乎無變化, 0.15=劇烈大小雨交替",
    )
    parser.add_argument(
        "--thunder-rate", type=float, default=2.0,
        help="每分鐘期望雷聲次數；0=無雷；事件排程用指數分布",
    )
    parser.add_argument(
        "--thunder-distance-range", type=float, nargs=2,
        default=[2.0, 8.0], metavar=("KM_MIN", "KM_MAX"),
        help="雷聲距離範圍（公里）；越近=越響, 低頻能量越強",
    )
    parser.add_argument(
        "--rain-gush", action=argparse.BooleanOptionalAction, default=True,
        help="大雷後雨量短暫增強（--rain-gush / --no-rain-gush）",
    )
    parser.add_argument(
        "--critters", action=argparse.BooleanOptionalAction, default=False,
        help="啟用蟲鳥層，雨強度低時自動出現（--critters / --no-critters）",
    )
    parser.add_argument(
        "--out", type=pathlib.Path, default=pathlib.Path("/tmp/ambience.wav"),
        help="輸出 WAV 路徑（float32, 48kHz）",
    )
    parser.add_argument(
        "--mp3", action="store_true", default=False,
        help="同時轉換 MP3（192kbps），輸出到 --out 同目錄",
    )
    parser.add_argument(
        "--stems", action="store_true", default=False,
        help="輸出分軌（rain/thunder/critters 各一個 WAV），存到 <out_stem>_stems/",
    )
    parser.add_argument(
        "--events-json", type=pathlib.Path, default=None,
        help="輸出雷聲事件表 JSON（含時刻、距離、強度、post-thunder boost 參數）",
    )
    parser.add_argument(
        "--drop-rate-scale", type=float, default=None,
        help="drop_rate_hz 縮放倍數（None=用 preset 值 800Hz）",
    )
    parser.add_argument(
        "--drop-amp-sigma", type=float, default=None,
        help="drop_amp_lognormal_sigma，越大音量差異越大（None=用 preset 值）",
    )
    parser.add_argument(
        "--spectral-tilt-hi", type=float, default=None,
        help="spectral_slope2_db_oct，高頻斜率 dB/oct（None=用 preset 值）",
    )
    parser.add_argument(
        "--spectral-tilt-lo", type=float, default=None,
        help="spectral_slope0_db_oct，低頻斜率 dB/oct（None=用 preset 值）",
    )
    parser.add_argument(
        "--bed-mix", type=float, default=None,
        help="床聲占比 [0~1]（None=用 preset 值）",
    )

    args = parser.parse_args()

    render(
        duration_s=args.duration,
        seed=args.seed,
        rain_intensity=args.rain_intensity,
        rain_variability=args.rain_variability,
        thunder_rate=args.thunder_rate,
        thunder_distance_range=tuple(args.thunder_distance_range),
        rain_gush=args.rain_gush,
        critters=args.critters,
        out_wav=args.out,
        mp3=args.mp3,
        stems=args.stems,
        events_json=args.events_json,
        drop_rate_scale=args.drop_rate_scale,
        drop_amp_sigma=args.drop_amp_sigma,
        spectral_tilt_hi=args.spectral_tilt_hi,
        spectral_tilt_lo=args.spectral_tilt_lo,
        bed_mix=args.bed_mix,
    )


if __name__ == "__main__":
    main()
