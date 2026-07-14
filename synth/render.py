"""
render.py — CLI 渲染入口
用法:
    python render.py --dur 60 --seed 1 [--preset config/preset_default.json]
                     [--out-dir out/stems]

輸出:
    rain_{seed}.wav
    thunder_{seed}.wav
    thunder_events_{seed}.json
"""

import argparse
import json
import pathlib
import time
import sys

import numpy as np
from scipy.io import wavfile

# 同目錄 import
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from rain import synthesize_rain
from thunder import synthesize_thunder_session
from wavio import write_wav


def main():
    parser = argparse.ArgumentParser(description="Rainforest ambience renderer")
    parser.add_argument("--dur", type=float, default=60.0, help="Duration in seconds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--preset", type=str,
                        default=str(pathlib.Path(__file__).parent.parent /
                                    "config/preset_default.json"),
                        help="Path to preset JSON")
    parser.add_argument("--out-dir", type=str,
                        default=str(pathlib.Path(__file__).parent.parent /
                                    "out/stems"),
                        help="Output directory")
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 載入 preset
    preset_path = pathlib.Path(args.preset)
    if preset_path.exists():
        with open(preset_path) as f:
            preset = json.load(f)
    else:
        preset = {}

    rain_cfg = preset.get("rain", {})
    thunder_cfg = preset.get("thunder", {})
    SR = int(preset.get("master", {}).get("sample_rate", 48000))

    t0 = time.time()

    # ---- 雨聲 ----
    rain_rng = np.random.default_rng(args.seed)
    rain_audio = synthesize_rain(
        duration_s=args.dur,
        sr=SR,
        rng=rain_rng,
        intensity=rain_cfg.get("intensity", 0.6),
        band_freqs_hz=rain_cfg.get("band_freqs_hz"),
        band_widths_hz=rain_cfg.get("band_widths_hz"),
        drop_rate_hz=float(rain_cfg.get("drop_rate_hz", 80.0)),
        drop_template_count=int(rain_cfg.get("drop_template_count", 40)),
    )
    rain_path = out_dir / f"rain_{args.seed}.wav"
    write_wav(rain_path, SR, rain_audio)

    # ---- 雷聲 ----
    thunder_rng = np.random.default_rng(args.seed + 1000)
    thunder_audio, thunder_events = synthesize_thunder_session(
        duration_s=args.dur,
        sr=SR,
        rng=thunder_rng,
        event_rate_per_hour=float(thunder_cfg.get("event_rate_per_hour", 6.0)),
        refractory_s=float(thunder_cfg.get("refractory_s", 30.0)),
        distance_km_range=tuple(thunder_cfg.get("distance_km_range", [1.0, 15.0])),
        reverb_tail_s=float(thunder_cfg.get("reverb_tail_s", 3.0)),
    )
    thunder_path = out_dir / f"thunder_{args.seed}.wav"
    write_wav(thunder_path, SR, thunder_audio)

    events_path = out_dir / f"thunder_events_{args.seed}.json"
    with open(events_path, "w") as f:
        json.dump({
            "sample_rate": SR,
            "duration_s": args.dur,
            "seed": args.seed,
            "events": thunder_events,
        }, f, indent=2)

    elapsed = time.time() - t0

    n_rain = len(rain_audio)
    n_thunder = len(thunder_audio)
    print(f"[render] dur={args.dur}s seed={args.seed} SR={SR}")
    print(f"  rain    -> {rain_path}  samples={n_rain}")
    print(f"  thunder -> {thunder_path}  samples={n_thunder}  events={len(thunder_events)}")
    print(f"  events  -> {events_path}")
    print(f"  elapsed: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
