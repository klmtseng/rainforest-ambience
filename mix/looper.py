"""
looper.py — 90s 無縫雨聲 loop 製作
技巧:
  1. 多渲 5s tail，做 equal-power crossfade 貼回起點
  2. LFO 週期鎖 loop_duration 整數分之一，保證首尾相位吻合

輸出: out/loops/rain_loop_90s.wav  (48kHz, 24-bit)

用法:
    python mix/looper.py [--seed SEED] [--preset PATH] [--out-dir PATH]
"""

import argparse
import json
import pathlib
import sys
import numpy as np
from scipy.io import wavfile

# 允許從專案根目錄匯入 synth/
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from rain import synthesize_rain
from wavio import write_wav


SR = 48000
LOOP_DUR_S = 90.0          # 目標 loop 長度
TAIL_DUR_S = 5.0           # 多渲的 crossfade tail
XFADE_DUR_S = 5.0          # equal-power crossfade 長度
TOTAL_DUR_S = LOOP_DUR_S + TAIL_DUR_S   # 95s

# LFO: rain.py 床層無內建 LFO，此處在 mix 層加一個緩慢音量 LFO
# 週期必須是 loop_duration 的整數分之一，以保證首尾相位相同
# 選 LOOP_DUR_S / N 其中 N 為正整數
# 這裡取 N=3 → 週期 30s，LFO 在 90s 剛好走完整 3 個週期
LFO_PERIOD_S = LOOP_DUR_S / 3.0   # 30s
LFO_DEPTH = 0.08                    # ±8% 音量調變，足夠但不突兀


def make_lfo(n_samples: int, sr: int, period_s: float, depth: float,
             phase_offset: float = 0.0) -> np.ndarray:
    """
    返回 shape (n_samples,) 的 LFO 包絡，值域 [1-depth, 1+depth]。
    phase_offset: 起始相位（弧度），用於對齊首尾。
    """
    t = np.arange(n_samples) / sr
    lfo = 1.0 + depth * np.sin(2 * np.pi * t / period_s + phase_offset)
    return lfo.astype(np.float64)


def make_rain_loop(seed: int, preset: dict) -> np.ndarray:
    """
    合成 TOTAL_DUR_S 雨聲，應用 LFO，然後用 equal-power crossfade
    把 tail 疊回起點，返回 LOOP_DUR_S 長度的 loop。
    """
    rain_cfg = preset.get("rain", {})

    # 渲染 95s 雨聲（含 5s tail）
    rng = np.random.default_rng(seed)
    rain_raw = synthesize_rain(
        duration_s=TOTAL_DUR_S,
        sr=SR,
        rng=rng,
        intensity=rain_cfg.get("intensity", 0.6),
        band_freqs_hz=rain_cfg.get("band_freqs_hz"),
        band_widths_hz=rain_cfg.get("band_widths_hz"),
        drop_rate_hz=float(rain_cfg.get("drop_rate_hz", 80.0)),
        drop_template_count=int(rain_cfg.get("drop_template_count", 40)),
    )

    n_total = len(rain_raw)  # 95s * SR
    n_loop = int(LOOP_DUR_S * SR)   # 90s * SR
    n_xfade = int(XFADE_DUR_S * SR)  # 5s * SR

    # 應用 LFO（週期鎖在 loop 長的整數分之一）
    # 相位 = 0 → t=0 與 t=LOOP_DUR_S 的 sin 值相同（週期整數倍）
    lfo_full = make_lfo(n_total, SR, LFO_PERIOD_S, LFO_DEPTH, phase_offset=0.0)
    rain_lfo = rain_raw * lfo_full

    # 取 loop 段和 tail 段
    loop_body = rain_lfo[:n_loop].copy()      # [0, 90s)
    tail = rain_lfo[n_loop:n_loop + n_xfade]  # [90s, 95s)

    # equal-power crossfade curve
    # fade_in: 用於把 tail 疊加到 loop 起點
    # fade_out: 把 loop 起點原來的內容淡出
    t_xf = np.linspace(0.0, 1.0, n_xfade, dtype=np.float64)
    fade_in = np.sqrt(t_xf)         # tail 淡入
    fade_out = np.sqrt(1.0 - t_xf)  # loop 起點淡出

    # 把 tail 做 equal-power 疊加到 loop_body 的起始 n_xfade 個樣本
    loop_body[:n_xfade] = loop_body[:n_xfade] * fade_out + tail * fade_in

    # 最終 clip 保護（-1dBFS = 0.891）
    peak = np.max(np.abs(loop_body))
    if peak > 0.891:
        loop_body = loop_body * (0.891 / peak)

    return loop_body.astype(np.float64)


def main():
    parser = argparse.ArgumentParser(description="90s 無縫雨聲 loop 製作")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", type=str,
                        default=str(ROOT / "config/preset_default.json"))
    parser.add_argument("--out-dir", type=str,
                        default=str(ROOT / "out/loops"))
    args = parser.parse_args()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    preset_path = pathlib.Path(args.preset)
    if preset_path.exists():
        with open(preset_path) as f:
            preset = json.load(f)
    else:
        preset = {}

    print(f"[looper] seed={args.seed}  loop={LOOP_DUR_S}s  xfade={XFADE_DUR_S}s  LFO_period={LFO_PERIOD_S}s")

    loop_audio = make_rain_loop(args.seed, preset)
    out_path = out_dir / "rain_loop_90s.wav"
    write_wav(out_path, SR, loop_audio)

    dur = len(loop_audio) / SR
    rms = np.sqrt(np.mean(loop_audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-12)
    peak = np.max(np.abs(loop_audio))
    peak_db = 20 * np.log10(peak + 1e-12)
    print(f"[looper] -> {out_path}")
    print(f"[looper]    dur={dur:.3f}s  RMS={rms_db:.2f}dBFS  peak={peak_db:.2f}dBFS")


if __name__ == "__main__":
    main()
