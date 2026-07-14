"""
looper_real_rain.py — 90s 無縫真實雨聲 loop 製作

使用 CC0 真實雨聲錄音（amb_rain2.wav，GoPro 野外錄音），
從中段取素材，做 equal-power crossfade 首尾接縫。

不需要 LFO 相位對齊（真實錄音非程序合成），
改用 xfade crossfade 做自然接縫。

輸出: out/loops/rain_loop_90s_real.wav  (48kHz, 24-bit)

用法:
    python mix/looper_real_rain.py [--source PATH] [--out-dir PATH]
"""

import argparse
import pathlib
import sys
import numpy as np
from scipy.io import wavfile

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))
from wavio import write_wav

SR = 48000
LOOP_DUR_S = 90.0          # 目標 loop 長度
XFADE_DUR_S = 5.0          # equal-power crossfade 長度（首尾接縫）
# 從素材中段取，跳過頭尾各 10s（避免錄音起頭/結尾的切換雜音）
SKIP_START_S = 10.0
SKIP_END_S = 10.0


def load_wav(path: pathlib.Path) -> tuple:
    """Load WAV, normalise to float64 [-1, 1], downmix to mono.

    ffmpeg pcm_s32le 轉出的 WAV 使用全 32-bit 範圍（2^31）。
    """
    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float64) / 2147483648.0  # ffmpeg pcm_s32le 全範圍
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x.astype(np.float64)


def make_real_rain_loop(source_wav: pathlib.Path) -> np.ndarray:
    """
    從真實雨聲錄音中取 90s + xfade tail，做 equal-power crossfade，
    返回 90s loop。
    """
    src_sr, src = load_wav(source_wav)

    # 重採樣到 48kHz（若來源不是）
    if src_sr != SR:
        raise ValueError(f"Source SR={src_sr} != {SR}. Pre-convert to 48kHz first.")

    n_skip_start = int(SKIP_START_S * SR)
    n_skip_end = int(SKIP_END_S * SR)
    usable = src[n_skip_start: len(src) - n_skip_end]

    n_loop = int(LOOP_DUR_S * SR)          # 90s
    n_xfade = int(XFADE_DUR_S * SR)        # 5s
    n_needed = n_loop + n_xfade            # 95s

    if len(usable) < n_needed:
        raise ValueError(
            f"Source too short: usable={len(usable)/SR:.1f}s, need={n_needed/SR:.1f}s"
        )

    # 從中段取素材：若素材夠長，從中間偏左取，確保首尾段落差異大（自然接縫）
    mid = len(usable) // 2
    start = max(0, mid - n_needed // 2)
    segment = usable[start: start + n_needed].copy()

    loop_body = segment[:n_loop].copy()     # [0, 90s)
    tail = segment[n_loop: n_loop + n_xfade]  # [90s, 95s)

    # equal-power crossfade curve
    t_xf = np.linspace(0.0, 1.0, n_xfade, dtype=np.float64)
    fade_in = np.sqrt(t_xf)          # tail 淡入（貼到 loop 起點）
    fade_out = np.sqrt(1.0 - t_xf)   # loop 起點原內容淡出

    loop_body[:n_xfade] = loop_body[:n_xfade] * fade_out + tail * fade_in

    # clip 保護（-1dBFS = 0.891）
    peak = np.max(np.abs(loop_body))
    if peak > 0.891:
        loop_body = loop_body * (0.891 / peak)

    return loop_body


def main():
    parser = argparse.ArgumentParser(description="90s 真實雨聲無縫 loop 製作")
    parser.add_argument(
        "--source",
        type=str,
        default=str(ROOT / "assets/curated/rain_real_amb2.wav"),
        help="來源 WAV（48kHz, mono）"
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(ROOT / "out/loops")
    )
    args = parser.parse_args()

    source_path = pathlib.Path(args.source)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[looper_real_rain] source={source_path.name}  loop={LOOP_DUR_S}s  xfade={XFADE_DUR_S}s")
    print(f"[looper_real_rain] skip_start={SKIP_START_S}s  skip_end={SKIP_END_S}s")

    loop_audio = make_real_rain_loop(source_path)

    out_path = out_dir / "rain_loop_90s_real.wav"
    write_wav(out_path, SR, loop_audio)

    dur = len(loop_audio) / SR
    rms = np.sqrt(np.mean(loop_audio ** 2))
    rms_db = 20 * np.log10(rms + 1e-12)
    peak = np.max(np.abs(loop_audio))
    peak_db = 20 * np.log10(peak + 1e-12)
    print(f"[looper_real_rain] -> {out_path}")
    print(f"[looper_real_rain]    dur={dur:.3f}s  RMS={rms_db:.2f}dBFS  peak={peak_db:.2f}dBFS")


if __name__ == "__main__":
    main()
