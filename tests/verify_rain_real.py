"""
verify_rain_real.py — 真實雨聲 loop 驗收腳本

驗收條件（全過才算 PASS，exit 0）：
  V_R1  loop 長度 = 90.000s（±0.5ms 容差）
  V_R2  無 clip（peak < 0dBFS）
  V_R3  首尾 RMS 差 < 1dB（取首/尾各 0.5s）
  V_R4  2-8kHz env-kurtosis >= 10（真實雨滴衝擊的脈衝特徵）
  V_R5  125-250Hz 頻帶相對能量 > 2-4kHz 頻帶（真實雨聲低頻豐富）

用法:
    python tests/verify_rain_real.py [--loop PATH]
"""

import argparse
import pathlib
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal

ROOT = pathlib.Path(__file__).parent.parent


def load(path: pathlib.Path):
    sr, x = wavfile.read(str(path))
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        # 本專案用 24-bit packed in int32（float64_to_int32 用 2^23-1 縮放）
        x = x.astype(np.float64) / (2 ** 23)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return sr, x.astype(np.float64)


def octave_band_rel(sr, x, flo, fhi):
    """Mean power in [flo, fhi) Hz, relative to full-band mean, in dB."""
    f, pxx = signal.welch(x, sr, nperseg=8192)
    band_mask = (f >= flo) & (f < fhi)
    full_mask = f > 20
    band_db = 10 * np.log10(pxx[band_mask].mean() + 1e-20)
    full_db = 10 * np.log10(pxx[full_mask].mean() + 1e-20)
    return band_db - full_db


def env_kurtosis_2_8k(sr, x):
    """Envelope kurtosis in the 2-8kHz band (5ms smoothed)."""
    sos = signal.butter(4, [2000, 8000], "bandpass", fs=sr, output="sos")
    b = signal.sosfilt(sos, x)
    env = np.abs(signal.hilbert(b))
    win = int(sr * 0.005)
    env_s = np.convolve(env, np.ones(win) / win, mode="valid")
    kurt = float(
        ((env_s - env_s.mean()) ** 4).mean() / (env_s.std() ** 4 + 1e-20)
    )
    return kurt


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x ** 2)) + 1e-12)


def main():
    parser = argparse.ArgumentParser(description="真實雨聲 loop 驗收")
    parser.add_argument(
        "--loop",
        type=str,
        default=str(ROOT / "out/loops/rain_loop_90s_real.wav"),
    )
    args = parser.parse_args()

    loop_path = pathlib.Path(args.loop)
    print(f"[verify_rain_real] 驗收檔案: {loop_path}")

    if not loop_path.exists():
        print(f"  FAIL: 檔案不存在 {loop_path}")
        sys.exit(1)

    sr, x = load(loop_path)
    dur = len(x) / sr
    results = []

    # V_R1 長度 90.000s ±0.5ms
    tol = 0.0005
    ok_r1 = abs(dur - 90.0) <= tol
    results.append(("V_R1 長度=90.000s", ok_r1,
                    f"dur={dur:.4f}s  tol=±{tol*1000:.1f}ms"))

    # V_R2 無 clip
    peak = float(np.max(np.abs(x)))
    ok_r2 = peak < 1.0
    peak_db = 20 * np.log10(peak + 1e-12)
    results.append(("V_R2 無clip", ok_r2,
                    f"peak={peak_db:.2f}dBFS (需<0dBFS)"))

    # V_R3 首尾 RMS 差 < 1dB
    n_half = int(0.5 * sr)   # 0.5s
    head_rms = rms_db(x[:n_half])
    tail_rms = rms_db(x[-n_half:])
    diff_rms = abs(head_rms - tail_rms)
    ok_r3 = diff_rms < 1.0
    results.append(("V_R3 首尾RMS差<1dB", ok_r3,
                    f"head={head_rms:.2f}dBFS  tail={tail_rms:.2f}dBFS  diff={diff_rms:.3f}dB"))

    # V_R4 2-8kHz env-kurtosis >= 10
    kurt = env_kurtosis_2_8k(sr, x)
    ok_r4 = kurt >= 10.0
    results.append(("V_R4 kurtosis>=10", ok_r4,
                    f"kurtosis={kurt:.1f}"))

    # V_R5 125-250Hz > 2-4kHz 相對能量
    rel_125_250 = octave_band_rel(sr, x, 125, 250)
    rel_2k_4k = octave_band_rel(sr, x, 2000, 4000)
    ok_r5 = rel_125_250 > rel_2k_4k
    results.append(("V_R5 125-250Hz>2-4kHz", ok_r5,
                    f"125-250Hz={rel_125_250:+.1f}dB  2-4kHz={rel_2k_4k:+.1f}dB"))

    # 輸出結果
    print()
    all_pass = True
    for name, ok, detail in results:
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] {name}  ({detail})")
        if not ok:
            all_pass = False

    print()
    if all_pass:
        print("verify_rain_real: ALL PASS")
        sys.exit(0)
    else:
        n_fail = sum(1 for _, ok, _ in results if not ok)
        print(f"verify_rain_real: {n_fail} FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
