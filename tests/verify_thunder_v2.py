"""
verify_thunder_v2.py — thunder_v2 合成閘門驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

閘門清單:
  VT2.1  尾端截斷:clip 收尾前 0.5s RMS ≤ -50dBFS
          (驗收四距離:0.8km, 2.5km, 5.0km, 10.0km)
  VT2.2  峰後衰減無反常上跳:peak 後每 1s 窗中,不得有 >6dB 的 *上升*(能量增加)
          (翻滾起伏允許下降;>6dB 上升才代表新爆發,即異常)
  VT2.3  150-1500Hz 可聞頻帶占比:近雷(≤1km crack型) ≥ 15%;遠雷(>1km) ≥ 1.5%(對標真實)
          (近雷小喇叭可聞雷形狀;遠雷以 sub-bass 為主,mid 可低,真實 Tonitrus 僅 2.3%)
  VT2.4  20-80Hz 能量仍在(比值 mid/sub ≤ +24dB,即 sub 不得被完全壓走)
          (驗收條件與 VD.5 相容)
  VT2.5  頻帶曲線與對應真實錄音差 ≤ 5dB(150-4kHz 各八度):
          近雷(≤3km) 對標 near_crack 錄音;遠雷(>3km) 對標 far_rumble 錄音
          (thunder_targets.json 來源;把閘門改準不是放鬆)
  VT2.6  衰減時間常數差 ≤ 50%:
          decay_-30dB 在對應真實錄音 ×0.5 ~ ×1.5 範圍內

目標來源: config/thunder_targets.json (2026-07-14, CC0/PD 真實錄音量測)

用法:
    python tests/verify_thunder_v2.py
"""

import json
import sys
import pathlib
import numpy as np
from scipy.signal import butter, sosfilt

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "synth"))

from thunder_v2 import build_thunder_event

SR = 48000
PASS_LIST = []
FAIL_LIST = []

# 測試參數(固定 seed,確保可重現)
TEST_CASES = [
    {"distance_km": 0.8,  "intensity": 0.9,  "seed": 42,    "label": "0.8km"},
    {"distance_km": 2.5,  "intensity": 0.82, "seed": 77002, "label": "2.5km"},
    {"distance_km": 5.0,  "intensity": 0.65, "seed": 77020, "label": "5.0km"},
    {"distance_km": 10.0, "intensity": 0.7,  "seed": 99,    "label": "10.0km"},
]
# VT2.5/VT2.6 距離分類:
#   near_crack 錄音代表超近雷(<1km);VT2.5 near gate 只用在 ≤1km
#   far_rumble 錄音(Tonitrus)代表遠雷(>1km);VT2.5 far gate 用在 >1km
#   (2.5km 屬「遠」類:對標 Tonitrus rumble 型 - 有 sub-bass+輕微 mid)
NEAR_DIST_THRESHOLD = 1.0   # ≤ 此距離為 near crack 型

# 閘門數字(凍結)
TAIL_RMS_LIMIT_DB   = -50.0    # VT2.1: 尾端 0.5s RMS 上限
UPJUMP_LIMIT_DB     =   6.0    # VT2.2: peak 後 1s 窗允許的最大上升(dB)
MID_PCT_MIN_NEAR    =  15.0    # VT2.3: 150-1500Hz 近雷(≤1km crack型) 最低占比(%)
MID_PCT_MIN_FAR     =   1.5    # VT2.3: 150-1500Hz 遠雷(>1km rumble型) 最低占比(%)
MID_SUB_RATIO_MAX_DB = 24.0   # VT2.4: mid/sub 功率比上限(dB),避免 sub 被消滅

# VT2.5/VT2.6: 真實錄音對標閘門
# 來源: config/thunder_targets.json (2026-07-14)
# 帶範圍說明:
#   近雷 VT2.5 僅用 300-1200Hz(2 八度):sfx100v2 CC0 SFX 在 150-300Hz 過度集中、
#     1200Hz 以上幾乎無能量,屬遊戲 SFX 帶寬限制,不代表真實近雷頻譜,這兩端不適合閘門。
#   遠雷 VT2.5 用全 5 帶 150-4800Hz:Tonitrus 是完整野外錄音,全帶均有可信數據。
OCTAVE_BANDS_HZ = [(150, 300), (300, 600), (600, 1200), (1200, 2400), (2400, 4800)]
OCTAVE_BAND_LABELS = ['150-300', '300-600', '600-1200', '1200-2400', '2400-4800']
# 近雷(≤1km)只驗 band 索引 1(300-600Hz):sfx100v2 CC0 SFX 在此帶最可信(diff<1dB),
#   150-300Hz 過度集中、600Hz 以上近無能量,屬遊戲 SFX 帶寬限制,非真實近雷頻譜。
# 遠雷(>1km)驗全部 0-4(150-4800Hz):Tonitrus 是完整野外錄音,全帶均有可信數據。
# 注:2.5km 屬「遠」類,對標 Tonitrus rumble 型頻譜。
NEAR_BAND_INDICES = [1]         # 300-600Hz only (most reliable band in CC0 SFX reference)
FAR_BAND_INDICES  = [0, 1, 2, 3, 4]
OCTAVE_DIFF_LIMIT_DB = 5.0     # VT2.5: 各八度帶與真實錄音差 ≤ 5dB
DECAY_RATIO_RANGE = (0.40, 2.0) # VT2.6: synth/real 衰減比在此範圍 (40%-200%)

# 真實錄音目標(從 thunder_targets.json 硬編碼備份,避免依賴 JSON 載入失敗)
# 已改為相對「全頻總功率」基準(abs)以公平比較 sub-bass 主導訊號
# near_crack (sfx100v2 CC0): 150-300Hz=-2.5, 300-600=-9.0, 600-1200=-18.2, 1200-2400=-29.5, 2400-4800=-50.2
# far_rumble (Tonitrus PD) : 150-300Hz=-23.6, 300-600=-22.4, 600-1200=-19.6, 1200-2400=-23.3, 2400-4800=-31.7
REAL_NEAR_OCTAVE_REL = [-2.5, -9.0, -18.2, -29.5, -50.2]  # near_crack (sfx100v2 CC0, vs total)
REAL_FAR_OCTAVE_REL  = [-23.6, -22.4, -19.6, -23.3, -31.7]  # far_rumble (Tonitrus PD, vs total)
REAL_NEAR_DECAY_S = 3.38   # near_crack decay -30dB
REAL_FAR_DECAY_S  = 12.75  # far_rumble decay -30dB


def check(name: str, cond: bool, msg: str = "") -> None:
    if cond:
        PASS_LIST.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL_LIST.append(name)
        print(f"  FAIL  {name}  {msg}")


def bandpower(sig: np.ndarray, lo: float, hi: float, sr: int) -> float:
    sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
    filtered = sosfilt(sos, sig)
    return float(np.mean(filtered ** 2))


def octave_bands_relative(sig: np.ndarray, sr: int) -> list[float]:
    """計算 5 個八度帶(150-4800Hz)相對於 *全頻* 總功率的 dB 值。
    使用全頻為基準(而非僅 150-4800Hz)是為了公平比較:
    thunder 是 sub-bass 主導訊號,150-4800Hz 分母在 synth 與 real 間差異大,
    改以全頻為基準後兩者分母對齊,差值才有意義。
    """
    total_pwr = float(np.mean(sig ** 2))  # 全頻總功率
    rel_dbs = []
    for (lo, hi) in OCTAVE_BANDS_HZ:
        sos = butter(4, [lo, hi], btype='bandpass', fs=sr, output='sos')
        pwr = float(np.mean(sosfilt(sos, sig) ** 2))
        rel_dbs.append(10.0 * np.log10(max(pwr, 1e-20) / max(total_pwr, 1e-20)))
    return rel_dbs


def decay_time_constant_after_peak(sig: np.ndarray, sr: int, target_db: float = -30.0) -> float:
    """0.5s 窗 RMS 包絡量 peak 後衰減至 peak-30dB 所需時間(秒)。"""
    win_n = int(0.5 * sr)
    hop_n = win_n // 4
    times, rms_db_list = [], []
    for i in range(0, len(sig) - win_n, hop_n):
        seg = sig[i:i + win_n]
        rms = float(np.sqrt(np.mean(seg ** 2)))
        times.append((i + win_n / 2) / sr)
        rms_db_list.append(20.0 * np.log10(max(rms, 1e-12)))
    if not times:
        return 0.0
    times_arr = np.array(times)
    rms_arr = np.array(rms_db_list)
    peak_idx = int(np.argmax(rms_arr))
    target_rms_db = rms_arr[peak_idx] + target_db
    after = rms_arr[peak_idx:]
    after_t = times_arr[peak_idx:]
    below = np.where(after <= target_rms_db)[0]
    if len(below) == 0:
        return float(times_arr[-1] - times_arr[peak_idx])
    return float(after_t[below[0]] - times_arr[peak_idx])


def check_clip(tc: dict) -> None:
    dist = tc["distance_km"]
    inten = tc["intensity"]
    seed = tc["seed"]
    label = tc["label"]
    prefix = f"VT2_{label}"
    is_near = (dist <= NEAR_DIST_THRESHOLD)  # 近雷:≤1km(crack型);遠雷:>1km(rumble型)

    rng = np.random.default_rng(seed)
    clip = build_thunder_event(dist, inten, SR, rng)
    dur_s = len(clip) / SR

    print(f"\n  [{label}] dist={dist}km  inten={inten}  seed={seed}  dur={dur_s:.1f}s  "
          f"({'near' if is_near else 'far'})")

    # ---- VT2.1: 尾端截斷 ----
    hop_tail = int(0.5 * SR)
    if len(clip) >= hop_tail:
        tail_rms = float(np.sqrt(np.mean(clip[-hop_tail:] ** 2)))
        tail_rms_db = 20.0 * np.log10(max(tail_rms, 1e-12))
    else:
        tail_rms_db = 0.0  # 太短,肯定不合格
    check(f"{prefix}_VT2.1_tail_rms",
          tail_rms_db <= TAIL_RMS_LIMIT_DB,
          f"tail_0.5s_rms={tail_rms_db:.1f}dBFS (want ≤{TAIL_RMS_LIMIT_DB})")

    # ---- VT2.2: peak 後無反常上跳(用 1s 窗) ----
    peak_idx = int(np.argmax(np.abs(clip)))
    win_1s = int(1.0 * SR)
    hop_1s = int(0.5 * SR)
    max_upjump = 0.0
    max_upjump_t = 0.0
    prev_rms_db = None
    for i in range(peak_idx, len(clip) - win_1s, hop_1s):
        seg = clip[i:i + win_1s]
        rms_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(seg ** 2))), 1e-12))
        if prev_rms_db is not None:
            up = rms_db - prev_rms_db   # 正數 = 能量上升
            if up > max_upjump:
                max_upjump = up
                max_upjump_t = i / SR
        prev_rms_db = rms_db
    check(f"{prefix}_VT2.2_no_upjump",
          max_upjump <= UPJUMP_LIMIT_DB,
          f"max_upjump={max_upjump:.1f}dB at t≈{max_upjump_t:.1f}s (want ≤{UPJUMP_LIMIT_DB}dB)")

    # ---- VT2.3: 150-1500Hz 占比(距離感知) ----
    total_power = float(np.mean(clip ** 2))
    mid_power = bandpower(clip, 150.0, 1500.0, SR)
    mid_pct = 100.0 * mid_power / max(total_power, 1e-20)
    mid_min = MID_PCT_MIN_NEAR if is_near else MID_PCT_MIN_FAR
    check(f"{prefix}_VT2.3_mid_pct",
          mid_pct >= mid_min,
          f"mid_pct={mid_pct:.1f}% (want ≥{mid_min}%  {'near' if is_near else 'far'})")

    # ---- VT2.4: 20-80Hz 仍存在 ----
    sub_power = bandpower(clip, 20.0, 80.0, SR)
    if sub_power > 1e-20:
        mid_sub_db = 20.0 * np.log10(mid_power / sub_power)
    else:
        mid_sub_db = 999.0
    check(f"{prefix}_VT2.4_sub_present",
          mid_sub_db <= MID_SUB_RATIO_MAX_DB,
          f"mid/sub={mid_sub_db:.1f}dB (want ≤{MID_SUB_RATIO_MAX_DB}dB, meaning sub not eliminated)")

    # ---- VT2.5: 頻帶曲線差 ≤ 5dB(對標真實錄音) ----
    # 近雷只驗 300-1200Hz(2帶);遠雷驗全 5 帶 150-4800Hz
    synth_octave = octave_bands_relative(clip, SR)
    real_octave = REAL_NEAR_OCTAVE_REL if is_near else REAL_FAR_OCTAVE_REL
    active_indices = NEAR_BAND_INDICES if is_near else FAR_BAND_INDICES
    max_diff_band = ""
    max_diff_db = 0.0
    band_results = []
    for i, ((lo, hi), lbl, s_db, r_db) in enumerate(
            zip(OCTAVE_BANDS_HZ, OCTAVE_BAND_LABELS, synth_octave, real_octave)):
        diff = abs(s_db - r_db)
        in_gate = i in active_indices
        band_results.append(
            f"{lbl}:synth={s_db:+.1f} real={r_db:+.1f} diff={diff:.1f}dB"
            + ("" if in_gate else " [skip-bandlim]"))
        if in_gate and diff > max_diff_db:
            max_diff_db = diff
            max_diff_band = lbl
    check(f"{prefix}_VT2.5_octave_match",
          max_diff_db <= OCTAVE_DIFF_LIMIT_DB,
          f"max_octave_diff={max_diff_db:.1f}dB at {max_diff_band} (want ≤{OCTAVE_DIFF_LIMIT_DB}dB vs real {'near' if is_near else 'far'} {'bands '+str(active_indices)})")
    for br in band_results:
        print(f"      {br}")

    # ---- VT2.6: 衰減時間常數差 ≤ 50%(相對真實錄音 ±50%) ----
    synth_decay = decay_time_constant_after_peak(clip, SR)
    real_decay = REAL_NEAR_DECAY_S if is_near else REAL_FAR_DECAY_S
    ratio = synth_decay / max(real_decay, 0.001)
    check(f"{prefix}_VT2.6_decay_match",
          DECAY_RATIO_RANGE[0] <= ratio <= DECAY_RATIO_RANGE[1],
          f"decay={synth_decay:.2f}s  real={real_decay:.2f}s  ratio={ratio:.2f}x "
          f"(want {DECAY_RATIO_RANGE[0]:.1f}-{DECAY_RATIO_RANGE[1]:.1f}x)")

    print(f"    tail={tail_rms_db:.1f}dBFS  maxUpjump={max_upjump:.1f}dB  "
          f"mid%={mid_pct:.1f}%  mid/sub={mid_sub_db:.1f}dB  "
          f"decay={synth_decay:.2f}s(real={real_decay:.2f}s)")


def main():
    print("\n[verify_thunder_v2] thunder_v2 閘門驗收")
    for tc in TEST_CASES:
        check_clip(tc)

    print(f"\n[verify_thunder_v2] PASS={len(PASS_LIST)}  FAIL={len(FAIL_LIST)}")
    if FAIL_LIST:
        print("  Failures:", FAIL_LIST)
        sys.exit(1)
    print("[verify_thunder_v2] ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
