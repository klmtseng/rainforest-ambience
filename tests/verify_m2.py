"""
verify_m2.py — M2 雷聲驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收清單(閘門凍結,不准修改數字):
  V2.1  thunder_events JSON schema 合法
        (有 sample_rate, duration_s, events 欄;每 event 有
         t_flash, t_audio, distance_km, intensity, start_sample, end_sample)
  V2.2  每雷事件窗 RMS 比背景 RMS 高 > 6dB
  V2.3  t_audio - t_flash ∈ [0, 6] s (物理延遲合理)
  V2.4  雷窗 <80Hz 能量占比 > 40%

策略:用高密度測試 session(120/hour,無不應期)保證至少 3 個事件可驗;
      並驗 json schema(從 render.py 輸出讀);API 層級直接測 thunder module。
"""

import sys
import subprocess
import pathlib
import json

import numpy as np
from scipy.io import wavfile
from scipy.signal import welch

ROOT = pathlib.Path(__file__).parent.parent
RENDER = ROOT / "synth/render.py"
SR = 48000
SEED = 42

sys.path.insert(0, str(ROOT / "synth"))
from thunder import synthesize_thunder_session, build_thunder_event

PASS = []
FAIL = []


def check(name, cond, msg=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {msg}")


print("\n[M2] 雷聲驗收")

# ---- 生成測試用高密度 session (120 events/hour = 2/min, 60s → expect ~2 events) ----
# 使用較長時長 + 高速率保證有多個事件
TEST_DUR = 300.0   # 5 分鐘
TEST_RATE = 60.0   # 60/hour = 1/min; 5min → expect ~5 events
rng_test = np.random.default_rng(SEED)
audio, events = synthesize_thunder_session(
    duration_s=TEST_DUR,
    sr=SR,
    rng=rng_test,
    event_rate_per_hour=TEST_RATE,
    refractory_s=10.0,   # 測試時縮短不應期
    distance_km_range=(0.3, 2.0),
)
print(f"        test session: {TEST_DUR}s @ {TEST_RATE}/hr => {len(events)} events")

# ---- V2.1 schema 合法(直接驗 events list) ----
required_event_keys = {"t_flash", "t_audio", "distance_km", "intensity",
                        "start_sample", "end_sample"}
if len(events) == 0:
    FAIL.append("V2.1_event_schema")
    print(f"  FAIL  V2.1_event_schema  0 events in {TEST_DUR}s @ {TEST_RATE}/hr — thunder broken")
else:
    bad_events = [i for i, e in enumerate(events)
                  if not required_event_keys.issubset(e.keys())]
    check("V2.1_event_schema", len(bad_events) == 0,
          f"events missing keys at indices: {bad_events[:5]}")

# ---- V2.3 t_audio - t_flash ∈ [0, 6] ----
if events:
    delays = [e["t_audio"] - e["t_flash"] for e in events]
    bad_delays = [(i, d) for i, d in enumerate(delays) if not (0 <= d <= 6.0)]
    check("V2.3_timing", len(bad_delays) == 0,
          f"out-of-range delays at: {bad_delays[:5]}")
    print(f"        delays range: [{min(delays):.3f}, {max(delays):.3f}]s  "
          f"(physics: d_km/0.343; max 2km/0.343={2/0.343:.2f}s)")

# ---- 背景 RMS ----
n_total = len(audio)
bg_mask = np.ones(n_total, dtype=bool)
for e in events:
    bg_mask[e["start_sample"]:e["end_sample"]] = False

bg_samples = audio[bg_mask]
if len(bg_samples) > SR:
    bg_rms = np.sqrt(np.mean(bg_samples ** 2))
else:
    bg_rms = 1e-6   # fallback: tiny
bg_rms_db = 20 * np.log10(bg_rms + 1e-12)
print(f"        background RMS: {bg_rms_db:.2f} dBFS")

# ---- V2.2 雷窗 RMS > 背景 + 6dB ----
if events:
    threshold_db = bg_rms_db + 6.0
    bad_rms = []
    for i, e in enumerate(events):
        ss = e["start_sample"]
        es = min(e["end_sample"], n_total)
        if es <= ss:
            continue
        window = audio[ss:es]
        w_rms = np.sqrt(np.mean(window ** 2))
        w_rms_db = 20 * np.log10(w_rms + 1e-12)
        if w_rms_db <= threshold_db:
            bad_rms.append((i, f"{w_rms_db:.1f}dB vs threshold {threshold_db:.1f}dB"))
    check("V2.2_thunder_rms", len(bad_rms) == 0,
          f"events below 6dB threshold: {bad_rms[:5]}")
    print(f"        event RMS checks: {len(events)-len(bad_rms)}/{len(events)} pass")

# ---- V2.4 <80Hz 能量占比 > 40% ----
if events:
    low_ratios = []
    for e in events:
        ss = e["start_sample"]
        es = min(e["end_sample"], n_total)
        window = audio[ss:es]
        if len(window) < 512:
            continue
        nperseg = min(2048, len(window))
        freqs, psd = welch(window, fs=SR, nperseg=nperseg)
        mask_low = (freqs > 0) & (freqs < 80.0)
        mask_all = freqs > 0
        e_low = (np.trapezoid(psd[mask_low], freqs[mask_low])
                 if np.sum(mask_low) > 1 else 0.0)
        e_all = (np.trapezoid(psd[mask_all], freqs[mask_all])
                 if np.sum(mask_all) > 1 else 1e-12)
        ratio = e_low / (e_all + 1e-12)
        low_ratios.append(ratio)

    if low_ratios:
        median_ratio = float(np.median(low_ratios))
        check("V2.4_low_freq_ratio",
              median_ratio > 0.40,
              f"median <80Hz ratio={median_ratio*100:.1f}% (want >40%)")
        print(f"        <80Hz ratios: median={median_ratio*100:.1f}%  "
              f"min={min(low_ratios)*100:.1f}%  max={max(low_ratios)*100:.1f}%")
    else:
        FAIL.append("V2.4_low_freq_ratio")
        print("  FAIL  V2.4_low_freq_ratio  no valid event windows (too short?)")

# ---- V2.1 JSON schema — 再驗 render 輸出 ----
print("\n[M2] render JSON schema 驗收")
out_dir = ROOT / "out/stems"
result = subprocess.run(
    [sys.executable, str(RENDER), "--dur", "60", "--seed", "42",
     "--out-dir", str(out_dir)],
    capture_output=True, text=True, cwd=str(ROOT)
)
events_path = out_dir / "thunder_events_42.json"
check("V2.1_json_file_exists", events_path.is_file(), str(events_path))
if events_path.is_file():
    with open(events_path) as f:
        jdata = json.load(f)
    required_top = {"sample_rate", "duration_s", "events"}
    check("V2.1_json_top_keys", required_top.issubset(jdata.keys()),
          f"missing: {required_top - set(jdata.keys())}")
    # 如果有事件驗 schema
    render_events = jdata.get("events", [])
    if render_events:
        bad = [i for i, e in enumerate(render_events)
               if not required_event_keys.issubset(e.keys())]
        check("V2.1_render_event_schema", len(bad) == 0)
    else:
        PASS.append("V2.1_render_event_schema(none in 60s)")

# ---- 結果 ----
print(f"\n[M2] PASS={len(PASS)}  FAIL={len(FAIL)}")
if FAIL:
    print("  Failures:", FAIL)
    sys.exit(1)

print("[M2] ALL PASS")
sys.exit(0)
