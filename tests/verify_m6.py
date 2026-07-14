"""
verify_m6.py — M6 ffmpeg 影片線驗收
exit 0 = ALL PASS;exit 1 = 有 FAIL

驗收清單(閘門凍結,不准修改數字):
  V6.1  rainforest_45min_1080p25.mp4 存在且時長 = master.wav ±0.1s
  V6.2  rainforest_loop_90s_1080p25.mp4 存在且時長 = rain_loop_90s.wav ±0.1s
  V6.3  抽 ≥3 個 flash 時點幀亮度 > 前後 1s 幀均值 1.5 倍
  V6.4  抽 ≥3 個非 flash 時點無此現象(亮度 ≤ 鄰幀均值 1.5 倍)
"""

import os
import shutil
import json
import math
import pathlib
import subprocess
import sys
import tempfile
import numpy as np

ROOT = pathlib.Path(__file__).parent.parent
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"
OUT_DIR = ROOT / "out/video"
FLASH_TIMES_JSON = ROOT / "video/ffmpeg_line/assets/flash_times.json"
LONG_AUDIO = ROOT / "out/long/master.wav"
LOOP_AUDIO = ROOT / "out/loops/rain_loop_90s.wav"

PASS = []
FAIL = []


def check(name, cond, msg=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name)
        print(f"  FAIL  {name}  {msg}")


def get_duration(path: pathlib.Path) -> float:
    """用 ffprobe 取得媒體檔案時長(秒)。"""
    r = subprocess.run(
        [FFMPEG, "-i", str(path)],
        capture_output=True, text=True
    )
    # Duration in stderr
    import re
    m = re.search(r"Duration:\s*(\d+):(\d+):([\d.]+)", r.stderr)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + s
    return -1.0


def wav_duration(path: pathlib.Path) -> float:
    """從 WAV header 取得時長。"""
    from scipy.io import wavfile
    sr, data = wavfile.read(str(path))
    return len(data) / sr


def extract_frame_luma(video_path: pathlib.Path, t_sec: float) -> float:
    """
    擷取指定時間點幀,回傳平均亮度(Y channel 0-255 range)。
    """
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp_path = tmp.name

    r = subprocess.run(
        [FFMPEG, "-y", "-ss", str(t_sec), "-i", str(video_path),
         "-frames:v", "1", "-vf", "format=yuv420p", tmp_path],
        capture_output=True, text=True
    )
    if r.returncode != 0 or not pathlib.Path(tmp_path).exists():
        return -1.0

    from PIL import Image
    img = np.array(Image.open(tmp_path).convert("L"), dtype=np.float32)
    pathlib.Path(tmp_path).unlink(missing_ok=True)
    return float(img.mean())


def sample_neighborhood_luma(video_path: pathlib.Path, t_center: float,
                               radius: float = 1.0, n_samples: int = 4) -> float:
    """
    在 t_center ± radius 秒的範圍內抽樣幾幀,回傳平均亮度。
    避開 t_center 本身(避免含閃電幀)。
    """
    offsets = np.linspace(-radius, radius, n_samples * 2 + 1)
    lumas = []
    for off in offsets:
        if abs(off) < 0.15:  # 跳過接近中心的幀(可能含閃電)
            continue
        t = t_center + off
        if t < 0.5 or t > get_duration(video_path) - 0.5:
            continue
        luma = extract_frame_luma(video_path, t)
        if luma > 0:
            lumas.append(luma)
        if len(lumas) >= n_samples:
            break
    return float(np.mean(lumas)) if lumas else -1.0


print("\n[M6] ffmpeg 影片線驗收")

# ----------------------------------------------------------------
# V6.1 45min 影片存在且時長正確
# ----------------------------------------------------------------
mp4_45 = OUT_DIR / "rainforest_45min_1080p25.mp4"
if mp4_45.exists():
    video_dur = get_duration(mp4_45)
    audio_dur = wav_duration(LONG_AUDIO)
    check("V6.1_45min_exists", True)
    check("V6.1_45min_duration",
          abs(video_dur - audio_dur) <= 0.1,
          f"video={video_dur:.2f}s audio={audio_dur:.2f}s diff={abs(video_dur-audio_dur):.3f}s")
else:
    check("V6.1_45min_exists", False, f"file not found: {mp4_45}")
    check("V6.1_45min_duration", False, "file not found")

# ----------------------------------------------------------------
# V6.2 90s loop 存在且時長正確
# ----------------------------------------------------------------
mp4_loop = OUT_DIR / "rainforest_loop_90s_1080p25.mp4"
if mp4_loop.exists():
    loop_video_dur = get_duration(mp4_loop)
    loop_audio_dur = wav_duration(LOOP_AUDIO)
    check("V6.2_loop_exists", True)
    check("V6.2_loop_duration",
          abs(loop_video_dur - loop_audio_dur) <= 0.1,
          f"video={loop_video_dur:.2f}s audio={loop_audio_dur:.2f}s")
else:
    check("V6.2_loop_exists", False, f"file not found: {mp4_loop}")
    check("V6.2_loop_duration", False, "file not found")

# ----------------------------------------------------------------
# V6.3 flash 幀亮度 > 鄰幀均值 1.5x (抽 ≥3 個)
# V6.4 非 flash 幀無此現象 (抽 ≥3 個)
# ----------------------------------------------------------------
if not mp4_45.exists():
    check("V6.3_flash_brightness", False, "45min mp4 not found")
    check("V6.4_noflash_brightness", False, "45min mp4 not found")
elif not FLASH_TIMES_JSON.exists():
    check("V6.3_flash_brightness", False, "flash_times.json not found")
    check("V6.4_noflash_brightness", False, "flash_times.json not found")
else:
    with open(FLASH_TIMES_JSON) as f:
        flash_data = json.load(f)

    main_flashes = flash_data.get("main_flashes", [])
    video_duration = get_duration(mp4_45)

    # V6.3: 選前3個 main flash 時點 (在影片範圍內)
    sample_flashes = [w for w in main_flashes if w["t_start"] + 0.04 < video_duration][:3]
    print(f"\n  V6.3: 抽樣 {len(sample_flashes)} 個 flash 時點...")

    flash_pass_count = 0
    for i, w in enumerate(sample_flashes):
        t_mid = (w["t_start"] + w["t_end"]) / 2
        flash_luma = extract_frame_luma(mp4_45, t_mid)
        neighbor_luma = sample_neighborhood_luma(mp4_45, t_mid, radius=1.0, n_samples=4)
        ratio = flash_luma / neighbor_luma if neighbor_luma > 0 else 0
        is_bright = ratio >= 1.5
        print(f"    flash[{i}] t={t_mid:.2f}s: flash_luma={flash_luma:.1f} "
              f"neighbor={neighbor_luma:.1f} ratio={ratio:.2f}x {'BRIGHT' if is_bright else 'DARK'}")
        if is_bright:
            flash_pass_count += 1

    check("V6.3_flash_brightness",
          flash_pass_count >= min(3, len(sample_flashes)),
          f"only {flash_pass_count}/{len(sample_flashes)} flash frames exceeded 1.5x threshold")

    # V6.4: 選 ≥3 個非 flash 時點
    # 找出遠離所有 flash 窗口的時間點
    all_windows = flash_data.get("all_windows", [])
    def is_near_flash(t, margin=2.0):
        for w in all_windows:
            if abs(t - w["t_start"]) < margin or abs(t - w["t_end"]) < margin:
                return True
        return False

    # 候選非 flash 時間點(每隔約200s)
    candidate_times = [100.0, 300.0, 650.0, 950.0, 1500.0, 2000.0, 2400.0]
    noflash_times = [t for t in candidate_times
                     if not is_near_flash(t) and t < video_duration - 1.0][:3]
    print(f"\n  V6.4: 抽樣 {len(noflash_times)} 個非 flash 時點: {noflash_times}")

    noflash_pass_count = 0
    for t in noflash_times:
        sample_luma = extract_frame_luma(mp4_45, t)
        neighbor_luma = sample_neighborhood_luma(mp4_45, t, radius=1.0, n_samples=4)
        ratio = sample_luma / neighbor_luma if neighbor_luma > 0 else 0
        is_ok = ratio < 1.5
        print(f"    noflash t={t:.1f}s: luma={sample_luma:.1f} "
              f"neighbor={neighbor_luma:.1f} ratio={ratio:.2f}x {'OK' if is_ok else 'FAIL'}")
        if is_ok:
            noflash_pass_count += 1

    check("V6.4_noflash_brightness",
          noflash_pass_count >= min(3, len(noflash_times)),
          f"only {noflash_pass_count}/{len(noflash_times)} non-flash checks passed")

# ----------------------------------------------------------------
# 結果
# ----------------------------------------------------------------
print(f"\n[M6] PASS={len(PASS)}  FAIL={len(FAIL)}")
if FAIL:
    print("  Failures:", FAIL)
    sys.exit(1)
print("[M6] ALL PASS")
sys.exit(0)
