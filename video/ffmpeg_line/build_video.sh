#!/usr/bin/env bash
# build_video.sh — 深山夜雨 ambience 影片產線 (M6)
#
# 輸出:
#   out/video/rainforest_45min_1080p25.mp4   (45min 主影片)
#   out/video/rainforest_loop_90s_1080p25.mp4 (90s loop 版,無雷無閃電)
#
# 用法:
#   bash video/ffmpeg_line/build_video.sh [--test]
#
#   --test  只渲 3min 切片(快速驗收用),輸出到 out/video/test_3min.mp4
#
# 依賴:
#   - video/ffmpeg_line/assets/base.png       (gen_base_image.py 產)
#   - video/ffmpeg_line/assets/flash_filter.txt (gen_flash_filter.py 產)
#   - out/long/master.wav                     (mixdown.py 產)
#   - out/loops/rain_loop_90s.wav             (looper.py 產)
#   - tools/ffmpeg/ffmpeg

set -euo pipefail

FFMPEG="${FFMPEG_BIN:-$(which ffmpeg)}"
PROJECT="$(cd "$(dirname "$0")/../.." && pwd)"

ASSETS_DIR="$PROJECT/video/ffmpeg_line/assets"
OUT_DIR="$PROJECT/out/video"
LONG_AUDIO="$PROJECT/out/long/master.wav"
LOOP_AUDIO="$PROJECT/out/loops/rain_loop_90s.wav"
BASE_IMG="$ASSETS_DIR/base.png"
FLASH_FILTER="$ASSETS_DIR/flash_filter.txt"

TEST_MODE=0
if [[ "${1:-}" == "--test" ]]; then
    TEST_MODE=1
fi

mkdir -p "$OUT_DIR"

# -----------------------------------------------------------------------
# Step 1: 生成素材 tiles
# -----------------------------------------------------------------------
echo "[M6] Step 1: 生成 assets..."

python3 "$PROJECT/video/ffmpeg_line/gen_base_image.py"
python3 "$PROJECT/video/ffmpeg_line/gen_flash_filter.py"

RAIN_NEAR="$ASSETS_DIR/rain_near.png"
RAIN_FAR="$ASSETS_DIR/rain_far.png"
MIST_TILE="$ASSETS_DIR/mist_tile.png"
WHITE_IMG="$ASSETS_DIR/white.png"

# 生成雨絲 tile(近層:3% column 有 streak)
python3 - <<'EOF'
import numpy as np
from PIL import Image
import pathlib
import sys

ASSETS_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "assets"
W, H = 1920, 1080
rng = np.random.default_rng(42)

# 近層雨絲:3% columns 亮, lum=150
cols_near = rng.random(W)
near = np.zeros((H, W), dtype=np.uint8)
for x in range(W):
    if cols_near[x] > 0.97:
        near[:, x] = 150
Image.fromarray(near).save(str(ASSETS_DIR / "rain_near.png"))

# 遠層雨絲:2.5% columns 亮, lum=80 (更細更暗)
cols_far = rng.random(W)
far = np.zeros((H, W), dtype=np.uint8)
for x in range(W):
    if cols_far[x] > 0.975:
        far[:, x] = 80
Image.fromarray(far).save(str(ASSETS_DIR / "rain_far.png"))

# 霧層:低解析度隨機噪聲,上採樣得到自然模糊感
mist_low = (rng.random((27, 48)) * 25).astype(np.uint8)
Image.fromarray(mist_low).save(str(ASSETS_DIR / "mist_tile.png"))

# 白色疊加圖
Image.fromarray(np.full((1080, 1920, 3), 255, dtype=np.uint8)).save(str(ASSETS_DIR / "white.png"))

print("[gen_tiles] rain_near.png, rain_far.png, mist_tile.png, white.png saved")
EOF

# -----------------------------------------------------------------------
# Step 2: 讀 flash_filter.txt 取得 enable 表達式和 alpha
# -----------------------------------------------------------------------
MAIN_ENABLE=$(grep "^MAIN_ENABLE=" "$FLASH_FILTER" | cut -d= -f2-)
ECHO_ENABLE=$(grep "^ECHO_ENABLE=" "$FLASH_FILTER" | cut -d= -f2-)
MAIN_ALPHA=$(grep "^MAIN_ALPHA=" "$FLASH_FILTER" | cut -d= -f2-)
ECHO_ALPHA=$(grep "^ECHO_ALPHA=" "$FLASH_FILTER" | cut -d= -f2-)

# colorchannelmixer aa= 值(float)
MAIN_AA=$MAIN_ALPHA
ECHO_AA=$ECHO_ALPHA

echo "[M6] Flash events loaded"
echo "  MAIN_ENABLE (first 80 chars): ${MAIN_ENABLE:0:80}..."
echo "  MAIN_AA=$MAIN_AA  ECHO_AA=$ECHO_AA"

# -----------------------------------------------------------------------
# Step 3: ffmpeg filtergraph 定義
# -----------------------------------------------------------------------
# Input 索引:
#   [0:v] = base.png (loop 1)
#   [1:v] = rain_near.png (loop 1)
#   [2:v] = rain_far.png (loop 1)
#   [3:v] = mist_tile.png (loop 1)
#   [4:v] = white.png (loop 1)  ← main flash overlay
#   [5:v] = white.png (loop 1)  ← echo flash overlay
#   [6:a] = audio (long or loop)
#
# filtergraph:
#   近層雨絲 scroll 快
#   遠層雨絲 scroll 慢
#   霧層 低解析度上採樣 + 緩慢橫移
#   底圖 + 近層 (screen blend) + 遠層 (screen blend) + 霧 (screen blend)
#   → main flash overlay → echo flash overlay
#
# 雨絲速度:
#   near = 0.22 (22% image per second, 1080*0.22 = 237px/s)
#   far  = 0.10 (10% image per second, 1080*0.10 = 108px/s)

build_fc() {
    local main_enable="$1"
    local echo_enable="$2"
    local main_aa="$3"
    local echo_aa="$4"
    echo "[1:v]scroll=vertical=0.22[near];[2:v]scroll=vertical=0.10[far];[3:v]scale=1920:1080:flags=bilinear,scroll=horizontal=0.003:vertical=0.001[mist];[0:v][near]blend=all_mode=screen:all_opacity=0.4[bn];[bn][far]blend=all_mode=screen:all_opacity=0.25[bf];[bf][mist]blend=all_mode=screen:all_opacity=0.07[combined];[4:v]format=rgba,colorchannelmixer=aa=${main_aa}[main_fl];[combined][main_fl]overlay=enable='${main_enable}'[f1];[5:v]format=rgba,colorchannelmixer=aa=${echo_aa}[echo_fl];[f1][echo_fl]overlay=enable='${echo_enable}'[out]"
}

# -----------------------------------------------------------------------
# Step 4a: 3min 測試片(迭代用)
# -----------------------------------------------------------------------
if [[ $TEST_MODE -eq 1 ]]; then
    echo "[M6] TEST MODE: 渲 3min 切片..."
    FC=$(build_fc "$MAIN_ENABLE" "$ECHO_ENABLE" "$MAIN_AA" "$ECHO_AA")
    start_ts=$(date +%s)
    "$FFMPEG" -y \
        -loop 1 -i "$BASE_IMG" \
        -loop 1 -i "$RAIN_NEAR" \
        -loop 1 -i "$RAIN_FAR" \
        -loop 1 -i "$MIST_TILE" \
        -loop 1 -i "$WHITE_IMG" \
        -loop 1 -i "$WHITE_IMG" \
        -i "$LONG_AUDIO" \
        -t 180 \
        -filter_complex "$FC" \
        -map "[out]" -map "6:a" \
        -c:v libx264 -crf 26 -preset veryfast -pix_fmt yuv420p \
        -c:a aac -b:a 192k \
        -movflags +faststart \
        "$OUT_DIR/test_3min.mp4"
    end_ts=$(date +%s)
    echo "[M6] TEST done  elapsed=$((end_ts - start_ts))s  -> $OUT_DIR/test_3min.mp4"
    exit 0
fi

# -----------------------------------------------------------------------
# Step 4b: 90s loop 版 (無雷無閃電)
# -----------------------------------------------------------------------
echo "[M6] 渲染 90s loop 版..."
FC_LOOP="[1:v]scroll=vertical=0.22[near];[2:v]scroll=vertical=0.10[far];[3:v]scale=1920:1080:flags=bilinear,scroll=horizontal=0.003:vertical=0.001[mist];[0:v][near]blend=all_mode=screen:all_opacity=0.4[bn];[bn][far]blend=all_mode=screen:all_opacity=0.25[bf];[bf][mist]blend=all_mode=screen:all_opacity=0.07[out]"
start_ts=$(date +%s)
"$FFMPEG" -y \
    -loop 1 -i "$BASE_IMG" \
    -loop 1 -i "$RAIN_NEAR" \
    -loop 1 -i "$RAIN_FAR" \
    -loop 1 -i "$MIST_TILE" \
    -i "$LOOP_AUDIO" \
    -t 90 \
    -filter_complex "$FC_LOOP" \
    -map "[out]" -map "4:a" \
    -c:v libx264 -crf 26 -preset veryfast -pix_fmt yuv420p \
    -c:a aac -b:a 192k \
    -movflags +faststart \
    "$OUT_DIR/rainforest_loop_90s_1080p25.mp4"
end_ts=$(date +%s)
echo "[M6] 90s loop done  elapsed=$((end_ts - start_ts))s  -> $OUT_DIR/rainforest_loop_90s_1080p25.mp4"

# -----------------------------------------------------------------------
# Step 5: 45min 主影片
# -----------------------------------------------------------------------
echo "[M6] 渲染 45min 主影片 (含閃電)..."
echo "  此步驟預計耗時 25-45 分鐘,前景同步執行..."
FC=$(build_fc "$MAIN_ENABLE" "$ECHO_ENABLE" "$MAIN_AA" "$ECHO_AA")
start_ts=$(date +%s)
"$FFMPEG" -y \
    -loop 1 -i "$BASE_IMG" \
    -loop 1 -i "$RAIN_NEAR" \
    -loop 1 -i "$RAIN_FAR" \
    -loop 1 -i "$MIST_TILE" \
    -loop 1 -i "$WHITE_IMG" \
    -loop 1 -i "$WHITE_IMG" \
    -i "$LONG_AUDIO" \
    -t 2700 \
    -filter_complex "$FC" \
    -map "[out]" -map "6:a" \
    -c:v libx264 -crf 26 -preset veryfast -pix_fmt yuv420p \
    -c:a aac -b:a 192k \
    -movflags +faststart \
    "$OUT_DIR/rainforest_45min_1080p25.mp4"
end_ts=$(date +%s)
elapsed=$((end_ts - start_ts))
elapsed_min=$(echo "scale=1; $elapsed/60" | bc)
echo "[M6] 45min done  elapsed=${elapsed}s (${elapsed_min}min)  -> $OUT_DIR/rainforest_45min_1080p25.mp4"
echo "[M6] ALL DONE"
