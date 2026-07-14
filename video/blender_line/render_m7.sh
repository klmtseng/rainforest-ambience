#!/usr/bin/env bash
# render_m7.sh — full M7 pipeline
# Usage: bash render_m7.sh [--frames N] [--samples N]
# Defaults: 260 frames, 8 samples
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
FFMPEG="${FFMPEG_BIN:-$(which ffmpeg)}"
FRAMES_DIR="$PROJ/out/video/frames_blender"
OUT_MP4="$PROJ/out/video/blender_loop_10s.mp4"
SCENE_PY="$PROJ/video/blender_line/scene_rain.py"

TOTAL_FRAMES="${1:-260}"
SAMPLES="${2:-8}"

mkdir -p "$FRAMES_DIR"

echo "=== M7 Render: $TOTAL_FRAMES frames @ $SAMPLES samples ==="
echo "Output: $FRAMES_DIR"
echo "Started: $(date)"

# Step 1: Blender render (synchronous, foreground)
blender -b --factory-startup -noaudio -E CYCLES \
    -P "$SCENE_PY" \
    -- --frames "$TOTAL_FRAMES" --outdir "$FRAMES_DIR/" --samples "$SAMPLES" \
    2>&1 | tee "$FRAMES_DIR/blender_render.log"

RENDER_EXIT=$?
if [ $RENDER_EXIT -ne 0 ]; then
    echo "ERROR: Blender exited with code $RENDER_EXIT"
    exit $RENDER_EXIT
fi

# Count actual frames
ACTUAL_FRAMES=$(ls "$FRAMES_DIR"/frame_*.png 2>/dev/null | wc -l)
echo "Frames rendered: $ACTUAL_FRAMES / $TOTAL_FRAMES"

if [ "$ACTUAL_FRAMES" -lt "$TOTAL_FRAMES" ]; then
    echo "ERROR: Expected $TOTAL_FRAMES frames, got $ACTUAL_FRAMES"
    exit 1
fi

# Step 2: ffmpeg — assemble 240-frame (10s) MP4
# Strategy: use 260 rendered frames
#   input A = frames 1-240 (main 10s loop body)
#   input B = frames 221-260 (40 frames covering the xfade tail)
#   xfade: fade, duration=20/24≈0.833s, offset=220/24≈9.167s
#   → output: 240 frames = exactly 10s loop that closes smoothly

echo "=== Assembling MP4 with xfade loop closure ==="
TAIL_DIR="$FRAMES_DIR/xfade_tail"
mkdir -p "$TAIL_DIR"

# Copy frames 221-260 as tail sequence, renaming to 0001-0040
IDX=1
for SRC_F in $(seq 221 260); do
    printf -v DEST "%04d" $IDX
    printf -v SRC "%04d" $SRC_F
    cp "$FRAMES_DIR/frame_${SRC}.png" "$TAIL_DIR/frame_${DEST}.png"
    IDX=$((IDX + 1))
done

XFADE_DUR="0.833"    # 20 frames / 24fps
XFADE_OFFSET="9.167" # 220 frames / 24fps

"$FFMPEG" -y \
    -framerate 24 -start_number 1 -i "$FRAMES_DIR/frame_%04d.png" \
    -framerate 24 -start_number 1 -i "$TAIL_DIR/frame_%04d.png" \
    -filter_complex "[0:v][1:v]xfade=transition=fade:duration=${XFADE_DUR}:offset=${XFADE_OFFSET}[out]" \
    -map "[out]" \
    -frames:v 240 \
    -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
    -r 24 \
    "$OUT_MP4" \
    2>&1 | tee "$FRAMES_DIR/ffmpeg_assemble.log"

FFMPEG_EXIT=$?
if [ $FFMPEG_EXIT -ne 0 ]; then
    echo "WARNING: Crossfade assembly failed (exit $FFMPEG_EXIT), falling back to simple 240-frame cut"
    "$FFMPEG" -y \
        -framerate 24 \
        -start_number 1 \
        -i "$FRAMES_DIR/frame_%04d.png" \
        -frames:v 240 \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        -r 24 \
        "$OUT_MP4" \
        2>&1 | tee "$FRAMES_DIR/ffmpeg_simple.log"
fi

echo "=== Done: $OUT_MP4 ==="
echo "Finished: $(date)"
