#!/usr/bin/env bash
# assemble_and_verify.sh — run after Blender render completes
# Assembles MP4 then runs verify_m7.py
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/../.." && pwd)"
FFMPEG="${FFMPEG_BIN:-$(which ffmpeg)}"
FFPROBE="${FFPROBE_BIN:-$(which ffprobe)}"
FRAMES_DIR="$PROJ/out/video/frames_blender"
OUT_MP4="$PROJ/out/video/blender_loop_10s.mp4"

echo "=== M7 Assembly & Verify ==="
echo "Started: $(date)"

# Count frames
ACTUAL_FRAMES=$(ls "$FRAMES_DIR"/frame_*.png 2>/dev/null | wc -l)
echo "PNG frames found: $ACTUAL_FRAMES"

if [ "$ACTUAL_FRAMES" -lt 240 ]; then
    echo "ERROR: Need at least 240 frames, found $ACTUAL_FRAMES"
    exit 1
fi

# If we have 260+ frames, use xfade; otherwise simple 240-frame cut
if [ "$ACTUAL_FRAMES" -ge 260 ]; then
    echo "=== Assembling with xfade (${ACTUAL_FRAMES} → 240 frames) ==="
    TAIL_DIR="$FRAMES_DIR/xfade_tail"
    mkdir -p "$TAIL_DIR"

    IDX=1
    for SRC_F in $(seq 221 260); do
        printf -v DEST "%04d" $IDX
        printf -v SRC "%04d" $SRC_F
        if [ -f "$FRAMES_DIR/frame_${SRC}.png" ]; then
            cp "$FRAMES_DIR/frame_${SRC}.png" "$TAIL_DIR/frame_${DEST}.png"
        fi
        IDX=$((IDX + 1))
    done

    TAIL_COUNT=$(ls "$TAIL_DIR"/frame_*.png 2>/dev/null | wc -l)
    echo "Tail frames copied: $TAIL_COUNT"

    "$FFMPEG" -y \
        -framerate 24 -start_number 1 -i "$FRAMES_DIR/frame_%04d.png" \
        -framerate 24 -start_number 1 -i "$TAIL_DIR/frame_%04d.png" \
        -filter_complex "[0:v][1:v]xfade=transition=fade:duration=0.833:offset=9.167[out]" \
        -map "[out]" \
        -frames:v 240 \
        -c:v libx264 -pix_fmt yuv420p -crf 18 -preset medium \
        -r 24 \
        "$OUT_MP4" \
        2>&1 | tee "$FRAMES_DIR/ffmpeg_assemble.log"
    FFMPEG_EXIT=$?

    if [ $FFMPEG_EXIT -ne 0 ]; then
        echo "WARNING: xfade failed, using simple 240-frame cut"
        "$FFMPEG" -y \
            -framerate 24 -start_number 1 \
            -i "$FRAMES_DIR/frame_%04d.png" \
            -frames:v 240 \
            -c:v libx264 -pix_fmt yuv420p -crf 18 \
            -r 24 \
            "$OUT_MP4" 2>&1 | tee "$FRAMES_DIR/ffmpeg_simple.log"
    fi
else
    echo "=== Simple 240-frame cut (${ACTUAL_FRAMES} frames available) ==="
    "$FFMPEG" -y \
        -framerate 24 -start_number 1 \
        -i "$FRAMES_DIR/frame_%04d.png" \
        -frames:v 240 \
        -c:v libx264 -pix_fmt yuv420p -crf 18 \
        -r 24 \
        "$OUT_MP4" 2>&1 | tee "$FRAMES_DIR/ffmpeg_simple.log"
fi

echo "=== MP4 info ==="
"$FFPROBE" -v error -select_streams v:0 -count_frames \
    -show_entries stream=nb_read_frames,r_frame_rate \
    -of json "$OUT_MP4" 2>&1

echo "=== Running verify_m7.py ==="
python3 "$PROJ/tests/verify_m7.py"
VERIFY_EXIT=$?

echo "=== Assembly complete: $(date) ==="
exit $VERIFY_EXIT
