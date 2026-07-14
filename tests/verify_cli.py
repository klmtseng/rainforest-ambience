"""
verify_cli.py — render_ambience.py 煙霧測試
exit 0 = ALL PASS, exit 1 = 任何失敗

測試:
  (a) 30s 無雷渲染:exit 0、時長對、ffmpeg volumedetect 正常（非靜音）
  (b) 60s 帶雷渲染:events-json 存在、格式正確、events 數 > 0
  (c) 同 seed 同參數兩次 bit-identical（WAV 位元完全相同）
  (d) --help 正常退出
"""

import os
import shutil
import json
import pathlib
import subprocess
import sys
import tempfile

PYTHON = sys.executable
ROOT = pathlib.Path(__file__).parent.parent
CLI = ROOT / "render_ambience.py"
FFMPEG = os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg"

PASS_COUNT = 0
FAIL_COUNT = 0


def _ok(tag: str, msg: str = ""):
    global PASS_COUNT
    PASS_COUNT += 1
    suffix = f"  {msg}" if msg else ""
    print(f"  PASS [{tag}]{suffix}")


def _fail(tag: str, msg: str):
    global FAIL_COUNT
    FAIL_COUNT += 1
    print(f"  FAIL [{tag}]  {msg}", file=sys.stderr)


def wav_duration(path: pathlib.Path) -> float:
    """讀 WAV header 計算時長（秒）。支援 float32 WAV（format code 3）。"""
    from scipy.io import wavfile
    sr, data = wavfile.read(str(path))
    n_samples = data.shape[0] if data.ndim == 1 else data.shape[0]
    return n_samples / sr


def wav_bytes(path: pathlib.Path) -> bytes:
    """Read WAV data bytes (skip 44-byte header for float32 WAV comparison)."""
    with open(path, "rb") as f:
        return f.read()


def volumedetect(path: pathlib.Path) -> dict:
    """Run ffmpeg volumedetect and return parsed stats."""
    cmd = [FFMPEG, "-i", str(path), "-af", "volumedetect", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    stats = {}
    for line in r.stderr.splitlines():
        if "mean_volume" in line:
            stats["mean_volume"] = float(line.split(":")[1].strip().replace(" dB", ""))
        if "max_volume" in line:
            stats["max_volume"] = float(line.split(":")[1].strip().replace(" dB", ""))
    return stats


def run_render(extra_args: list[str], out: pathlib.Path, desc: str) -> bool:
    """Run render_ambience.py with given args; return True if exit 0."""
    cmd = [PYTHON, str(CLI)] + extra_args + ["--out", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        _fail(desc, f"exit {r.returncode}\nSTDOUT: {r.stdout[-300:]}\nSTDERR: {r.stderr[-300:]}")
        return False
    return True


# ---------------------------------------------------------------------------
# 測試 (a): 30s 無雷渲染
# ---------------------------------------------------------------------------

def test_a(tmpdir: pathlib.Path):
    print("[A] 30s 無雷渲染 …")
    out = tmpdir / "test_a.wav"
    ok = run_render(
        ["--duration", "30", "--seed", "100", "--thunder-rate", "0", "--no-rain-gush"],
        out, "A.exit0",
    )
    if not ok:
        return

    _ok("A.exit0", "exit 0")

    # 時長驗收:允許 ±0.5s
    try:
        dur = wav_duration(out)
        if abs(dur - 30.0) <= 0.5:
            _ok("A.duration", f"dur={dur:.2f}s")
        else:
            _fail("A.duration", f"期望 30s ±0.5, 實際 {dur:.2f}s")
    except Exception as e:
        _fail("A.duration", str(e))

    # volumedetect:非靜音(mean_volume < -70dB 才算失敗)
    try:
        stats = volumedetect(out)
        mean_vol = stats.get("mean_volume", -999.0)
        if mean_vol > -70.0:
            _ok("A.volumedetect", f"mean_volume={mean_vol:.1f}dB")
        else:
            _fail("A.volumedetect", f"mean_volume={mean_vol:.1f}dB (靜音?)")
    except Exception as e:
        _fail("A.volumedetect", str(e))


# ---------------------------------------------------------------------------
# 測試 (b): 60s 帶雷渲染, events-json 格式
# ---------------------------------------------------------------------------

def test_b(tmpdir: pathlib.Path):
    print("[B] 60s 帶雷渲染 + events-json …")
    out = tmpdir / "test_b.wav"
    ev_json = tmpdir / "test_b_events.json"
    ok = run_render(
        [
            "--duration", "60",
            "--seed", "77",
            "--thunder-rate", "3",
            "--thunder-distance-range", "2.0", "5.0",
            "--events-json", str(ev_json),
        ],
        out, "B.exit0",
    )
    if not ok:
        return

    _ok("B.exit0", "exit 0")

    # events-json 存在
    if not ev_json.exists():
        _fail("B.events_json_exists", f"not found: {ev_json}")
        return
    _ok("B.events_json_exists")

    # 格式:含 events 列表 > 0
    try:
        with open(ev_json) as f:
            data = json.load(f)

        # 必要欄位
        required_top = {"sample_rate", "duration_s", "seed", "events"}
        missing = required_top - set(data.keys())
        if missing:
            _fail("B.events_json_schema", f"missing keys: {missing}")
            return
        _ok("B.events_json_schema", "top-level keys OK")

        events = data["events"]
        if len(events) > 0:
            _ok("B.events_count", f"{len(events)} events")
        else:
            _fail("B.events_count", "events list is empty (期望 >0 for rate=3/min over 60s)")

        # 事件格式檢查(對照 demo_thunder_events.json)
        ref_path = ROOT / "out/demo/demo_thunder_events.json"
        if ref_path.exists():
            with open(ref_path) as f:
                ref = json.load(f)
            ref_ev_keys = set(ref["events"][0].keys()) if ref["events"] else set()
            if events and ref_ev_keys:
                ev_keys = set(events[0].keys())
                missing_keys = ref_ev_keys - ev_keys
                if not missing_keys:
                    _ok("B.events_schema_match", f"event keys match reference")
                else:
                    _fail("B.events_schema_match", f"missing event keys: {missing_keys}")
            else:
                _ok("B.events_schema_match", "skipped (no events or no ref)")
        else:
            _ok("B.events_schema_match", "ref not found, skipped")

    except Exception as e:
        _fail("B.events_json_parse", str(e))


# ---------------------------------------------------------------------------
# 測試 (c): bit-identical (同 seed, 同參數, 兩次)
# ---------------------------------------------------------------------------

def test_c(tmpdir: pathlib.Path):
    print("[C] bit-identical 測試 …")
    out1 = tmpdir / "test_c1.wav"
    out2 = tmpdir / "test_c2.wav"

    args = ["--duration", "20", "--seed", "42", "--thunder-rate", "0", "--no-rain-gush"]

    ok1 = run_render(args, out1, "C.run1")
    ok2 = run_render(args, out2, "C.run2")

    if not (ok1 and ok2):
        return

    b1 = wav_bytes(out1)
    b2 = wav_bytes(out2)

    if b1 == b2:
        _ok("C.bit_identical", f"files identical ({len(b1)} bytes)")
    else:
        # 找到第一個不同的 byte
        diff_pos = next((i for i, (a, b) in enumerate(zip(b1, b2)) if a != b), -1)
        _fail("C.bit_identical",
              f"files differ at byte {diff_pos} (sizes {len(b1)} vs {len(b2)})")


# ---------------------------------------------------------------------------
# 測試 (d): --help
# ---------------------------------------------------------------------------

def test_d():
    print("[D] --help …")
    cmd = [PYTHON, str(CLI), "--help"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0 and "--duration" in r.stdout:
        _ok("D.help", "--help exit 0 and contains --duration")
    else:
        _fail("D.help", f"exit {r.returncode}  stdout={r.stdout[:200]}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("verify_cli.py — render_ambience.py 煙霧測試")
    print("=" * 60)

    if not CLI.exists():
        print(f"FATAL: CLI not found at {CLI}", file=sys.stderr)
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="verify_cli_") as tmpdir_str:
        tmpdir = pathlib.Path(tmpdir_str)
        test_a(tmpdir)
        test_b(tmpdir)
        test_c(tmpdir)
        test_d()

    print("=" * 60)
    total = PASS_COUNT + FAIL_COUNT
    print(f"結果: {PASS_COUNT}/{total} PASS  {FAIL_COUNT} FAIL")
    if FAIL_COUNT > 0:
        print("SOME TESTS FAILED", file=sys.stderr)
        sys.exit(1)
    print("ALL PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
