#!/usr/bin/env python3
"""
verify_m3.py — M3 acceptance gate: CC0 asset fetch & curation
Exit 0 = PASS, Exit 1 = FAIL (with details)

Checks:
  1. oga_manifest.json exists and all entries have non-null SHA256 for downloaded files
  2. SHA256 values match actual raw files on disk
  3. licenses.csv exists, every row has non-empty evidence_url and license=CC0 or Public Domain
  4. curated/ contains ≥8 bird/insect WAV files
  5. Each curated WAV passes: duration ≥5s, sample_rate ≥22050Hz, mean_volume > -60dB
"""

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MANIFEST = PROJECT_ROOT / "assets" / "fetch" / "oga_manifest.json"
LICENSES_CSV = PROJECT_ROOT / "assets" / "curated" / "licenses.csv"
RAW_DIR = PROJECT_ROOT / "assets" / "raw"
CURATED_DIR = PROJECT_ROOT / "assets" / "curated"

FFMPEG = Path(os.environ.get("FFMPEG_BIN") or shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = Path(os.environ.get("FFPROBE_BIN") or shutil.which("ffprobe") or "ffprobe")

FAILS = []
PASSES = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  FAIL: {msg}")


def ok(msg: str) -> None:
    PASSES.append(msg)
    print(f"  OK:   {msg}")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def ffprobe_duration_sr(path: Path) -> tuple[float, int]:
    """Return (duration_seconds, sample_rate)."""
    result = subprocess.run(
        [
            str(FFPROBE), "-v", "quiet",
            "-show_entries", "stream=duration,sample_rate",
            "-of", "default=noprint_wrappers=1",
            str(path),
        ],
        capture_output=True, text=True,
    )
    dur = 0.0
    sr = 0
    for line in result.stdout.splitlines():
        if line.startswith("duration="):
            try:
                dur = float(line.split("=")[1])
            except ValueError:
                pass
        elif line.startswith("sample_rate="):
            try:
                sr = int(line.split("=")[1])
            except ValueError:
                pass
    return dur, sr


def ffmpeg_mean_volume(path: Path) -> float:
    """Return mean_volume in dB (negative value). -999 on error."""
    # Note: volumedetect filter outputs to stderr even with -v error
    result = subprocess.run(
        [
            str(FFMPEG), "-v", "error",
            "-i", str(path),
            "-af", "volumedetect",
            "-f", "null", "/dev/null",
        ],
        capture_output=True, text=True,
    )
    # volumedetect lines come from ffmpeg's filter log, use loglevel info
    # Re-run without -v suppression to capture filter output
    result2 = subprocess.run(
        [
            str(FFMPEG),
            "-i", str(path),
            "-af", "volumedetect",
            "-f", "null", "/dev/null",
        ],
        capture_output=True, text=True,
    )
    for line in result2.stderr.splitlines():
        if "mean_volume" in line:
            try:
                return float(line.split("mean_volume:")[1].split("dB")[0].strip())
            except (IndexError, ValueError):
                pass
    return -999.0


# ── Check 1: manifest exists ──────────────────────────────────────────────────
print("\n[C1] oga_manifest.json exists")
if not MANIFEST.exists():
    fail("oga_manifest.json not found")
else:
    ok("oga_manifest.json found")
    with open(MANIFEST) as f:
        manifest = json.load(f)

    # ── Check 2: SHA256 matches raw files ──────────────────────────────────────
    print("\n[C2] SHA256 verification for downloaded raw files")
    sha_errors = 0
    sha_ok = 0
    for source in manifest.get("sources", []):
        for fe in source.get("files", []):
            stored = fe.get("sha256")
            raw_fname = fe.get("raw_filename")
            if stored is None:
                continue  # skip entries without sha256 (not yet downloaded)
            raw_path = RAW_DIR / raw_fname
            if not raw_path.exists():
                fail(f"Raw file missing: {raw_fname}")
                sha_errors += 1
                continue
            computed = sha256_file(raw_path)
            if computed != stored:
                fail(f"SHA256 mismatch: {raw_fname}\n    stored={stored}\n    disk  ={computed}")
                sha_errors += 1
            else:
                sha_ok += 1
    if sha_errors == 0:
        ok(f"All {sha_ok} SHA256 values match")
    else:
        fail(f"{sha_errors} SHA256 mismatches found")

# ── Check 3: licenses.csv ─────────────────────────────────────────────────────
print("\n[C3] licenses.csv integrity")
if not LICENSES_CSV.exists():
    fail("licenses.csv not found")
else:
    ok("licenses.csv found")
    csv_errors = 0
    csv_rows = 0
    ALLOWED_LICENSES = {"CC0", "Public Domain"}
    with open(LICENSES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            csv_rows += 1
            fname = row.get("檔名", "").strip()
            lic = row.get("授權", "").strip()
            evidence = row.get("證據URL", "").strip()
            if not evidence:
                fail(f"Empty evidence_url for row: {fname}")
                csv_errors += 1
            if lic not in ALLOWED_LICENSES:
                fail(f"License '{lic}' not in allowed set {ALLOWED_LICENSES} for: {fname}")
                csv_errors += 1
    if csv_errors == 0:
        ok(f"All {csv_rows} rows have valid license and non-empty evidence_url")
    else:
        fail(f"{csv_errors} issues in licenses.csv")

# ── Check 4 & 5: curated/ WAV files ──────────────────────────────────────────
print("\n[C4] curated/ contains ≥8 insect/bird WAV files (by category column in licenses.csv)")
wav_files = sorted(CURATED_DIR.glob("*.wav"))

# Build category map from licenses.csv
category_map: dict[str, str] = {}
if LICENSES_CSV.exists():
    with open(LICENSES_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row.get("檔名", "").strip()
            cat = row.get("category", "").strip()
            if fname:
                category_map[fname] = cat

insect_bird_wavs = [w for w in wav_files
                    if category_map.get(w.name, "") in ("insect", "bird")]
if len(insect_bird_wavs) < 8:
    fail(f"Only {len(insect_bird_wavs)} insect/bird WAV files in curated/ (need ≥8); "
         f"total WAVs={len(wav_files)}, excluded rain_other/unknown="
         f"{len(wav_files) - len(insect_bird_wavs)}")
else:
    ok(f"{len(insect_bird_wavs)} insect/bird WAV files found in curated/ "
       f"(total WAVs={len(wav_files)})")

print("\n[C5] Each curated WAV: duration≥5s, sample_rate≥22050Hz, mean_volume>-60dB")
probe_errors = 0
for wav in wav_files:
    dur, sr = ffprobe_duration_sr(wav)
    vol = ffmpeg_mean_volume(wav)

    issues = []
    if dur < 5.0:
        issues.append(f"duration={dur:.2f}s < 5s")
    if sr < 22050:
        issues.append(f"sample_rate={sr} < 22050")
    if vol <= -60.0:
        issues.append(f"mean_volume={vol:.1f}dB ≤ -60dB")

    if issues:
        fail(f"{wav.name}: {'; '.join(issues)}")
        probe_errors += 1
    else:
        ok(f"{wav.name}: {dur:.1f}s | {sr}Hz | {vol:.1f}dB")

if probe_errors == 0 and len(insect_bird_wavs) >= 8:
    ok("All curated WAVs pass ffprobe filter")

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"PASS: {len(PASSES)}  FAIL: {len(FAILS)}")
if FAILS:
    print("\nFailed checks:")
    for f in FAILS:
        print(f"  - {f}")
    print("\nResult: FAIL")
    sys.exit(1)
else:
    print("\nResult: ALL PASS ✓")
    sys.exit(0)
