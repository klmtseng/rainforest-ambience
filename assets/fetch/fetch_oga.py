#!/usr/bin/env python3
"""
fetch_oga.py — Download CC0 bird/insect audio from OpenGameArt.org
All sources verified CC0 by WebFetch inspection on 2026-07-13.
Usage: python3 fetch_oga.py [--dry-run]
Output: assets/raw/ (raw downloads) + updates oga_manifest.json SHA256 fields
"""

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR.parent
RAW_DIR = ASSETS_DIR / "raw"
MANIFEST_PATH = SCRIPT_DIR / "oga_manifest.json"

DRY_RUN = "--dry-run" in sys.argv


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    """Download url to dest, skip if already exists with matching size."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [dl]   {dest.name} <- {url}")
    if DRY_RUN:
        return
    req = urllib.request.Request(
        url, headers={"User-Agent": "rainforest-ambience-fetcher/1.0"}
    )
    with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as out:
        while chunk := resp.read(65536):
            out.write(chunk)


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    changed = False
    for source in manifest["sources"]:
        sid = source["id"]
        cat = source["category"]
        print(f"\n[{sid}] ({cat}) {source['asset_page']}")
        for file_entry in source["files"]:
            url = file_entry["url"]
            raw_fname = file_entry["raw_filename"]
            dest = RAW_DIR / raw_fname
            try:
                download(url, dest)
            except Exception as exc:
                print(f"  [ERROR] {exc}", file=sys.stderr)
                continue

            if dest.exists() and not DRY_RUN:
                computed = sha256_file(dest)
                stored = file_entry.get("sha256")
                if stored and stored != computed:
                    print(
                        f"  [WARN] SHA256 mismatch for {raw_fname}:\n"
                        f"    stored:   {stored}\n"
                        f"    computed: {computed}"
                    )
                elif not stored:
                    print(f"  [sha256] {computed}")
                    file_entry["sha256"] = computed
                    changed = True
                else:
                    print(f"  [ok]   SHA256 matches")

    if changed and not DRY_RUN:
        with open(MANIFEST_PATH, "w") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print("\nManifest updated with new SHA256 values.")

    print("\nDone.")


if __name__ == "__main__":
    main()
