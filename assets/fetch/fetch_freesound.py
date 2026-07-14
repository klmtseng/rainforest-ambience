#!/usr/bin/env python3
"""
fetch_freesound.py — Optional: download CC0 sounds from Freesound.org
Requires env FREESOUND_TOKEN (obtain free at freesound.org/apiv2/apply/).
If token is absent, prints a skip message and exits 0.
Usage: FREESOUND_TOKEN=xxx python3 fetch_freesound.py
"""

import os
import sys
import json
import hashlib
import urllib.request
import urllib.parse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
ASSETS_DIR = SCRIPT_DIR.parent
RAW_DIR = ASSETS_DIR / "raw"
FREESOUND_MANIFEST = SCRIPT_DIR / "freesound_manifest.json"

TOKEN = os.environ.get("FREESOUND_TOKEN", "")

# Search queries: only CC0 licensed sounds
SEARCHES = [
    {"query": "crickets night", "filter": "license:\"Creative Commons 0\"", "max": 5},
    {"query": "bird forest ambient", "filter": "license:\"Creative Commons 0\"", "max": 5},
]

BASE_URL = "https://freesound.org/apiv2"


def api_get(path: str, params: dict) -> dict:
    params["token"] = TOKEN
    url = f"{BASE_URL}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "rainforest-ambience/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not TOKEN:
        print(
            "[freesound] FREESOUND_TOKEN not set — skipping Freesound download.\n"
            "To enable: register at https://freesound.org/apiv2/apply/ "
            "and re-run with FREESOUND_TOKEN=<your_token>"
        )
        return 0

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for search in SEARCHES:
        print(f"\nSearching Freesound: {search['query']!r} (CC0 only)")
        try:
            data = api_get(
                "/search/text/",
                {
                    "query": search["query"],
                    "filter": search["filter"],
                    "fields": "id,name,license,previews,download,duration,samplerate",
                    "page_size": search["max"],
                },
            )
        except Exception as exc:
            print(f"  [ERROR] Search failed: {exc}", file=sys.stderr)
            continue

        for sound in data.get("results", []):
            sid = sound["id"]
            name = sound["name"]
            dur = sound.get("duration", 0)
            sr = sound.get("samplerate", 0)
            lic = sound.get("license", "")
            print(f"  {sid}: {name!r} | {dur:.1f}s | {sr}Hz | {lic}")

            # Only keep CC0, ≥5s, ≥22050Hz (match curated criteria)
            if dur < 5.0 or sr < 22050:
                print(f"    [skip] fails duration/samplerate filter")
                continue

            # Download preview (HQ preview doesn't need OAuth)
            preview_url = sound.get("previews", {}).get("preview-hq-ogg", "")
            if not preview_url:
                print(f"    [skip] no HQ preview")
                continue

            safe_name = f"freesound_{sid}_{name[:40].replace('/', '_').replace(' ', '_')}.ogg"
            dest = RAW_DIR / safe_name
            if not dest.exists():
                print(f"    [dl] {safe_name}")
                try:
                    req = urllib.request.Request(
                        preview_url,
                        headers={"User-Agent": "rainforest-ambience/1.0"},
                    )
                    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as out:
                        while chunk := resp.read(65536):
                            out.write(chunk)
                except Exception as exc:
                    print(f"    [ERROR] {exc}", file=sys.stderr)
                    continue
            else:
                print(f"    [skip] already exists")

            results.append(
                {
                    "id": sid,
                    "name": name,
                    "license": "CC0",
                    "evidence_url": f"https://freesound.org/people/sound/{sid}/",
                    "duration": dur,
                    "samplerate": sr,
                    "raw_filename": safe_name,
                    "sha256": sha256_file(dest) if dest.exists() else None,
                }
            )

    with open(FREESOUND_MANIFEST, "w") as f:
        json.dump({"sources": results}, f, indent=2)
    print(f"\nFreesound manifest written: {FREESOUND_MANIFEST}")
    print(f"Total qualifying downloads: {len(results)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
