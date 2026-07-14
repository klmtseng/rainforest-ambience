# rainforest-ambience

A procedural rainforest night ambience generator. Produces hours of non-repeating, parametric rain, thunder, and critter audio — entirely from synthesised DSP and CC0 field recordings, with no proprietary samples.

## What it does

- **Rain synthesis** — statistically matched to real recordings using multi-band spectral shaping, log-normal drop amplitude distribution, and OU-process intensity drift
- **Thunder synthesis** — physically-modelled rumble + crack + convolution reverb, distance-scaled and event-scheduled with exponential inter-arrival times
- **Critter layer** — 8 CC0 insect and bird recordings, dynamically layered when rain intensity is low
- **McDermott perceptual framework** — synthesis targets match human auditory texture perception: band-envelope CV, inter-band Pearson correlation, and spectral tilt
- **Machine gate verified** — every milestone gated by `tests/verify_*.py` with frozen thresholds (exit 0 = pass)
- **Bit-identical reproducibility** — same seed always produces identical output (uses `numpy.random.default_rng`)

## Hardware requirements

Python 3.12+, numpy, scipy, ffmpeg — runs on a CPU-only laptop (tested on i5-3210M, no GPU, no AVX2).

## Quick start

```bash
# Install dependencies
pip install numpy scipy

# Set ffmpeg path (or add ffmpeg to PATH)
export FFMPEG_BIN=/usr/bin/ffmpeg   # or wherever ffmpeg lives

# 5 minutes of light rain, no thunder
python render_ambience.py --duration 300 --thunder-rate 0 --out /tmp/rain.wav

# 30 minutes of thunderstorm with critters
python render_ambience.py --duration 1800 --seed 42 --thunder-rate 2 --critters --out /tmp/storm.wav

# 10 seconds quick preview + MP3
python render_ambience.py --duration 10 --seed 77 --thunder-rate 3 --mp3 --out /tmp/test.wav
```

## CLI parameters

| Flag | Default | Description |
|---|---|---|
| `--duration` | 300.0 | Output length in seconds (1800 = 30 min, 2700 = 45 min) |
| `--seed` | 77 | Random seed; same seed + same params → bit-identical WAV |
| `--rain-intensity` | 0.55 | OU mean intensity [0–1]; higher = heavier rain |
| `--rain-variability` | 0.08 | OU sigma; 0 = steady, 0.15 = heavy fluctuation |
| `--thunder-rate` | 2.0 | Expected thunder strikes per minute; 0 = silent |
| `--thunder-distance-range KM_MIN KM_MAX` | 2.0 8.0 | Distance range in km; closer = louder, more low-end |
| `--rain-gush / --no-rain-gush` | enabled | Brief rain intensity boost after close thunder |
| `--critters / --no-critters` | disabled | Enable insect/bird layer (appears when rain is light) |
| `--out` | /tmp/ambience.wav | Output WAV path (float32, 48 kHz) |
| `--mp3` | off | Also encode MP3 at 192 kbps alongside WAV |
| `--stems` | off | Write separate rain/thunder/critters stems |
| `--events-json PATH` | off | Write thunder event table (time, distance, intensity) |
| `--drop-rate-scale` | preset | Scale factor on raindrop density |
| `--drop-amp-sigma` | preset | Log-normal sigma for drop amplitude variation |
| `--spectral-tilt-hi` | preset | High-frequency slope in dB/octave |
| `--spectral-tilt-lo` | preset | Low-frequency slope in dB/octave |
| `--bed-mix` | preset | Bed-to-drop mix ratio [0–1] |

## Interactive tuning panel

```bash
python panel/server.py
# Open http://localhost:8765/ in a browser
```

The panel renders 4–30 second previews in real time as you adjust sliders. Presets can be saved to `config/`.

## Environment variables

| Variable | Purpose |
|---|---|
| `FFMPEG_BIN` | Path to ffmpeg binary (falls back to `which ffmpeg`) |
| `FFPROBE_BIN` | Path to ffprobe binary (falls back to `which ffprobe`) |

## Output format

- 48000 Hz, float32 WAV (IEEE PCM_FLOAT; readable by ffmpeg, Audacity, DAWs)
- Internal processing in float64
- loudnorm target: −23 LUFS

## Project structure

```
render_ambience.py   single CLI entry point
synth/               DSP modules (rain_v4, thunder_v2, wavio, walk)
mix/                 mixdown pipeline, looper, demo mix
panel/               browser-based tuning UI
config/              JSON presets
assets/curated/      8 CC0 insect/bird recordings + licenses.csv
assets/fetch/        scripts to re-download CC0 assets
tests/               machine-gate verification scripts (verify_cli.py, verify_rain_v4.py, …)
video/               optional video render pipeline (ffmpeg + Blender)
```

## Running tests

```bash
export FFMPEG_BIN=/usr/bin/ffmpeg

# Core smoke test (exit 0 = all pass)
python tests/verify_cli.py

# Perceptual quality gates
python tests/verify_rain_v4.py
```

## Asset licenses

All audio assets in `assets/curated/` are CC0 (public domain). See `assets/curated/licenses.csv` for per-file source URLs and attribution evidence.

## License

Code: MIT License — see `LICENSE`.

Audio assets (`assets/curated/`): CC0 / Public Domain — see `assets/curated/licenses.csv`.
