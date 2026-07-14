"""Compare real rain recording vs synthesized rain: spectrum + temporal texture."""
import numpy as np
from scipy.io import wavfile
from scipy import signal

def load(path, max_s=None):
    sr, x = wavfile.read(path)
    if x.dtype == np.int16:
        x = x / 32768.0
    elif x.dtype == np.int32:
        x = x / 2147483648.0
    if x.ndim > 1:
        x = x.mean(axis=1)
    if max_s:
        x = x[: int(sr * max_s)]
    return sr, x.astype(np.float64)

def octave_bands(sr, x):
    f, pxx = signal.welch(x, sr, nperseg=8192)
    edges = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    out = []
    for i in range(len(edges) - 1):
        m = (f >= edges[i]) & (f < edges[i + 1])
        out.append(10 * np.log10(pxx[m].mean() + 1e-20))
    total = 10 * np.log10(pxx[(f > 20)].mean() + 1e-20)
    return edges, np.array(out) - total  # relative dB per band

def temporal_stats(sr, x):
    # envelope in the droplet-transient band (2-8 kHz)
    sos = signal.butter(4, [2000, 8000], "bandpass", fs=sr, output="sos")
    b = signal.sosfilt(sos, x)
    env = np.abs(signal.hilbert(b))
    # smooth to 5ms
    win = int(sr * 0.005)
    env_s = np.convolve(env, np.ones(win) / win, mode="valid")
    crest = 20 * np.log10(np.max(np.abs(b)) / (np.std(b) + 1e-12))
    kurt = float(((env_s - env_s.mean()) ** 4).mean() / (env_s.std() ** 4 + 1e-20))
    cv = float(env_s.std() / (env_s.mean() + 1e-12))
    # modulation spectrum peak 1-30 Hz of the envelope
    fm, pm = signal.welch(env_s - env_s.mean(), sr, nperseg=1 << 16)
    mmask = (fm > 0.5) & (fm < 30)
    return crest, kurt, cv

for name, path, cap in [
    ("REAL  (pinterest)", "/tmp/real_rain.wav", 39),
    ("SYNTH (loop 90s) ", str(ROOT / "out/loops/rain_loop_90s.wav"), 90),
]:
    sr, x = load(path, cap)
    edges, bands = octave_bands(sr, x)
    crest, kurt, cv = temporal_stats(sr, x)
    print(f"== {name}  sr={sr}")
    labels = ["31", "63", "125", "250", "500", "1k", "2k", "4k", "8k"]
    print("   octave rel dB: " + "  ".join(f"{l}:{b:+.1f}" for l, b in zip(labels, bands)))
    print(f"   2-8k crest={crest:.1f} dB  env-kurtosis={kurt:.1f}  env-CV={cv:.3f}")
