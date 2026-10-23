"""Measure the first 40 ms of notes: crest factor and arrival time per frequency band.

A phase-coherent start of many inharmonic partials shows up as a high crest factor and as a
systematic spread of arrival times across bands (a chirp, heard as a 'zap').
Usage: python3 -I onset.py LABEL=FILE.wav ...
"""
import sys
import numpy as np
from scipy.io import wavfile
from scipy.signal import butter, sosfiltfilt, hilbert

BANDS = [(150, 400), (400, 1000), (1000, 2500), (2500, 6000)]


def load(p):
    sr, x = wavfile.read(p)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 2:
        x /= 32768.0
    on = int(np.argmax(np.abs(x) > 0.02 * np.abs(x).max()))
    return sr, x[max(0, on - int(0.005 * sr)): on + int(0.06 * sr)]


for arg in sys.argv[1:]:
    label, path = arg.split("=", 1)
    sr, x = load(path)
    seg = x[: int(0.045 * sr)]
    crest = 20 * np.log10(np.abs(seg).max() / (np.sqrt((seg ** 2).mean()) + 1e-12))
    arrivals = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        env = np.abs(hilbert(sosfiltfilt(sos, x)))
        pk = env.max()
        arrivals.append(1000 * np.argmax(env > 0.5 * pk) / sr if pk > 1e-6 else float("nan"))
    hf = []
    for lo, hi in BANDS:
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        y = sosfiltfilt(sos, seg)
        hf.append(10 * np.log10((y ** 2).mean() + 1e-15))
    hf = np.array(hf) - max(hf)
    print(f"{label:>14}: crest {crest:5.1f} dB | half-level arrival ms per band "
          + " ".join(f"{a:5.1f}" for a in arrivals)
          + " | band energy dB " + " ".join(f"{v:6.1f}" for v in hf))
