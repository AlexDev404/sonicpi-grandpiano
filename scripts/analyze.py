"""Measure per-partial frequency, level and two-stage decay from piano recordings.

Usage: python3 -I analyze.py REF_DIR OUT_JSON
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.optimize import curve_fit
from scipy.signal import fftconvolve

NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
         "G#": 8, "A": 9, "A#": 10, "B": 11}
MAX_PARTIALS = int(__import__('os').environ.get('MAX_PARTIALS', '96'))
ENV_RATE = 200          # envelope samples per second


def note_num(name):
    pitch, octave = name[:-1], int(name[-1])
    return 12 * (octave + 1) + NAMES[pitch]


def load(path):
    sr, x = wavfile.read(path)
    x = x.astype(np.float64)
    if x.dtype.kind == "i" or np.abs(x).max() > 2:
        x /= 32768.0
    return sr, x


def find_onset(x, sr):
    env = np.abs(x)
    thr = 0.05 * env.max()
    i = int(np.argmax(env > thr))
    return max(0, i - int(0.002 * sr))


def peak_near(spec, freqs, target, tol):
    lo, hi = np.searchsorted(freqs, [target * (1 - tol), target * (1 + tol)])
    if hi <= lo + 1:
        return None, 0.0
    j = lo + int(np.argmax(spec[lo:hi]))
    # parabolic interpolation
    if 0 < j < len(spec) - 1:
        a, b, c = np.log(spec[j - 1] + 1e-12), np.log(spec[j] + 1e-12), np.log(spec[j + 1] + 1e-12)
        p = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) != 0 else 0
        return freqs[j] + p * (freqs[1] - freqs[0]), spec[j]
    return freqs[j], spec[j]


def find_partials(x, sr, f0):
    seg = x[int(0.03 * sr): int(0.03 * sr) + int(1.0 * sr)]
    n = 1 << int(np.ceil(np.log2(len(seg) * 4)))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n))
    freqs = np.fft.rfftfreq(n, 1 / sr)
    noise = np.median(spec)
    found = []
    B = 0.0
    fund = f0
    for k in range(1, MAX_PARTIALS + 1):
        target = k * fund * np.sqrt(1 + B * k * k)
        if target > min(sr / 2 * 0.9, 16000):
            break
        f, mag = peak_near(spec, freqs, target, 0.03 if k > 1 else 0.04)
        if f is None or mag < noise * 30:
            continue
        found.append((k, f, mag))
        if k == 1:
            fund = f
        # refine inharmonicity once we have a few partials
        if len(found) >= 3:
            ks = np.array([p[0] for p in found], float)
            fs = np.array([p[1] for p in found])
            # (f_k / k)^2 = f0^2 + f0^2 B k^2  -> linear in k^2
            yy = (fs / ks) ** 2
            A = np.vstack([np.ones_like(ks), ks ** 2]).T
            c0, c1 = np.linalg.lstsq(A, yy, rcond=None)[0]
            if c0 > 0:
                fund = np.sqrt(c0)
                B = max(0.0, c1 / c0)
    return fund, B, found


def envelope(x, sr, f, bw):
    t = np.arange(len(x)) / sr
    z = x * np.exp(-2j * np.pi * f * t)
    win = int(sr / bw)
    kernel = np.hanning(win)
    kernel /= kernel.sum()
    env = 2 * np.abs(fftconvolve(z, kernel, mode="same"))
    step = sr // ENV_RATE
    return env[::step]


def two_exp(t, a1, tau1, a2, tau2):
    return a1 * np.exp(-t / tau1) + a2 * np.exp(-t / tau2)


def fit_decay(env):
    t = np.arange(len(env)) / ENV_RATE
    pk = int(np.argmax(env[: int(0.15 * ENV_RATE)]))
    e = env[pk:]
    t = t[: len(e)]
    floor = max(np.median(env[-ENV_RATE:]) * 2, e[0] * 1e-3)
    usable = np.where(e > floor)[0]
    end = usable[-1] if len(usable) else len(e) - 1
    end = max(end, int(0.3 * ENV_RATE))
    e, t = e[:end], t[:end]
    a0 = e[0]
    w = 1.0 / np.maximum(e, a0 * 1e-3)   # fit in roughly relative error
    try:
        p, _ = curve_fit(two_exp, t, e, p0=[a0 * 0.6, 0.3, a0 * 0.4, 3.0],
                         bounds=([0, 0.01, 0, 0.05], [a0 * 3, 5, a0 * 3, 60]),
                         sigma=1 / w, maxfev=20000)
    except Exception:
        p = [a0, 1.0, 0.0, 1.0]
    a1, t1, a2, t2 = p
    if t1 > t2:                        # keep stage 1 as the fast one
        a1, t1, a2, t2 = a2, t2, a1, t1
    return dict(peak=float(a0), a_fast=float(a1), tau_fast=float(t1),
                a_slow=float(a2), tau_slow=float(t2), attack_s=float(pk / ENV_RATE))


def analyze_file(path):
    name = path.stem
    n = note_num(name)
    f0 = 440 * 2 ** ((n - 69) / 12)
    sr, x = load(path)
    x = x[find_onset(x, sr):]
    fund, B, found = find_partials(x, sr, f0)
    bw = max(8.0, min(40.0, fund / 3))
    parts = []
    for k, f, _ in found:
        env = envelope(x, sr, f, bw)
        d = fit_decay(env)
        d.update(k=k, freq=float(f), ratio=float(f / fund))
        parts.append(d)
    return dict(name=name, note=n, f0_nominal=f0, fund=float(fund), B=float(B),
                partials=parts)


def main():
    ref, out = Path(sys.argv[1]), Path(sys.argv[2])
    results = []
    for p in sorted(ref.glob("*.wav"), key=lambda p: note_num(p.stem)):
        r = analyze_file(p)
        results.append(r)
        top = sorted(r["partials"], key=lambda q: -q["peak"])[:4]
        print(f"{r['name']:>4} n={r['note']:3d} B={r['B']:.2e} partials={len(r['partials']):2d} "
              f"loudest k={[q['k'] for q in top]} "
              f"k1 tau=({r['partials'][0]['tau_fast']:.2f},{r['partials'][0]['tau_slow']:.2f})"
              if r["partials"] else r["name"])
    out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
