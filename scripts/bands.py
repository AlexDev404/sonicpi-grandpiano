"""Octave-band level comparison, real vs synth, over time.

Usage: python3 -I bands.py SYNTH_DIR PREFIX [NOTE_NAMES...]
  SYNTH_DIR/PREFIX_<NOTE>.wav are compared with /home/claude/ref/<NOTE>.wav
Prints per-note mean absolute band error (dB) at each time, and band detail with -v.
Levels are normalised to each file's own total level at 0-50 ms, so only spectral
balance and decay shape are compared (not absolute loudness).
"""
import sys
import numpy as np
from scipy.io import wavfile

BANDS = [(22, 45), (45, 90), (90, 180), (180, 355), (355, 710), (710, 1400),
         (1400, 2800), (2800, 5600), (5600, 11200), (11200, 18000)]
TIMES = [(0.0, 0.05), (0.1, 0.25), (0.4, 0.8), (1.0, 2.0), (2.5, 4.0)]


def load(p):
    sr, x = wavfile.read(p)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 2:
        x /= 32768.0
    on = int(np.argmax(np.abs(x) > 0.05 * np.abs(x).max()))
    return sr, x[max(0, on - 22):]


def band_levels(sr, x, t0, t1):
    s = x[int(t0 * sr): int(t1 * sr)]
    n = 1 << int(np.ceil(np.log2(max(len(s), 2048))))
    win = np.hanning(len(s))
    sp = np.abs(np.fft.rfft(s * win, n)) ** 2 / (win ** 2).sum()   # power per sample
    f = np.fft.rfftfreq(n, 1 / sr)
    return np.array([10 * np.log10(sp[(f >= a) & (f < b)].sum() + 1e-12) for a, b in BANDS])


NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
         "G#": 8, "A": 9, "A#": 10, "B": 11}


def f0_of(name):
    n = 12 * (int(name[-1]) + 1) + NAMES[name[:-1]]
    return 440 * 2 ** ((n - 69) / 12)


def main():
    sdir, prefix = sys.argv[1], sys.argv[2]
    names = [a for a in sys.argv[3:] if a != "-v"]
    verbose = "-v" in sys.argv
    summary = []
    for nm in names:
        sr, a = load(f"{__import__('os').environ.get('REF_DIR', '/home/claude/ref')}/{nm}.wav")
        _, b = load(f"{sdir}/{prefix}_{nm}.wav")
        # recording noise floor per band: quietest 0.5 s stretch of the whole file
        nf = np.min([band_levels(sr, a, t, t + 0.5) for t in np.arange(0.5, len(a) / sr - 0.5, 0.5)], axis=0)
        ra = band_levels(sr, a, 0, 0.05)
        rb = band_levels(sr, b, 0, 0.05)
        ref_a = 10 * np.log10((10 ** (ra / 10)).sum())
        ref_b = 10 * np.log10((10 ** (rb / 10)).sum())
        errs = []
        rows = []
        above_f0 = np.array([b > 0.9 * f0_of(nm) for _, b in BANDS])   # skip hum below the note
        for t0, t1 in TIMES:
            la = band_levels(sr, a, t0, t1) - ref_a
            lb = band_levels(sr, b, t0, t1) - ref_b
            floor = la.max() - 40         # only bands within 40 dB of the loudest
            relevant = (la > floor) & (la + ref_a > nf + 6) & above_f0   # above noise, not hum
            d = (np.maximum(lb, floor) - la)[relevant]
            errs.append(np.mean(np.abs(d)) if len(d) else 0)
            rows.append((t0, t1, la, lb))
        summary.append((nm, errs))
        print(f"{nm:>4} mean |err| dB per time window:", " ".join(f"{e:5.1f}" for e in errs),
              f"  overall {np.mean(errs):5.1f}")
        if verbose:
            print("      band Hz:      " + " ".join(f"{a:>6d}" for a, _ in BANDS))
            for t0, t1, la, lb in rows:
                print(f"   {t0:3.1f}-{t1:3.1f}s real " + " ".join(f"{v:6.1f}" for v in la))
                print(f"            synth" + " ".join(f"{v:6.1f}" for v in lb))
    print("MEAN over notes:", round(float(np.mean([np.mean(e) for _, e in summary])), 2))


if __name__ == "__main__":
    main()
