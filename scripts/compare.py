"""Compare a rendered note with the real recording: level envelope and brightness over time.

Usage: python3 -I compare.py REAL.wav SYNTH.wav [synth_start_seconds]
"""
import sys
import numpy as np
from scipy.io import wavfile


def load(p, start=0.0):
    sr, x = wavfile.read(p)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 2:
        x /= 32768.0
    x = x[int(start * sr):]
    on = int(np.argmax(np.abs(x) > 0.05 * np.abs(x).max()))
    return sr, x[on:]


def stats(sr, x, t, w=0.05):
    s = x[int(t * sr): int((t + w) * sr)]
    if len(s) < 16:
        return -120.0, 0.0
    rms = np.sqrt((s ** 2).mean()) + 1e-12
    sp = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    f = np.fft.rfftfreq(len(s), 1 / sr)
    cen = (sp * f).sum() / (sp.sum() + 1e-20)
    return 20 * np.log10(rms), cen


def main():
    real, synth = sys.argv[1], sys.argv[2]
    start = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    sr1, a = load(real)
    sr2, b = load(synth, start)
    pa, pb = stats(sr1, a, 0.0, 0.03)[0], stats(sr2, b, 0.0, 0.03)[0]
    print(f"{'t(s)':>6} {'real dB':>8} {'synth dB':>9} {'real Hz':>8} {'synth Hz':>9}  (dB relative to each attack)")
    for t in (0.0, 0.1, 0.3, 0.6, 1.0, 2.0, 3.0, 4.0):
        ra, ca = stats(sr1, a, t)
        rb, cb = stats(sr2, b, t)
        print(f"{t:6.1f} {ra - pa:8.1f} {rb - pb:9.1f} {ca:8.0f} {cb:9.0f}")
    print(f"absolute attack level: real {pa:.1f} dBFS, synth {pb:.1f} dBFS")


if __name__ == "__main__":
    main()
