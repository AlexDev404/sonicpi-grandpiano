"""Score the first 64 ms of notes against the real recordings.

For third-octave bands from 250 Hz to 12 kHz, in 4 ms frames:
  dist  = mean |synth - real| band level (dB), each file normalised to its own loudest frame/band
  rough = mean frame-to-frame jump (dB) in the synth minus the same for the real piano
          (positive = synth flickers more than a real piano at the attack)
Usage: python3 -I onset_score.py SYNTH_DIR PREFIX NOTE...
"""
import os
import sys
import numpy as np
from scipy.io import wavfile

REF = os.environ.get("REF_DIR", "/home/claude/ref")
CENTERS = [250 * 2 ** (i / 3) for i in range(0, 17)]       # 250 Hz .. ~10 kHz
FRAME = 0.004
N_FRAMES = 16


def load(p):
    sr, x = wavfile.read(p)
    x = x.astype(np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if np.abs(x).max() > 2:
        x /= 32768.0
    on = int(np.argmax(np.abs(x) > 0.02 * np.abs(x).max()))
    return sr, x[max(0, on - int(0.002 * sr)):]


def grid(sr, x):
    w = int(FRAME * 2 * sr)            # 8 ms windows, 4 ms hop
    hop = int(FRAME * sr)
    n = 4096
    f = np.fft.rfftfreq(n, 1 / sr)
    out = np.zeros((N_FRAMES, len(CENTERS)))
    for i in range(N_FRAMES):
        s = x[i * hop: i * hop + w]
        if len(s) < w:
            s = np.pad(s, (0, w - len(s)))
        sp = np.abs(np.fft.rfft(s * np.hanning(w), n)) ** 2
        for j, c in enumerate(CENTERS):
            m = (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
            out[i, j] = 10 * np.log10(sp[m].sum() + 1e-15)
    return out - out.max()


def main():
    sdir, prefix, names = sys.argv[1], sys.argv[2], sys.argv[3:]
    D, R = [], []
    for nm in names:
        sr, a = load(f"{REF}/{nm}.wav")
        _, b = load(f"{sdir}/{prefix}_{nm}.wav")
        ga, gb = grid(sr, a), grid(sr, b)
        mask = ga > -40
        dist = np.mean(np.abs(np.maximum(gb, -40) - ga)[mask])
        ra = np.mean(np.abs(np.diff(np.maximum(ga, -40), axis=0))[mask[1:]])
        rb = np.mean(np.abs(np.diff(np.maximum(gb, -40), axis=0))[mask[1:]])
        D.append(dist); R.append(rb - ra)
        print(f"{nm:>4}: dist {dist:5.1f} dB   flicker synth {rb:4.1f} vs real {ra:4.1f} dB/frame")
    print(f"MEAN dist {np.mean(D):.2f}  extra flicker {np.mean(R):+.2f}")


if __name__ == "__main__":
    main()
