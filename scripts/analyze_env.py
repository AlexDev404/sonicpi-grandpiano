"""Measure each partial's level (dB relative to its own peak) at fixed checkpoint times.

Usage: python3 -I analyze_env.py REF_DIR ANALYSIS_JSON OUT_JSON
Uses partial frequencies already found by analyze.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import load, find_onset, envelope, ENV_RATE  # noqa: E402

CHECK = [0.0, 0.05, 0.15, 0.4, 1.0, 2.0, 4.0, 8.0, 16.0]
MIN_DB = -100.0


def checkpoint_db(env):
    pk = int(np.argmax(env[: int(0.15 * ENV_RATE)]))
    e = env[pk:]
    peak = e[0]
    floor = max(np.median(env[-ENV_RATE:]) * 1.5, peak * 1e-5)
    db = 20 * np.log10(np.maximum(e, 1e-12) / peak)
    valid_end = np.where(e > floor * 2)[0]
    last = valid_end[-1] if len(valid_end) else 0
    out = []
    for t in CHECK:
        i = int(t * ENV_RATE)
        w = max(1, int(0.15 * t * ENV_RATE))           # average over +-15 % of t
        if i + w <= last:
            seg = db[max(0, i - w): i + w + 1]
            out.append(float(np.mean(seg)))
        else:
            out.append(None)                             # below the noise floor here
    # extrapolate past the noise floor using the slope of the last good stretch
    good = [j for j, v in enumerate(out) if v is not None]
    if not good:
        return [0.0] + [MIN_DB] * (len(CHECK) - 1)
    g = good[-1]
    if g >= 1:
        t0, t1 = CHECK[good[-2]] if len(good) > 1 else 0.0, CHECK[g]
        v0 = out[good[-2]] if len(good) > 1 else 0.0
        slope = min((out[g] - v0) / max(t1 - t0, 1e-3), -1.0)   # dB per second, at least -1
    else:
        slope = -20.0
    for j in range(g + 1, len(CHECK)):
        out[j] = max(MIN_DB, out[g] + slope * (CHECK[j] - CHECK[g]))
    out[0] = 0.0
    return out


def main():
    ref, analysis, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    data = json.loads(analysis.read_text())
    result = []
    for d in data:
        sr, x = load(ref / f"{d['name']}.wav")
        x = x[find_onset(x, sr):]
        bw = max(8.0, min(40.0, d["fund"] / 3))
        parts = {}
        for p in d["partials"]:
            env = envelope(x, sr, p["freq"], bw)
            parts[p["k"]] = checkpoint_db(env)
        result.append(dict(note=d["note"], name=d["name"], env=parts))
        c = parts.get(1)
        print(d["name"], "k1:", [round(v) for v in c] if c else None)
    out.write_text(json.dumps(dict(check=CHECK, notes=result)))


if __name__ == "__main__":
    main()
