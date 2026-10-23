"""Build an analysis-based additive piano SynthDef for Sonic Pi.

Usage: python3 -I build_grandpiano.py DATA_DIR REF_DIR_PREFIX OUT_DIR
  DATA_DIR holds analysis_v2.json, analysis.json (v6), analysis_v12.json,
           env_v2.json, env_v6.json, env_v12.json and (optionally) noise.json
  REF_DIR_PREFIX e.g. ./ref  (reads ref_v2/, ref/, ref_v12/ WAVs to measure hammer
           noise; only needed when DATA_DIR has no noise.json yet)
  ENV_SMOOTH env var: smoothing width in semitones for the envelopes (default 2.0)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from supriya import Envelope, synthdef
from supriya.ugens import (
    BPF, DetectSilence, EnvGen, LPF, HPF, Out, Pan2, PinkNoise, Select, SinOsc,
    WhiteNoise, Mix,
)

K = 16                                   # partials per note
NOTES = list(range(21, 109, 3))          # A0 .. C8, 30 table points
VEL = {"v2": 2 / 16, "v6": 6 / 16, "v12": 12 / 16}
FLOOR_DB = -90.0
ENV_SMOOTH = float(__import__('os').environ.get('ENV_SMOOTH', '2.0'))
SMOOTH_ST = 4.0                          # smoothing width in semitones


# ---------------------------------------------------------------- tables ---
def load_layer(path):
    data = json.loads(Path(path).read_text())
    return {d["note"]: d for d in data}


def raw_tables(layer):
    """Arrays [note_idx, k] of log-amp, fast fraction, log tau fast/slow, plus validity."""
    n = len(NOTES)
    la = np.full((n, K), np.nan)
    ff = np.full((n, K), np.nan)
    ltf = np.full((n, K), np.nan)
    lts = np.full((n, K), np.nan)
    B = np.full(n, np.nan)
    for i, note in enumerate(NOTES):
        d = layer.get(note)
        if not d:
            continue
        if d["B"] > 0:
            B[i] = math.log(d["B"])
        f0 = d["fund"]
        for p in d["partials"]:
            k = p["k"]
            if k > K:
                continue
            la[i, k - 1] = math.log(max(p["peak"], 1e-9))
            tot = p["a_fast"] + p["a_slow"]
            if tot <= 0:
                continue
            frac = p["a_fast"] / tot
            tf, ts = p["tau_fast"], p["tau_slow"]
            sane = 0.012 < tf < 4.9 and 0.06 < ts < 59 and ts > tf * 1.5
            if sane:
                ff[i, k - 1] = min(max(frac, 0.02), 0.98)
                ltf[i, k - 1] = math.log(tf)
                lts[i, k - 1] = math.log(ts)
        # partials that should exist (below 16 kHz) but were not detected -> very quiet
        for k in range(1, K + 1):
            if np.isnan(la[i, k - 1]) and k * f0 < 16000:
                la[i, k - 1] = math.log(10 ** (FLOOR_DB / 20))
    return la, ff, ltf, lts, B


def smooth_rows(tab, robust=True, width=None):
    """Smooth each partial's column across notes with outlier-robust Gaussian weights."""
    out = np.empty_like(tab)
    x = np.array(NOTES, float)
    for k in range(tab.shape[1]):
        y = tab[:, k]
        ok = ~np.isnan(y)
        for i, n in enumerate(x):
            if ok.sum() == 0:
                out[i, k] = np.nan
                continue
            w = np.exp(-0.5 * ((x[ok] - n) / (width or SMOOTH_ST)) ** 2)
            yy = y[ok]
            if robust and ok.sum() >= 4:
                med = np.average(yy, weights=w)
                dev = np.abs(yy - med)
                mad = np.average(dev, weights=w) + 1e-6
                w = w * (dev < 2.5 * mad)
                if w.sum() == 0:
                    w = np.exp(-0.5 * ((x[ok] - n) / SMOOTH_ST) ** 2)
            out[i, k] = np.average(yy, weights=w)
    return out


def fill_missing_k(tab, default):
    """Partials with no data at a note inherit from the partial below (or a default)."""
    for i in range(tab.shape[0]):
        for k in range(tab.shape[1]):
            if np.isnan(tab[i, k]):
                tab[i, k] = tab[i, k - 1] if k > 0 else default
    return tab


def smooth_vec(v):
    return smooth_rows(v[:, None])[:, 0]


def build_tables(analysis_dir):
    layers = {
        "v2": load_layer(Path(analysis_dir) / "analysis_v2.json"),
        "v6": load_layer(Path(analysis_dir) / "analysis.json"),
        "v12": load_layer(Path(analysis_dir) / "analysis_v12.json"),
    }
    raw = {k: raw_tables(v) for k, v in layers.items()}
    la = {k: smooth_rows(raw[k][0]) for k in raw}
    # decay shapes: pool all three layers for stability (decay is mostly velocity-independent)
    ff = smooth_rows(np.nanmean(np.stack([raw[k][1] for k in raw]), axis=0))
    ltf = smooth_rows(np.nanmean(np.stack([raw[k][2] for k in raw]), axis=0))
    lts = smooth_rows(np.nanmean(np.stack([raw[k][3] for k in raw]), axis=0))
    ff = fill_missing_k(ff, 0.6)
    ltf = fill_missing_k(ltf, math.log(0.3))
    lts = fill_missing_k(lts, math.log(2.0))
    # stage-2 decay must stay slower than stage 1
    lts = np.maximum(lts, ltf + math.log(2.0))
    B = smooth_vec(np.nanmean(np.stack([raw[k][4] for k in raw]), axis=0))
    B = np.where(np.isnan(B), math.log(1e-3), B)
    # top notes have too few partials to estimate B; extrapolate the trend
    x = np.array(NOTES, float)
    good = x <= 90
    slope, icpt = np.polyfit(x[good & (x >= 60)], B[good & (x >= 60)], 1)
    B = np.where(x > 90, icpt + slope * x, B)

    la2, la6, la12 = la["v2"], la["v6"], la["v12"]
    slo = (la6 - la2) / (VEL["v6"] - VEL["v2"])
    shi = (la12 - la6) / (VEL["v12"] - VEL["v6"])
    env = env_tables(analysis_dir)
    return dict(la6=la6, slo=slo, shi=shi, B=B, env=env)


def env_tables(analysis_dir):
    """Checkpoint levels in dB, [check_idx][note_idx, k], pooled over velocity layers."""
    pooled = []
    for f in ("env_v2.json", "env_v6.json", "env_v12.json"):
        d = json.loads((Path(analysis_dir) / f).read_text())
        check = d["check"]
        by_note = {n["note"]: n["env"] for n in d["notes"]}
        arr = np.full((len(check), len(NOTES), K), np.nan)
        for i, note in enumerate(NOTES):
            for kstr, levels in by_note.get(note, {}).items():
                k = int(kstr)
                if k <= K:
                    arr[:, i, k - 1] = levels
        pooled.append(arr)
    arr = np.nanmean(np.stack(pooled), axis=0)
    out = []
    for c in range(arr.shape[0]):
        t = smooth_rows(arr[c], width=ENV_SMOOTH)
        t = fill_missing_k(t, -60.0)
        out.append(np.clip(t, -100.0, 0.0))
    # keep each partial's envelope non-increasing after the first checkpoint
    for c in range(2, len(out)):
        out[c] = np.minimum(out[c], out[c - 1] + 1.5)
    out[0][:] = 0.0
    globals()["CHECK"] = check
    return out


# ----------------------------------------------------------- hammer noise ---
NAMES = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7,
         "G#": 8, "A": 9, "A#": 10, "B": 11}


def note_num(name):
    return 12 * (int(name[-1]) + 1) + NAMES[name[:-1]]


def noise_levels(ref_dir, layer):
    """RMS of the non-harmonic part of the first 40 ms, in a low and a high band."""
    lows, highs = np.full(len(NOTES), np.nan), np.full(len(NOTES), np.nan)
    for p in Path(ref_dir).glob("*.wav"):
        n = note_num(p.stem)
        if n not in NOTES:
            continue
        sr, x = wavfile.read(p)
        x = x.astype(np.float64)
        if np.abs(x).max() > 2:
            x /= 32768.0
        on = int(np.argmax(np.abs(x) > 0.05 * np.abs(x).max()))
        seg = x[max(0, on - 44): max(0, on - 44) + int(0.04 * sr)]
        spec = np.fft.rfft(seg * np.hanning(len(seg)))
        freqs = np.fft.rfftfreq(len(seg), 1 / sr)
        f0 = layer[n]["fund"] if n in layer else 440 * 2 ** ((n - 69) / 12)
        mask = np.ones_like(freqs, bool)
        for q in (layer[n]["partials"] if n in layer else []):
            mask &= np.abs(freqs - q["freq"]) > max(30.0, 0.3 * f0)
        pw = np.abs(spec) ** 2 * mask
        norm = (np.hanning(len(seg)) ** 2).sum() / 2
        i = NOTES.index(n)
        lows[i] = math.sqrt(pw[(freqs > 40) & (freqs < 400)].sum() / norm / len(seg) * 2)
        highs[i] = math.sqrt(pw[(freqs > 1500) & (freqs < 9000)].sum() / norm / len(seg) * 2)
    return np.log(np.maximum(lows, 1e-7)), np.log(np.maximum(highs, 1e-7))


# ------------------------------------------------------------- synthdef ---
def make_synthdef(T, NZ, gain_norm):
    n_idx = len(NOTES)

    def interp(table_col, i0, i1, fr):
        vals = [float(v) for v in table_col]
        return Select.kr(selector=i0, sources=vals) * (1 - fr) + Select.kr(selector=i1, sources=vals) * fr

    rates = ["ir"] * 9 + ["kr"] * 2

    @synthdef(*rates)
    def grandpiano(note=60, vel=0.5, hard=0.5, decay=1.0, thump=1.0, sustain=6.0, release=1.0,
              stereo_width=0.4, out_bus=0, amp=1.0, pan=0.0):
        x = ((note - NOTES[0]) / 3.0).clip(0, n_idx - 1)
        fr = x.fractional_part()
        i0 = x - fr
        i1 = (i0 + 1).clip(0, n_idx - 1)
        v = vel.clip(0.02, 1.0)
        freq = note.midi_to_hz()
        Bc = interp(T["B"], i0, i1, fr).exponential()
        is_low = v < VEL["v6"]
        dv = v - VEL["v6"]
        tilt = (hard - 0.5) * 1.5

        partials = []
        for k in range(1, K + 1):
            c = k - 1
            la = interp(T["la6"][:, c], i0, i1, fr)
            slope = interp(T["slo"][:, c], i0, i1, fr) * is_low + interp(T["shi"][:, c], i0, i1, fr) * (1 - is_low)
            a = (la + slope * dv + tilt * math.log(k)).clip(-25, 3).exponential()
            fk = freq * k * (1 + Bc * k * k).sqrt()
            a = a * (fk < 19000)
            levels = [interp(T["env"][ci][:, c], i0, i1, fr).clip(-100, 0).db_to_amplitude().max(1e-5)
                      for ci in range(len(CHECK))]
            durs = [CHECK[j + 1] - CHECK[j] for j in range(len(CHECK) - 1)]
            penv = EnvGen.ar(envelope=Envelope(amplitudes=levels, durations=durs,
                                               curves=["exponential"] * len(durs)),
                             time_scale=decay)
            # two slightly detuned strings -> gentle beating, mostly heard in the long tail
            cents = 0.25 + 0.9 * ((k * 0.618) % 1.0)
            det = 2 ** (cents / 1200)
            phase = (k * 1.7) % 6.283
            s1 = SinOsc.ar(frequency=fk, phase=phase)
            s2 = SinOsc.ar(frequency=fk * det, phase=phase)
            string = s1 * 0.85 + s2 * 0.15
            partials.append(string * a * penv)
        sig = Mix.new(partials)

        # hammer noise: low "thump" and a short high "click", levels measured per note
        nl6 = interp(NZ["low6"], i0, i1, fr)
        nls = interp(NZ["low_slope"], i0, i1, fr)
        nh6 = interp(NZ["high6"], i0, i1, fr)
        nhs = interp(NZ["high_slope"], i0, i1, fr)
        low_lvl = (nl6 + nls * dv).clip(-25, 3).exponential() * thump
        high_lvl = (nh6 + nhs * dv).clip(-25, 3).exponential() * thump
        thud_env = EnvGen.ar(envelope=Envelope(amplitudes=[0, 1, 0], durations=[0.002, 0.045], curves=[1, -6]))
        click_env = EnvGen.ar(envelope=Envelope(amplitudes=[0, 1, 0], durations=[0.0005, 0.012], curves=[1, -6]))
        thud = LPF.ar(source=LPF.ar(source=PinkNoise.ar(), frequency=350), frequency=350) * thud_env * low_lvl * 4.0
        click = BPF.ar(source=WhiteNoise.ar(), frequency=3500, reciprocal_of_q=1.2) * click_env * high_lvl * 2.0
        sig = sig + thud + click

        # hammer contact softens the very start; harder / higher = quicker
        att = (0.004 - 0.0025 * v - (note - 60) * 0.00003).clip(0.0008, 0.006)
        key = EnvGen.ar(
            envelope=Envelope(amplitudes=[0, 1, 1, 0], durations=[att, sustain, release], curves=[2, 0, -4]),
            done_action=2)
        sig = sig * key * gain_norm * amp
        position = (pan + (note - 60) / 44.0 * stereo_width).clip(-1, 1)
        DetectSilence.ar(source=sig, threshold=1e-5, time=0.3, done_action=2)
        Out.ar(bus=out_bus, source=Pan2.ar(source=sig, position=position))

    return grandpiano


def main():
    analysis_dir, ref_prefix, out_dir = sys.argv[1:4]
    T = build_tables(analysis_dir)
    layers = {
        "v2": load_layer(Path(analysis_dir) / "analysis_v2.json"),
        "v6": load_layer(Path(analysis_dir) / "analysis.json"),
        "v12": load_layer(Path(analysis_dir) / "analysis_v12.json"),
    }
    cache = Path(analysis_dir) / "noise.json"
    if cache.exists():
        NZ = {k: np.array(v) for k, v in json.loads(cache.read_text()).items()}
    else:
        nz = {}
        for lay, d in (("v2", ref_prefix + "_v2"), ("v6", ref_prefix), ("v12", ref_prefix + "_v12")):
            lo, hi = noise_levels(d, layers[lay])
            nz[lay] = (smooth_vec(lo), smooth_vec(hi))
        dv = VEL["v12"] - VEL["v2"]
        NZ = dict(low6=nz["v6"][0], high6=nz["v6"][1],
                  low_slope=(nz["v12"][0] - nz["v2"][0]) / dv,
                  high_slope=(nz["v12"][1] - nz["v2"][1]) / dv)
        cache.write_text(json.dumps({k: v.tolist() for k, v in NZ.items()}))
    # normalise: at vel 0.5 the loudest partial of middle C peaks around 0.25
    i_c4 = NOTES.index(60)
    la_c4 = T["la6"][i_c4] + T["shi"][i_c4] * (0.5 - VEL["v6"])
    peak_sum = np.exp(la_c4).sum()
    gain_norm = float(1.05 / peak_sum)
    sd = make_synthdef(T, NZ, gain_norm)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = sd.compile()
    (out / "grandpiano.scsyndef").write_bytes(data)
    Path(out / "piano_tables.json").write_text(json.dumps(
        {k: np.asarray(v).tolist() for k, v in {**T, **NZ}.items() if k != "env"} | {"gain_norm": gain_norm}))
    print(f"wrote {out/'grandpiano.scsyndef'} ({len(data)} bytes), gain_norm={gain_norm:.4f}")
    for n in (36, 60, 84):
        i = NOTES.index(n)
        print(n, "k1 env dB:", np.round([T["env"][c][i, 0] for c in range(len(CHECK))]).astype(int).tolist())


if __name__ == "__main__":
    main()
