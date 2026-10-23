"""Build an analysis-based additive piano SynthDef for Sonic Pi.

Usage: python3 -I build_grandpiano.py DATA_DIR REF_DIR_PREFIX OUT_DIR
  DATA_DIR holds analysis_v2.json, analysis.json (v6), analysis_v12.json,
           env_v2.json, env_v6.json, env_v12.json and (optionally) noise.json
  REF_DIR_PREFIX e.g. ./ref  (reads ref_v2/, ref/, ref_v12/ WAVs to measure hammer
           noise; only needed when DATA_DIR has no noise.json yet)
  Env vars (defaults reproduce the shipped synth):
    K=96          partials per note
    K_SIN=24      partials 1..K_SIN get individual oscillators with full measured envelopes
    AMP_SMOOTH=0  smoothing of partial levels across notes, semitones (0 = keep each note's spectrum)
    ENV_SMOOTH=2.0  smoothing of envelopes across notes, semitones
    ENV_K_SMOOTH=0  1 = also smooth envelopes across neighbouring partials
    TABLE_RATE=ir table lookups computed once per note (scalar rate); 'kr' = every block (~4x the CPU)
    KMODE=klang   partials above K_SIN: 'klang' = 4 groups of phase-scrambled sine banks (default),
                  'klank' = impulse-excited resonator banks (can give a 'zap' at the attack)
    KL_SINGLE=1   one Klang bank per group with a shared fast/slow mix (half the sines)
    PHASES=rand   quasi-random partial phases (avoids a coherent per-cycle chirp); 'lin' = old
    ATT_MODE=note string attack per note: ATT_BASS (0.016 s at A0) down to 2 ms at the top, ATT_CURVE=2
    CLICK_F=2500, CLICK_GAIN=0.35   hammer click centre and level
    HLPF=0        1 = felt-hammer low-pass opening over the attack (measured no benefit)
    SYNTH_NAME    name inside the SynthDef (default grandpiano; sonic-pi-grandpiano for a synth
                  registered in Sonic Pi's synthinfo.rb)
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from supriya import Envelope, synthdef
from supriya.enums import CalculationRate
from supriya.ugens import (
    BPF, DetectSilence, EnvGen, LPF, HPF, Out, Pan2, PinkNoise, Select, SinOsc,
    WhiteNoise, Mix, Klank, Impulse,
)

from supriya.ugens import UGen, param, ugen


@ugen(ar=True)
class Klang(UGen):
    """SuperCollider core Klang: a bank of fixed sine oscillators (freq, amp, phase per partial)."""
    frequency_scale = param(1)
    frequency_offset = param(0)
    specifications = param(unexpanded=True)


def _make_select_ir():
    @ugen(ar=True, kr=True, ir=True)
    class Select(UGen):          # same scsynth UGen as supriya's Select, but scalar rate allowed
        selector = param()
        sources = param(unexpanded=True)
    return Select


SelectIR = _make_select_ir()

GROUP_BOUNDS = [25, 33, 45, 61, 97]     # partial groups that share fast/slow decay times
K = int(__import__('os').environ.get('K', '96'))   # partials per note
K_PAIR = 16                              # partials that get a detuned second string
K_SIN = int(__import__('os').environ.get('K_SIN', '24'))   # partials with full measured envelopes (SinOsc)
KLANK_GAIN = 1 / 0.514                   # Klank rings at 0.514 x its amplitude (measured)
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
        # partials below 16 kHz that were not detected: just under the weakest detected
        # one (they were below the detection threshold), not silent
        found = la[i][~np.isnan(la[i])]
        miss_level = (found.min() - math.log(2.0)) if len(found) else math.log(10 ** (FLOOR_DB / 20))
        top_k = max([p["k"] for p in d["partials"]] + [1])
        for k in range(1, K + 1):
            if np.isnan(la[i, k - 1]) and k * f0 < 16000:
                # beyond the highest partial ever detected, fall off quickly instead
                extra = max(0, k - top_k) * math.log(10 ** (3 / 20))
                la[i, k - 1] = miss_level - extra
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
            if w.sum() <= 1e-12:      # no measurement anywhere near: use the nearest one
                out[i, k] = yy[np.argmin(np.abs(x[ok] - n))]
            else:
                out[i, k] = np.average(yy, weights=w)
    return out


def fill_missing_k(tab, default):
    """Partials with no data at a note inherit from the partial below (or a default)."""
    for i in range(tab.shape[0]):
        for k in range(tab.shape[1]):
            if np.isnan(tab[i, k]):
                tab[i, k] = tab[i, k - 1] if k > 0 else default
    return tab


def smooth_k(tab):
    out = tab.copy()
    for k in range(K_PAIR, tab.shape[1] - 1):
        out[:, k] = 0.25 * tab[:, k - 1] + 0.5 * tab[:, k] + 0.25 * tab[:, k + 1]
    return out


def fill_nan_notes(tab):
    """Fill missing table entries by interpolating across notes, without smoothing."""
    out = tab.copy()
    x = np.array(NOTES, float)
    for k in range(tab.shape[1]):
        y = tab[:, k]
        ok = ~np.isnan(y)
        if ok.sum() == 0:
            out[:, k] = math.log(10 ** (FLOOR_DB / 20))
        elif ok.sum() < len(y):
            out[:, k] = np.interp(x, x[ok], y[ok])
    return out


def fit_two_stage(env_tabs, check):
    """For every (note, k): fraction f of a fast decay tau1 plus (1-f) of a slow decay tau2
    that best matches the measured checkpoint levels (dB)."""
    t = np.array(check)
    fs = np.linspace(0.0, 0.97, 14)
    t1s = np.geomspace(0.03, 2.0, 16)
    t2s = np.geomspace(0.3, 40.0, 18)
    F, T1, T2 = np.meshgrid(fs, t1s, t2s, indexing="ij")
    F, T1, T2 = F.ravel(), T1.ravel(), T2.ravel()
    M = F[:, None] * np.exp(-t[None, :] / T1[:, None]) + (1 - F[:, None]) * np.exp(-t[None, :] / T2[:, None])
    MdB = 20 * np.log10(np.maximum(M, 1e-6))
    L = np.stack(env_tabs, axis=-1)            # [note, k, check]
    n_notes, n_k, _ = L.shape
    ff = np.zeros((n_notes, n_k)); tau1 = np.ones((n_notes, n_k)); tau2 = np.ones((n_notes, n_k))
    for i in range(n_notes):
        for k in range(n_k):
            y = L[i, k]
            w = (y > -80).astype(float)
            err = (((MdB - y[None, :]) ** 2) * w[None, :]).sum(axis=1)
            j = int(np.argmin(err))
            ff[i, k], tau1[i, k], tau2[i, k] = F[j], T1[j], T2[j]
    return ff, np.log(tau1), np.log(tau2)


def smooth_vec(v):
    return smooth_rows(v[:, None])[:, 0]


def build_tables(analysis_dir):
    layers = {
        "v2": load_layer(Path(analysis_dir) / "analysis_v2.json"),
        "v6": load_layer(Path(analysis_dir) / "analysis.json"),
        "v12": load_layer(Path(analysis_dir) / "analysis_v12.json"),
    }
    raw = {k: raw_tables(v) for k, v in layers.items()}
    AMP_SMOOTH = float(__import__('os').environ.get('AMP_SMOOTH', '0'))
    if AMP_SMOOTH > 0:
        la = {k: smooth_rows(raw[k][0], robust=False, width=AMP_SMOOTH) for k in raw}
    else:   # keep each sampled note's own spectrum; only fill gaps
        la = {k: fill_nan_notes(raw[k][0]) for k in raw}
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
    kff, kt1, kt2 = fit_two_stage(env, CHECK)
    # per group: amplitude-weighted average (log) decay times
    G = len(GROUP_BOUNDS) - 1
    gt1 = np.zeros((len(NOTES), G)); gt2 = np.zeros((len(NOTES), G))
    for g in range(G):
        ks = [k - 1 for k in range(GROUP_BOUNDS[g], GROUP_BOUNDS[g + 1]) if k <= kt1.shape[1]]
        w = np.exp(la6[:, ks]) + 1e-12
        gt1[:, g] = (w * kt1[:, ks]).sum(1) / w.sum(1)
        gt2[:, g] = (w * kt2[:, ks]).sum(1) / w.sum(1)
    gkf = np.zeros((len(NOTES), G))
    for g in range(G):
        ks = [k - 1 for k in range(GROUP_BOUNDS[g], GROUP_BOUNDS[g + 1]) if k <= kff.shape[1]]
        w = np.exp(la6[:, ks]) + 1e-12
        gkf[:, g] = (w * kff[:, ks]).sum(1) / w.sum(1)
    return dict(la6=la6, slo=slo, shi=shi, B=B, env=env, kff=kff, kt1=kt1, kt2=kt2, gt1=gt1, gt2=gt2, gkf=gkf)


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
        if __import__('os').environ.get('ENV_K_SMOOTH', '0') == '1':
            t = smooth_k(t)
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
        if __import__('os').environ.get('TABLE_RATE', 'ir') == 'ir':   # computed once when the note starts, not every block
            sel = lambda i: SelectIR.ir(selector=i, sources=vals)
        else:
            sel = lambda i: Select.kr(selector=i, sources=vals)
        return sel(i0) * (1 - fr) + sel(i1) * fr

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

        sig = None
        spec_fast, spec_slow = [], []
        G = len(GROUP_BOUNDS) - 1
        kl_fast = [[] for _ in range(G)]; kl_slow = [[] for _ in range(G)]
        use_klang = __import__('os').environ.get('KMODE', 'klang') == 'klang'
        for k in range(1, K + 1):
            c = k - 1
            la = interp(T["la6"][:, c], i0, i1, fr)
            slope = interp(T["slo"][:, c], i0, i1, fr) * is_low + interp(T["shi"][:, c], i0, i1, fr) * (1 - is_low)
            a = (la + slope * dv + tilt * math.log(k)).clip(-25, 3).exponential()
            fk = freq * k * (1 + Bc * k * k).sqrt()
            a = a * (fk < 19000)
            if k > K_SIN:
                kf = interp(T["kff"][:, c], i0, i1, fr)
                t60_1 = interp(T["kt1"][:, c], i0, i1, fr).exponential() * math.log(1000)
                t60_2 = interp(T["kt2"][:, c], i0, i1, fr).exponential() * math.log(1000)
                if use_klang:
                    g = max(i for i in range(G) if GROUP_BOUNDS[i] <= k)
                    ph = ((k * k * 0.6180339887 + k * 0.381966) % 1.0) * 6.283
                    kl_fast[g] += [fk, a * kf, ph]
                    kl_slow[g] += [fk, a * (1 - kf), ph]
                else:
                    spec_fast += [fk, a * kf * KLANK_GAIN, t60_1]
                    spec_slow += [fk, a * (1 - kf) * KLANK_GAIN, t60_2]
                continue
            levels = [interp(T["env"][ci][:, c], i0, i1, fr).clip(-100, 0).db_to_amplitude().max(1e-5)
                      for ci in range(len(CHECK))]
            durs = [CHECK[j + 1] - CHECK[j] for j in range(len(CHECK) - 1)]
            penv = EnvGen.kr(envelope=Envelope(amplitudes=levels, durations=durs,
                                               curves=["exponential"] * len(durs)),
                             time_scale=decay)
            # two slightly detuned strings -> gentle beating, mostly heard in the long tail
            cents = 0.25 + 0.9 * ((k * 0.618) % 1.0)
            det = 2 ** (cents / 1200)
            if __import__('os').environ.get('PHASES', 'rand') == 'rand':   # quasi-random phases: no coherent pulse/chirp
                phase = ((k * k * 0.6180339887 + k * 0.381966) % 1.0) * 6.283
            else:
                phase = (k * 1.7) % 6.283
            s1 = SinOsc.ar(frequency=fk, phase=phase)
            if k <= K_PAIR:
                s2 = SinOsc.ar(frequency=fk * det, phase=phase)
                string = s1 * 0.85 + s2 * 0.15
            else:
                string = s1
            part = string * a * penv
            sig = part if sig is None else sig + part   # running sum keeps few buffers alive

        single = __import__('os').environ.get('KL_SINGLE', '1') == '1'
        for g in range(G):
            if not kl_fast[g]:
                continue
            if single:   # one bank per group: partial levels exact, fast/slow mix shared per group
                specs = []
                for j in range(0, len(kl_fast[g]), 3):
                    specs += [kl_fast[g][j], kl_fast[g][j + 1] + kl_slow[g][j + 1], kl_fast[g][j + 2]]
                gk = interp(T["gkf"][:, g], i0, i1, fr)
                e_f = EnvGen.kr(envelope=Envelope(amplitudes=[1, 1e-4], durations=[1], curves=["exponential"]),
                                time_scale=interp(T["gt1"][:, g], i0, i1, fr).exponential() * math.log(1e4) * decay)
                e_s = EnvGen.kr(envelope=Envelope(amplitudes=[1, 1e-4], durations=[1], curves=["exponential"]),
                                time_scale=interp(T["gt2"][:, g], i0, i1, fr).exponential() * math.log(1e4) * decay)
                bank = Klang._new_expanded(calculation_rate=CalculationRate.AUDIO, frequency_scale=1,
                                           frequency_offset=0, specifications=specs)
                sig = sig + bank * (e_f * gk + e_s * (1 - gk))
                continue
            for specs, tab in ((kl_fast[g], "gt1"), (kl_slow[g], "gt2")):
                tau = interp(T[tab][:, g], i0, i1, fr).exponential()
                genv = EnvGen.kr(envelope=Envelope(amplitudes=[1, 1e-4], durations=[1], curves=["exponential"]),
                                 time_scale=tau * math.log(1e4) * decay)
                bank = Klang._new_expanded(calculation_rate=CalculationRate.AUDIO, frequency_scale=1,
                                           frequency_offset=0, specifications=specs)
                sig = sig + bank * genv

        if spec_fast:
            if __import__('os').environ.get('KEXC', 'impulse') == 'noise':   # 2 ms noise burst: random phase per mode
                burst = EnvGen.ar(envelope=Envelope(amplitudes=[0, 1, 0], durations=[0.001, 0.001], curves=['sine', 'sine']))
                exc = WhiteNoise.ar() * burst * float(__import__('os').environ.get('KEXC_GAIN', '0.26'))
            else:
                exc = Impulse.ar(frequency=0)
            for spec in (spec_fast, spec_slow):
                bank = Klank._new_expanded(calculation_rate=CalculationRate.AUDIO, source=exc,
                                           frequency_scale=1, frequency_offset=0, decay_scale=decay,
                                           specifications=spec)
                sig = sig + bank

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
        click_f = float(__import__('os').environ.get('CLICK_F', '2500'))
        click = BPF.ar(source=WhiteNoise.ar(), frequency=click_f, reciprocal_of_q=1.2) * click_env * high_lvl * 2.0 * float(__import__('os').environ.get('CLICK_GAIN', '0.35'))
        if __import__('os').environ.get('ATT_MODE', 'note') == 'note':   # string build-up: ~12 ms in the bass to ~2 ms at the top
            att_bass = float(__import__('os').environ.get('ATT_BASS', '0.016'))
            s_att = (att_bass - (note - 21) * (att_bass - 0.002) / 63.0).clip(0.002, att_bass) * (1.15 - 0.3 * v)
            s_curve = float(__import__('os').environ.get('ATT_CURVE', '2'))
            s_env = EnvGen.ar(envelope=Envelope(amplitudes=[0, 1], durations=[1], curves=[s_curve]), time_scale=s_att)
            sig = sig * s_env
        sig = sig + thud + click
        if __import__('os').environ.get('HLPF', '0') == '1':   # felt hammer: the top end opens up over the first few ms (slower in the bass)
            sweep_t = (0.012 - (note - 36) * 0.0003).clip(0.002, 0.02)
            cut0 = (freq * 3).max(300)
            open_env = EnvGen.ar(envelope=Envelope(amplitudes=[0, 1], durations=[1], curves=[-3]), time_scale=sweep_t)
            cutoff = (cut0.log() + open_env * ((19000 / cut0).log())).exponential()
            sig = LPF.ar(source=sig, frequency=cutoff)

        # hammer contact softens the very start; harder / higher = quicker
        att = (0.004 - 0.0025 * v - (note - 60) * 0.00003).clip(0.0008, 0.006)
        if __import__('os').environ.get('ATT_MODE', 'note') == 'note':
            att = 0.0005   # strings have their own attack above; this only de-clicks
        key = EnvGen.ar(
            envelope=Envelope(amplitudes=[0, 1, 1, 0], durations=[att, sustain, release], curves=[2, 0, -4]),
            done_action=2)
        sig = sig * key * gain_norm * amp
        position = (pan + (note - 60) / 44.0 * stereo_width).clip(-1, 1)
        DetectSilence.ar(source=sig, threshold=1e-5, time=0.3, done_action=2)
        Out.ar(bus=out_bus, source=Pan2.ar(source=sig, position=position))

    return grandpiano


def rename_synthdef(data, new_name):
    """Change the SynthDef name stored in a compiled single-def .scsyndef file."""
    assert data[:4] == b"SCgf"
    n = data[10]
    nb = new_name.encode()
    return data[:10] + bytes([len(nb)]) + nb + data[11 + n:]


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
    synth_name = __import__('os').environ.get('SYNTH_NAME', 'grandpiano')
    if synth_name != "grandpiano":
        data = rename_synthdef(data, synth_name)
    (out / f"{synth_name}.scsyndef").write_bytes(data)
    Path(out / "piano_tables.json").write_text(json.dumps(
        {k: np.asarray(v).tolist() for k, v in {**T, **NZ}.items() if k not in ("env",)} | {"gain_norm": gain_norm}))
    print(f"wrote {out / (synth_name + '.scsyndef')} ({len(data)} bytes), gain_norm={gain_norm:.4f}")
    for n in (36, 60, 84):
        i = NOTES.index(n)
        print(n, "k1 env dB:", np.round([T["env"][c][i, 0] for c in range(len(CHECK))]).astype(int).tolist())


if __name__ == "__main__":
    main()
