"""Render notes from a compiled .scsyndef with scsynth in non-realtime mode.

Usage:
  python3 -I render.py SYNTHDEF_FILE NAME OUT.wav DURATION "t:note[:k=v,...]" ...
Each event string: start time, note number, optional extra controls.
e.g.  0:60  0.5:64:vel=0.9  1.0:67
"""
import os
import struct
import subprocess
import sys
import tempfile


def osc_str(s):
    b = s.encode() + b"\0"
    return b + b"\0" * ((4 - len(b) % 4) % 4)


def osc_msg(addr, *args):
    tags, data = ",", b""
    for a in args:
        if isinstance(a, int):
            tags += "i"; data += struct.pack(">i", a)
        elif isinstance(a, float):
            tags += "f"; data += struct.pack(">f", a)
        else:
            tags += "s"; data += osc_str(str(a))
    return osc_str(addr) + osc_str(tags) + data


def bundle(t, msgs):
    secs = int(t)
    frac = int((t - secs) * (1 << 32))
    body = osc_str("#bundle") + struct.pack(">II", secs, frac)
    for m in msgs:
        body += struct.pack(">i", len(m)) + m
    return struct.pack(">i", len(body)) + body


def main():
    sd_path, name, out, dur = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
    events = sys.argv[5:]
    score = bundle(0.0, [osc_msg("/d_load", os.path.abspath(sd_path))])
    node = 1000
    for ev in events:
        parts = ev.split(":")
        t, note = float(parts[0]), float(parts[1])
        ctl = ["note", note]
        if len(parts) > 2 and parts[2]:
            for kv in parts[2].split(","):
                k, v = kv.split("=")
                ctl += [k, float(v)]
        score += bundle(t + 0.001, [osc_msg("/s_new", name, node, 0, 0, *ctl)])
        node += 1
    score += bundle(dur, [osc_msg("/c_set", 0, 0)])
    with tempfile.NamedTemporaryFile(suffix=".osc", delete=False) as f:
        f.write(score)
        score_path = f.name
    r = subprocess.run(["scsynth", "-N", score_path, "_", out, "44100", "WAV", "float",
                        "-o", "2"], capture_output=True, text=True)
    os.unlink(score_path)
    errs = [l for l in (r.stdout + r.stderr).splitlines()
            if "FAILURE" in l or "ERROR" in l or "not found" in l.lower() or "exception" in l.lower()]
    if r.returncode != 0 or errs:
        print("scsynth problems:", r.returncode, *errs[:10], sep="\n  ")
    print("rendered", out)


if __name__ == "__main__":
    main()
