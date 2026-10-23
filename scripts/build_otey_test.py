import sys
from pathlib import Path

from supriya import synthdef
from supriya.ugens import UGen, param, ugen, Line, Out, Pan2, DetectSilence


@ugen(ar=True)
class OteyPiano(UGen):
    frequency = param(440)
    vel = param(1)
    t_gate = param(0)
    rmin = param(0.35)
    rmax = param(2)
    rampl = param(4)
    rampr = param(8)
    rcore = param(1)
    lmin = param(0.07)
    lmax = param(1.4)
    lampl = param(-4)
    lampr = param(4)
    rho = param(1)
    e = param(1)
    zb = param(1)
    zh = param(0)
    mh = param(1)
    k = param(0.2)
    alpha = param(1)
    p = param(1)
    hpos = param(0.142)
    loss = param(1)
    detune = param(0.0003)
    hammer_type = param(1)


@synthdef()
def oteytest(note=60, amp=1, vel=0.7, pan=0, out_bus=0):
    gate = Line.kr(start=1, stop=0, duration=0.01)
    sig = OteyPiano.ar(frequency=note.midi_to_hz(), vel=vel, t_gate=gate)
    DetectSilence.ar(source=sig, threshold=0.0001, time=0.5, done_action=2)
    Out.ar(bus=out_bus, source=Pan2.ar(source=sig * amp, position=pan))


out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
(out / "oteytest.scsyndef").write_bytes(oteytest.compile())
print("ok")
