import os, sys, subprocess, tempfile
import numpy as np
from scipy.io import wavfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import osc_msg, bundle
from supriya import synthdef
from supriya.ugens import Klank, Impulse, Out, Line
from supriya.enums import CalculationRate

@synthdef("ir")
def kcal(freq=440):
    sig = Klank._new_expanded(calculation_rate=CalculationRate.AUDIO, source=Impulse.ar(frequency=0), frequency_scale=1, frequency_offset=0, decay_scale=1, specifications=[freq * 1, 1.0, 2.0])
    Line.kr(start=0, stop=1, duration=1.5, done_action=2)
    Out.ar(bus=0, source=sig)

d = tempfile.mkdtemp(); open(f"{d}/kcal.scsyndef","wb").write(kcal.compile())
for f in (30, 100, 300, 1000, 3000, 8000):
    sc = bundle(0.0,[osc_msg("/d_load", f"{d}/kcal.scsyndef")]) + bundle(0.01,[osc_msg("/s_new","kcal",1000,0,0,"freq",float(f))]) + bundle(1.0,[osc_msg("/c_set",0,0)])
    t = tempfile.NamedTemporaryFile(suffix=".osc", delete=False); t.write(sc); t.close()
    out = f"{d}/k.wav"
    subprocess.run(["scsynth","-N",t.name,"_",out,"44100","WAV","float","-o","1"],capture_output=True)
    sr, x = wavfile.read(out); x = x.astype(float)
    seg = x[int(0.1*sr):int(0.1*sr)+int(sr*max(0.05, 3/f))]
    print(f"{f:6d} Hz: peak {np.abs(seg).max():.5f} at 0.1 s  (expected decay to {10**(-3*0.1/2):.3f} of start)")
