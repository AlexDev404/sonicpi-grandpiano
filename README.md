# grandpiano: an analysis-based piano SynthDef for Sonic Pi

`grandpiano.scsyndef` is a pure-scsynth additive piano. It uses only core UGens
(SinOsc, EnvGen, Select, LPF, BPF, noise), so it works on Sonic Pi's SuperSonic
engine, which can't load sc3-plugins such as OteyPiano.

Each note is 16 slightly inharmonic partials. Every partial follows a measured level
curve with 9 checkpoints from 0 to 16 s. The curves come from the Salamander Grand
Piano (a Yamaha C5) at 30 notes (A0 to C8, every 3 semitones) and three velocity layers.
A short hammer thump and click is added on top.

## Use it in Sonic Pi

```ruby
# enable external synths first:  sonicpi config set enable-external-synths 1  (then reboot)
load_synthdefs "C:/path/to/folder/containing/grandpiano.scsyndef"

synth "grandpiano", note: 60, vel: 0.5
```

| opt | default | what it does |
|---|---|---|
| `note` | 60 | MIDI number (use `note(:c4)` for names) |
| `vel` | 0.5 | strike strength, 0-1: loudness and brightness |
| `sustain` | 6 | seconds the key is held |
| `release` | 1 | damper fade after `sustain` |
| `hard` | 0.5 | brightness tilt (higher = brighter) |
| `decay` | 1 | stretches/shrinks the natural decay |
| `thump` | 1 | hammer noise level |
| `stereo_width` | 0.4 | low notes left, high notes right |
| `amp`, `pan`, `out_bus` | 1, 0, 0 | as usual |

Note: the synth is named `grandpiano` because Sonic Pi would resolve `"piano"` to its built-in `:piano`.

## Rebuild it

```bash
pip install supriya numpy scipy        # supriya compiles SynthDefs without sclang

# from the included measurements (no samples needed); reproduces grandpiano.scsyndef exactly
python3 scripts/build_grandpiano.py data ./ref ./out
```

`ENV_SMOOTH=2.0` (env var) sets how much the envelopes are smoothed across neighbouring
notes, in semitones. The output gain is set in `main()` (`gain_norm`), and matches Sonic Pi's
built-in `:piano` at default settings.

## Re-measure from the recordings (full pipeline)

1. Get the samples (about 5 MB per layer, Salamander Grand V3 via npm):
   ```bash
   for v in 2 6 12; do npm pack @audio-samples/piano-velocity$v; done
   # unpack each; convert to mono 44.1 kHz WAVs named C4.wav, D#4.wav ...:
   #   layer 6 -> ref/, layer 2 -> ref_v2/, layer 12 -> ref_v12/
   ffmpeg -i C4v6.ogg -ac 1 -ar 44100 ref/C4.wav
   ```
2. Find the partials (frequency, inharmonicity, peak level):
   ```bash
   python3 scripts/analyze.py ref     data/analysis.json
   python3 scripts/analyze.py ref_v2  data/analysis_v2.json
   python3 scripts/analyze.py ref_v12 data/analysis_v12.json
   ```
3. Measure each partial's level at the checkpoint times:
   ```bash
   python3 scripts/analyze_env.py ref     data/analysis.json     data/env_v6.json
   python3 scripts/analyze_env.py ref_v2  data/analysis_v2.json  data/env_v2.json
   python3 scripts/analyze_env.py ref_v12 data/analysis_v12.json data/env_v12.json
   ```
4. Delete `data/noise.json` so the hammer noise is measured again from the WAVs, then build:
   ```bash
   python3 scripts/build_grandpiano.py data ./ref ./out
   ```

## Check it

Both of these need `scsynth` (e.g. `apt install supercollider-server`).

```bash
# render notes offline: "start_time:note[:opt=value,...]"
python3 scripts/render.py out/grandpiano.scsyndef grandpiano c4.wav 5 "0:60:vel=0.375"
# compare level and brightness over time against the real recording
python3 scripts/compare.py ref/C4.wav c4.wav
```

`vel=0.375` corresponds to Salamander layer 6. The layers map as layer / 16 (2 → 0.125, 12 → 0.75).

## Files

- `grandpiano.scsyndef`: the built synth
- `scripts/analyze.py`: partial detection and inharmonicity fit
- `scripts/analyze_env.py`: per-partial checkpoint envelopes
- `scripts/build_grandpiano.py`: smoothing, velocity model and SynthDef generation
- `scripts/render.py`: non-realtime scsynth renderer (writes the OSC score itself)
- `scripts/compare.py`: real vs synth level and brightness table
- `scripts/build_otey_test.py`: OteyPiano test SynthDef (sc3-plugins scsynth only; not SuperSonic)
- `data/`: measurements derived from the Salamander Grand Piano

## Credits

Measurements come from the Salamander Grand Piano by Alexander Holm, licensed CC BY 3.0.
No audio samples are included.
