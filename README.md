# grandpiano: an analysis-based piano SynthDef for Sonic Pi

`grandpiano.scsyndef` is a pure-scsynth additive piano. It uses only core UGens
(SinOsc, Klank, EnvGen, Select, LPF, BPF, noise), so it works on Sonic Pi's SuperSonic
engine, which can't load sc3-plugins such as OteyPiano.

Each note has up to 96 slightly inharmonic partials, all measured from the Salamander
Grand Piano (a Yamaha C5) at 30 notes (A0 to C8, every 3 semitones) and three velocity layers:

- **Partials 1-24** each get their own oscillator and follow their measured level curve
  (9 checkpoints, 0-16 s). The first 16 also get a slightly detuned second string.
- **Partials 25-96** run in 4 groups of `Klang` sine banks (one UGen per group and decay
  stage). Each partial keeps its measured level and its own fast/slow mix; each group shares
  fast and slow decay times. This is what gives bass notes their bright "zing" (real A0 has
  energy up to its 80th partial), at little CPU cost and well within scsynth's limit on how
  many audio signals can be in flight at once.
- **Quasi-random partial phases.** If all partials start in phase, the stretched (inharmonic)
  partials of the bass re-form a pulse every cycle that smears into a falling chirp, heard as a
  "zap" at the attack. (An earlier version had this.)
- **The strings build up over about 16 ms at A0, down to 2 ms at the top,** as measured from the
  recordings. The hammer thump and click start instantly on top.
- Each sampled note keeps its own spectrum. Notes in between interpolate between the two
  nearest sampled notes.

## Use it in Sonic Pi

Two copies of the same synth are included. They differ only in the name stored inside:

- **`grandpiano.scsyndef`** loads as an external synth:
  ```ruby
  # enable external synths first:  sonicpi config set enable-external-synths 1  (then reboot)
  load_synthdefs "C:/path/to/folder/containing/grandpiano.scsyndef"

  # external synths get no note-name or use_transpose handling, so convert notes yourself:
  define :pno do |n, **opts|
    play note(n) + current_transpose, **opts
  end

  use_synth "grandpiano"
  pno :c4, vel: 0.5
  ```
- **`sonic-pi-grandpiano.scsyndef`** is for registering it as a built-in `:grandpiano` (a
  `SynthGrandPiano` entry in `synthinfo.rb` with `synth_name "grandpiano"`, plus this file in
  `etc/synthdefs/compiled`). Sonic Pi then handles note names, `use_transpose` and chords
  itself, so use plain `play`. Don't use the `pno` helper with it, or notes get transposed twice.

**CPU:** each note's table lookups are computed once when the note starts, and the upper
partials share one sine bank per group. 32 overlapping notes take about 35% of one core
(measured offline). An earlier build re-read its tables on every audio block and needed
about 176%, which caused "Timing error: can't keep up" in fast passages. Notes ring for
`sustain` + `release` (7 s by default), so very dense passages stack many notes. A shorter
`sustain` lightens the load further.

| opt | default | what it does |
|---|---|---|
| `note` | 60 | MIDI number (note names work only with the built-in registration, or via `pno`) |
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

The settings are env vars, documented at the top of `build_grandpiano.py`: `K`, `K_SIN`,
`AMP_SMOOTH`, `ENV_SMOOTH`, `ENV_K_SMOOTH`, `TABLE_RATE`, `KMODE`, `KL_SINGLE`, `PHASES`,
`ATT_MODE`, `ATT_BASS`, `CLICK_*` and `SYNTH_NAME`. The defaults reproduce `grandpiano.scsyndef`;
`SYNTH_NAME=sonic-pi-grandpiano` writes the built-in copy.

## Re-measure from the recordings (full pipeline)

1. Get the samples (about 5 MB per layer, Salamander Grand V3 via npm):
   ```bash
   for v in 2 6 12; do npm pack @audio-samples/piano-velocity$v; done
   # unpack; convert to mono 44.1 kHz WAVs named C4.wav, D#4.wav ...:
   #   layer 6 -> ref/, layer 2 -> ref_v2/, layer 12 -> ref_v12/
   ffmpeg -i C4v6.ogg -ac 1 -ar 44100 ref/C4.wav
   ```
2. Find the partials (up to `MAX_PARTIALS`, default 96):
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

These need `scsynth` (e.g. `apt install supercollider-server`).

```bash
# render a note offline: "start_time:note[:opt=value,...]"
python3 scripts/render.py out/grandpiano.scsyndef grandpiano renders/g_C4.wav 4.5 "0:60:vel=0.375,stereo_width=0"

# octave-band error vs the real recording over five time windows (lower is better).
# Ignores the recording's noise floor and the 50 Hz mains hum below the note.
python3 scripts/bands.py renders g C4 [-v]          # REF_DIR=... to compare other layers

# the first 64 ms: third-octave bands in 4 ms frames vs the real attack, plus how much the
# synth flickers frame to frame compared with the real piano (a "zap" shows up as extra flicker)
python3 scripts/onset_score.py renders g A1 C2 A2 C3

# level and brightness table for one note
python3 scripts/compare.py ref/C4.wav renders/g_C4.wav
```

`vel=0.375` corresponds to Salamander layer 6. The layers map as layer / 16 (2 → 0.125, 12 → 0.75).

Mean error at vel 0.375 (dB, lower is better):

| | first version | 96-partial version | this version |
|---|---|---|---|
| bass tone (A0-A2) | 4.9 | 2.1 | 2.1 |
| middle tone (C3-C5) | 3.3 | 2.3 | 2.3 |
| treble tone (C6, C7) | 3.8 | 3.3 | 3.3 |
| attack, bass (first 64 ms) | 9.9 | 6.5 | 4.8 |
| attack, middle | 4.6 | 3.3 | 2.7 |
| attack flicker vs real piano | +0.8 | +1.3 | -1.1 |

## Files

- `grandpiano.scsyndef`, `sonic-pi-grandpiano.scsyndef`: the built synth (external / built-in name)
- `scripts/analyze.py`: partial detection and inharmonicity fit
- `scripts/analyze_env.py`: per-partial checkpoint envelopes
- `scripts/build_grandpiano.py`: tables, velocity model, Klank fits and SynthDef generation
- `scripts/render.py`: non-realtime scsynth renderer (writes the OSC score itself)
- `scripts/bands.py`, `scripts/onset_score.py`, `scripts/onset.py`, `scripts/compare.py`: real vs synth measurements
- `scripts/klank_cal.py`: measures Klank's output gain (0.514 x amplitude), used by `KMODE=klank`
- `scripts/build_otey_test.py`: OteyPiano test SynthDef (sc3-plugins scsynth only; not SuperSonic)
- `data/`: measurements derived from the Salamander Grand Piano

## Credits

Measurements come from the Salamander Grand Piano by Alexander Holm, licensed CC BY 3.0.
No audio samples are included.
