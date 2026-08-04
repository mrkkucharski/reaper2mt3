# reaper2mt3

Renders verified REAPER projects into aligned audio/MIDI training examples for
[MT3](https://github.com/magenta/mt3), with a manifest and contract checks.

Input is the projects produced and auditioned via
[`midi2reaper`](../midi2reaper). Output satisfies `../DATA_CONTRACT.md`.

The `.RPP` is deliberately the input rather than the original MIDI: it carries
the soundfont assignment you verified by ear, so a correction made in REAPER — a
swapped soundfont, a renamed part, an edited note — carries through instead of
being overwritten.

## Requirements

macOS, though nothing here is macOS-specific except the soundfont library path.
Versions in brackets are what this has been verified against.

### 1. FluidSynth  [2.5.7]

```sh
brew install fluid-synth
```

Renders the audio. See [Why FluidSynth](#why-fluidsynth-and-not-sflt) — it is
deliberately *not* the same program that plays the projects you audition.

### 2. A soundfont library  [`/Users/Shared/Soundfonts`]

Projects reference soundfonts by absolute path, so the library must be wherever
it was when the projects were built. `midi2reaper` writes paths under
`/Users/Shared/Soundfonts`. Nothing needs to be configured here; the paths come
out of the project file.

### 3. Verified REAPER projects

Produced by `midi2reaper build`. REAPER itself is not required — this tool
parses the `.RPP` directly and never launches it.

### 4. Python  [3.11]

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Runtime dependencies are
`mido` and `numpy`; `pytest` for the tests.

## Usage

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .

.venv/bin/reaper2mt3 build ../reaper/generated -o ../data/pilot
.venv/bin/reaper2mt3 check ../data/pilot          # re-run contract checks
```

```text
data/pilot/
  manifest.jsonl
  midi/train/ex_0002.mid      audio/train/ex_0002.wav
  midi/test/ex_0001.mid       audio/test/ex_0001.wav
```

A non-zero exit means at least one example failed a check; each failure names
the `DATA_CONTRACT.md` check number that caught it.

Useful flags: `--test-fraction` (default 0.2), `--sample-rate` (44100),
`--tail-seconds` (2.0), `--gain` (0.6), `--keep-stems` to retain per-part
renders for debugging.

## How it works

### Why FluidSynth and not SFLT

Hosting SFLT in [Pedalboard](https://github.com/spotify/pedalboard) was tried
and does not work. SFLT loads its soundfont only inside `initialize()`
(`sflt-nih/src/lib.rs:1240`), which the host calls once *before* any state can
be injected; a soundfont set afterwards is stored but never loaded, and the
plugin renders silence. Injecting through `raw_state` (JUCE base64) and through
`preset_data` (a rebuilt VST3 preset with recomputed chunk offsets) both fail
the same way, as does forcing re-initialisation by changing sample rate. The
tell is that `patch_number` never snaps to a valid preset, which
`auto_find_preset` would do had the soundfont loaded.

FluidSynth reads the same `.sf2` with the same bank and patch, runs headless and
offline at roughly 90× realtime, and is the conventional renderer for
SoundFont-derived transcription corpora.

**The consequence is that REAPER is an approximate audition, not an exact
preview** — same soundfont and preset, different synthesis engine, so timbre
differs in detail.

### Labels are not preset numbers

Two numbering systems meet here and must not be confused. A part is *rendered*
with the soundfont's own bank and patch, which are arbitrary — `Power Guitar
1.sf2` sits at patch 0. It is *labelled* with the General MIDI program its
canonical track name declares, which is the training target. The corpus MIDI
carries the label, never the patch.

### The track-name interface

REAPER track names are the contract between the two repos. A guitar playing
chordal accompaniment is titled `<slug>:rhythm` — `distortion-guitar:rhythm` —
and **every other** part is titled `<slug>` alone, including guitars that lead:
`distortion-guitar`, `tenor-sax`, `drums`. Either may be followed by ` | ` and
anything human-readable, which is ignored.

`rhythm` is the only annotation; there is no `:lead`. Absence claims only that
the part is not chordal accompaniment. `rhythm` on a non-guitar
(`tenor-sax:rhythm`) and any explicit `:lead` are both rejected as corrupt
labels rather than accepted quietly.

The two repos share no code, so `tests/test_names.py` pins the grammar against
names taken verbatim from generated projects, checks all 128 programs
round-trip, and asserts no two programs collapse to the same slug.

### Rendering

Each part renders in its own FluidSynth process, which keeps one soundfont
loaded at a time and isolates crashes. Reverb and chorus are off by default for
reproducibility. Stems are summed to mono and peak-normalised to −1 dBFS.

Splits are assigned per *source*, never per render, and by ranking a hash rather
than thresholding it — thresholding is stable but not proportional, and produced
an empty test split on nine sources at 0.2.

## Known limits

- What you audition in REAPER is not bit-identical to what is rendered, because
  SFLT and FluidSynth are different synthesis engines reading the same file.
- FluidSynth appends its own post-roll on top of `--tail-seconds`, so the
  manifest records the tail actually present in the file (4.0 s at the default
  2.0 s request) rather than the value requested.
- Overlapping identical pitches within a part are paired first-in-first-out.
  Counts and onsets survive exactly; nested durations may swap between the two
  notes, which MIDI itself cannot disambiguate.
- The contract checks verify structure and alignment, not that a part *sounds*
  like its label. A soundfont mismatched to its program passes every check.
- No augmentation. Each source yields exactly one render, so a corpus built this
  way has no timbral variation.
