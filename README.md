# reaper2mt3

Pairs REAPER projects with their manually rendered audio into aligned
audio/MIDI training examples for [MT3](https://github.com/magenta/mt3), with a
manifest and contract checks.

Input is projects produced by [`midi2reaper`](../midi2reaper), auditioned and
rendered to audio in REAPER by hand. Output satisfies `../DATA_CONTRACT.md`.

The `.RPP` is deliberately the label source rather than the original MIDI: it
carries the instrument assignment you verified by ear (an SFLT soundfont or a
real plugin chain), so a correction made in REAPER — a swapped instrument, a
renamed part, an edited note — carries through instead of being overwritten.

## Requirements

macOS, though nothing here is macOS-specific except the soundfont library path.

### 1. Verified, rendered REAPER projects

Produced by `midi2reaper build`, then rendered to a `.wav` or `.flac` by hand
in REAPER with the same base name as the `.RPP` (`Song.RPP` + `Song.wav`, or
`Song.flac`). REAPER itself is not required to run this tool — it parses the
`.RPP` directly.

### 2. Python  [3.11]

Python 3.11+ and [uv](https://docs.astral.sh/uv/). Runtime dependencies are
`mido`, `numpy` and `soundfile`; `pytest` for the tests.

## Usage

```sh
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e .

.venv/bin/reaper2mt3 import ../reaper/generated -o ../data/pilot
.venv/bin/reaper2mt3 check ../data/pilot          # re-run contract checks
```

`import` scans a directory for `<name>.RPP` + `<name>.wav`/`.flac` pairs
(checked in that order — if a project somehow has both, `.wav` wins), copies
the audio byte-for-byte in whatever format it finds, writes a corpus MIDI
from the RPP's labels and notes, and runs the full `DATA_CONTRACT.md`
acceptance list against what it wrote. Format detection is by file content
(`soundfile`/`libsndfile`), not extension, so WAV and FLAC examples are
validated through the exact same code path — nothing downstream needs to
know or care which a given example used. **No audio is synthesized** —
nothing here calls a renderer of any kind.

```text
data/pilot/
  manifest.jsonl
  midi/train/ex_0002.mid      audio/train/ex_0002.wav
  midi/test/ex_0001.mid       audio/test/ex_0001.flac
```

A non-zero exit means at least one example failed a check; each failure names
the `DATA_CONTRACT.md` check number that caught it. `--test-fraction` (default
0.2) controls the split.

### Pitch-range exceptions

A note outside the accepted range (check 8) is rejected unless explicitly
approved. Pass `--exceptions <file.json>`, keyed by `source_midi_id`:

```json
{
  "Metallica-Enter Sandman-07-28-2026": {
    "drums": { "extra_pitches": [17, 18], "reason": "deliberate low-mapped special hits" }
  }
}
```

The approval is written into `approved_exceptions` on the manifest record
itself, so a later `reaper2mt3 check` — run without `--exceptions`, in a
different process, possibly much later — still honours it. It widens the range
only for the exact pitches on the exact named track, never the whole example.

## How it works

### Labels are not preset numbers

Two numbering systems meet here and must not be confused. A soundfont-backed
part is *rendered* with the soundfont's own bank and patch, which are
arbitrary — `Power Guitar 1.sf2` sits at patch 0. It is *labelled* with the
General MIDI program its canonical track name declares, which is the training
target. The corpus MIDI carries the label, never the patch — and a part driven
by a real instrument plugin chain has no bank/patch at all, only the label.

### A part is identified by its name, not by SFLT

Reading used to look for an SFLT instance to recognize a part, which silently
dropped every track `midi2reaper`'s chain library had moved onto a real
instrument (Kontakt, Guitar Rig, Ample Bass, BeatBuddy...) — on one project
that meant 4 of 6 parts vanished with no error. A part is now recognized by its
canonical track name regardless of what's driving it; the plugin chain is
still captured, as `instrument_plugins`, for provenance.

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

### Splits

Assigned per *source*, never per render, and by ranking a hash rather than
thresholding it — thresholding is stable but not proportional, and produced an
empty test split on nine sources at 0.2.

## `build`: the FluidSynth path

A second command, `reaper2mt3 build`, renders audio itself via FluidSynth
instead of importing an existing render — see `build --help`. It exists because
Pedalboard cannot host SFLT (SFLT loads its soundfont only inside
`initialize()`, before any state can be injected, and every state-injection
route tried renders silence — see git history for the detail), so FluidSynth
was the fallback for automated rendering. `import` is the workflow actually in
use: rendering is done by hand in REAPER, with real instrument chains this
path cannot reach.

`build`'s known limits, if used: what you audition in REAPER is not
bit-identical to a FluidSynth render, since they're different synthesis
engines; FluidSynth appends its own post-roll on top of `--tail-seconds`, so
the manifest records the tail actually present rather than the value
requested; and it works only for SFLT-backed parts, never chain-driven ones.

## Known limits

- Overlapping identical pitches within a part are paired first-in-first-out.
  Counts and onsets survive exactly; nested durations may swap between the two
  notes, which MIDI itself cannot disambiguate.
- The contract checks verify structure and alignment, not that a part *sounds*
  like its label. An instrument mismatched to its program passes every check.
- No augmentation. Each source yields exactly one render, so a corpus built
  this way has no timbral variation.
- `import` trusts the paired audio's actual format; a render that isn't mono
  16-bit 44.1 kHz is flagged by check 6, not corrected, regardless of whether
  it's WAV or FLAC.
