"""Write MT3 training examples and check them against DATA_CONTRACT.md.

Two numbering systems meet here and must not be confused. A part is *rendered*
with the soundfont's own bank and patch, which are arbitrary — `Power Guitar
1.sf2` sits at patch 0. A part is *labelled* with the General MIDI program its
canonical track name declares, which is the training target. The corpus MIDI
therefore carries the label, never the patch.
"""

from __future__ import annotations

import hashlib
import json
import wave
from dataclasses import dataclass
from pathlib import Path

import mido

from . import gm
from .render import DRUM_CHANNEL, RenderSettings, mix, render_part, write_wav
from .rppread import Project, ProjectPart
from .sf2 import read_presets

PITCH_MIN, PITCH_MAX = 21, 108
PERCUSSION_MIN, PERCUSSION_MAX = 27, 87


@dataclass
class Example:
    example_id: str
    split: str
    record: dict
    problems: list[str]


def write_corpus_midi(project: Project, path: Path) -> None:
    """Type-1 MIDI: a conductor track plus one canonically named track per part."""
    midi = mido.MidiFile(type=1, ticks_per_beat=project.ppq)

    conductor = mido.MidiTrack()
    midi.tracks.append(conductor)
    conductor.append(mido.MetaMessage("track_name", name="conductor", time=0))
    numerator, denominator = project.time_signature
    conductor.append(mido.MetaMessage("time_signature", numerator=numerator,
                                      denominator=denominator, time=0))
    previous = 0
    for tick, tempo in project.tempo_events():
        conductor.append(mido.MetaMessage("set_tempo", tempo=tempo, time=tick - previous))
        previous = tick

    for index, part in enumerate(project.parts):
        track = mido.MidiTrack()
        midi.tracks.append(track)
        channel = DRUM_CHANNEL if part.is_drum else _melodic_channel(index)
        track.append(mido.MetaMessage("track_name", name=part.canonical_name, time=0))
        if not part.is_drum:
            track.append(mido.Message("program_change", channel=channel,
                                      program=part.program, time=0))

        events: list[tuple[int, int, mido.Message]] = []
        for note in part.notes:
            events.append((note.start, 1, mido.Message("note_on", channel=channel,
                                                       note=note.pitch, velocity=note.velocity)))
            events.append((note.end, 0, mido.Message("note_off", channel=channel, note=note.pitch)))
        events.sort(key=lambda e: (e[0], e[1]))

        previous = 0
        for tick, _, message in events:
            message.time = tick - previous
            track.append(message)
            previous = tick

    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))


def _melodic_channel(index: int) -> int:
    """Channels 0-15 skipping 9, which GM reserves for percussion."""
    channel = index % 15
    return channel if channel < DRUM_CHANNEL else channel + 1


def build_example(
    project: Project,
    out_root: Path,
    example_id: str,
    split: str,
    settings: RenderSettings,
    work_dir: Path,
    renderer: str,
) -> Example:
    midi_path = Path("midi") / split / f"{example_id}.mid"
    audio_path = Path("audio") / split / f"{example_id}.wav"

    stems = [render_part(part, project, work_dir / example_id, settings) for part in project.parts]
    audio = mix(stems, settings)
    write_wav(audio, out_root / audio_path, settings.sample_rate)
    write_corpus_midi(project, out_root / midi_path)

    # FluidSynth appends its own post-roll on top of the requested tail, so the
    # manifest records what the file actually contains rather than what was asked
    # for. The contract permits a fixed tail; it has to be the documented one.
    performance_seconds = project.seconds_at(project.last_tick)
    tail_seconds = round(len(audio) / settings.sample_rate - performance_seconds, 3)

    record = {
        "id": example_id,
        "split": split,
        "audio_path": str(audio_path),
        "midi_path": str(midi_path),
        "sample_rate_hz": settings.sample_rate,
        "channels": 1,
        "source_midi_id": project.name,
        "source_project": str(project.path),
        "source_project_sha256": _sha256(project.path),
        "soundfont_library_root": "/Users/Shared/Soundfonts",
        "parts": [_part_record(part) for part in project.parts],
        "skipped_tracks": [{"name": name, "reason": "no SFLT instance in project"}
                           for name in project.unparsed_tracks],
        "renderer": renderer,
        "effects_chain": settings.effects,
        "performance_seconds": round(performance_seconds, 3),
        "tail_seconds": tail_seconds,
        "tail_seconds_requested": settings.tail_seconds,
        "duration_seconds": round(len(audio) / settings.sample_rate, 3),
        "normalization": settings.normalization,
    }
    return Example(example_id, split, record, validate_example(record, out_root))


def _part_record(part: ProjectPart) -> dict:
    return {
        "track_name": part.canonical_name,
        "program": part.program,
        "is_drum": part.is_drum,
        "rhythm": part.rhythm,
        "note_count": part.note_count,
        "soundfont": str(part.soundfont),
        "bank": part.bank,
        "patch": part.patch,
        "reaper_track_name": part.track_name,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_example(record: dict, out_root: Path) -> list[str]:
    """The acceptance checks in DATA_CONTRACT.md, run against what was written."""
    problems: list[str] = []
    parts = record["parts"]

    if not any(not p["is_drum"] and gm.is_guitar(p["program"]) for p in parts):
        problems.append("check 1: no guitar part")

    pairs = [(p["program"], p["is_drum"], p["rhythm"]) for p in parts]
    if len(pairs) != len(set(pairs)):
        problems.append("check 4: two parts share a (program, rhythm) pair")

    for part in parts:
        expected = gm.track_name(part["program"], part["is_drum"], part["rhythm"])
        if part["track_name"] != expected:
            problems.append(f"check 2: {part['track_name']} is not canonical ({expected})")

        soundfont = Path(part["soundfont"])
        if not soundfont.exists():
            problems.append(f"check 10: missing soundfont {soundfont}")
        else:
            presets = read_presets(soundfont)
            if presets is None:
                problems.append(f"check 10: unreadable soundfont {soundfont.name}")
            elif not any(x.bank == part["bank"] and x.patch == part["patch"] for x in presets):
                problems.append(
                    f"check 10: {soundfont.name} lacks bank {part['bank']} patch {part['patch']}"
                )

    midi_path = out_root / record["midi_path"]
    audio_path = out_root / record["audio_path"]

    if not midi_path.exists():
        problems.append("check 7: MIDI missing")
    else:
        problems += _check_midi(midi_path, parts)

    if not audio_path.exists():
        problems.append("check 6: WAV missing")
    else:
        problems += _check_audio(audio_path, record["sample_rate_hz"])

    if not record.get("renderer") or not record.get("normalization"):
        problems.append("check 11: renderer or normalization metadata missing")

    return problems


def _check_midi(path: Path, parts: list[dict]) -> list[str]:
    problems: list[str] = []
    midi = mido.MidiFile(str(path))
    by_name = {p["track_name"]: p for p in parts}
    seen: dict[str, int] = {}

    for track in midi.tracks:
        name = next((m.name for m in track if m.type == "track_name"), "")
        if name in ("", "conductor"):
            continue
        part = by_name.get(name)
        if part is None:
            problems.append(f"check 5: MIDI track {name} is not in the manifest")
            continue

        programs = [m.program for m in track if m.type == "program_change"]
        if not part["is_drum"]:
            if len(programs) != 1:
                problems.append(f"check 3: {name} has {len(programs)} program changes")
            elif programs[0] != part["program"]:
                problems.append(f"check 3: {name} carries program {programs[0]}, "
                                f"labelled {part['program']}")
        if any(m.type == "control_change" and m.control in (0, 32) for m in track):
            problems.append(f"check 3: {name} contains a bank-select event")

        notes = [m.note for m in track if m.type == "note_on" and m.velocity > 0]
        seen[name] = len(notes)
        low, high = (PERCUSSION_MIN, PERCUSSION_MAX) if part["is_drum"] else (PITCH_MIN, PITCH_MAX)
        out_of_range = [n for n in notes if not low <= n <= high]
        if out_of_range:
            problems.append(f"check 8: {name} has {len(out_of_range)} notes outside {low}-{high}")

    for part in parts:
        if seen.get(part["track_name"]) != part["note_count"]:
            problems.append(f"check 5: {part['track_name']} has {seen.get(part['track_name'])} "
                            f"notes in MIDI, manifest says {part['note_count']}")
    return problems


def _check_audio(path: Path, sample_rate: int) -> list[str]:
    problems: list[str] = []
    with wave.open(str(path)) as handle:
        if handle.getnchannels() != 1:
            problems.append(f"check 6: WAV has {handle.getnchannels()} channels, expected mono")
        if handle.getframerate() != sample_rate:
            problems.append(f"check 6: WAV is {handle.getframerate()} Hz, expected {sample_rate}")
        if handle.getsampwidth() != 2:
            problems.append("check 6: WAV is not 16-bit PCM")
        frames = handle.readframes(handle.getnframes())
    if not frames or max(frames) == 0:
        problems.append("check 6: WAV is silent")
    return problems


def assign_splits(source_ids: list[str], test_fraction: float) -> dict[str, str]:
    """Stable per-source split that also guarantees the test count.

    Thresholding a per-item hash is stable but not proportional — on nine
    sources at 0.2 it produced an empty test split. Ranking by hash keeps the
    assignment deterministic and independent of input order while making the
    size exact, and splitting per source (never per render) is what stops
    alternate renders of one performance leaking across splits.
    """
    ordered = sorted(set(source_ids), key=lambda s: hashlib.sha256(s.encode()).hexdigest())
    if test_fraction <= 0 or not ordered:
        return {name: "train" for name in ordered}
    count = max(1, round(len(ordered) * test_fraction)) if test_fraction > 0 else 0
    count = min(count, max(0, len(ordered) - 1))  # never leave the train split empty
    test = set(ordered[:count])
    return {name: ("test" if name in test else "train") for name in ordered}


def write_manifest(examples: list[Example], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for example in examples:
            handle.write(json.dumps(example.record) + "\n")
