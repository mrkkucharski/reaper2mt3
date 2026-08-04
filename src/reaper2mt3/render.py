"""Render parts to audio with FluidSynth.

Pedalboard cannot drive SFLT: SFLT loads its soundfont only inside
`initialize()`, which the host calls once before any state can be injected, so
a soundfont set afterwards is stored but never loaded and the plugin renders
silence. FluidSynth reads the same `.sf2` with the same bank and patch, runs
headless and offline, and is the conventional renderer for SoundFont-derived
transcription corpora.

Each part is rendered in its own FluidSynth process. That keeps one soundfont
loaded at a time, isolates crashes, and matches the process-isolation advice for
dataset rendering.
"""

from __future__ import annotations

import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import mido
import numpy as np

from .rppread import Project, ProjectPart

DRUM_CHANNEL = 9
GM_DRUM_BANK = 128


@dataclass(frozen=True)
class RenderSettings:
    sample_rate: int = 44100
    tail_seconds: float = 2.0
    gain: float = 0.6
    reverb: bool = False
    chorus: bool = False
    peak_dbfs: float = -1.0

    @property
    def normalization(self) -> str:
        return f"peak:{self.peak_dbfs:g}dBFS"

    @property
    def effects(self) -> str:
        return f"reverb={'on' if self.reverb else 'off'},chorus={'on' if self.chorus else 'off'}"


class RenderError(RuntimeError):
    pass


def fluidsynth_version() -> str:
    binary = shutil.which("fluidsynth")
    if binary is None:
        raise RenderError("fluidsynth not found on PATH — install it with `brew install fluid-synth`")
    out = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout
    for line in out.splitlines():
        if "version" in line.lower():
            return line.strip()
    return "fluidsynth (unknown version)"


def write_part_midi(part: ProjectPart, project: Project, path: Path, settings: RenderSettings) -> None:
    """One-track MIDI selecting the part's preset, for FluidSynth to render."""
    midi = mido.MidiFile(ticks_per_beat=project.ppq)
    track = mido.MidiTrack()
    midi.tracks.append(track)

    channel = DRUM_CHANNEL if part.is_drum else 0
    events: list[tuple[int, int, mido.Message]] = []

    for tick, tempo in project.tempo_events():
        events.append((tick, 0, mido.MetaMessage("set_tempo", tempo=tempo)))

    # FluidSynth routes channel 10 to the percussion bank itself; sending an
    # explicit bank 128 through CC0 is impossible (CC values stop at 127).
    if not part.is_drum:
        events.append((0, 1, mido.Message("control_change", channel=channel, control=0,
                                          value=part.bank & 0x7F)))
        events.append((0, 1, mido.Message("control_change", channel=channel, control=32, value=0)))
    events.append((0, 2, mido.Message("program_change", channel=channel, program=part.patch & 0x7F)))

    for note in part.notes:
        events.append((note.start, 4, mido.Message("note_on", channel=channel, note=note.pitch,
                                                   velocity=note.velocity)))
        events.append((note.end, 3, mido.Message("note_off", channel=channel, note=note.pitch)))

    events.sort(key=lambda e: (e[0], e[1]))
    previous = 0
    for tick, _, message in events:
        message.time = tick - previous
        track.append(message)
        previous = tick

    # Convert the tail through the tempo in force at the end, so the rendered
    # release is the requested length regardless of the piece's tempo.
    final_tempo = project.tempo_events()[-1][1] if project.tempo_events() else 500_000
    tail_ticks = int(settings.tail_seconds * 1e6 / final_tempo * project.ppq)
    track.append(mido.MetaMessage("end_of_track", time=max(1, tail_ticks)))
    midi.save(str(path))


def render_part(part: ProjectPart, project: Project, work_dir: Path, settings: RenderSettings) -> np.ndarray:
    """Render one part to a float32 mono array."""
    if not part.soundfont.exists():
        raise RenderError(f"missing soundfont {part.soundfont}")

    work_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(part.canonical_name)
    midi_path = work_dir / f"{stem}.mid"
    wav_path = work_dir / f"{stem}.wav"
    write_part_midi(part, project, midi_path, settings)

    command = [
        shutil.which("fluidsynth") or "fluidsynth",
        "-ni", "-F", str(wav_path),
        "-r", str(settings.sample_rate),
        "-g", str(settings.gain),
        "-o", f"synth.reverb.active={'yes' if settings.reverb else 'no'}",
        "-o", f"synth.chorus.active={'yes' if settings.chorus else 'no'}",
        "-o", "synth.sample-rate=" + str(settings.sample_rate),
        str(part.soundfont), str(midi_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not wav_path.exists():
        raise RenderError(f"fluidsynth failed for {part.canonical_name}: {result.stderr.strip()[:300]}")

    return _read_wav_mono(wav_path)


def _safe_stem(name: str) -> str:
    return name.replace(":", "-").replace("/", "-")


def _read_wav_mono(path: Path) -> np.ndarray:
    with wave.open(str(path)) as handle:
        frames = handle.readframes(handle.getnframes())
        channels = handle.getnchannels()
        width = handle.getsampwidth()
    if width != 2:
        raise RenderError(f"unexpected sample width {width} in {path.name}")
    audio = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return audio


def mix(stems: list[np.ndarray], settings: RenderSettings) -> np.ndarray:
    """Sum stems to one mono track and peak-normalise."""
    if not stems:
        raise RenderError("nothing to mix")
    length = max(len(s) for s in stems)
    total = np.zeros(length, dtype=np.float32)
    for stem in stems:
        total[: len(stem)] += stem

    peak = float(np.max(np.abs(total)))
    if peak > 0:
        total *= (10 ** (settings.peak_dbfs / 20.0)) / peak
    return total


def write_wav(audio: np.ndarray, path: Path, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(audio, -1.0, 1.0)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes((clipped * 32767).astype("<i2").tobytes())
