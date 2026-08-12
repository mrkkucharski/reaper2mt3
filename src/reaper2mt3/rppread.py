"""Read a REAPER project back into parts, notes and a tempo map.

The RPP is the source of truth for stage two, not the original MIDI: it carries
the soundfont assignment you verified by ear, and any correction made in REAPER
— a swapped soundfont, a renamed part, an edited note — is honoured rather than
silently overwritten by re-deriving from the source arrangement.
"""

from __future__ import annotations

import base64
import json
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

from . import gm

SFLT_MARKER = '<VST "VST3i: SFLT'
# REAPER breaks each base64 block into 128-character lines and closes the block
# with a shorter one, which is how the three blocks are told apart.
B64_LINE_WIDTH = 128
_EVENT = re.compile(r"^[Ee]\s+(\d+)\s+([0-9a-fA-F]{2})\s+([0-9a-fA-F]{2})\s+([0-9a-fA-F]{2})\s*$")

# A part is now identified by its track name, not by finding an SFLT instance:
# midi2reaper splices real instrument FX chains (Kontakt, Guitar Rig, Ample
# Bass...) in place of SFLT wherever a tuned chain exists, and a reader that
# only recognized SFLT silently dropped every one of those tracks. The name
# stays the identifier and one instrument name is captured for provenance
# regardless of which plugin family produced it.
_PLUGIN_LINE = re.compile(r'^<(VST|JS|AU|CLAP)\s+(?:"([^"]*)"|(\S+))')
_FORMAT_PREFIX = re.compile(r"^(VST3i?|VSTi?|AUi?|CLAPi?):\s*")


def _plugin_name(line: str) -> str | None:
    match = _PLUGIN_LINE.match(line.strip())
    if not match:
        return None
    name = match.group(2) if match.group(2) is not None else match.group(3)
    return _FORMAT_PREFIX.sub("", name.strip())


@dataclass(frozen=True)
class Note:
    start: int
    end: int
    pitch: int
    velocity: int
    channel: int


@dataclass
class ProjectPart:
    track_name: str
    program: int | None
    is_drum: bool
    rhythm: bool | None
    # Present only for an SFLT-backed part. A part driven by a real instrument
    # chain carries none of these; `instrument_plugins` is its provenance instead.
    soundfont: Path | None
    bank: int | None
    patch: int | None
    instrument_plugins: list[str] = field(default_factory=list)
    notes: list[Note] = field(default_factory=list)
    muted: bool = False
    soloed: bool = False

    @property
    def canonical_name(self) -> str:
        return gm.track_name(self.program, self.is_drum, self.rhythm)

    @property
    def note_count(self) -> int:
        return len(self.notes)


@dataclass
class Project:
    path: Path
    ppq: int
    tempo_points: list[tuple[float, float]]  # (seconds, bpm)
    time_signature: tuple[int, int]
    parts: list[ProjectPart] = field(default_factory=list)
    unparsed_tracks: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.path.stem

    def tempo_events(self) -> list[tuple[int, int]]:
        """(tick, microseconds per beat) reconstructed from the REAPER envelope.

        Envelope points are stored in seconds with square shape, so tempo holds
        constant between points and each tick offset follows from the previous
        point's tempo.
        """
        points = self.tempo_points or [(0.0, 120.0)]
        events: list[tuple[int, int]] = []
        tick = 0.0
        previous_seconds, previous_bpm = 0.0, points[0][1]

        for seconds, bpm in points:
            tick += (seconds - previous_seconds) * previous_bpm / 60.0 * self.ppq
            events.append((int(round(tick)), int(round(60_000_000 / bpm))))
            previous_seconds, previous_bpm = seconds, bpm

        deduped: list[tuple[int, int]] = []
        for entry in events:
            if not deduped or deduped[-1][1] != entry[1]:
                deduped.append(entry)
        return deduped

    @property
    def last_tick(self) -> int:
        return max((n.end for p in self.parts for n in p.notes), default=0)

    def seconds_at(self, tick: int) -> float:
        events = self.tempo_events() or [(0, 500_000)]
        seconds = 0.0
        previous_tick, previous_tempo = 0, events[0][1]
        for event_tick, tempo in events:
            if event_tick >= tick:
                break
            seconds += (event_tick - previous_tick) / self.ppq * previous_tempo / 1e6
            previous_tick, previous_tempo = event_tick, tempo
        return seconds + (tick - previous_tick) / self.ppq * previous_tempo / 1e6


def read_project(path: Path) -> Project:
    lines = path.read_text(encoding="utf-8", errors="replace").split("\n")
    ppq = 960
    tempo_points: list[tuple[float, float]] = []
    time_signature = (4, 4)
    parts: list[ProjectPart] = []
    unparsed: list[str] = []

    track_name: str | None = None
    pending: dict | None = None

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()

        if stripped.startswith("TEMPO "):
            bits = stripped.split()
            if len(bits) >= 4:
                time_signature = (int(bits[2]), int(bits[3]))
            if not tempo_points:
                tempo_points.append((0.0, float(bits[1])))
        elif stripped.startswith("PT "):
            bits = stripped.split()
            tempo_points.append((float(bits[0 + 1]), float(bits[2])))
        elif stripped.startswith("<TRACK "):
            _flush(parts, unparsed, pending, track_name)
            # Initialized as soon as a track starts, not on first SFLT sighting,
            # so a track driven by any other plugin still gets its notes read.
            track_name, pending = None, {}
        elif stripped.startswith("NAME ") and track_name is None:
            track_name = _unquote(stripped[5:].strip())
        elif stripped.startswith("MUTESOLO ") and pending is not None:
            bits = stripped.split()
            if len(bits) >= 3:
                pending["muted"] = bits[1] != "0"
                pending["soloed"] = bits[2] != "0"
        elif stripped.startswith(SFLT_MARKER) and pending is not None:
            payload, i = _collect_chunk(lines, i)
            state = _decode_state(payload)["fields"]
            if state["file"] != "null":
                pending["soundfont"] = Path(json.loads(state["file"]))
                pending["bank"] = int(float(state["bank"]))
                pending["patch"] = int(float(state["patch"]))
        elif stripped.startswith(("<VST", "<JS", "<AU", "<CLAP")) and pending is not None:
            name = _plugin_name(stripped)
            if name:
                pending.setdefault("plugins", []).append(name)
        elif stripped.startswith("HASDATA "):
            bits = stripped.split()
            if len(bits) >= 3 and bits[2].isdigit():
                ppq = int(bits[2])
        elif stripped.startswith(("E ", "e ")) and pending is not None:
            notes, i = _read_events(lines, i)
            pending.setdefault("notes", []).extend(notes)

        i += 1

    _flush(parts, unparsed, pending, track_name)

    # A single tempo is written as TEMPO alone; keep the envelope authoritative.
    if len(tempo_points) > 1:
        tempo_points = tempo_points[1:] if tempo_points[1][0] == 0.0 else tempo_points

    return Project(
        path=path,
        ppq=ppq,
        tempo_points=sorted(tempo_points),
        time_signature=time_signature,
        parts=parts,
        unparsed_tracks=unparsed,
    )


def _flush(parts, unparsed, pending, track_name) -> None:
    if pending is None:
        # No <TRACK block was ever opened for this name -- happens only before
        # the first track in the file, where track_name is also always None.
        if track_name:
            unparsed.append(track_name)
        return
    parsed = gm.parse_track_name(track_name or "")
    if parsed is None:
        unparsed.append(track_name or "(unnamed)")
        return
    program, is_drum, rhythm = parsed
    parts.append(
        ProjectPart(
            track_name=track_name,
            program=program,
            is_drum=is_drum,
            rhythm=rhythm,
            soundfont=pending.get("soundfont"),
            bank=pending.get("bank"),
            patch=pending.get("patch"),
            instrument_plugins=pending.get("plugins", []),
            notes=sorted(pending.get("notes", []), key=lambda n: n.start),
            muted=pending.get("muted", False),
            soloed=pending.get("soloed", False),
        )
    )


def _unquote(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'`":
        return text[1:-1]
    return text


def _collect_chunk(lines: list[str], start: int) -> tuple[list[str], int]:
    payload = []
    i = start + 1
    while not lines[i].strip().startswith(">"):
        payload.append(lines[i].strip())
        i += 1
    return payload, i


def _decode_state(payload: list[str]) -> dict:
    blocks, current = [], []
    for line in payload:
        current.append(line)
        if len(line) < B64_LINE_WIDTH:
            blocks.append(base64.b64decode("".join(current)))
            current = []
    length = struct.unpack("<I", blocks[1][:4])[0]
    return json.loads(blocks[1][8 : 8 + length].decode())


def _read_events(lines: list[str], start: int) -> tuple[list[Note], int]:
    """Consume a run of `E <delta> <status> <data1> <data2>` lines into notes."""
    notes: list[Note] = []
    # Merged parts stack identical pitches, so each (channel, pitch) holds a
    # queue and note-offs are matched first-in-first-out. A plain dict would
    # discard every note but the last of an overlapping run.
    open_notes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    tick = 0
    i = start

    while i < len(lines):
        match = _EVENT.match(lines[i].strip())
        if match is None:
            break
        delta, status, d1, d2 = match.groups()
        tick += int(delta)
        status, pitch, value = int(status, 16), int(d1, 16), int(d2, 16)
        kind, channel = status & 0xF0, status & 0x0F

        if kind == 0x90 and value > 0:
            open_notes.setdefault((channel, pitch), []).append((tick, value))
        elif kind in (0x80, 0x90):
            queue = open_notes.get((channel, pitch))
            if queue:
                begin, velocity = queue.pop(0)
                notes.append(Note(begin, max(tick, begin + 1), pitch, velocity, channel))
        i += 1

    return notes, i - 1
