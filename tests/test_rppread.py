"""A part is identified by its track name, not by finding an SFLT instance.

midi2reaper splices real instrument chains (Kontakt, Guitar Rig, Ample Bass...)
in place of SFLT wherever a tuned chain exists. A reader that only recognized
SFLT silently dropped every chain-driven track -- these tests are what caught
that and what pin the fix.
"""

import base64
import json
import struct
from pathlib import Path

from reaper2mt3.rppread import read_project


_B64_LINE_WIDTH = 128


def _b64_lines(data: bytes) -> list[str]:
    """Split into REAPER's own 128-char lines, exactly as a real project
    would: a block is closed by the first line under the width, so a payload
    landing exactly on the boundary would silently run into the next block."""
    text = base64.b64encode(data).decode()
    lines = [text[i : i + _B64_LINE_WIDTH] for i in range(0, len(text), _B64_LINE_WIDTH)]
    return lines or [""]


def _sflt_chunk(soundfont: str, bank: int, patch: int) -> str:
    """A minimal but genuinely decodable SFLT state chunk.

    `_decode_state` only inspects the second base64 block (a u32 JSON length,
    a u32 constant, the JSON, then padding) and ignores the first entirely, so
    the first block can be a single dummy byte.
    """
    state = {"version": "0.10.0", "fields": {
        "file": json.dumps(soundfont), "bank": str(bank), "patch": str(patch),
    }}
    payload = json.dumps(state).encode()
    block1 = struct.pack("<II", len(payload), 1) + payload + b"\x00" * 8
    lines = _b64_lines(b"X") + _b64_lines(block1)
    body = "\n".join(f"        {line}" for line in lines)
    return "\n".join([
        '      <VST "VST3i: SFLT (ash taylor) (34 out)" SFLT.vst3 0 "" 955354538{4} ""',
        body,
        "      >",
        "      FXID {A0000000-0000-0000-0000-000000000001}",
    ])


SFLT_CHUNK = _sflt_chunk("/tmp/x.sf2", 0, 27)

CHAIN_CHUNK = """      <VST "VST3i: Kontakt 8 (Native Instruments) (64 out)" "Kontakt 8.vst3" 0 "" 9 ""
        AAAA
      >
      FXID {A0000000-0000-0000-0000-000000000002}
      <VST "VST3: Guitar Rig 7 (Native Instruments)" "Guitar Rig 7.vst3" 0 "" 1 ""
        BBBB
      >
      FXID {A0000000-0000-0000-0000-000000000003}"""


def _track(name: str, fx_chunk: str, notes: str, mutesolo: str | None = None) -> str:
    lines = ["  <TRACK {G}", f'    NAME "{name}"']
    if mutesolo is not None:
        lines.append(f"    MUTESOLO {mutesolo}")
    lines += [
        "    <FXCHAIN",
        fx_chunk,
        "    >",
        "    <ITEM",
        "      <SOURCE MIDI",
        "        HASDATA 1 480 QN",
        notes,
        "      >",
        "    >",
        "  >",
    ]
    return "\n".join(lines)


def _project(*tracks: str) -> str:
    return "\n".join(['<REAPER_PROJECT 0.1 "7.74" 0 0', "  TEMPO 120 4 4 0", *tracks, ">"])


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "project.RPP"
    path.write_text(text)
    return path


def test_chain_driven_track_is_recognized(tmp_path):
    """The original bug: a track with no SFLT instance vanished entirely."""
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(_track("distortion-guitar", CHAIN_CHUNK, notes))
    project = read_project(_write(tmp_path, text))

    assert len(project.parts) == 1
    assert project.unparsed_tracks == []
    part = project.parts[0]
    assert part.note_count == 1
    assert part.soundfont is None
    assert part.bank is None and part.patch is None


def test_chain_driven_track_records_plugin_provenance(tmp_path):
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(_track("distortion-guitar", CHAIN_CHUNK, notes))
    project = read_project(_write(tmp_path, text))

    assert project.parts[0].instrument_plugins == [
        "Kontakt 8 (Native Instruments) (64 out)",
        "Guitar Rig 7 (Native Instruments)",
    ]


def test_sflt_driven_track_still_reads_soundfont(tmp_path):
    """Regression guard: fixing the chain-blind bug must not break the
    original SFLT path."""
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(_track("electric-guitar-clean:rhythm", SFLT_CHUNK, notes))
    project = read_project(_write(tmp_path, text))

    part = project.parts[0]
    assert part.soundfont == Path("/tmp/x.sf2")
    assert part.bank == 0 and part.patch == 27
    assert part.instrument_plugins == []


def test_mixed_project_recovers_every_part(tmp_path):
    """The real bug: Deep Purple recovered only its two SFLT parts (organ,
    clarinet) and silently dropped guitar, bass and drums -- all chain-driven."""
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(
        _track("rock-organ", SFLT_CHUNK, notes),
        _track("overdriven-guitar:rhythm", CHAIN_CHUNK, notes),
        _track("drums", CHAIN_CHUNK, notes),
    )
    project = read_project(_write(tmp_path, text))

    assert {p.canonical_name for p in project.parts} == {
        "rock-organ", "overdriven-guitar:rhythm", "drums",
    }
    assert project.unparsed_tracks == []


def test_unrecognized_track_name_still_reported_unparsed(tmp_path):
    """A track with real content but no canonical name is still skipped and
    named -- this must not regress into silent loss either."""
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(_track("Random Track Name", CHAIN_CHUNK, notes))
    project = read_project(_write(tmp_path, text))

    assert project.parts == []
    assert project.unparsed_tracks == ["Random Track Name"]


def test_default_track_is_neither_muted_nor_soloed(tmp_path):
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(_track("drums", CHAIN_CHUNK, notes))
    project = read_project(_write(tmp_path, text))

    assert project.parts[0].muted is False
    assert project.parts[0].soloed is False


def test_muted_track_is_reported(tmp_path):
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(_track("drums", CHAIN_CHUNK, notes, mutesolo="1 0 0"))
    project = read_project(_write(tmp_path, text))

    assert project.parts[0].muted is True
    assert project.parts[0].soloed is False


def test_soloed_track_is_reported(tmp_path):
    """REAPER's solo field is not just a bool (e.g. 2 for solo-in-front), so
    any non-zero value must count as soloed."""
    notes = "        E 0 90 3c 64\n        E 480 80 3c 00"
    text = _project(_track("drums", CHAIN_CHUNK, notes, mutesolo="0 2 0"))
    project = read_project(_write(tmp_path, text))

    assert project.parts[0].muted is False
    assert project.parts[0].soloed is True
