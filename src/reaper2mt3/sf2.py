"""Read a SoundFont's preset table.

Only enough to satisfy acceptance check 10 — that the bank and patch a project
names really exist in the file it points at. Choosing soundfonts is
`midi2reaper`'s job; this module only confirms the choice is still valid.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

PHDR_RECORD_SIZE = 38


@dataclass(frozen=True)
class Preset:
    bank: int
    patch: int
    name: str


def read_presets(path: Path) -> list[Preset] | None:
    """The file's presets, or None if it is not a readable SoundFont.

    Sample data is never read; only the `phdr` chunk is parsed, so this stays
    cheap even on large banks.
    """
    try:
        with open(path, "rb") as handle:
            if handle.read(4) != b"RIFF":
                return None
            handle.seek(12)  # skip RIFF size and the 'sfbk' form type
            while True:
                header = handle.read(8)
                if len(header) < 8:
                    return None
                chunk_id, size = struct.unpack("<4sI", header)
                if chunk_id != b"LIST":
                    handle.seek(size + (size & 1), 1)
                    continue
                if handle.read(4) != b"pdta":
                    handle.seek(size - 4 + (size & 1), 1)
                    continue
                end = handle.tell() + size - 4
                while handle.tell() < end:
                    sub = handle.read(8)
                    if len(sub) < 8:
                        return None
                    sub_id, sub_size = struct.unpack("<4sI", sub)
                    if sub_id == b"phdr":
                        return _parse_phdr(handle.read(sub_size))
                    handle.seek(sub_size + (sub_size & 1), 1)
                return None
    except (OSError, struct.error):
        return None


def _parse_phdr(raw: bytes) -> list[Preset]:
    presets = []
    for offset in range(0, len(raw), PHDR_RECORD_SIZE):
        record = raw[offset : offset + PHDR_RECORD_SIZE]
        if len(record) < PHDR_RECORD_SIZE:
            break
        name = record[:20].split(b"\0")[0].decode("latin-1").strip()
        patch, bank = struct.unpack("<HH", record[20:24])
        presets.append(Preset(bank=bank, patch=patch, name=name))
    return presets[:-1]  # drop the terminal EOP record
