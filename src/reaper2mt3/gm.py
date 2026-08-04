"""General MIDI program names and the canonical part-name grammar.

This is the interface between the two tools: `midi2reaper` writes REAPER track
names as `<slug>:<role>`, and this module reads them back. The slugs derive
mechanically from the General MIDI program list, which is a fixed standard, so
both sides agree without sharing code. `tests/test_names.py` pins the grammar.
"""

from __future__ import annotations

import re

PROGRAM_NAMES = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
    "Clavi", "Celesta", "Glockenspiel", "Music Box", "Vibraphone", "Marimba",
    "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ",
    "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
    "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)", "Electric Guitar (jazz)",
    "Electric Guitar (clean)", "Electric Guitar (muted)", "Overdriven Guitar",
    "Distortion Guitar", "Guitar Harmonics", "Acoustic Bass",
    "Electric Bass (finger)", "Electric Bass (pick)", "Fretless Bass",
    "Slap Bass 1", "Slap Bass 2", "Synth Bass 1", "Synth Bass 2", "Violin",
    "Viola", "Cello", "Contrabass", "Tremolo Strings", "Pizzicato Strings",
    "Orchestral Harp", "Timpani", "String Ensemble 1", "String Ensemble 2",
    "Synth Strings 1", "Synth Strings 2", "Choir Aahs", "Voice Oohs",
    "Synth Voice", "Orchestra Hit", "Trumpet", "Trombone", "Tuba",
    "Muted Trumpet", "French Horn", "Brass Section", "Synth Brass 1",
    "Synth Brass 2", "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax",
    "Oboe", "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute",
    "Recorder", "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle",
    "Ocarina", "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
    "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)", "Lead 7 (fifths)",
    "Lead 8 (bass + lead)", "Pad 1 (new age)", "Pad 2 (warm)",
    "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)", "Pad 6 (metallic)",
    "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)", "FX 2 (soundtrack)",
    "FX 3 (crystal)", "FX 4 (atmosphere)", "FX 5 (brightness)",
    "FX 6 (goblins)", "FX 7 (echoes)", "FX 8 (sci-fi)", "Sitar", "Banjo",
    "Shamisen", "Koto", "Kalimba", "Bag pipe", "Fiddle", "Shanai",
    "Tinkle Bell", "Agogo", "Steel Drums", "Woodblock", "Taiko Drum",
    "Melodic Tom", "Synth Drum", "Reverse Cymbal", "Guitar Fret Noise",
    "Breath Noise", "Seashore", "Bird Tweet", "Telephone Ring", "Helicopter",
    "Applause", "Gunshot",
]

DRUM_SLUG = "drums"

# Guitar anchors corpus membership: an example without one is rejected.
GUITAR_PROGRAMS = range(24, 32)


def program_name(program: int) -> str:
    return PROGRAM_NAMES[program]


def slug(text: str) -> str:
    text = text.lower().replace("+", " plus ")
    return re.sub(r"[^a-z0-9]+", "-", text).strip("-")


def program_slug(program: int) -> str:
    return slug(PROGRAM_NAMES[program])


def is_guitar(program: int | None) -> bool:
    return program is not None and program in GUITAR_PROGRAMS


def track_name(program: int | None, is_drum: bool, rhythm: bool | None) -> str:
    """Canonical part name required by DATA_CONTRACT.md.

    Guitars are named `<slug>:<role>`; everything else is named `<slug>` alone,
    because role is a guitar-only annotation. The absence of a colon is what
    says "this part takes no role", so it is not an omission to tidy up.
    """
    part = DRUM_SLUG if is_drum else program_slug(program)
    if rhythm is None:
        return part
    return f"{part}:{'rhythm' if rhythm else 'lead'}"


_SLUG_TO_PROGRAM = {program_slug(p): p for p in range(128)}


def parse_track_name(name: str) -> tuple[int | None, bool, bool | None] | None:
    """Inverse of `track_name`: returns (program, is_drum, rhythm).

    `rhythm` is None for a part that carries no role. Only the canonical prefix
    is read, so anything appended for human benefit — the source track, a
    `[vocal→instrument]` marker — is ignored and a project renamed in REAPER
    still parses.
    """
    head = name.split("|", 1)[0].strip()
    slug_part, separator, role = head.partition(":")
    slug_part = slug_part.strip()

    if separator:
        role = role.strip().split()[0] if role.strip() else ""
        if role not in ("rhythm", "lead"):
            return None
        rhythm: bool | None = role == "rhythm"
    else:
        rhythm = None

    if slug_part == DRUM_SLUG:
        # Percussion is not a guitar, so it may not carry a role either.
        return (None, True, None) if rhythm is None else None
    program = _SLUG_TO_PROGRAM.get(slug_part)
    if program is None:
        return None
    # Only guitars may carry a role, and a guitar without one is unlabelled.
    if (rhythm is not None) != is_guitar(program):
        return None
    return program, False, rhythm
