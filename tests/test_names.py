"""Pin the `<slug>:<role>` grammar.

This is the only interface between `midi2reaper`, which writes these names into
REAPER track titles, and this tool, which reads them back as training labels.
The two repos share no code, so a silent change to either side would mis-label
a corpus rather than fail loudly. These cases fix the grammar in place.
"""

import pytest

from reaper2mt3 import gm

# Names taken verbatim from projects midi2reaper generated.
REAL_TRACK_NAMES = [
    ("distortion-guitar:rhythm", 30, False, True),
    ("electric-guitar-clean:lead", 27, False, False),
    ("acoustic-guitar-steel:rhythm", 25, False, True),
    ("acoustic-guitar-nylon:lead", 24, False, False),
    ("overdriven-guitar:rhythm", 29, False, True),
    ("electric-guitar-jazz:lead", 26, False, False),
    ("electric-bass-finger:rhythm", 33, False, True),
    ("electric-bass-pick:rhythm", 34, False, True),
    ("fretless-bass:rhythm", 35, False, True),
    ("tenor-sax:lead", 66, False, False),
    ("choir-aahs:rhythm", 52, False, True),
    ("voice-oohs:lead", 53, False, False),
    ("synth-voice:lead", 54, False, False),
    ("rock-organ:rhythm", 18, False, True),
    ("drawbar-organ:rhythm", 16, False, True),
    ("electric-piano-1:rhythm", 4, False, True),
    ("acoustic-grand-piano:lead", 0, False, False),
    ("lead-8-bass-plus-lead:lead", 87, False, False),
    ("lead-3-calliope:lead", 82, False, False),
    ("orchestral-harp:lead", 46, False, False),
    ("trombone:rhythm", 57, False, True),
    ("clarinet:lead", 71, False, False),
    ("recorder:rhythm", 74, False, True),
    ("drums:rhythm", None, True, True),
]


@pytest.mark.parametrize("name,program,is_drum,rhythm", REAL_TRACK_NAMES)
def test_names_parse_to_expected_labels(name, program, is_drum, rhythm):
    assert gm.parse_track_name(name) == (program, is_drum, rhythm)


@pytest.mark.parametrize("name,program,is_drum,rhythm", REAL_TRACK_NAMES)
def test_names_are_reproduced_exactly(name, program, is_drum, rhythm):
    assert gm.track_name(program, is_drum, rhythm) == name


@pytest.mark.parametrize("program", range(128))
def test_every_program_round_trips(program):
    for rhythm in (True, False):
        name = gm.track_name(program, False, rhythm)
        assert gm.parse_track_name(name) == (program, False, rhythm)


def test_slugs_are_unique_across_all_programs():
    """Two programs sharing a slug would make labels ambiguous."""
    slugs = [gm.program_slug(p) for p in range(128)]
    assert len(set(slugs)) == 128


def test_human_suffix_is_ignored():
    name = "tenor-sax:lead | Kurt Cobain | Vocals [vocal→instrument]"
    assert gm.parse_track_name(name) == (66, False, False)


@pytest.mark.parametrize("name", ["Lead Guitar", "not-an-instrument:rhythm", "drums", "", "x:solo"])
def test_unparseable_names_are_rejected(name):
    assert gm.parse_track_name(name) is None
