"""Pin the part-name grammar.

This is the only interface between `midi2reaper`, which writes these names into
REAPER track titles, and this tool, which reads them back as training labels.
The two repos share no code, so a silent change to either side would mis-label
a corpus rather than fail loudly. These cases fix the grammar in place.

Guitars are named `<slug>:<role>`; every other part is named `<slug>` alone,
because role is a guitar-only annotation.
"""

import pytest

from reaper2mt3 import gm

# Names taken verbatim from projects midi2reaper generated.
REAL_TRACK_NAMES = [
    ("acoustic-grand-piano", 0, False, None),
    ("acoustic-guitar-nylon:lead", 24, False, False),
    ("acoustic-guitar-nylon:rhythm", 24, False, True),
    ("acoustic-guitar-steel:lead", 25, False, False),
    ("acoustic-guitar-steel:rhythm", 25, False, True),
    ("choir-aahs", 52, False, None),
    ("clarinet", 71, False, None),
    ("distortion-guitar:lead", 30, False, False),
    ("distortion-guitar:rhythm", 30, False, True),
    ("drawbar-organ", 16, False, None),
    ("drums", None, True, None),
    ("electric-bass-finger", 33, False, None),
    ("electric-bass-pick", 34, False, None),
    ("electric-guitar-clean:lead", 27, False, False),
    ("electric-guitar-clean:rhythm", 27, False, True),
    ("electric-guitar-jazz:lead", 26, False, False),
    ("electric-guitar-jazz:rhythm", 26, False, True),
    ("electric-piano-1", 4, False, None),
    ("fretless-bass", 35, False, None),
    ("lead-3-calliope", 82, False, None),
    ("lead-8-bass-plus-lead", 87, False, None),
    ("orchestral-harp", 46, False, None),
    ("overdriven-guitar:lead", 29, False, False),
    ("overdriven-guitar:rhythm", 29, False, True),
    ("recorder", 74, False, None),
    ("rock-organ", 18, False, None),
    ("synth-voice", 54, False, None),
    ("tenor-sax", 66, False, None),
    ("trombone", 57, False, None),
    ("viola", 41, False, None),
    ("voice-oohs", 53, False, None),
]


@pytest.mark.parametrize("name,program,is_drum,rhythm", REAL_TRACK_NAMES)
def test_names_parse_to_expected_labels(name, program, is_drum, rhythm):
    assert gm.parse_track_name(name) == (program, is_drum, rhythm)


@pytest.mark.parametrize("name,program,is_drum,rhythm", REAL_TRACK_NAMES)
def test_names_are_reproduced_exactly(name, program, is_drum, rhythm):
    assert gm.track_name(program, is_drum, rhythm) == name


@pytest.mark.parametrize("program", range(128))
def test_every_program_round_trips(program):
    if gm.is_guitar(program):
        for rhythm in (True, False):
            name = gm.track_name(program, False, rhythm)
            assert gm.parse_track_name(name) == (program, False, rhythm)
    else:
        name = gm.track_name(program, False, None)
        assert gm.parse_track_name(name) == (program, False, None)


def test_slugs_are_unique_across_all_programs():
    """Two programs sharing a slug would make labels ambiguous."""
    slugs = [gm.program_slug(p) for p in range(128)]
    assert len(set(slugs)) == 128


def test_human_suffix_is_ignored():
    name = "electric-guitar-clean:lead | John Frusciante | Gretsch White Falcon"
    assert gm.parse_track_name(name) == (27, False, False)


def test_vocal_marker_suffix_is_ignored():
    assert gm.parse_track_name("tenor-sax | Vocals [vocal→instrument]") == (66, False, None)


@pytest.mark.parametrize("name", ["Lead Guitar", "not-an-instrument:rhythm", "", "x:solo"])
def test_unparseable_names_are_rejected(name):
    assert gm.parse_track_name(name) is None


@pytest.mark.parametrize("name", ["tenor-sax:rhythm", "drums:rhythm", "acoustic-grand-piano:lead"])
def test_role_on_a_non_guitar_is_rejected(name):
    """Role is guitar-only, so a role on anything else is a corrupt label rather
    than something to accept quietly."""
    assert gm.parse_track_name(name) is None


@pytest.mark.parametrize("name", ["electric-guitar-clean", "distortion-guitar"])
def test_guitar_without_a_role_is_rejected(name):
    """Guitars must carry a role; one without is equally a corrupt label."""
    assert gm.parse_track_name(name) is None
