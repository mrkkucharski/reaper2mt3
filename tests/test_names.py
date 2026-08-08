"""Pin the part-name grammar.

This is the only interface between `midi2reaper`, which writes these names into
REAPER track titles, and this tool, which reads them back as training labels.
The two repos share no code, so a silent change to either side would mis-label
a corpus rather than fail loudly. These cases fix the grammar in place.

A guitar playing chordal accompaniment is named `<slug>:rhythm`; every other
part is named `<slug>` alone. There is deliberately no `lead`.
"""

import pytest

from reaper2mt3 import gm

# Names taken verbatim from projects midi2reaper generated.
REAL_TRACK_NAMES = [
    ("acoustic-grand-piano", 0, False, None),
    ("acoustic-guitar-nylon", 24, False, None),
    ("acoustic-guitar-nylon:rhythm", 24, False, True),
    ("acoustic-guitar-steel", 25, False, None),
    ("acoustic-guitar-steel:rhythm", 25, False, True),
    ("choir-aahs", 52, False, None),
    ("clarinet", 71, False, None),
    ("distortion-guitar", 30, False, None),
    ("distortion-guitar:rhythm", 30, False, True),
    ("drawbar-organ", 16, False, None),
    ("drums", None, True, None),
    ("electric-bass-finger", 33, False, None),
    ("electric-bass-pick", 34, False, None),
    ("electric-guitar-clean", 27, False, None),
    ("electric-guitar-clean:rhythm", 27, False, True),
    ("electric-guitar-jazz", 26, False, None),
    ("electric-guitar-jazz:rhythm", 26, False, True),
    ("electric-piano-1", 4, False, None),
    ("fretless-bass", 35, False, None),
    ("lead-3-calliope", 82, False, None),
    ("lead-8-bass-plus-lead", 87, False, None),
    ("orchestral-harp", 46, False, None),
    ("overdriven-guitar", 29, False, None),
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
    name = gm.track_name(program, False, None)
    assert gm.parse_track_name(name) == (program, False, None)
    if gm.is_guitar(program):
        annotated = gm.track_name(program, False, True)
        assert gm.parse_track_name(annotated) == (program, False, True)


def test_slugs_are_unique_across_all_programs():
    """Two programs sharing a slug would make labels ambiguous."""
    slugs = [gm.program_slug(p) for p in range(128)]
    assert len(set(slugs)) == 128


def test_human_suffix_is_ignored():
    name = "electric-guitar-clean:rhythm | John Frusciante | Gretsch White Falcon"
    assert gm.parse_track_name(name) == (27, False, True)


@pytest.mark.parametrize("name", [
    "distortion-guitar:rhythm trailing text",
    "distortion-guitar:rhythm | human display name",
])
def test_strict_corpus_name_rejects_nonterminal_rhythm_suffix(name):
    assert gm.parse_track_name(name) == (30, False, True)
    assert gm.parse_track_name(name, strict_corpus_name=True) is None


def test_vocal_marker_suffix_is_ignored():
    assert gm.parse_track_name("tenor-sax | Vocals [vocal→instrument]") == (66, False, None)


@pytest.mark.parametrize("name", ["Lead Guitar", "not-an-instrument:rhythm", "", "x:solo"])
def test_unparseable_names_are_rejected(name):
    assert gm.parse_track_name(name) is None


@pytest.mark.parametrize("name", ["tenor-sax:rhythm", "drums:rhythm", "acoustic-grand-piano:rhythm"])
def test_rhythm_on_a_non_guitar_is_rejected(name):
    """rhythm is guitar-only, so it on anything else is a corrupt label rather
    than something to accept quietly."""
    assert gm.parse_track_name(name) is None


@pytest.mark.parametrize(
    "name", ["electric-guitar-clean:lead", "distortion-guitar:lead", "tenor-sax:lead"]
)
def test_lead_is_not_part_of_the_grammar(name):
    """Absence of `:rhythm` is the only way a part says it is not accompaniment;
    an explicit `:lead` is a label from an older grammar and is rejected."""
    assert gm.parse_track_name(name) is None


@pytest.mark.parametrize("name", ["electric-guitar-clean", "distortion-guitar"])
def test_unannotated_guitar_is_valid(name):
    """A guitar that is not chordal accompaniment simply carries no annotation."""
    program, is_drum, rhythm = gm.parse_track_name(name)
    assert gm.is_guitar(program) and not is_drum and rhythm is None
