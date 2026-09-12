"""Advisory note-level checks.

These never fail lint. A slide ramp is unambiguous and blocks import; an
isolated very short note might be a dead/muted note the arranger meant, so it
is surfaced for a human to look at rather than acted on.
"""

from reaper2mt3 import noteqa
from reaper2mt3.rppread import read_project

from test_slides import PPQ, STEP, DUR, LONG, _write, _ramp


def _short_notes(tmp_path, notes):
    path = _write(tmp_path, notes)
    project = read_project(path)
    ordered = sorted(project.parts[0].notes, key=lambda n: (n.start, n.pitch))
    return noteqa.find_isolated_short_notes(ordered, project.seconds_at)


def test_isolated_sliver_is_reported(tmp_path):
    """~15 ms with clear space after it: a click, not a played note."""
    notes = [(0, LONG, 60, 0x5F),          # a normal note
             (LONG, LONG + 15, 45, 0x5F),  # ~15 ms sliver, then silence
             (LONG + 900, LONG + 900 + LONG, 62, 0x5F)]
    hits = _short_notes(tmp_path, notes)
    assert len(hits) == 1
    assert hits[0].pitches == (45,)
    assert hits[0].duration_ms < 30
    assert hits[0].gap_after_ms > 80


def test_simultaneous_slivers_count_as_one_place(tmp_path):
    """A two-note dead-note stab is one thing to look at, not two."""
    notes = [(0, LONG, 60, 0x5F),
             (LONG, LONG + 15, 40, 0x5F),
             (LONG + 2, LONG + 17, 45, 0x5F),
             (LONG + 900, LONG + 900 + LONG, 62, 0x5F)]
    hits = _short_notes(tmp_path, notes)
    assert len(hits) == 1
    assert hits[0].pitches == (40, 45)


def test_short_notes_inside_a_fast_figure_are_not_reported(tmp_path):
    """Shortness alone is not the signal -- a fast run is made of short notes.

    Without the isolation requirement this would flag every step of every
    ramp, which is noise rather than review.
    """
    hits = _short_notes(tmp_path, _ramp(0, 60, 10))
    assert hits == []


def test_a_short_note_butted_against_neighbours_is_not_reported(tmp_path):
    """Space on at least one side is what makes it stand out."""
    notes = [(0, 200, 60, 0x5F),
             (200, 215, 61, 0x5F),        # short, but hemmed in on both sides
             (215, 415, 62, 0x5F)]
    hits = _short_notes(tmp_path, notes)
    assert hits == []


def test_normal_length_isolated_note_is_not_reported(tmp_path):
    notes = [(0, LONG, 60, 0x5F),
             (LONG + 900, LONG + 900 + LONG, 62, 0x5F)]
    hits = _short_notes(tmp_path, notes)
    assert hits == []


def test_drum_tracks_are_skipped(tmp_path):
    """Drum notes are all short by nature; flagging them would be pure noise."""
    notes = [(0, LONG, 60, 0x5F),
             (LONG, LONG + 15, 45, 0x5F),
             (LONG + 900, LONG + 900 + LONG, 62, 0x5F)]
    path = _write(tmp_path, notes, name="drums")
    project = read_project(path)
    assert all(p.is_drum for p in project.parts) or not project.parts
    assert noteqa.scan_project(project) == {}


def _figures(tmp_path, notes, covered=frozenset()):
    path = _write(tmp_path, notes)
    project = read_project(path)
    ordered = sorted(project.parts[0].notes, key=lambda n: (n.start, n.pitch))
    return noteqa.find_small_slide_figures(
        ordered, project.seconds_at, project.ppq, set(covered))


def test_a_two_step_chord_fall_is_reported_as_a_small_figure(tmp_path):
    """Hey You bar 28.4: a triad falling two semitones off a held chord.

    Below RUN_MIN_STEPS and SLIDE_MIN_SEMITONES, so `fix-slides` will never
    touch it -- which is exactly why it has to be surfaced rather than
    silently left behind.
    """
    step = 40                                       # ~42 ms, finer than a 32nd
    notes = [(0, LONG, 60, 0x5F), (0, LONG, 64, 0x5F)]           # held chord
    for k, (a, b) in enumerate(((59, 63), (58, 62))):
        notes += [(LONG + k * step, LONG + (k + 1) * step, a, 0x5F),
                  (LONG + k * step, LONG + (k + 1) * step, b, 0x5F)]
    hits = _figures(tmp_path, notes)
    assert len(hits) == 1
    assert (hits[0].steps, hits[0].voices, hits[0].span) == (2, 2, 1)
    assert hits[0].pitches == (59, 63)               # the chord it falls from


def test_parallel_voices_do_not_hide_the_figure(tmp_path):
    """Interleaved voices read 59, 63, 58, 62 -- never stepwise note by note.

    The main detector learned this once already; the advisory has to group by
    chord too or multi-voice falls stay invisible.
    """
    step = 40
    notes = [(0, LONG, 60, 0x5F), (0, LONG, 64, 0x5F)]
    for k, (a, b) in enumerate(((59, 63), (58, 62))):
        notes += [(LONG + k * step, LONG + (k + 1) * step, a, 0x5F),
                  (LONG + k * step, LONG + (k + 1) * step, b, 0x5F)]
    ordered = sorted((n for n in notes), key=lambda n: (n[0], n[2]))
    assert [n[2] for n in ordered][2:] == [59, 63, 58, 62], "the trap is present"
    assert len(_figures(tmp_path, notes)) == 1


def test_a_figure_already_handled_by_the_converter_is_not_repeated(tmp_path):
    """Whatever `fix-slides` converts or skips must not warn as well."""
    notes = _ramp(LONG, 60, 3)
    notes.insert(0, (0, LONG, 59, 0x5F))            # a sustained origin
    plain = _figures(tmp_path, notes)
    assert plain, "the figure is reportable on its own"
    covered = {i for i, _ in enumerate(sorted(
        read_project(_write(tmp_path, notes)).parts[0].notes,
        key=lambda n: (n.start, n.pitch)))}
    assert _figures(tmp_path, notes, covered) == []


def test_isolated_notes_with_no_sustained_neighbour_are_not_figures(tmp_path):
    """Two stray short notes are not a slide without something to slide from."""
    step = 40
    notes = [(0, step, 59, 0x5F), (step, 2 * step, 58, 0x5F),
             (LONG * 4, LONG * 5, 40, 0x5F)]
    assert _figures(tmp_path, notes) == []


def test_position_is_reported_in_reaper_transport_format(tmp_path):
    notes = [(0, LONG, 60, 0x5F),
             (LONG, LONG + 15, 45, 0x5F),
             (LONG + 900, LONG + 900 + LONG, 62, 0x5F)]
    hits = _short_notes(tmp_path, notes)
    # One quarter note in at 4/4 is bar 1, beat 2.
    assert hits[0].where(PPQ, (4, 4)).startswith("bar 1.2.00 @ 0:00.5")


def test_position_honours_the_time_signature_denominator(tmp_path):
    """Several projects here are in 3/16, where a bar is 0.75 QN, not 3.

    Assuming a /4 denominator put every reported position in such a project
    out by a factor of four -- caught against REAPER's own ruler.
    """
    notes = [(0, LONG, 60, 0x5F),
             (LONG, LONG + 15, 45, 0x5F),
             (LONG + 900, LONG + 900 + LONG, 62, 0x5F)]
    hits = _short_notes(tmp_path, notes)
    # A quarter note is four sixteenths, so in 3/16 that is bar 2, beat 2.
    assert hits[0].where(PPQ, (3, 16)).startswith("bar 2.2.00 @")
    # ...where the same tick in 4/4 is only bar 1, beat 2.
    assert hits[0].where(PPQ, (4, 4)).startswith("bar 1.2.00 @")
