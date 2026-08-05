import mido
import pytest

from reaper2mt3 import gm
from reaper2mt3.dataset import assign_splits, load_splits, write_corpus_midi, write_splits
from reaper2mt3.rppread import Note, Project, ProjectPart, _read_events


def make_project(parts, ppq=480, tempo_points=None):
    return Project(
        path=__import__("pathlib").Path("fake.RPP"),
        ppq=ppq,
        tempo_points=tempo_points or [(0.0, 120.0)],
        time_signature=(4, 4),
        parts=parts,
    )


def part(program, is_drum=False, rhythm=True, patch=0, notes=None):
    from pathlib import Path

    return ProjectPart(
        track_name=gm.track_name(program, is_drum, rhythm),
        program=program,
        is_drum=is_drum,
        rhythm=rhythm,
        soundfont=Path("/tmp/x.sf2"),
        bank=0,
        patch=patch,
        notes=notes if notes is not None else [Note(0, 480, 60, 100, 0)],
    )


@pytest.mark.parametrize(
    "program,is_drum,rhythm",
    [(30, False, True), (27, False, None), (66, False, None), (None, True, None)],
)
def test_track_name_round_trips(program, is_drum, rhythm):
    name = gm.track_name(program, is_drum, rhythm)
    assert gm.parse_track_name(name) == (program, is_drum, rhythm)


def test_track_name_parses_with_human_suffix():
    name = "tenor-sax | Kurt Cobain | Vocals [vocal→instrument]"
    assert gm.parse_track_name(name) == (66, False, None)


def test_unknown_track_name_rejected():
    assert gm.parse_track_name("Lead Guitar") is None
    assert gm.parse_track_name("not-an-instrument:rhythm") is None


def test_corpus_midi_carries_label_program_not_soundfont_patch(tmp_path):
    """The soundfont's patch number is arbitrary — Power Guitar 1.sf2 sits at
    patch 0 — so the label must come from the canonical name, not the preset."""
    project = make_project([part(30, patch=0)])
    out = tmp_path / "ex.mid"
    write_corpus_midi(project, out)

    midi = mido.MidiFile(str(out))
    programs = [m.program for t in midi.tracks for m in t if m.type == "program_change"]
    assert programs == [30]


def test_corpus_midi_puts_drums_on_channel_ten(tmp_path):
    project = make_project([part(None, is_drum=True)])
    out = tmp_path / "ex.mid"
    write_corpus_midi(project, out)

    midi = mido.MidiFile(str(out))
    channels = {m.channel for t in midi.tracks for m in t if m.type == "note_on"}
    assert channels == {9}
    assert not [m for t in midi.tracks for m in t if m.type == "program_change"]


def test_corpus_midi_keeps_melodic_parts_off_channel_ten(tmp_path):
    project = make_project([part(24 + i, rhythm=bool(i % 2)) for i in range(12)])
    out = tmp_path / "ex.mid"
    write_corpus_midi(project, out)

    midi = mido.MidiFile(str(out))
    channels = {m.channel for t in midi.tracks for m in t if m.type == "note_on"}
    assert 9 not in channels


def test_split_is_proportional_and_never_empty():
    """Thresholding a per-item hash gave an empty test split on nine sources."""
    names = [f"song_{i}" for i in range(9)]
    splits = assign_splits(names, 0.2)
    assert sum(v == "test" for v in splits.values()) == 2
    assert sum(v == "train" for v in splits.values()) == 7


def test_split_is_stable_and_order_independent():
    names = [f"song_{i}" for i in range(9)]
    assert assign_splits(names, 0.2) == assign_splits(list(reversed(names)), 0.2)


def test_split_leaves_train_non_empty_for_one_source():
    assert assign_splits(["only"], 0.5) == {"only": "train"}


def test_growing_the_corpus_does_not_reassign_existing_splits():
    """The regression this exists to close (PROJECT_LOG.md, 2026-08-05,
    'Corpus rebuilt from reviewed -REV projects'): ranking the whole set by
    hash makes the train/test boundary shift purely because the total count
    changed, silently moving a source that was never touched between splits.

    `song_6` is not an arbitrary choice: with the pre-`existing` algorithm,
    7 sources at test_fraction=0.2 round to a 1-source test slice that
    excludes it, but 8 sources round to a 2-source slice that includes it --
    a real flip caused only by one unrelated source arriving, not by any
    decision about `song_6` itself. Confirmed by brute force against the old
    implementation before writing this test, so a regression here is
    guaranteed to be caught rather than passing by coincidence.
    """
    names = [f"song_{i}" for i in range(7)]
    first = assign_splits(names, 0.2)
    assert first["song_6"] == "train"

    grown = names + ["song_7"]
    without_freezing = assign_splits(grown, 0.2)
    assert without_freezing["song_6"] == "test"  # the bug, reproduced

    frozen = assign_splits(grown, 0.2, existing=first)
    assert frozen["song_6"] == "train"  # frozen at its original assignment


def test_new_arrivals_are_still_split_among_themselves():
    existing = {"old_1": "train", "old_2": "test"}
    new_names = [f"song_{i}" for i in range(9)]

    splits = assign_splits(list(existing) + new_names, 0.2, existing=existing)

    assert splits["old_1"] == "train"
    assert splits["old_2"] == "test"
    new_only = {name: splits[name] for name in new_names}
    assert sum(v == "test" for v in new_only.values()) == 2
    assert sum(v == "train" for v in new_only.values()) == 7


def test_a_source_dropped_from_existing_is_freshly_assignable():
    """`existing` only freezes ids it actually names -- a source id it has
    never seen is free to be assigned like any other new arrival."""
    assert assign_splits(["only"], 0.5, existing={}) == assign_splits(["only"], 0.5)


def test_write_splits_then_load_splits_round_trips(tmp_path):
    splits = {"song_a": "train", "song_b": "test"}

    write_splits(splits, tmp_path)

    assert load_splits(tmp_path) == splits


def test_load_splits_is_empty_for_a_dataset_root_with_no_splits_file(tmp_path):
    assert load_splits(tmp_path) == {}


def test_overlapping_identical_pitches_all_survive():
    """Merged parts stack the same pitch; FIFO pairing must keep every note."""
    lines = [
        "E 0 90 3c 64",
        "E 10 90 3c 50",
        "E 10 80 3c 00",
        "E 10 80 3c 00",
    ]
    notes, _ = _read_events(lines, 0)
    assert len(notes) == 2
    assert sorted((n.start, n.end) for n in notes) == [(0, 20), (10, 30)]


def test_tempo_events_follow_the_envelope():
    project = make_project([part(30)], ppq=480, tempo_points=[(0.0, 120.0), (2.0, 60.0)])
    events = project.tempo_events()
    assert events[0] == (0, 500_000)
    # two seconds at 120 bpm is four beats
    assert events[1][0] == 4 * 480
    assert events[1][1] == 1_000_000


def test_seconds_at_inverts_tempo_events():
    project = make_project([part(30)], ppq=480, tempo_points=[(0.0, 120.0), (2.0, 60.0)])
    assert project.seconds_at(4 * 480) == pytest.approx(2.0)
    assert project.seconds_at(8 * 480) == pytest.approx(6.0)  # 4 beats at 60 bpm
