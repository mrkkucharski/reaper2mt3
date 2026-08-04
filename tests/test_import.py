"""`import_example` pairs an already-rendered WAV with labels from its RPP.

No audio is synthesized here -- render.py's FluidSynth path is never called.
These tests pin that: the imported WAV bytes are checked against the source
file, not against anything render.py could have produced.
"""

import wave
from pathlib import Path

import pytest

from reaper2mt3.dataset import import_example
from reaper2mt3.rppread import Note, Project, ProjectPart


def make_project(parts, ppq=480, tempo_points=None, path=None):
    return Project(
        path=path or Path("fake.RPP"),
        ppq=ppq,
        tempo_points=tempo_points or [(0.0, 120.0)],
        time_signature=(4, 4),
        parts=parts,
        unparsed_tracks=[],
    )


def guitar_part(program=30, rhythm=True, plugins=None, notes=None):
    from reaper2mt3 import gm

    return ProjectPart(
        track_name=gm.track_name(program, False, rhythm),
        program=program,
        is_drum=False,
        rhythm=rhythm,
        soundfont=None,
        bank=None,
        patch=None,
        instrument_plugins=plugins if plugins is not None else ["Kontakt 8", "Guitar Rig 7"],
        notes=notes if notes is not None else [Note(0, 480, 52, 100, 0)],
    )


def write_wav(path: Path, seconds: float, channels=1, sampwidth=2, rate=44100):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "w") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sampwidth)
        handle.setframerate(rate)
        frame_count = int(seconds * rate)
        handle.writeframes((b"\x10\x20" * channels) * frame_count)
    return path


def write_rpp(path: Path, content: str = "fake project content") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return path


def test_import_copies_wav_without_rendering(tmp_path):
    """The defining property: the output WAV is a byte-for-byte copy of the
    source, never a render.py product."""
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    project = make_project([guitar_part()], path=rpp)

    out = tmp_path / "dataset"
    example = import_example(project, wav, out, "ex_0001", "train", "REAPER (manual)")

    written = out / example.record["audio_path"]
    assert written.read_bytes() == wav.read_bytes()


def test_import_never_touches_render_module(tmp_path, monkeypatch):
    """Belt and braces: importing must not call the renderer at all. Patched on
    the names as bound into dataset.py's own namespace, since that is what
    import_example actually calls."""
    import reaper2mt3.dataset as dataset_module

    def _forbidden(*args, **kwargs):
        raise AssertionError("import_example must not call the renderer")

    monkeypatch.setattr(dataset_module, "render_part", _forbidden)
    monkeypatch.setattr(dataset_module, "mix", _forbidden)

    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    project = make_project([guitar_part()], path=rpp)

    import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "REAPER (manual)")


def test_import_writes_corpus_midi_from_project(tmp_path):
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    project = make_project([guitar_part(program=30, rhythm=True)], path=rpp)

    out = tmp_path / "dataset"
    example = import_example(project, wav, out, "ex_0001", "train", "REAPER (manual)")

    import mido
    midi = mido.MidiFile(str(out / example.record["midi_path"]))
    names = [m.name for t in midi.tracks for m in t if m.type == "track_name"]
    assert "distortion-guitar:rhythm" in names


def test_import_records_chain_provenance_not_soundfont(tmp_path):
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    project = make_project([guitar_part(plugins=["Kontakt 8", "Guitar Rig 7"])], path=rpp)

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    part = example.record["parts"][0]
    assert part["soundfont"] is None
    assert part["instrument_plugins"] == ["Kontakt 8", "Guitar Rig 7"]


def test_import_flags_part_with_neither_soundfont_nor_chain(tmp_path):
    """A part with nothing driving it is a real problem, not silence to ignore."""
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    project = make_project([guitar_part(plugins=[])], path=rpp)

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    assert any("check 10" in p for p in example.problems)


@pytest.mark.parametrize("channels,sampwidth,rate", [(2, 2, 44100), (1, 3, 44100), (1, 2, 48000)])
def test_import_flags_wav_that_does_not_meet_the_contract(tmp_path, channels, sampwidth, rate):
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0, channels=channels,
                    sampwidth=sampwidth, rate=rate)
    project = make_project([guitar_part()], path=rpp)

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    assert any("check 6" in p for p in example.problems)


def test_import_accepts_a_conforming_wav(tmp_path):
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0, channels=1, sampwidth=2, rate=44100)
    project = make_project([guitar_part()], path=rpp)

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    assert not any("check 6" in p for p in example.problems)


def test_import_flags_audio_ending_well_before_the_performance(tmp_path):
    """A render cut off mid-performance must not pass silently."""
    rpp = write_rpp(tmp_path / "song.RPP")
    long_notes = [Note(i * 480, i * 480 + 480, 52, 100, 0) for i in range(200)]  # ~100s @ 120bpm
    project = make_project([guitar_part(notes=long_notes)], path=rpp)
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)  # nowhere near 100s

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    assert any("check 12" in p and "truncated" in p for p in example.problems)


def test_import_flags_audio_running_far_past_the_performance(tmp_path):
    rpp = write_rpp(tmp_path / "song.RPP")
    project = make_project([guitar_part(notes=[Note(0, 480, 52, 100, 0)])], path=rpp)  # ~0.25s
    wav = write_wav(tmp_path / "song.wav", seconds=60.0)

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    assert any("check 12" in p and "misalignment" in p for p in example.problems)


def test_out_of_range_pitch_is_flagged_without_an_exception(tmp_path):
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    low_note = [Note(0, 480, 17, 100, 0)]  # below the accepted guitar range
    project = make_project([guitar_part(notes=low_note)], path=rpp)

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    assert any("check 8" in p for p in example.problems)


def test_approved_pitch_exception_clears_check_8(tmp_path):
    """DATA_CONTRACT.md's escape hatch: material outside the accepted range is
    rejected unless explicitly approved and recorded as an exception."""
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    low_note = [Note(0, 480, 17, 100, 0)]
    project = make_project([guitar_part(notes=low_note)], path=rpp)
    track_name = project.parts[0].canonical_name
    exceptions = {track_name: {"extra_pitches": [17], "reason": "test"}}

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r",
                             pitch_exceptions=exceptions)
    assert not any("check 8" in p for p in example.problems)


def test_pitch_exception_is_recorded_in_the_manifest(tmp_path):
    """The approval must be auditable from the manifest alone, not only from
    whatever produced it, so a later `reaper2mt3 check` still honours it."""
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    project = make_project([guitar_part(notes=[Note(0, 480, 17, 100, 0)])], path=rpp)
    track_name = project.parts[0].canonical_name
    exceptions = {track_name: {"extra_pitches": [17], "reason": "deliberate low hit"}}

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r",
                             pitch_exceptions=exceptions)
    assert example.record["approved_exceptions"] == exceptions


def test_exception_does_not_widen_the_range_for_other_pitches(tmp_path):
    """Approving pitch 17 must not accidentally clear an unrelated pitch 5."""
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    notes = [Note(0, 480, 17, 100, 0), Note(480, 960, 5, 100, 0)]
    project = make_project([guitar_part(notes=notes)], path=rpp)
    track_name = project.parts[0].canonical_name
    exceptions = {track_name: {"extra_pitches": [17], "reason": "test"}}

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r",
                             pitch_exceptions=exceptions)
    assert any("check 8" in p and "1 notes" in p for p in example.problems)


def test_exception_on_one_track_does_not_cover_another(tmp_path):
    rpp = write_rpp(tmp_path / "song.RPP")
    wav = write_wav(tmp_path / "song.wav", seconds=1.0)
    project = make_project([guitar_part(notes=[Note(0, 480, 17, 100, 0)])], path=rpp)
    exceptions = {"some-other-track": {"extra_pitches": [17], "reason": "test"}}

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r",
                             pitch_exceptions=exceptions)
    assert any("check 8" in p for p in example.problems)


def test_import_accepts_a_short_reasonable_tail(tmp_path):
    rpp = write_rpp(tmp_path / "song.RPP")
    project = make_project([guitar_part(notes=[Note(0, 480, 52, 100, 0)])], path=rpp)  # ~0.25s
    wav = write_wav(tmp_path / "song.wav", seconds=3.0)  # a few seconds' tail, not 30+

    example = import_example(project, wav, tmp_path / "dataset", "ex_0001", "train", "r")
    assert not any("check 12" in p for p in example.problems)
