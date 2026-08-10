import hashlib
import json
from pathlib import Path

import mido
import pytest

from reaper2mt3.finalization import FinalizationInput, load_finalization_input, preflight, provenance_record
from reaper2mt3.rppread import Note, Project, ProjectPart


def _sidecar(*, aliases=None, policy=None, label_ppq=960, label_notes=None):
    return FinalizationInput(
        {
            "schema_version": "procgen.reaper-render-provenance/v1",
            "repositories": {name: "a" * 40 for name in ("procgen", "mt3", "midi2reaper", "reaper2mt3")},
            "template": {"id": "midi2reaper/sflt-v1", "version": "1"},
            "vst_profiles": [{"id": "procgen/test/v1", "version": "1"}],
            "renderer_tracks": aliases or [],
            "event_policy": policy or {},
        },
        label_ppq,
        label_notes or {},
    )


def _part(notes, name="distortion-guitar:rhythm"):
    return ProjectPart(name, 30, False, True, None, None, None, notes=notes)


def _project(*parts, ppq=960):
    return Project(Path("fixture.RPP"), ppq, [(0.0, 120.0)], (4, 4), list(parts))


def test_preflight_removes_keyswitches_and_reports_source_cc_audit():
    part = _part([Note(0, 1, 12, 100, 0), Note(2, 3, 52, 100, 0)])
    part.source_cc_count = 3
    project = _project(part)
    label = [Note(2, 3, 52, 100, 0)]
    report = preflight(project, _sidecar(
        policy={"default": {"keyswitch_ranges": [[0, 20]], "keyswitch_action": "remove"}},
        label_notes={"distortion-guitar:rhythm": label},
    ))
    assert [note.pitch for note in project.parts[0].notes] == [52]
    assert report["source_event_audit"][0]["source_cc_events"] == 3
    assert report["source_event_audit"][0]["source_cc_action"] == "dropped"
    assert report["source_event_audit"][0]["keyswitch_action"] == "removed"


def test_preflight_rejects_keyswitch_under_reject_policy():
    with pytest.raises(ValueError, match="keyswitch"):
        preflight(_project(_part([Note(0, 1, 12, 100, 0)])), _sidecar(policy={"default": {
            "keyswitch_ranges": [[0, 20]], "keyswitch_action": "reject",
        }}))


def test_preflight_collapses_attested_identical_double_track():
    notes = [Note(0, 10, 52, 100, 0)]
    project = _project(_part(notes), _part(notes))
    label = [Note(0, 10, 52, 100, 0)]
    report = preflight(project, _sidecar(
        aliases=[
            {"renderer_track_id": "left", "authoritative_symbolic_part": "distortion-guitar:rhythm"},
            {"renderer_track_id": "right", "authoritative_symbolic_part": "distortion-guitar:rhythm"},
        ],
        label_notes={"distortion-guitar:rhythm": label},
    ))
    assert len(project.parts) == 1
    assert len(report["collapsed_renderer_duplicates"]) == 1


def test_preflight_rejects_unattested_duplicate_before_writing():
    project = _project(_part([Note(0, 10, 52, 100, 0)]), _part([Note(0, 10, 53, 100, 0)]))
    with pytest.raises(ValueError, match="unattested duplicate"):
        preflight(project, _sidecar())


def test_finalization_provenance_carries_all_revisions_and_versions():
    record = provenance_record(_sidecar(), {"source_event_audit": [], "collapsed_renderer_duplicates": []}, "ex_0001")
    assert set(record["repositories"]) == {"procgen", "mt3", "midi2reaper", "reaper2mt3"}
    assert record["template"]["version"] == "1"
    assert record["vst_profiles"][0]["version"] == "1"


def test_preflight_sources_corpus_notes_from_label_midi_not_renderer_notes():
    """The RPP's notes are attack-latency-compensated and lead-in-shifted
    (#64); shipped ground truth must come from label_midi instead (#65)."""
    renderer_notes = [Note(90, 190, 52, 100, 0)]  # scheduled early to compensate
    label_notes = [Note(120, 220, 52, 100, 0)]  # the true, uncompensated onset
    project = _project(_part(renderer_notes))
    preflight(project, _sidecar(label_notes={"distortion-guitar:rhythm": label_notes}))
    assert project.parts[0].notes == label_notes


def test_preflight_rejects_label_midi_missing_a_part_track():
    project = _project(_part([Note(0, 10, 52, 100, 0)]))
    with pytest.raises(ValueError, match="no track for distortion-guitar:rhythm"):
        preflight(project, _sidecar(label_notes={}))


def test_preflight_rejects_label_midi_pitch_mismatch():
    project = _project(_part([Note(0, 10, 52, 100, 0)]))
    with pytest.raises(ValueError, match="do not match"):
        preflight(project, _sidecar(label_notes={"distortion-guitar:rhythm": [Note(0, 10, 53, 100, 0)]}))


def test_preflight_rejects_label_midi_ppq_mismatch():
    project = _project(_part([Note(0, 10, 52, 100, 0)]), ppq=960)
    with pytest.raises(ValueError, match="ppq"):
        preflight(project, _sidecar(label_ppq=480, label_notes={"distortion-guitar:rhythm": [Note(0, 10, 52, 100, 0)]}))


def _write_label_midi(path: Path) -> None:
    midi = mido.MidiFile(type=1, ticks_per_beat=960)
    conductor = mido.MidiTrack()
    midi.tracks.append(conductor)
    conductor.append(mido.MetaMessage("track_name", name="conductor", time=0))
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("track_name", name="distortion-guitar:rhythm", time=0))
    track.append(mido.Message("program_change", channel=0, program=30, time=0))
    track.append(mido.Message("note_on", channel=0, note=52, velocity=96, time=0))
    track.append(mido.Message("note_off", channel=0, note=52, velocity=0, time=480))
    midi.save(str(path))


def _provenance_json(*, label_path: Path, label_sha256: str) -> dict:
    return {
        "schema_version": "procgen.reaper-render-provenance/v1",
        "repositories": {name: "a" * 40 for name in ("procgen", "mt3", "midi2reaper", "reaper2mt3")},
        "template": {"id": "midi2reaper/sflt-v1", "version": "1"},
        "vst_profiles": [{"id": "procgen/test/v1", "version": "1"}],
        "renderer_tracks": [],
        "event_policy": {},
        "label_midi": {"path": str(label_path), "sha256": label_sha256},
    }


def test_load_finalization_input_parses_label_midi_notes(tmp_path):
    label_path = tmp_path / "label.mid"
    _write_label_midi(label_path)
    sha256 = hashlib.sha256(label_path.read_bytes()).hexdigest()
    sidecar_path = tmp_path / "render-provenance.json"
    sidecar_path.write_text(json.dumps(_provenance_json(label_path=label_path, label_sha256=sha256)))

    sidecar = load_finalization_input(sidecar_path)
    assert sidecar.label_ppq == 960
    assert sidecar.label_notes["distortion-guitar:rhythm"] == [Note(0, 480, 52, 96, 0)]
    assert "conductor" not in sidecar.label_notes


def test_load_finalization_input_rejects_label_midi_checksum_mismatch(tmp_path):
    label_path = tmp_path / "label.mid"
    _write_label_midi(label_path)
    sidecar_path = tmp_path / "render-provenance.json"
    sidecar_path.write_text(json.dumps(_provenance_json(label_path=label_path, label_sha256="0" * 64)))

    with pytest.raises(ValueError, match="SHA-256"):
        load_finalization_input(sidecar_path)


def test_load_finalization_input_rejects_missing_label_midi_field(tmp_path):
    sidecar_path = tmp_path / "render-provenance.json"
    record = _provenance_json(label_path=tmp_path / "label.mid", label_sha256="0" * 64)
    del record["label_midi"]
    sidecar_path.write_text(json.dumps(record))

    with pytest.raises(ValueError, match="label_midi"):
        load_finalization_input(sidecar_path)
