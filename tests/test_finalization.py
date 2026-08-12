from pathlib import Path

import pytest

from reaper2mt3.finalization import FinalizationInput, preflight, provenance_record
from reaper2mt3.rppread import Note, Project, ProjectPart


def _sidecar(*, aliases=None, policy=None):
    return FinalizationInput({
        "schema_version": "procgen.reaper-render-provenance/v1",
        "repositories": {name: "a" * 40 for name in ("procgen", "mt3", "midi2reaper", "reaper2mt3")},
        "template": {"id": "midi2reaper/sflt-v1", "version": "1"},
        "vst_profiles": [{"id": "procgen/test/v1", "version": "1"}],
        "renderer_tracks": aliases or [],
        "event_policy": policy or {},
    })


def _part(notes, name="distortion-guitar:rhythm"):
    return ProjectPart(name, 30, False, True, None, None, None, notes=notes)


def _project(*parts):
    return Project(Path("fixture.RPP"), 960, [(0.0, 120.0)], (4, 4), list(parts))


def test_preflight_removes_keyswitches_and_reports_source_cc_audit():
    part = _part([Note(0, 1, 12, 100, 0), Note(2, 3, 52, 100, 0)])
    part.source_cc_count = 3
    project = _project(part)
    report = preflight(project, _sidecar(policy={"default": {
        "keyswitch_ranges": [[0, 20]], "keyswitch_action": "remove",
    }}))
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
    report = preflight(project, _sidecar(aliases=[
        {"renderer_track_id": "left", "authoritative_symbolic_part": "distortion-guitar:rhythm"},
        {"renderer_track_id": "right", "authoritative_symbolic_part": "distortion-guitar:rhythm"},
    ]))
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
