"""`_find_audio` picks whichever supported audio format sits next to an RPP.

`lint` checks are covered here too: it reads projects directly (no rendering
or import side effects), so it is exercised through `_lint` the same way
`_find_audio` is, rather than needing a full `build`/`import` fixture.
"""

import json
from argparse import Namespace
from pathlib import Path

from reaper2mt3.cli import _find_audio, _lint


def test_finds_wav(tmp_path):
    rpp = tmp_path / "song.RPP"
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"fake")
    assert _find_audio(rpp) == wav


def test_finds_flac(tmp_path):
    rpp = tmp_path / "song.RPP"
    flac = tmp_path / "song.flac"
    flac.write_bytes(b"fake")
    assert _find_audio(rpp) == flac


def test_prefers_wav_when_both_exist(tmp_path):
    rpp = tmp_path / "song.RPP"
    wav = tmp_path / "song.wav"
    wav.write_bytes(b"fake")
    (tmp_path / "song.flac").write_bytes(b"fake")
    assert _find_audio(rpp) == wav


def test_returns_none_when_neither_exists(tmp_path):
    rpp = tmp_path / "song.RPP"
    assert _find_audio(rpp) is None


def _minimal_project(*tracks: str) -> str:
    return "\n".join(['<REAPER_PROJECT 0.1 "7.74" 0 0', "  TEMPO 120 4 4 0", *tracks, ">"])


def _minimal_track(name: str, mutesolo: str = "0 0 0") -> str:
    return "\n".join([
        "  <TRACK {G}",
        f'    NAME "{name}"',
        f"    MUTESOLO {mutesolo}",
        "  >",
    ])


def _write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "project.RPP"
    path.write_text(text)
    return path


def test_lint_clean_project_reports_no_problems(tmp_path, capsys):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums")))
    code = _lint(Namespace(projects=[rpp], vocabulary=None))
    assert code == 0
    assert "0 problem(s)" in capsys.readouterr().out


def test_lint_flags_non_canonical_track_name(tmp_path, capsys):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("Random Track Name")))
    code = _lint(Namespace(projects=[rpp], vocabulary=None))
    assert code == 1
    assert "non-canonical track name: Random Track Name" in capsys.readouterr().out


def test_lint_flags_muted_track(tmp_path, capsys):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums", mutesolo="1 0 0")))
    code = _lint(Namespace(projects=[rpp], vocabulary=None))
    assert code == 1
    assert "muted track: drums" in capsys.readouterr().out


def test_lint_flags_soloed_track(tmp_path, capsys):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums", mutesolo="0 2 0")))
    code = _lint(Namespace(projects=[rpp], vocabulary=None))
    assert code == 1
    assert "soloed track: drums" in capsys.readouterr().out


def test_lint_flags_out_of_vocabulary_instrument(tmp_path, capsys):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums")))
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(["acoustic-guitar-nylon"]))  # drums not included
    code = _lint(Namespace(projects=[rpp], vocabulary=vocab))
    assert code == 1
    assert "out-of-vocabulary instrument: drums" in capsys.readouterr().out


def test_lint_allows_rhythm_variant_via_base_slug(tmp_path, capsys):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("distortion-guitar:rhythm")))
    vocab = tmp_path / "vocab.json"
    vocab.write_text(json.dumps(["distortion-guitar"]))  # base slug, no ':rhythm' entry
    code = _lint(Namespace(projects=[rpp], vocabulary=vocab))
    assert code == 0
    assert "0 problem(s)" in capsys.readouterr().out


def test_lint_missing_vocabulary_file_errors(tmp_path, capsys):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums")))
    code = _lint(Namespace(projects=[rpp], vocabulary=tmp_path / "missing.json"))
    assert code == 2
    assert "no vocabulary file" in capsys.readouterr().err


def test_lint_collects_projects_from_a_directory(tmp_path, capsys):
    _write(tmp_path, _minimal_project(_minimal_track("drums")))
    code = _lint(Namespace(projects=[tmp_path], vocabulary=None))
    assert code == 0
    assert "1 project(s) checked" in capsys.readouterr().out
