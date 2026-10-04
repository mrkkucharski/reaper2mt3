"""`_find_audio` picks whichever supported audio format sits next to an RPP.

`lint` checks are covered here too: it reads projects directly (no rendering
or import side effects), so it is exercised through `_lint` the same way
`_find_audio` is, rather than needing a full `build`/`import` fixture.
"""

import json
from argparse import Namespace
from pathlib import Path

from reaper2mt3.cli import _find_audio, _lint, _render
from reaper2mt3.render import RenderError


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


def _track_with_midi_item(name: str, notes: str) -> str:
    return "\n".join([
        "  <TRACK {G}",
        f'    NAME "{name}"',
        "    MUTESOLO 0 0 0",
        "    <ITEM",
        "      <SOURCE MIDI",
        "        HASDATA 1 480 QN",
        notes,
        "      >",
        "    >",
        "  >",
    ])


def test_lint_flags_bend_missing_rpn_declaration(tmp_path, capsys):
    """A hand-drawn bend with no RPN 0,0=12 declaration is exactly what
    niepokonani.RPP was missing (PROJECT_LOG.md, 2026-10-04) -- caught here,
    before it reaches mt3/scripts/build_guitar_pilot_tfrecord.py, which
    rejects it outright at TFRecord-build time instead."""
    notes = "        E 0 90 3c 64\n        E 100 e0 00 50\n        E 380 80 3c 00"
    rpp = _write(tmp_path, _minimal_project(_track_with_midi_item("distortion-guitar", notes)))
    code = _lint(Namespace(projects=[rpp], vocabulary=None))
    assert code == 1
    out = capsys.readouterr().out
    assert "1/1 bend event(s) missing an RPN 0,0=12 semitone declaration" in out


def test_lint_allows_bend_with_rpn_declaration(tmp_path, capsys):
    notes = ("        E 0 b0 65 00\n        E 0 b0 64 00\n        E 0 b0 06 0c\n"
             "        E 0 90 3c 64\n        E 100 e0 00 50\n        E 380 80 3c 00")
    rpp = _write(tmp_path, _minimal_project(_track_with_midi_item("distortion-guitar", notes)))
    code = _lint(Namespace(projects=[rpp], vocabulary=None))
    assert code == 0
    assert "0 problem(s)" in capsys.readouterr().out


def _render_args(tmp_path, **overrides):
    defaults = dict(
        projects=[tmp_path / "song.RPP"],
        out_dir=None,
        format="flac",
        sample_rate=44100,
        channels=1,
        reaper_binary=Path("/Applications/REAPER.app/Contents/MacOS/REAPER"),
        timeout=300,
        force=False,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_render_skips_existing_output_without_force(tmp_path, capsys, monkeypatch):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums")))
    rpp.rename(tmp_path / "song.RPP")
    (tmp_path / "song.flac").write_bytes(b"already rendered")

    def fail_if_called(*a, **k):
        raise AssertionError("render_project should not run when the output already exists")

    monkeypatch.setattr("reaper2mt3.cli.render_project_via_reaper", fail_if_called)
    code = _render(_render_args(tmp_path))
    assert code == 0
    out = capsys.readouterr().out
    assert "SKIP" in out
    assert "1 rendered, 1 skipped" not in out  # sanity: this is the 0-rendered path
    assert "0 rendered, 1 skipped, 0 failed" in out


def test_render_force_overwrites_existing_output(tmp_path, capsys, monkeypatch):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums")))
    rpp.rename(tmp_path / "song.RPP")
    output = tmp_path / "song.flac"
    output.write_bytes(b"stale")

    def fake_render(path, *, out_dir, settings, force):
        output.write_bytes(b"fresh")
        return output

    monkeypatch.setattr("reaper2mt3.cli.render_project_via_reaper", fake_render)
    code = _render(_render_args(tmp_path, force=True))
    assert code == 0
    assert output.read_bytes() == b"fresh"
    assert "1 rendered, 0 skipped, 0 failed" in capsys.readouterr().out


def test_render_reports_failure_and_nonzero_exit(tmp_path, capsys, monkeypatch):
    rpp = _write(tmp_path, _minimal_project(_minimal_track("drums")))
    rpp.rename(tmp_path / "song.RPP")

    def fake_render(path, *, out_dir, settings, force):
        raise RenderError("REAPER not found")

    monkeypatch.setattr("reaper2mt3.cli.render_project_via_reaper", fake_render)
    code = _render(_render_args(tmp_path))
    assert code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert "0 rendered, 0 skipped, 1 failed" in out


def test_render_no_projects_found_errors(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    code = _render(_render_args(tmp_path, projects=[empty_dir]))
    assert code == 2
    assert "no REAPER projects found" in capsys.readouterr().err
