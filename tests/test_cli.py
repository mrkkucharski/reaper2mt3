"""`_find_audio` picks whichever supported audio format sits next to an RPP."""

from pathlib import Path

from reaper2mt3.cli import _find_audio


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
