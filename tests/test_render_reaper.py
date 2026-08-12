"""`render_project` drives REAPER via an injected runner, so these tests never
invoke REAPER itself -- they verify the generated command/script and the
file-handling logic (existing-output handling, output validation) around it.
"""

import subprocess

import numpy as np
import pytest
import soundfile as sf

from reaper2mt3.render import RenderError
from reaper2mt3.render_reaper import (
    RenderReaperSettings,
    output_path_for,
    render_project,
)


def _write_tone(path, *, sample_rate=44100, channels=1, seconds=1.0, silent=False):
    frames = int(sample_rate * seconds)
    if silent:
        samples = np.zeros((frames, channels), dtype=np.int16)
    else:
        tone = (np.sin(np.linspace(0, 440 * seconds, frames)) * 10000).astype(np.int16)
        samples = np.tile(tone.reshape(-1, 1), (1, channels))
    if channels == 1:
        samples = samples.reshape(-1)
    sf.write(path, samples, sample_rate, subtype="PCM_16")


def _fake_reaper_binary(tmp_path):
    binary = tmp_path / "REAPER"
    binary.write_text("#!/bin/sh\n")
    binary.chmod(0o755)
    return binary


def _rpp(tmp_path, name="song.RPP"):
    path = tmp_path / name
    path.write_text('<REAPER_PROJECT 0.1 "7.74" 0 0\n>\n')
    return path


def test_output_path_defaults_next_to_project(tmp_path):
    rpp = _rpp(tmp_path)
    assert output_path_for(rpp, None, "flac") == tmp_path / "song.flac"
    assert output_path_for(rpp, None, "wav") == tmp_path / "song.wav"


def test_output_path_honours_out_dir(tmp_path):
    rpp = _rpp(tmp_path)
    out_dir = tmp_path / "elsewhere"
    assert output_path_for(rpp, out_dir, "flac") == out_dir / "song.flac"


def test_rejects_non_rpp_file(tmp_path):
    not_rpp = tmp_path / "song.txt"
    not_rpp.write_text("nope")
    with pytest.raises(RenderError, match="not a .RPP file"):
        render_project(not_rpp)


def test_rejects_missing_reaper_binary(tmp_path):
    rpp = _rpp(tmp_path)
    settings = RenderReaperSettings(reaper_binary=tmp_path / "no-such-REAPER")
    with pytest.raises(RenderError, match="REAPER not found"):
        render_project(rpp, settings=settings)


def test_refuses_to_overwrite_without_force(tmp_path):
    rpp = _rpp(tmp_path)
    _write_tone(tmp_path / "song.flac")
    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    with pytest.raises(RenderError, match="already exists"):
        render_project(rpp, settings=settings)


def test_force_removes_existing_output_before_rendering(tmp_path):
    rpp = _rpp(tmp_path)
    existing = tmp_path / "song.flac"
    _write_tone(existing, seconds=0.1)
    stale_bytes = existing.read_bytes()

    def runner(command, timeout):
        _write_tone(tmp_path / "song.flac")

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    output = render_project(rpp, settings=settings, force=True, runner=runner)
    assert output.read_bytes() != stale_bytes


def test_generated_command_shape(tmp_path):
    rpp = _rpp(tmp_path)
    seen = {}

    def runner(command, timeout):
        seen["command"] = command
        seen["timeout"] = timeout
        script_path = command[4]
        seen["script"] = open(script_path).read()
        _write_tone(tmp_path / "song.flac")

    binary = _fake_reaper_binary(tmp_path)
    settings = RenderReaperSettings(reaper_binary=binary, timeout_seconds=42)
    render_project(rpp, settings=settings, runner=runner)

    command = seen["command"]
    assert command[0] == str(binary)
    assert command[1] == "-newinst"
    assert command[2] == "-nosplash"
    assert command[3] == str(rpp)
    assert command[5] == "-closeall:nosave:exit"
    assert seen["timeout"] == 42

    script = seen["script"]
    assert "RENDER_CHANNELS', 1" in script
    assert "RENDER_SRATE', 44100" in script
    assert "Main_OnCommand(42230, 0)" in script  # render using most recent settings
    assert "Main_OnCommand(40004, 0)" in script  # close project, so nothing is left dirty
    assert "noprompt:" in script  # swaps to a blank project rather than prompting to save


def test_non_ascii_project_name_is_valid_lua_not_json_unicode_escape(tmp_path):
    """A real corpus file ("...Pamiętasz...") hit exactly this: json.dumps's
    default \\uXXXX escape is valid JSON but not valid Lua, so REAPER's
    interpreter rejected the generated script outright with a syntax error
    and never rendered anything."""
    rpp = _rpp(tmp_path, name="Budka Suflera-Jolka Jolka Pamiętasz.RPP")
    seen = {}

    def runner(command, timeout):
        seen["script"] = open(command[4], encoding="utf-8").read()
        _write_tone(tmp_path / "Budka Suflera-Jolka Jolka Pamiętasz.flac")

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    render_project(rpp, settings=settings, runner=runner)

    script = seen["script"]
    assert "Pamiętasz" in script
    assert "\\u" not in script


def test_timeout_raises_render_error_with_actionable_message(tmp_path):
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        raise subprocess.TimeoutExpired(cmd=command, timeout=timeout)

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    with pytest.raises(RenderError, match="currently open in another REAPER window"):
        render_project(rpp, settings=settings, runner=runner)


def test_nonzero_exit_raises_render_error(tmp_path):
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        raise subprocess.CalledProcessError(returncode=1, cmd=command)

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    with pytest.raises(RenderError, match="REAPER exited 1"):
        render_project(rpp, settings=settings, runner=runner)


def test_missing_output_after_successful_run_raises(tmp_path):
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        pass  # REAPER "succeeded" but never wrote the file

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    with pytest.raises(RenderError, match="did not produce"):
        render_project(rpp, settings=settings, runner=runner)


def test_wrong_channel_count_is_rejected(tmp_path):
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        _write_tone(tmp_path / "song.flac", channels=2)

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path), channels=1)
    with pytest.raises(RenderError, match="rendered 2 channel"):
        render_project(rpp, settings=settings, runner=runner)


def test_wrong_sample_rate_is_rejected(tmp_path):
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        _write_tone(tmp_path / "song.flac", sample_rate=48000)

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path), sample_rate=44100)
    with pytest.raises(RenderError, match="48000 Hz"):
        render_project(rpp, settings=settings, runner=runner)


def test_silent_output_is_rejected(tmp_path):
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        _write_tone(tmp_path / "song.flac", silent=True)

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    with pytest.raises(RenderError, match="is silent"):
        render_project(rpp, settings=settings, runner=runner)


def test_audio_with_a_long_silent_intro_is_not_rejected(tmp_path):
    """A real corpus song ("Zombie") has 8.7s of legitimate quiet intro
    before its first note -- past any short preview window, which once
    flagged it as a failed render. The whole file must be checked, not
    just its opening seconds."""
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        path = tmp_path / "song.flac"
        _write_tone(path, seconds=1.0, silent=True)
        intro = sf.read(path, dtype="int16")[0]
        _write_tone(path, seconds=1.0)
        body = sf.read(path, dtype="int16")[0]
        sf.write(path, np.concatenate([intro] * 10 + [body]), 44100, subtype="PCM_16")

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    output = render_project(rpp, settings=settings, runner=runner)
    assert output.exists()


def test_successful_render_returns_output_path(tmp_path):
    rpp = _rpp(tmp_path)

    def runner(command, timeout):
        _write_tone(tmp_path / "song.flac")

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path))
    output = render_project(rpp, settings=settings, runner=runner)
    assert output == tmp_path / "song.flac"
    assert output.exists()


def test_wav_format_selects_wav_extension_and_code(tmp_path):
    rpp = _rpp(tmp_path)
    seen = {}

    def runner(command, timeout):
        script_path = command[4]
        seen["script"] = open(script_path).read()
        _write_tone(tmp_path / "song.wav")

    settings = RenderReaperSettings(reaper_binary=_fake_reaper_binary(tmp_path), audio_format="wav")
    output = render_project(rpp, settings=settings, runner=runner)
    assert output == tmp_path / "song.wav"
    assert "ZXZhdxAAAQ==" in seen["script"]
