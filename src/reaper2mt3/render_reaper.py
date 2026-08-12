"""Render an already-tuned REAPER project's own mix to audio using REAPER
itself, headlessly -- not FluidSynth.

`render.py`'s FluidSynth path (`build`) exists because Pedalboard cannot host
SFLT, but it only ever reaches SFLT-backed parts: a chain-driven part (real
Kontakt, Guitar Rig, Ample Sound instruments spliced in by `midi2reaper`) has
no FluidSynth equivalent at all. The projects this repo actually cares about
are exactly the chain-driven, hand-tuned ones, so every session that needed a
fresh render after editing a project's instruments has re-derived the same
answer by hand: drive REAPER's own command line with a short generated
ReaScript, using the documented "render project, using the most recent render
settings" action (42230) instead of clicking through the Render dialog.

`procgen/src/procgen/renderers/reaper_worker.py` already does this for
procgen's own synthetic corpus, but it is built around procgen's
build-result/provenance contract (a `midi2reaper.build-result/v1` JSON, a
pinned library manifest, mix *and* per-part stems) that a hand-tuned real
song's project simply doesn't carry. Rather than import that module and force
its contract onto something it was never built for, this reimplements the
same core technique -- ReaScript render-settings + action 42230 + a
throwaway discard project so REAPER never prompts to save -- scoped to what
this repo actually needs: one full mix per project, in this corpus's own
established format.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import soundfile as sf

from .render import RenderError

DEFAULT_REAPER_BINARY = Path("/Applications/REAPER.app/Contents/MacOS/REAPER")

# REAPER's RENDER_FORMAT project-info value is an opaque base64 blob: a
# reversed four-byte format code (FLAC's 'calf', WAV's 'evaw') followed by
# format-specific parameters (bit depth, compression level). Not something to
# construct from scratch -- both were extracted verbatim from a real
# project's own saved <RENDER_CFG> block (recorded once by REAPER's own
# Render dialog, never guessed), so a render made this way matches this
# corpus's established settings exactly rather than approximating them.
FLAC_FORMAT_B64 = "Y2FsZhAAAAAFAAAA"
WAV_FORMAT_B64 = "ZXZhdxAAAQ=="

_FORMAT_CODES = {"flac": FLAC_FORMAT_B64, "wav": WAV_FORMAT_B64}
_FORMAT_EXTENSIONS = {"flac": ".flac", "wav": ".wav"}

# Runs `command`, raising on a non-zero exit or after `timeout` seconds.
# Injectable so tests can verify the generated command/script without
# actually invoking REAPER.
Runner = Callable[[Sequence[str], int], None]


@dataclass(frozen=True)
class RenderReaperSettings:
    sample_rate: int = 44100
    channels: int = 1
    audio_format: str = "flac"  # "flac" or "wav"
    reaper_binary: Path = DEFAULT_REAPER_BINARY
    timeout_seconds: int = 300


def output_path_for(project_path: Path, out_dir: Path | None, audio_format: str) -> Path:
    destination_dir = (out_dir or project_path.parent).resolve()
    return destination_dir / f"{project_path.stem}{_FORMAT_EXTENSIONS[audio_format]}"


def render_project(
    project_path: Path,
    *,
    out_dir: Path | None = None,
    settings: RenderReaperSettings | None = None,
    force: bool = False,
    runner: Runner | None = None,
) -> Path:
    """Render `project_path`'s own mix, via REAPER, to `out_dir` (default:
    next to the project, matching every other render already in this corpus).

    Raises `RenderError` if the output already exists and `force` is not
    set, if REAPER cannot be found at `settings.reaper_binary`, if the
    render subprocess fails or times out, or if the resulting file is
    missing or fails the same channel/sample-rate/silence checks
    `dataset._check_audio` runs during `check` -- so a broken render is
    caught here, at render time, not silently imported later.
    """
    settings = settings or RenderReaperSettings()
    project_path = project_path.resolve()
    if not project_path.is_file() or project_path.suffix.upper() != ".RPP":
        raise RenderError(f"not a .RPP file: {project_path}")
    if not settings.reaper_binary.exists():
        raise RenderError(f"REAPER not found at {settings.reaper_binary}")

    output_path = output_path_for(project_path, out_dir, settings.audio_format)
    if output_path.exists():
        if not force:
            raise RenderError(f"{output_path} already exists (pass force=True to overwrite)")
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run = runner or _subprocess_runner
    with tempfile.TemporaryDirectory(prefix="reaper2mt3-render-") as tmp:
        tmp_path = Path(tmp)
        # Rendering marks the project dirty (render settings are project
        # state), so before quitting the dedicated `-newinst` worker, REAPER
        # is pointed at a disposable blank project via the documented
        # `noprompt:` path -- the same technique that lets a headless render
        # exit without ever offering to save the RPP it just rendered.
        discard_project = tmp_path / "discard-after-render.RPP"
        discard_project.write_text('<REAPER_PROJECT 0.1 "7.0/x64" 0\n>\n', encoding="utf-8")

        script_path = tmp_path / "render.lua"
        script_path.write_text(
            _lua_render_script(output_path.parent, project_path.stem, settings, discard_project),
            encoding="utf-8",
        )

        command = [
            str(settings.reaper_binary), "-newinst", "-nosplash",
            str(project_path), str(script_path), "-closeall:nosave:exit",
        ]
        try:
            run(command, settings.timeout_seconds)
        except subprocess.TimeoutExpired as error:
            raise RenderError(
                f"{project_path.name}: REAPER did not finish within "
                f"{settings.timeout_seconds}s. If this project is currently open in "
                "another REAPER window, close it first -- a project the GUI already "
                "has open can block a headless render indefinitely without ever "
                "raising an error."
            ) from error
        except subprocess.CalledProcessError as error:
            raise RenderError(f"{project_path.name}: REAPER exited {error.returncode}") from error

    if not output_path.exists():
        raise RenderError(f"{project_path.name}: render did not produce {output_path}")
    _validate_output(output_path, settings)
    return output_path


def _validate_output(path: Path, settings: RenderReaperSettings) -> None:
    info = sf.info(path)
    if info.channels != settings.channels:
        raise RenderError(
            f"{path.name}: rendered {info.channels} channel(s), expected {settings.channels}")
    if info.samplerate != settings.sample_rate:
        raise RenderError(
            f"{path.name}: rendered at {info.samplerate} Hz, expected {settings.sample_rate}")
    if not info.frames:
        raise RenderError(f"{path.name}: rendered file has no audio frames")
    # The whole file, matching dataset._check_audio's own silence check --
    # a several-second preview isn't enough: a real corpus song ("Zombie")
    # has 8.7s of legitimate quiet intro before its first note, well past a
    # 5s preview window, and was flagged as a failed render for it.
    samples, _ = sf.read(path, dtype="int16")
    if not samples.any():
        raise RenderError(f"{path.name}: rendered audio is silent -- render may have failed")


def _lua_render_script(
    out_dir: Path, pattern: str, settings: RenderReaperSettings, discard_project: Path,
) -> str:
    # JSON string literals are valid Lua string literals and protect paths
    # with spaces/quotes without asking a shell to interpret them -- but only
    # with ensure_ascii=False. json.dumps's default escapes any non-ASCII
    # character (a song title with a diacritic, e.g. "Pamiętasz") as \uXXXX,
    # which is valid JSON but not valid Lua: Lua has no bare \u escape at all
    # (5.3+'s is spelled \u{XXXX}, with braces), so REAPER's Lua interpreter
    # fails the whole script with "missing '{'" and never renders -- caught
    # via a real corpus file that hit exactly this.
    format_code = _FORMAT_CODES[settings.audio_format]
    return "\n".join([
        "local project = 0",
        "reaper.GetSetProjectInfo(project, 'RENDER_SETTINGS', 0, true)",   # full mix, no stems
        "reaper.GetSetProjectInfo(project, 'RENDER_BOUNDSFLAG', 1, true)", # entire project
        f"reaper.GetSetProjectInfo(project, 'RENDER_CHANNELS', {settings.channels}, true)",
        f"reaper.GetSetProjectInfo(project, 'RENDER_SRATE', {settings.sample_rate}, true)",
        "reaper.GetSetProjectInfo_String(project, 'RENDER_FORMAT', "
        f"'{format_code}', true)",
        "reaper.GetSetProjectInfo_String(project, 'RENDER_FILE', "
        + _lua_string(str(out_dir)) + ", true)",
        "reaper.GetSetProjectInfo_String(project, 'RENDER_PATTERN', "
        + _lua_string(pattern) + ", true)",
        "reaper.Main_OnCommand(42230, 0)",  # render project, using the most recent render settings
        "reaper.Main_openProject("
        + _lua_string("noprompt:" + str(discard_project)) + ")",
        "reaper.Main_OnCommand(40004, 0)",  # close current project
        "",
    ])


def _lua_string(value: str) -> str:
    """A double-quoted Lua string literal for `value`.

    JSON and Lua string-literal syntax agree on every escape this needs
    (`\"`, `\\`, control characters) except JSON's `\\uXXXX`, so
    `ensure_ascii=False` keeps non-ASCII characters as literal UTF-8 bytes
    instead -- which Lua accepts directly, and which the .lua script file is
    written out as (`encoding="utf-8"`, below).
    """
    return json.dumps(value, ensure_ascii=False)


def _subprocess_runner(command: Sequence[str], timeout: int) -> None:
    subprocess.run(command, check=True, timeout=timeout)


__all__ = [
    "DEFAULT_REAPER_BINARY",
    "FLAC_FORMAT_B64",
    "WAV_FORMAT_B64",
    "RenderReaperSettings",
    "Runner",
    "output_path_for",
    "render_project",
]
