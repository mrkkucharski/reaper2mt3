"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .dataset import (
    assign_splits,
    build_example,
    import_example,
    load_splits,
    validate_example,
    write_manifest,
    write_splits,
)
from .render import RenderError, RenderSettings, fluidsynth_version
from .rppread import read_project


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="reaper2mt3", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="render verified REAPER projects into training examples")
    build.add_argument("projects", nargs="+", type=Path, help="RPP files or directories")
    build.add_argument("-o", "--out", type=Path, required=True, help="dataset root")
    build.add_argument("--test-fraction", type=float, default=0.2)
    build.add_argument("--sample-rate", type=int, default=44100)
    build.add_argument("--tail-seconds", type=float, default=2.0)
    build.add_argument("--gain", type=float, default=0.6)
    build.add_argument("--keep-stems", action="store_true", help="retain per-part renders")

    check = sub.add_parser("check", help="re-run the contract checks on an existing dataset")
    check.add_argument("dataset", type=Path, help="dataset root containing manifest.jsonl")

    imp = sub.add_parser(
        "import",
        help="pair already-rendered audio (WAV or FLAC) with labels extracted from their RPP "
             "-- no rendering",
    )
    imp.add_argument("source", type=Path,
                     help="directory containing <name>.RPP + <name>.wav/.flac pairs, "
                          "or a single RPP")
    imp.add_argument("-o", "--out", type=Path, required=True, help="dataset root")
    imp.add_argument("--test-fraction", type=float, default=0.2)
    imp.add_argument("--renderer", default="REAPER (manual render)",
                     help="free-text description recorded in the manifest")
    imp.add_argument("--exceptions", type=Path,
                     help="JSON file of approved out-of-range pitches, keyed by "
                          "source_midi_id -- see DATA_CONTRACT.md's pitch range exception")

    lint = sub.add_parser(
        "lint",
        help="check REAPER projects for non-canonical track names, "
             "out-of-vocabulary instruments, and muted/soloed tracks -- "
             "before rendering or import, not instead of it",
    )
    lint.add_argument("projects", nargs="+", type=Path, help="RPP files or directories")
    lint.add_argument("--vocabulary", type=Path,
                      help="JSON file listing allowed canonical base slugs (without any "
                           "':rhythm' suffix); omit to skip the out-of-vocabulary check")

    args = parser.parse_args(argv)
    if args.command == "check":
        return _check(args)
    if args.command == "import":
        return _import(args)
    if args.command == "lint":
        return _lint(args)
    return _build(args)


def _collect(items: list[Path]) -> list[Path]:
    projects: list[Path] = []
    for item in items:
        if item.is_dir():
            projects += sorted(item.rglob("*.RPP"))
        else:
            projects.append(item)
    return projects


def _build(args: argparse.Namespace) -> int:
    try:
        renderer = fluidsynth_version()
    except RenderError as error:
        print(error, file=sys.stderr)
        return 2

    settings = RenderSettings(
        sample_rate=args.sample_rate, tail_seconds=args.tail_seconds, gain=args.gain
    )
    paths = _collect(args.projects)
    if not paths:
        print("no REAPER projects found", file=sys.stderr)
        return 2

    splits = assign_splits([p.stem for p in paths], args.test_fraction, load_splits(args.out))
    work_dir = Path(tempfile.mkdtemp(prefix="reaper2mt3-"))
    examples, failures = [], 0

    try:
        for index, path in enumerate(paths, start=1):
            project = read_project(path)
            if not project.parts:
                print(f"SKIP    {path.name}: no canonically named parts found")
                failures += 1
                continue

            example_id = f"ex_{index:04d}"
            split = splits[project.name]
            try:
                example = build_example(project, args.out, example_id, split, settings,
                                        work_dir, renderer)
            except RenderError as error:
                print(f"FAIL    {path.name}: {error}")
                failures += 1
                continue

            examples.append(example)
            print(f"{'OK  ' if not example.problems else 'PROB'}    {example_id} [{split}] "
                  f"{path.stem[:40]:<40} {len(project.parts)} parts")
            for problem in example.problems:
                print(f"          {problem}")

        if examples:
            write_manifest(examples, args.out / "manifest.jsonl")
            write_splits(splits, args.out)
        if args.keep_stems:
            shutil.copytree(work_dir, args.out / "stems", dirs_exist_ok=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    clean = sum(1 for e in examples if not e.problems)
    print(f"\n{len(examples)} example(s) written to {args.out} "
          f"({clean} clean, {len(examples) - clean} with problems, {failures} failed)")
    return 1 if failures or clean != len(examples) else 0


AUDIO_EXTENSIONS = (".wav", ".flac")


def _find_audio(rpp: Path) -> Path | None:
    """The audio next to an RPP, whichever supported format it was rendered
    in. Checked in this order, so a project with both (e.g. re-rendered from
    WAV to FLAC without removing the old file) deterministically picks WAV,
    matching the format every pre-FLAC project already uses."""
    for ext in AUDIO_EXTENSIONS:
        candidate = rpp.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def _import(args: argparse.Namespace) -> int:
    exceptions: dict[str, dict] = {}
    if args.exceptions:
        if not args.exceptions.exists():
            print(f"no exceptions file at {args.exceptions}", file=sys.stderr)
            return 2
        exceptions = json.loads(args.exceptions.read_text())

    rpp_files = sorted(args.source.rglob("*.RPP")) if args.source.is_dir() else [args.source]
    pairs: list[tuple[Path, Path]] = []
    for rpp in rpp_files:
        audio = _find_audio(rpp)
        if audio is None:
            print(f"SKIP    {rpp.name}: no matching .wav or .flac next to it")
            continue
        pairs.append((rpp, audio))

    if not pairs:
        print("no RPP/audio pairs found", file=sys.stderr)
        return 2

    splits = assign_splits([rpp.stem for rpp, _ in pairs], args.test_fraction, load_splits(args.out))
    examples, failures = [], 0

    for index, (rpp, audio) in enumerate(pairs, start=1):
        project = read_project(rpp)
        if not project.parts:
            print(f"SKIP    {rpp.name}: no canonically named parts found")
            failures += 1
            continue

        example_id = f"ex_{index:04d}"
        split = splits[project.name]
        example = import_example(project, audio, args.out, example_id, split, args.renderer,
                                 pitch_exceptions=exceptions.get(project.name))
        examples.append(example)
        print(f"{'OK  ' if not example.problems else 'PROB'}    {example_id} [{split}] "
              f"{rpp.stem[:40]:<40} {len(project.parts)} parts")
        for problem in example.problems:
            print(f"          {problem}")
        if project.unparsed_tracks:
            for name in project.unparsed_tracks:
                print(f"          skipped track: {name[:70]}")

    if examples:
        write_manifest(examples, args.out / "manifest.jsonl")
        write_splits(splits, args.out)

    clean = sum(1 for e in examples if not e.problems)
    print(f"\n{len(examples)} example(s) imported to {args.out} "
          f"({clean} clean, {len(examples) - clean} with problems, {failures} failed)")
    return 1 if failures or clean != len(examples) else 0


def _check(args: argparse.Namespace) -> int:
    manifest = args.dataset / "manifest.jsonl"
    if not manifest.exists():
        print(f"no manifest at {manifest}", file=sys.stderr)
        return 2

    total = 0
    problems = 0
    for line in manifest.read_text().splitlines():
        record = json.loads(line)
        total += 1
        found = validate_example(record, args.dataset)
        problems += len(found)
        if found:
            print(f"{record['id']}:")
            for problem in found:
                print(f"  {problem}")

    print(f"{total} example(s) checked, {problems} problem(s)")
    return 1 if problems else 0


def _lint(args: argparse.Namespace) -> int:
    vocabulary: set[str] | None = None
    if args.vocabulary:
        if not args.vocabulary.exists():
            print(f"no vocabulary file at {args.vocabulary}", file=sys.stderr)
            return 2
        vocabulary = set(json.loads(args.vocabulary.read_text()))

    paths = _collect(args.projects)
    if not paths:
        print("no REAPER projects found", file=sys.stderr)
        return 2

    total_problems = 0
    for path in paths:
        project = read_project(path)
        problems: list[str] = []

        for name in project.unparsed_tracks:
            problems.append(f"non-canonical track name: {name[:70]}")

        for part in project.parts:
            base_slug = part.canonical_name.split(":", 1)[0]
            if vocabulary is not None and base_slug not in vocabulary:
                problems.append(f"out-of-vocabulary instrument: {part.canonical_name} "
                               f"({part.track_name[:60]})")
            if part.muted:
                problems.append(f"muted track: {part.canonical_name} ({part.track_name[:60]})")
            if part.soloed:
                problems.append(f"soloed track: {part.canonical_name} ({part.track_name[:60]})")

        print(f"{'OK  ' if not problems else 'PROB'}    {path.name[:70]:<70} "
              f"{len(project.parts)} part(s)")
        for problem in problems:
            print(f"          {problem}")
        total_problems += len(problems)

    print(f"\n{len(paths)} project(s) checked, {total_problems} problem(s)")
    return 1 if total_problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
