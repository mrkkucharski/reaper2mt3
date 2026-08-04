"""Command line entry point."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

from .dataset import assign_splits, build_example, validate_example, write_manifest
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

    args = parser.parse_args(argv)
    return _check(args) if args.command == "check" else _build(args)


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

    splits = assign_splits([p.stem for p in paths], args.test_fraction)
    work_dir = Path(tempfile.mkdtemp(prefix="reaper2mt3-"))
    examples, failures = [], 0

    try:
        for index, path in enumerate(paths, start=1):
            project = read_project(path)
            if not project.parts:
                print(f"SKIP    {path.name}: no SFLT parts found")
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
        if args.keep_stems:
            shutil.copytree(work_dir, args.out / "stems", dirs_exist_ok=True)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    clean = sum(1 for e in examples if not e.problems)
    print(f"\n{len(examples)} example(s) written to {args.out} "
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


if __name__ == "__main__":
    raise SystemExit(main())
