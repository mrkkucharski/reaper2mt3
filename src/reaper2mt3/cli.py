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
from .finalization import load_finalization_input, preflight, provenance_record
from .render import RenderError, RenderSettings, fluidsynth_version
from .render_reaper import (
    DEFAULT_REAPER_BINARY,
    RenderReaperSettings,
    output_path_for,
    render_project as render_project_via_reaper,
)
from . import noteqa, slides
from .rppread import (
    AMPLE_GUITAR_MARKER,
    AMPLE_MARKER,
    PITCH_BEND_RANGE_SEMITONES,
    read_project,
)


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
    imp.add_argument("--allow-slide-ramps", action="store_true",
                     help="import projects that still contain Guitar-Pro slide ramps "
                          "instead of failing on them (see `reaper2mt3 fix-slides`)")
    imp.add_argument("--allow-polyphonic-slides", action="store_true",
                     help="import projects containing slide ramps overlapped by other "
                          "notes in the same part; these have no converter yet")
    imp.add_argument("--render-provenance", type=Path, required=True,
                     help="procgen.reaper-render-provenance/v1 sidecar; supplies pinned "
                          "revisions, renderer aliases, and per-part event policy")

    render = sub.add_parser(
        "render",
        help="headlessly render already-tuned REAPER projects to audio via REAPER itself "
             "-- reaches real plugin chains build's FluidSynth path can't",
    )
    render.add_argument("projects", nargs="+", type=Path, help="RPP files or directories")
    render.add_argument("--out-dir", type=Path,
                        help="defaults to each project's own directory")
    render.add_argument("--format", choices=["flac", "wav"], default="flac")
    render.add_argument("--sample-rate", type=int, default=44100)
    render.add_argument("--channels", type=int, default=1)
    render.add_argument("--reaper-binary", type=Path, default=DEFAULT_REAPER_BINARY)
    render.add_argument("--timeout", type=int, default=300,
                        help="seconds to wait for one project before giving up")
    render.add_argument("--force", action="store_true",
                        help="remove an existing render first instead of skipping it")

    lint = sub.add_parser(
        "lint",
        help="check REAPER projects for non-canonical track names, "
             "out-of-vocabulary instruments, and muted/soloed tracks -- "
             "before rendering or import, not instead of it",
    )
    lint.add_argument("projects", nargs="+", type=Path, help="RPP files or directories")
    lint.add_argument("--allow-slide-ramps", action="store_true",
                      help="don't fail on Guitar-Pro slide ramps that still need "
                           "pitch-bend conversion (see `reaper2mt3 fix-slides`)")
    lint.add_argument("--allow-polyphonic-slides", action="store_true",
                      help="don't fail on slide ramps overlapped by other notes in the "
                           "same part; these have no automatic conversion yet")
    lint.add_argument("--no-warnings", action="store_true",
                      help="skip advisory checks (isolated very short notes); "
                           "these never affect the exit code either way")
    lint.add_argument("--vocabulary", type=Path,
                      help="JSON file listing allowed canonical base slugs (without any "
                           "':rhythm' suffix); omit to skip the out-of-vocabulary check")

    fix = sub.add_parser(
        "fix-slides",
        help="rewrite Guitar-Pro slide ramps as one sustained note plus pitch bend",
    )
    fix.add_argument("projects", nargs="+", type=Path, help="RPP files or directories")
    fix.add_argument("--bend-range", type=int, default=slides.DEFAULT_BEND_RANGE,
                     help=f"semitones declared via RPN 0 "
                          f"(default {slides.DEFAULT_BEND_RANGE}); a run needing more "
                          f"than this aborts rather than clipping")
    fix.add_argument("--track", action="append", dest="tracks",
                     help="only convert this track NAME (repeatable; default all)")
    fix.add_argument("--dry-run", action="store_true",
                     help="report what would change without writing")
    fix.add_argument("--report", type=Path, help="write a JSON report here")

    args = parser.parse_args(argv)
    if args.command == "check":
        return _check(args)
    if args.command == "import":
        return _import(args)
    if args.command == "render":
        return _render(args)
    if args.command == "lint":
        return _lint(args)
    if args.command == "fix-slides":
        return _fix_slides(args)
    return _build(args)


def _collect(items: list[Path]) -> list[Path]:
    projects: list[Path] = []
    for item in items:
        if item.is_dir():
            # Skip backup folders: a backup RPP shares its stem with the live
            # project, so scanning both would register two `source_midi_id`s
            # for one song (see slides.BACKUP_DIR_NAMES).
            projects += sorted(p for p in item.rglob("*.RPP")
                               if not slides.is_backup_path(p))
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
    try:
        sidecar = load_finalization_input(args.render_provenance)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2

    # `_collect`'s backup exclusion applies here too: importing a backup
    # alongside its live project would register two `source_midi_id`s for one
    # song and risk a near-duplicate pair landing in different splits.
    rpp_files = _collect([args.source]) if args.source.is_dir() else [args.source]
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
        project = read_project(rpp, strict_corpus_names=False)
        try:
            preflight_report = preflight(project, sidecar)
        except ValueError as error:
            print(f"FAIL    {rpp.name}: {error}")
            failures += 1
            continue
        if not project.parts:
            print(f"SKIP    {rpp.name}: no canonically named parts found")
            failures += 1
            continue

        found = slides.scan_project(project)
        blocking = []
        if not getattr(args, "allow_slide_ramps", False):
            n = sum(len(runs) for runs, _s in found.values())
            if n:
                blocking.append(f"{n} convertible slide gesture(s) "
                                "-- run `reaper2mt3 fix-slides`")
        if not getattr(args, "allow_polyphonic_slides", False):
            n = sum(len(sk) for _r, sk in found.values())
            if n:
                blocking.append(f"{n} polyphonic slide ramp(s) "
                                "-- pass --allow-polyphonic-slides to accept")
        if blocking:
            print(f"FAIL    {rpp.name}: " + "; ".join(blocking))
            failures += 1
            continue

        example_id = f"ex_{index:04d}"
        split = splits[project.name]
        example = import_example(project, audio, args.out, example_id, split, args.renderer,
                                 pitch_exceptions=exceptions.get(project.name))
        provenance_path = Path("provenance") / split / f"{example_id}.json"
        full_provenance_path = args.out / provenance_path
        full_provenance_path.parent.mkdir(parents=True, exist_ok=True)
        full_provenance_path.write_text(
            json.dumps(provenance_record(sidecar, preflight_report, example_id),
                       indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        example.record["provenance_path"] = str(provenance_path)
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


def _render(args: argparse.Namespace) -> int:
    paths = _collect(args.projects)
    if not paths:
        print("no REAPER projects found", file=sys.stderr)
        return 2

    settings = RenderReaperSettings(
        sample_rate=args.sample_rate,
        channels=args.channels,
        audio_format=args.format,
        reaper_binary=args.reaper_binary,
        timeout_seconds=args.timeout,
    )

    rendered = failures = skipped = 0
    for path in paths:
        existing = output_path_for(path, args.out_dir, args.format)
        if existing.exists() and not args.force:
            print(f"SKIP    {path.name[:60]:<60} {existing.name} already exists")
            skipped += 1
            continue
        try:
            output = render_project_via_reaper(
                path, out_dir=args.out_dir, settings=settings, force=args.force)
        except RenderError as error:
            print(f"FAIL    {path.name}: {error}")
            failures += 1
            continue
        size_mb = output.stat().st_size / 1e6
        print(f"OK      {path.name[:60]:<60} -> {output.name} ({size_mb:.1f} MB)")
        rendered += 1

    print(f"\n{rendered} rendered, {skipped} skipped, {failures} failed "
          f"(of {len(paths)} project(s))")
    return 1 if failures else 0


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


MAX_LOCATED_LINES = 40


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
    total_warnings = 0
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

            # Ample Sound plugins don't honour an incoming MIDI RPN Pitch
            # Bend Sensitivity message -- only the instrument's own saved
            # `Bend Range` parameter controls how far its wheel actually
            # bends. The pb12 label vocabulary assumes 12 semitones; any
            # other value silently mismatches label and audio wherever this
            # part actually carries bend events (PROJECT_LOG.md, 2026-09-12).
            if part.bend_count and any(AMPLE_MARKER in p for p in part.instrument_plugins):
                if part.bend_range != 12:
                    seen = "unknown" if part.bend_range is None else str(part.bend_range)
                    problems.append(
                        f"Ample instrument bend range is {seen}, not 12: "
                        f"{part.canonical_name} ({part.track_name[:60]}), "
                        f"{part.bend_count} bend event(s)")

                # Ample Guitar (not Bass) also needs Poly Bender on, or bent
                # notes don't retrigger/overlap correctly -- separate from
                # Bend Range, which only scales how far the wheel travels.
                if any(AMPLE_GUITAR_MARKER in p for p in part.instrument_plugins):
                    if not part.poly_bender:
                        seen = ("unknown" if part.poly_bender is None
                               else str(part.poly_bender))
                        problems.append(
                            f"Ample Guitar Poly Bender is off ({seen}): "
                            f"{part.canonical_name} ({part.track_name[:60]}), "
                            f"{part.bend_count} bend event(s)")

            # Independent of which instrument is loaded (an Ample plugin's
            # own Bend Range parameter above is a different, playback-only
            # setting it never reads this RPN from): every bend event needs
            # an RPN 0,0=12 declaration (CC 101=0, 100=0, 6=12) in effect on
            # its channel, or mt3/scripts/build_guitar_pilot_tfrecord.py
            # rejects it outright when splicing real bend data into a
            # training example (PROJECT_LOG.md, 2026-10-04 -- a hand-drawn
            # bend in REAPER doesn't get this declaration for free; only
            # scripts/generate_bend_gestures.py and `fix-slides` add it).
            if part.bends_missing_rpn:
                problems.append(
                    f"{part.bends_missing_rpn}/{part.bend_count} bend event(s) missing "
                    f"an RPN 0,0={PITCH_BEND_RANGE_SEMITONES} semitone declaration: "
                    f"{part.canonical_name} ({part.track_name[:60]})")

        # Guitar-Pro slide ramps. A failure, not a warning: these silently
        # teach the model to over-emit notes, and a warning would be
        # overlooked. `reaper2mt3 fix-slides` converts them.
        #
        # Anything with a position goes into one list ordered by bar, so the
        # output reads as a work list to walk through once rather than a
        # per-track grouping that jumps back and forth along the timeline.
        sig = project.time_signature or (4, 4)
        located: list[tuple[int, str, str]] = []

        for track_name, (runs, skipped) in slides.scan_project(project).items():
            if runs and not getattr(args, "allow_slide_ramps", False):
                # Not a per-place list: the fix is to run `fix-slides`, once.
                notes = sum(r.note_count for r in runs)
                written = sum(r.voices for r in runs)
                problems.append(
                    f"slide gestures needing pitch-bend conversion: {track_name[:44]} "
                    f"-- {len(runs)} gesture(s), {notes} notes -> {written}, "
                    f"max bend {max(r.max_bend for r in runs)} st "
                    f"(run `reaper2mt3 fix-slides`)")
            if skipped and not getattr(args, "allow_polyphonic_slides", False):
                for run in skipped:
                    located.append((
                        run.start_tick, "FIX ",
                        f"{run.where(project.ppq, sig)}  slide ramp, "
                        f"{run.note_count} notes ({run.reason})  {track_name[:38]}"))
                total_problems += len(skipped)

        # Advisory: surfaced for review, never a reason to fail. An isolated
        # short note may be a dead/muted note the arranger meant.
        if not getattr(args, "no_warnings", False):
            for track_name, hits in noteqa.scan_project(project).items():
                for hit in hits:
                    located.append((
                        hit.start_tick, "warn",
                        f"{hit.where(project.ppq, sig)}  short note "
                        f"{hit.duration_ms:.0f}ms {list(hit.pitches)}  "
                        f"{track_name[:38]}"))
                total_warnings += len(hits)

            # Slide-shaped, but below the converter's floors. `fix-slides`
            # will never touch these, so they need a human decision rather
            # than a failure that no command can clear.
            for track_name, figures in noteqa.scan_small_slide_figures(project).items():
                for figure in figures:
                    located.append((
                        figure.start_tick, "warn",
                        f"{figure.where(project.ppq, sig)}  small slide figure, "
                        f"{figure.steps} steps {figure.voices}v span "
                        f"{figure.span} @{figure.step_ms:.0f}ms from "
                        f"{list(figure.pitches)}  {track_name[:38]}"))
                total_warnings += len(figures)

        located.sort(key=lambda item: item[0])
        has_fix = any(severity == "FIX " for _t, severity, _m in located)

        status = "PROB" if (problems or has_fix) else ("WARN" if located else "OK  ")
        print(f"{status}    {path.name[:70]:<70} {len(project.parts)} part(s)")
        for problem in problems:
            print(f"          {problem}")
        for _tick, severity, message in located[:MAX_LOCATED_LINES]:
            print(f"    {severity}  {message}")
        if len(located) > MAX_LOCATED_LINES:
            print(f"          ... +{len(located) - MAX_LOCATED_LINES} more "
                  f"(use --no-warnings to show only what fails)")
        total_problems += len(problems)

    summary = f"\n{len(paths)} project(s) checked, {total_problems} problem(s)"
    if total_warnings:
        summary += f", {total_warnings} warning(s) (advisory, not failing)"
    print(summary)
    return 1 if total_problems else 0


def _fix_slides(args: argparse.Namespace) -> int:
    paths = _collect(args.projects)
    if not paths:
        print("no REAPER projects found", file=sys.stderr)
        return 2

    only = set(args.tracks) if args.tracks else None
    report, converted_total, skipped_total, failures = [], 0, 0, 0

    for path in paths:
        result = slides.convert_project(
            path, bend_range=args.bend_range, dry_run=args.dry_run, only_tracks=only)
        if result.error:
            print(f"FAIL    {path.name[:70]}: {result.error}")
            failures += 1
            continue
        converted_total += result.converted
        skipped_total += result.skipped
        if not result.tracks or (not result.converted and not result.skipped):
            print(f"OK      {path.name[:70]}    no slide ramps")
            continue

        verb = "would convert" if args.dry_run else "converted"
        print(f"{'DRY ' if args.dry_run else 'FIX '}    {path.name[:70]}")
        for track in result.tracks:
            if track.converted:
                notes = sum(r.note_count for r in track.converted)
                written = sum(r.voices for r in track.converted)
                print(f"          {track.track_name[:48]}: {verb} "
                      f"{len(track.converted)} gesture(s), {notes} notes -> "
                      f"{written}  (max bend "
                      f"{max(r.max_bend for r in track.converted)} st)")
                for run in track.converted:
                    print(f"            {run.at_seconds:9.3f}s {run.shape:8s} "
                          f"{run.kind:9s} {run.voices}v "
                          f"{list(run.pitches)} bend {run.bend_from:+d}->{run.bend_to:+d}")
            if track.skipped:
                print(f"          {track.track_name[:48]}: skipped "
                      f"{len(track.skipped)} polyphonic run(s)")
        if result.backup:
            print(f"          backup: {result.backup}")
        report.append({
            "project": str(path),
            "backup": str(result.backup) if result.backup else None,
            "tracks": [{
                "track": t.track_name,
                "notes_before": t.notes_before, "notes_after": t.notes_after,
                "events_before": t.events_before, "events_after": t.events_after,
                "converted": [{
                    "at_seconds": round(r.at_seconds, 3), "kind": r.kind,
                    "shape": r.shape, "voices": r.voices,
                    "notes_dropped": r.note_count, "pitches": list(r.pitches),
                    "bend_from": r.bend_from, "bend_to": r.bend_to,
                    "max_bend": r.max_bend,
                } for r in t.converted],
                "skipped": [{
                    "at_seconds": round(s.at_seconds, 3), "reason": s.reason,
                    "notes": s.note_count, "overlapping": s.overlapping,
                } for s in t.skipped],
            } for t in result.tracks],
        })

    if args.report:
        args.report.write_text(json.dumps(report, indent=2))
        print(f"\nreport written to {args.report}")

    print(f"\n{len(paths)} project(s), {converted_total} run(s) "
          f"{'to convert' if args.dry_run else 'converted'}, "
          f"{skipped_total} polyphonic run(s) skipped, {failures} failure(s)")
    if skipped_total:
        print("polyphonic runs are left alone: pitch bend is per-channel, so "
              "bending under a chord would detune the chord")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
