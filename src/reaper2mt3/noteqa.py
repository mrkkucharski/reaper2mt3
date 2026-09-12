"""Note-level quality checks that are not about slides.

`slides.py` rewrites what it can prove is an articulation artifact. What is
left over still deserves a human eye, and this holds the checks that ask for
one rather than acting on their own.

Everything here is advisory: `reaper2mt3 lint` reports these as WARN and they
do not fail the exit code. A slide ramp is unambiguous and blocks import; an
isolated short note might be a dead/muted note the arranger meant, so it is
surfaced for review instead of being changed.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Callable, Sequence

from . import slides
from .slides import NoteLike, _clusters, bar_beat

# A note this short cannot be a played note once it is through an amp chain:
# it reads as a click. Guitar Pro writes dead/muted "x" notes this way.
SHORT_NOTE_SECONDS = 0.030
# ...and surrounded by this much space, it stands alone rather than being one
# step of a fast figure, which is what makes it worth a look.
ISOLATION_SECONDS = 0.080


@dataclass(frozen=True)
class IsolatedShortNote:
    """A very short note (or simultaneous pair) standing on its own."""

    pitches: tuple[int, ...]
    start_tick: int
    at_seconds: float
    duration_ms: float
    gap_before_ms: float
    gap_after_ms: float

    def where(self, ppq: int, time_signature: tuple[int, int] = (4, 4)) -> str:
        minutes, rest = divmod(self.at_seconds, 60)
        return (f"bar {bar_beat(self.start_tick, ppq, time_signature)} "
                f"@ {int(minutes)}:{rest:06.3f}")


def find_isolated_short_notes(
    notes: Sequence[NoteLike],
    seconds_at: Callable[[int], float],
) -> list[IsolatedShortNote]:
    """Very short notes with clear space around them.

    Shortness alone is not enough -- a fast figure is made of short notes and
    is perfectly real. It is shortness *plus* isolation that marks something
    the arranger probably did not intend as a pitched note.
    """
    if len(notes) < 2:
        return []

    groups = _clusters(notes, seconds_at)
    found: list[IsolatedShortNote] = []
    for index, group in enumerate(groups):
        duration = statistics.median(
            seconds_at(notes[i].end) - seconds_at(notes[i].start) for i in group)
        if duration >= SHORT_NOTE_SECONDS:
            continue

        start = min(seconds_at(notes[i].start) for i in group)
        end = max(seconds_at(notes[i].end) for i in group)
        # Only gaps that exist count. A missing neighbour is absence of
        # evidence, not space: treating it as infinite would flag the first and
        # last step of every ramp, which is exactly what this must not do.
        gaps = []
        if index:
            gaps.append(start - max(seconds_at(notes[i].end) for i in groups[index - 1]))
        if index + 1 < len(groups):
            gaps.append(min(seconds_at(notes[i].start) for i in groups[index + 1]) - end)
        if not gaps or max(gaps) < ISOLATION_SECONDS:
            continue
        before = gaps[0] if index else float("inf")
        after = gaps[-1] if index + 1 < len(groups) else float("inf")

        found.append(IsolatedShortNote(
            pitches=tuple(sorted(notes[i].pitch for i in group)),
            start_tick=min(notes[i].start for i in group),
            at_seconds=start,
            duration_ms=round(duration * 1000, 1),
            gap_before_ms=round(min(before, 9999) * 1000, 1),
            gap_after_ms=round(min(after, 9999) * 1000, 1),
        ))
    return found


# A slide gesture smaller than the converter's floors: too few steps, or too
# small an interval, to rewrite safely. Reported so a human can judge it.
SMALL_FIGURE_MIN_STEPS = 2
# Steps up to a 32nd note count here. The converter needs corroboration above
# RAMP_STEP_FILLER_QN; this only *reports*, so it uses the wider band and picks
# up chord falls like Hey You's organ at bar 28.4 (133.9 ms = 0.125 QN).
SMALL_FIGURE_MAX_QN = slides.RAMP_STEP_MAX_QN
# What counts as a sustained note adjoining the figure, at either end.
ADJOINING_MIN_QN = 0.10


@dataclass(frozen=True)
class SmallSlideFigure:
    """A slide-shaped run the converter deliberately will not touch."""

    pitches: tuple[int, ...]      # the chord the figure starts from
    start_tick: int
    at_seconds: float
    steps: int                    # chords in the run, not notes
    voices: int
    span: int                     # semitones travelled, per voice
    step_ms: float

    def where(self, ppq: int, time_signature: tuple[int, int] = (4, 4)) -> str:
        minutes, rest = divmod(self.at_seconds, 60)
        return (f"bar {bar_beat(self.start_tick, ppq, time_signature)} "
                f"@ {int(minutes)}:{rest:06.3f}")


def find_small_slide_figures(
    notes: Sequence[NoteLike],
    seconds_at: Callable[[int], float],
    ppq: int,
    covered: set[int],
) -> list[SmallSlideFigure]:
    """Monotonic short runs adjoining a sustained note, below the fix floors.

    `find_glides` needs RUN_MIN_STEPS steps spanning SLIDE_MIN_SEMITONES. A
    chord falling two semitones over two steps is the same articulation at a
    size the converter will not rewrite -- and being neither converted nor
    skipped, it was invisible. `covered` excludes anything already accounted
    for, so nothing is reported twice.

    Chord-wise like the real detector, not note-wise: voices of a parallel
    slide interleave in pitch order (56, 59, 64, 55, 58, 63) and never look
    stepwise, which is how Hey You's organ fall at bar 28.4 stayed hidden.
    Spacing is judged in musical time for the same reason the converter's is --
    at 56 BPM a 32nd-note step is 134 ms and overruns RUN_GAP_SECONDS.
    """
    groups = _clusters(notes, seconds_at)
    short = {i for i, n in enumerate(notes)
             if (n.end - n.start) / ppq < SMALL_FIGURE_MAX_QN}

    runs, run, direction = [], [], 0
    for group in groups:
        # A sustained chord ends the figure rather than joining it -- it is the
        # note being slid into. Chaining it in and rejecting the run for
        # containing a long note would lose every slide-in.
        if not set(group) <= short:
            runs.append(run)
            run, direction = [], 0
            continue
        step = None if not run else slides._step_delta(notes, run[-1], group)
        gap = None if not run else (
            notes[group[0]].start - notes[run[-1][0]].start) / ppq
        if run and not (step is not None and gap is not None
                        and 0 < gap < SMALL_FIGURE_MAX_QN
                        and (not direction or (1 if step > 0 else -1) == direction)):
            runs.append(run)
            run, direction = [], 0
        if step is not None and run:
            direction = 1 if step > 0 else -1
        run.append(group)
    runs.append(run)

    found: list[SmallSlideFigure] = []
    for chain in runs:
        if not SMALL_FIGURE_MIN_STEPS <= len(chain) < slides.RUN_MIN_STEPS:
            continue
        run = [i for group in chain for i in group]
        if set(run) & covered:
            continue
        head, tail = notes[run[0]], notes[run[-1]]
        # A sustained note at either end is what makes it a slide rather than
        # two stray notes: the pitch it leaves from, or the one it arrives at.
        adjoins = any(
            (n.end - n.start) / ppq >= ADJOINING_MIN_QN
            and (abs(n.pitch - head.pitch) in (1, 2)
                 and abs(seconds_at(n.end) - seconds_at(head.start))
                 < slides.ANCHOR_JOIN_SECONDS
                 or abs(n.pitch - tail.pitch) in (1, 2)
                 and abs(seconds_at(n.start) - seconds_at(tail.end))
                 < slides.ANCHOR_JOIN_SECONDS)
            for n in notes)
        if not adjoins:
            continue
        durations = [seconds_at(notes[i].end) - seconds_at(notes[i].start) for i in run]
        first = sorted(notes[i].pitch for i in chain[0])
        last = sorted(notes[i].pitch for i in chain[-1])
        found.append(SmallSlideFigure(
            pitches=tuple(first),
            start_tick=min(notes[i].start for i in chain[0]),
            at_seconds=seconds_at(min(notes[i].start for i in chain[0])),
            steps=len(chain),
            voices=len(chain[0]),
            span=abs(last[0] - first[0]),
            step_ms=round(statistics.median(durations) * 1000, 1),
        ))
    return found


def scan_project(project, only_tracks: set[str] | None = None
                 ) -> dict[str, list[IsolatedShortNote]]:
    """Advisory checks over a parsed project, keyed by track name."""
    found = {}
    for part in project.parts:
        if part.is_drum:
            continue          # drums are all short by nature
        if only_tracks and part.track_name not in only_tracks:
            continue
        ordered = sorted(part.notes, key=lambda n: (n.start, n.pitch))
        hits = find_isolated_short_notes(ordered, project.seconds_at)
        if hits:
            found[part.track_name] = hits
    return found


def scan_small_slide_figures(project, only_tracks: set[str] | None = None
                             ) -> dict[str, list[SmallSlideFigure]]:
    """Sub-threshold slide figures per track, excluding what slides.py reports."""
    found = {}
    for part in project.parts:
        if part.is_drum:
            continue
        if only_tracks and part.track_name not in only_tracks:
            continue
        ordered = sorted(part.notes, key=lambda n: (n.start, n.pitch))
        glides, skipped = slides.find_glides(ordered, project.seconds_at, project.ppq)
        covered = ({i for g in glides for i in g.drop}
                   | {i for s in skipped for i in s.indices})
        hits = find_small_slide_figures(
            ordered, project.seconds_at, project.ppq, covered)
        if hits:
            found[part.track_name] = hits
    return found
