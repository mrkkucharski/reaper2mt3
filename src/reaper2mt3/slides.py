"""Detection and repair of Guitar-Pro slide gestures in REAPER projects.

Guitar Pro exports a slide as a run of chromatic micro-notes. In the corpus
that becomes 10-30 note events where a transcriber would write one, which is
the supervision that taught the model to over-emit -- see
`reports/corpus_audit_2026-08-27.md`.

One detector covers every shape the export produces, because they are the same
gesture at different voice counts:

  bare       a ramp on its own, either into a sustained destination note
             ("slide-in") or dying away ("slide-out").
  parallel   two or more voices ramping in lockstep -- a power chord slid up
             or down the neck. Invisible to a scan that walks notes in pitch
             order, because interleaved voices read 48, 55, 49, 56, ... and
             never look chromatic.
  anchored   a sustained note per voice at the pitch the ramp starts from,
             overlapping the ramp's head. Physically impossible on one string
             -- the arranger left the chord ringing *and* wrote the slide --
             so the sustained note is the slide's origin, not an obstruction.

The repair rewrites a gesture as one sustained note per voice at the *anchor*
pitch plus a single shared `0xE0` bend trajectory. All voices move by the same
interval at the same instants, so one channel-wide bend serves the whole chord.
`rppread._read_events` handles `0x90`/`0x80` and counts `0xB0`; a `0xE0` event
falls through while still advancing `tick`, so the bend is rendered into the
audio by REAPER but never reaches the corpus MIDI. The label becomes one note
per voice, which is the convention MT3 itself is built around
(`ignore_pitch_bends=True`, `mt3/tasks.py`).

An anchored gesture's sustained note is *absorbed*: it and the ramp become a
single note, so the impossible hold-and-slide overlap disappears rather than
being trimmed or preserved.

Detection and repair share `find_glides`, so what `reaper2mt3 lint` reports is
by construction exactly what `reaper2mt3 fix-slides` rewrites.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol, Sequence

# Detector thresholds. Only ramps built from micro-notes are export artifacts:
# a chromatic run played at normal note lengths is real playing (Deep Purple's
# "Child In Time" is the motivating counter-example).
CLUSTER_SECONDS = 0.010        # onsets this close are one chord of a ramp
# Observed in the corpus: voices of a chord are staggered by <5 ms, while
# consecutive steps of a ramp are >=17 ms apart, so 10 ms separates them
# with margin at both ends.
RUN_GAP_SECONDS = 0.12
RUN_MIN_STEPS = 4              # minimum chords in a ramp
# A 4-step ramp of single semitones can only span 3, so this is close to the
# floor. It was 5 while chains ignored direction and a wobble could chain into
# a long run needing a span test to exclude it; monotonic chains do that job
# now.
SLIDE_MIN_SEMITONES = 3

# Ramp steps are measured in musical time, not seconds. Guitar Pro subdivides a
# slide on a note-value grid, so a step's length in seconds scales with tempo:
# the same 1/10-of-a-quarter figure measures 23 ms in a fast song and 74 ms in a
# slow one. A seconds threshold therefore rewrote the fast copies and left the
# slow ones in place -- Pink Floyd's "Hey You" at 56 BPM kept five ramps whose
# steps were plain 64th notes, while identical gestures elsewhere were repaired.
#
# A step of a 32nd-note triplet (1/12 of a quarter) or finer is subdivision
# filler at any tempo: nothing in this corpus plays that fast deliberately.
RAMP_STEP_FILLER_QN = 0.10
# From there up to a 32nd note the shape is genuinely ambiguous -- Deep Purple's
# "Child In Time" plays real 16th-note chromatic runs and Metallica's "Nothing
# Else Matters" a real scale in octaves, both in this band -- so corroboration
# is required: strict semitone steps beneath a sustained origin.
RAMP_STEP_MAX_QN = 0.126

# A ramp's final step this much longer than the rest is a sustained
# destination; a note this long overlapping the head is a sustained origin.
DESTINATION_MIN_SECONDS = 0.100
ANCHOR_MIN_SECONDS = 0.100
# How far a sustained note's end may sit from the ramp's start and still be
# that ramp's origin. Voices in these exports are staggered by a few ms, so one
# voice can end exactly at the ramp start while its partner ends just after;
# a strict test would call one an anchor and the other an obstruction.
ANCHOR_JOIN_SECONDS = 0.030

BEND_CENTRE = 8192
BEND_STEP_SECONDS = 0.005      # ~200 Hz updates: smooth without bloating the RPP
DEFAULT_BEND_RANGE = 12

EVENT = re.compile(
    r"^(?P<indent>\s*)[Ee]\s+(?P<delta>\d+)\s+(?P<status>[0-9a-fA-F]{2})\s+"
    r"(?P<d1>[0-9a-fA-F]{2})\s+(?P<d2>[0-9a-fA-F]{2})\s*$")

# Directory names never scanned for projects: our own conversion backups and
# REAPER's automatic ones. A backup RPP shares its stem with the live project,
# so importing both would register two `source_midi_id`s for one song and risk
# a near-duplicate pair landing in different splits.
BACKUP_DIR_NAMES = {"backup", "backups"}
BACKUP_DIR_NAME = "backup"


class NoteLike(Protocol):
    start: int
    end: int
    pitch: int
    velocity: int
    channel: int


@dataclass(frozen=True)
class Glide:
    """One slide gesture, ready to be written as notes + a shared bend.

    Every voice shares one bend trajectory in semitones, which is what makes a
    channel-wide bend correct: the voices move together by the same interval.
    """

    kind: str                     # "slide-in" | "slide-out"
    shape: str                    # "bare" | "parallel" | "anchored"
    pitches: tuple[int, ...]      # anchor pitch per voice -- what the label says
    velocities: tuple[int, ...]
    note_start: int               # tick: note-on for every voice
    note_end: int                 # tick: note-off for every voice
    glide_start: int              # tick the bend leaves centre
    glide_end: int                # tick the bend arrives
    bend_from: int                # semitones relative to the anchor
    bend_to: int
    channel: int
    drop: tuple[int, ...]         # note indices this replaces
    at_seconds: float

    @property
    def voices(self) -> int:
        return len(self.pitches)

    @property
    def note_count(self) -> int:
        """Notes removed; the gesture writes `voices` notes back."""
        return len(self.drop)

    @property
    def max_bend(self) -> int:
        return max(abs(self.bend_from), abs(self.bend_to))


@dataclass(frozen=True)
class SkippedRun:
    """A ramp that is not safe to convert.

    Carries its own position so a lint failure says where to go and fix it by
    hand, rather than only that something is wrong.
    """

    indices: tuple[int, ...]
    reason: str                   # "polyphonic"
    at_seconds: float
    overlapping: int
    start_tick: int = 0

    @property
    def note_count(self) -> int:
        return len(self.indices)

    def where(self, ppq: int, time_signature: tuple[int, int] = (4, 4)) -> str:
        """`bar.beat @ m:ss` -- REAPER's transport format plus a clock time."""
        minutes, rest = divmod(self.at_seconds, 60)
        return (f"bar {bar_beat(self.start_tick, ppq, time_signature)} "
                f"@ {int(minutes)}:{rest:06.3f}")


def bar_beat(tick: int, ppq: int, time_signature: tuple[int, int] = (4, 4)) -> str:
    """REAPER's transport position, `bar.beat`, honouring the *denominator*.

    A beat is the note value the signature's denominator names, not always a
    quarter note: in 3/16 a bar is three sixteenths, i.e. 0.75 QN. Assuming /4
    put every reported position in such a project out by a factor of four.
    """
    numerator, denominator = time_signature
    beats = (tick / ppq) / (4.0 / denominator)
    bar = int(beats // numerator) + 1
    beat = beats % numerator + 1
    return f"{bar}.{beat:.2f}"


def is_backup_path(path: Path) -> bool:
    """True if any directory component is a backup folder."""
    return any(part.lower() in BACKUP_DIR_NAMES for part in path.parts[:-1])


def _clusters(notes: Sequence[NoteLike],
              seconds_at: Callable[[int], float]) -> list[list[int]]:
    """Groups note indices into chords: onsets close together *and* sounding together.

    Onset proximity alone is not enough. The fastest ramps in this corpus step
    every 3-5 ms, inside the chord window, so their steps were merged in pairs:
    six steps read as three two-note "chords", dropping the chain under
    RUN_MIN_STEPS so it was reported as an unconvertible residual instead of
    being repaired. Child In Time and Enter Sandman were mostly this.

    What separates them is whether the notes ever sound at once. A chord's
    voices overlap -- they are struck together and ring together. Ramp steps are
    strictly sequential: each ends on the tick the next begins.
    """
    groups: list[list[int]] = []
    current = [0]
    for i in range(1, len(notes)):
        close = (seconds_at(notes[i].start) - seconds_at(notes[current[0]].start)
                 < CLUSTER_SECONDS)
        sounding = notes[i].start < max(notes[j].end for j in current)
        if close and sounding:
            current.append(i)
        else:
            groups.append(current)
            current = [i]
    groups.append(current)
    return groups


def _step_delta(notes, a: Sequence[int], b: Sequence[int]) -> int | None:
    """The single semitone step from chord `a` to chord `b`, or None."""
    if len(a) != len(b):
        return None
    pa = sorted(notes[i].pitch for i in a)
    pb = sorted(notes[i].pitch for i in b)
    deltas = {y - x for x, y in zip(pa, pb)}
    if len(deltas) != 1:
        return None
    delta = deltas.pop()
    return delta if 1 <= abs(delta) <= 2 else None


def _strict_semitones(notes, chain: Sequence[Sequence[int]]) -> bool:
    """Every step exactly one semitone -- how an interpolated ramp is written.

    Real playing in the ambiguous band mixes tones with semitones: the Hotel
    California solo lick and Nothing Else Matters' octave scale both step by 2
    as well as 1, while every ghost-sustain ramp found in this corpus is strict.
    Coarser ramps do exist -- whole-tone steps of 5-10 ms -- but those are far
    below RAMP_STEP_FILLER_QN and never reach this test.
    """
    steps = (_step_delta(notes, chain[k], chain[k + 1]) for k in range(len(chain) - 1))
    return all(d is not None and abs(d) == 1 for d in steps)


def _is_filler(step_qn: float, *, anchored: bool, strict: bool,
               dies_with_anchor: bool) -> bool:
    """Whether a ramp's steps are export filler rather than played notes."""
    if anchored and dies_with_anchor:
        # The ramp stops at the very instant its sustained origin does. Two
        # independent voices do not repeatedly end on the same tick; this is one
        # gesture the exporter split in two, so step length says nothing.
        return True
    if step_qn < RAMP_STEP_FILLER_QN:
        return True
    return anchored and strict and step_qn < RAMP_STEP_MAX_QN


def _ramp_chains(notes: Sequence[NoteLike], groups: Sequence[Sequence[int]],
                 seconds_at: Callable[[int], float]) -> list[list[Sequence[int]]]:
    """Maximal chains of chords transposed by a constant small step.

    A chain must keep one direction. Without that, a slide up followed by a
    slide back down chains into a single run whose *net* span is near zero, and
    both real gestures are rejected for being too small (Iron Man's 53->60->55
    figure is the motivating case).
    """
    chains, i = [], 0
    while i < len(groups) - 1:
        chain = [groups[i]]
        j = i
        direction = 0
        while j + 1 < len(groups):
            head, nxt = groups[j], groups[j + 1]
            if seconds_at(notes[nxt[0]].start) - seconds_at(notes[head[0]].start) >= RUN_GAP_SECONDS:
                break
            delta = _step_delta(notes, head, nxt)
            if delta is None:
                break
            step_direction = 1 if delta > 0 else -1
            if direction and step_direction != direction:
                break
            direction = step_direction
            chain.append(nxt)
            j += 1
        if len(chain) >= RUN_MIN_STEPS:
            following = groups[j + 1] if j + 1 < len(groups) else None
            chains.append((chain, following, direction))
            i = j + 1
        else:
            i += 1
    return chains


def _landing(notes: Sequence[NoteLike], chain: Sequence[Sequence[int]],
             following: Sequence[int] | None, direction: int,
             seconds_at: Callable[[int], float]) -> list[int] | None:
    """A sustained chord the ramp lands on, even if it has extra voices.

    The chain itself requires equal voice counts step to step, so a ramp that
    lands on a fuller chord -- one string sliding into a two-note chord, which
    is ordinary guitar writing -- breaks before its own destination and loses
    the semitones that make it a slide. Matched per-pitch, like anchors.
    """
    if not following or not direction:
        return None
    landing = []
    for i in sorted(chain[-1], key=lambda i: notes[i].pitch):
        pitch = notes[i].pitch
        match = [
            k for k in following
            if 1 <= abs(notes[k].pitch - pitch) <= 2
            and (notes[k].pitch - pitch) * direction > 0
            and seconds_at(notes[k].end) - seconds_at(notes[k].start)
            >= DESTINATION_MIN_SECONDS]
        if len(match) != 1:
            return None
        landing.append(match[0])
    return landing


def _residual_ramps(notes: Sequence[NoteLike],
                    seconds_at: Callable[[int], float],
                    ppq: int,
                    reported: set[int]) -> list[SkippedRun]:
    """Micro-note ramps the detector neither converted nor already flagged.

    A chain that breaks -- most often because two independent voices interleave
    and the cluster sizes stop matching -- yields no gesture *and* no skip, so
    without this the ramp is silently left in place. Reported so lint can point
    at it for a manual fix rather than saying nothing.

    Normal-length notes are stepped over rather than ending the run. A sustained
    voice that merely *starts on the same tick* as a ramp step used to truncate
    the run, and a 4-step ramp losing its colliding first step fell under
    RUN_MIN_STEPS and vanished from both paths (Hey You, bar 34.4.75). The
    RUN_GAP_SECONDS test between consecutive micro-notes still ends the run, so
    stepping over a long note cannot join two unrelated ramps.
    """
    runs: list[SkippedRun] = []
    current: list[int] = []

    def flush():
        if len(current) >= RUN_MIN_STEPS:
            runs.append(SkippedRun(
                tuple(current), "unconverted micro-note ramp",
                seconds_at(notes[current[0]].start), 0,
                notes[current[0]].start))

    for i, note in enumerate(notes):
        # Per note there is no chain to corroborate, so only the unambiguous
        # threshold applies.
        micro = (note.end - note.start) / ppq < RAMP_STEP_FILLER_QN
        if not micro:
            continue
        if i in reported:
            flush()
            current = []
            continue
        if current:
            previous = notes[current[-1]]
            contiguous = (1 <= abs(note.pitch - previous.pitch) <= 2
                          and seconds_at(note.start) - seconds_at(previous.start)
                          < RUN_GAP_SECONDS)
            if not contiguous:
                flush()
                current = []
        current.append(i)
    flush()
    return runs


def find_glides(
    notes: Sequence[NoteLike],
    seconds_at: Callable[[int], float],
    ppq: int,
) -> tuple[list[Glide], list[SkippedRun]]:
    """Splits a part's notes into convertible glides and unsafe ramps.

    Ramps of normal-length notes are real playing and appear in neither list.
    A ramp overlapped by a note that is not its own sustained origin is
    skipped: a channel-wide bend would drag that note out of tune.
    """
    if len(notes) < RUN_MIN_STEPS:
        return [], []

    glides: list[Glide] = []
    skipped: list[SkippedRun] = []
    groups = _clusters(notes, seconds_at)

    for chain, following, direction in _ramp_chains(notes, groups, seconds_at):
        flat = [i for step in chain for i in step]
        durations = [seconds_at(notes[i].end) - seconds_at(notes[i].start) for i in flat]

        # A long final step is a sustained destination, not part of the ramp.
        ramp = list(chain)
        destination: Sequence[int] | None = None
        last = [seconds_at(notes[i].end) - seconds_at(notes[i].start) for i in chain[-1]]
        body = [d for i, d in zip(flat, durations) if i not in set(chain[-1])]
        if (len(chain) > RUN_MIN_STEPS and body
                and statistics.median(last) >= DESTINATION_MIN_SECONDS
                and statistics.median(last) > 3 * statistics.median(body)):
            destination = chain[-1]
            ramp = chain[:-1]
        else:
            # The ramp may land on a chord with more voices than itself, which
            # ends the chain one step early; recover that landing per-pitch.
            landing = _landing(notes, chain, following, direction, seconds_at)
            if landing:
                destination = landing
                flat = flat + landing

        head, tail_step = ramp[0], ramp[-1]
        head_pitches = sorted(notes[i].pitch for i in head)
        voices = len(head)
        ramp_start = min(notes[i].start for i in head)
        ramp_end = max(notes[i].end for i in tail_step)

        # Sustained origins: one per voice, adjacent in pitch to the ramp head,
        # already sounding when the ramp begins. The arranger's impossible
        # hold-and-slide; the sustained note is the slide's origin.
        member = {i for step in chain for i in step}
        anchors = [
            i for i, n in enumerate(notes)
            if i not in member and n.start < ramp_start
            and seconds_at(n.end) >= seconds_at(ramp_start) - ANCHOR_JOIN_SECONDS
            and seconds_at(n.end) - seconds_at(n.start) >= ANCHOR_MIN_SECONDS
            and any(abs(n.pitch - p) in (1, 2) for p in head_pitches)]

        gesture_start = ramp_start
        anchored = len(anchors) == voices and voices > 0
        if anchored:
            gesture_start = min(notes[i].start for i in anchors)

        # Are these steps filler, or real playing? Measured over the ramp alone:
        # a sustained destination is a real note, and averaging it in inflated
        # the figure enough to hide short filler behind it (Child In Time's
        # 60 -> 61, 62 -> 63 slide-in, whose 15 ms steps read as 54 ms).
        step_qn = statistics.median(
            (notes[i].end - notes[i].start) / ppq for step in ramp for i in step)
        dies_with_anchor = anchored and abs(
            max(seconds_at(notes[i].end) for i in anchors)
            - seconds_at(ramp_end)) < CLUSTER_SECONDS
        if not _is_filler(step_qn, anchored=anchored,
                          strict=_strict_semitones(notes, ramp),
                          dies_with_anchor=dies_with_anchor):
            continue                      # real chromatic playing

        # Never relabel a note that already exists -- only micro-notes are
        # removed. A sustained origin keeps its own pitch and bends away; a
        # sustained destination is left untouched entirely. Only where an end
        # of the gesture has no real note does one get written.
        ramp_only = {i for step in ramp for i in step}
        if destination is not None and anchored:
            # Hold, slide, hold. Both real notes survive; the origin bends
            # across the ramp and the destination is not touched at all.
            kind = "slide-out"
            pitches = sorted(notes[i].pitch for i in anchors)
            velocities = [notes[i].velocity for i in sorted(
                anchors, key=lambda i: notes[i].pitch)]
            note_start = gesture_start
            note_end = ramp_end
            glide_start = ramp_start
            glide_end = ramp_end
            bend_from = 0
            bend_to = sorted(notes[i].pitch for i in destination)[0] - pitches[0]
            drop = set(anchors) | ramp_only
        elif destination is not None:
            # No real origin note: the destination absorbs the ramp, which is
            # the "slide into" ornament a transcriber writes as one note.
            kind = "slide-in"
            pitches = sorted(notes[i].pitch for i in destination)
            velocities = [notes[i].velocity for i in sorted(
                destination, key=lambda i: notes[i].pitch)]
            note_start = ramp_start
            note_end = max(notes[i].end for i in destination)
            glide_start = ramp_start
            glide_end = min(notes[i].start for i in destination)
            bend_from = head_pitches[0] - pitches[0]
            bend_to = 0
            drop = ramp_only | set(destination)
        elif anchored:
            # A fall away from a real note: it keeps its pitch and its onset,
            # and simply plays on through the ramp while bending.
            kind = "slide-out"
            pitches = sorted(notes[i].pitch for i in anchors)
            velocities = [notes[i].velocity for i in sorted(
                anchors, key=lambda i: notes[i].pitch)]
            note_start = gesture_start
            note_end = ramp_end
            glide_start = ramp_start
            glide_end = ramp_end
            bend_from = 0
            bend_to = sorted(notes[i].pitch for i in tail_step)[0] - pitches[0]
            drop = set(anchors) | ramp_only
        else:
            # Nothing real at either end; one note is written at the origin.
            kind = "slide-out"
            pitches = head_pitches
            velocities = [notes[i].velocity for i in sorted(
                head, key=lambda i: notes[i].pitch)]
            note_start = ramp_start
            note_end = ramp_end
            glide_start = ramp_start
            glide_end = ramp_end
            bend_from = 0
            bend_to = sorted(notes[i].pitch for i in tail_step)[0] - pitches[0]
            drop = ramp_only

        if abs(bend_to - bend_from) < SLIDE_MIN_SEMITONES:
            continue

        # Only a note sounding while the bend is *away from centre* can be
        # detuned by it. A note that stops as the ramp begins -- very common,
        # since that is usually the chord the slide leaves from -- is safe, and
        # so is anything after the note has ended and the bend has recentred.
        covered = member | set(anchors) | set(drop)
        obstructions = [
            i for i, n in enumerate(notes)
            if i not in covered and n.start < note_end and n.end > glide_start]
        if obstructions:
            skipped.append(SkippedRun(tuple(flat), "polyphonic",
                                      seconds_at(ramp_start), len(obstructions),
                                      ramp_start))
            continue

        # Every voice must need the same bend, or one channel cannot serve them.
        target = destination if (destination is not None and anchored) else tail_step
        if destination is not None and not anchored:
            source = sorted(notes[i].pitch for i in head)
            offsets = {s - p for s, p in zip(source, pitches)}
        else:
            ends = sorted(notes[i].pitch for i in target)
            offsets = {e - p for e, p in zip(ends, pitches)}
        if len(offsets) != 1:
            skipped.append(SkippedRun(tuple(flat), "voices need different bends",
                                      seconds_at(ramp_start), voices, ramp_start))
            continue

        shape = "anchored" if anchored else ("parallel" if voices > 1 else "bare")
        glides.append(Glide(
            kind=kind, shape=shape,
            pitches=tuple(pitches), velocities=tuple(velocities),
            note_start=note_start, note_end=note_end,
            glide_start=glide_start, glide_end=glide_end,
            bend_from=bend_from, bend_to=bend_to,
            channel=notes[head[0]].channel,
            drop=tuple(sorted(drop)),
            at_seconds=seconds_at(note_start),
        ))

    reported = {i for g in glides for i in g.drop}
    reported |= {i for s in skipped for i in s.indices}
    skipped.extend(_residual_ramps(notes, seconds_at, ppq, reported))
    skipped.sort(key=lambda s: s.at_seconds)
    return glides, skipped


# --------------------------------------------------------------------------
# Rewriting
# --------------------------------------------------------------------------


@dataclass
class _ParsedNote:
    """A note plus the indices of the E-lines that opened and closed it."""

    start: int
    end: int
    pitch: int
    velocity: int
    channel: int
    on_event: int
    off_event: int


def parse_events(lines: Sequence[str], lo: int, hi: int) -> list[list[int]]:
    """E-lines in [lo,hi) -> [absolute_tick, status, d1, d2, line_index]."""
    events: list[list[int]] = []
    tick = 0
    for i in range(lo, hi):
        match = EVENT.match(lines[i])
        if match is None:
            continue
        tick += int(match.group("delta"))
        events.append([tick, int(match.group("status"), 16),
                       int(match.group("d1"), 16), int(match.group("d2"), 16), i])
    return events


def build_notes(events: Sequence[Sequence[int]]) -> list[_ParsedNote]:
    """Pairs note-ons to note-offs FIFO per (channel, pitch), as rppread does."""
    open_notes: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    notes: list[_ParsedNote] = []
    for index, (tick, status, d1, d2, _line) in enumerate(events):
        kind, channel = status & 0xF0, status & 0x0F
        if kind == 0x90 and d2 > 0:
            open_notes.setdefault((channel, d1), []).append((tick, d2, index))
        elif kind in (0x80, 0x90):
            queue = open_notes.get((channel, d1))
            if queue:
                start, velocity, on_index = queue.pop(0)
                notes.append(_ParsedNote(start, max(tick, start + 1), d1,
                                         velocity, channel, on_index, index))
    notes.sort(key=lambda n: (n.start, n.pitch))
    return notes


def locate_track_block(lines: Sequence[str], track_name: str) -> tuple[int, int]:
    """Line range of the <TRACK> block whose track-level NAME is `track_name`.

    Bounded by `<TRACK` markers and matched on the track-level NAME line, never
    on MIDI-item label text -- item labels are cosmetic and go stale.
    """
    starts = [i for i, line in enumerate(lines) if line.strip().startswith("<TRACK")]
    name_index = next(
        (i for i, line in enumerate(lines)
         if line.strip().startswith("NAME ") and track_name in line), None)
    if name_index is None:
        raise ValueError(f"no track-level NAME matching {track_name!r}")
    start = max(s for s in starts if s < name_index)
    end = min([s for s in starts if s > name_index] + [len(lines)])
    return start, end


def _bend_bytes(semitones: float, bend_range: int) -> tuple[int, int]:
    fraction = max(-1.0, min(1.0, semitones / bend_range))
    value = max(0, min(16383, int(round(BEND_CENTRE + fraction * 8191))))
    return value & 0x7F, (value >> 7) & 0x7F


def _replacement_events(glide: Glide, seconds_at: Callable[[int], float],
                        bend_range: int) -> list[list[int]]:
    """Notes + one shared bend for a gesture, whatever its voice count."""
    channel = glide.channel
    events: list[list[int]] = []

    # Bend range (RPN 0), re-asserted per gesture because some plugins reset
    # RPN state on transport stop or program change.
    for controller, value in ((0x65, 0x00), (0x64, 0x00), (0x06, bend_range)):
        events.append([glide.note_start, 0xB0 | channel, controller, value])

    lsb, msb = _bend_bytes(glide.bend_from, bend_range)
    events.append([glide.note_start, 0xE0 | channel, lsb, msb])
    for pitch, velocity in zip(glide.pitches, glide.velocities):
        events.append([glide.note_start, 0x90 | channel, pitch, velocity])

    # An anchored gesture holds at the starting offset while the note sustains,
    # then departs only once the ramp begins.
    if glide.glide_start > glide.note_start:
        events.append([glide.glide_start, 0xE0 | channel, lsb, msb])

    span = seconds_at(glide.glide_end) - seconds_at(glide.glide_start)
    steps = max(1, int(round(span / BEND_STEP_SECONDS)))
    for step in range(1, steps + 1):
        fraction = step / steps
        tick = int(round(glide.glide_start
                         + (glide.glide_end - glide.glide_start) * fraction))
        semitones = glide.bend_from + (glide.bend_to - glide.bend_from) * fraction
        lsb, msb = _bend_bytes(semitones, bend_range)
        events.append([tick, 0xE0 | channel, lsb, msb])

    for pitch in glide.pitches:
        events.append([glide.note_end, 0x80 | channel, pitch, 0x00])
    # Recentre after the note so later notes are not left detuned.
    events.append([glide.note_end, 0xE0 | channel, 0x00, 0x40])
    return events


def _event_sort_key(event: Sequence[int]) -> tuple[int, int]:
    # At equal ticks: note-offs, then RPN, then bends, then note-ons -- so the
    # bend is always in place before the note it applies to sounds.
    rank = {0x80: 0, 0xB0: 1, 0xE0: 2, 0x90: 3}.get(event[1] & 0xF0, 4)
    return event[0], rank


def backup_path(rpp: Path) -> Path:
    """Where a project's pre-conversion copy is kept: `<dir>/backup/<name>`."""
    return rpp.parent / BACKUP_DIR_NAME / rpp.name


@dataclass
class TrackResult:
    track_name: str
    converted: list[Glide]
    skipped: list[SkippedRun]
    notes_before: int
    notes_after: int
    events_before: int
    events_after: int


@dataclass
class ProjectResult:
    path: Path
    tracks: list[TrackResult]
    backup: Path | None = None
    error: str | None = None

    @property
    def converted(self) -> int:
        return sum(len(t.converted) for t in self.tracks)

    @property
    def skipped(self) -> int:
        return sum(len(t.skipped) for t in self.tracks)

    @property
    def changed(self) -> bool:
        return self.converted > 0


def scan_project(project, only_tracks: set[str] | None = None
                 ) -> dict[str, tuple[list[Glide], list[SkippedRun]]]:
    """Detection only, straight off a parsed project. Used by lint."""
    found = {}
    for part in project.parts:
        if only_tracks and part.track_name not in only_tracks:
            continue
        notes = sorted(part.notes, key=lambda n: (n.start, n.pitch))
        glides, skipped = find_glides(notes, project.seconds_at, project.ppq)
        if glides or skipped:
            found[part.track_name] = (glides, skipped)
    return found


def _emit_events(events: Sequence[Sequence[int]], indent: str) -> list[str]:
    lines, previous = [], 0
    for tick, status, d1, d2 in events:
        lines.append(f"{indent}E {tick - previous} {status:02x} {d1:02x} {d2:02x}")
        previous = tick
    return lines


def convert_project(rpp: Path, *, bend_range: int = DEFAULT_BEND_RANGE,
                    dry_run: bool = False,
                    only_tracks: set[str] | None = None) -> ProjectResult:
    """Rewrites every convertible glide in `rpp`, in place, with a backup.

    Aborts without writing on any inconsistency: a parse that disagrees with
    `rppread`, a gesture needing more bend than declared, or a post-write check
    that fails. This project has been bitten enough by tools that exit 0 having
    silently done the wrong thing.
    """
    from .rppread import read_project

    project = read_project(rpp)
    seconds_at = project.seconds_at
    # Work in bytes: text-mode I/O silently translates CRLF to LF, which would
    # make the backup differ from the original and rewrite every line ending in
    # the converted file. A minority of REAPER projects really are CRLF.
    original_bytes = rpp.read_bytes()
    text = original_bytes.decode("utf-8", errors="replace")
    newline = "\r\n" if "\r\n" in text else "\n"
    lines = text.split(newline)
    if newline == "\r\n" and any("\n" in line for line in lines):
        return ProjectResult(rpp, [], error="mixed line endings -- refusing to rewrite")

    parts = [p for p in project.parts
             if not only_tracks or p.track_name in only_tracks]
    if not parts:
        return ProjectResult(rpp, [], error="no matching tracks")

    # Collect per-track edits first, then splice from the last block backwards
    # so earlier line indices stay valid.
    edits: list[tuple[int, int, list[str], TrackResult]] = []
    for part in parts:
        try:
            lo, hi = locate_track_block(lines, part.track_name)
        except ValueError as error:
            return ProjectResult(rpp, [], error=str(error))
        events = parse_events(lines, lo, hi)
        if not events:
            continue
        notes = build_notes(events)

        # Compare as multisets: rppread appends a note when its note-off is
        # seen, while build_notes sorts by onset, so the orderings differ even
        # when the content is identical.
        mine = sorted((n.start, n.end, n.pitch, n.velocity) for n in notes)
        theirs = sorted((n.start, n.end, n.pitch, n.velocity) for n in part.notes)
        if mine != theirs:
            if len(mine) != len(theirs):
                detail = f"parsed {len(mine)} notes but rppread reports {len(theirs)}"
            else:
                differing = sum(1 for a, b in zip(mine, theirs) if a != b)
                detail = f"{differing} of {len(mine)} notes differ from rppread's reading"
            return ProjectResult(
                rpp, [], error=f"{part.track_name}: {detail} -- refusing to rewrite")

        glides, skipped = find_glides(notes, seconds_at, project.ppq)
        if not glides:
            if skipped:
                edits.append((lo, hi, [], TrackResult(
                    part.track_name, [], skipped, len(notes), len(notes),
                    len(events), len(events))))
            continue

        over = [g for g in glides if g.max_bend > bend_range]
        if over:
            return ProjectResult(
                rpp, [], error=f"{part.track_name}: {len(over)} gesture(s) need more "
                               f"than {bend_range} semitones of bend (max "
                               f"{max(g.max_bend for g in over)})")

        dropped = {i for g in glides for note_index in g.drop
                   for i in (notes[note_index].on_event, notes[note_index].off_event)}
        kept = [list(e[:4]) for i, e in enumerate(events) if i not in dropped]
        for glide in glides:
            kept.extend(_replacement_events(glide, seconds_at, bend_range))
        kept.sort(key=_event_sort_key)

        indent = EVENT.match(lines[events[0][4]]).group("indent")
        new_lines = _emit_events(kept, indent)
        removed = sum(g.note_count for g in glides)
        written = sum(g.voices for g in glides)
        edits.append((events[0][4], events[-1][4] + 1, new_lines, TrackResult(
            part.track_name, glides, skipped, len(notes),
            len(notes) - removed + written, len(events), len(kept))))

    results = [t for _lo, _hi, _new, t in edits]
    if not any(t.converted for t in results):
        return ProjectResult(rpp, results)
    if dry_run:
        return ProjectResult(rpp, results)

    updated = list(lines)
    for lo, hi, new_lines, track in sorted(edits, key=lambda e: -e[0]):
        if track.converted:
            updated[lo:hi] = new_lines

    backup = backup_path(rpp)
    backup.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing backup: the first one is the pristine
    # pre-conversion original and must survive repeated runs. Copied as raw
    # bytes so it is byte-identical, line endings included.
    if not backup.exists():
        backup.write_bytes(original_bytes)

    rpp.write_bytes(newline.join(updated).encode("utf-8"))

    verified = read_project(rpp)
    problems = []
    if verified.last_tick != project.last_tick:
        problems.append(f"last_tick changed {project.last_tick} -> {verified.last_tick}")
    converted_names = {t.track_name for t in results if t.converted}
    before = {p.track_name: [(n.start, n.end, n.pitch, n.velocity) for n in p.notes]
              for p in project.parts}
    for part in verified.parts:
        if part.track_name in converted_names:
            continue
        now = [(n.start, n.end, n.pitch, n.velocity) for n in part.notes]
        if before.get(part.track_name) != now:
            problems.append(f"untouched track {part.track_name!r} changed")
    if problems:
        rpp.write_bytes(original_bytes)
        return ProjectResult(rpp, results, backup=backup,
                             error="post-write check failed, reverted: "
                                   + "; ".join(problems))
    return ProjectResult(rpp, results, backup=backup)
