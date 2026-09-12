"""Guitar-Pro slide gestures: detection, classification and pitch-bend repair.

The corpus audit traced the model's runaway chromatic-ramp output to slides
exported as runs of chromatic micro-notes. One detector covers all three shapes
the export produces -- bare, parallel (a chord slid in lockstep) and anchored
(a sustained origin note the arranger left ringing through the slide).

These pin the things that must stay true: a chromatic run of *normal-length*
notes is real playing and is never touched; one bend must serve every voice of
a chord; and an anchored gesture's impossible hold-and-slide overlap must be
absorbed, not preserved.
"""

from pathlib import Path

from reaper2mt3 import slides
from reaper2mt3.rppread import read_project


CHAIN_CHUNK = """      <VST "VST3i: Kontakt 8 (Native Instruments) (64 out)" "Kontakt 8.vst3" 0 "" 9 ""
        AAAA
      >
      FXID {A0000000-0000-0000-0000-000000000002}"""

PPQ = 480      # ticks per quarter note; at the fixture's TEMPO 120, 1 tick ~ 1.04 ms
STEP = 20      # ~21 ms between ramp steps, as in the real exports
DUR = 16       # ~17 ms micro-note
LONG = PPQ     # a 500 ms sustained note


def _events(notes):
    """[(start_tick, end_tick, pitch, velocity)] -> RPP E-lines."""
    points = []
    for start, end, pitch, velocity in notes:
        points.append((start, 0x90, pitch, velocity))
        points.append((end, 0x80, pitch, 0))
    points.sort(key=lambda p: (p[0], p[1] == 0x90))
    lines, previous = [], 0
    for tick, status, pitch, value in points:
        lines.append(f"        E {tick - previous} {status:02x} {pitch:02x} {value:02x}")
        previous = tick
    return "\n".join(lines)


def _project(notes, name="electric-guitar-clean", tempo=120):
    return "\n".join([
        '<REAPER_PROJECT 0.1 "7.74" 0 0',
        f"  TEMPO {tempo} 4 4 0",
        "  <TRACK {G}",
        f'    NAME "{name}"',
        "    <FXCHAIN",
        CHAIN_CHUNK,
        "    >",
        "    <ITEM",
        "      <SOURCE MIDI",
        f"        HASDATA 1 {PPQ} QN",
        _events(notes),
        "      >",
        "    >",
        "  >",
        ">",
    ])


def _write(tmp_path, notes, name="electric-guitar-clean", tempo=120):
    path = tmp_path / "project.RPP"
    path.write_text(_project(notes, name, tempo))
    return path


def _ramp(start, first_pitch, count, step=STEP, dur=DUR, ascending=True, offset=0):
    """A chromatic ramp of micro-notes. `offset` staggers a parallel voice."""
    out = []
    for i in range(count):
        pitch = first_pitch + (i if ascending else -i)
        tick = start + i * step + offset
        out.append((tick, tick + dur, pitch, 0x5F))
    return out


def _glides(tmp_path, notes, tempo=120):
    path = _write(tmp_path, notes, tempo=tempo)
    project = read_project(path)
    ordered = sorted(project.parts[0].notes, key=lambda n: (n.start, n.pitch))
    return slides.find_glides(ordered, project.seconds_at, project.ppq)


# --------------------------------------------------------------------------
# Detection -- bare
# --------------------------------------------------------------------------


def test_ramp_into_sustained_note_is_a_slide_in_anchored_on_destination(tmp_path):
    """Micro-notes into a long note: the long note is the real one."""
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    runs, skipped = _glides(tmp_path, notes)
    assert skipped == []
    assert len(runs) == 1
    run = runs[0]
    assert run.kind == "slide-in" and run.shape == "bare" and run.voices == 1
    assert run.pitches == (69,)          # destination, not the ramp's first pitch
    assert (run.bend_from, run.bend_to) == (-9, 0)
    assert run.max_bend == 9


def test_fall_into_silence_is_a_slide_out_anchored_on_origin(tmp_path):
    """A descending gliss dying away: the first note is the real one."""
    notes = _ramp(0, 75, 9, ascending=False)
    runs, _ = _glides(tmp_path, notes)
    assert len(runs) == 1
    assert runs[0].kind == "slide-out" and runs[0].shape == "bare"
    assert runs[0].pitches == (75,)      # the plucked note, bending away
    assert (runs[0].bend_from, runs[0].bend_to) == (0, -8)


def test_chromatic_run_of_normal_length_notes_is_real_playing(tmp_path):
    """Deep Purple's "Child In Time" case: must never be rewritten.

    Spaced closely enough to form a chain, but each note is long enough to be
    played rather than an articulation artifact.
    """
    notes = _ramp(0, 60, 10, step=100, dur=100)
    runs, skipped = _glides(tmp_path, notes)
    assert runs == [] and skipped == []


def test_a_ramp_faster_than_the_chord_window_is_not_read_as_chords(tmp_path):
    """Steps closer together than CLUSTER_SECONDS -- the fastest ramps here.

    Grouping on onset proximity alone merged consecutive steps into two-note
    "chords", halving the step count until the chain fell under RUN_MIN_STEPS
    and the ramp was reported as an unconvertible residual. Most of Enter
    Sandman's and Child In Time's blocked runs were this.
    """
    step = 5                                    # ~5 ms at the fixture's tempo
    assert step * (60 / 120 / PPQ) < slides.CLUSTER_SECONDS, "inside the chord window"
    notes = _ramp(0, 51, 6, step=step, dur=step)
    notes.append((6 * step, 6 * step + LONG, 57, 0x5F))
    runs, skipped = _glides(tmp_path, notes)
    assert len(runs) == 1 and skipped == []
    assert runs[0].kind == "slide-in"
    assert runs[0].pitches == (57,)             # absorbed into the real note


def test_a_chord_struck_together_is_still_one_cluster(tmp_path):
    """The other half of the rule: voices that overlap are a chord.

    Guarding the fix above -- if it split real chords, every parallel slide
    would stop being detected.
    """
    notes = [(0, LONG, 48, 0x5F), (2, LONG, 55, 0x5F), (4, LONG, 60, 0x5F)]
    path = _write(tmp_path, notes)
    project = read_project(path)
    ordered = sorted(project.parts[0].notes, key=lambda n: (n.start, n.pitch))
    assert slides._clusters(ordered, project.seconds_at) == [[0, 1, 2]]


def test_the_same_ramp_is_detected_at_every_tempo(tmp_path):
    """A ramp is a fixed note value, so tempo must not change the verdict.

    Guitar Pro subdivides a slide on a rhythmic grid, so a step measured in
    seconds stretches as the song slows. While the threshold was in seconds
    this ramp converted at 120 BPM and was silently left in place at 56 --
    Pink Floyd's "Hey You" kept five of them (`PROJECT_LOG.md`, 2026-08-28).
    """
    # 30 ticks at PPQ 480 is a 64th note: 15.6 ms at 120 BPM, 66.9 ms at 56.
    notes = _ramp(0, 60, 6, step=30, dur=30)
    notes.append((6 * 30, 6 * 30 + LONG, 66, 0x5F))
    verdicts = {}
    for tempo in (56, 120, 200):
        runs, _ = _glides(tmp_path, notes, tempo=tempo)
        verdicts[tempo] = [(g.kind, g.pitches, g.bend_from, g.bend_to) for g in runs]
    assert len(verdicts[120]) == 1, "the fast case was already detected"
    assert verdicts[56] == verdicts[120] == verdicts[200]


def test_ghost_sustain_dying_with_its_ramp_is_converted_however_long_the_steps(tmp_path):
    """The shape that survived Hey You's first conversion pass.

    A sustained note rings through a chromatic ramp and both stop on the same
    tick -- impossible on one string, and two independent voices do not
    repeatedly end together. That coincidence is the evidence, so it holds even
    where the step length alone would read as playable.
    """
    # 64 ticks at PPQ 480 is coarser than RAMP_STEP_MAX_QN, so nothing but the
    # shared ending can justify converting this.
    step = 64
    assert step / PPQ > slides.RAMP_STEP_MAX_QN
    ramp = _ramp(LONG, 61, 6, step=step, dur=step, ascending=False)
    notes = [(0, LONG + 6 * step, 62, 0x5F)] + ramp            # anchor ends with it
    runs, skipped = _glides(tmp_path, notes)
    assert len(runs) == 1 and skipped == []
    assert runs[0].shape == "anchored"
    assert runs[0].pitches == (62,)          # the sustained note keeps its pitch
    assert runs[0].bend_to == 56 - 62        # and bends to where the ramp ended
    # The ghost overlap is gone: every ramp note plus the anchor is replaced.
    assert runs[0].note_count == len(ramp) + 1


def test_a_real_scale_in_the_ambiguous_band_is_left_alone(tmp_path):
    """Metallica's "Nothing Else Matters" case: a played run, not a slide.

    In the band where step length alone cannot decide, conversion needs strict
    semitone steps. A scale mixes tones with semitones, so it stays put even
    with a sustained note behind it and steps as short as a ramp's.
    """
    step = 60
    assert slides.RAMP_STEP_FILLER_QN <= step / PPQ < slides.RAMP_STEP_MAX_QN
    pitches = [57, 59, 61, 62, 64, 66, 67]                     # major scale
    notes = [(LONG + i * step, LONG + (i + 1) * step, p, 0x5F)
             for i, p in enumerate(pitches)]
    notes.append((0, LONG, 55, 0x5F))                          # a sustained origin
    runs, _ = _glides(tmp_path, notes)
    assert runs == []


def test_short_interval_wobble_is_not_a_slide(tmp_path):
    """A 1-semitone trill is a bend/vibrato figure, below the slide threshold."""
    notes = [(i * STEP, i * STEP + DUR, 60 + (i % 2), 0x5F) for i in range(10)]
    runs, _ = _glides(tmp_path, notes)
    assert runs == []


# --------------------------------------------------------------------------
# Detection -- parallel and anchored
# --------------------------------------------------------------------------


def test_parallel_chord_slide_is_one_gesture_with_a_shared_bend(tmp_path):
    """A power chord slid up the neck: two voices, one bend.

    Invisible to a scan that walks notes in pitch order -- interleaved, the
    voices read 48, 55, 49, 56, ... and never look chromatic.
    """
    notes = _ramp(0, 48, 9) + _ramp(0, 55, 9, offset=2)   # a fifth apart
    runs, skipped = _glides(tmp_path, notes)
    assert skipped == []
    assert len(runs) == 1
    run = runs[0]
    assert run.shape == "parallel" and run.voices == 2
    assert run.pitches == (48, 55)
    assert (run.bend_from, run.bend_to) == (0, 8)
    assert run.note_count == 18          # both voices' ramp notes, replaced by 2


def test_sustained_origin_is_absorbed_not_treated_as_an_obstruction(tmp_path):
    """The Iron Man pattern.

    The arranger left a chord ringing *and* wrote the slide, so the same string
    would have to hold and slide at once. The sustained note is the slide's
    origin: it and the ramp become one note.
    """
    sustain = [(0, 700, 52, 0x5F), (2, 702, 59, 0x5F)]
    ramp = (_ramp(680, 51, 9, ascending=False)
            + _ramp(680, 58, 9, ascending=False, offset=2))
    runs, skipped = _glides(tmp_path, sustain + ramp)
    assert skipped == [], "the sustained origin must not count as an obstruction"
    assert len(runs) == 1
    run = runs[0]
    assert run.shape == "anchored" and run.voices == 2
    assert run.pitches == (52, 59)       # written at the sustained pitches
    assert (run.bend_from, run.bend_to) == (0, -9)
    assert run.note_start == 0           # starts where the sustain did
    assert run.note_end == max(n[1] for n in ramp)
    assert run.glide_start == 680        # bend holds at centre until the ramp
    assert run.note_count == 20          # 2 sustained + 18 ramp, replaced by 2


def test_ghost_overlap_is_gone_after_conversion(tmp_path):
    """No note may still sound while the slide it belongs to runs."""
    sustain = [(0, 700, 52, 0x5F), (2, 702, 59, 0x5F)]
    ramp = (_ramp(680, 51, 9, ascending=False)
            + _ramp(680, 58, 9, ascending=False, offset=2))
    path = _write(tmp_path, sustain + ramp)
    result = slides.convert_project(path)
    assert result.error is None and result.converted == 1

    after = sorted(read_project(path).parts[0].notes, key=lambda n: (n.start, n.pitch))
    assert len(after) == 2, f"expected 2 notes, got {[(n.pitch, n.start) for n in after]}"
    assert sorted(n.pitch for n in after) == [52, 59]
    # Nothing overlaps a ramp any more, because the ramp is gone entirely.
    assert all(n.start == 0 for n in after)


def test_a_held_note_on_another_string_still_blocks_conversion(tmp_path):
    """A sustained note that is *not* the slide's origin would be detuned."""
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    notes.append((0, 9 * STEP + LONG, 40, 0x5F))     # a bass note, far in pitch
    runs, skipped = _glides(tmp_path, notes)
    assert runs == []
    assert len(skipped) == 1 and skipped[0].reason == "polyphonic"


# --------------------------------------------------------------------------
# Backup-path exclusion
# --------------------------------------------------------------------------


def test_backup_directories_are_excluded():
    assert slides.is_backup_path(Path("set/backup/Song-REV.RPP"))
    assert slides.is_backup_path(Path("set/Backups/Song-REV.RPP"))
    assert not slides.is_backup_path(Path("set/Song-REV.RPP"))
    # Only directory components count, never the filename itself.
    assert not slides.is_backup_path(Path("set/backup.RPP"))


# --------------------------------------------------------------------------
# Conversion
# --------------------------------------------------------------------------


def test_conversion_replaces_ramp_with_one_note_and_preserves_timing(tmp_path):
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    notes.append((3000, 3000 + LONG, 55, 0x5F))       # an unrelated later note
    path = _write(tmp_path, notes)
    before = read_project(path)

    result = slides.convert_project(path)
    assert result.error is None and result.converted == 1

    after = read_project(path)
    assert after.last_tick == before.last_tick
    # 10 ramp notes collapse to 1; the unrelated note survives untouched.
    assert len(after.parts[0].notes) == len(before.parts[0].notes) - 9
    assert any(n.pitch == 55 for n in after.parts[0].notes)
    anchored = [n for n in after.parts[0].notes if n.pitch == 69]
    assert len(anchored) == 1
    assert anchored[0].start == 0          # note now starts where the ramp did


def test_conversion_writes_bend_events_and_recentres(tmp_path):
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    path = _write(tmp_path, notes)
    slides.convert_project(path)

    raw = [slides.EVENT.match(l) for l in path.read_text().split("\n")]
    parsed = [(int(m.group("status"), 16), int(m.group("d1"), 16),
               int(m.group("d2"), 16)) for m in raw if m]
    bends = [(d1, d2) for status, d1, d2 in parsed if status & 0xF0 == 0xE0]
    assert bends, "expected pitch-bend events"
    # Starts fully bent down 9 semitones, ends recentred.
    assert bends[0] != (0x00, 0x40)
    assert bends[-1] == (0x00, 0x40), "must recentre or later notes stay detuned"
    # RPN 0 declares the bend range.
    ccs = [(d1, d2) for status, d1, d2 in parsed if status & 0xF0 == 0xB0]
    assert (0x65, 0x00) in ccs and (0x64, 0x00) in ccs
    assert any(d1 == 0x06 for d1, _ in ccs)


def test_bend_stays_within_the_declared_range(tmp_path):
    """A gesture needing more bend than declared aborts instead of clipping."""
    notes = _ramp(0, 40, 20)                           # 19 semitones
    notes.append((20 * STEP, 20 * STEP + LONG, 59, 0x5F))
    path = _write(tmp_path, notes)
    original = path.read_text()
    result = slides.convert_project(path, bend_range=12)
    assert result.error is not None and "semitones of bend" in result.error
    assert path.read_text() == original, "must not write when it cannot represent it"


def test_backup_is_written_once_and_never_clobbered(tmp_path):
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    path = _write(tmp_path, notes)
    original = path.read_text()

    slides.convert_project(path)
    backup = slides.backup_path(path)
    assert backup.exists() and backup.read_text() == original

    # A second run finds nothing to convert and must leave the backup alone.
    slides.convert_project(path)
    assert backup.read_text() == original


def test_dry_run_reports_without_writing(tmp_path):
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    path = _write(tmp_path, notes)
    original = path.read_text()
    result = slides.convert_project(path, dry_run=True)
    assert result.converted == 1
    assert path.read_text() == original
    assert not slides.backup_path(path).exists()


def test_converted_project_is_clean_on_a_second_pass(tmp_path):
    """Conversion is idempotent: the result contains no detectable gesture."""
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    path = _write(tmp_path, notes)
    slides.convert_project(path)
    project = read_project(path)
    ordered = sorted(project.parts[0].notes, key=lambda n: (n.start, n.pitch))
    runs, skipped = slides.find_glides(ordered, project.seconds_at, project.ppq)
    assert runs == [] and skipped == []


def test_crlf_projects_keep_their_line_endings_and_backup_is_byte_identical(tmp_path):
    """A minority of real REAPER projects are CRLF.

    Text-mode I/O silently translates CRLF to LF, which made the backup differ
    from the original and rewrote every line ending in the converted file --
    a whole-file diff for a ten-note change. Caught on two of fifty projects.
    """
    notes = _ramp(0, 60, 9)
    notes.append((9 * STEP, 9 * STEP + LONG, 69, 0x5F))
    path = tmp_path / "project.RPP"
    original = _project(notes).replace("\n", "\r\n").encode("utf-8")
    path.write_bytes(original)

    result = slides.convert_project(path)
    assert result.error is None and result.converted == 1

    assert slides.backup_path(path).read_bytes() == original
    converted = path.read_bytes()
    assert b"\r\n" in converted, "converted file lost its CRLF line endings"
    assert b"\n" not in converted.replace(b"\r\n", b""), "mixed line endings"


def test_hold_slide_hold_keeps_both_real_notes(tmp_path):
    """A power chord slid between two held positions.

    Both held chords are real notes and must keep their own pitches: only the
    micro-note ramp between them is removed. Merging all three into one note at
    the destination pitch would relabel the first held chord -- the same
    label/audio mismatch the conversion exists to remove.
    """
    origin = [(0, 200, 55, 0x5F), (2, 202, 62, 0x5F)]
    ramp = (_ramp(200, 54, 4, ascending=False)
            + _ramp(200, 61, 4, ascending=False, offset=2))
    dest_start = 200 + 4 * STEP
    destination = [(dest_start, dest_start + 200, 50, 0x5F),
                   (dest_start + 2, dest_start + 202, 57, 0x5F)]
    runs, skipped = _glides(tmp_path, origin + ramp + destination)
    assert skipped == []
    assert len(runs) == 1
    run = runs[0]
    assert run.shape == "anchored" and run.voices == 2
    assert run.pitches == (55, 62)        # the ORIGIN, not the destination
    assert (run.bend_from, run.bend_to) == (0, -5)
    assert run.note_start == 0            # keeps the origin's own onset
    assert run.note_end == max(n[1] for n in ramp)
    # The destination notes are not part of what this gesture replaces.
    assert run.note_count == len(origin) + len(ramp)


def test_hold_slide_hold_leaves_the_destination_untouched(tmp_path):
    """After conversion the destination chord is still its own pair of notes."""
    origin = [(0, 200, 55, 0x5F), (2, 202, 62, 0x5F)]
    ramp = (_ramp(200, 54, 4, ascending=False)
            + _ramp(200, 61, 4, ascending=False, offset=2))
    dest_start = 200 + 4 * STEP
    destination = [(dest_start, dest_start + 200, 50, 0x5F),
                   (dest_start + 2, dest_start + 202, 57, 0x5F)]
    path = _write(tmp_path, origin + ramp + destination)
    assert slides.convert_project(path).error is None

    after = sorted(read_project(path).parts[0].notes, key=lambda n: (n.start, n.pitch))
    assert [n.pitch for n in after] == [55, 62, 50, 57]
    kept = [n for n in after if n.pitch in (50, 57)]
    assert [n.start for n in kept] == [dest_start, dest_start + 2]
    assert [n.end for n in kept] == [dest_start + 200, dest_start + 202]


def test_a_slide_up_then_back_down_is_two_gestures_not_one(tmp_path):
    """Iron Man's 53->60->55 figure.

    A chain that ignores direction swallows both slides into one run whose
    *net* span is near zero, so both are rejected for being too small and the
    micro-notes survive untouched. A chain must keep one direction.
    """
    up = _ramp(0, 53, 8)                       # 53 -> 60
    down = _ramp(8 * STEP, 59, 8, ascending=False)   # 59 -> 52
    runs, _ = _glides(tmp_path, up + down)
    assert len(runs) == 2, f"expected two gestures, got {len(runs)}"
    assert runs[0].bend_to > 0 and runs[1].bend_to < 0
    assert runs[0].pitches == (53,)
    assert runs[1].pitches == (59,)


def test_ramp_landing_on_a_fuller_chord_still_finds_its_destination(tmp_path):
    """One string sliding into a two-note chord -- ordinary guitar writing.

    The chain requires equal voice counts step to step, so a ramp that lands on
    a fuller chord breaks one step early and loses the semitones that make it a
    slide: 52->47 measures as 52->48 and falls under the threshold. The landing
    is matched per-pitch instead, the way anchors already are.
    """
    anchor = [(0, 300, 52, 0x5F)]                    # rings through the ramp
    ramp = _ramp(200, 51, 4, ascending=False)        # 51 -> 48
    landing = [(280, 680, 47, 0x5F), (282, 682, 54, 0x5F)]
    runs, skipped = _glides(tmp_path, anchor + ramp + landing)
    assert skipped == []
    assert len(runs) == 1
    run = runs[0]
    assert run.shape == "anchored" and run.pitches == (52,)
    assert (run.bend_from, run.bend_to) == (0, -5)   # the full gesture, not 52->48
    assert run.note_count == 5                       # anchor + 4 ramp notes


def test_a_four_step_ramp_spanning_three_semitones_is_still_a_slide(tmp_path):
    """The smallest a real ramp can be: 4 steps of one semitone spans 3.

    The span floor was 5 while chains ignored direction and a wobble could
    chain into a long run; monotonic chains exclude wobbles on their own now,
    and a 3-semitone slide with a sustained origin is ordinary guitar writing.
    """
    anchor = [(0, 300, 69, 0x5F)]
    ramp = _ramp(200, 68, 4, ascending=False)     # 68 -> 65
    runs, _ = _glides(tmp_path, anchor + ramp)
    assert len(runs) == 1
    assert runs[0].pitches == (69,)
    assert (runs[0].bend_from, runs[0].bend_to) == (0, -4)
