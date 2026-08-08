"""Fail-closed corpus-finalization preflight and provenance records."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .rppread import Note, Project, ProjectPart

RENDER_PROVENANCE_SCHEMA = "procgen.reaper-render-provenance/v1"
FINALIZATION_PROVENANCE_SCHEMA = "reaper2mt3.finalization-provenance/v1"
REQUIRED_REPOSITORIES = ("procgen", "mt3", "midi2reaper", "reaper2mt3")


@dataclass(frozen=True)
class FinalizationInput:
    record: dict

    @property
    def aliases(self) -> list[dict]:
        value = self.record.get("renderer_tracks", [])
        return value if isinstance(value, list) else []

    def policy_for(self, canonical_name: str) -> dict:
        policy = self.record.get("event_policy", {})
        if not isinstance(policy, dict):
            raise ValueError("finalization sidecar event_policy must be an object")
        parts = policy.get("parts", {})
        if not isinstance(parts, dict):
            raise ValueError("finalization sidecar event_policy.parts must be an object")
        selected = parts.get(canonical_name, policy.get("default", {}))
        if not isinstance(selected, dict):
            raise ValueError(f"event policy for {canonical_name} must be an object")
        return selected


def load_finalization_input(path: Path) -> FinalizationInput:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid render provenance sidecar: {path}") from exc
    if record.get("schema_version") != RENDER_PROVENANCE_SCHEMA:
        raise ValueError("finalization requires procgen.reaper-render-provenance/v1")
    repositories = record.get("repositories")
    if not isinstance(repositories, dict) or any(
        not isinstance(repositories.get(name), str) or not repositories[name]
        for name in REQUIRED_REPOSITORIES
    ):
        raise ValueError("render provenance lacks all four pinned repository revisions")
    template = record.get("template")
    profiles = record.get("vst_profiles")
    if not isinstance(template, dict) or not template.get("id") or not template.get("version"):
        raise ValueError("render provenance lacks template id/version")
    if not isinstance(profiles, list) or any(
        not isinstance(p, dict) or not p.get("id") or not p.get("version") for p in profiles
    ):
        raise ValueError("render provenance lacks VST profile versions")
    return FinalizationInput(record)


def preflight(project: Project, sidecar: FinalizationInput) -> dict:
    """Apply declared source-event policy and collapse only attested aliases.

    This mutates the in-memory project solely after every error condition has
    been checked.  Callers therefore run it before creating MIDI, audio, or
    manifest files.
    """
    if project.invalid_corpus_names:
        raise ValueError("non-terminal canonical corpus names: " + ", ".join(project.invalid_corpus_names))

    planned: list[tuple[ProjectPart, list[Note], dict]] = []
    for part in project.parts:
        policy = sidecar.policy_for(part.canonical_name)
        ranges = policy.get("keyswitch_ranges", [])
        action = policy.get("keyswitch_action", "reject")
        if action not in ("reject", "remove"):
            raise ValueError(f"invalid keyswitch action for {part.canonical_name}: {action}")
        if not isinstance(ranges, list) or any(
            not isinstance(r, list) or len(r) != 2 or not all(isinstance(v, int) for v in r)
            for r in ranges
        ):
            raise ValueError(f"invalid keyswitch ranges for {part.canonical_name}")
        keyswitches = [n for n in part.notes if any(low <= n.pitch <= high for low, high in ranges)]
        if keyswitches and action == "reject":
            raise ValueError(f"keyswitch notes in {part.track_name}: {len(keyswitches)}")
        planned.append((part, [n for n in part.notes if n not in keyswitches], {
            "track_name": part.track_name,
            "canonical_name": part.canonical_name,
            "source_cc_events": part.source_cc_count,
            "source_cc_action": "dropped" if part.source_cc_count else "none",
            "keyswitch_notes": len(keyswitches),
            "keyswitch_action": "removed" if keyswitches else "none",
        }))

    groups: dict[tuple[int | None, bool, bool | None], list[tuple[ProjectPart, list[Note], dict]]] = {}
    for item in planned:
        part = item[0]
        groups.setdefault((part.program, part.is_drum, part.rhythm), []).append(item)
    selected: list[tuple[ProjectPart, list[Note], dict]] = []
    collapsed: list[dict] = []
    for key, group in groups.items():
        if len(group) == 1:
            selected.extend(group)
            continue
        canonical_name = group[0][0].canonical_name
        aliases = [a for a in sidecar.aliases if isinstance(a, dict)
                   and a.get("authoritative_symbolic_part") == canonical_name]
        streams = [notes for _, notes, _ in group]
        if len(aliases) < len(group) or any(stream != streams[0] for stream in streams[1:]):
            raise ValueError(f"unattested duplicate corpus part {key}")
        selected.append(group[0])
        collapsed.append({"canonical_name": canonical_name, "renderer_tracks": aliases})

    project.parts = [part for part, notes, _ in selected]
    for part, notes, _ in selected:
        part.notes = notes
    return {
        "source_event_audit": [audit for _, _, audit in planned],
        "collapsed_renderer_duplicates": collapsed,
    }


def provenance_record(sidecar: FinalizationInput, preflight_report: dict, example_id: str) -> dict:
    return {
        "schema_version": FINALIZATION_PROVENANCE_SCHEMA,
        "version": 1,
        "example_id": example_id,
        "repositories": sidecar.record["repositories"],
        "template": sidecar.record["template"],
        "vst_profiles": sidecar.record["vst_profiles"],
        "renderer_tracks": sidecar.aliases,
        **preflight_report,
    }
