"""Preview — the heart of the feature (design §6.3).

A dry-run that shows the complete outcome of a Plan without touching disk:
before/after folder trees, per-file `from→to` diffs, the tags that would be
added, conflicts flagged inline, and live counts. Pure read.
"""

from __future__ import annotations

import os
import sqlite3
from collections import Counter, defaultdict
from pathlib import PurePosixPath
from typing import Any

from find_and_seek.organize import plan_store


def _tree(paths: list[str]) -> dict[str, Any]:
    """Nest a flat list of file paths into {folder: {...}, '_files': [names]}."""
    root: dict[str, Any] = {}
    for p in paths:
        parts = PurePosixPath(p).parts
        node = root
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node.setdefault("_files", []).append(parts[-1])
    return root


def _detect_conflicts(
    conn: sqlite3.Connection, actions: list[dict[str, Any]]
) -> dict[int, str]:
    """Map action_id → conflict reason. Read-only precondition checks."""
    conflicts: dict[int, str] = {}

    # Destination collisions: two accepted-or-pending moves landing on one path,
    # or a destination that already exists on disk and isn't the source itself.
    dest_counts: dict[str, int] = defaultdict(int)
    for a in actions:
        to_path = a["payload"].get("to_path")
        if to_path and a["action_type"] in ("move", "rename"):
            dest_counts[to_path] += 1

    for a in actions:
        pl = a["payload"]
        aid = a["id"]
        from_path = pl.get("from_path")
        to_path = pl.get("to_path")

        # Source moved/vanished since indexing → never act on stale state.
        if from_path and not os.path.exists(from_path):
            conflicts[aid] = "source missing since indexing — will be skipped"
            continue
        if a["action_type"] in ("move", "rename") and to_path:
            if dest_counts[to_path] > 1:
                conflicts[aid] = "name collision: multiple files target this path"
            elif from_path != to_path and os.path.exists(to_path):
                conflicts[aid] = "destination already exists — would need a suffix"
    return conflicts


def build_preview(
    conn: sqlite3.Connection,
    plan_id: int,
    diff_limit: int | None = None,
    include_trees: bool = True,
) -> dict[str, Any] | None:
    """Full dry-run view for a stored Plan, or None if the plan is unknown.

    `diff_limit` caps the per-action diff list (counts/conflicts stay full-scan)
    and `include_trees=False` drops the before/after trees — both keep the API
    payload small for the menu-bar UI, which only renders a sample + the totals.
    """
    plan = plan_store.get_plan(conn, plan_id)
    if not plan:
        return None
    actions = plan_store.get_actions(conn, plan_id)
    conflicts = _detect_conflicts(conn, actions)

    before_paths: list[str] = []
    after_paths: list[str] = []
    diffs: list[dict[str, Any]] = []

    for a in actions:
        pl = a["payload"]
        atype = a["action_type"]
        if atype == "create_dir":
            continue
        from_path = pl.get("from_path")
        if from_path:
            before_paths.append(from_path)
        if atype in ("move", "rename"):
            after_paths.append(pl.get("to_path", from_path))
        elif atype == "quarantine_duplicate":
            after_paths.append(f"~/.findandseek/quarantine/plan-{plan_id}/{PurePosixPath(from_path).name}")
        # For duplicates, resolve the *kept* copy's path so the UI can show
        # "duplicate of …" and let the user compare the two.
        dup_of_path = None
        if atype == "quarantine_duplicate" and pl.get("dup_of_file_id"):
            row = conn.execute(
                "SELECT path FROM files WHERE id=?", (pl["dup_of_file_id"],)
            ).fetchone()
            dup_of_path = row["path"] if row else None
        diffs.append({
            "action_id": a["id"],
            "action_type": atype,
            "file_id": a["file_id"],
            "from_path": from_path,
            "to_path": pl.get("to_path"),
            "dup_of_path": dup_of_path,
            "renamed": pl.get("renamed", False),
            "reason": pl.get("reason"),
            "tags": pl.get("tags", []),
            "decision": a["decision"],
            "conflict": conflicts.get(a["id"]),
        })

    # The resulting structure: true per-destination counts across ALL moves (not
    # just the sampled diffs) so the UI can show the proposed folder tree.
    home = os.path.expanduser("~")
    folder_counts: Counter[str] = Counter()
    for a in actions:
        if a["action_type"] in ("move", "rename"):
            to = a["payload"].get("to_path")
            if to:
                folder_counts[str(PurePosixPath(to).parent)] += 1
    folder_summary = [
        {"folder": f.replace(home, "~", 1) if f.startswith(home) else f, "count": c}
        for f, c in folder_counts.most_common()
    ]

    counts = plan["summary"]["counts"] if plan.get("summary") else {}
    total_diffs = len(diffs)
    if diff_limit is not None:
        diffs = diffs[:diff_limit]
    return {
        "plan_id": plan_id,
        "status": plan["status"],
        "strategy": plan["strategy"],
        "generated_at": plan.get("created_at"),
        "files_in_scope": plan["summary"].get("files_in_scope") if plan.get("summary") else None,
        "summary_line": _summary_line(counts, len(conflicts)),
        "counts": {**counts, "conflicts": len(conflicts)},
        "folder_summary": folder_summary,
        "diff_count": total_diffs,
        "before_tree": _tree(before_paths) if include_trees else None,
        "after_tree": _tree(after_paths) if include_trees else None,
        "diffs": diffs,
    }


def _summary_line(counts: dict[str, int], conflicts: int) -> str:
    parts = [
        f"{counts.get('move', 0)} moves",
        f"{counts.get('rename', 0)} renames",
        f"{counts.get('tag', 0)} tags",
        f"{counts.get('quarantine_duplicate', 0)} duplicates",
    ]
    if conflicts:
        parts.append(f"{conflicts} conflicts to resolve")
    return " · ".join(parts)
