from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from .dual_camera import DualFrame, save_dual_snapshot


_UNSAFE_NODE_CHARS = re.compile(r"[^a-z0-9]+")


def normalize_node_label(label: str) -> str:
    normalized = _UNSAFE_NODE_CHARS.sub("_", label.strip().lower()).strip("_")
    if not normalized:
        raise ValueError("node label must contain at least one letter or digit")
    return normalized


def save_experiment_node_snapshot(
    run_dir: Path,
    run_id: str,
    node: str,
    frames: Mapping[str, DualFrame],
    sequence: int | None = None,
    timestamp: datetime | None = None,
    robot_state: Mapping[str, Any] | None = None,
    notes: str = "",
) -> Path:
    safe_node = normalize_node_label(node)
    if sequence is not None and sequence < 1:
        raise ValueError("sequence must be positive when provided")

    tag = safe_node if sequence is None else f"{sequence:02d}_{safe_node}"
    snapshot_root = run_dir / "evidence" / "node_snapshots"
    snapshot_dir = save_dual_snapshot(snapshot_root, frames, timestamp=timestamp, tag=tag)

    metadata_path = snapshot_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "schema": "tool_manipulation.real_robot.node_snapshot.v1",
            "run_id": run_id,
            "node": safe_node,
            "node_label": node,
            "sequence": sequence,
            "robot_state": dict(robot_state or {}),
            "notes": notes,
        }
    )
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    return snapshot_dir


@dataclass
class ExperimentNodeRecorder:
    run_dir: Path
    run_id: str
    frame_provider: Callable[[], Mapping[str, DualFrame]]
    clock: Callable[[], datetime] = datetime.now
    next_sequence: int = 1

    def capture(
        self,
        node: str,
        robot_state: Mapping[str, Any] | None = None,
        notes: str = "",
    ) -> Path:
        sequence = self.next_sequence
        self.next_sequence += 1
        return save_experiment_node_snapshot(
            run_dir=self.run_dir,
            run_id=self.run_id,
            node=node,
            frames=self.frame_provider(),
            sequence=sequence,
            timestamp=self.clock(),
            robot_state=robot_state,
            notes=notes,
        )
