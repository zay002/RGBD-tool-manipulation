from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    position: list[float]
    approach: list[float]
    closing: list[float]
    width_m: float
    score: float = 0.0
    collision_free: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def generate_topdown_grasps(points: np.ndarray, max_candidates: int = 16, width_margin_m: float = 0.02) -> list[GraspCandidate]:
    """Generate simple top-down parallel-jaw candidates around the cloud bounding box."""

    if len(points) == 0:
        return []

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = 0.5 * (mins + maxs)
    extents = np.maximum(maxs - mins, 1e-4)
    z_top = maxs[2] + 0.035
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    angles = np.linspace(0.0, np.pi, max_candidates, endpoint=False)
    candidates: list[GraspCandidate] = []
    for angle in angles:
        closing = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        lateral_width = abs(closing[0]) * extents[0] + abs(closing[1]) * extents[1]
        width = float(lateral_width + width_margin_m)
        candidates.append(
            GraspCandidate(
                position=[float(center[0]), float(center[1]), float(z_top)],
                approach=approach.tolist(),
                closing=closing.tolist(),
                width_m=width,
            )
        )
    return candidates

