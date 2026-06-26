from __future__ import annotations

import dataclasses

import numpy as np

from sim_pipeline.grasp_generation.geometric_parallel_jaw import GraspCandidate


def score_grasps(
    candidates: list[GraspCandidate],
    object_points: np.ndarray,
    table_z_m: float | None = None,
    max_gripper_width_m: float = 0.22,
) -> list[GraspCandidate]:
    """Score candidates using simple width, clearance, and centroid alignment checks."""

    if len(object_points) == 0:
        return []

    centroid = object_points.mean(axis=0)
    mins = object_points.min(axis=0)
    maxs = object_points.max(axis=0)
    table_z = float(mins[2] - 0.002) if table_z_m is None else float(table_z_m)

    scored: list[GraspCandidate] = []
    for candidate in candidates:
        pos = np.array(candidate.position, dtype=np.float32)
        closing = np.array(candidate.closing, dtype=np.float32)
        width_ok = candidate.width_m <= max_gripper_width_m
        above_table = pos[2] > table_z + 0.01
        centered = 1.0 / (1.0 + 50.0 * float(np.linalg.norm(pos[:2] - centroid[:2])))
        width_score = max(0.0, 1.0 - candidate.width_m / max_gripper_width_m)
        orientation_bonus = 0.5 + 0.5 * abs(float(closing[0]))
        score = float(centered * (0.55 + 0.35 * width_score + 0.10 * orientation_bonus))
        collision_free = bool(width_ok and above_table and pos[2] > maxs[2])
        if not collision_free:
            score *= 0.1
        scored.append(dataclasses.replace(candidate, score=score, collision_free=collision_free))

    return sorted(scored, key=lambda item: item.score, reverse=True)
