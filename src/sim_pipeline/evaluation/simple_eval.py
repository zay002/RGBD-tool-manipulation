from __future__ import annotations

from sim_pipeline.grasp_generation.geometric_parallel_jaw import GraspCandidate


def summarize_grasp_set(candidates: list[GraspCandidate]) -> dict[str, float | int]:
    if not candidates:
        return {"num_candidates": 0, "num_collision_free": 0, "best_score": 0.0}
    return {
        "num_candidates": len(candidates),
        "num_collision_free": sum(1 for item in candidates if item.collision_free),
        "best_score": max(item.score for item in candidates),
    }

