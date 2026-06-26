#!/usr/bin/env python3
"""Run a dependency-light simulated RGB-D to grasp-candidate pipeline check."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sim_pipeline.completion_inference.geometric_fallback import complete_by_bbox_symmetry  # noqa: E402
from sim_pipeline.evaluation.simple_eval import summarize_grasp_set  # noqa: E402
from sim_pipeline.grasp_generation.geometric_parallel_jaw import generate_topdown_grasps  # noqa: E402
from sim_pipeline.grasp_scoring.simple_collision import score_grasps  # noqa: E402
from sim_pipeline.partial_cloud_generation.synthetic_partial import generate_synthetic_partial_cloud  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/sim_pipeline/pipeline_check"))
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    scene = generate_synthetic_partial_cloud(width=args.width, height=args.height)
    partial = scene["partial_cloud"]
    completed = complete_by_bbox_symmetry(partial, target_points=4096)
    candidates = generate_topdown_grasps(completed, max_candidates=16)
    scored = score_grasps(candidates, completed)
    summary = summarize_grasp_set(scored)
    result = {
        "image_size": [args.width, args.height],
        "partial_points": int(len(partial)),
        "completed_points": int(len(completed)),
        "grasp_summary": summary,
        "best_grasp": scored[0].to_dict() if scored else None,
    }
    with (args.out_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
