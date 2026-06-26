#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Contact-GraspNet PyTorch on a depth+segmap npz.")
    parser.add_argument("input_npz")
    parser.add_argument("output_json")
    parser.add_argument("--repo", default="src/external/contact_graspnet_pytorch")
    parser.add_argument("--checkpoint", default="checkpoints/contact_graspnet")
    parser.add_argument("--first-thres", type=float, default=0.03)
    parser.add_argument("--second-thres", type=float, default=0.01)
    parser.add_argument("--filter-thres", type=float, default=0.004)
    return parser


def _quat_xyzw_from_rotation(rotation: np.ndarray) -> list[float]:
    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
            qw = (r[2, 1] - r[1, 2]) / s
            qx = 0.25 * s
            qy = (r[0, 1] + r[1, 0]) / s
            qz = (r[0, 2] + r[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
            qw = (r[0, 2] - r[2, 0]) / s
            qx = (r[0, 1] + r[1, 0]) / s
            qy = 0.25 * s
            qz = (r[1, 2] + r[2, 1]) / s
        else:
            s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
            qw = (r[1, 0] - r[0, 1]) / s
            qx = (r[0, 2] + r[2, 0]) / s
            qy = (r[1, 2] + r[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    return [float(v) for v in quat]


def main() -> int:
    args = build_parser().parse_args()
    repo = Path(args.repo).resolve()
    import sys

    sys.path.insert(0, str(repo))
    from contact_graspnet_pytorch import config_utils
    from contact_graspnet_pytorch.checkpoints import CheckpointIO
    from contact_graspnet_pytorch.contact_grasp_estimator import GraspEstimator
    from contact_graspnet_pytorch.data import load_available_input_data

    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    ckpt = repo / args.checkpoint
    segmap, rgb, depth, cam_k, _pc_full, _pc_colors = load_available_input_data(args.input_npz)
    cfg = config_utils.load_config(str(ckpt), batch_size=1, arg_configs=[])
    cfg["TEST"]["first_thres"] = args.first_thres
    cfg["TEST"]["second_thres"] = args.second_thres
    cfg["TEST"]["filter_thres"] = args.filter_thres

    estimator = GraspEstimator(cfg)
    CheckpointIO(checkpoint_dir=str(ckpt / "checkpoints"), model=estimator.model).load("model.pt")
    estimator.model.eval()
    with torch.no_grad():
        pc_full, pc_segments, _pc_colors = estimator.extract_point_clouds(
            depth,
            cam_k,
            segmap=segmap,
            rgb=rgb,
            z_range=[0.2, 1.2],
        )
        pred, scores, contact_pts, openings = estimator.predict_scene_grasps(
            pc_full,
            pc_segments=pc_segments,
            local_regions=True,
            filter_grasps=True,
            forward_passes=1,
        )

    candidates: list[dict[str, object]] = []
    for key, grasps in pred.items():
        key_scores = np.asarray(scores[key]).reshape(-1)
        key_contacts = np.asarray(contact_pts[key]).reshape(-1, 3)
        key_openings = np.asarray(openings[key]).reshape(-1)
        for index, grasp in enumerate(np.asarray(grasps).reshape(-1, 4, 4)):
            contact = key_contacts[index]
            rotation = grasp[:3, :3]
            approach = rotation[:, 2]
            approach /= max(float(np.linalg.norm(approach)), 1e-8)
            approach_pose = contact - approach * 0.08
            candidates.append(
                {
                    "segment_id": int(key),
                    "score": float(key_scores[index]),
                    "width_m": float(key_openings[index]) if index < len(key_openings) else 0.08,
                    "contact_point_m": [float(v) for v in contact],
                    "rotation_matrix": rotation.tolist(),
                    "closing": [float(v) for v in rotation[:, 0]],
                    "approach": [float(v) for v in approach],
                    "grasp_pose_camera": {
                        "translation_m": [float(v) for v in contact],
                        "quaternion_xyzw": _quat_xyzw_from_rotation(rotation),
                    },
                    "approach_pose_camera": {
                        "translation_m": [float(v) for v in approach_pose],
                        "quaternion_xyzw": _quat_xyzw_from_rotation(rotation),
                    },
                }
            )

    candidates.sort(key=lambda item: float(item["score"]), reverse=True)
    payload = {
        "status": "ok" if candidates else "rejected",
        "reject_reason": None if candidates else "contact_graspnet_no_candidates",
        "num_candidates": len(candidates),
        "best": candidates[0] if candidates else None,
        "candidates": candidates[:20],
        "torch_cuda": bool(torch.cuda.is_available()),
        "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    }
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return 0 if candidates else 3


if __name__ == "__main__":
    raise SystemExit(main())
