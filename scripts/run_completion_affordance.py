#!/usr/bin/env python3
"""Run masked-depth completion and affordance segmentation.

This script is the public integration point used by ``run_server_inference.py``.
It supports two modes:

1. If a user supplies an importable learned backend through
   ``COMPLETION_AFFORDANCE_BACKEND`` or ``--backend-module``, the backend is
   called with the normalized partial cloud and may return learned completion
   and affordance labels.
2. Otherwise, a deterministic geometric completion and handle-like affordance
   heuristic is used. The fallback is intended for environment checks and for
   validating the request/result contract; it should not be reported as a
   learned model.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

import numpy as np


def read_pfm(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        header = f.readline().decode("ascii").strip()
        if header not in {"Pf", "PF"}:
            raise ValueError(f"{path} is not a PFM file")
        dims = f.readline().decode("ascii").strip()
        while dims.startswith("#"):
            dims = f.readline().decode("ascii").strip()
        width, height = [int(v) for v in dims.split()]
        scale = float(f.readline().decode("ascii").strip())
        dtype = "<f4" if scale < 0 else ">f4"
        channels = 1 if header == "Pf" else 3
        data = np.frombuffer(f.read(), dtype=dtype)
    if channels == 3:
        return data.reshape((height, width, channels)).astype(np.float32)
    return data.reshape((height, width)).astype(np.float32)


def read_mask(path: Path) -> np.ndarray:
    from PIL import Image

    return np.array(Image.open(path).convert("L")) > 127


def depth_to_cloud(depth_m: np.ndarray, mask: np.ndarray, hfov: float, min_depth: float, max_depth: float) -> np.ndarray:
    height, width = depth_m.shape
    fx = width / (2.0 * np.tan(hfov * 0.5))
    fy = fx
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    valid = mask & np.isfinite(depth_m) & (depth_m >= min_depth) & (depth_m <= max_depth)
    v, u = np.nonzero(valid)
    z = depth_m[v, u].astype(np.float32)
    x = (u.astype(np.float32) - cx) * z / fx
    y = (v.astype(np.float32) - cy) * z / fy
    return np.column_stack([x, y, z]).astype(np.float32)


def sample_points(points: np.ndarray, n_points: int, seed: int = 2026) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError("Cannot sample from an empty point cloud")
    rng = np.random.default_rng(seed)
    replace = points.shape[0] < n_points
    idx = rng.choice(points.shape[0], size=n_points, replace=replace)
    return points[idx].astype(np.float32)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    center = points.mean(axis=0).astype(np.float32)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale < 1e-6:
        scale = 1.0
    return (centered / scale).astype(np.float32), center, scale


def voxel_downsample(points: np.ndarray, max_points: int) -> np.ndarray:
    if len(points) <= max_points:
        return points.astype(np.float32)
    idx = np.linspace(0, len(points) - 1, max_points).astype(np.int64)
    return points[idx].astype(np.float32)


def geometric_completion(partial_norm: np.ndarray, target_points: int) -> np.ndarray:
    center = 0.5 * (partial_norm.min(axis=0) + partial_norm.max(axis=0))
    mirror_x = partial_norm.copy()
    mirror_x[:, 0] = 2.0 * center[0] - mirror_x[:, 0]
    mirror_y = partial_norm.copy()
    mirror_y[:, 1] = 2.0 * center[1] - mirror_y[:, 1]
    completed = np.concatenate([partial_norm, mirror_x, mirror_y], axis=0)
    order = np.lexsort((completed[:, 2], completed[:, 1], completed[:, 0]))
    return voxel_downsample(completed[order], target_points)


def heuristic_affordance_labels(points_norm: np.ndarray, affordance_label: int) -> np.ndarray:
    """Mark an elongated central band as a handle-like affordance region."""

    labels = np.zeros((len(points_norm),), dtype=np.int64)
    if len(points_norm) == 0:
        return labels
    centered = points_norm - points_norm.mean(axis=0, keepdims=True)
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    main_axis = eigvecs[:, int(np.argmax(eigvals))]
    coord = centered @ main_axis
    radial = np.linalg.norm(centered - np.outer(coord, main_axis), axis=1)
    coord_lo, coord_hi = np.quantile(coord, [0.20, 0.80])
    radial_cut = np.quantile(radial, 0.55)
    handle = (coord >= coord_lo) & (coord <= coord_hi) & (radial <= radial_cut)
    if handle.sum() < max(16, int(0.05 * len(points_norm))):
        handle = radial <= np.quantile(radial, 0.35)
    labels[handle] = int(affordance_label)
    return labels


def propagate_labels(src_points: np.ndarray, src_labels: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree

        _, idx = cKDTree(src_points).query(query_points, k=1, workers=-1)
        return src_labels[idx].astype(np.int64)
    except Exception:
        labels = []
        for start in range(0, len(query_points), 2048):
            chunk = query_points[start : start + 2048]
            dist = ((chunk[:, None, :] - src_points[None, :, :]) ** 2).sum(axis=-1)
            labels.append(src_labels[dist.argmin(axis=1)])
        return np.concatenate(labels).astype(np.int64)


def run_backend(module_name: str, partial_norm: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    module = importlib.import_module(module_name)
    if not hasattr(module, "run_completion_affordance"):
        raise AttributeError(f"{module_name} must define run_completion_affordance(partial_norm, args)")
    payload = module.run_completion_affordance(partial_norm=partial_norm, args=args)
    if not isinstance(payload, dict):
        raise TypeError("backend must return a dict")
    if "completed_normalized" not in payload or "completed_labels" not in payload:
        raise KeyError("backend must return completed_normalized and completed_labels")
    return payload


def write_ply(path: Path, points: np.ndarray, labels: np.ndarray | None = None, affordance_label: int = 1) -> None:
    if labels is None:
        colors = np.tile(np.array([[180, 188, 196]], dtype=np.uint8), (points.shape[0], 1))
    else:
        colors = np.where(
            labels[:, None] == affordance_label,
            np.array([[220, 48, 48]], dtype=np.uint8),
            np.array([[160, 168, 176]], dtype=np.uint8),
        )
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.7f} {point[1]:.7f} {point[2]:.7f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth-pfm", type=Path, required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seedformer-checkpoint", type=Path, default=None)
    parser.add_argument("--dgcnn-checkpoint", type=Path, default=None)
    parser.add_argument("--backend-module", default=None)
    parser.add_argument("--hfov", type=float, default=1.22)
    parser.add_argument("--partial-points", type=int, default=2048)
    parser.add_argument("--completed-points", type=int, default=4096)
    parser.add_argument("--seg-points", type=int, default=2048)
    parser.add_argument("--min-depth", type=float, default=0.05)
    parser.add_argument("--max-depth", type=float, default=2.5)
    parser.add_argument("--affordance-label", type=int, default=1)
    parser.add_argument("--object-category", default=None)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    depth = read_pfm(args.depth_pfm)
    mask = read_mask(args.mask)
    partial_camera_all = depth_to_cloud(depth, mask, args.hfov, args.min_depth, args.max_depth)
    partial_camera = sample_points(partial_camera_all, args.partial_points)
    partial_norm, center, scale = normalize_points(partial_camera)

    backend_module = args.backend_module
    if backend_module is None:
        import os

        backend_module = os.environ.get("COMPLETION_AFFORDANCE_BACKEND")

    backend_name = "geometric_completion_affordance_heuristic"
    backend_payload: dict[str, Any] = {}
    if backend_module:
        backend_payload = run_backend(backend_module, partial_norm, args)
        completed_norm = np.asarray(backend_payload["completed_normalized"], dtype=np.float32)
        completed_labels = np.asarray(backend_payload["completed_labels"], dtype=np.int64)
        backend_name = str(backend_payload.get("backend", backend_module))
    else:
        completed_norm = geometric_completion(partial_norm, args.completed_points)
        completed_labels = heuristic_affordance_labels(completed_norm, args.affordance_label)

    completed_camera = (completed_norm * scale + center).astype(np.float32)
    partial_labels = propagate_labels(completed_camera, completed_labels, partial_camera_all)

    np.save(args.out_dir / "partial_camera_all.npy", partial_camera_all)
    np.save(args.out_dir / "partial_camera_2048.npy", partial_camera)
    np.save(args.out_dir / "partial_normalized_2048.npy", partial_norm)
    np.save(args.out_dir / "completed_normalized.npy", completed_norm)
    np.save(args.out_dir / "completed_camera.npy", completed_camera)
    np.save(args.out_dir / "completed_affordance_labels.npy", completed_labels)
    np.save(args.out_dir / "partial_affordance_labels.npy", partial_labels)
    write_ply(args.out_dir / "partial_camera_all_labeled.ply", partial_camera_all, partial_labels, args.affordance_label)
    write_ply(args.out_dir / "completed_camera_labeled.ply", completed_camera, completed_labels, args.affordance_label)

    completed_affordance = completed_labels == args.affordance_label
    partial_affordance = partial_labels == args.affordance_label
    summary = {
        "backend": backend_name,
        "backend_module": backend_module,
        "learned_backend_supplied": bool(backend_module),
        "seedformer_checkpoint": str(args.seedformer_checkpoint) if args.seedformer_checkpoint else None,
        "dgcnn_checkpoint": str(args.dgcnn_checkpoint) if args.dgcnn_checkpoint else None,
        "object_category": args.object_category,
        "device": args.device,
        "partial_raw_points": int(partial_camera_all.shape[0]),
        "partial_sampled_points": int(partial_norm.shape[0]),
        "completed_points": int(completed_norm.shape[0]),
        "affordance_label": int(args.affordance_label),
        "completed_affordance_points": int(completed_affordance.sum()),
        "completed_affordance_ratio": float(completed_affordance.mean()) if completed_affordance.size else 0.0,
        "partial_affordance_points": int(partial_affordance.sum()),
        "normalization": {"center_camera": center.tolist(), "scale": scale},
        "outputs_are_restored_to_camera_coordinates": True,
        "backend_extra": {k: v for k, v in backend_payload.items() if k not in {"completed_normalized", "completed_labels"}},
    }
    with (args.out_dir / "completion_segmentation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
