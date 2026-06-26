from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel_size: float = 0.004, max_points: int = 4096) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32)
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, unique_idx = np.unique(keys, axis=0, return_index=True)
    sampled = points[np.sort(unique_idx)]
    if len(sampled) > max_points:
        stride = int(np.ceil(len(sampled) / max_points))
        sampled = sampled[::stride][:max_points]
    return sampled.astype(np.float32)


def complete_by_bbox_symmetry(partial_points: np.ndarray, target_points: int = 4096) -> np.ndarray:
    """A deterministic geometry fallback until learned completion checkpoints are installed.

    It mirrors the visible points across the cloud bounding-box center along camera x/y axes.
    This is not a learned completion model; it is only a pipeline-check substitute.
    """

    if partial_points.ndim != 2 or partial_points.shape[1] != 3:
        raise ValueError(f"partial_points must be Nx3, got shape {partial_points.shape}")
    if len(partial_points) == 0:
        return partial_points.astype(np.float32)

    center = 0.5 * (partial_points.min(axis=0) + partial_points.max(axis=0))
    mirrored_x = partial_points.copy()
    mirrored_x[:, 0] = 2.0 * center[0] - mirrored_x[:, 0]
    mirrored_y = partial_points.copy()
    mirrored_y[:, 1] = 2.0 * center[1] - mirrored_y[:, 1]
    completed = np.concatenate([partial_points, mirrored_x, mirrored_y], axis=0)
    return voxel_downsample(completed, max_points=target_points)
