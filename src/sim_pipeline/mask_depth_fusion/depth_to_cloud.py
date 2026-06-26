from __future__ import annotations

import numpy as np


def depth_to_point_cloud(
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics,
    min_depth_m: float = 0.05,
    max_depth_m: float = 5.0,
) -> np.ndarray:
    """Convert a masked depth image to an Nx3 point cloud in camera coordinates."""

    if depth_m.ndim != 2:
        raise ValueError(f"depth_m must be HxW, got shape {depth_m.shape}")
    if mask.shape != depth_m.shape:
        raise ValueError(f"mask shape {mask.shape} must match depth shape {depth_m.shape}")

    valid = mask.astype(bool) & np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    v, u = np.nonzero(valid)
    z = depth_m[v, u].astype(np.float32)
    x = (u.astype(np.float32) - float(intrinsics.cx)) * z / float(intrinsics.fx)
    y = (v.astype(np.float32) - float(intrinsics.cy)) * z / float(intrinsics.fy)
    return np.column_stack((x, y, z)).astype(np.float32)

