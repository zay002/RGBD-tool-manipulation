from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class CameraIntrinsicsLike(Protocol):
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float


def _valid_depth_mask(
    depth_uint16: np.ndarray,
    intrinsics: CameraIntrinsicsLike,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    max_units = int(max_depth_m / intrinsics.depth_scale_m)
    return (depth_uint16 > 0) & (depth_uint16 < 65535) & (depth_uint16 <= max_units)


def depth_rgb_to_points(
    depth_uint16: np.ndarray,
    color_bgr: np.ndarray,
    intrinsics: CameraIntrinsicsLike,
    max_depth_m: float = 3.0,
) -> np.ndarray:
    if depth_uint16.ndim != 2:
        raise ValueError("depth_uint16 must have shape HxW")
    if color_bgr.ndim != 3 or color_bgr.shape[2] != 3:
        raise ValueError("color_bgr must have shape HxWx3")
    if color_bgr.shape[:2] != depth_uint16.shape:
        raise ValueError("color_bgr and depth_uint16 must have matching height and width")

    valid = _valid_depth_mask(depth_uint16, intrinsics, max_depth_m=max_depth_m)
    if not np.any(valid):
        return np.empty((0, 6), dtype=np.float64)

    ys, xs = np.nonzero(valid)
    z = depth_uint16[ys, xs].astype(np.float64) * intrinsics.depth_scale_m
    x = (xs.astype(np.float64) - intrinsics.cx) * z / intrinsics.fx
    y = (ys.astype(np.float64) - intrinsics.cy) * z / intrinsics.fy
    rgb = color_bgr[ys, xs, ::-1].astype(np.float64)
    return np.column_stack([x, y, z, rgb])


def write_ascii_ply(path: Path, points_xyz_rgb: np.ndarray) -> int:
    if points_xyz_rgb.ndim != 2 or points_xyz_rgb.shape[1] != 6:
        raise ValueError("points_xyz_rgb must have shape Nx6")
    path.parent.mkdir(parents=True, exist_ok=True)
    count = int(points_xyz_rgb.shape[0])
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {count}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for x, y, z, r, g, b in points_xyz_rgb:
            file.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    return count


def make_depth_visualization(
    depth_uint16: np.ndarray,
    intrinsics: CameraIntrinsicsLike,
    min_depth_m: float = 0.25,
    max_depth_m: float = 1.25,
) -> np.ndarray:
    if depth_uint16.ndim != 2:
        raise ValueError("depth_uint16 must have shape HxW")
    depth_m = depth_uint16.astype(np.float32) * float(intrinsics.depth_scale_m)
    valid = _valid_depth_mask(depth_uint16, intrinsics, max_depth_m=10.0)
    normalized = np.clip((depth_m - min_depth_m) / (max_depth_m - min_depth_m), 0.0, 1.0)
    gray = (255.0 * (1.0 - normalized)).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = (0, 0, 0)
    color[depth_uint16 == 65535] = (255, 0, 255)
    return color
