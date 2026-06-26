from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_image_size(cls, width: int, height: int, fov_degrees: float = 60.0) -> "CameraIntrinsics":
        focal = 0.5 * width / np.tan(np.deg2rad(fov_degrees) * 0.5)
        return cls(width=width, height=height, fx=float(focal), fy=float(focal), cx=(width - 1) / 2, cy=(height - 1) / 2)


def render_synthetic_box_scene(
    width: int = 320,
    height: int = 240,
    object_depth_m: float = 0.72,
    table_depth_m: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, CameraIntrinsics]:
    """Render a deterministic RGB-D tabletop scene with one segmented box-like target.

    This is intentionally simple and dependency-light. It provides a stable pipeline check for
    the downstream geometry pipeline before heavyweight renderers or detectors are installed.
    """

    intrinsics = CameraIntrinsics.from_image_size(width, height)
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[..., :] = np.array([38, 43, 49], dtype=np.uint8)
    depth = np.full((height, width), table_depth_m, dtype=np.float32)
    mask = np.zeros((height, width), dtype=bool)

    x0, x1 = int(width * 0.42), int(width * 0.58)
    y0, y1 = int(height * 0.38), int(height * 0.62)
    yy, xx = np.mgrid[y0:y1, x0:x1]

    mask[y0:y1, x0:x1] = True
    slope_x = (xx - xx.mean()) / max(1.0, (x1 - x0))
    slope_y = (yy - yy.mean()) / max(1.0, (y1 - y0))
    depth[y0:y1, x0:x1] = object_depth_m + 0.035 * slope_y - 0.015 * slope_x
    rgb[y0:y1, x0:x1] = np.array([210, 184, 92], dtype=np.uint8)

    # Add a small unsegmented distractor on the table.
    dx0, dx1 = int(width * 0.71), int(width * 0.82)
    dy0, dy1 = int(height * 0.50), int(height * 0.62)
    rgb[dy0:dy1, dx0:dx1] = np.array([78, 150, 170], dtype=np.uint8)
    depth[dy0:dy1, dx0:dx1] = table_depth_m - 0.08

    return rgb, depth, mask, intrinsics
