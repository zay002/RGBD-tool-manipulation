from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .point_cloud_export import depth_rgb_to_points, make_depth_visualization, write_ascii_ply


@dataclass(frozen=True)
class CameraIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    depth_scale_m: float


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_request_directory(
    run_dir: Path,
    request_metadata: Mapping[str, Any],
    rgb_bgr: np.ndarray,
    depth_uint16: np.ndarray,
    intrinsics: CameraIntrinsics,
    tf_base_camera: Mapping[str, Any],
) -> Path:
    request_dir = run_dir / "request"
    request_dir.mkdir(parents=True, exist_ok=True)

    if rgb_bgr.ndim != 3 or rgb_bgr.shape[2] != 3:
        raise ValueError("rgb_bgr must have shape HxWx3")
    if depth_uint16.ndim != 2:
        raise ValueError("depth_uint16 must have shape HxW")
    if depth_uint16.dtype != np.uint16:
        raise ValueError("depth_uint16 must use dtype uint16")

    if not cv2.imwrite(str(request_dir / "rgb.png"), rgb_bgr):
        raise RuntimeError("Failed to write rgb.png")
    if not cv2.imwrite(str(request_dir / "depth_uint16.png"), depth_uint16):
        raise RuntimeError("Failed to write depth_uint16.png")
    if not cv2.imwrite(str(request_dir / "depth_vis.png"), make_depth_visualization(depth_uint16, intrinsics)):
        raise RuntimeError("Failed to write depth_vis.png")
    write_ascii_ply(
        request_dir / "cloud_camera.ply",
        depth_rgb_to_points(depth_uint16, rgb_bgr, intrinsics),
    )

    _write_json(request_dir / "request.json", dict(request_metadata))
    _write_json(request_dir / "intrinsics.json", asdict(intrinsics))
    _write_json(request_dir / "tf_base_camera.json", dict(tf_base_camera))
    return request_dir


def capture_realsense_frame(serial: str = "") -> tuple[np.ndarray, np.ndarray, CameraIntrinsics]:
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    if serial:
        cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipe.start(cfg)
    align = rs.align(rs.stream.color)
    try:
        frames = None
        for _ in range(30):
            frames = align.process(pipe.wait_for_frames(5000))
        if frames is None:
            raise RuntimeError("RealSense returned no frames")
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("RealSense did not return both depth and color frames")

        color_profile = color_frame.profile.as_video_stream_profile()
        intr = color_profile.intrinsics
        intrinsics = CameraIntrinsics(
            width=int(intr.width),
            height=int(intr.height),
            fx=float(intr.fx),
            fy=float(intr.fy),
            cx=float(intr.ppx),
            cy=float(intr.ppy),
            depth_scale_m=float(profile.get_device().first_depth_sensor().get_depth_scale()),
        )
        rgb_bgr = np.asanyarray(color_frame.get_data()).copy()
        depth_uint16 = np.asanyarray(depth_frame.get_data()).copy()
        return rgb_bgr, depth_uint16, intrinsics
    finally:
        pipe.stop()


def realsense_pipeline_check(serial: str = "") -> dict[str, Any]:
    rgb_bgr, depth_uint16, intrinsics = capture_realsense_frame(serial)
    center_y = depth_uint16.shape[0] // 2
    center_x = depth_uint16.shape[1] // 2
    return {
        "capture_ok": True,
        "width": intrinsics.width,
        "height": intrinsics.height,
        "depth_scale_m": intrinsics.depth_scale_m,
        "center_depth_m": float(depth_uint16[center_y, center_x]) * intrinsics.depth_scale_m,
        "color_shape": list(rgb_bgr.shape),
        "depth_shape": list(depth_uint16.shape),
    }
