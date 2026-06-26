from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from .point_cloud_export import depth_rgb_to_points, make_depth_visualization, write_ascii_ply
from .realsense_capture import CameraIntrinsics


@dataclass(frozen=True)
class DualFrame:
    serial: str
    color_bgr: np.ndarray
    depth_uint16: np.ndarray | None
    intrinsics: CameraIntrinsics


def resolve_camera_roles(
    devices: list[Mapping[str, str]],
    wrist_serial: str,
    wide_serial: str,
) -> dict[str, Mapping[str, str]]:
    if wrist_serial == wide_serial:
        raise ValueError("wrist_serial and wide_serial must be different")
    by_serial = {device.get("serial_number", ""): device for device in devices}
    missing = [
        label
        for label, serial in (("wrist", wrist_serial), ("wide", wide_serial))
        if serial not in by_serial
    ]
    if missing:
        available = ", ".join(sorted(serial for serial in by_serial if serial))
        raise ValueError(f"Missing {', '.join(missing)} camera. Available serials: {available}")
    return {
        "wrist": by_serial[wrist_serial],
        "wide": by_serial[wide_serial],
    }


def _camera_metadata(frame: DualFrame) -> dict[str, Any]:
    return {
        "serial": frame.serial,
        "intrinsics": asdict(frame.intrinsics),
        "depth_available": frame.depth_uint16 is not None,
        "depth_visualization": None,
        "point_cloud": None,
    }


def save_dual_snapshot(
    save_root: Path,
    frames: Mapping[str, DualFrame],
    timestamp: datetime | None = None,
    tag: str = "auto",
) -> Path:
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    snapshot_dir = save_root / f"{stamp}_{tag}"
    snapshot_dir.mkdir(parents=True, exist_ok=False)

    metadata: dict[str, Any] = {
        "timestamp": stamp,
        "tag": tag,
        "cameras": {},
    }
    for role in ("wrist", "wide"):
        frame = frames[role]
        color_path = snapshot_dir / f"{role}_rgb.png"
        depth_path = snapshot_dir / f"{role}_depth_uint16.png"
        depth_vis_path = snapshot_dir / f"{role}_depth_vis.png"
        cloud_path = snapshot_dir / f"{role}_cloud.ply"
        if not cv2.imwrite(str(color_path), frame.color_bgr):
            raise RuntimeError(f"Failed to write {color_path}")
        camera_metadata = _camera_metadata(frame)
        if frame.depth_uint16 is not None:
            if not cv2.imwrite(str(depth_path), frame.depth_uint16):
                raise RuntimeError(f"Failed to write {depth_path}")
            depth_vis = make_depth_visualization(frame.depth_uint16, frame.intrinsics)
            if not cv2.imwrite(str(depth_vis_path), depth_vis):
                raise RuntimeError(f"Failed to write {depth_vis_path}")
            points = depth_rgb_to_points(frame.depth_uint16, frame.color_bgr, frame.intrinsics)
            point_count = write_ascii_ply(cloud_path, points)
            camera_metadata["depth_visualization"] = depth_vis_path.name
            camera_metadata["point_cloud"] = {
                "path": cloud_path.name,
                "format": "ply_ascii_xyzrgb",
                "points": point_count,
            }
        metadata["cameras"][role] = camera_metadata

    (snapshot_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return snapshot_dir


def make_preview_canvas(frames: Mapping[str, DualFrame], auto_enabled: bool, save_root: Path) -> np.ndarray:
    panels = []
    for role in ("wrist", "wide"):
        image = frames[role].color_bgr.copy()
        label = f"{role.upper()} {frames[role].serial}"
        cv2.rectangle(image, (0, 0), (image.shape[1], 34), (0, 0, 0), thickness=-1)
        cv2.putText(image, label, (10, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        panels.append(image)
    height = min(panel.shape[0] for panel in panels)
    resized = [
        cv2.resize(panel, (int(panel.shape[1] * height / panel.shape[0]), height))
        for panel in panels
    ]
    canvas = cv2.hconcat(resized)
    status = f"auto={'ON' if auto_enabled else 'OFF'} | s snapshot | a toggle auto | q quit | {save_root}"
    cv2.rectangle(canvas, (0, canvas.shape[0] - 34), (canvas.shape[1], canvas.shape[0]), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        status[:120],
        (10, canvas.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
    )
    return canvas
