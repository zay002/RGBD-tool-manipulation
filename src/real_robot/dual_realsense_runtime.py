from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .camera_options import apply_color_options
from .dual_camera import DualFrame, resolve_camera_roles
from .realsense_capture import CameraIntrinsics


@dataclass
class RunningCamera:
    role: str
    serial: str
    pipeline: object
    profile: object
    depth_enabled: bool


def list_realsense_devices() -> list[dict[str, str]]:
    import pyrealsense2 as rs

    devices = []
    for dev in rs.context().query_devices():
        record = {}
        for key in (
            rs.camera_info.name,
            rs.camera_info.serial_number,
            rs.camera_info.firmware_version,
            rs.camera_info.usb_type_descriptor,
        ):
            try:
                record[str(key).split(".")[-1]] = dev.get_info(key)
            except Exception:
                pass
        devices.append(record)
    return devices


def start_camera(
    role: str,
    serial: str,
    width: int,
    height: int,
    fps: int,
    depth_enabled: bool,
    color_exposure: float | None = None,
    color_gain: float | None = None,
    color_brightness: float | None = None,
    color_auto_exposure: bool | None = None,
    auto_exposure_priority: bool | None = None,
) -> RunningCamera:
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    if depth_enabled:
        cfg.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    profile = pipeline.start(cfg)
    for sensor in profile.get_device().query_sensors():
        try:
            sensor_name = sensor.get_info(rs.camera_info.name)
        except Exception:
            sensor_name = ""
        if "RGB" in sensor_name:
            apply_color_options(
                sensor,
                rs.option,
                exposure=color_exposure,
                gain=color_gain,
                brightness=color_brightness,
                auto_exposure=color_auto_exposure,
                auto_exposure_priority=auto_exposure_priority,
            )
    return RunningCamera(
        role=role,
        serial=serial,
        pipeline=pipeline,
        profile=profile,
        depth_enabled=depth_enabled,
    )


def frame_from_camera(camera: RunningCamera, warmup_frames: int = 30) -> DualFrame:
    if warmup_frames < 1:
        raise ValueError("warmup_frames must be at least 1")
    frames = None
    for _ in range(warmup_frames):
        frames = camera.pipeline.wait_for_frames(5000)
    if frames is None:
        raise RuntimeError(f"{camera.role} camera returned no frames")
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not color_frame:
        raise RuntimeError(f"{camera.role} camera did not return color")
    if camera.depth_enabled and not depth_frame:
        raise RuntimeError(f"{camera.role} camera did not return depth")
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    intrinsics = CameraIntrinsics(
        width=int(intr.width),
        height=int(intr.height),
        fx=float(intr.fx),
        fy=float(intr.fy),
        cx=float(intr.ppx),
        cy=float(intr.ppy),
        depth_scale_m=float(camera.profile.get_device().first_depth_sensor().get_depth_scale()),
    )
    return DualFrame(
        serial=camera.serial,
        color_bgr=np.asanyarray(color_frame.get_data()).copy(),
        depth_uint16=np.asanyarray(depth_frame.get_data()).copy() if depth_frame else None,
        intrinsics=intrinsics,
    )


def capture_dual_realsense_once(
    wrist_serial: str,
    wide_serial: str,
    width: int = 640,
    height: int = 480,
    fps: int = 15,
    wide_depth: bool = False,
    color_exposure: float | None = None,
    color_gain: float | None = None,
    color_brightness: float | None = None,
    wrist_color_exposure: float | None = None,
    wide_color_exposure: float | None = None,
    wrist_color_gain: float | None = None,
    wide_color_gain: float | None = None,
    wrist_color_auto: bool = False,
    wide_color_auto: bool = False,
    auto_exposure_priority: bool = False,
    warmup_frames: int = 30,
) -> dict[str, DualFrame]:
    devices = list_realsense_devices()
    resolve_camera_roles(devices, wrist_serial=wrist_serial, wide_serial=wide_serial)

    wrist_exposure = wrist_color_exposure if wrist_color_exposure is not None else color_exposure
    wide_exposure = wide_color_exposure if wide_color_exposure is not None else color_exposure
    wrist_gain = wrist_color_gain if wrist_color_gain is not None else color_gain
    wide_gain = wide_color_gain if wide_color_gain is not None else color_gain
    wrist_auto = True if wrist_color_auto else (False if wrist_exposure is not None else None)
    wide_auto = True if wide_color_auto else (False if wide_exposure is not None else None)

    cameras = [
        start_camera(
            "wrist",
            wrist_serial,
            width,
            height,
            fps,
            True,
            wrist_exposure,
            wrist_gain,
            color_brightness,
            wrist_auto,
            auto_exposure_priority if wrist_exposure is None else False,
        ),
        start_camera(
            "wide",
            wide_serial,
            width,
            height,
            fps,
            wide_depth,
            wide_exposure,
            wide_gain,
            color_brightness,
            wide_auto,
            auto_exposure_priority if wide_exposure is None else False,
        ),
    ]
    try:
        return {
            camera.role: frame_from_camera(camera, warmup_frames=warmup_frames)
            for camera in cameras
        }
    finally:
        for camera in cameras:
            camera.pipeline.stop()
