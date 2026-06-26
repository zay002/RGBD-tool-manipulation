from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


TAG_FAMILY_ALIASES = {
    "tag42standard": "tagStandard41h12",
    "tagstandard41h12": "tagStandard41h12",
    "tagstandard52h13": "tagStandard52h13",
}


def normalize_tag_family(value: str) -> str:
    compact = value.strip()
    return TAG_FAMILY_ALIASES.get(compact.lower(), compact)


def transform_from_rotation_translation(rotation: np.ndarray, translation: Iterable[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    matrix[:3, 3] = np.asarray(list(translation), dtype=np.float64).reshape(3)
    return matrix


def pose_vector_to_transform(
    pose: Iterable[float],
    rotation_format: str = "rotvec",
) -> np.ndarray:
    values = np.asarray(list(pose), dtype=np.float64)
    if values.shape != (6,):
        raise ValueError("robot pose must contain [x, y, z, rx, ry, rz]")
    translation = values[:3]
    rotation_values = values[3:]
    if rotation_format == "rotvec":
        rotation = Rotation.from_rotvec(rotation_values).as_matrix()
    elif rotation_format == "rpy":
        rotation = Rotation.from_euler("xyz", rotation_values).as_matrix()
    else:
        raise ValueError("rotation_format must be 'rotvec' or 'rpy'")
    return transform_from_rotation_translation(rotation, translation)


def invert_transform(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = matrix[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ matrix[:3, 3]
    return inverse


@dataclass(frozen=True)
class TagBoardLayout:
    first_id: int
    rows: int
    cols: int
    tag_size_m: float
    pitch_m: float
    order: str = "row-major"

    @property
    def tag_ids(self) -> list[int]:
        return [self.first_id + idx for idx in range(self.rows * self.cols)]

    def board_T_tag(self, tag_id: int) -> np.ndarray:
        offset = int(tag_id) - self.first_id
        if offset < 0 or offset >= self.rows * self.cols:
            raise KeyError(f"tag id {tag_id} is outside board layout")
        if self.order == "row-major":
            row = offset // self.cols
            col = offset % self.cols
        elif self.order == "col-major":
            col = offset // self.rows
            row = offset % self.rows
        else:
            raise ValueError("order must be 'row-major' or 'col-major'")
        return transform_from_rotation_translation(np.eye(3), [col * self.pitch_m, row * self.pitch_m, 0.0])


def build_tag_board_layout(
    first_id: int,
    rows: int,
    cols: int,
    tag_size_m: float,
    pitch_m: float,
    order: str = "row-major",
) -> TagBoardLayout:
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be positive")
    if tag_size_m <= 0 or pitch_m <= 0:
        raise ValueError("tag_size_m and pitch_m must be positive")
    return TagBoardLayout(
        first_id=int(first_id),
        rows=int(rows),
        cols=int(cols),
        tag_size_m=float(tag_size_m),
        pitch_m=float(pitch_m),
        order=order,
    )


def camera_T_board_from_detection(
    tag_id: int,
    camera_T_detected_tag: np.ndarray,
    layout: TagBoardLayout,
) -> np.ndarray:
    return np.asarray(camera_T_detected_tag, dtype=np.float64).reshape(4, 4) @ invert_transform(layout.board_T_tag(tag_id))


def board_corner_object_points(tag_id: int, layout: TagBoardLayout) -> np.ndarray:
    tag_t_board = layout.board_T_tag(tag_id)
    center_x, center_y = tag_t_board[:2, 3]
    half = layout.tag_size_m / 2.0
    return np.array(
        [
            [center_x - half, center_y + half, 0.0],
            [center_x + half, center_y + half, 0.0],
            [center_x + half, center_y - half, 0.0],
            [center_x - half, center_y - half, 0.0],
        ],
        dtype=np.float64,
    )


def estimate_camera_T_board_from_detections(
    detections: Iterable[Any],
    layout: TagBoardLayout,
    camera_params: Iterable[float],
) -> dict[str, Any]:
    fx, fy, cx, cy = [float(value) for value in camera_params]
    camera_matrix = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    object_points = []
    image_points = []
    tag_ids = []
    allowed_ids = set(layout.tag_ids)
    for detection in detections:
        tag_id = int(detection.tag_id)
        if tag_id not in allowed_ids:
            continue
        corners = np.asarray(detection.corners, dtype=np.float64).reshape(4, 2)
        object_points.append(board_corner_object_points(tag_id, layout))
        image_points.append(corners)
        tag_ids.append(tag_id)
    if not object_points:
        raise ValueError("No detections match the configured tag board layout")

    object_array = np.concatenate(object_points, axis=0)
    image_array = np.concatenate(image_points, axis=0)
    success, rvec, tvec = cv2.solvePnP(
        object_array,
        image_array,
        camera_matrix,
        None,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        raise RuntimeError("cv2.solvePnP failed for AprilTag board detections")
    rotation, _ = cv2.Rodrigues(rvec)
    camera_t_board = transform_from_rotation_translation(rotation, np.asarray(tvec).reshape(3))
    projected, _ = cv2.projectPoints(object_array, rvec, tvec, camera_matrix, None)
    residuals = projected.reshape(-1, 2) - image_array
    reprojection_rmse_px = float(np.sqrt(np.mean(np.sum(residuals**2, axis=1))))
    return {
        "camera_T_board": camera_t_board,
        "tag_ids": tag_ids,
        "point_count": int(object_array.shape[0]),
        "reprojection_rmse_px": reprojection_rmse_px,
    }


def transform_error(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    a_mat = np.asarray(a, dtype=np.float64).reshape(4, 4)
    b_mat = np.asarray(b, dtype=np.float64).reshape(4, 4)
    translation_error = float(np.linalg.norm(a_mat[:3, 3] - b_mat[:3, 3]))
    rotation_delta = a_mat[:3, :3].T @ b_mat[:3, :3]
    rotation_error = float(np.degrees(Rotation.from_matrix(rotation_delta).magnitude()))
    return translation_error, rotation_error


def _sample_matrix(sample: Mapping[str, Any], key: str) -> np.ndarray:
    value = sample.get(key)
    if value is None:
        raise KeyError(f"sample missing {key}")
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{key} must be a 4x4 transform matrix")
    return matrix


def _target_residuals(samples: list[Mapping[str, Any]], tcp_t_camera: np.ndarray) -> dict[str, Any]:
    base_t_tags = []
    for sample in samples:
        base_t_tcp = _sample_matrix(sample, "base_T_tcp")
        camera_t_target = _sample_matrix(sample, "camera_T_target" if "camera_T_target" in sample else "camera_T_tag")
        base_t_tags.append(base_t_tcp @ tcp_t_camera @ camera_t_target)
    translations = np.stack([matrix[:3, 3] for matrix in base_t_tags], axis=0)
    mean_translation = translations.mean(axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)
    ref_rotation = base_t_tags[0][:3, :3]
    rotation_errors = [
        float(np.degrees(Rotation.from_matrix(ref_rotation.T @ matrix[:3, :3]).magnitude()))
        for matrix in base_t_tags
    ]
    return {
        "samples": len(samples),
        "translation_rmse_m": float(np.sqrt(np.mean(translation_errors**2))),
        "translation_max_m": float(np.max(translation_errors)),
        "rotation_max_deg": float(np.max(rotation_errors)),
        "base_T_tag_mean_translation_m": mean_translation.tolist(),
    }


def solve_eye_in_hand(samples: list[Mapping[str, Any]], method: int = cv2.CALIB_HAND_EYE_TSAI) -> dict[str, Any]:
    if len(samples) < 3:
        raise ValueError("At least 3 hand-eye samples are required; 8-15 varied poses are recommended")
    rotations_gripper2base = []
    translations_gripper2base = []
    rotations_target2cam = []
    translations_target2cam = []
    for sample in samples:
        base_t_tcp = _sample_matrix(sample, "base_T_tcp")
        camera_t_tag = _sample_matrix(sample, "camera_T_target" if "camera_T_target" in sample else "camera_T_tag")
        rotations_gripper2base.append(base_t_tcp[:3, :3])
        translations_gripper2base.append(base_t_tcp[:3, 3])
        rotations_target2cam.append(camera_t_tag[:3, :3])
        translations_target2cam.append(camera_t_tag[:3, 3])

    rotation, translation = cv2.calibrateHandEye(
        rotations_gripper2base,
        translations_gripper2base,
        rotations_target2cam,
        translations_target2cam,
        method=method,
    )
    tcp_t_camera = transform_from_rotation_translation(rotation, np.asarray(translation).reshape(3))
    return {
        "schema": "tool_manipulation.handeye.v1",
        "convention": "base_T_camera = base_T_tcp @ tcp_T_camera",
        "tcp_T_camera": tcp_t_camera.tolist(),
        "camera_T_tcp": invert_transform(tcp_t_camera).tolist(),
        "residuals": _target_residuals(samples, tcp_t_camera),
    }


def build_grasp_offset_record(
    translation_m: Iterable[float],
    rotation_rotvec: Iterable[float],
    reference: str,
) -> dict[str, Any]:
    translation = [float(value) for value in translation_m]
    rotvec = [float(value) for value in rotation_rotvec]
    transform = pose_vector_to_transform([*translation, *rotvec], rotation_format="rotvec")
    return {
        "schema": "tool_manipulation.grasp_offset.v1",
        "reference": reference,
        "translation_m": translation,
        "rotation_rotvec": rotvec,
        "tcp_T_grasp": transform.tolist(),
        "note": "The gripper grasp TCP is separate from wrist camera hand-eye calibration.",
    }


@dataclass(frozen=True)
class HandEyeSample:
    sample_id: int
    base_T_tcp: list[list[float]]
    camera_T_tag: list[list[float]]
    robot_pose_tcp: list[float]
    tag_id: int
    tag_family: str
    tag_size_m: float
