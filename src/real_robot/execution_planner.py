from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RzPlan:
    rz: float
    delta_from_home: float
    target_angle: float


def wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def bounded_rz_from_closing_axis(
    *,
    home_rz: float,
    home_local_x_angle: float,
    target_closing_xy: tuple[float, float] | list[float] | np.ndarray,
    tool_closing_yaw_offset_rad: float = 0.0,
    bidirectional_axis: bool = False,
) -> RzPlan:
    closing = np.asarray(target_closing_xy, dtype=float).reshape(-1)
    if closing.shape[0] < 2:
        raise ValueError("target_closing_xy must contain at least x and y")
    if float(np.linalg.norm(closing[:2])) < 1e-8:
        return RzPlan(rz=float(home_rz), delta_from_home=0.0, target_angle=float(home_local_x_angle))

    target_angle = math.atan2(float(closing[1]), float(closing[0]))
    calibrated_target_angle = target_angle + float(tool_closing_yaw_offset_rad)
    candidates = [calibrated_target_angle]
    if bidirectional_axis:
        candidates.append(calibrated_target_angle + math.pi)
    chosen_target_angle = min(
        candidates,
        key=lambda angle: abs(wrap_to_pi(angle - float(home_local_x_angle))),
    )
    delta = wrap_to_pi(chosen_target_angle - float(home_local_x_angle))
    rz = float(home_rz) + delta
    return RzPlan(rz=wrap_to_pi(rz), delta_from_home=delta, target_angle=chosen_target_angle)


def rotvec_to_matrix(rotvec: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    rv = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(rv))
    if theta < 1e-9:
        return np.eye(3)
    axis = rv / theta
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def rpy_to_matrix(rpy: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    roll, pitch, yaw = np.asarray(rpy, dtype=float).reshape(3)
    cr, sr = math.cos(float(roll)), math.sin(float(roll))
    cp, sp = math.cos(float(pitch)), math.sin(float(pitch))
    cy, sy = math.cos(float(yaw)), math.sin(float(yaw))
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    return rz @ ry @ rx


def tcp_from_grasp_point_and_exec_rotvec(
    grasp_point_base: tuple[float, float, float] | list[float] | np.ndarray,
    exec_rotvec: tuple[float, float, float] | list[float] | np.ndarray,
    tcp_to_grasp_m: tuple[float, float, float] | list[float] | np.ndarray = (0.0, 0.0, 0.2),
) -> np.ndarray:
    point = np.asarray(grasp_point_base, dtype=float).reshape(3)
    offset = np.asarray(tcp_to_grasp_m, dtype=float).reshape(3)
    return point - rotvec_to_matrix(exec_rotvec) @ offset


def tcp_from_grasp_point_and_exec_rpy(
    grasp_point_base: tuple[float, float, float] | list[float] | np.ndarray,
    exec_rpy: tuple[float, float, float] | list[float] | np.ndarray,
    tcp_to_grasp_m: tuple[float, float, float] | list[float] | np.ndarray = (0.0, 0.0, 0.2),
) -> np.ndarray:
    point = np.asarray(grasp_point_base, dtype=float).reshape(3)
    offset = np.asarray(tcp_to_grasp_m, dtype=float).reshape(3)
    return point - rpy_to_matrix(exec_rpy) @ offset
