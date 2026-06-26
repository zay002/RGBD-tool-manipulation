from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping


class HardwareActionError(RuntimeError):
    """Raised when a hardware-changing action is attempted without consent."""


def _parse_bool(value: str | bool | int) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on", "high"}:
        return True
    if normalized in {"0", "false", "no", "n", "off", "low"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def _get_required(env: Mapping[str, str], key: str) -> str:
    value = env.get(key)
    if value is None or value == "":
        raise KeyError(f"Missing required environment variable {key}")
    return value


@dataclass(frozen=True)
class ClientConfig:
    server_user: str
    server_host: str
    server_port: int
    server_root: str
    aubo_robot_ip: str
    aubo_rpc_port: int
    aubo_rtde_port: int
    gripper_do_index: int
    gripper_do_close_value: bool
    gripper_do_open_value: bool
    realsense_serial: str
    client_run_root: Path
    wide_realsense_serial: str = ""
    aubo_rpc_user: str = "aubo"
    aubo_rpc_password: str = ""

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "ClientConfig":
        source = os.environ if env is None else env
        return cls(
            server_user=_get_required(source, "TMR_SERVER_USER"),
            server_host=_get_required(source, "TMR_SERVER_HOST"),
            server_port=int(source.get("TMR_SERVER_PORT", "22")),
            server_root=_get_required(source, "TMR_SERVER_ROOT"),
            aubo_robot_ip=_get_required(source, "AUBO_ROBOT_IP"),
            aubo_rpc_port=int(source.get("AUBO_RPC_PORT", "30005")),
            aubo_rtde_port=int(source.get("AUBO_RTDE_PORT", "9012")),
            gripper_do_index=int(source.get("TMR_GRIPPER_DO_INDEX", "1")),
            gripper_do_close_value=_parse_bool(
                source.get("TMR_GRIPPER_DO_CLOSE_VALUE", "true")
            ),
            gripper_do_open_value=_parse_bool(
                source.get("TMR_GRIPPER_DO_OPEN_VALUE", "false")
            ),
            realsense_serial=source.get(
                "WRIST_REALSENSE_SERIAL",
                source.get("REALSENSE_SERIAL", ""),
            ),
            client_run_root=Path(source.get("TMR_CLIENT_RUN_ROOT", Path.home() / "tool_manipulation_runs")),
            wide_realsense_serial=source.get("WIDE_REALSENSE_SERIAL", ""),
            aubo_rpc_user=source.get("AUBO_RPC_USER", "aubo"),
            aubo_rpc_password=source.get("AUBO_RPC_PASSWORD", ""),
        )


def require_hardware_action(confirmed: bool, action: str) -> None:
    if not confirmed:
        raise HardwareActionError(
            f"Refusing to {action}. Re-run with --execute-hardware-action "
            "after the workspace is clear and the emergency stop is reachable."
        )


def gripper_output_value(config: ClientConfig, action: str) -> bool:
    normalized = action.strip().lower()
    if normalized == "close":
        return config.gripper_do_close_value
    if normalized == "open":
        return config.gripper_do_open_value
    raise ValueError(f"Unknown gripper action {action!r}; expected 'open' or 'close'")


def generate_run_id(scene: int, trial: int, now: datetime | None = None) -> str:
    stamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return f"real_{stamp}_scene{scene:02d}_trial{trial:02d}"


def build_request_metadata(
    run_id: str,
    query: str,
    target: str,
    camera_frame: str = "wrist_camera_color_optical_frame",
    base_frame: str = "aubo_base_link",
    depth_units: str = "millimeter",
    return_artifacts: bool = True,
) -> dict:
    return {
        "run_id": run_id,
        "query": query,
        "target": target,
        "camera_frame": camera_frame,
        "base_frame": base_frame,
        "depth_units": depth_units,
        "return_artifacts": return_artifacts,
    }


def quality_gates_pass(result: Mapping[str, object]) -> bool:
    if result.get("status") != "ok":
        return False
    gates = result.get("quality_gates")
    if not isinstance(gates, Mapping):
        return False
    required = (
        "detection_ok",
        "mask_ok",
        "depth_ok",
        "completion_ok",
        "affordance_ok",
    )
    return all(gates.get(key) is True for key in required)
