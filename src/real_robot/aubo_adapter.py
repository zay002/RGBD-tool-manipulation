from __future__ import annotations

from collections.abc import Callable
from collections.abc import Sequence
from typing import Any

from .contracts import ClientConfig, gripper_output_value, require_hardware_action


def _default_client_factory() -> Any:
    import pyaubo_sdk

    return pyaubo_sdk.RpcClient()


class AuboSession:
    """Small wrapper around the AUBO RPC SDK with safety gates for IO changes."""

    def __init__(
        self,
        config: ClientConfig,
        client_factory: Callable[[], Any] | None = None,
        request_timeout_ms: int = 1000,
    ) -> None:
        self.config = config
        self.client_factory = client_factory or _default_client_factory
        self.request_timeout_ms = request_timeout_ms
        self.client: Any | None = None
        self.robot: Any | None = None

    def connect(self) -> Any:
        client = self.client_factory()
        client.setRequestTimeout(self.request_timeout_ms)
        rc = client.connect(self.config.aubo_robot_ip, self.config.aubo_rpc_port)
        if rc != 0:
            raise RuntimeError(f"AUBO RPC connect failed with code {rc}")

        names = client.getRobotNames()
        if not names and hasattr(client, "login"):
            if not self.config.aubo_rpc_password:
                raise RuntimeError(
                    "AUBO RPC returned no robot names before login. "
                    "Set AUBO_RPC_PASSWORD in the environment and retry."
                )
            login_rc = client.login(
                self.config.aubo_rpc_user,
                self.config.aubo_rpc_password,
            )
            if login_rc != 0:
                raise RuntimeError(f"AUBO RPC login failed with code {login_rc}")
            names = client.getRobotNames()
        if not names:
            raise RuntimeError("AUBO RPC connected but returned no robot names")

        self.client = client
        self.robot = client.getRobotInterface(names[0])
        return self.robot

    def startup(self, execute_hardware_action: bool) -> int:
        require_hardware_action(execute_hardware_action, "startup AUBO robot")
        robot = self._require_robot()
        return int(robot.getRobotManage().startup())

    def tcp_pose(self) -> list[float]:
        robot = self._require_robot()
        pose = robot.getRobotState().getTcpPose()
        return [float(value) for value in pose]

    def set_speed_fraction(self, value: float, execute_hardware_action: bool) -> int:
        require_hardware_action(
            execute_hardware_action,
            f"set AUBO speed fraction to {value}",
        )
        robot = self._require_robot()
        return int(robot.getMotionControl().setSpeedFraction(float(value)))

    def move_linear_pose(
        self,
        pose: Sequence[float],
        execute_hardware_action: bool,
        acceleration: float = 0.2,
        velocity: float = 0.2,
        blend_radius: float = 0.0,
        duration: float = 0.0,
    ) -> int:
        require_hardware_action(execute_hardware_action, "move AUBO TCP linearly")
        pose_values = [float(value) for value in pose]
        if len(pose_values) != 6:
            raise ValueError("AUBO TCP pose must contain 6 values: x y z rx ry rz")
        robot = self._require_robot()
        motion = robot.getMotionControl()
        if not hasattr(motion, "moveLine"):
            raise RuntimeError("AUBO MotionControl does not expose moveLine")
        return int(
            motion.moveLine(
                pose_values,
                float(acceleration),
                float(velocity),
                float(blend_radius),
                float(duration),
            )
        )

    def set_gripper(self, action: str, execute_hardware_action: bool) -> int:
        require_hardware_action(execute_hardware_action, f"{action} gripper via DO")
        robot = self._require_robot()
        value = gripper_output_value(self.config, action)
        io = robot.getIoControl()
        return int(io.setStandardDigitalOutput(self.config.gripper_do_index, value))

    def _require_robot(self) -> Any:
        if self.robot is None:
            return self.connect()
        return self.robot
