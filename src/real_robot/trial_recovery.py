from __future__ import annotations

from collections.abc import Sequence

from .aubo_adapter import AuboSession


def restore_home_and_close_gripper(
    session: AuboSession,
    home_pose: Sequence[float],
    *,
    execute_hardware_action: bool,
) -> None:
    """Return to the saved home pose and leave DO01 high/closed at trial end."""

    session.move_linear_pose(
        home_pose,
        execute_hardware_action=execute_hardware_action,
    )
    session.set_gripper(
        "close",
        execute_hardware_action=execute_hardware_action,
    )
