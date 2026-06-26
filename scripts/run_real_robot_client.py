#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, TextIO

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from real_robot.aubo_adapter import AuboSession
from real_robot.contracts import (
    HardwareActionError,
    ClientConfig,
    build_request_metadata,
    generate_run_id,
)
from real_robot.transfer import build_transfer_commands, run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Client host hub for modular RGB-D tool manipulation experiments."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check-env", help="Parse environment and import required packages.")
    subparsers.add_parser("camera-check", help="Capture one RealSense RGB-D frame.")
    subparsers.add_parser("aubo-probe", help="Connect to AUBO RPC without startup, IO, or motion.")

    startup = subparsers.add_parser("aubo-startup", help="Call RobotManage.startup().")
    startup.add_argument(
        "--execute-hardware-action",
        action="store_true",
        help="Required because startup changes the physical robot enable state.",
    )

    for name, help_text in [
        ("gripper-open", "Set configured DO low/open."),
        ("gripper-close", "Set configured DO high/close."),
    ]:
        cmd = subparsers.add_parser(name, help=help_text)
        cmd.add_argument(
            "--execute-hardware-action",
            action="store_true",
            help="Required for commands that change physical hardware state.",
        )

    capture = subparsers.add_parser("capture-request", help="Capture RGB-D and write a request folder.")
    capture.add_argument("--scene", type=int, required=True)
    capture.add_argument("--trial", type=int, required=True)
    capture.add_argument("--query", default="grasp the wrench handle")
    capture.add_argument("--target", default="wrench")
    capture.add_argument(
        "--tf-base-camera-json",
        required=True,
        help="Path to a JSON file storing T_base_camera at capture time.",
    )

    capture_node = subparsers.add_parser(
        "capture-node",
        help="Capture wrist + wide scene images for a named real-experiment node.",
    )
    capture_node.add_argument("--run-id", required=True)
    capture_node.add_argument("--node", required=True, help="Experiment node label, e.g. before_grasp.")
    capture_node.add_argument("--sequence", type=int, default=None)
    capture_node.add_argument("--notes", default="")
    capture_node.add_argument(
        "--robot-state-json",
        default=None,
        help="Optional JSON file with robot state to embed in node snapshot metadata.",
    )
    capture_node.add_argument("--wrist-serial", default=None)
    capture_node.add_argument("--wide-serial", default=None)
    capture_node.add_argument("--width", type=int, default=640)
    capture_node.add_argument("--height", type=int, default=480)
    capture_node.add_argument("--fps", type=int, default=15)
    capture_node.add_argument(
        "--warmup-frames",
        type=int,
        default=30,
        help="Frames to discard after starting RealSense streams before saving the node snapshot.",
    )
    capture_node.add_argument("--wide-depth", action="store_true")
    capture_node.add_argument("--color-exposure", type=float, default=None)
    capture_node.add_argument("--color-gain", type=float, default=None)
    capture_node.add_argument("--color-brightness", type=float, default=None)
    capture_node.add_argument("--wrist-color-exposure", type=float, default=None)
    capture_node.add_argument("--wide-color-exposure", type=float, default=None)
    capture_node.add_argument("--wrist-color-gain", type=float, default=None)
    capture_node.add_argument("--wide-color-gain", type=float, default=None)
    capture_node.add_argument("--wrist-color-auto", action="store_true")
    capture_node.add_argument("--wide-color-auto", action="store_true")
    capture_node.add_argument("--auto-exposure-priority", action="store_true")

    roundtrip = subparsers.add_parser(
        "server-roundtrip",
        help="Upload request, run server inference, and download result.",
    )
    roundtrip.add_argument("--run-id", required=True)
    roundtrip.add_argument(
        "--execute-transfer",
        action="store_true",
        help="Actually run ssh/scp commands. Without this, print the command plan.",
    )
    return parser


def _load_config(env: Mapping[str, str]) -> ClientConfig:
    return ClientConfig.from_env(env)


def _print_json(stdout: TextIO, value: Mapping[str, Any]) -> None:
    print(json.dumps(value, indent=2, ensure_ascii=False), file=stdout)


def _check_env(config: ClientConfig, stdout: TextIO) -> int:
    imports = {}
    for module_name in ("pyaubo_sdk", "pyrealsense2", "paramiko", "cv2", "numpy", "open3d"):
        try:
            __import__(module_name)
            imports[module_name] = "ok"
        except Exception as exc:  # pragma: no cover - only hit on broken lab envs
            imports[module_name] = f"missing: {exc}"
    _print_json(
        stdout,
        {
            "config_ok": True,
            "server": f"{config.server_user}@{config.server_host}:{config.server_port}",
            "server_root": config.server_root,
            "aubo": f"{config.aubo_robot_ip}:{config.aubo_rpc_port}",
            "aubo_rpc_user": config.aubo_rpc_user,
            "realsense_serial": config.realsense_serial,
            "wide_realsense_serial": config.wide_realsense_serial,
            "gripper_do_index": config.gripper_do_index,
            "gripper_close_value": config.gripper_do_close_value,
            "gripper_open_value": config.gripper_do_open_value,
            "imports": imports,
        },
    )
    return 0


def _aubo_probe(config: ClientConfig, client_factory: Any, stdout: TextIO) -> int:
    session = AuboSession(config, client_factory=client_factory)
    session.connect()
    _print_json(stdout, {"aubo_probe_ok": True, "robot_ip": config.aubo_robot_ip})
    return 0


def _set_gripper(
    config: ClientConfig,
    action: str,
    execute_hardware_action: bool,
    client_factory: Any,
    stdout: TextIO,
) -> int:
    session = AuboSession(config, client_factory=client_factory)
    rc = session.set_gripper(action, execute_hardware_action=execute_hardware_action)
    _print_json(
        stdout,
        {
            "gripper_action": action,
            "do_index": config.gripper_do_index,
            "sdk_return_code": rc,
        },
    )
    return 0


def _aubo_startup(
    config: ClientConfig,
    execute_hardware_action: bool,
    client_factory: Any,
    stdout: TextIO,
) -> int:
    session = AuboSession(config, client_factory=client_factory)
    rc = session.startup(execute_hardware_action=execute_hardware_action)
    _print_json(stdout, {"aubo_startup": "requested", "sdk_return_code": rc})
    return 0


def _capture_request(config: ClientConfig, args: argparse.Namespace, stdout: TextIO) -> int:
    from real_robot.realsense_capture import capture_realsense_frame, write_request_directory

    run_id = generate_run_id(args.scene, args.trial)
    run_dir = config.client_run_root / run_id
    tf_path = Path(args.tf_base_camera_json)
    tf_base_camera = json.loads(tf_path.read_text(encoding="utf-8"))
    rgb_bgr, depth_uint16, intrinsics = capture_realsense_frame(config.realsense_serial)
    metadata = build_request_metadata(
        run_id=run_id,
        query=args.query,
        target=args.target,
    )
    request_dir = write_request_directory(
        run_dir=run_dir,
        request_metadata=metadata,
        rgb_bgr=rgb_bgr,
        depth_uint16=depth_uint16,
        intrinsics=intrinsics,
        tf_base_camera=tf_base_camera,
    )
    _print_json(stdout, {"run_id": run_id, "request_dir": str(request_dir)})
    return 0


def _capture_node(
    config: ClientConfig,
    args: argparse.Namespace,
    stdout: TextIO,
    dual_frame_provider: Any,
) -> int:
    from real_robot.dual_realsense_runtime import capture_dual_realsense_once
    from real_robot.experiment_recorder import save_experiment_node_snapshot

    wrist_serial = args.wrist_serial or config.realsense_serial
    wide_serial = args.wide_serial or config.wide_realsense_serial
    if not wrist_serial:
        raise ValueError("Missing wrist RealSense serial; set REALSENSE_SERIAL or pass --wrist-serial")
    if not wide_serial:
        raise ValueError("Missing wide RealSense serial; set WIDE_REALSENSE_SERIAL or pass --wide-serial")

    provider = dual_frame_provider or capture_dual_realsense_once
    frames = provider(
        wrist_serial=wrist_serial,
        wide_serial=wide_serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_frames=args.warmup_frames,
        wide_depth=args.wide_depth,
        color_exposure=args.color_exposure,
        color_gain=args.color_gain,
        color_brightness=args.color_brightness,
        wrist_color_exposure=args.wrist_color_exposure,
        wide_color_exposure=args.wide_color_exposure,
        wrist_color_gain=args.wrist_color_gain,
        wide_color_gain=args.wide_color_gain,
        wrist_color_auto=args.wrist_color_auto,
        wide_color_auto=args.wide_color_auto,
        auto_exposure_priority=args.auto_exposure_priority,
    )
    robot_state = {}
    if args.robot_state_json:
        robot_state = json.loads(Path(args.robot_state_json).read_text(encoding="utf-8"))
    run_dir = config.client_run_root / args.run_id
    snapshot_dir = save_experiment_node_snapshot(
        run_dir=run_dir,
        run_id=args.run_id,
        node=args.node,
        frames=frames,
        sequence=args.sequence,
        robot_state=robot_state,
        notes=args.notes,
    )
    _print_json(
        stdout,
        {
            "run_id": args.run_id,
            "node": args.node,
            "snapshot_dir": str(snapshot_dir),
        },
    )
    return 0


def _server_roundtrip(config: ClientConfig, args: argparse.Namespace, stdout: TextIO) -> int:
    run_root = config.client_run_root / args.run_id
    commands = build_transfer_commands(config, args.run_id, run_root)
    command_plan = {
        "mkdir_remote": commands.mkdir_remote,
        "upload_request": commands.upload_request,
        "run_remote": commands.run_remote,
        "download_result": commands.download_result,
    }
    if not args.execute_transfer:
        _print_json(stdout, {"dry_run": True, "commands": command_plan})
        return 0

    (run_root / "result").mkdir(parents=True, exist_ok=True)
    for command in (
        commands.mkdir_remote,
        commands.upload_request,
        commands.run_remote,
        commands.download_result,
    ):
        run_command(command)
    _print_json(stdout, {"server_roundtrip_ok": True, "run_id": args.run_id})
    return 0


def main(
    argv: list[str] | None = None,
    env: Mapping[str, str] | None = None,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    client_factory: Any = None,
    dual_frame_provider: Any = None,
) -> int:
    out = stdout or sys.stdout
    err = stderr or sys.stderr
    parser = build_parser()
    args = parser.parse_args(argv)
    source_env = os.environ if env is None else env

    try:
        config = _load_config(source_env)
        if args.command == "check-env":
            return _check_env(config, out)
        if args.command == "camera-check":
            from real_robot.realsense_capture import realsense_pipeline_check

            _print_json(out, realsense_pipeline_check(config.realsense_serial))
            return 0
        if args.command == "aubo-probe":
            return _aubo_probe(config, client_factory, out)
        if args.command == "aubo-startup":
            return _aubo_startup(config, args.execute_hardware_action, client_factory, out)
        if args.command == "gripper-open":
            return _set_gripper(config, "open", args.execute_hardware_action, client_factory, out)
        if args.command == "gripper-close":
            return _set_gripper(config, "close", args.execute_hardware_action, client_factory, out)
        if args.command == "capture-request":
            return _capture_request(config, args, out)
        if args.command == "capture-node":
            return _capture_node(config, args, out, dual_frame_provider)
        if args.command == "server-roundtrip":
            return _server_roundtrip(config, args, out)
        parser.error(f"unknown command {args.command}")
        return 2
    except HardwareActionError as exc:
        print(str(exc), file=err)
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
