from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

from .contracts import ClientConfig


@dataclass(frozen=True)
class ServerPaths:
    remote_inbox: str
    remote_outbox: str


@dataclass(frozen=True)
class TransferCommands:
    mkdir_remote: list[str]
    upload_request: list[str]
    run_remote: list[str]
    download_result: list[str]


def _server(config: ClientConfig) -> str:
    return f"{config.server_user}@{config.server_host}"


def _ssh_options() -> list[str]:
    known_hosts = Path.home() / ".ssh" / "tool_manipulation_known_hosts"
    return [
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
    ]


def build_server_paths(config: ClientConfig, run_id: str) -> ServerPaths:
    root = config.server_root.rstrip("/")
    return ServerPaths(
        remote_inbox=f"{root}/outputs/real_robot/inbox/{run_id}",
        remote_outbox=f"{root}/outputs/real_robot/outbox/{run_id}",
    )


def build_transfer_commands(
    config: ClientConfig,
    run_id: str,
    run_root: Path,
) -> TransferCommands:
    server = _server(config)
    paths = build_server_paths(config, run_id)
    request_dir = run_root / "request" / "."
    result_dir = run_root / "result"
    port = str(config.server_port)
    mkdir_remote = [
        "ssh",
        "-p",
        port,
        *_ssh_options(),
        server,
        f"mkdir -p '{paths.remote_inbox}' '{paths.remote_outbox}'",
    ]
    upload_request = [
        "scp",
        "-P",
        port,
        *_ssh_options(),
        "-r",
        str(request_dir),
        f"{server}:{paths.remote_inbox}/",
    ]
    run_remote = [
        "ssh",
        "-p",
        port,
        *_ssh_options(),
        server,
        (
            f"cd '{config.server_root}' && "
            f"./scripts/run_real_robot_inference.sh "
            f"'{paths.remote_inbox}' '{paths.remote_outbox}'"
        ),
    ]
    download_result = [
        "scp",
        "-P",
        port,
        *_ssh_options(),
        "-r",
        f"{server}:{paths.remote_outbox}/.",
        str(result_dir),
    ]
    return TransferCommands(
        mkdir_remote=mkdir_remote,
        upload_request=upload_request,
        run_remote=run_remote,
        download_result=download_result,
    )


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True)
