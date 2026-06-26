# Real Robot Pipeline

This document describes the public request/result contract. It intentionally
uses template values for hardware and network settings.

## Roles

- Client host:
  - captures wrist RGB-D frames
  - stores camera intrinsics and the current base-to-camera transform
  - uploads a request folder to a GPU server
  - downloads the result folder
  - validates gates, transforms the grasp pose, and controls robot execution
- GPU server:
  - reads the request folder
  - runs target grounding, masking, depth lifting, completion, affordance
    segmentation, grasp generation, and ranking
  - writes a result JSON and optional visual artifacts

## Required Environment Variables

Client-side:

```bash
export TMR_SERVER_USER="<server-user>"
export TMR_SERVER_HOST="<server-host>"
export TMR_SERVER_PORT="22"
export TMR_SERVER_ROOT="/path/to/server/repo"

export AUBO_ROBOT_IP="<robot-controller-ip>"
export AUBO_RPC_PORT="30005"
export AUBO_RTDE_PORT="9012"
export AUBO_RPC_USER="<controller-user>"
export AUBO_RPC_PASSWORD="<controller-password>"

export TMR_GRIPPER_DO_INDEX="1"
export TMR_GRIPPER_DO_CLOSE_VALUE="true"
export TMR_GRIPPER_DO_OPEN_VALUE="false"

export WRIST_REALSENSE_SERIAL="<optional-camera-serial>"
export TMR_CLIENT_RUN_ROOT="$HOME/tool_manipulation_runs"
```

Do not commit real values for these variables.

## Client Commands

The client entry point is:

```bash
python scripts/run_real_robot_client.py <command>
```

Non-motion checks:

```bash
python scripts/run_real_robot_client.py check-env
python scripts/run_real_robot_client.py camera-check
python scripts/run_real_robot_client.py aubo-probe
```

Request capture:

```bash
python scripts/run_real_robot_client.py capture-request \
  --scene 1 \
  --trial 1 \
  --query "grasp the tool handle" \
  --target tool \
  --tf-base-camera-json calibration/base_T_camera.json
```

Server round trip:

```bash
python scripts/run_real_robot_client.py server-roundtrip \
  --run-id real_YYYYMMDD_HHMMSS_scene01_trial01 \
  --execute-transfer
```

Hardware-changing commands require an explicit flag:

```bash
python scripts/run_real_robot_client.py aubo-startup --execute-hardware-action
python scripts/run_real_robot_client.py gripper-open --execute-hardware-action
python scripts/run_real_robot_client.py gripper-close --execute-hardware-action
```

## Request Folder

Each request folder contains:

```text
request.json
rgb.png
depth_uint16.png
intrinsics.json
tf_base_camera.json
```

`request.json`:

```json
{
  "run_id": "real_YYYYMMDD_HHMMSS_scene01_trial01",
  "query": "grasp the tool handle",
  "target": "tool",
  "camera_frame": "wrist_camera_color_optical_frame",
  "base_frame": "robot_base_link",
  "depth_units": "millimeter",
  "return_artifacts": true
}
```

`intrinsics.json`:

```json
{
  "width": 640,
  "height": 480,
  "fx": 600.0,
  "fy": 600.0,
  "cx": 320.0,
  "cy": 240.0,
  "depth_scale_m": 0.001
}
```

`tf_base_camera.json` stores a 4-by-4 homogeneous matrix:

```json
{
  "T_base_camera": [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
  ]
}
```

## Result Folder

The server writes `result.json`:

```json
{
  "status": "ok",
  "run_id": "real_YYYYMMDD_HHMMSS_scene01_trial01",
  "target_box_xyxy": [100, 100, 300, 300],
  "mask_area_px": 12000,
  "partial_points": 2048,
  "completion_status": "ok",
  "affordance_status": "ok",
  "grasp_pose_camera": {
    "translation_m": [0.0, 0.0, 0.5],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "approach_pose_camera": {
    "translation_m": [0.0, 0.0, 0.6],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "quality_gates": {
    "detection_ok": true,
    "mask_ok": true,
    "depth_ok": true,
    "completion_ok": true,
    "affordance_ok": true
  }
}
```

If any gate fails, the server should write `"status": "rejected"` and a
human-readable `reject_reason`.

## Safety

Hardware-changing commands require `--execute-hardware-action`. This is a
deliberate guard to prevent accidental startup, gripper actuation, or robot
motion during setup.
