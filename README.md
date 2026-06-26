# Modular Object-Centric Tool Manipulation

This repository is an anonymized, source-only release for an object-centric
RGB-D manipulation pipeline. The code is organized so that each module can be
replaced independently:

```text
wrist RGB-D
-> open-vocabulary target grounding
-> 2D target mask
-> masked depth to object partial cloud
-> optional object completion
-> optional affordance-part segmentation
-> grasp candidate generation and ranking
-> client-side robot safety checks and execution
```

The release intentionally does not include training data, private robot logs,
model checkpoints, raw CAD datasets, manuscript files, or machine-specific
configuration files.

## Implemented Components

| Path | What is implemented |
| --- | --- |
| `src/sim_pipeline/mask_depth_fusion` | Masked RGB-D back-projection into camera-frame object point clouds. |
| `src/sim_pipeline/completion_inference` | Completion interface plus a deterministic geometric fallback for contract testing. |
| `src/sim_pipeline/grasp_generation` | Parallel-jaw grasp candidate representation and geometric candidate generator. |
| `src/sim_pipeline/grasp_scoring` | Clearance and collision-style scoring utilities for grasp ranking. |
| `src/real_robot/realsense_capture.py` | RGB-D capture, depth visualization, request-folder serialization, and point-cloud export. |
| `src/real_robot/transfer.py` | SSH/SCP request and result transfer commands between client host and inference server. |
| `src/real_robot/aubo_adapter.py` | Robot and gripper adapter with explicit safety gates for hardware-changing actions. |
| `src/real_robot/execution_planner.py` | Grasp-axis to robot-orientation utilities and TCP offset handling. |
| `src/real_robot/handeye_calibration.py` | Hand-eye and grasp-offset records used by the client executor. |
| `scripts/run_server_inference.py` | Server-side request processor with hooks for grounding, masking, completion, affordance segmentation, and Contact-Grasp style ranking. |
| `scripts/run_real_robot_client.py` | Client-side command-line entry point for environment checks, RGB-D capture, server round trip, robot probing, and guarded gripper commands. |
| `examples/wrench_occlusion_scene` | A redistributable cluttered wrench scene, primitive SDF, and representative RGB-D and perception-stage images. |

## Quick Start

Create an environment and install the base dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements/base.txt
```

Verify the local contract:

```bash
PYTHONPATH="$PWD/src" python scripts/run_pipeline_check.py --out-dir outputs/pipeline_check
```

Expected output:

```text
outputs/pipeline_check/summary.json
```

This command exercises the same point-cloud and grasp-candidate data structures
used by the server path. It does not replace learned models; it verifies that a
fresh checkout has a working Python environment before external weights are
installed.

Install role-specific dependencies when needed:

```bash
python -m pip install -r requirements/server.txt   # GPU inference host
python -m pip install -r requirements/client.txt   # RGB-D and transfer host
python -m pip install -r requirements/hardware.txt # RealSense runtime
```

## Deployment Split

The code supports a two-machine deployment:

- A client host captures RGB-D, talks to the robot controller and gripper, and
  performs safety checks.
- A GPU server runs the heavy perception and geometry modules.

Machine-specific values are read from environment variables. Do not hard-code
server addresses, passwords, robot IPs, or calibration files in the repository.
See `docs/REAL_ROBOT_PIPELINE.md` for the expected request/result contract.

The client host can be any machine that has camera and robot connectivity. It
does not need to be a notebook computer.

## Full Pipeline Interfaces

The release includes the server entry point and the adapter scripts used to
connect common external backends:

- `scripts/run_server_inference.py` processes a request folder and writes
  `result.json`.
- `scripts/run_completion_affordance.py` converts masked depth into partial
  points, runs an optional completion and affordance backend, and writes
  camera-frame point clouds and labels.
- `scripts/run_contact_graspnet_pytorch.py` adapts a separately installed
  Contact-GraspNet-style repository to the result schema.

Set `COMPLETION_AFFORDANCE_BACKEND` to an importable Python module that defines:

```python
def run_completion_affordance(partial_norm, args) -> dict:
    return {
        "completed_normalized": completed_points,  # Nx3 float32
        "completed_labels": affordance_labels,     # N int labels
        "backend": "your_backend_name",
    }
```

If this variable is unset, the script writes valid artifacts with a deterministic
geometric completion and handle-like affordance heuristic. That fallback is for
contract testing only and is marked as
`geometric_completion_affordance_heuristic` in the output summary.

## Example Scene

`examples/wrench_occlusion_scene/wrench_occlusion_primitive.sdf` provides a
redistributable primitive approximation of the cluttered wrench scene. The
included images show the intended visual evidence:

- wrist RGB image
- wrist depth image
- open-vocabulary target detection
- mask overlay
- masked depth foreground
- global success and failure views

The scene is intentionally simple and does not vendor third-party robot meshes.
Replace the primitive robot and gripper with your own licensed assets when
running full-fidelity simulation.

## External Models and Weights

The repository does not vendor third-party model code or weights, but the server
entry point exposes the integration points. Typical module choices include:

- open-vocabulary detector: YOLO-World or Grounding DINO
- instance mask: SAM or SAM2
- object completion: SeedFormer, PoinTr, AdaPoinTr, or another point completion
  model
- affordance segmentation: DGCNN PartSeg, PointNet++, PointNeXt, or another
  point segmentation model
- grasp generation: Contact-GraspNet, AnyGrasp, GPD, GPG, or a custom generator

See `docs/THIRD_PARTY_ACKNOWLEDGEMENTS.md` for attribution notes.

See `docs/MODEL_AND_RESOURCE_SETUP.md` for the expected locations of optional
model weights, robot meshes, and external repositories.

## Release Policy

This release is source-only. Do not commit:

- model weights or checkpoints
- raw RGB-D logs
- training datasets
- private calibration files
- machine-specific environment files
- generated point clouds, tensors, or experiment folders
