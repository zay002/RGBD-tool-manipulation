# Model and Resource Setup

This repository intentionally ships source code and lightweight illustrative
assets only. Learned model weights, external repositories, licensed robot
models, and training data must be obtained separately.

## Expected Runtime Layout

The server-side entry point accepts overrides through environment variables.
The following layout is used when no override is provided:

```text
weights/
  detector/
    yoloworld.pt
  segmenter/
    sam.pt
  completion/
    ckpt-best.pth
  affordance/
    ckpt-best.pth

external/
  contact_graspnet_pytorch/
  seedformer/
  point_completion_models/
  point_segmentation_models/

assets/
  robots/
  grippers/
  tools/
```

The repository does not require this full layout for import or the lightweight
pipeline check. It is the recommended convention for full experiments.

## Environment Overrides

Common server-side overrides:

```bash
export YOLOWORLD_WEIGHTS=/path/to/yolo-world.pt
export SAM_CHECKPOINT=/path/to/sam.pt
export SEEDFORMER_CHECKPOINT=/path/to/completion_ckpt.pth
export DGCNN_CHECKPOINT=/path/to/affordance_ckpt.pth
export CONTACT_GRASPNET_ROOT=/path/to/contact_graspnet_pytorch
export FULL_PIPELINE_DEVICE=cuda:0
export COMPLETION_AFFORDANCE_BACKEND=your_package.your_backend
```

`COMPLETION_AFFORDANCE_BACKEND` is optional. If it is not set,
`scripts/run_completion_affordance.py` writes valid point-cloud artifacts using
a deterministic geometric completion and a handle-like affordance heuristic.
This fallback is for request/response contract testing only; full experiments
should provide learned completion and affordance backends.

## Completion and Affordance Backend Contract

The backend module must define:

```python
def run_completion_affordance(partial_norm, args) -> dict:
    return {
        "completed_normalized": completed_points,  # shape: N x 3, float32
        "completed_labels": affordance_labels,     # shape: N, int64
        "backend": "model_name_or_experiment_id",
    }
```

`partial_norm` is a normalized 2048-point object cloud. The script restores the
completed output to camera coordinates, propagates labels to the visible cloud,
and writes:

```text
completed_camera.npy
completed_affordance_labels.npy
partial_camera_all.npy
partial_affordance_labels.npy
completion_segmentation_summary.json
```

If your completion and segmentation models are separate, call both inside this
backend function. The public repository intentionally keeps this model-specific
code outside the core pipeline because checkpoint formats and third-party model
repositories vary.

## Replaceable Modules

The public code does not assume one fixed backbone. The following interfaces are
expected:

- Detector: returns one target bounding box and confidence for the query.
- Masker: returns a binary target mask from the RGB image and prompt.
- Depth lifting: converts masked depth and intrinsics into a camera-frame point
  cloud.
- Completion: maps a partial object cloud to a completed object cloud in a
  normalized frame and stores the inverse transform.
- Affordance segmentation: assigns part labels or probabilities to points.
- Grasp generator: returns candidate 6-DoF grasps with approach and closing
  axes.
- Client executor: checks reachability, workspace limits, collision constraints,
  and hardware safety before motion.

## Data Policy

Do not commit raw robot logs, training datasets, calibration files, checkpoints,
or generated point clouds. If reproducibility requires these assets, publish
them as a separate release artifact with its own license and manifest.
