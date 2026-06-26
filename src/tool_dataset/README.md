# Tool Mesh Dataset Generation

This module contains the source code used to build an object-centric tool
completion and affordance-segmentation dataset from licensed or user-supplied
industrial tool meshes. The repository does not include raw meshes, generated
datasets, checkpoints, or experiment logs.

## Expected Mesh Layout

Preferred public layout:

```text
<MESH_ROOT>/
  <category>/
    <object_id>/
      models/
        obj_<object_id>.ply
      models_parts/
        obj_<object_id>_handle.ply
        obj_<object_id>_not.ply
```

The builder also accepts a compatibility layout with
`*_no_depth/*_dataset_<category>/models` and `models_parts` folders.

Supported category folder names are:

```text
adjustable_wrenches
combinational_wrenches
fixed_joint_pliers
hammers
locking_pliers
power_drills
ratchets
screwdrivers
```

## Build Dataset

```bash
python src/tool_dataset/build_tool_mesh_dataset.py \
  --mesh-root /path/to/tool_meshes \
  --out-dir src/datasets/ToolMesh-mini \
  --n-complete 16384 \
  --n-partial 2048 \
  --n-views 24 \
  --target-train-samples-per-category 1000 \
  --eval-partials-per-object 4 \
  --visible-ratio-min 0.38 \
  --visible-ratio-max 0.70 \
  --crop-ratio-min 0.32 \
  --crop-ratio-max 0.90 \
  --partial-jitter-std 0.002 \
  --n-seg-points 4096 \
  --seed 2026
```

The output layout is:

```text
src/datasets/ToolMesh-mini/
  completion/
    ToolMeshMini.json
    partial_metadata.json
    train|val|test/
      partial/<category>/<sample_id>/00.h5
      gt/<category>/<sample_id>.h5
  segmentation/
    manifest.json
    train|val|test/*.npz
  README.md
```

Completion samples are object-level only. Partial clouds are created from the
same complete object by retaining a virtual visible side and applying a
camera-plane crop. Scene geometry, neighboring tools, table surfaces, and
background points are not included.

Segmentation samples use binary labels: `1=handle` and `0=not_handle`.

## Optional Partial Modes

The builder includes optional full-view and thin-surface partial modes. They are
disabled by default:

```bash
--full-view-prob 0.0
--thin-partial-prob 0.0
--mask-hole-prob 0.0
```

Use these only when intentionally creating a new dataset variant and retraining
all completion and segmentation models.

## Train Affordance Segmentation

```bash
python src/tool_dataset/train_affordance_segmentation.py \
  --data-root src/datasets/ToolMesh-mini \
  --out-dir outputs/tool_dataset/affordance_dgcnn_partseg \
  --model dgcnn_partseg \
  --epochs 120 \
  --batch-size 16 \
  --n-points 2048 \
  --num-workers 4
```

The training script includes MLP, PointNet, PointNet++-lite, DGCNN, and
DGCNN-PartSeg-style binary segmentation models. Checkpoints and metrics are
written under `outputs/` and should not be committed.
