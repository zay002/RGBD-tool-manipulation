# Reproducibility Notes

This release is intended to make the engineering pipeline auditable without
redistributing private datasets, checkpoints, or hardware logs.

## What Is Included

- Python source for RGB-D request creation, point-cloud conversion, geometric
  fallback completion, grasp-candidate generation, scoring, and client/server
  transfer.
- Source code for building an object-centric tool completion and affordance
  segmentation dataset from user-supplied meshes, plus point-cloud affordance
  segmentation training code.
- Client-side adapters for a robot controller, digital-output gripper control,
  RealSense RGB-D capture, hand-eye records, and guarded hardware actions.
- A server-side inference entry point plus adapter scripts for detector,
  mask, completion, affordance, and Contact-Grasp-style grasp-generation
  backends.
- A primitive SDF scene and representative rendered images for a cluttered
  wrench acquisition scenario.
- Example configuration and environment templates.

## What Must Be Supplied Separately

- Detector and segmenter weights.
- Point-cloud completion and affordance segmentation checkpoints.
- Contact-style grasp generator code and checkpoints, if that backend is used.
- A Python completion and affordance backend module, unless the deterministic
  geometric fallback is sufficient for checking the file contract.
- Licensed robot, gripper, and tool meshes for high-fidelity simulation.
- Camera calibration and robot hand-eye calibration.
- Training datasets and physical robot logs.
- Raw tool meshes and generated dataset files. The generation code is included,
  but users must supply meshes with compatible complete-object and handle-part
  files.

## Verification Levels

1. Source import check:

   ```bash
   python -m compileall -q src scripts
   ```

2. Geometry contract check:

   ```bash
   PYTHONPATH="$PWD/src" python scripts/run_pipeline_check.py --out-dir outputs/pipeline_check
   ```

3. Dataset code environment check:

   ```bash
   python -m pip install -r requirements/dataset.txt
   python src/tool_dataset/build_tool_mesh_dataset.py --help
   python src/tool_dataset/train_affordance_segmentation.py --help
   ```

4. Client environment check:

   ```bash
   python scripts/run_real_robot_client.py check-env
   ```

5. Completion and affordance contract:

   ```bash
   python scripts/run_completion_affordance.py \
     --depth-pfm /path/to/depth_m.pfm \
     --mask /path/to/mask.png \
     --out-dir outputs/completion_contract
   ```

6. Server request processing:

   ```bash
   PYTHONPATH="$PWD/src" python scripts/run_server_inference.py \
     /path/to/request_dir \
     /path/to/result_dir
   ```

7. Hardware execution:

   Hardware-changing commands must be run only after calibration, workspace
   checks, and emergency-stop access have been verified.

## Expected Request and Result Files

The formal request/result schema is described in
`docs/REAL_ROBOT_PIPELINE.md`. A run is considered reproducible if another
system can:

- create the required request folder from a wrist RGB-D frame;
- run the server-side inference entry point with equivalent module backends;
- produce a `result.json` containing gate status and grasp poses;
- apply the client-side safety checks before any robot motion.
