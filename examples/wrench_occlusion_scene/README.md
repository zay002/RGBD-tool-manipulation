# Wrench Occlusion Scene Example

This directory contains a small redistributable example for a cluttered
tool-acquisition scene.

Files:

- `wrench_occlusion_primitive.sdf`: a primitive SDF scene with a table, target
  wrench approximation, distractor tools, a simplified robot/gripper silhouette,
  and a wrist camera.
- `images/wrist_rgb.png`: rendered wrist RGB view.
- `images/wrist_depth.png`: rendered wrist depth visualization.
- `images/target_detection.png`: example target grounding visualization.
- `images/mask_overlay.png`: example target mask overlay.
- `images/masked_depth.png`: example masked depth foreground.
- `images/global_success.png`: example successful return-to-hover view.
- `images/global_failure.png`: example failure/outcome comparison view.

The images are documentation artifacts only. They are not training data and do
not contain real robot logs.

The SDF avoids private or third-party mesh paths by using primitive geometry.
Replace the primitive robot, gripper, and tool shapes with your own licensed
assets for high-fidelity simulation.

