from __future__ import annotations

import numpy as np

from sim_pipeline.mask_depth_fusion.depth_to_cloud import depth_to_point_cloud
from sim_pipeline.rgbd_rendering.synthetic_box_scene import render_synthetic_box_scene


def generate_synthetic_partial_cloud(width: int = 320, height: int = 240) -> dict[str, np.ndarray | object]:
    rgb, depth, mask, intrinsics = render_synthetic_box_scene(width=width, height=height)
    partial_cloud = depth_to_point_cloud(depth, mask, intrinsics)
    return {
        "rgb": rgb,
        "depth": depth,
        "mask": mask,
        "intrinsics": intrinsics,
        "partial_cloud": partial_cloud,
    }

