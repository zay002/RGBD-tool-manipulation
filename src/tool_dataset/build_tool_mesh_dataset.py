#!/usr/bin/env python3
"""Build object-level ToolMesh-mini completion and affordance datasets.

Inputs are source industrial tool object meshes:
- complete object mesh: models/obj_*.ply
- affordance part mesh: models_parts/obj_*_handle.ply
- non-affordance part mesh: models_parts/obj_*_not.ply

Outputs:
- Completion3D-style HDF5 point clouds for SeedFormer.
- NPZ point clouds with binary labels for affordance part segmentation.

Completion samples are object-level only. The default ToolMesh-mini profile keeps
points visible from virtual camera directions, then applies a 2D camera-plane
crop to simulate segmented single-object RGB-D observations. Scene folders and
multi-object annotations are intentionally not used here. Experimental thin
surface patches are available through explicit flags but are disabled by
default because the reference pipeline uses the stable crop-based v1 split.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import trimesh
from scipy.spatial import cKDTree


INDUSTRIAL_KEYWORDS = (
    "adjustable_wrenches",
    "combinational_wrenches",
    "fixed_joint_pliers",
    "hammers",
    "locking_pliers",
    "power_drills",
    "ratchets",
    "screwdrivers",
)


@dataclass(frozen=True)
class ObjectRecord:
    category: str
    object_id: str
    complete_mesh: Path
    handle_mesh: Path
    not_mesh: Path

    @property
    def model_id(self) -> str:
        return f"{self.category}_{self.object_id}"


def discover_objects(mesh_root: Path) -> list[ObjectRecord]:
    records: list[ObjectRecord] = []

    # Preferred public layout:
    # <mesh_root>/<category>/<object_id>/models/obj.ply
    # <mesh_root>/<category>/<object_id>/models_parts/obj_handle.ply
    # <mesh_root>/<category>/<object_id>/models_parts/obj_not.ply
    for object_dir in sorted(mesh_root.glob("*/*")):
        models_dir = object_dir / "models"
        parts_dir = object_dir / "models_parts"
        if not models_dir.is_dir() or not parts_dir.is_dir():
            continue
        category = object_dir.parent.name
        if not any(key in category for key in INDUSTRIAL_KEYWORDS):
            continue
        for mesh_path in sorted(models_dir.glob("obj_*.ply")):
            object_id = mesh_path.stem.removeprefix("obj_")
            handle = parts_dir / f"obj_{object_id}_handle.ply"
            not_part = parts_dir / f"obj_{object_id}_not.ply"
            if handle.exists() and not_part.exists():
                records.append(ObjectRecord(category, object_id, mesh_path, handle, not_part))

    if records:
        return records

    # Compatibility layout used by some mesh exports:
    # <mesh_root>/*_no_depth/*_dataset_<category>/{models,models_parts}/...
    for dataset_dir in sorted(mesh_root.glob("*_no_depth/*_dataset_*")):
        category = dataset_dir.name.split("_dataset_", 1)[-1]
        if not any(key in category for key in INDUSTRIAL_KEYWORDS):
            continue
        models_dir = dataset_dir / "models"
        parts_dir = dataset_dir / "models_parts"
        for mesh_path in sorted(models_dir.glob("obj_*.ply")):
            object_id = mesh_path.stem.removeprefix("obj_")
            handle = parts_dir / f"obj_{object_id}_handle.ply"
            not_part = parts_dir / f"obj_{object_id}_not.ply"
            if handle.exists() and not_part.exists():
                records.append(ObjectRecord(category, object_id, mesh_path, handle, not_part))
    return records


def load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    if mesh.vertices.size == 0:
        raise ValueError(f"empty mesh: {path}")
    return mesh


def normalize_points(points: np.ndarray) -> np.ndarray:
    points = points.astype(np.float32, copy=False)
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.linalg.norm(points, axis=1).max()
    if scale > 0:
        points = points / scale
    return points.astype(np.float32)


def sample_surface(mesh: trimesh.Trimesh, n_points: int, rng: np.random.Generator) -> np.ndarray:
    # trimesh uses numpy global state internally; seed it for reproducible sampling.
    state = np.random.get_state()
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    try:
        points, _ = trimesh.sample.sample_surface(mesh, n_points)
    finally:
        np.random.set_state(state)
    return points.astype(np.float32)


def sample_or_pad(points: np.ndarray, n_points: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        raise ValueError("cannot sample an empty point set")
    replace = points.shape[0] < n_points
    choice = rng.choice(points.shape[0], n_points, replace=replace)
    return points[choice].astype(np.float32)


def camera_basis(view_dir: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    direction = view_dir / (np.linalg.norm(view_dir) + 1e-8)
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(direction @ up)) > 0.92:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = np.cross(up, direction)
    right = right / (np.linalg.norm(right) + 1e-8)
    cam_up = np.cross(direction, right)
    cam_up = cam_up / (np.linalg.norm(cam_up) + 1e-8)
    return direction.astype(np.float32), right.astype(np.float32), cam_up.astype(np.float32)


def apply_uv_holes(
    points: np.ndarray,
    uv: np.ndarray,
    rng: np.random.Generator,
    probability: float,
    max_holes: int = 3,
) -> np.ndarray:
    if points.shape[0] < 128 or rng.random() > probability:
        return points
    keep = np.ones(points.shape[0], dtype=bool)
    extent = np.maximum(uv.max(axis=0) - uv.min(axis=0), 1e-4)
    for _ in range(int(rng.integers(1, max_holes + 1))):
        center = uv[int(rng.integers(0, uv.shape[0]))]
        radius = extent * float(rng.uniform(0.06, 0.18))
        hole = (
            ((uv[:, 0] - center[0]) / max(float(radius[0]), 1e-4)) ** 2
            + ((uv[:, 1] - center[1]) / max(float(radius[1]), 1e-4)) ** 2
        ) <= 1.0
        keep &= ~hole
    if int(keep.sum()) < 64:
        return points
    return points[keep].astype(np.float32)


def virtual_view_crop_partial(
    points: np.ndarray,
    view_dir: np.ndarray,
    n_partial: int,
    rng: np.random.Generator,
    visible_ratio: tuple[float, float],
    crop_ratio: tuple[float, float],
    jitter_std: float,
) -> np.ndarray:
    direction, right, cam_up = camera_basis(view_dir)
    depths = points @ direction
    ratio = float(rng.uniform(visible_ratio[0], visible_ratio[1]))
    keep_count = max(n_partial, int(points.shape[0] * ratio))
    keep_count = min(points.shape[0], keep_count)
    order = np.argsort(depths)[-keep_count:]
    visible = points[order]

    uv = np.stack([visible @ right, visible @ cam_up], axis=1)
    extent = np.maximum(uv.max(axis=0) - uv.min(axis=0), 1e-4)
    crop_scale = float(rng.uniform(crop_ratio[0], crop_ratio[1]))
    if crop_scale >= 0.999:
        center = (uv.max(axis=0) + uv.min(axis=0)) * 0.5
    else:
        center_idx = int(rng.integers(0, visible.shape[0]))
        center = uv[center_idx]
    half_size = extent * crop_scale * 0.5
    crop_mask = np.logical_and(
        np.abs(uv[:, 0] - center[0]) <= half_size[0],
        np.abs(uv[:, 1] - center[1]) <= half_size[1],
    )
    cropped = visible[crop_mask]
    if cropped.shape[0] < max(64, n_partial // 8):
        cropped = visible

    partial = sample_or_pad(cropped, n_partial, rng)
    if jitter_std > 0:
        partial = partial + rng.normal(0.0, jitter_std, size=partial.shape).astype(np.float32)
    return partial.astype(np.float32)


def full_visible_view_partial(
    points: np.ndarray,
    view_dir: np.ndarray,
    n_partial: int,
    rng: np.random.Generator,
    visible_ratio: tuple[float, float],
    jitter_std: float,
) -> np.ndarray:
    direction = view_dir / (np.linalg.norm(view_dir) + 1e-8)
    depths = points @ direction
    ratio = float(rng.uniform(visible_ratio[0], visible_ratio[1]))
    keep_count = max(n_partial, int(points.shape[0] * ratio))
    keep_count = min(points.shape[0], keep_count)
    visible = points[np.argsort(depths)[-keep_count:]]
    partial = sample_or_pad(visible, n_partial, rng)
    if jitter_std > 0:
        partial = partial + rng.normal(0.0, jitter_std, size=partial.shape).astype(np.float32)
    return partial.astype(np.float32)


def thin_surface_patch_partial(
    points: np.ndarray,
    view_dir: np.ndarray,
    n_partial: int,
    rng: np.random.Generator,
    visible_ratio: tuple[float, float],
    crop_ratio: tuple[float, float],
    jitter_std: float,
    mask_hole_prob: float,
) -> np.ndarray:
    del visible_ratio, jitter_std
    direction, right, cam_up = camera_basis(view_dir)
    depths = points @ direction

    uv_all = np.stack([points @ right, points @ cam_up], axis=1)
    uv_min = uv_all.min(axis=0)
    uv_extent = np.maximum(uv_all.max(axis=0) - uv_min, 1e-4)

    # Approximate a depth-camera observation from the complete point cloud:
    # each UV cell keeps only the front-most surface layer. This produces a
    # genuine single-view sheet cut from the complete object instead of random
    # floating patches.
    grid_size = 96
    ij = np.floor((uv_all - uv_min) / uv_extent * (grid_size - 1)).astype(np.int32)
    ij = np.clip(ij, 0, grid_size - 1)
    cell_id = ij[:, 0] * grid_size + ij[:, 1]
    max_depth = np.full(grid_size * grid_size, -np.inf, dtype=np.float32)
    np.maximum.at(max_depth, cell_id, depths.astype(np.float32))
    depth_span = max(float(np.ptp(depths)), 1e-4)
    layer_thickness = depth_span * float(rng.uniform(0.018, 0.045))
    visible_mask = depths >= (max_depth[cell_id] - layer_thickness)
    visible = points[visible_mask]
    visible_uv = uv_all[visible_mask]
    if visible.shape[0] < max(128, n_partial // 8):
        keep_count = max(n_partial, int(points.shape[0] * 0.25))
        keep_count = min(points.shape[0], keep_count)
        visible_idx = np.argsort(depths)[-keep_count:]
        visible = points[visible_idx]
        visible_uv = uv_all[visible_idx]

    extent = np.maximum(visible_uv.max(axis=0) - visible_uv.min(axis=0), 1e-4)
    scale = float(rng.uniform(crop_ratio[0], crop_ratio[1]))
    if scale >= 0.98:
        center = (visible_uv.max(axis=0) + visible_uv.min(axis=0)) * 0.5
    else:
        center = visible_uv[int(rng.integers(0, visible.shape[0]))]
    half_size = extent * scale * 0.5
    crop_mask = np.logical_and(
        np.abs(visible_uv[:, 0] - center[0]) <= half_size[0],
        np.abs(visible_uv[:, 1] - center[1]) <= half_size[1],
    )
    cropped = visible[crop_mask]
    cropped_uv = visible_uv[crop_mask]
    if cropped.shape[0] < max(128, n_partial // 8):
        cropped = visible
        cropped_uv = visible_uv

    cropped = apply_uv_holes(cropped, cropped_uv, rng, probability=mask_hole_prob)

    partial = sample_or_pad(cropped, n_partial, rng)
    # Keep thin-surface partials on the complete point-cloud surface. Sensor
    # noise belongs in the RGB-D simulation stage, not in this mesh-derived
    # supervision target.
    return partial.astype(np.float32)


def mixed_real_observation_partial(
    points: np.ndarray,
    view_dir: np.ndarray,
    n_partial: int,
    rng: np.random.Generator,
    visible_ratio: tuple[float, float],
    crop_ratio: tuple[float, float],
    thin_visible_ratio: tuple[float, float],
    thin_crop_ratio: tuple[float, float],
    jitter_std: float,
    full_view_prob: float,
    thin_partial_prob: float,
    mask_hole_prob: float,
) -> tuple[np.ndarray, str]:
    mode_draw = rng.random()
    if mode_draw < full_view_prob:
        return (
            full_visible_view_partial(
                points,
                view_dir,
                n_partial,
                rng,
                visible_ratio=visible_ratio,
                jitter_std=jitter_std,
            ),
            "full_visible_view",
        )
    if mode_draw < full_view_prob + thin_partial_prob:
        return (
            thin_surface_patch_partial(
                points,
                view_dir,
                n_partial,
                rng,
                visible_ratio=thin_visible_ratio,
                crop_ratio=thin_crop_ratio,
                jitter_std=jitter_std,
                mask_hole_prob=mask_hole_prob,
            ),
            "thin_surface_patch",
        )
    partial = virtual_view_crop_partial(
        points,
        view_dir,
        n_partial,
        rng,
        visible_ratio=visible_ratio,
        crop_ratio=crop_ratio,
        jitter_std=jitter_std,
    )
    return partial, "view_random_crop"


def link_or_copy_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        import shutil

        shutil.copy2(source, target)


def build_sample_schedule(
    records: list[ObjectRecord],
    split: str,
    target_train_samples_per_category: int,
    eval_partials_per_object: int,
) -> dict[str, list[tuple[ObjectRecord, int]]]:
    by_category: dict[str, list[ObjectRecord]] = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)

    schedule: dict[str, list[tuple[ObjectRecord, int]]] = {}
    for category, items in by_category.items():
        if split == "train":
            target = max(len(items), target_train_samples_per_category)
            per_object = math.ceil(target / len(items))
            samples = [(record, aug_idx) for aug_idx in range(per_object) for record in items]
            schedule[category] = samples[:target]
        else:
            samples = [(record, aug_idx) for record in items for aug_idx in range(eval_partials_per_object)]
            schedule[category] = samples
    return schedule


def augmented_sample_id(record: ObjectRecord, aug_idx: int) -> str:
    if aug_idx == 0:
        return record.model_id
    return f"{record.model_id}_aug{aug_idx:04d}"


def sample_view(views: list[np.ndarray], aug_idx: int, rng: np.random.Generator) -> np.ndarray:
    base = views[aug_idx % len(views)]
    noise = rng.normal(0.0, 0.18, size=3).astype(np.float32)
    view = base + noise
    norm = np.linalg.norm(view)
    if norm <= 1e-6:
        return base
    return (view / norm).astype(np.float32)


def virtual_view_partial(points: np.ndarray, view_dir: np.ndarray, n_partial: int) -> np.ndarray:
    direction = view_dir / (np.linalg.norm(view_dir) + 1e-8)
    depths = points @ direction
    keep_count = max(n_partial, int(points.shape[0] * 0.45))
    order = np.argsort(depths)[-keep_count:]
    visible = points[order]
    if visible.shape[0] >= n_partial:
        choice = np.random.choice(visible.shape[0], n_partial, replace=False)
    else:
        choice = np.random.choice(visible.shape[0], n_partial, replace=True)
    return visible[choice].astype(np.float32)


def fibonacci_views(n_views: int) -> list[np.ndarray]:
    views: list[np.ndarray] = []
    golden = math.pi * (3.0 - math.sqrt(5.0))
    for i in range(n_views):
        z = 1.0 - (2.0 * (i + 0.5) / n_views)
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        theta = golden * i
        views.append(np.array([math.cos(theta) * radius, math.sin(theta) * radius, z], dtype=np.float32))
    return views


def write_h5(path: Path, points: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("data", data=points.astype(np.float32))


def split_records(records: list[ObjectRecord], seed: int) -> dict[str, list[ObjectRecord]]:
    by_category: dict[str, list[ObjectRecord]] = {}
    for record in records:
        by_category.setdefault(record.category, []).append(record)

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    for category, items in by_category.items():
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * 0.15))
        n_val = max(1, round(n * 0.15))
        splits["test"].extend(items[:n_test])
        splits["val"].extend(items[n_test:n_test + n_val])
        splits["train"].extend(items[n_test + n_val:])
    return splits


def build_completion(
    splits: dict[str, list[ObjectRecord]],
    out_dir: Path,
    n_complete: int,
    n_partial: int,
    n_views: int,
    seed: int,
    target_train_samples_per_category: int,
    eval_partials_per_object: int,
    visible_ratio: tuple[float, float],
    crop_ratio: tuple[float, float],
    thin_visible_ratio: tuple[float, float],
    thin_crop_ratio: tuple[float, float],
    jitter_std: float,
    full_view_prob: float,
    thin_partial_prob: float,
    mask_hole_prob: float,
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    views = fibonacci_views(n_views)
    categories: dict[str, dict[str, object]] = {}
    complete_cache: dict[str, Path] = {}
    partial_metadata: list[dict[str, object]] = []

    for split, records in splits.items():
        sample_schedule = build_sample_schedule(
            records,
            split,
            target_train_samples_per_category=target_train_samples_per_category,
            eval_partials_per_object=eval_partials_per_object,
        )
        for category in sorted(sample_schedule):
            for record, aug_idx in sample_schedule[category]:
                base_gt_path = complete_cache.get(record.model_id)
                if base_gt_path is None:
                    mesh = load_mesh(record.complete_mesh)
                    complete = normalize_points(sample_surface(mesh, n_complete, rng))
                    base_gt_path = out_dir / "completion" / "_gt_objects" / category / f"{record.model_id}.h5"
                    write_h5(base_gt_path, complete)
                    complete_cache[record.model_id] = base_gt_path
                else:
                    with h5py.File(base_gt_path) as handle:
                        complete = handle["data"][:].astype(np.float32)

                sample_id = augmented_sample_id(record, aug_idx)
                categories.setdefault(category, {
                    "taxonomy_id": category,
                    "taxonomy_name": category,
                    "train": [],
                    "val": [],
                    "test": [],
                })
                categories[category][split].append(sample_id)

                gt_path = out_dir / "completion" / split / "gt" / category / f"{sample_id}.h5"
                link_or_copy_file(base_gt_path, gt_path)
                view = sample_view(views, aug_idx, rng)
                partial, partial_mode = mixed_real_observation_partial(
                    complete,
                    view,
                    n_partial,
                    rng,
                    visible_ratio=visible_ratio,
                    crop_ratio=crop_ratio,
                    thin_visible_ratio=thin_visible_ratio,
                    thin_crop_ratio=thin_crop_ratio,
                    jitter_std=jitter_std,
                    full_view_prob=full_view_prob,
                    thin_partial_prob=thin_partial_prob,
                    mask_hole_prob=mask_hole_prob,
                )
                write_h5(
                    out_dir / "completion" / split / "partial" / category / sample_id / "00.h5",
                    partial,
                )
                partial_metadata.append({
                    "split": split,
                    "category": category,
                    "model_id": record.model_id,
                    "sample_id": sample_id,
                    "aug_idx": aug_idx,
                    "partial_mode": partial_mode,
                    "partial_path": str(
                        (Path("completion") / split / "partial" / category / sample_id / "00.h5").as_posix()
                    ),
                })

    category_list = [categories[name] for name in sorted(categories)]
    with (out_dir / "completion" / "ToolMeshMini.json").open("w", encoding="utf-8") as fh:
        json.dump(category_list, fh, ensure_ascii=False, indent=2)
    with (out_dir / "completion" / "partial_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(partial_metadata, fh, ensure_ascii=False, indent=2)
    return category_list


def build_segmentation(
    splits: dict[str, list[ObjectRecord]],
    out_dir: Path,
    n_points: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed + 17)
    manifest: dict[str, list[dict[str, str]]] = {"train": [], "val": [], "test": []}

    for split, records in splits.items():
        for record in records:
            handle_mesh = load_mesh(record.handle_mesh)
            not_mesh = load_mesh(record.not_mesh)
            n_handle = max(1, n_points // 2)
            n_not = n_points - n_handle
            handle_points = sample_surface(handle_mesh, n_handle, rng)
            not_points = sample_surface(not_mesh, n_not, rng)
            points = np.concatenate([handle_points, not_points], axis=0)
            labels = np.concatenate([
                np.ones(n_handle, dtype=np.int64),
                np.zeros(n_not, dtype=np.int64),
            ])

            points = normalize_points(points)
            order = rng.permutation(points.shape[0])
            points = points[order]
            labels = labels[order]

            sample_path = out_dir / "segmentation" / split / f"{record.model_id}.npz"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                sample_path,
                points=points.astype(np.float32),
                labels=labels.astype(np.int64),
                category=record.category,
                object_id=record.object_id,
            )
            manifest[split].append({
                "path": str(sample_path.relative_to(out_dir)),
                "category": record.category,
                "object_id": record.object_id,
                "model_id": record.model_id,
            })

    with (out_dir / "segmentation" / "manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=2)


def write_readme(out_dir: Path, categories: Iterable[dict[str, object]], args: argparse.Namespace) -> None:
    lines = [
        "# ToolMesh-mini",
        "",
        "Object-level dataset derived from source industrial tool meshes.",
        "",
        "Completion target: single complete object mesh from `models/obj_*.ply`.",
        "Completion partial input: object-level visible-view crops sampled from the same object.",
        "The default profile is the stable crop-based ToolMesh-mini v1 split.",
        "Optional full-view and thin-surface modes are experimental and disabled unless their probabilities are set explicitly.",
        "Affordance segmentation labels: `1=handle`, `0=not_handle` from `models_parts` meshes.",
        "",
        "Scene folders are not used to build completion samples.",
        "Training completion samples are augmented object observations, not additional CAD identities.",
        "",
        "## Categories",
    ]
    for category in categories:
        lines.append(
            f"- {category['taxonomy_name']}: "
            f"train={len(category['train'])}, val={len(category['val'])}, test={len(category['test'])}"
        )
    lines.extend([
        "",
        "## Build Parameters",
        f"- complete points: {args.n_complete}",
        f"- partial points: {args.n_partial}",
        f"- virtual view pool: {args.n_views}",
        f"- target train completion samples per category: {args.target_train_samples_per_category}",
        f"- eval partials per object: {args.eval_partials_per_object}",
        f"- visible ratio range: {args.visible_ratio_min}-{args.visible_ratio_max}",
        f"- crop ratio range: {args.crop_ratio_min}-{args.crop_ratio_max}",
        f"- full visible view probability: {args.full_view_prob}",
        f"- thin partial probability: {args.thin_partial_prob}",
        f"- thin visible ratio range: {args.thin_visible_ratio_min}-{args.thin_visible_ratio_max}",
        f"- thin crop ratio range: {args.thin_crop_ratio_min}-{args.thin_crop_ratio_max}",
        f"- mask hole probability: {args.mask_hole_prob}",
        f"- partial jitter std: {args.partial_jitter_std}",
        f"- segmentation points: {args.n_seg_points}",
        f"- seed: {args.seed}",
    ])
    with (out_dir / "README.md").open("w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh-root", type=Path, default=Path("tool_meshes"))
    parser.add_argument("--out-dir", type=Path, default=Path("src/datasets/ToolMesh-mini"))
    parser.add_argument("--n-complete", type=int, default=16384)
    parser.add_argument("--n-partial", type=int, default=2048)
    parser.add_argument("--n-views", type=int, default=24)
    parser.add_argument("--target-train-samples-per-category", type=int, default=1000)
    parser.add_argument("--eval-partials-per-object", type=int, default=4)
    parser.add_argument("--visible-ratio-min", type=float, default=0.38)
    parser.add_argument("--visible-ratio-max", type=float, default=0.70)
    parser.add_argument("--crop-ratio-min", type=float, default=0.32)
    parser.add_argument("--crop-ratio-max", type=float, default=0.90)
    parser.add_argument("--full-view-prob", type=float, default=0.0)
    parser.add_argument("--thin-partial-prob", type=float, default=0.0)
    parser.add_argument("--thin-visible-ratio-min", type=float, default=0.04)
    parser.add_argument("--thin-visible-ratio-max", type=float, default=0.24)
    parser.add_argument("--thin-crop-ratio-min", type=float, default=0.12)
    parser.add_argument("--thin-crop-ratio-max", type=float, default=0.48)
    parser.add_argument("--mask-hole-prob", type=float, default=0.0)
    parser.add_argument("--partial-jitter-std", type=float, default=0.002)
    parser.add_argument("--n-seg-points", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.visible_ratio_min <= 0 or args.visible_ratio_max > 1 or args.visible_ratio_min > args.visible_ratio_max:
        raise SystemExit("--visible-ratio-min/max must satisfy 0 < min <= max <= 1")
    if args.crop_ratio_min <= 0 or args.crop_ratio_max > 1 or args.crop_ratio_min > args.crop_ratio_max:
        raise SystemExit("--crop-ratio-min/max must satisfy 0 < min <= max <= 1")
    if args.thin_visible_ratio_min <= 0 or args.thin_visible_ratio_max > 1 or args.thin_visible_ratio_min > args.thin_visible_ratio_max:
        raise SystemExit("--thin-visible-ratio-min/max must satisfy 0 < min <= max <= 1")
    if args.thin_crop_ratio_min <= 0 or args.thin_crop_ratio_max > 1 or args.thin_crop_ratio_min > args.thin_crop_ratio_max:
        raise SystemExit("--thin-crop-ratio-min/max must satisfy 0 < min <= max <= 1")
    if not 0 <= args.full_view_prob <= 1:
        raise SystemExit("--full-view-prob must satisfy 0 <= value <= 1")
    if not 0 <= args.thin_partial_prob <= 1:
        raise SystemExit("--thin-partial-prob must satisfy 0 <= value <= 1")
    if args.full_view_prob + args.thin_partial_prob > 1:
        raise SystemExit("--full-view-prob + --thin-partial-prob must be <= 1")
    if not 0 <= args.mask_hole_prob <= 1:
        raise SystemExit("--mask-hole-prob must satisfy 0 <= value <= 1")
    if args.target_train_samples_per_category < 1:
        raise SystemExit("--target-train-samples-per-category must be positive")
    if args.eval_partials_per_object < 1:
        raise SystemExit("--eval-partials-per-object must be positive")
    return args


def main() -> None:
    args = parse_args()
    records = discover_objects(args.mesh_root)
    if not records:
        raise SystemExit(f"No source industrial tool objects found under {args.mesh_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    splits = split_records(records, args.seed)
    categories = build_completion(
        splits,
        args.out_dir,
        args.n_complete,
        args.n_partial,
        args.n_views,
        args.seed,
        args.target_train_samples_per_category,
        args.eval_partials_per_object,
        (args.visible_ratio_min, args.visible_ratio_max),
        (args.crop_ratio_min, args.crop_ratio_max),
        (args.thin_visible_ratio_min, args.thin_visible_ratio_max),
        (args.thin_crop_ratio_min, args.thin_crop_ratio_max),
        args.partial_jitter_std,
        args.full_view_prob,
        args.thin_partial_prob,
        args.mask_hole_prob,
    )
    build_segmentation(splits, args.out_dir, args.n_seg_points, args.seed)
    write_readme(args.out_dir, categories, args)
    print(f"Built ToolMesh-mini at {args.out_dir}")
    print(f"Objects: {len(records)}")
    print("Splits:", {name: len(items) for name, items in splits.items()})


if __name__ == "__main__":
    main()
