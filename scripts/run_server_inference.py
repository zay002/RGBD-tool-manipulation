#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, TextIO

import cv2
import numpy as np

from sim_pipeline.completion_inference.geometric_fallback import complete_by_bbox_symmetry
from sim_pipeline.evaluation.simple_eval import summarize_grasp_set
from sim_pipeline.grasp_generation.geometric_parallel_jaw import GraspCandidate
from sim_pipeline.grasp_generation.geometric_parallel_jaw import generate_topdown_grasps
from sim_pipeline.grasp_scoring.simple_collision import score_grasps


REQUIRED_FILES = (
    "request.json",
    "rgb.png",
    "depth_uint16.png",
    "intrinsics.json",
    "tf_base_camera.json",
)


class Intrinsics:
    def __init__(
        self,
        width: int,
        height: int,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        depth_scale_m: float,
    ) -> None:
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.depth_scale_m = depth_scale_m


def _load_request(request_dir: Path) -> dict[str, Any]:
    missing = [name for name in REQUIRED_FILES if not (request_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Request directory {request_dir} is missing required files: {', '.join(missing)}"
        )
    return json.loads((request_dir / "request.json").read_text(encoding="utf-8"))


def _load_intrinsics(request_dir: Path) -> Intrinsics:
    raw = json.loads((request_dir / "intrinsics.json").read_text(encoding="utf-8"))
    return Intrinsics(
        width=int(raw["width"]),
        height=int(raw["height"]),
        fx=float(raw["fx"]),
        fy=float(raw["fy"]),
        cx=float(raw["cx"]),
        cy=float(raw["cy"]),
        depth_scale_m=float(raw["depth_scale_m"]),
    )


def _rejected_result(run_id: str, reason: str) -> dict[str, Any]:
    return {
        "status": "rejected",
        "run_id": run_id,
        "target_box_xyxy": [0, 0, 0, 0],
        "mask_area_px": 0,
        "partial_points": 0,
        "completion_status": "not_run",
        "affordance_status": "not_run",
        "grasp_pose_camera": {
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "approach_pose_camera": {
            "translation_m": [0.0, 0.0, 0.0],
            "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "quality_gates": {
            "detection_ok": False,
            "mask_ok": False,
            "depth_ok": False,
            "completion_ok": False,
            "affordance_ok": False,
        },
        "reject_reason": reason,
    }


def _write_ply(path: Path, points: np.ndarray, color: tuple[int, int, int] = (180, 188, 196)) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    with path.open("w", encoding="ascii", newline="\n") as file:
        file.write("ply\nformat ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\nproperty float y\nproperty float z\n")
        file.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for x, y, z in points:
            file.write(f"{x:.6f} {y:.6f} {z:.6f} {color[0]} {color[1]} {color[2]}\n")
    return int(len(points))


def _write_pfm(path: Path, image: np.ndarray) -> None:
    data = np.asarray(image, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("PFM depth image must be HxW")
    with path.open("wb") as file:
        file.write(b"Pf\n")
        file.write(f"{data.shape[1]} {data.shape[0]}\n".encode("ascii"))
        file.write(b"-1.0\n")
        np.flipud(data).astype("<f4").tofile(file)


def _depth_to_points(depth_uint16: np.ndarray, mask: np.ndarray, intr: Intrinsics) -> np.ndarray:
    depth_m = depth_uint16.astype(np.float32) * float(intr.depth_scale_m)
    valid = mask & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 2.5)
    v, u = np.nonzero(valid)
    z = depth_m[v, u]
    x = (u.astype(np.float32) - intr.cx) * z / intr.fx
    y = (v.astype(np.float32) - intr.cy) * z / intr.fy
    return np.column_stack([x, y, z]).astype(np.float32)


def _points_to_image_mask(
    points_camera: np.ndarray,
    intr: Intrinsics,
    shape: tuple[int, int],
    dilate_iterations: int = 2,
) -> np.ndarray:
    points = np.asarray(points_camera, dtype=np.float32).reshape(-1, 3)
    mask = np.zeros(shape, dtype=np.uint8)
    if len(points) == 0:
        return mask.astype(bool)
    z = points[:, 2]
    valid = np.isfinite(z) & (z > 0.05) & (z < 2.5)
    if not np.any(valid):
        return mask.astype(bool)
    pts = points[valid]
    u = np.rint(pts[:, 0] * intr.fx / pts[:, 2] + intr.cx).astype(np.int32)
    v = np.rint(pts[:, 1] * intr.fy / pts[:, 2] + intr.cy).astype(np.int32)
    inside = (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    mask[v[inside], u[inside]] = 255
    if dilate_iterations > 0 and int(mask.sum()) > 0:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=dilate_iterations)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask > 0


def _largest_point_component(points: np.ndarray, radius_m: float = 0.018, min_points: int = 32) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    if len(pts) < min_points:
        return pts
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(pts)
        visited = np.zeros(len(pts), dtype=bool)
        best: list[int] = []
        for start in range(len(pts)):
            if visited[start]:
                continue
            stack = [start]
            visited[start] = True
            component: list[int] = []
            while stack:
                idx = stack.pop()
                component.append(idx)
                for nb in tree.query_ball_point(pts[idx], r=radius_m):
                    if not visited[nb]:
                        visited[nb] = True
                        stack.append(int(nb))
            if len(component) > len(best):
                best = component
        if len(best) >= min_points:
            return pts[np.asarray(best, dtype=np.int64)]
        return pts
    except Exception:
        return pts


def _write_contact_graspnet_input(
    path: Path,
    rgb_bgr: np.ndarray,
    depth_uint16: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
) -> None:
    depth_m = depth_uint16.astype(np.float32) * float(intr.depth_scale_m)
    segmap = np.zeros(depth_uint16.shape, dtype=np.uint8)
    segmap[mask] = 1
    camera_k = np.array(
        [[intr.fx, 0.0, intr.cx], [0.0, intr.fy, intr.cy], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    np.savez(path, depth=depth_m, K=camera_k, segmap=segmap, rgb=rgb_bgr)


def _try_contact_graspnet_pytorch(
    output_dir: Path,
    rgb_bgr: np.ndarray,
    depth_uint16: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
    timeout_s: float = 30.0,
) -> dict[str, Any] | None:
    project_root = Path(__file__).resolve().parents[2]
    repo = Path(os.environ.get("CONTACT_GRASPNET_ROOT", project_root / "src" / "external" / "contact_graspnet_pytorch"))
    helper = project_root / "scripts" / "run_contact_graspnet_pytorch.py"
    python_exe = os.environ.get(
        "CONTACT_GRASPNET_PYTHON",
        "python",
    )
    if not repo.is_dir() or not helper.is_file():
        return None
    contact_dir = output_dir / "contact_graspnet"
    contact_dir.mkdir(parents=True, exist_ok=True)
    input_npz = contact_dir / "input_depth_seg.npz"
    output_json = contact_dir / "result.json"
    _write_contact_graspnet_input(input_npz, rgb_bgr, depth_uint16, mask, intr)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo}{os.pathsep}{env.get('PYTHONPATH', '')}"
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    try:
        completed = subprocess.run(
            [
                python_exe,
                str(helper),
                str(input_npz),
                str(output_json),
                "--repo",
                str(repo),
                "--first-thres",
                os.environ.get("CONTACT_GRASPNET_FIRST_THRES", "0.0"),
                "--second-thres",
                os.environ.get("CONTACT_GRASPNET_SECOND_THRES", "0.0"),
                "--filter-thres",
                os.environ.get("CONTACT_GRASPNET_FILTER_THRES", "0.001"),
            ],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        (contact_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return None
    (contact_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (contact_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    if not output_json.is_file():
        return None
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    if payload.get("status") != "ok" or not payload.get("best"):
        return None
    return payload


def _latest_checkpoint(project_root: Path, pattern: str) -> Path | None:
    candidates = sorted(project_root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def _resolve_seedformer_checkpoint(project_root: Path) -> Path:
    override = os.environ.get("SEEDFORMER_CHECKPOINT")
    if override:
        return Path(override)
    for candidate in (
        _latest_checkpoint(project_root, "weights/completion/*/ckpt-best.pth"),
        project_root / "weights/completion/ckpt-best.pth",
    ):
        if candidate is not None and candidate.is_file():
            return candidate
    return project_root / "missing_seedformer_checkpoint.pth"


def _resolve_dgcnn_checkpoint(project_root: Path) -> Path:
    override = os.environ.get("DGCNN_CHECKPOINT")
    if override:
        return Path(override)
    for candidate in (
        _latest_checkpoint(project_root, "weights/affordance/*/ckpt-best.pth"),
        project_root / "weights/affordance/ckpt-best.pth",
    ):
        if candidate is not None and candidate.is_file():
            return candidate
    return project_root / "missing_dgcnn_checkpoint.pth"


def _try_completion_affordance(
    output_dir: Path,
    depth_uint16: np.ndarray,
    mask: np.ndarray,
    intr: Intrinsics,
    object_category: str | None = None,
    timeout_s: float = 120.0,
) -> dict[str, Any] | None:
    project_root = Path(__file__).resolve().parents[2]
    helper = project_root / "scripts" / "run_completion_affordance.py"
    python_exe = os.environ.get(
        "FULL_PIPELINE_PYTHON",
        "python",
    )
    seedformer_checkpoint = _resolve_seedformer_checkpoint(project_root)
    dgcnn_checkpoint = _resolve_dgcnn_checkpoint(project_root)
    if not helper.is_file():
        return None

    full_dir = output_dir / "completion_dgcnn"
    full_dir.mkdir(parents=True, exist_ok=True)
    depth_pfm = full_dir / "depth_m.pfm"
    mask_path = full_dir / "mask.png"
    _write_pfm(depth_pfm, depth_uint16.astype(np.float32) * float(intr.depth_scale_m))
    cv2.imwrite(str(mask_path), (mask.astype(np.uint8) * 255))
    hfov = float(2.0 * np.arctan(float(intr.width) / (2.0 * float(intr.fx))))
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{project_root}{os.pathsep}{project_root / 'src'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    try:
        completed = subprocess.run(
            [
                python_exe,
                str(helper),
                "--depth-pfm",
                str(depth_pfm),
                "--mask",
                str(mask_path),
                "--out-dir",
                str(full_dir),
                "--seedformer-checkpoint",
                str(seedformer_checkpoint),
                "--dgcnn-checkpoint",
                str(dgcnn_checkpoint),
                "--backend-module",
                os.environ.get("COMPLETION_AFFORDANCE_BACKEND", ""),
                "--hfov",
                str(hfov),
                "--affordance-label",
                os.environ.get("DGCNN_AFFORDANCE_LABEL", "1"),
                "--object-category",
                object_category or "",
                "--device",
                os.environ.get("FULL_PIPELINE_DEVICE", "cuda:0"),
            ],
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        (full_dir / "error.txt").write_text(str(exc), encoding="utf-8")
        return None
    (full_dir / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
    (full_dir / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
    completed_path = full_dir / "completed_camera.npy"
    labels_path = full_dir / "completed_affordance_labels.npy"
    partial_path = full_dir / "partial_camera_all.npy"
    partial_labels_path = full_dir / "partial_affordance_labels.npy"
    summary_path = full_dir / "completion_segmentation_summary.json"
    if completed.returncode != 0 or not completed_path.is_file() or not labels_path.is_file():
        return None
    completed_camera = np.load(completed_path).astype(np.float32)
    completed_labels = np.load(labels_path).astype(np.int64)
    partial_camera = np.load(partial_path).astype(np.float32) if partial_path.is_file() else np.zeros((0, 3), dtype=np.float32)
    partial_labels = (
        np.load(partial_labels_path).astype(np.int64)
        if partial_labels_path.is_file()
        else np.zeros((len(partial_camera),), dtype=np.int64)
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    return {
        "output_dir": str(full_dir),
        "completed_camera": completed_camera,
        "completed_labels": completed_labels,
        "partial_camera": partial_camera,
        "partial_labels": partial_labels,
        "summary": summary,
        "seedformer_checkpoint": str(seedformer_checkpoint),
        "dgcnn_checkpoint": str(dgcnn_checkpoint),
    }


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return mask.astype(bool)
    areas = stats[1:, cv2.CC_STAT_AREA]
    label = int(np.argmax(areas) + 1)
    return labels == label


def _parse_detector_class(query: str, target: str) -> str:
    text = f"{query} {target}".lower()
    for term in ("screwdriver", "scissors", "wrench", "pliers", "hammer", "drill", "screw", "nut", "bolt"):
        if term in text:
            return term
    return target or "object"


def _parse_affordance_category(query: str, target: str, detector_class: str) -> str:
    text = f"{query} {target}".lower()
    if "wrench" in text:
        if "combo" in text or "combination" in text or "combinational" in text:
            return "combo wrench"
        if "adjustable" in text or "spanner" in text:
            return "adjustable wrench"
    if "ratchet" in text:
        return "ratchet"
    if "locking" in text and ("plier" in text or "pliers" in text):
        return "locking pliers"
    return detector_class


def _parse_yoloworld_candidate_classes(detector_class: str) -> list[str]:
    raw = os.environ.get(
        "YOLOWORLD_CANDIDATE_CLASSES",
        "screwdriver,scissors,wrench,pliers,hammer,drill,screw,nut,bolt",
    )
    classes: list[str] = []
    for item in raw.split(","):
        item = item.strip()
        if item and item not in classes:
            classes.append(item)
    if detector_class not in classes:
        classes.insert(0, detector_class)
    return classes or [detector_class]


def _allow_nontarget_yoloworld_detection() -> bool:
    return os.environ.get("YOLOWORLD_ALLOW_NONTARGET", "").strip().lower() in {"1", "true", "yes", "on"}


def _target_prompt_variants(detector_class: str) -> list[str]:
    base = detector_class.strip().lower() or "object"
    variants = {
        "screwdriver": [
            "screwdriver",
            "black screwdriver",
            "red screwdriver",
            "Phillips screwdriver",
            "flathead screwdriver",
        ],
        "wrench": [
            "wrench",
            "spanner",
            "adjustable wrench",
            "open-end wrench",
            "metal wrench",
        ],
        "scissors": ["scissors", "shears", "metal scissors"],
        "pliers": ["pliers", "long nose pliers", "wire pliers"],
        "hammer": ["hammer", "claw hammer"],
        "drill": ["drill", "electric drill", "power drill"],
        "screw": ["screw", "metal screw", "bolt-like screw"],
        "nut": ["nut", "hex nut", "metal nut"],
        "bolt": ["bolt", "metal bolt", "hex bolt"],
    }
    prompts = variants.get(base, [base])
    return list(dict.fromkeys([item for item in prompts if item]))


def _box_area_ratio(box: np.ndarray, width: int, height: int) -> float:
    x1, y1, x2, y2 = [float(v) for v in box]
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / max(float(width * height), 1.0)


def _select_yolo_box(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int,
    height: int,
    detector_class: str,
) -> int:
    del boxes, width, height, detector_class
    return int(np.argmax(scores))


def _run_yoloworld_group(
    yolo_cls: Any,
    yolo_weights: Path,
    rgb_path: Path,
    classes: list[str],
    conf: float,
    imgsz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    yolo = yolo_cls(str(yolo_weights))
    yolo.set_classes(classes)
    det = yolo(str(rgb_path), conf=conf, imgsz=imgsz, verbose=False)[0]
    if det.boxes is None or len(det.boxes) == 0:
        return None
    boxes = det.boxes.xyxy.detach().cpu().numpy()
    scores = det.boxes.conf.detach().cpu().numpy()
    class_ids = det.boxes.cls.detach().cpu().numpy().astype(np.int64)
    return boxes, scores, class_ids


def _candidate_debug_rows(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    classes: list[str],
    width: int,
    height: int,
) -> list[dict[str, Any]]:
    rows = []
    order = np.argsort(-scores)[:50]
    for idx in order:
        class_id = int(class_ids[idx])
        rows.append(
            {
                "class": classes[class_id] if 0 <= class_id < len(classes) else str(class_id),
                "class_id": class_id,
                "score": float(scores[idx]),
                "box": [float(v) for v in boxes[idx].tolist()],
                "box_area_ratio": _box_area_ratio(boxes[idx], width, height),
            }
        )
    return rows


def _enhance_for_yoloworld(rgb_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_l = clahe.apply(l_channel)
    enhanced = cv2.merge((enhanced_l, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
    blurred = cv2.GaussianBlur(enhanced_bgr, (0, 0), 1.0)
    return cv2.addWeighted(enhanced_bgr, 1.25, blurred, -0.25, 0)


def _parse_crop_box(value: str, width: int, height: int) -> tuple[int, int, int, int] | None:
    try:
        parts = [float(item.strip()) for item in value.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    if all(0.0 <= item <= 1.0 for item in parts):
        x1, y1, x2, y2 = [int(round(v)) for v in (parts[0] * width, parts[1] * height, parts[2] * width, parts[3] * height)]
    else:
        x1, y1, x2, y2 = [int(round(v)) for v in parts]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return x1, y1, x2, y2


def _prepare_yoloworld_views(
    rgb_path: Path,
    rgb_bgr: np.ndarray,
    perception_dir: Path,
) -> list[dict[str, Any]]:
    height, width = rgb_bgr.shape[:2]
    requested = {
        item.strip().lower()
        for item in os.environ.get("YOLOWORLD_PREPROCESS_VIEWS", "original,clahe").split(",")
        if item.strip()
    }
    views: list[dict[str, Any]] = [
        {
            "name": "original",
            "path": rgb_path,
            "offset_xy": [0.0, 0.0],
            "size": [width, height],
        }
    ]
    if "clahe" in requested:
        enhanced_path = perception_dir / "yoloworld_preprocess_clahe.png"
        cv2.imwrite(str(enhanced_path), _enhance_for_yoloworld(rgb_bgr))
        views.append(
            {
                "name": "clahe",
                "path": enhanced_path,
                "offset_xy": [0.0, 0.0],
                "size": [width, height],
            }
        )
    crop = _parse_crop_box(os.environ.get("YOLOWORLD_WORKSPACE_CROP", ""), width, height)
    if crop is not None:
        x1, y1, x2, y2 = crop
        crop_path = perception_dir / "yoloworld_preprocess_workspace_crop.png"
        cv2.imwrite(str(crop_path), rgb_bgr[y1:y2, x1:x2])
        views.append(
            {
                "name": "workspace_crop",
                "path": crop_path,
                "offset_xy": [float(x1), float(y1)],
                "size": [int(x2 - x1), int(y2 - y1)],
            }
        )
    return views


def _map_boxes_to_original(boxes: np.ndarray, view: dict[str, Any]) -> np.ndarray:
    mapped = boxes.copy()
    offset_x, offset_y = [float(v) for v in view.get("offset_xy", [0.0, 0.0])]
    mapped[:, [0, 2]] += offset_x
    mapped[:, [1, 3]] += offset_y
    return mapped


def _sam_prompt_points(detector_class: str, box: list[float]) -> tuple[list[list[float]], list[int]] | None:
    if os.environ.get("SAM_BOX_WITH_POINTS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        return None
    if detector_class.lower() not in {"screwdriver", "wrench", "hammer", "drill"}:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(1.0, x2 - x1)
    h = max(1.0, y2 - y1)
    points: list[list[float]] = []
    labels: list[int] = []
    if h >= w:
        x = 0.5 * (x1 + x2)
        for ratio in (0.35, 0.55, 0.76):
            points.append([x, y1 + ratio * h])
            labels.append(1)
    else:
        y = 0.5 * (y1 + y2)
        for ratio in (0.30, 0.50, 0.72):
            points.append([x1 + ratio * w, y])
            labels.append(1)
    for px, py in (
        (x1 + 0.08 * w, y1 + 0.08 * h),
        (x2 - 0.08 * w, y1 + 0.08 * h),
        (x1 + 0.08 * w, y2 - 0.08 * h),
        (x2 - 0.08 * w, y2 - 0.08 * h),
    ):
        points.append([px, py])
        labels.append(0)
    return points, labels


def _try_yoloworld_sam_mask(
    request_dir: Path,
    output_dir: Path,
    rgb_bgr: np.ndarray,
    request: dict[str, Any],
) -> tuple[np.ndarray, list[int], dict[str, Any]] | None:
    try:
        from ultralytics import SAM, YOLO
    except Exception as exc:
        (output_dir / "perception_error.txt").write_text(
            f"ultralytics import failed: {exc}",
            encoding="utf-8",
        )
        return None

    project_root = Path(__file__).resolve().parents[2]
    yolo_weights = Path(os.environ.get("YOLOWORLD_WEIGHTS", project_root / "yolov8s-worldv2.pt"))
    sam_weights = Path(os.environ.get("SAM_WEIGHTS", project_root / "sam_b.pt"))
    if not yolo_weights.is_file() or not sam_weights.is_file():
        return None

    detector_class = _parse_detector_class(
        str(request.get("query") or ""),
        str(request.get("target") or ""),
    )
    candidate_classes = _parse_yoloworld_candidate_classes(detector_class)
    perception_dir = output_dir / "perception"
    perception_dir.mkdir(parents=True, exist_ok=True)
    rgb_path = request_dir / "rgb.png"
    try:
        conf = float(os.environ.get("YOLOWORLD_CONF", "0.0005"))
        imgsz = int(os.environ.get("YOLOWORLD_IMGSZ", "960"))
        detection_views = _prepare_yoloworld_views(rgb_path, rgb_bgr, perception_dir)
        prompt_groups: list[tuple[str, list[str], bool]] = [
            ("target_class", [detector_class], False),
            ("target_phrases", _target_prompt_variants(detector_class), False),
        ]
        if os.environ.get("YOLOWORLD_USE_CANDIDATE_FALLBACK", "1").strip().lower() in {"1", "true", "yes", "on"}:
            prompt_groups.append(("candidate_classes", candidate_classes, True))

        group_debug: dict[str, Any] = {}
        selected_group = ""
        selected_classes: list[str] = []
        selected_view_names: list[str] = []
        boxes = scores = class_ids = None
        num_detector_boxes = 0
        for group_name, classes, filter_to_target in prompt_groups:
            view_rows = []
            all_boxes = []
            all_scores = []
            all_class_ids = []
            all_view_names: list[str] = []
            for view in detection_views:
                result = _run_yoloworld_group(YOLO, yolo_weights, view["path"], classes, conf, imgsz)
                if result is None:
                    view_rows.append({"view": view["name"], "num_boxes": 0})
                    continue
                view_boxes, view_scores, view_class_ids = result
                mapped_boxes = _map_boxes_to_original(view_boxes, view)
                all_boxes.append(mapped_boxes)
                all_scores.append(view_scores)
                all_class_ids.append(view_class_ids)
                all_view_names.extend([str(view["name"])] * len(mapped_boxes))
                view_rows.append(
                    {
                        "view": view["name"],
                        "num_boxes": int(len(mapped_boxes)),
                        "top_candidates": _candidate_debug_rows(
                            mapped_boxes,
                            view_scores,
                            view_class_ids,
                            classes,
                            rgb_bgr.shape[1],
                            rgb_bgr.shape[0],
                        )[:8],
                    }
                )
            if not all_boxes:
                group_debug[group_name] = {"classes": classes, "views": view_rows, "num_boxes": 0}
                continue
            group_boxes = np.concatenate(all_boxes, axis=0)
            group_scores = np.concatenate(all_scores, axis=0)
            group_class_ids = np.concatenate(all_class_ids, axis=0)
            group_view_names = all_view_names
            num_raw = int(len(group_boxes))
            group_debug[group_name] = {
                "classes": classes,
                "views": view_rows,
                "num_boxes": num_raw,
                "top_candidates": _candidate_debug_rows(
                    group_boxes,
                    group_scores,
                    group_class_ids,
                    classes,
                    rgb_bgr.shape[1],
                    rgb_bgr.shape[0],
                )[:12],
            }
            if filter_to_target and len(classes) > 1 and not _allow_nontarget_yoloworld_detection():
                target_ids = [idx for idx, name in enumerate(classes) if name == detector_class]
                target_mask = np.isin(group_class_ids, target_ids)
                if not target_mask.any():
                    observed = sorted({classes[int(idx)] for idx in group_class_ids if 0 <= int(idx) < len(classes)})
                    group_debug[group_name]["observed_candidate_classes"] = observed
                    group_debug[group_name]["reason"] = "no_target_class_box"
                    continue
                group_boxes = group_boxes[target_mask]
                group_scores = group_scores[target_mask]
                group_class_ids = group_class_ids[target_mask]
                group_view_names = [name for name, keep in zip(group_view_names, target_mask.tolist()) if keep]
            boxes, scores, class_ids = group_boxes, group_scores, group_class_ids
            num_detector_boxes = num_raw
            selected_group = group_name
            selected_classes = classes
            selected_view_names = group_view_names
            break
        (perception_dir / "yoloworld_candidate_debug.json").write_text(
            json.dumps(group_debug, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        if boxes is None or scores is None or class_ids is None or len(boxes) == 0:
            return None

        best_idx = _select_yolo_box(boxes, scores, rgb_bgr.shape[1], rgb_bgr.shape[0], detector_class)
        best_box = [float(v) for v in boxes[best_idx].tolist()]
        best_score = float(scores[best_idx])
        selected_class_id = int(class_ids[best_idx])
        selected_view = selected_view_names[best_idx] if best_idx < len(selected_view_names) else "unknown"
        selected_class = (
            selected_classes[selected_class_id]
            if 0 <= selected_class_id < len(selected_classes)
            else detector_class
        )
        canonical_selected_class = detector_class if selected_group == "target_phrases" else selected_class

        sam = SAM(str(sam_weights))
        sam_points = _sam_prompt_points(detector_class, best_box)
        sam_mode = os.environ.get("SAM_PROMPT_MODE", "box_points").strip().lower()
        if sam_points is None:
            seg = sam(str(rgb_path), bboxes=[best_box], verbose=False)[0]
            sam_prompt: dict[str, Any] = {"box_xyxy": best_box}
        else:
            points, labels = sam_points
            if sam_mode == "points_only":
                seg = sam(str(rgb_path), points=points, labels=labels, verbose=False)[0]
                sam_prompt = {"points": points, "labels": labels, "mode": sam_mode}
            else:
                seg = sam(str(rgb_path), bboxes=[best_box], points=[points], labels=[labels], verbose=False)[0]
                sam_prompt = {
                    "box_xyxy": best_box,
                    "points": points,
                    "labels": labels,
                    "mode": "box_points",
                }
        if seg.masks is None or len(seg.masks.data) == 0:
            return None
        mask = seg.masks.data[0].detach().cpu().numpy() > 0.5
        mask = _largest_component(mask)
        if int(mask.sum()) < 500:
            return None

        box_i = [int(round(v)) for v in best_box]
        overlay = rgb_bgr.copy()
        overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([40, 210, 80], dtype=np.uint8)).astype(np.uint8)
        cv2.rectangle(overlay, (box_i[0], box_i[1]), (box_i[2], box_i[3]), (0, 255, 255), 2)
        cv2.imwrite(str(perception_dir / "yoloworld_sam_overlay.png"), overlay)
        cv2.imwrite(str(perception_dir / "yoloworld_sam_mask.png"), (mask.astype(np.uint8) * 255))
        payload = {
            "segmentation": "yoloworld_sam",
            "detector": "YOLOWorld",
            "detector_class": detector_class,
            "detector_prompt_group": selected_group,
            "detector_candidate_classes": candidate_classes,
            "selected_detector_prompts": selected_classes,
            "selected_detector_prompt": selected_class,
            "selected_detector_class": canonical_selected_class,
            "selected_detector_class_id": selected_class_id,
            "selected_detector_view": selected_view,
            "detector_confidence": best_score,
            "detector_box_xyxy": best_box,
            "detector_imgsz": imgsz,
            "detector_conf_threshold": conf,
            "detector_preprocess_views": [
                {
                    **view,
                    "path": str(view["path"]),
                }
                for view in detection_views
            ],
            "num_detector_boxes": num_detector_boxes,
            "num_target_detector_boxes": int(len(boxes)),
            "sam": "Ultralytics SAM",
            "sam_prompt": sam_prompt,
            "mask_pixels": int(mask.sum()),
            "weights": {
                "yolo": str(yolo_weights),
                "sam": str(sam_weights),
            },
        }
        (perception_dir / "wrist_perception_summary.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return mask.astype(bool), box_i, payload
    except Exception as exc:
        (perception_dir / "error.txt").write_text(f"{type(exc).__name__}: {exc}", encoding="utf-8")
        return None


def _segment_object_from_rgbd(rgb_bgr: np.ndarray, depth_uint16: np.ndarray, intr: Intrinsics) -> tuple[np.ndarray, str]:
    depth_m = depth_uint16.astype(np.float32) * float(intr.depth_scale_m)
    valid = (depth_uint16 > 0) & np.isfinite(depth_m) & (depth_m > 0.05) & (depth_m < 2.5)
    if int(valid.sum()) < 1000:
        return np.zeros(depth_uint16.shape, dtype=bool), "too_few_valid_depth_pixels"

    # In the wrist setup the target tool is closer to the camera than the tag board/table.
    plane_depth = float(np.percentile(depth_m[valid], 65.0))
    protrusion = valid & (depth_m < plane_depth - 0.018)

    hsv = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    not_black_tag = value > 55
    not_white_board = saturation > 18
    protrusion_component = _largest_component(protrusion)
    mask = protrusion & not_black_tag & not_white_board

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=2).astype(bool)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=1).astype(bool)
    mask = _largest_component(mask)

    if int(mask.sum()) < 500 or int(mask.sum()) < int(protrusion_component.sum()) * 0.45:
        # Synthetic tests and low-texture tools may not pass the color filter; use depth protrusion.
        mask = protrusion_component
        reason = "depth_protrusion_fallback"
    else:
        reason = "depth_protrusion_color_filtered"
    return mask.astype(bool), reason


def _box_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _quaternion_xyzw_from_axes(closing: np.ndarray, approach: np.ndarray) -> list[float]:
    x_axis = closing / max(float(np.linalg.norm(closing)), 1e-8)
    z_axis = approach / max(float(np.linalg.norm(approach)), 1e-8)
    y_axis = np.cross(z_axis, x_axis)
    y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-8)
    x_axis = np.cross(y_axis, z_axis)
    rotation = np.column_stack([x_axis, y_axis, z_axis]).astype(np.float64)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rotation)))
        if idx == 0:
            s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            qw = (rotation[2, 1] - rotation[1, 2]) / s
            qx = 0.25 * s
            qy = (rotation[0, 1] + rotation[1, 0]) / s
            qz = (rotation[0, 2] + rotation[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            qw = (rotation[0, 2] - rotation[2, 0]) / s
            qx = (rotation[0, 1] + rotation[1, 0]) / s
            qy = 0.25 * s
            qz = (rotation[1, 2] + rotation[2, 1]) / s
        else:
            s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            qw = (rotation[1, 0] - rotation[0, 1]) / s
            qx = (rotation[0, 2] + rotation[2, 0]) / s
            qy = (rotation[1, 2] + rotation[2, 1]) / s
            qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    quat /= max(float(np.linalg.norm(quat)), 1e-8)
    return [float(v) for v in quat]


def _pose_from_candidate(candidate: Any, z_offset_m: float = 0.0) -> dict[str, Any]:
    position = np.asarray(candidate.position, dtype=np.float32).copy()
    position[2] += float(z_offset_m)
    approach = np.asarray(candidate.approach, dtype=np.float32)
    closing = np.asarray(candidate.closing, dtype=np.float32)
    return {
        "translation_m": [float(v) for v in position],
        "quaternion_xyzw": _quaternion_xyzw_from_axes(closing, approach),
    }


def _contact_candidate_to_grasp_candidate(contact_payload: dict[str, Any]) -> GraspCandidate:
    best = contact_payload["best"]
    return GraspCandidate(
        position=best["grasp_pose_camera"]["translation_m"],
        approach=best["approach"],
        closing=best["closing"],
        width_m=float(best.get("width_m") or 0.08),
        score=float(best.get("score") or 0.0),
        collision_free=True,
    )


def _generate_pca_handle_grasps(points: np.ndarray, width_margin_m: float = 0.005) -> list[GraspCandidate]:
    if len(points) < 16:
        return []
    xy = points[:, :2].astype(np.float32)
    center_xy = xy.mean(axis=0)
    centered = xy - center_xy
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    projected = centered @ vectors
    extents = np.maximum(np.ptp(projected, axis=0), 1e-4)
    center = points.mean(axis=0).astype(np.float32)
    center[:2] = center_xy
    # RealSense optical-frame z is depth, not height. The top-down simulator uses
    # z-up and adds clearance; here a larger z moves behind the object and can
    # transform below the table. Use the object's far-side surface as the grasp
    # center depth, while approach poses subtract depth to move upward/toward the camera.
    center[2] = float(points[:, 2].max() + 0.0005)
    approach = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    candidates: list[GraspCandidate] = []
    for sign in (1.0, -1.0):
        closing_xy = vectors[:, 1] * sign
        closing = np.array([closing_xy[0], closing_xy[1], 0.0], dtype=np.float32)
        candidates.append(
            GraspCandidate(
                position=[float(v) for v in center],
                approach=approach.tolist(),
                closing=closing.tolist(),
                width_m=float(extents[1] + width_margin_m),
            )
        )
    return candidates


def _select_handle_midsection(points: np.ndarray) -> np.ndarray:
    if len(points) < 32:
        return points
    xy = points[:, :2].astype(np.float32)
    center = xy.mean(axis=0)
    covariance = np.cov((xy - center).T)
    values, vectors = np.linalg.eigh(covariance)
    long_axis = vectors[:, int(np.argmax(values))]
    projected = (xy - center) @ long_axis
    lo, hi = np.quantile(projected, [0.20, 0.80])
    selected = points[(projected >= lo) & (projected <= hi)]
    return selected if len(selected) >= 32 else points


def _select_handle_affordance_mask(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    if len(xs) < 64:
        return mask.astype(bool)
    pixels = np.column_stack([xs, ys]).astype(np.float32)
    center = pixels.mean(axis=0)
    covariance = np.cov((pixels - center).T)
    values, vectors = np.linalg.eigh(covariance)
    vectors = vectors[:, np.argsort(values)[::-1]]
    candidates: list[tuple[float, np.ndarray]] = []
    for axis_idx in (0, 1):
        long_axis = vectors[:, axis_idx]
        cross_axis = vectors[:, 1 - axis_idx]
        long_coord = (pixels - center) @ long_axis
        cross_coord = (pixels - center) @ cross_axis
        bins = np.linspace(float(long_coord.min()), float(long_coord.max()), 36)
        selected = np.zeros(len(pixels), dtype=bool)
        widths: list[float] = []
        members: list[np.ndarray] = []
        for low, high in zip(bins[:-1], bins[1:]):
            member = (long_coord >= low) & (long_coord < high)
            if int(member.sum()) < 25:
                continue
            widths.append(float(np.ptp(cross_coord[member])))
            members.append(member)
        if not widths:
            continue
        width_ref = float(np.percentile(widths, 35.0))
        for width, member in zip(widths, members):
            if width <= max(width_ref * 1.20, width_ref + 8.0):
                selected |= member
        if int(selected.sum()) < 200:
            continue
        sel_long = long_coord[selected]
        sel_cross = cross_coord[selected]
        length = float(np.ptp(sel_long))
        width = max(float(np.ptp(sel_cross)), 1.0)
        aspect = length / width
        score = float(selected.sum()) * aspect / width
        candidates.append((score, selected))

    if not candidates:
        return mask.astype(bool)
    selected = max(candidates, key=lambda item: item[0])[1]
    out = np.zeros(mask.shape, dtype=bool)
    chosen = pixels[selected].astype(np.int32)
    out[chosen[:, 1], chosen[:, 0]] = True
    kernel = np.ones((5, 5), np.uint8)
    out = cv2.morphologyEx(out.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=1).astype(bool)
    return _largest_component(out)


def _save_overlays(
    output_dir: Path,
    rgb_bgr: np.ndarray,
    mask: np.ndarray,
    box: list[int],
    affordance_mask: np.ndarray | None = None,
) -> None:
    overlays = output_dir / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    overlay = rgb_bgr.copy()
    overlay[mask] = (0.55 * overlay[mask] + 0.45 * np.array([40, 210, 80], dtype=np.uint8)).astype(np.uint8)
    if affordance_mask is not None:
        overlay[affordance_mask] = (
            0.45 * overlay[affordance_mask] + 0.55 * np.array([30, 60, 255], dtype=np.uint8)
        ).astype(np.uint8)
    x0, y0, x1, y1 = box
    if x1 > x0 and y1 > y0:
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 255), 2)
    cv2.imwrite(str(overlays / "01_rgb_detection.png"), overlay)
    cv2.imwrite(str(overlays / "02_mask.png"), (mask.astype(np.uint8) * 255))
    masked = rgb_bgr.copy()
    masked[~mask] = (0.25 * masked[~mask]).astype(np.uint8)
    cv2.imwrite(str(overlays / "03_masked_depth.png"), masked)
    cv2.imwrite(str(overlays / "06_affordance_completed.png"), overlay)
    cv2.imwrite(str(overlays / "07_affordance_visible.png"), overlay)


def run_contract_entrypoint(request_dir: Path, output_dir: Path) -> Path:
    request = _load_request(request_dir)
    run_id = str(request.get("run_id") or request_dir.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(exist_ok=True)
    (output_dir / "clouds").mkdir(exist_ok=True)

    rgb_bgr = cv2.imread(str(request_dir / "rgb.png"), cv2.IMREAD_COLOR)
    depth_uint16 = cv2.imread(str(request_dir / "depth_uint16.png"), cv2.IMREAD_UNCHANGED)
    if rgb_bgr is None or depth_uint16 is None:
        result = _rejected_result(run_id, "failed_to_read_rgb_or_depth")
    else:
        intr = _load_intrinsics(request_dir)
        perception_payload: dict[str, Any] | None = None
        mask_override_path = request_dir / "mask_override.png"
        detector_category = _parse_detector_class(
            str(request.get("instruction") or request.get("query") or ""),
            str(request.get("target") or ""),
        )
        object_category = _parse_affordance_category(
            str(request.get("instruction") or request.get("query") or ""),
            str(request.get("target") or ""),
            detector_category,
        )
        if mask_override_path.is_file():
            override = cv2.imread(str(mask_override_path), cv2.IMREAD_GRAYSCALE)
            if override is None or override.shape != depth_uint16.shape:
                result = _rejected_result(run_id, "invalid_mask_override")
                result_path = output_dir / "result.json"
                result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
                return result_path
            mask = override > 127
            segmentation_backend = "manual_mask_override"
            box = _box_from_mask(mask)
        else:
            yolo_sam = _try_yoloworld_sam_mask(
                request_dir,
                output_dir,
                rgb_bgr,
                request,
            )
            if yolo_sam is not None:
                mask, box, perception_payload = yolo_sam
                segmentation_backend = "yoloworld_sam"
                if object_category in {"", "object"}:
                    object_category = str(
                        perception_payload.get("selected_detector_class")
                        or perception_payload.get("detector_class")
                        or ""
                    )
            else:
                mask, segmentation_backend = _segment_object_from_rgbd(rgb_bgr, depth_uint16, intr)
                box = _box_from_mask(mask)
        affordance_mask = _select_handle_affordance_mask(mask)
        partial = _depth_to_points(depth_uint16, mask, intr)
        if len(partial) < 800:
            result = _rejected_result(run_id, f"too_few_partial_points:{len(partial)}")
            result["target_box_xyxy"] = box
            result["mask_area_px"] = int(mask.sum())
            result["partial_points"] = int(len(partial))
        else:
            full_payload = _try_completion_affordance(
                output_dir,
                depth_uint16,
                mask,
                intr,
                object_category=object_category,
            )
            if full_payload is not None:
                completed = full_payload["completed_camera"]
                completed_labels = full_payload["completed_labels"]
                affordance_label = int(full_payload["summary"].get("affordance_label", 1))
                affordance_points = completed[completed_labels == affordance_label]
                visible_affordance_points = full_payload["partial_camera"][
                    full_payload["partial_labels"] == affordance_label
                ]
                raw_affordance_count = int(len(affordance_points))
                raw_visible_affordance_count = int(len(visible_affordance_points))
                affordance_points = _largest_point_component(affordance_points, radius_m=0.02, min_points=64)
                visible_affordance_points = _largest_point_component(
                    visible_affordance_points,
                    radius_m=0.02,
                    min_points=32,
                )
                contact_points = _select_handle_midsection(visible_affordance_points)
                contact_mask = _points_to_image_mask(
                    contact_points,
                    intr,
                    depth_uint16.shape,
                    dilate_iterations=1,
                )
                refined_contact_mask = contact_mask & affordance_mask
                if int(refined_contact_mask.sum()) >= 300:
                    contact_mask = _largest_component(refined_contact_mask)
                    contact_mask_source = "DGCNN_visible_affordance_midsection_intersect_2d_handle"
                else:
                    contact_mask_source = "DGCNN_visible_affordance_midsection_projection"
                if int(contact_mask.sum()) < 300:
                    contact_mask = affordance_mask
                    contact_mask_source = "fallback_2d_affordance_mask"
                completion_backend = "SeedFormer"
                affordance_backend = "DGCNN_partseg"
                completion_affordance_summary = full_payload["summary"]
                completion_affordance_summary["connected_component_filter"] = {
                    "radius_m": 0.02,
                    "completed_before": raw_affordance_count,
                    "completed_after": int(len(affordance_points)),
                    "visible_before": raw_visible_affordance_count,
                    "visible_after": int(len(visible_affordance_points)),
                }
            else:
                completed = complete_by_bbox_symmetry(partial, target_points=4096)
                affordance_points = _depth_to_points(depth_uint16, affordance_mask, intr)
                affordance_points = _select_handle_midsection(affordance_points)
                visible_affordance_points = affordance_points
                contact_mask = affordance_mask
                contact_mask_source = "fallback_2d_affordance_mask"
                completion_backend = "sim_pipeline.complete_by_bbox_symmetry"
                affordance_backend = "thin_midsection_selected_from_2d_mask_then_projected_to_points"
                completion_affordance_summary = None
            cv2.imwrite(str(output_dir / "overlays" / "08_contact_graspnet_segmap.png"), (contact_mask.astype(np.uint8) * 255))
            contact_overlay = rgb_bgr.copy()
            contact_overlay[contact_mask] = (
                0.45 * contact_overlay[contact_mask] + 0.55 * np.array([255, 80, 40], dtype=np.uint8)
            ).astype(np.uint8)
            cv2.imwrite(str(output_dir / "overlays" / "09_contact_graspnet_segmap_overlay.png"), contact_overlay)
            contact_payload = _try_contact_graspnet_pytorch(output_dir, rgb_bgr, depth_uint16, contact_mask, intr)
            if contact_payload is None and contact_mask_source != "fallback_2d_affordance_mask":
                retry_mask = cv2.dilate(
                    contact_mask.astype(np.uint8),
                    np.ones((7, 7), np.uint8),
                    iterations=2,
                ).astype(bool)
                retry_mask &= mask
                cv2.imwrite(
                    str(output_dir / "overlays" / "10_contact_graspnet_segmap_retry.png"),
                    (retry_mask.astype(np.uint8) * 255),
                )
                retry_overlay = rgb_bgr.copy()
                retry_overlay[retry_mask] = (
                    0.45 * retry_overlay[retry_mask] + 0.55 * np.array([255, 80, 40], dtype=np.uint8)
                ).astype(np.uint8)
                cv2.imwrite(str(output_dir / "overlays" / "11_contact_graspnet_segmap_retry_overlay.png"), retry_overlay)
                retry_payload = _try_contact_graspnet_pytorch(output_dir, rgb_bgr, depth_uint16, retry_mask, intr)
                if retry_payload is not None:
                    contact_payload = retry_payload
                    contact_mask = retry_mask
                    contact_mask_source += "_dilated_retry"
            if contact_payload is not None:
                contact_candidate = _contact_candidate_to_grasp_candidate(contact_payload)
                scored = [contact_candidate]
                grasp_generation_backend = "contact_graspnet_pytorch"
            else:
                candidates = _generate_pca_handle_grasps(affordance_points, width_margin_m=0.005)
                if not candidates:
                    candidates = generate_topdown_grasps(affordance_points, max_candidates=24, width_margin_m=0.005)
                scored = score_grasps(candidates, completed, table_z_m=-10.0, max_gripper_width_m=0.095)
                grasp_generation_backend = "pca_handle_axis_then_sim_pipeline_scoring"
                affordance_backend = "thin_midsection_selected_from_2d_mask_then_projected_to_points"
            best = scored[0] if scored else None
            if best is None or not best.collision_free:
                result = _rejected_result(run_id, "no_collision_free_grasp_candidate")
                result["target_box_xyxy"] = box
                result["mask_area_px"] = int(mask.sum())
                result["partial_points"] = int(len(partial))
            else:
                result = {
                    "status": "ok",
                    "run_id": run_id,
                    "target_box_xyxy": box,
                    "mask_area_px": int(mask.sum()),
                    "partial_points": int(len(partial)),
                    "completion_status": "ok",
                    "affordance_status": "ok",
                    "grasp_pose_camera": _pose_from_candidate(best, z_offset_m=0.0),
                    "approach_pose_camera": _pose_from_candidate(best, z_offset_m=-0.08),
                    "quality_gates": {
                        "detection_ok": True,
                        "mask_ok": int(mask.sum()) >= 500,
                        "depth_ok": int(len(partial)) >= 800,
                        "completion_ok": int(len(completed)) > 0,
                        "affordance_ok": int(len(affordance_points)) >= 16,
                    },
                    "reject_reason": None,
                    "backend": {
                        "segmentation": segmentation_backend,
                        "perception_2d": perception_payload,
                        "completion": completion_backend,
                        "affordance": affordance_backend,
                        "grasp_generation": grasp_generation_backend,
                        "grasp_scoring": "contact_graspnet_score" if contact_payload is not None else "sim_pipeline.score_grasps",
                        "grasp_summary": summarize_grasp_set(scored),
                        "best_grasp": best.to_dict(),
                        "contact_graspnet": {
                            "used": contact_payload is not None,
                            "num_candidates": int(contact_payload.get("num_candidates", 0)) if contact_payload else 0,
                            "torch_device": contact_payload.get("torch_device") if contact_payload else None,
                            "segmap_source": contact_mask_source,
                            "segmap_pixels": int(contact_mask.sum()),
                        },
                        "completion_affordance": {
                            "used": full_payload is not None,
                            "summary": completion_affordance_summary,
                            "seedformer_checkpoint": full_payload.get("seedformer_checkpoint") if full_payload else None,
                            "dgcnn_checkpoint": full_payload.get("dgcnn_checkpoint") if full_payload else None,
                        },
                    },
                }
            _write_ply(output_dir / "clouds" / "partial_camera.ply", partial, (80, 150, 220))
            _write_ply(output_dir / "clouds" / "complete_object.ply", completed, (180, 188, 196))
            _write_ply(output_dir / "clouds" / "affordance_complete.ply", affordance_points, (220, 48, 48))
            _write_ply(output_dir / "clouds" / "affordance_visible.ply", visible_affordance_points, (220, 48, 48))
        _save_overlays(output_dir, rgb_bgr, mask, box, affordance_mask)

    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Server-side real-robot request/result contract entrypoint."
    )
    parser.add_argument("request_dir")
    parser.add_argument("output_dir")
    return parser


def main(argv: list[str] | None = None, stderr: TextIO | None = None) -> int:
    err = stderr or sys.stderr
    args = build_parser().parse_args(argv)
    try:
        run_contract_entrypoint(Path(args.request_dir), Path(args.output_dir))
        return 0
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
        print(f"request invalid: {exc}", file=err)
        return 2
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=err)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
