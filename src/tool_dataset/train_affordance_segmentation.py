#!/usr/bin/env python3
"""Train point-cloud affordance part segmenters on ToolMesh-mini."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    return pairwise_distance.topk(k=k, dim=-1)[1]


def get_graph_feature(x: torch.Tensor, k: int) -> torch.Tensor:
    batch_size, num_dims, num_points = x.size()
    idx = knn(x, k=k)
    device = x.device
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)
    x_t = x.transpose(2, 1).contiguous()
    feature = x_t.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x_t = x_t.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x_t, x_t), dim=3).permute(0, 3, 1, 2).contiguous()
    return feature


def square_distance(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    return torch.cdist(src, dst, p=2) ** 2


def index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=points.device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        distance = torch.minimum(distance, dist)
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def knn_group(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    sqrdists = square_distance(new_xyz, xyz)
    return sqrdists.topk(k=k, dim=-1, largest=False)[1]


class PointMLPSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        batch_size, num_points, _ = points.shape
        x = points.reshape(batch_size * num_points, 3)
        return self.net(x).view(batch_size, num_points, 2)


class DGCNNSeg(nn.Module):
    def __init__(self, k: int = 20):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.head = nn.Sequential(
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(2, 1).contiguous()
        x1 = self.conv1(get_graph_feature(x, self.k)).max(dim=-1)[0]
        x2 = self.conv2(get_graph_feature(x1, self.k)).max(dim=-1)[0]
        x3 = self.conv3(get_graph_feature(x2, self.k)).max(dim=-1)[0]
        local = torch.cat((x1, x2, x3), dim=1)
        global_feature = self.conv4(local).max(dim=-1, keepdim=True)[0].repeat(1, 1, points.shape[1])
        return self.head(torch.cat((local, global_feature), dim=1)).transpose(2, 1).contiguous()


class DGCNNPartSeg(nn.Module):
    """Official-scale DGCNN part-segmentation style network for binary affordance labels."""

    def __init__(self, k: int = 20, emb_dims: int = 1024, num_categories: int = 8):
        super().__init__()
        self.k = k
        self.emb_dims = emb_dims
        self.num_categories = num_categories
        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 64, 1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(
            nn.Conv1d(192, emb_dims, 1, bias=False),
            nn.BatchNorm1d(emb_dims),
            nn.LeakyReLU(0.2),
        )
        self.label_conv = nn.Sequential(
            nn.Conv1d(num_categories, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.head = nn.Sequential(
            nn.Conv1d(emb_dims + 64 + 192, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(256, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, points: torch.Tensor, category_ids: torch.Tensor | None = None) -> torch.Tensor:
        num_points = points.shape[1]
        x = points.transpose(2, 1).contiguous()

        x = self.conv1(get_graph_feature(x, self.k))
        x = self.conv2(x)
        x1 = x.max(dim=-1)[0]

        x = self.conv3(get_graph_feature(x1, self.k))
        x = self.conv4(x)
        x2 = x.max(dim=-1)[0]

        x = self.conv5(get_graph_feature(x2, self.k))
        x3 = x.max(dim=-1)[0]

        local = torch.cat((x1, x2, x3), dim=1)
        global_feature = self.conv6(local).max(dim=-1, keepdim=True)[0].repeat(1, 1, num_points)
        if category_ids is None:
            category_ids = torch.zeros(points.shape[0], dtype=torch.long, device=points.device)
        one_hot = F.one_hot(category_ids.clamp_min(0), num_classes=self.num_categories).float().unsqueeze(-1)
        category_feature = self.label_conv(one_hot).repeat(1, 1, num_points)
        return self.head(torch.cat((global_feature, category_feature, local), dim=1)).transpose(2, 1).contiguous()


class PointNetSeg(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv1d(3, 64, 1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Conv1d(512, 256, 1, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(256, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        x = points.transpose(2, 1).contiguous()
        local = self.local(x)
        global_feature = local.max(dim=-1, keepdim=True)[0].repeat(1, 1, points.shape[1])
        return self.head(torch.cat((local, global_feature), dim=1)).transpose(2, 1).contiguous()


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint: int, k: int, in_channel: int, mlp: list[int]):
        super().__init__()
        self.npoint = npoint
        self.k = k
        last_channel = in_channel + 3
        layers = []
        for out_channel in mlp:
            layers.extend([
                nn.Conv2d(last_channel, out_channel, 1, bias=False),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True),
            ])
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(self, xyz: torch.Tensor, points: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        fps_idx = farthest_point_sample(xyz, self.npoint)
        new_xyz = index_points(xyz, fps_idx)
        idx = knn_group(self.k, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)
        if points is not None:
            grouped_points = index_points(points.transpose(1, 2), idx)
            grouped_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1)
        else:
            grouped_points = grouped_xyz_norm
        grouped_points = grouped_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(grouped_points).max(dim=-1)[0]
        return new_xyz, new_points


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel: int, mlp: list[int]):
        super().__init__()
        layers = []
        last_channel = in_channel
        for out_channel in mlp:
            layers.extend([
                nn.Conv1d(last_channel, out_channel, 1, bias=False),
                nn.BatchNorm1d(out_channel),
                nn.ReLU(inplace=True),
            ])
            last_channel = out_channel
        self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        xyz1: torch.Tensor,
        xyz2: torch.Tensor,
        points1: torch.Tensor | None,
        points2: torch.Tensor,
    ) -> torch.Tensor:
        dists = square_distance(xyz1, xyz2)
        dists, idx = dists.sort(dim=-1)
        dists = dists[:, :, :3]
        idx = idx[:, :, :3]
        dist_recip = 1.0 / (dists + 1e-8)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        interpolated_points = torch.sum(index_points(points2.transpose(1, 2), idx) * weight.unsqueeze(-1), dim=2)
        interpolated_points = interpolated_points.transpose(1, 2).contiguous()
        if points1 is not None:
            new_points = torch.cat([points1, interpolated_points], dim=1)
        else:
            new_points = interpolated_points
        return self.mlp(new_points)


class PointNet2LiteSeg(nn.Module):
    def __init__(self, k: int = 24):
        super().__init__()
        self.sa1 = PointNetSetAbstraction(npoint=512, k=k, in_channel=0, mlp=[64, 64, 128])
        self.sa2 = PointNetSetAbstraction(npoint=128, k=k, in_channel=128, mlp=[128, 128, 256])
        self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, 128])
        self.fp1 = PointNetFeaturePropagation(in_channel=131, mlp=[128, 128, 128])
        self.head = nn.Sequential(
            nn.Conv1d(128, 128, 1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Conv1d(128, 2, 1),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        xyz = points.contiguous()
        l1_xyz, l1_points = self.sa1(xyz, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        xyz_features = xyz.transpose(1, 2).contiguous()
        l0_points = self.fp1(xyz, l1_xyz, xyz_features, l1_points)
        return self.head(l0_points).transpose(2, 1).contiguous()


class ToolSegDataset(Dataset):
    def __init__(self, root: Path, split: str, n_points: int, return_category: bool = False):
        self.root = root
        self.split = split
        self.n_points = n_points
        self.return_category = return_category
        with (root / "segmentation" / "manifest.json").open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        self.categories = sorted({item["category"] for rows in manifest.values() for item in rows})
        self.category_to_idx = {category: idx for idx, category in enumerate(self.categories)}
        self.items = manifest[split]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        data = np.load(self.root / item["path"])
        points = data["points"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        replace = points.shape[0] < self.n_points
        choice = np.random.choice(points.shape[0], self.n_points, replace=replace)
        if not self.return_category:
            return torch.from_numpy(points[choice]), torch.from_numpy(labels[choice])
        category_id = self.category_to_idx[item["category"]]
        return torch.from_numpy(points[choice]), torch.from_numpy(labels[choice]), torch.tensor(category_id, dtype=torch.long)


def compute_label_weights(dataset: ToolSegDataset, num_classes: int = 2) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.float64)
    for item in dataset.items:
        data = np.load(dataset.root / item["path"])
        labels = data["labels"].astype(np.int64)
        counts += np.bincount(labels, minlength=num_classes)[:num_classes]
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    weights = weights / weights.mean()
    return torch.from_numpy(weights.astype(np.float32))


def unpack_batch(batch, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if len(batch) == 3:
        points, labels, category_ids = batch
        return points.to(device), labels.to(device), category_ids.to(device)
    points, labels = batch
    return points.to(device), labels.to(device), None


def forward_model(model: nn.Module, points: torch.Tensor, category_ids: torch.Tensor | None) -> torch.Tensor:
    if category_ids is None:
        return model(points)
    return model(points, category_ids)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    correct = 0
    total = 0
    intersection = torch.zeros(2, device=device)
    union = torch.zeros(2, device=device)
    for batch in loader:
        points, labels, category_ids = unpack_batch(batch, device)
        pred = forward_model(model, points, category_ids).argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
        for cls in range(2):
            p = pred == cls
            y = labels == cls
            intersection[cls] += (p & y).sum()
            union[cls] += (p | y).sum()
    iou = intersection / union.clamp_min(1)
    return {
        "acc": correct / max(total, 1),
        "miou": iou.mean().item(),
        "not_handle_iou": iou[0].item(),
        "handle_iou": iou[1].item(),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("src/datasets/ToolMesh-mini"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/tool_dataset/affordance_dgcnn"))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--n-points", type=int, default=2048)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--model", choices=("mlp", "pointnet", "pointnet2", "dgcnn", "dgcnn_partseg"), default="dgcnn")
    parser.add_argument("--emb-dims", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--class-weight-mode", choices=("none", "balanced"), default="balanced")
    parser.add_argument("--early-stop-patience", type=int, default=25)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_category = args.model == "dgcnn_partseg"
    val_dataset = ToolSegDataset(args.data_root, "val", args.n_points, return_category=use_category)
    test_dataset = ToolSegDataset(args.data_root, "test", args.n_points, return_category=use_category)
    args.categories = val_dataset.categories
    args.num_categories = len(val_dataset.categories)

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
    )
    if args.model == "mlp":
        model = PointMLPSeg().to(device)
    elif args.model == "pointnet":
        model = PointNetSeg().to(device)
    elif args.model == "pointnet2":
        model = PointNet2LiteSeg(k=args.k).to(device)
    elif args.model == "dgcnn_partseg":
        model = DGCNNPartSeg(k=args.k, emb_dims=args.emb_dims, num_categories=args.num_categories).to(device)
    else:
        model = DGCNNSeg(k=args.k).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    train_dataset = ToolSegDataset(args.data_root, "train", args.n_points, return_category=use_category)
    class_weights = (
        compute_label_weights(train_dataset).to(device)
        if args.class_weight_mode == "balanced"
        else None
    )

    if args.eval_only:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required with --eval-only")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model"])
        metrics = {
            "params": param_count,
            "trainable_params": trainable_count,
            "class_weight_mode": args.class_weight_mode,
            "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
            "val": evaluate(model, val_loader, device),
            "test": evaluate(model, test_loader, device),
        }
        with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
            json.dump(metrics, fh, ensure_ascii=False, indent=2)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_miou = -1.0
    epochs_without_improvement = 0
    best_path = args.out_dir / "ckpt-best.pth"
    log_path = args.out_dir / "training.log"
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            model.train()
            losses = []
            start = time.time()
            for batch in train_loader:
                points, labels, category_ids = unpack_batch(batch, device)
                logits = forward_model(model, points, category_ids)
                loss = F.cross_entropy(logits.view(-1, 2), labels.view(-1), weight=class_weights)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
            scheduler.step()
            metrics = evaluate(model, val_loader, device)
            message = (
                f"epoch={epoch} loss={np.mean(losses):.5f} "
                f"val_acc={metrics['acc']:.4f} val_miou={metrics['miou']:.4f} "
                f"handle_iou={metrics['handle_iou']:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
                f"time={time.time() - start:.1f}s"
            )
            print(message)
            log.write(message + "\n")
            log.flush()
            if metrics["miou"] > best_miou + args.early_stop_min_delta:
                best_miou = metrics["miou"]
                epochs_without_improvement = 0
                torch.save({
                    "model": model.state_dict(),
                    "args": vars(args),
                    "categories": val_dataset.categories,
                    "metrics": metrics,
                    "params": param_count,
                    "trainable_params": trainable_count,
                    "class_weight_mode": args.class_weight_mode,
                    "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
                }, best_path)
            else:
                epochs_without_improvement += 1
                if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                    stop_message = (
                        f"early_stop epoch={epoch} best_val_miou={best_miou:.4f} "
                        f"patience={args.early_stop_patience}"
                    )
                    print(stop_message)
                    log.write(stop_message + "\n")
                    log.flush()
                    break

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    final_metrics = {
        "params": param_count,
        "trainable_params": trainable_count,
        "class_weight_mode": args.class_weight_mode,
        "class_weights": class_weights.detach().cpu().tolist() if class_weights is not None else None,
        "best_val_in_training": checkpoint["metrics"],
        "val": evaluate(model, val_loader, device),
        "test": evaluate(model, test_loader, device),
    }
    with (args.out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
        json.dump(final_metrics, fh, ensure_ascii=False, indent=2)
    print(json.dumps(final_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
