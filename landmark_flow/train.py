"""Train the conditional rectified-flow landmark corrector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal cluster envs
    def tqdm(iterable, **_kwargs):
        return iterable

from landmark_flow.constants import CORRUPTION_TYPE_TO_ID, LANDMARK_NAMES, SPAN_BUCKET_TO_ID
from landmark_flow.model import LandmarkFlowModel

ID_TO_CORRUPTION_TYPE = {v: k for k, v in CORRUPTION_TYPE_TO_ID.items()}
ID_TO_SPAN_BUCKET = {v: k for k, v in SPAN_BUCKET_TO_ID.items()}


class LandmarkDataset(Dataset):
    def __init__(self, npz_path: Path) -> None:
        data = np.load(npz_path)
        self.x = torch.from_numpy(data["x_corrupt"]).float()
        self.y = torch.from_numpy(data["y_clean"]).float()
        self.mask = torch.from_numpy(data["low_conf_mask"]).float()
        self.corruption_type = torch.from_numpy(data["corruption_type"]).long() if "corruption_type" in data.files else torch.zeros_like(self.mask).long()
        self.span_bucket = torch.from_numpy(data["span_bucket"]).long() if "span_bucket" in data.files else torch.zeros_like(self.mask).long()

    def __len__(self) -> int:
        return self.y.shape[0]

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.mask[idx], self.corruption_type[idx], self.span_bucket[idx]


def _masked_l1(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1)
    return (torch.abs(pred - target) * weights).sum() / weights.sum().clamp_min(1.0)


def _linear_interpolation(condition: torch.Tensor, threshold: float = 0.35) -> torch.Tensor:
    xy = condition[..., :2].detach().cpu().numpy()
    vis = condition[..., 2].detach().cpu().numpy()
    out = xy.copy()
    frames = np.arange(xy.shape[1])
    for b in range(xy.shape[0]):
        for j in range(xy.shape[2]):
            good = vis[b, :, j] >= threshold
            if good.sum() >= 2:
                out[b, :, j, 0] = np.interp(frames, frames[good], xy[b, good, j, 0])
                out[b, :, j, 1] = np.interp(frames, frames[good], xy[b, good, j, 1])
            elif good.sum() == 1:
                out[b, :, j] = xy[b, good, j][0]
    return torch.from_numpy(out).to(condition.device, dtype=condition.dtype)


def _accumulate_metrics(
    totals: Dict[str, float],
    prefix: str,
    pred: torch.Tensor,
    interp: torch.Tensor,
    clean: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    weights = mask.unsqueeze(-1)
    count = weights.sum().item()
    if count <= 0:
        return
    total_count_key = f"{prefix}_count"
    totals[total_count_key] = totals.get(total_count_key, 0.0) + count
    totals[f"{prefix}_l1"] = totals.get(f"{prefix}_l1", 0.0) + (torch.abs(pred - clean) * weights).sum().item()
    totals[f"{prefix}_interp_l1"] = totals.get(f"{prefix}_interp_l1", 0.0) + (torch.abs(interp - clean) * weights).sum().item()
    totals[f"{prefix}_l2"] = totals.get(f"{prefix}_l2", 0.0) + (((pred - clean) ** 2) * weights).sum().item()
    totals[f"{prefix}_interp_l2"] = totals.get(f"{prefix}_interp_l2", 0.0) + (((interp - clean) ** 2) * weights).sum().item()
    dist = torch.linalg.norm(pred - clean, dim=-1)
    interp_dist = torch.linalg.norm(interp - clean, dim=-1)
    totals[f"{prefix}_pck_0_02"] = totals.get(f"{prefix}_pck_0_02", 0.0) + ((dist < 0.02).float() * mask).sum().item()
    totals[f"{prefix}_interp_pck_0_02"] = totals.get(f"{prefix}_interp_pck_0_02", 0.0) + ((interp_dist < 0.02).float() * mask).sum().item()


def _finalize_metrics(totals: Dict[str, float], prefix: str, output_prefix: str) -> Dict[str, float]:
    count = totals.get(f"{prefix}_count", 0.0)
    if count <= 0:
        return {}
    l2 = totals.get(f"{prefix}_l2", 0.0) / count
    interp_l2 = totals.get(f"{prefix}_interp_l2", 0.0) / count
    return {
        f"{output_prefix}masked_l1": totals.get(f"{prefix}_l1", 0.0) / count,
        f"{output_prefix}interp_masked_l1": totals.get(f"{prefix}_interp_l1", 0.0) / count,
        f"{output_prefix}masked_l2": l2,
        f"{output_prefix}interp_masked_l2": interp_l2,
        f"{output_prefix}masked_rmse": l2 ** 0.5,
        f"{output_prefix}interp_masked_rmse": interp_l2 ** 0.5,
        f"{output_prefix}pck_0_02": totals.get(f"{prefix}_pck_0_02", 0.0) / count,
        f"{output_prefix}interp_pck_0_02": totals.get(f"{prefix}_interp_pck_0_02", 0.0) / count,
        f"{output_prefix}count": count,
    }


@torch.no_grad()
def evaluate(
    model: LandmarkFlowModel,
    loader: DataLoader,
    device: torch.device,
    steps: int,
    threshold: float = 0.35,
) -> Dict[str, float]:
    model.eval()
    totals: Dict[str, float] = {}
    for condition, clean, mask, corruption_type, span_bucket in loader:
        condition = condition.to(device)
        clean = clean.to(device)
        mask = mask.to(device)
        corruption_type = corruption_type.to(device)
        span_bucket = span_bucket.to(device)
        estimate = condition[..., :2].clone()
        for step in range(steps):
            t = torch.full((estimate.shape[0],), float(step) / max(steps, 1), device=device)
            estimate = estimate + model(estimate, condition, t) / max(steps, 1)
        interp = _linear_interpolation(condition, threshold=threshold)

        _accumulate_metrics(totals, "overall", estimate, interp, clean, mask)
        for bucket_id, bucket_name in ID_TO_SPAN_BUCKET.items():
            if bucket_id == 0:
                continue
            bucket_mask = mask * (span_bucket == bucket_id).float()
            _accumulate_metrics(totals, f"span_{bucket_name}", estimate, interp, clean, bucket_mask)
        for type_id, type_name in ID_TO_CORRUPTION_TYPE.items():
            type_mask = mask * (corruption_type == type_id).float()
            _accumulate_metrics(totals, f"type_{type_name}", estimate, interp, clean, type_mask)

    metrics = _finalize_metrics(totals, "overall", "")
    for bucket_name in ["short", "medium", "long"]:
        metrics.update(_finalize_metrics(totals, f"span_{bucket_name}", f"span_{bucket_name}_"))
    for type_name in ID_TO_CORRUPTION_TYPE.values():
        metrics.update(_finalize_metrics(totals, f"type_{type_name}", f"type_{type_name}_"))
    return metrics


def train(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_loader = DataLoader(
        LandmarkDataset(args.data_dir / "train.npz"),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(LandmarkDataset(args.data_dir / "val.npz"), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(LandmarkDataset(args.data_dir / "test.npz"), batch_size=args.batch_size, shuffle=False)

    model = LandmarkFlowModel(hidden_size=args.hidden_size, dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_val = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for condition, clean, mask, _corruption_type, _span_bucket in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            condition = condition.to(device)
            clean = clean.to(device)
            mask = mask.to(device)
            corrupt_xy = condition[..., :2]
            t = torch.rand((condition.shape[0],), device=device)
            y_t = (1.0 - t[:, None, None, None]) * corrupt_xy + t[:, None, None, None] * clean
            target_velocity = clean - corrupt_xy
            pred_velocity = model(y_t, condition, t)

            masked = mask.unsqueeze(-1)
            recon_loss = (torch.nn.functional.smooth_l1_loss(pred_velocity, target_velocity, reduction="none") * masked).sum()
            recon_loss = recon_loss / masked.sum().clamp_min(1.0)
            smooth_loss = torch.mean(torch.abs(pred_velocity[:, 1:] - pred_velocity[:, :-1]))
            loss = recon_loss + args.smooth_weight * smooth_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            running += loss.item()

        scheduler.step()
        val_metrics = evaluate(model, val_loader, device, args.inference_steps, threshold=args.threshold)
        record = {"epoch": epoch, "loss": running / max(len(train_loader), 1), **{f"val_{k}": v for k, v in val_metrics.items()}}
        history.append(record)
        print(json.dumps(record, sort_keys=True))

        if val_metrics["masked_l1"] < best_val:
            best_val = val_metrics["masked_l1"]
            torch.save({"model_state": model.state_dict(), "model_config": model.config, "epoch": epoch}, args.output_dir / "best.pt")

    checkpoint = torch.load(args.output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate(model, test_loader, device, args.inference_steps, threshold=args.threshold)
    serializable_args = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    metadata = {
        "landmark_order": LANDMARK_NAMES,
        "window_size": train_loader.dataset.y.shape[1],
        "threshold": args.threshold,
        "normalization": "per-clip orthographic projection to approximately [0, 1]",
        "model_config": model.config,
        "training_args": serializable_args,
        "history": history,
        "test_metrics": test_metrics,
    }
    torch.save({"model_state": model.state_dict(), "metadata": metadata}, args.output_dir / "landmark_flow_corrector.pt")
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"test": test_metrics}, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/landmark_flow"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/landmark_flow"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--smooth-weight", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
