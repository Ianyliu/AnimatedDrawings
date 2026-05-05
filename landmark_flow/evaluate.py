"""Evaluate an exported landmark flow checkpoint on a prepared dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from landmark_flow.model import LandmarkFlowModel
from landmark_flow.train import LandmarkDataset, evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/landmark_flow"))
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    payload = torch.load(args.checkpoint, map_location=device)
    metadata = payload.get("metadata", {})
    config = metadata.get("model_config", payload.get("model_config", {}))
    model = LandmarkFlowModel(**config).to(device)
    model.load_state_dict(payload["model_state"])

    loader = DataLoader(LandmarkDataset(args.data_dir / f"{args.split}.npz"), batch_size=args.batch_size, shuffle=False)
    metrics = evaluate(model, loader, device, args.inference_steps, threshold=args.threshold)
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
