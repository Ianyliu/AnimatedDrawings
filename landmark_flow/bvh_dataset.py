"""Build synthetic 2D landmark correction datasets from BVH files."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - convenience fallback for minimal cluster envs
    def tqdm(iterable, **_kwargs):
        return iterable

from animated_drawings.model.bvh import BVH
from landmark_flow.constants import BVH_ALIASES, CORRUPTION_TYPE_TO_ID, LANDMARK_NAMES, SPAN_BUCKET_TO_ID

LIMB_GROUPS = [
    [1, 3, 5],
    [2, 4, 6],
    [7, 9, 11],
    [8, 10, 12],
]

LEFT_RIGHT_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)]


def _rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ rx @ ry


def _find_joint(joint_names: List[str], aliases: List[str]) -> Optional[str]:
    lower_to_name = {name.lower(): name for name in joint_names}
    for alias in aliases:
        if alias.lower() in lower_to_name:
            return lower_to_name[alias.lower()]
    for alias in aliases:
        alias_lower = alias.lower()
        for name in joint_names:
            if name.lower().endswith(alias_lower):
                return name
    return None


def _map_required_joints(bvh: BVH) -> Optional[Dict[str, str]]:
    names = bvh.get_joint_names()
    mapping: Dict[str, str] = {}
    for landmark in LANDMARK_NAMES:
        found = _find_joint(names, BVH_ALIASES[landmark])
        if found is None:
            return None
        mapping[landmark] = found
    return mapping


def _extract_world_positions(bvh_path: Path) -> Tuple[np.ndarray, Dict[str, str]]:
    bvh = BVH.from_file(str(bvh_path))
    mapping = _map_required_joints(bvh)
    if mapping is None:
        raise ValueError("missing one or more required landmark joints")

    joints = [bvh.root_joint.get_transform_by_name(mapping[name]) for name in LANDMARK_NAMES]
    if any(joint is None for joint in joints):
        raise ValueError("joint lookup failed after alias mapping")

    frames = np.zeros((bvh.frame_max_num, len(LANDMARK_NAMES), 3), dtype=np.float32)
    for frame_idx in range(bvh.frame_max_num):
        bvh.apply_frame(frame_idx)
        bvh.root_joint.update_transforms(update_ancestors=True)
        for joint_idx, joint in enumerate(joints):
            frames[frame_idx, joint_idx] = joint.get_world_position()
    return frames, mapping


def _project_clip(positions: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    yaw = rng.uniform(math.radians(-70.0), math.radians(70.0))
    pitch = rng.uniform(math.radians(-15.0), math.radians(15.0))
    roll = rng.uniform(math.radians(-8.0), math.radians(8.0))
    rotated = positions @ _rotation_matrix(yaw, pitch, roll).T
    xy = rotated[..., [0, 1]]
    center = xy.reshape(-1, 2).mean(axis=0, keepdims=True)
    xy = xy - center
    scale = np.maximum(np.abs(xy).max(), 1e-6)
    xy = xy / scale * rng.uniform(0.34, 0.46)
    xy = xy + np.array(rng.uniform(0.45, 0.55, size=(2,)), dtype=np.float32)
    return np.clip(xy, 0.0, 1.0).astype(np.float32)


def _span_bucket(length: int) -> str:
    if length <= 8:
        return "short"
    if length <= 16:
        return "medium"
    return "long"


def _mark_corruption(
    low_mask: np.ndarray,
    corruption_type: np.ndarray,
    span_bucket: np.ndarray,
    frames: slice,
    joints: List[int],
    kind: str,
    bucket: str = "none",
) -> None:
    low_mask[frames, joints] = True
    corruption_type[frames, joints] = CORRUPTION_TYPE_TO_ID[kind]
    span_bucket[frames, joints] = SPAN_BUCKET_TO_ID[bucket]


def _corrupt(
    clean_xy: np.ndarray,
    rng: np.random.Generator,
    min_span: int,
    max_span: int,
    whole_limb_prob: float,
    swap_prob: float,
    high_conf_wrong_prob: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    t, j, _ = clean_xy.shape
    corrupt = clean_xy + rng.normal(0.0, 0.008, size=clean_xy.shape).astype(np.float32)
    visibility = np.ones((t, j), dtype=np.float32)
    low_mask = np.zeros((t, j), dtype=bool)
    corruption_type = np.zeros((t, j), dtype=np.int16)
    span_bucket = np.zeros((t, j), dtype=np.int16)

    random_mask = rng.random((t, j)) < rng.uniform(0.03, 0.12)
    low_mask[random_mask] = True
    corruption_type[random_mask] = CORRUPTION_TYPE_TO_ID["random"]

    for joint_idx in range(j):
        if rng.random() < 0.65:
            start = int(rng.integers(0, max(1, t - min_span)))
            length = int(rng.integers(min_span, min(max_span, t - start) + 1))
            bucket = _span_bucket(length)
            _mark_corruption(low_mask, corruption_type, span_bucket, slice(start, start + length), [joint_idx], f"{bucket}_span", bucket)
            drift = np.linspace(0.0, 1.0, length, dtype=np.float32)[:, None]
            drift_vec = rng.normal(0.0, 0.035, size=(1, 2)).astype(np.float32)
            corrupt[start:start + length, joint_idx] += drift * drift_vec

    if rng.random() < whole_limb_prob:
        joints = LIMB_GROUPS[int(rng.integers(0, len(LIMB_GROUPS)))]
        start = int(rng.integers(0, max(1, t - min_span)))
        length = int(rng.integers(min_span, min(max_span, t - start) + 1))
        bucket = _span_bucket(length)
        _mark_corruption(low_mask, corruption_type, span_bucket, slice(start, start + length), joints, "whole_limb", bucket)
        corrupt[start:start + length, joints] += rng.normal(0.0, 0.05, size=(length, len(joints), 2)).astype(np.float32)

    spike_count = int(rng.integers(0, 4))
    for _ in range(spike_count):
        frame_idx = int(rng.integers(0, t))
        joint_idx = int(rng.integers(0, j))
        corrupt[frame_idx, joint_idx] += rng.normal(0.0, 0.08, size=(2,)).astype(np.float32)
        _mark_corruption(low_mask, corruption_type, span_bucket, slice(frame_idx, frame_idx + 1), [joint_idx], "spike")

    if rng.random() < swap_prob:
        start = int(rng.integers(0, max(1, t - 4)))
        length = int(rng.integers(2, min(max(8, min_span), t - start) + 1))
        bucket = _span_bucket(length)
        for left_idx, right_idx in LEFT_RIGHT_PAIRS:
            corrupt[start:start + length, [left_idx, right_idx]] = corrupt[start:start + length, [right_idx, left_idx]]
            _mark_corruption(low_mask, corruption_type, span_bucket, slice(start, start + length), [left_idx, right_idx], "swap", bucket)

    high_conf_wrong_mask = np.zeros((t, j), dtype=bool)
    if rng.random() < high_conf_wrong_prob:
        start = int(rng.integers(0, max(1, t - min_span)))
        length = int(rng.integers(min_span, min(max_span, t - start) + 1))
        joint_idx = int(rng.integers(0, j))
        bucket = _span_bucket(length)
        _mark_corruption(low_mask, corruption_type, span_bucket, slice(start, start + length), [joint_idx], "high_conf_wrong", bucket)
        corrupt[start:start + length, joint_idx] += rng.normal(0.0, 0.075, size=(length, 2)).astype(np.float32)
        high_conf_wrong_mask[start:start + length, joint_idx] = True

    corrupt[low_mask] += rng.normal(0.0, 0.035, size=corrupt[low_mask].shape).astype(np.float32)
    visibility[low_mask] = rng.uniform(0.02, 0.34, size=int(low_mask.sum())).astype(np.float32)
    visibility[high_conf_wrong_mask] = rng.uniform(0.6, 1.0, size=int(high_conf_wrong_mask.sum())).astype(np.float32)
    corrupt = np.clip(corrupt, 0.0, 1.0).astype(np.float32)
    return corrupt, visibility, low_mask.astype(np.float32), corruption_type, span_bucket


def _iter_bvh_files(roots: Iterable[Path]) -> List[Path]:
    files: List[Path] = []
    for root in roots:
        if root.is_file() and root.suffix.lower() == ".bvh":
            files.append(root)
        elif root.exists():
            files.extend(sorted(root.rglob("*.bvh")))
    return sorted(set(path.resolve() for path in files))


def build_dataset(
    bvh_roots: List[Path],
    output_dir: Path,
    clip_len: int,
    stride: int,
    seed: int,
    max_files: Optional[int],
    min_span: int,
    max_span: int,
    whole_limb_prob: float,
    swap_prob: float,
    high_conf_wrong_prob: float,
) -> None:
    rng = np.random.default_rng(seed)
    random.seed(seed)
    bvh_files = _iter_bvh_files(bvh_roots)
    if max_files is not None:
        bvh_files = bvh_files[:max_files]
    if not bvh_files:
        raise FileNotFoundError("no .bvh files found")

    output_dir.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejected = []
    clips = {
        "x_corrupt": [],
        "y_clean": [],
        "low_conf_mask": [],
        "corruption_type": [],
        "span_bucket": [],
        "source": [],
        "start_frame": [],
    }

    for bvh_path in tqdm(bvh_files, desc="BVH files"):
        try:
            positions, mapping = _extract_world_positions(bvh_path)
        except Exception as exc:
            rejected.append({"path": str(bvh_path), "reason": str(exc)})
            continue
        if positions.shape[0] < clip_len:
            rejected.append({"path": str(bvh_path), "reason": f"only {positions.shape[0]} frames"})
            continue

        accepted.append({"path": str(bvh_path), "frames": int(positions.shape[0]), "mapping": mapping})
        for start in range(0, positions.shape[0] - clip_len + 1, stride):
            clean = _project_clip(positions[start:start + clip_len], rng)
            corrupt, visibility, low_mask, corruption_type, span_bucket = _corrupt(
                clean,
                rng,
                min_span=min_span,
                max_span=max_span,
                whole_limb_prob=whole_limb_prob,
                swap_prob=swap_prob,
                high_conf_wrong_prob=high_conf_wrong_prob,
            )
            condition = np.concatenate([corrupt, visibility[..., None], low_mask[..., None]], axis=-1)
            clips["x_corrupt"].append(condition.astype(np.float32))
            clips["y_clean"].append(clean.astype(np.float32))
            clips["low_conf_mask"].append(low_mask.astype(np.float32))
            clips["corruption_type"].append(corruption_type)
            clips["span_bucket"].append(span_bucket)
            clips["source"].append(str(bvh_path))
            clips["start_frame"].append(start)

    if not clips["y_clean"]:
        raise RuntimeError("no clips generated; check BVH joint naming and clip length")

    arrays = {
        "x_corrupt": np.stack(clips["x_corrupt"]),
        "y_clean": np.stack(clips["y_clean"]),
        "low_conf_mask": np.stack(clips["low_conf_mask"]),
        "corruption_type": np.stack(clips["corruption_type"]),
        "span_bucket": np.stack(clips["span_bucket"]),
        "source": np.array(clips["source"]),
        "start_frame": np.array(clips["start_frame"], dtype=np.int32),
    }

    sources = sorted(set(clips["source"]))
    rng.shuffle(sources)
    if len(sources) >= 3:
        val_count = max(1, int(round(0.1 * len(sources))))
        test_count = max(1, int(round(0.1 * len(sources))))
        train_count = max(1, len(sources) - val_count - test_count)
        train_sources = set(sources[:train_count])
        val_sources = set(sources[train_count:train_count + val_count])
        test_sources = set(sources[train_count + val_count:])
    elif len(sources) == 2:
        train_sources = {sources[0]}
        val_sources = set()
        test_sources = {sources[1]}
    else:
        train_sources = set(sources)
        val_sources = set()
        test_sources = set()

    source_array = arrays["source"]
    splits = {
        "train": np.flatnonzero(np.isin(source_array, list(train_sources))),
        "val": np.flatnonzero(np.isin(source_array, list(val_sources))),
        "test": np.flatnonzero(np.isin(source_array, list(test_sources))),
    }
    for split_name, split_indices in splits.items():
        np.savez_compressed(output_dir / f"{split_name}.npz", **{k: v[split_indices] for k, v in arrays.items()})

    metadata = {
        "landmark_order": LANDMARK_NAMES,
        "clip_len": clip_len,
        "stride": stride,
        "seed": seed,
        "corruption": {
            "min_span": min_span,
            "max_span": max_span,
            "whole_limb_prob": whole_limb_prob,
            "swap_prob": swap_prob,
            "high_conf_wrong_prob": high_conf_wrong_prob,
            "type_ids": CORRUPTION_TYPE_TO_ID,
            "span_bucket_ids": SPAN_BUCKET_TO_ID,
        },
        "num_clips": int(len(clips["y_clean"])),
        "splits": {name: int(len(split_indices)) for name, split_indices in splits.items()},
        "split_by": "bvh_file",
        "accepted_files": accepted,
        "rejected_files": rejected,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bvh-root", action="append", type=Path, default=[], help="BVH file or directory; repeatable")
    parser.add_argument("--output-dir", type=Path, default=Path("data/landmark_flow"))
    parser.add_argument("--clip-len", type=int, default=31)
    parser.add_argument("--stride", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--min-span", type=int, default=3)
    parser.add_argument("--max-span", type=int, default=30)
    parser.add_argument("--whole-limb-prob", type=float, default=0.35)
    parser.add_argument("--swap-prob", type=float, default=0.20)
    parser.add_argument("--high-conf-wrong-prob", type=float, default=0.0)
    args = parser.parse_args()

    roots = args.bvh_root or [Path("examples/bvh")]
    build_dataset(
        roots,
        args.output_dir,
        args.clip_len,
        args.stride,
        args.seed,
        args.max_files,
        args.min_span,
        args.max_span,
        args.whole_limb_prob,
        args.swap_prob,
        args.high_conf_wrong_prob,
    )


if __name__ == "__main__":
    main()
