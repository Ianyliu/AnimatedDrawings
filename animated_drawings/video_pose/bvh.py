"""Conversion from pose landmarks into a MediaPipe-compatible BVH."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import yaml

from animated_drawings.video_pose.constants import (
    BVH_CONTROL_BONES,
    BVH_ROOT,
    BVH_ROTATION_JOINTS,
    MEDIAPIPE_MOTION_CFG,
    MEDIAPIPE_REQUIRED_LANDMARKS,
    BvhJointSpec,
)
from animated_drawings.video_pose.types import PoseFrame, PoseSequence, PoseVideoError


class PoseToBvhConverter:
    """Convert MediaPipe-style landmark sequences to a simple BVH motion."""

    def __init__(self, coordinate_scale: float = 1000.0) -> None:
        self.coordinate_scale = coordinate_scale

    def convert(self, sequence: PoseSequence, bvh_path: Path, motion_config_path: Optional[Path] = None) -> Path:
        if not sequence.frames:
            raise PoseVideoError("Cannot convert an empty pose sequence to BVH.")

        bvh_path.parent.mkdir(exist_ok=True, parents=True)
        first_root = self._root_image_position(sequence.frames[0])
        lines = ["HIERARCHY"]
        _write_joint(lines, BVH_ROOT, indent=0, root=True)
        lines.extend([
            "MOTION",
            f"Frames: {sequence.frame_count}",
            f"Frame Time: {sequence.frame_time}",
        ])

        last_positions: Optional[Dict[str, np.ndarray]] = None
        for frame in sequence.frames:
            positions = self._frame_positions(frame, last_positions)
            if positions is None:
                raise PoseVideoError("Pose sequence did not contain enough landmarks for BVH conversion.")
            last_positions = positions
            root_pos = self._root_translation(frame, first_root)
            local_y_rotations = self._local_y_rotations(positions)
            frame_values = [root_pos[0], root_pos[1], root_pos[2]]
            for joint_name in BVH_ROTATION_JOINTS:
                # Channels are Zrotation, Xrotation, Yrotation.
                frame_values.extend([0.0, 0.0, local_y_rotations.get(joint_name, 0.0)])
            lines.append(" ".join(f"{value:.8f}" for value in frame_values))

        with bvh_path.open("w") as f:
            f.write("\n".join(lines))
            f.write("\n")

        if motion_config_path is not None:
            write_motion_config_for_bvh(bvh_path, motion_config_path, sequence.frame_time)

        return bvh_path

    def _frame_positions(
        self,
        frame: PoseFrame,
        fallback: Optional[Dict[str, np.ndarray]],
    ) -> Optional[Dict[str, np.ndarray]]:
        if not all(name in frame.landmarks for name in MEDIAPIPE_REQUIRED_LANDMARKS):
            return fallback

        raw = {name: np.array(frame.landmarks[name][:3], dtype=np.float32) for name in MEDIAPIPE_REQUIRED_LANDMARKS}
        hip_img = (raw["LEFT_HIP"] + raw["RIGHT_HIP"]) / 2.0
        shoulder_img = (raw["LEFT_SHOULDER"] + raw["RIGHT_SHOULDER"]) / 2.0

        def to_bvh(point: np.ndarray) -> np.ndarray:
            return np.array(
                [
                    (point[0] - hip_img[0]) * self.coordinate_scale,
                    0.0,
                    (hip_img[1] - point[1]) * self.coordinate_scale,
                ],
                dtype=np.float32,
            )

        positions = {
            "Hip": np.array([0.0, 0.0, 0.0], dtype=np.float32),
            "RightHip": to_bvh(raw["RIGHT_HIP"]),
            "RightKnee": to_bvh(raw["RIGHT_KNEE"]),
            "RightAnkle": to_bvh(raw["RIGHT_ANKLE"]),
            "LeftHip": to_bvh(raw["LEFT_HIP"]),
            "LeftKnee": to_bvh(raw["LEFT_KNEE"]),
            "LeftAnkle": to_bvh(raw["LEFT_ANKLE"]),
            "Thorax": to_bvh(shoulder_img),
            "LeftShoulder": to_bvh(raw["LEFT_SHOULDER"]),
            "LeftElbow": to_bvh(raw["LEFT_ELBOW"]),
            "LeftWrist": to_bvh(raw["LEFT_WRIST"]),
            "RightShoulder": to_bvh(raw["RIGHT_SHOULDER"]),
            "RightElbow": to_bvh(raw["RIGHT_ELBOW"]),
            "RightWrist": to_bvh(raw["RIGHT_WRIST"]),
        }

        positions["Spine"] = positions["Hip"] + 0.5 * (positions["Thorax"] - positions["Hip"])
        nose = to_bvh(raw["NOSE"])
        neck_candidate = positions["Thorax"] + 0.35 * (nose - positions["Thorax"])
        if np.linalg.norm(neck_candidate - positions["Thorax"]) < 1e-4:
            neck_candidate = positions["Thorax"] + np.array([0.0, 0.0, 60.0], dtype=np.float32)
        positions["Neck"] = neck_candidate

        return positions

    def _root_image_position(self, frame: PoseFrame) -> np.ndarray:
        if "LEFT_HIP" not in frame.landmarks or "RIGHT_HIP" not in frame.landmarks:
            return np.array([0.5, 0.5, 0.0], dtype=np.float32)
        return (
            np.array(frame.landmarks["LEFT_HIP"][:3], dtype=np.float32)
            + np.array(frame.landmarks["RIGHT_HIP"][:3], dtype=np.float32)
        ) / 2.0

    def _root_translation(self, frame: PoseFrame, first_root: np.ndarray) -> np.ndarray:
        cur_root = self._root_image_position(frame)
        return np.array(
            [
                (cur_root[0] - first_root[0]) * self.coordinate_scale,
                0.0,
                (first_root[1] - cur_root[1]) * self.coordinate_scale,
            ],
            dtype=np.float32,
        )

    def _local_y_rotations(self, positions: Dict[str, np.ndarray]) -> Dict[str, float]:
        global_angles: Dict[str, float] = {"Hip": 0.0}
        local_angles: Dict[str, float] = {"Hip": 0.0}

        for joint_name, (prox_name, dist_name, rest_vec, parent_name) in BVH_CONTROL_BONES.items():
            target_vec = positions[dist_name] - positions[prox_name]
            global_angle = _signed_y_angle(np.array(rest_vec, dtype=np.float32), target_vec)
            parent_angle = global_angles.get(parent_name or "Hip", 0.0)
            global_angles[joint_name] = global_angle
            local_angles[joint_name] = _normalize_degrees(global_angle - parent_angle)

        return local_angles


def write_motion_config_for_bvh(bvh_path: Path, motion_config_path: Path, frame_time: float) -> Path:
    motion_config_path.parent.mkdir(exist_ok=True, parents=True)
    cfg = dict(MEDIAPIPE_MOTION_CFG)
    cfg["filepath"] = str(bvh_path.resolve())
    cfg["frame_time"] = float(frame_time)
    with motion_config_path.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return motion_config_path


def _write_joint(lines: list[str], spec: BvhJointSpec, indent: int, root: bool = False) -> None:
    prefix = "    " * indent
    joint_type = "ROOT" if root else "JOINT"
    lines.append(f"{prefix}{joint_type} {spec.name}")
    lines.append(f"{prefix}{{")
    lines.append(f"{prefix}    OFFSET {_format_offset(spec.offset)}")
    if root:
        lines.append(f"{prefix}    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation")
    else:
        lines.append(f"{prefix}    CHANNELS 3 Zrotation Xrotation Yrotation")

    for child in spec.children:
        _write_joint(lines, child, indent + 1)

    if spec.end_offset is not None:
        lines.append(f"{prefix}    End Site")
        lines.append(f"{prefix}    {{")
        lines.append(f"{prefix}        OFFSET {_format_offset(spec.end_offset)}")
        lines.append(f"{prefix}    }}")

    lines.append(f"{prefix}}}")


def _format_offset(offset: tuple[float, float, float]) -> str:
    return " ".join(str(float(value)) for value in offset)


def _signed_y_angle(rest_vec: np.ndarray, target_vec: np.ndarray) -> float:
    rest = np.array([rest_vec[0], rest_vec[2]], dtype=np.float32)
    target = np.array([target_vec[0], target_vec[2]], dtype=np.float32)
    rest_norm = np.linalg.norm(rest)
    target_norm = np.linalg.norm(target)
    if rest_norm < 1e-6 or target_norm < 1e-6:
        return 0.0

    rest /= rest_norm
    target /= target_norm
    dot = float(np.clip(np.dot(rest, target), -1.0, 1.0))
    det = float(rest[0] * target[1] - rest[1] * target[0])
    return _normalize_degrees(-math.degrees(math.atan2(det, dot)))


def _normalize_degrees(angle: float) -> float:
    while angle <= -180.0:
        angle += 360.0
    while angle > 180.0:
        angle -= 360.0
    return angle
