"""Pose sequence post-processing and quality reporting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

from animated_drawings.video_pose.types import PoseFrame, PoseQualityReport, PoseSequence


LEFT_RIGHT_PAIRS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_ELBOW", "RIGHT_ELBOW"),
    ("LEFT_WRIST", "RIGHT_WRIST"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_KNEE", "RIGHT_KNEE"),
    ("LEFT_ANKLE", "RIGHT_ANKLE"),
]


@dataclass(frozen=True)
class PosePostprocessConfig:
    visibility_threshold: float = 0.35
    smoothing_window: int = 5
    root_jump_threshold: float = 0.12
    foot_contact_velocity: float = 0.012
    foot_contact_y: float = 0.72


class DefaultPosePostprocessor:
    """Repair, smooth, and quality-check MediaPipe-style pose landmarks."""

    def __init__(self, config: PosePostprocessConfig | None = None) -> None:
        self.config = config or PosePostprocessConfig()

    def process(self, sequence: PoseSequence) -> Tuple[PoseSequence, PoseQualityReport]:
        frames = _clone_frames(sequence.frames)
        names = _landmark_names(sequence, frames)
        original = _frames_to_array(frames, names)

        detection_coverage = _detection_coverage(original)
        repaired, repaired_count = _repair_low_confidence(original, self.config.visibility_threshold)
        smoothed = _smooth_landmarks(repaired, self.config.smoothing_window)
        left_right_corrections = _fix_left_right_swaps(smoothed, names)
        root_jump_corrections = _cleanup_root_motion(smoothed, names, self.config.root_jump_threshold)
        foot_stabilizations = _stabilize_feet(
            smoothed,
            names,
            velocity_threshold=self.config.foot_contact_velocity,
            contact_y=self.config.foot_contact_y,
        )

        processed_frames = _array_to_frames(frames, names, smoothed)
        total_slots = max(1, len(frames) * max(1, len(names)))
        metrics = {
            "detection_coverage": detection_coverage,
            "repaired_landmark_ratio": repaired_count / total_slots,
            "repaired_landmarks": float(repaired_count),
            "left_right_corrections": float(left_right_corrections),
            "root_jump_corrections": float(root_jump_corrections),
            "foot_stabilizations": float(foot_stabilizations),
        }
        report = PoseQualityReport(warnings=_quality_warnings(metrics), metrics=metrics)
        processed = PoseSequence(
            fps=sequence.fps,
            width=sequence.width,
            height=sequence.height,
            landmark_names=list(sequence.landmark_names),
            frames=processed_frames,
            quality_report=report,
        )
        return processed, report


def _clone_frames(frames: Iterable[PoseFrame]) -> List[PoseFrame]:
    return [
        PoseFrame(
            timestamp=frame.timestamp,
            landmarks={name: [float(value) for value in values] for name, values in frame.landmarks.items()},
        )
        for frame in frames
    ]


def _landmark_names(sequence: PoseSequence, frames: List[PoseFrame]) -> List[str]:
    names = list(sequence.landmark_names)
    for frame in frames:
        for name in frame.landmarks:
            if name not in names:
                names.append(name)
    return names


def _frames_to_array(frames: List[PoseFrame], names: List[str]) -> np.ndarray:
    data = np.full((len(frames), len(names), 4), np.nan, dtype=np.float32)
    for frame_idx, frame in enumerate(frames):
        for landmark_idx, name in enumerate(names):
            values = frame.landmarks.get(name)
            if not values:
                continue
            padded = list(values[:4])
            while len(padded) < 4:
                padded.append(1.0 if len(padded) == 3 else 0.0)
            data[frame_idx, landmark_idx, :] = np.array(padded[:4], dtype=np.float32)
    return data


def _array_to_frames(frames: List[PoseFrame], names: List[str], data: np.ndarray) -> List[PoseFrame]:
    out: List[PoseFrame] = []
    for frame_idx, frame in enumerate(frames):
        landmarks: Dict[str, List[float]] = {}
        for landmark_idx, name in enumerate(names):
            values = data[frame_idx, landmark_idx]
            if np.isnan(values[:3]).any():
                continue
            visibility = 0.0 if np.isnan(values[3]) else float(np.clip(values[3], 0.0, 1.0))
            landmarks[name] = [
                float(np.clip(values[0], -1.0, 2.0)),
                float(np.clip(values[1], -1.0, 2.0)),
                float(values[2]),
                visibility,
            ]
        out.append(PoseFrame(timestamp=frame.timestamp, landmarks=landmarks))
    return out


def _detection_coverage(data: np.ndarray) -> float:
    if data.shape[0] == 0:
        return 0.0
    visible = np.isfinite(data[:, :, :3]).all(axis=2)
    return float(np.count_nonzero(visible.any(axis=1)) / data.shape[0])


def _repair_low_confidence(data: np.ndarray, visibility_threshold: float) -> Tuple[np.ndarray, int]:
    repaired = data.copy()
    repaired_count = 0
    frame_count, landmark_count, _ = repaired.shape
    frame_positions = np.arange(frame_count)

    for landmark_idx in range(landmark_count):
        visibility = repaired[:, landmark_idx, 3]
        valid = (
            np.isfinite(repaired[:, landmark_idx, :3]).all(axis=1)
            & np.isfinite(visibility)
            & (visibility >= visibility_threshold)
        )
        if not np.any(valid):
            continue
        low = ~valid
        repaired_count += int(np.count_nonzero(low))
        valid_positions = frame_positions[valid]
        for axis in range(3):
            values = repaired[:, landmark_idx, axis]
            repaired[:, landmark_idx, axis] = np.interp(frame_positions, valid_positions, values[valid])
        repaired[:, landmark_idx, 3] = np.maximum(
            np.nan_to_num(repaired[:, landmark_idx, 3], nan=0.0),
            np.where(low, visibility_threshold, repaired[:, landmark_idx, 3]),
        )

    return repaired, repaired_count


def _smooth_landmarks(data: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or data.shape[0] < 3:
        return data
    window = max(3, min(window, data.shape[0]))
    if window % 2 == 0:
        window -= 1
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = data.copy()
    pad = window // 2

    for landmark_idx in range(data.shape[1]):
        for axis in range(3):
            values = data[:, landmark_idx, axis]
            if not np.isfinite(values).all():
                continue
            padded = np.pad(values, (pad, pad), mode="edge")
            smoothed[:, landmark_idx, axis] = np.convolve(padded, kernel, mode="valid")
    return smoothed


def _fix_left_right_swaps(data: np.ndarray, names: List[str]) -> int:
    index = {name: idx for idx, name in enumerate(names)}
    required = [pair for pair in LEFT_RIGHT_PAIRS if pair[0] in index and pair[1] in index]
    if not required:
        return 0
    corrections = 0
    for frame_idx in range(data.shape[0]):
        anchors = [
            data[frame_idx, index[left], 0] > data[frame_idx, index[right], 0]
            for left, right in required
            if np.isfinite(data[frame_idx, index[left], 0]) and np.isfinite(data[frame_idx, index[right], 0])
        ]
        if len(anchors) < 2 or anchors.count(True) < max(2, int(round(len(anchors) * 0.65))):
            continue
        for left, right in required:
            left_idx = index[left]
            right_idx = index[right]
            data[frame_idx, [left_idx, right_idx], :] = data[frame_idx, [right_idx, left_idx], :]
        corrections += 1
    return corrections


def _cleanup_root_motion(data: np.ndarray, names: List[str], threshold: float) -> int:
    index = {name: idx for idx, name in enumerate(names)}
    if "LEFT_HIP" not in index or "RIGHT_HIP" not in index or data.shape[0] < 2:
        return 0
    left_idx = index["LEFT_HIP"]
    right_idx = index["RIGHT_HIP"]
    centers = (data[:, left_idx, :2] + data[:, right_idx, :2]) / 2.0
    corrections = 0
    finite = np.isfinite(centers).all(axis=1)
    for frame_idx in range(1, data.shape[0]):
        if not finite[frame_idx] or not finite[frame_idx - 1]:
            continue
        delta = centers[frame_idx] - centers[frame_idx - 1]
        if float(np.linalg.norm(delta)) <= threshold:
            continue
        shift = delta * 0.65
        data[frame_idx:, :, 0] -= shift[0]
        data[frame_idx:, :, 1] -= shift[1]
        centers[frame_idx:] -= shift
        corrections += 1
    return corrections


def _stabilize_feet(data: np.ndarray, names: List[str], velocity_threshold: float, contact_y: float) -> int:
    index = {name: idx for idx, name in enumerate(names)}
    corrections = 0
    for name in ("LEFT_ANKLE", "RIGHT_ANKLE"):
        landmark_idx = index.get(name)
        if landmark_idx is None or data.shape[0] < 3:
            continue
        points = data[:, landmark_idx, :2]
        finite = np.isfinite(points).all(axis=1)
        for frame_idx in range(1, data.shape[0]):
            if not finite[frame_idx] or not finite[frame_idx - 1]:
                continue
            velocity = float(np.linalg.norm(points[frame_idx] - points[frame_idx - 1]))
            if points[frame_idx, 1] < contact_y or velocity > velocity_threshold:
                continue
            data[frame_idx, landmark_idx, :2] = data[frame_idx - 1, landmark_idx, :2]
            points[frame_idx] = points[frame_idx - 1]
            corrections += 1
    return corrections


def _quality_warnings(metrics: Dict[str, float]) -> List[str]:
    warnings: List[str] = []
    if metrics["detection_coverage"] < 0.85:
        warnings.append("Pose detection was intermittent; use a clearer full-body source video if the motion looks unstable.")
    if metrics["repaired_landmark_ratio"] > 0.18:
        warnings.append("Many low-confidence landmarks were repaired; review the source pose overlay before rendering.")
    if metrics["left_right_corrections"] > 0:
        warnings.append("Possible left/right landmark swaps were corrected in the source motion.")
    if metrics["root_jump_corrections"] > 0:
        warnings.append("Large root-motion jumps were smoothed; fast camera motion may still affect the result.")
    if metrics["foot_stabilizations"] > 8:
        warnings.append("Foot contact stabilization was applied; check for foot sliding in the rendered animation.")
    return warnings
