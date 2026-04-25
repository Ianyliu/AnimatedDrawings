"""Typed data structures for video pose estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol


class PoseVideoError(RuntimeError):
    """Base exception for video pose pipeline failures."""


class VideoDurationError(PoseVideoError, ValueError):
    """Raised when a video exceeds the configured duration limit."""


@dataclass
class PoseFrame:
    timestamp: float
    landmarks: Dict[str, List[float]]

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "landmarks": self.landmarks,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PoseFrame":
        return cls(
            timestamp=float(data["timestamp"]),
            landmarks={name: [float(v) for v in values] for name, values in data["landmarks"].items()},
        )


@dataclass
class PoseSequence:
    fps: float
    width: int
    height: int
    landmark_names: List[str]
    frames: List[PoseFrame]

    @property
    def frame_count(self) -> int:
        return len(self.frames)

    @property
    def frame_time(self) -> float:
        return 1.0 / self.fps if self.fps > 0 else 1.0 / 30.0

    @property
    def duration(self) -> float:
        return self.frame_count * self.frame_time

    def to_dict(self) -> dict:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "landmark_names": self.landmark_names,
            "frames": [frame.to_dict() for frame in self.frames],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PoseSequence":
        return cls(
            fps=float(data["fps"]),
            width=int(data["width"]),
            height=int(data["height"]),
            landmark_names=list(data["landmark_names"]),
            frames=[PoseFrame.from_dict(frame) for frame in data["frames"]],
        )

    def write_json(self, output_path: Path) -> None:
        output_path.parent.mkdir(exist_ok=True, parents=True)
        with output_path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def read_json(cls, input_path: Path) -> "PoseSequence":
        with input_path.open("r") as f:
            return cls.from_dict(json.load(f))


@dataclass
class MotionBuildResult:
    pose_sequence_path: Optional[Path]
    overlay_video_path: Optional[Path]
    bvh_path: Path
    motion_config_path: Path


class PoseEstimator(Protocol):
    def estimate(self, video_path: Path, max_seconds: int = 10) -> PoseSequence:
        """Estimate a pose sequence from a video."""


class PosePostprocessor(Protocol):
    def process(self, sequence: PoseSequence) -> PoseSequence:
        """Refine or smooth a pose sequence."""
