"""Video helpers for the pose estimation pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2

from animated_drawings.video_pose.constants import POSE_CONNECTIONS
from animated_drawings.video_pose.types import PoseSequence, PoseVideoError, VideoDurationError


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int

    @property
    def duration(self) -> float:
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def read_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise PoseVideoError(f"Could not open video: {video_path}")

        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0:
            fps = 30.0

        return VideoMetadata(
            fps=fps,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        )
    finally:
        cap.release()


def validate_video_duration(video_path: Path, max_seconds: int) -> VideoMetadata:
    metadata = read_video_metadata(video_path)
    if metadata.frame_count > 0 and metadata.duration > max_seconds + 0.05:
        raise VideoDurationError(
            f"Video duration is {metadata.duration:.2f}s; maximum allowed is {max_seconds}s."
        )
    return metadata


def write_pose_overlay(
    video_path: Path,
    sequence: PoseSequence,
    output_path: Path,
    max_seconds: int = 10,
    max_frames: Optional[int] = None,
) -> Path:
    """Write a video with the estimated pose overlaid on the source frames."""

    validate_video_duration(video_path, max_seconds)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise PoseVideoError(f"Could not open video: {video_path}")

        fps = sequence.fps if sequence.fps > 0 else float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or sequence.width)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or sequence.height)
        limit = max_frames or sequence.frame_count

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise PoseVideoError(f"Could not create overlay video: {output_path}")

        frame_idx = 0
        try:
            while frame_idx < limit:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_idx < sequence.frame_count:
                    _draw_pose(frame, sequence.frames[frame_idx])
                writer.write(frame)
                frame_idx += 1
        finally:
            writer.release()
    finally:
        cap.release()

    return output_path


def _draw_pose(frame, pose_frame) -> None:
    height, width = frame.shape[:2]

    for start_name, end_name in POSE_CONNECTIONS:
        start = pose_frame.landmarks.get(start_name)
        end = pose_frame.landmarks.get(end_name)
        if start is None or end is None:
            continue
        x1, y1 = _to_pixel(start, width, height)
        x2, y2 = _to_pixel(end, width, height)
        cv2.line(frame, (x1, y1), (x2, y2), (20, 220, 120), 2)

    for landmark in pose_frame.landmarks.values():
        x, y = _to_pixel(landmark, width, height)
        cv2.circle(frame, (x, y), 4, (20, 40, 240), -1)
        cv2.circle(frame, (x, y), 4, (255, 255, 255), 1)


def _to_pixel(landmark, width: int, height: int) -> tuple[int, int]:
    x = int(round(max(0.0, min(1.0, landmark[0])) * (width - 1)))
    y = int(round(max(0.0, min(1.0, landmark[1])) * (height - 1)))
    return x, y
