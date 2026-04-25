"""High-level video-to-motion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from animated_drawings.video_pose.bvh import PoseToBvhConverter
from animated_drawings.video_pose.constants import DEFAULT_MAX_SECONDS
from animated_drawings.video_pose.estimators import MediaPipePoseEstimator
from animated_drawings.video_pose.types import MotionBuildResult, PoseEstimator
from animated_drawings.video_pose.video import write_pose_overlay


def build_motion_from_video(
    video_path: Path,
    out_dir: Path,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    estimator: Optional[PoseEstimator] = None,
) -> MotionBuildResult:
    out_dir.mkdir(exist_ok=True, parents=True)

    pose_estimator = estimator or MediaPipePoseEstimator()
    sequence = pose_estimator.estimate(video_path, max_seconds=max_seconds)

    pose_sequence_path = out_dir / "pose_sequence.json"
    overlay_video_path = out_dir / "pose_overlay.mp4"
    bvh_path = out_dir / "motion.bvh"
    motion_config_path = out_dir / "motion.yaml"

    sequence.write_json(pose_sequence_path)
    write_pose_overlay(video_path, sequence, overlay_video_path, max_seconds=max_seconds)
    PoseToBvhConverter().convert(sequence, bvh_path, motion_config_path)

    return MotionBuildResult(
        pose_sequence_path=pose_sequence_path,
        overlay_video_path=overlay_video_path,
        bvh_path=bvh_path,
        motion_config_path=motion_config_path,
    )
