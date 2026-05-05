"""High-level video-to-motion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from animated_drawings.video_pose.bvh import PoseToBvhConverter
from animated_drawings.video_pose.constants import DEFAULT_MAX_SECONDS
from animated_drawings.video_pose.estimators import create_pose_estimator
from animated_drawings.video_pose.landmark_flow_corrector import (
    create_landmark_flow_corrector,
    merge_flow_metrics,
)
from animated_drawings.video_pose.postprocessors import DefaultPosePostprocessor, PosePostprocessConfig
from animated_drawings.video_pose.types import MotionBuildResult, PoseEstimator, PosePostprocessor, PoseQualityReport
from animated_drawings.video_pose.video import write_pose_overlay


def build_motion_from_video(
    video_path: Path,
    out_dir: Path,
    max_seconds: int = DEFAULT_MAX_SECONDS,
    estimator: Optional[PoseEstimator] = None,
    estimator_name: str = "mediapipe",
    estimator_config: Optional[dict[str, Any]] = None,
    postprocessor: Optional[PosePostprocessor] = None,
    landmark_flow_model: Optional[Path] = None,
    landmark_flow_enabled: Optional[bool] = None,
    landmark_flow_threshold: float = 0.5,
) -> MotionBuildResult:
    out_dir.mkdir(exist_ok=True, parents=True)

    pose_estimator = estimator or create_pose_estimator(estimator_name, estimator_config)
    sequence = pose_estimator.estimate(video_path, max_seconds=max_seconds)

    flow_corrector, flow_metrics = create_landmark_flow_corrector(
        landmark_flow_model,
        threshold=landmark_flow_threshold,
        enabled=landmark_flow_enabled,
    )
    if flow_corrector is not None:
        sequence, flow_metrics = flow_corrector.correct_sequence(sequence)

    pose_postprocessor = postprocessor or DefaultPosePostprocessor(
        PosePostprocessConfig(visibility_threshold=landmark_flow_threshold)
    )
    postprocessed = pose_postprocessor.process(sequence)
    if isinstance(postprocessed, tuple):
        sequence, quality_report = postprocessed
    else:
        sequence = postprocessed
        quality_report = sequence.quality_report or PoseQualityReport(warnings=[], metrics={})
    quality_report.metrics = merge_flow_metrics(quality_report.metrics, flow_metrics)
    sequence.quality_report = quality_report

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
        quality_report=quality_report,
    )
