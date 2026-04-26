from pathlib import Path
import subprocess

import cv2
import numpy as np
import pytest
import yaml

from animated_drawings.model.bvh import BVH
from animated_drawings.video_pose import PoseFrame, PoseSequence, PoseToBvhConverter, PoseVideoError
from animated_drawings.video_pose.constants import MEDIAPIPE_REQUIRED_LANDMARKS
from animated_drawings.video_pose.types import VideoDurationError
from animated_drawings.video_pose.video import _select_metadata_fps, transcode_to_browser_mp4, validate_video_duration
import animated_drawings.video_pose.video as video_helpers


def test_pose_sequence_to_bvh_round_trip(tmp_path: Path):
    sequence = PoseSequence(
        fps=30.0,
        width=640,
        height=480,
        landmark_names=MEDIAPIPE_REQUIRED_LANDMARKS,
        frames=[_pose_frame(0.0), _pose_frame(0.04), _pose_frame(0.08)],
    )
    bvh_path = tmp_path / "motion.bvh"
    motion_cfg_path = tmp_path / "motion.yaml"

    PoseToBvhConverter().convert(sequence, bvh_path, motion_cfg_path)

    bvh = BVH.from_file(str(bvh_path))
    assert bvh.frame_max_num == 3
    assert bvh.frame_time == pytest.approx(1.0 / 30.0)
    assert "LeftShoulder" in bvh.get_joint_names()
    assert "RightWrist" in bvh.get_joint_names()

    with motion_cfg_path.open("r") as f:
        motion_cfg = yaml.safe_load(f)
    assert motion_cfg["filepath"] == str(bvh_path.resolve())
    assert motion_cfg["groundplane_joint"] == "LeftAnkle"
    assert motion_cfg["up"] == "+z"


def test_video_duration_validation_rejects_long_video(tmp_path: Path):
    video_path = tmp_path / "long.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (32, 32))
    if not writer.isOpened():
        pytest.skip("OpenCV could not create an mp4 test file")
    try:
        frame = np.zeros((32, 32, 3), dtype=np.uint8)
        for _ in range(11):
            writer.write(frame)
    finally:
        writer.release()

    with pytest.raises(VideoDurationError):
        validate_video_duration(video_path, max_seconds=10)


def test_unreliable_webcam_fps_is_derived_from_duration():
    fps = _select_metadata_fps(
        raw_fps=1000.0,
        frame_count=130,
        probed_duration=4.37,
        probed_fps=None,
    )

    assert fps == pytest.approx(29.75, abs=0.01)


def test_strict_browser_transcode_requires_ffmpeg(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "input.mp4"
    video_path.write_bytes(b"mp4")
    monkeypatch.setattr(video_helpers.shutil, "which", lambda name: None)

    with pytest.raises(PoseVideoError, match="ffmpeg is required"):
        transcode_to_browser_mp4(video_path, strict=True)


def test_strict_browser_transcode_surfaces_timeout(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "input.mp4"
    output_path = tmp_path / "output.mp4"
    video_path.write_bytes(b"mp4")
    monkeypatch.setattr(video_helpers.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    def fake_run(command, capture_output, text, timeout):
        raise subprocess.TimeoutExpired(command, timeout=timeout)

    monkeypatch.setattr(video_helpers.subprocess, "run", fake_run)

    with pytest.raises(PoseVideoError, match="timed out"):
        transcode_to_browser_mp4(video_path, output_path, strict=True, timeout=1.0)


def _pose_frame(offset: float) -> PoseFrame:
    landmarks = {
        "NOSE": [0.50, 0.18, 0.0, 1.0],
        "LEFT_SHOULDER": [0.38, 0.34, 0.0, 1.0],
        "RIGHT_SHOULDER": [0.62, 0.34, 0.0, 1.0],
        "LEFT_ELBOW": [0.30 - offset, 0.48 - offset, 0.0, 1.0],
        "RIGHT_ELBOW": [0.70 + offset, 0.48, 0.0, 1.0],
        "LEFT_WRIST": [0.26 - offset, 0.62 - offset, 0.0, 1.0],
        "RIGHT_WRIST": [0.74 + offset, 0.62, 0.0, 1.0],
        "LEFT_HIP": [0.43, 0.62, 0.0, 1.0],
        "RIGHT_HIP": [0.57, 0.62, 0.0, 1.0],
        "LEFT_KNEE": [0.42, 0.78, 0.0, 1.0],
        "RIGHT_KNEE": [0.58, 0.78, 0.0, 1.0],
        "LEFT_ANKLE": [0.41, 0.94, 0.0, 1.0],
        "RIGHT_ANKLE": [0.59, 0.94, 0.0, 1.0],
    }
    return PoseFrame(timestamp=offset, landmarks=landmarks)
