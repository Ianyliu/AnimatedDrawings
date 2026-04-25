"""Pose estimator implementations."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Dict, List

import cv2

from animated_drawings.video_pose.types import PoseFrame, PoseSequence, PoseVideoError
from animated_drawings.video_pose.video import validate_video_duration


class MediaPipePoseEstimator:
    """MediaPipe-backed human pose estimator.

    MediaPipe is imported lazily so code that only uses the BVH writer or tests
    can run without the optional runtime dependency installed.
    """

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

    def estimate(self, video_path: Path, max_seconds: int = 10) -> PoseSequence:
        mpl_cache_dir = Path(tempfile.gettempdir()) / "animated_drawings_mpl"
        mpl_cache_dir.mkdir(exist_ok=True, parents=True)
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

        try:
            import mediapipe as mp
        except ImportError as e:
            raise PoseVideoError(
                "MediaPipe is not installed. Install the video app dependencies before estimating video pose."
            ) from e

        metadata = validate_video_duration(video_path, max_seconds)
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise PoseVideoError(f"Could not open video: {video_path}")

        mp_pose = mp.solutions.pose
        landmark_names = [landmark.name for landmark in mp_pose.PoseLandmark]
        frames: List[PoseFrame] = []

        try:
            with mp_pose.Pose(
                static_image_mode=False,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            ) as pose:
                frame_idx = 0
                max_frames = int(round(metadata.fps * max_seconds)) if metadata.fps > 0 else None
                while max_frames is None or frame_idx < max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose.process(rgb)
                    landmarks: Dict[str, List[float]] = {}
                    if results.pose_landmarks:
                        for idx, landmark in enumerate(results.pose_landmarks.landmark):
                            landmarks[landmark_names[idx]] = [
                                float(landmark.x),
                                float(landmark.y),
                                float(landmark.z),
                                float(getattr(landmark, "visibility", 0.0)),
                            ]

                    frames.append(PoseFrame(timestamp=frame_idx / metadata.fps, landmarks=landmarks))
                    frame_idx += 1
        finally:
            cap.release()

        _fill_missing_landmark_frames(frames)
        if not frames or not frames[0].landmarks:
            raise PoseVideoError("No human pose was detected in the video.")

        return PoseSequence(
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            landmark_names=landmark_names,
            frames=frames,
        )


def _fill_missing_landmark_frames(frames: List[PoseFrame]) -> None:
    first_valid = next((frame.landmarks for frame in frames if frame.landmarks), None)
    if first_valid is None:
        return

    last_valid = first_valid
    for frame in frames:
        if frame.landmarks:
            last_valid = frame.landmarks
        else:
            frame.landmarks = {name: list(values) for name, values in last_valid.items()}

    first_idx = next(idx for idx, frame in enumerate(frames) if frame.landmarks)
    for frame in frames[:first_idx]:
        frame.landmarks = {name: list(values) for name, values in first_valid.items()}
