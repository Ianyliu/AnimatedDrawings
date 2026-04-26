"""Pose estimator implementations."""

from __future__ import annotations

import os
import pickle
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from animated_drawings.video_pose.constants import MEDIAPIPE_REQUIRED_LANDMARKS
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


class RandomForestPoseEstimator:
    """scikit-learn artifact-backed pose estimator.

    The model artifact can be either a plain estimator or a dict with:
    model, landmark_names, and input_size. Predictions must return flattened
    landmark values in x/y/z[/visibility] order for each landmark.
    """

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        self.model, self.landmark_names, self.input_size = _load_model_artifact(self.model_path)

    def estimate(self, video_path: Path, max_seconds: int = 10) -> PoseSequence:
        metadata = validate_video_duration(video_path, max_seconds)
        features = _video_features(video_path, metadata.fps, max_seconds, self.input_size)
        if features.size == 0:
            raise PoseVideoError("No readable video frames were available for pose estimation.")
        predictions = np.asarray(self.model.predict(features), dtype=np.float32)
        frames = _predictions_to_frames(predictions, self.landmark_names, metadata.fps)
        return PoseSequence(
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            landmark_names=list(self.landmark_names),
            frames=frames,
        )


class CatBoostPoseEstimator:
    """CatBoost artifact-backed pose estimator."""

    def __init__(self, model_path: Path) -> None:
        self.model_path = Path(model_path)
        try:
            from catboost import CatBoostRegressor
        except ImportError as e:
            raise PoseVideoError(
                "CatBoost pose estimation requires the optional catboost dependency. "
                "Install animated_drawings[catboost] or choose MediaPipe."
            ) from e

        if not self.model_path.exists():
            raise PoseVideoError(f"CatBoost model artifact was not found: {self.model_path}")
        self.model = CatBoostRegressor()
        self.model.load_model(str(self.model_path))
        self.landmark_names = list(MEDIAPIPE_REQUIRED_LANDMARKS)
        self.input_size = (64, 64)

    def estimate(self, video_path: Path, max_seconds: int = 10) -> PoseSequence:
        metadata = validate_video_duration(video_path, max_seconds)
        features = _video_features(video_path, metadata.fps, max_seconds, self.input_size)
        if features.size == 0:
            raise PoseVideoError("No readable video frames were available for pose estimation.")
        predictions = np.asarray(self.model.predict(features), dtype=np.float32)
        frames = _predictions_to_frames(predictions, self.landmark_names, metadata.fps)
        return PoseSequence(
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
            landmark_names=list(self.landmark_names),
            frames=frames,
        )


def create_pose_estimator(name: str = "mediapipe", config: Optional[dict[str, Any]] = None):
    config = config or {}
    estimator_name = (name or "mediapipe").strip().lower()
    if estimator_name == "mediapipe":
        return MediaPipePoseEstimator()
    if estimator_name == "random_forest":
        model_path = config.get("random_forest_model") or config.get("model_path")
        if not model_path:
            raise PoseVideoError("Random Forest pose estimation requires VIDEO_APP_RANDOM_FOREST_MODEL.")
        return RandomForestPoseEstimator(Path(model_path))
    if estimator_name == "catboost":
        model_path = config.get("catboost_model") or config.get("model_path")
        if not model_path:
            raise PoseVideoError("CatBoost pose estimation requires VIDEO_APP_CATBOOST_MODEL.")
        return CatBoostPoseEstimator(Path(model_path))
    raise PoseVideoError(f"Unknown pose estimator: {name}")


def available_pose_estimators(config: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
    config = config or {}
    estimators = [
        {"id": "mediapipe", "name": "MediaPipe", "configured": True, "available": True},
    ]
    rf_model = config.get("random_forest_model")
    estimators.append(
        {
            "id": "random_forest",
            "name": "Random Forest",
            "configured": bool(rf_model),
            "available": bool(rf_model and Path(rf_model).exists()),
        }
    )
    cb_model = config.get("catboost_model")
    estimators.append(
        {
            "id": "catboost",
            "name": "CatBoost",
            "configured": bool(cb_model),
            "available": bool(cb_model and Path(cb_model).exists() and _module_available("catboost")),
        }
    )
    return estimators


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


def _load_model_artifact(model_path: Path):
    if not model_path.exists():
        raise PoseVideoError(f"Pose model artifact was not found: {model_path}")
    try:
        import joblib

        artifact = joblib.load(model_path)
    except Exception:
        with model_path.open("rb") as f:
            artifact = pickle.load(f)

    if isinstance(artifact, dict):
        model = artifact.get("model")
        landmark_names = artifact.get("landmark_names") or MEDIAPIPE_REQUIRED_LANDMARKS
        input_size = tuple(artifact.get("input_size") or (64, 64))
    else:
        model = artifact
        landmark_names = MEDIAPIPE_REQUIRED_LANDMARKS
        input_size = (64, 64)
    if model is None or not hasattr(model, "predict"):
        raise PoseVideoError("Pose model artifact must contain an object with a predict() method.")
    if len(input_size) != 2:
        raise PoseVideoError("Pose model input_size must be a two-value width/height tuple.")
    return model, list(landmark_names), (int(input_size[0]), int(input_size[1]))


def _video_features(video_path: Path, fps: float, max_seconds: int, input_size: tuple[int, int]) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PoseVideoError(f"Could not open video: {video_path}")
    max_frames = int(round(fps * max_seconds)) if fps > 0 else None
    features: List[np.ndarray] = []
    try:
        frame_idx = 0
        while max_frames is None or frame_idx < max_frames:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(gray, input_size, interpolation=cv2.INTER_AREA)
            features.append((resized.astype(np.float32) / 255.0).reshape(-1))
            frame_idx += 1
    finally:
        cap.release()
    return np.vstack(features) if features else np.empty((0, input_size[0] * input_size[1]), dtype=np.float32)


def _predictions_to_frames(predictions: np.ndarray, landmark_names: List[str], fps: float) -> List[PoseFrame]:
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
    values_per_landmark = 4 if predictions.shape[1] == len(landmark_names) * 4 else 3
    expected = len(landmark_names) * values_per_landmark
    if predictions.shape[1] != expected:
        raise PoseVideoError(
            f"Pose model produced {predictions.shape[1]} values per frame; expected {expected}."
        )
    reshaped = predictions.reshape(predictions.shape[0], len(landmark_names), values_per_landmark)
    frames: List[PoseFrame] = []
    frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0
    for frame_idx, frame_values in enumerate(reshaped):
        landmarks = {}
        for landmark_idx, name in enumerate(landmark_names):
            values = frame_values[landmark_idx]
            visibility = float(values[3]) if values_per_landmark == 4 else 1.0
            landmarks[name] = [float(values[0]), float(values[1]), float(values[2]), visibility]
        frames.append(PoseFrame(timestamp=frame_idx * frame_time, landmarks=landmarks))
    return frames


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
