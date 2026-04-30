"""Live webcam pose helpers for driving drawings without BVH files."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import numpy.typing as npt

from animated_drawings.config import RetargetConfig
from animated_drawings.video_pose.constants import BVH_ROTATION_JOINTS, MEDIAPIPE_REQUIRED_LANDMARKS, POSE_CONNECTIONS
from animated_drawings.video_pose.types import PoseFrame, PoseVideoError


TRACKING_COLOR = (70, 190, 90)
PARTIAL_COLOR = (0, 185, 255)
LOST_COLOR = (75, 75, 230)
PAUSED_COLOR = (210, 165, 55)
CAMERA_ERROR_COLOR = (55, 55, 220)


@dataclass(frozen=True)
class PoseTrackingStatus:
    state: str
    message: str
    missing_landmarks: Tuple[str, ...] = ()
    color: Tuple[int, int, int] = TRACKING_COLOR


@dataclass(frozen=True)
class CausalPoseSmootherConfig:
    alpha: float = 0.35
    visibility_threshold: float = 0.35


def analyze_pose_frame(
    frame: PoseFrame,
    required_landmarks: Optional[Sequence[str]] = None,
    visibility_threshold: float = 0.35,
) -> PoseTrackingStatus:
    required = tuple(required_landmarks or MEDIAPIPE_REQUIRED_LANDMARKS)
    if not frame.landmarks:
        return PoseTrackingStatus(
            state="lost",
            message="No pose detected. Step into view.",
            missing_landmarks=required,
            color=LOST_COLOR,
        )

    missing = tuple(
        name
        for name in required
        if name not in frame.landmarks or not _landmark_valid(frame.landmarks[name], visibility_threshold)
    )
    if missing:
        missing_groups = _friendly_missing_groups(missing)
        missing_text = "/".join(missing_groups) if missing_groups else "required joints"
        return PoseTrackingStatus(
            state="partial",
            message=f"Step back: full body not in view. Missing {missing_text}.",
            missing_landmarks=missing,
            color=PARTIAL_COLOR,
        )

    return PoseTrackingStatus(
        state="tracking",
        message="Tracking full body.",
        missing_landmarks=(),
        color=TRACKING_COLOR,
    )


def paused_status() -> PoseTrackingStatus:
    return PoseTrackingStatus(
        state="paused",
        message="Paused. Press Space to resume.",
        color=PAUSED_COLOR,
    )


def camera_error_status() -> PoseTrackingStatus:
    return PoseTrackingStatus(
        state="camera_error",
        message="Could not read webcam frame. Check camera access.",
        color=CAMERA_ERROR_COLOR,
    )


def draw_pose_overlay(
    frame_bgr: npt.NDArray[np.uint8],
    pose_frame: PoseFrame,
    status: PoseTrackingStatus,
) -> npt.NDArray[np.uint8]:
    overlay = frame_bgr.copy()
    if not pose_frame.landmarks:
        return overlay

    height, width = overlay.shape[:2]
    color = tuple(int(channel) for channel in status.color)
    muted = (135, 135, 135)

    for start_name, end_name in POSE_CONNECTIONS:
        start = pose_frame.landmarks.get(start_name)
        end = pose_frame.landmarks.get(end_name)
        if start is None or end is None:
            continue
        if not _coords_finite(start) or not _coords_finite(end):
            continue
        cv2.line(
            overlay,
            _landmark_pixel(start, width, height),
            _landmark_pixel(end, width, height),
            color if _landmark_valid(start, 0.0) and _landmark_valid(end, 0.0) else muted,
            2,
            cv2.LINE_AA,
        )

    missing = set(status.missing_landmarks)
    for name, values in pose_frame.landmarks.items():
        if not _coords_finite(values):
            continue
        point_color = muted if name in missing else color
        cv2.circle(overlay, _landmark_pixel(values, width, height), 4, point_color, -1, cv2.LINE_AA)

    return overlay


def compose_live_dashboard(
    camera_bgr: npt.NDArray[np.uint8],
    animation_bgr: npt.NDArray[np.uint8],
    status: PoseTrackingStatus,
    *,
    pane_size: int,
    paused: bool = False,
    active_figure: Optional[str] = None,
    controls: Optional[str] = None,
) -> npt.NDArray[np.uint8]:
    top_h = 64
    bottom_h = 48
    gap = 16
    margin = 16
    pane_h = pane_size
    pane_w = pane_size
    width = margin * 2 + pane_w * 2 + gap
    height = top_h + pane_h + bottom_h + margin
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (width, top_h), (34, 37, 42), -1)
    cv2.rectangle(canvas, (0, 0), (10, top_h), status.color, -1)
    state_text = status.state.replace("_", " ").upper()
    if paused and status.state != "paused":
        state_text = f"{state_text} / PAUSED"
    _put_text(canvas, state_text, (margin, 24), 0.58, (255, 255, 255), thickness=2)
    _put_text(canvas, status.message, (margin, 50), 0.58, (224, 229, 235), thickness=1)

    left_x = margin
    right_x = margin + pane_w + gap
    pane_y = top_h
    _paste(canvas, _fit_to_square(camera_bgr, pane_size), left_x, pane_y)
    _paste(canvas, _fit_to_square(animation_bgr, pane_size), right_x, pane_y)

    cv2.rectangle(canvas, (left_x, pane_y), (left_x + pane_w, pane_y + pane_h), (35, 35, 35), 1)
    cv2.rectangle(canvas, (right_x, pane_y), (right_x + pane_w, pane_y + pane_h), (35, 35, 35), 1)
    _put_label(canvas, "Webcam Pose", (left_x + 12, pane_y + 28))
    animation_label = "Animated Drawing"
    if active_figure:
        animation_label = f"Figure: {active_figure}"
    _put_label(canvas, animation_label, (right_x + 12, pane_y + 28))

    controls_y = top_h + pane_h + 32
    controls = controls or "Space pause/resume   R reset pose   Q/Esc quit   Keep full body in frame"
    _put_text(canvas, controls, (margin, controls_y), 0.58, (45, 48, 53), thickness=1)

    return canvas


class LiveMediaPipePoseEstimator:
    """MediaPipe pose estimator for one webcam frame at a time."""

    def __init__(
        self,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as e:
            raise PoseVideoError(
                "MediaPipe is not installed. Install the video app dependencies before estimating webcam pose."
            ) from e

        self._mp_pose = mp.solutions.pose
        self.landmark_names = [landmark.name for landmark in self._mp_pose.PoseLandmark]
        self._pose = self._mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def estimate_frame(self, frame_bgr: npt.NDArray[np.uint8], timestamp: float) -> PoseFrame:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._pose.process(rgb)
        landmarks: Dict[str, List[float]] = {}
        if results.pose_landmarks:
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                landmarks[self.landmark_names[idx]] = [
                    float(landmark.x),
                    float(landmark.y),
                    float(landmark.z),
                    float(getattr(landmark, "visibility", 0.0)),
                ]
        return PoseFrame(timestamp=timestamp, landmarks=landmarks)

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "LiveMediaPipePoseEstimator":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


class CausalPoseSmoother:
    """One-way landmark repair and EMA smoothing for live pose streams."""

    def __init__(self, config: Optional[CausalPoseSmootherConfig] = None) -> None:
        self.config = config or CausalPoseSmootherConfig()
        self._last_good: Dict[str, List[float]] = {}

    def reset(self) -> None:
        self._last_good.clear()

    def process(self, frame: PoseFrame) -> PoseFrame:
        smoothed: Dict[str, List[float]] = {}
        names = set(self._last_good)
        names.update(frame.landmarks)

        for name in names:
            values = frame.landmarks.get(name)
            previous = self._last_good.get(name)
            if values is None or not _landmark_valid(values, self.config.visibility_threshold):
                if previous is not None:
                    smoothed[name] = list(previous)
                continue

            current = _pad_landmark(values)
            if previous is not None:
                repaired = [
                    self.config.alpha * current[idx] + (1.0 - self.config.alpha) * previous[idx]
                    for idx in range(3)
                ]
                repaired.append(current[3])
            else:
                repaired = current

            smoothed[name] = repaired
            self._last_good[name] = list(repaired)

        return PoseFrame(timestamp=frame.timestamp, landmarks=smoothed)


class LivePoseRetargeter:
    """Retargeter-compatible live pose driver.

    It implements the subset of the BVH Retargeter contract consumed by
    AnimatedDrawing.update(): bvh_joint_names and get_retargeted_frame_data().
    """

    def __init__(
        self,
        retarget_cfg: RetargetConfig,
        *,
        root_mode: str = "locked",
        depth_mode: str = "flat",
        visibility_threshold: float = 0.35,
        root_scale: float = 1.0,
    ) -> None:
        if root_mode not in ("locked", "hip"):
            raise ValueError("root_mode must be 'locked' or 'hip'.")
        if depth_mode not in ("flat", "mediapipe-z"):
            raise ValueError("depth_mode must be 'flat' or 'mediapipe-z'.")

        self.retarget_cfg = retarget_cfg
        self.root_mode = root_mode
        self.depth_mode = depth_mode
        self.visibility_threshold = visibility_threshold
        self.root_scale = float(root_scale)
        self.bvh_joint_names = list(BVH_ROTATION_JOINTS)
        self.character_start_loc: npt.NDArray[np.float32] = np.array(retarget_cfg.char_start_loc, dtype=np.float32)

        self._root_reference: Optional[npt.NDArray[np.float32]] = None
        self._last_positions: Optional[Dict[str, npt.NDArray[np.float32]]] = None
        self._current_orientations: Dict[str, float] = {}
        self._current_depths: Dict[str, float] = {name: 0.0 for name in self.bvh_joint_names}
        self._current_root_position: npt.NDArray[np.float32] = np.array(self.character_start_loc, dtype=np.float32)

        self._set_current_from_positions(_neutral_positions(), initialize_reference=False)

    def reset_root_reference(self) -> None:
        self._root_reference = None
        if self._last_positions is not None:
            self._root_reference = np.array(self._last_positions["Hip"][:2], dtype=np.float32)
            self._current_root_position = np.array(self.character_start_loc, dtype=np.float32)

    def update_pose(self, frame: PoseFrame) -> bool:
        positions = self._positions_from_frame(frame)
        if positions is None:
            return False
        self._set_current_from_positions(positions, initialize_reference=True)
        return True

    def get_retargeted_frame_data(
        self, time: float
    ) -> tuple[Dict[str, float], Dict[str, float], npt.NDArray[np.float32]]:
        return (
            dict(self._current_orientations),
            dict(self._current_depths),
            np.array(self._current_root_position, dtype=np.float32),
        )

    def _set_current_from_positions(
        self,
        positions: Dict[str, npt.NDArray[np.float32]],
        *,
        initialize_reference: bool,
    ) -> None:
        self._last_positions = positions
        self._current_orientations = self._orientations_from_positions(positions)
        self._current_depths = self._depths_from_positions(positions)
        self._current_root_position = self._root_from_positions(positions, initialize_reference=initialize_reference)

    def _positions_from_frame(self, frame: PoseFrame) -> Optional[Dict[str, npt.NDArray[np.float32]]]:
        return _positions_from_landmarks(frame.landmarks, self.visibility_threshold)

    def _orientations_from_positions(self, positions: Dict[str, npt.NDArray[np.float32]]) -> Dict[str, float]:
        orientations: Dict[str, float] = {}
        for char_joint_name, (prox_name, dist_name) in self.retarget_cfg.char_joint_bvh_joints_mapping.items():
            prox = positions.get(prox_name)
            dist = positions.get(dist_name)
            if prox is None or dist is None:
                continue
            orientations[char_joint_name] = _orientation_degrees(prox, dist)
        return orientations

    def _depths_from_positions(self, positions: Dict[str, npt.NDArray[np.float32]]) -> Dict[str, float]:
        if self.depth_mode == "flat":
            return {name: 0.0 for name in self.bvh_joint_names}
        return {name: float(positions.get(name, np.zeros(3, dtype=np.float32))[2]) for name in self.bvh_joint_names}

    def _root_from_positions(
        self,
        positions: Dict[str, npt.NDArray[np.float32]],
        *,
        initialize_reference: bool,
    ) -> npt.NDArray[np.float32]:
        if self.root_mode == "locked":
            return np.array(self.character_start_loc, dtype=np.float32)

        hip_xy = np.array(positions["Hip"][:2], dtype=np.float32)
        if self._root_reference is None:
            if initialize_reference:
                self._root_reference = hip_xy
            return np.array(self.character_start_loc, dtype=np.float32)

        delta = (hip_xy - self._root_reference) * self.root_scale
        return np.array(
            [
                self.character_start_loc[0] + delta[0],
                self.character_start_loc[1] + delta[1],
                self.character_start_loc[2],
            ],
            dtype=np.float32,
        )


def _landmark_valid(values: List[float], visibility_threshold: float) -> bool:
    if len(values) < 3:
        return False
    coords = values[:3]
    if not all(math.isfinite(float(value)) for value in coords):
        return False
    visibility = float(values[3]) if len(values) >= 4 else 1.0
    return math.isfinite(visibility) and visibility >= visibility_threshold


def _coords_finite(values: List[float]) -> bool:
    return len(values) >= 2 and math.isfinite(float(values[0])) and math.isfinite(float(values[1]))


def _landmark_pixel(values: List[float], width: int, height: int) -> Tuple[int, int]:
    x = int(round(float(values[0]) * (width - 1)))
    y = int(round(float(values[1]) * (height - 1)))
    return int(np.clip(x, 0, width - 1)), int(np.clip(y, 0, height - 1))


def _friendly_missing_groups(missing_landmarks: Sequence[str]) -> List[str]:
    groups = [
        ("head", ("NOSE",)),
        ("shoulders", ("LEFT_SHOULDER", "RIGHT_SHOULDER")),
        ("elbows", ("LEFT_ELBOW", "RIGHT_ELBOW")),
        ("hands", ("LEFT_WRIST", "RIGHT_WRIST")),
        ("hips", ("LEFT_HIP", "RIGHT_HIP")),
        ("knees", ("LEFT_KNEE", "RIGHT_KNEE")),
        ("ankles", ("LEFT_ANKLE", "RIGHT_ANKLE")),
    ]
    missing = set(missing_landmarks)
    return [label for label, names in groups if any(name in missing for name in names)]


def _fit_to_square(frame_bgr: npt.NDArray[np.uint8], size: int) -> npt.NDArray[np.uint8]:
    if frame_bgr.size == 0:
        return np.full((size, size, 3), 235, dtype=np.uint8)
    height, width = frame_bgr.shape[:2]
    scale = min(size / max(1, width), size / max(1, height))
    out_w = max(1, int(round(width * scale)))
    out_h = max(1, int(round(height * scale)))
    resized = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
    square = np.full((size, size, 3), 232, dtype=np.uint8)
    x = (size - out_w) // 2
    y = (size - out_h) // 2
    square[y : y + out_h, x : x + out_w] = resized
    return square


def _paste(canvas: npt.NDArray[np.uint8], image: npt.NDArray[np.uint8], x: int, y: int) -> None:
    height, width = image.shape[:2]
    canvas[y : y + height, x : x + width] = image


def _put_label(canvas: npt.NDArray[np.uint8], text: str, origin: Tuple[int, int]) -> None:
    x, y = origin
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.rectangle(canvas, (x - 8, y - text_h - 10), (x + text_w + 8, y + 8), (30, 32, 36), -1)
    _put_text(canvas, text, (x, y), 0.62, (255, 255, 255), thickness=2)


def _put_text(
    canvas: npt.NDArray[np.uint8],
    text: str,
    origin: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    cv2.putText(canvas, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _pad_landmark(values: List[float]) -> List[float]:
    padded = [float(value) for value in values[:4]]
    while len(padded) < 4:
        padded.append(1.0 if len(padded) == 3 else 0.0)
    return padded


def _landmark_to_live_point(values: List[float]) -> npt.NDArray[np.float32]:
    return np.array([float(values[0]), 1.0 - float(values[1]), float(values[2])], dtype=np.float32)


def _positions_from_landmarks(
    landmarks: Dict[str, List[float]],
    visibility_threshold: float,
) -> Optional[Dict[str, npt.NDArray[np.float32]]]:
    if not all(
        name in landmarks and _landmark_valid(landmarks[name], visibility_threshold)
        for name in MEDIAPIPE_REQUIRED_LANDMARKS
    ):
        return None

    raw = {name: _landmark_to_live_point(landmarks[name]) for name in MEDIAPIPE_REQUIRED_LANDMARKS}
    hip = (raw["LEFT_HIP"] + raw["RIGHT_HIP"]) / 2.0
    thorax = (raw["LEFT_SHOULDER"] + raw["RIGHT_SHOULDER"]) / 2.0
    nose = raw["NOSE"]
    neck = thorax + 0.35 * (nose - thorax)

    return {
        "Hip": hip,
        "RightHip": raw["RIGHT_HIP"],
        "RightKnee": raw["RIGHT_KNEE"],
        "RightAnkle": raw["RIGHT_ANKLE"],
        "LeftHip": raw["LEFT_HIP"],
        "LeftKnee": raw["LEFT_KNEE"],
        "LeftAnkle": raw["LEFT_ANKLE"],
        "Spine": hip + 0.5 * (thorax - hip),
        "Thorax": thorax,
        "Neck": neck,
        "LeftShoulder": raw["LEFT_SHOULDER"],
        "LeftElbow": raw["LEFT_ELBOW"],
        "LeftWrist": raw["LEFT_WRIST"],
        "RightShoulder": raw["RIGHT_SHOULDER"],
        "RightElbow": raw["RIGHT_ELBOW"],
        "RightWrist": raw["RIGHT_WRIST"],
    }


def _orientation_degrees(prox: npt.NDArray[np.float32], dist: npt.NDArray[np.float32]) -> float:
    vec = dist[:2] - prox[:2]
    norm = float(np.linalg.norm(vec))
    if norm < 1e-6:
        return 0.0
    vec = vec / norm
    theta = math.degrees(math.atan2(float(vec[1]), float(vec[0])) - math.atan2(1.0, 0.0))
    theta = theta % 360.0
    if theta < 0.0:
        theta += 360.0
    return float(theta)


def _neutral_positions() -> Dict[str, npt.NDArray[np.float32]]:
    frame = PoseFrame(
        timestamp=0.0,
        landmarks={
            "NOSE": [0.50, 0.18, 0.0, 1.0],
            "LEFT_SHOULDER": [0.38, 0.34, 0.0, 1.0],
            "RIGHT_SHOULDER": [0.62, 0.34, 0.0, 1.0],
            "LEFT_ELBOW": [0.30, 0.48, 0.0, 1.0],
            "RIGHT_ELBOW": [0.70, 0.48, 0.0, 1.0],
            "LEFT_WRIST": [0.26, 0.62, 0.0, 1.0],
            "RIGHT_WRIST": [0.74, 0.62, 0.0, 1.0],
            "LEFT_HIP": [0.43, 0.62, 0.0, 1.0],
            "RIGHT_HIP": [0.57, 0.62, 0.0, 1.0],
            "LEFT_KNEE": [0.42, 0.78, 0.0, 1.0],
            "RIGHT_KNEE": [0.58, 0.78, 0.0, 1.0],
            "LEFT_ANKLE": [0.41, 0.94, 0.0, 1.0],
            "RIGHT_ANKLE": [0.59, 0.94, 0.0, 1.0],
        },
    )
    positions = _positions_from_landmarks(frame.landmarks, visibility_threshold=0.0)
    assert positions is not None
    return positions
