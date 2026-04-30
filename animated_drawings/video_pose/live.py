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


@dataclass(frozen=True)
class LiveDashboardLayout:
    width: int
    height: int
    margin: int
    gap: int
    top_h: int
    pane_y: int
    bottom_y: int
    upload_rect: Tuple[int, int, int, int]


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
    upload_status: Optional[str] = None,
    figures: Optional[Sequence[str]] = None,
    active_figure_index: Optional[int] = None,
) -> npt.NDArray[np.uint8]:
    layout = _dashboard_layout(pane_size)
    width = layout.width
    height = layout.height
    margin = layout.margin
    gap = layout.gap
    top_h = layout.top_h
    pane_y = layout.pane_y
    bottom_y = layout.bottom_y
    upload_rect = layout.upload_rect
    canvas = np.full((height, width, 3), (241, 243, 239), dtype=np.uint8)

    cv2.rectangle(canvas, (0, 0), (width, top_h), (32, 34, 38), -1)
    cv2.rectangle(canvas, (0, 0), (12, top_h), status.color, -1)
    state_text = status.state.replace("_", " ").upper()
    if paused and status.state != "paused":
        state_text = f"{state_text} / PAUSED"
    _draw_status_pill(canvas, state_text, status.color, (margin, 24))
    header_message = status.message
    if upload_status:
        header_message = f"{status.message}  |  {upload_status}"
    _put_fitted_text(
        canvas,
        header_message,
        (margin, 60),
        width - margin * 2,
        0.56,
        (226, 231, 235),
        thickness=1,
    )

    left_x = margin
    right_x = margin + pane_size + gap
    _paste(canvas, _fit_to_square(camera_bgr, pane_size), left_x, pane_y)
    _paste(canvas, _fit_to_square(animation_bgr, pane_size), right_x, pane_y)

    _draw_pane_frame(canvas, left_x, pane_y, pane_size, "Webcam Pose")
    animation_label = "Animated Drawing"
    if active_figure:
        animation_label = f"Figure: {active_figure}"
    _draw_pane_frame(canvas, right_x, pane_y, pane_size, animation_label)

    _draw_upload_tile(canvas, upload_rect, upload_status)
    _draw_figure_carousel(
        canvas,
        figures=figures,
        active_figure=active_figure,
        active_figure_index=active_figure_index,
        origin=(margin, bottom_y + 22),
        max_width=upload_rect[0] - margin - 16,
    )
    controls = controls or "Space pause   R reset   U upload   C choose rig   [/] figure   Q quit"
    _put_fitted_text(
        canvas,
        controls,
        (margin, bottom_y + 92),
        upload_rect[0] - margin - 18,
        0.54,
        (55, 58, 61),
        thickness=1,
    )

    return canvas


def live_dashboard_upload_rect(pane_size: int) -> Tuple[int, int, int, int]:
    """Return the upload tile hit area for the dashboard generated above."""

    return _dashboard_layout(pane_size).upload_rect


def _dashboard_layout(pane_size: int) -> LiveDashboardLayout:
    pane_size = max(120, int(pane_size))
    margin = 18
    gap = 18
    top_h = 82
    pane_y = top_h + 14
    bottom_h = 118
    width = margin * 2 + pane_size * 2 + gap
    bottom_y = pane_y + pane_size
    height = bottom_y + bottom_h
    upload_w = min(184, max(136, pane_size - 28))
    upload_h = 74
    upload_rect = (
        width - margin - upload_w,
        bottom_y + 22,
        width - margin,
        bottom_y + 22 + upload_h,
    )
    return LiveDashboardLayout(
        width=width,
        height=height,
        margin=margin,
        gap=gap,
        top_h=top_h,
        pane_y=pane_y,
        bottom_y=bottom_y,
        upload_rect=upload_rect,
    )


def _draw_status_pill(
    canvas: npt.NDArray[np.uint8],
    text: str,
    color: Tuple[int, int, int],
    origin: Tuple[int, int],
) -> None:
    x, y = origin
    scale = 0.52
    thickness = 2
    text_w, text_h = _text_size(text, scale, thickness)
    pad_x = 12
    pad_y = 7
    rect = (x, y - text_h - pad_y, x + text_w + pad_x * 2, y + pad_y)
    cv2.rectangle(canvas, rect[:2], rect[2:], _darken(color, 0.72), -1)
    cv2.rectangle(canvas, rect[:2], rect[2:], color, 1)
    _put_text(canvas, text, (x + pad_x, y), scale, (255, 255, 255), thickness=thickness)


def _draw_pane_frame(canvas: npt.NDArray[np.uint8], x: int, y: int, size: int, label: str) -> None:
    cv2.rectangle(canvas, (x - 1, y - 1), (x + size + 1, y + size + 1), (210, 213, 209), 1)
    cv2.rectangle(canvas, (x, y), (x + size, y + size), (42, 44, 48), 1)
    _put_label(canvas, label, (x + 12, y + 29), max_width=size - 24)


def _draw_upload_tile(
    canvas: npt.NDArray[np.uint8],
    rect: Tuple[int, int, int, int],
    upload_status: Optional[str],
) -> None:
    x1, y1, x2, y2 = rect
    accent = (58, 132, 230)
    fill = (255, 255, 255)
    if upload_status:
        lowered = upload_status.lower()
        if any(token in lowered for token in ("failed", "error", "could not", "unsupported", "too large")):
            accent = (65, 78, 214)
        elif any(token in lowered for token in ("ready", "added", "complete", "switched")):
            accent = TRACKING_COLOR
        elif any(token in lowered for token in ("analyzing", "upload", "preparing", "choose")):
            accent = PARTIAL_COLOR

    cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), (205, 209, 205), 1)
    cv2.rectangle(canvas, (x1, y1), (x1 + 7, y2), accent, -1)
    _put_text(canvas, "U", (x1 + 21, y1 + 31), 0.72, accent, thickness=2)
    _put_fitted_text(canvas, "Upload drawing", (x1 + 52, y1 + 28), x2 - x1 - 62, 0.55, (33, 36, 39), thickness=2)
    subtitle = "PNG, JPG, WebP"
    if upload_status:
        subtitle = upload_status
    _put_fitted_text(canvas, subtitle, (x1 + 21, y1 + 57), x2 - x1 - 32, 0.46, (86, 89, 92), thickness=1)


def _draw_figure_carousel(
    canvas: npt.NDArray[np.uint8],
    *,
    figures: Optional[Sequence[str]],
    active_figure: Optional[str],
    active_figure_index: Optional[int],
    origin: Tuple[int, int],
    max_width: int,
) -> None:
    x, y = origin
    _put_text(canvas, "Figures", (x, y), 0.48, (95, 98, 101), thickness=1)
    chip_x = x
    chip_y = y + 16
    names = list(figures or [])
    if not names and active_figure:
        names = [active_figure]
        active_figure_index = 0

    if not names:
        _put_fitted_text(canvas, "No figures loaded", (chip_x, chip_y + 25), max_width, 0.52, (70, 73, 76), thickness=1)
        return

    active_index = active_figure_index if active_figure_index is not None else 0
    visible = names[:9]
    for idx, name in enumerate(visible):
        label = f"{idx + 1} {name}"
        scale = 0.48
        thickness = 1
        text_w, text_h = _text_size(label, scale, thickness)
        chip_w = min(max(54, text_w + 22), 118)
        if chip_x + chip_w > x + max_width:
            remaining = len(visible) - idx
            if remaining > 0 and chip_x + 54 <= x + max_width:
                _draw_chip(canvas, f"+{remaining}", (chip_x, chip_y), 54, False)
            break
        _draw_chip(canvas, label, (chip_x, chip_y), chip_w, idx == active_index)
        chip_x += chip_w + 8


def _draw_chip(
    canvas: npt.NDArray[np.uint8],
    text: str,
    origin: Tuple[int, int],
    width: int,
    active: bool,
) -> None:
    x, y = origin
    height = 32
    fill = (37, 93, 71) if active else (255, 255, 255)
    border = (37, 93, 71) if active else (200, 205, 200)
    text_color = (255, 255, 255) if active else (50, 53, 56)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + width, y + height), border, 1)
    _put_fitted_text(canvas, text, (x + 10, y + 21), width - 18, 0.48, text_color, thickness=1)


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


def _put_label(
    canvas: npt.NDArray[np.uint8],
    text: str,
    origin: Tuple[int, int],
    max_width: Optional[int] = None,
) -> None:
    x, y = origin
    scale = 0.58
    thickness = 2
    if max_width is not None:
        text = _ellipsize_text(text, max_width, scale, thickness)
    text_w, text_h = _text_size(text, scale, thickness)
    cv2.rectangle(canvas, (x - 8, y - text_h - 10), (x + text_w + 8, y + 8), (30, 32, 36), -1)
    _put_text(canvas, text, (x, y), scale, (255, 255, 255), thickness=thickness)


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


def _put_fitted_text(
    canvas: npt.NDArray[np.uint8],
    text: str,
    origin: Tuple[int, int],
    max_width: int,
    scale: float,
    color: Tuple[int, int, int],
    *,
    thickness: int,
) -> None:
    fitted = _ellipsize_text(text, max(1, max_width), scale, thickness)
    _put_text(canvas, fitted, origin, scale, color, thickness=thickness)


def _ellipsize_text(text: str, max_width: int, scale: float, thickness: int) -> str:
    if _text_size(text, scale, thickness)[0] <= max_width:
        return text
    suffix = "..."
    if _text_size(suffix, scale, thickness)[0] > max_width:
        return ""

    lo = 0
    hi = len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].rstrip() + suffix
        if _text_size(candidate, scale, thickness)[0] <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best or suffix


def _text_size(text: str, scale: float, thickness: int) -> Tuple[int, int]:
    (width, height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    return width, height


def _darken(color: Tuple[int, int, int], amount: float) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(channel * amount))) for channel in color)


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
