"""Optional landmark-flow postprocessor for low-confidence MediaPipe landmarks."""

from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

import numpy as np

from animated_drawings.video_pose.types import PoseFrame, PoseSequence


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LANDMARK_FLOW_MODEL = REPO_ROOT / "outputs/landmark_flow/landmark_flow_corrector.pt"
DEFAULT_LANDMARK_FLOW_THRESHOLD = 0.5
DEFAULT_LANDMARK_FLOW_STEPS = 8
DEFAULT_LANDMARK_FLOW_WINDOW = 31
FLOW_METRIC_KEYS = (
    "flow_model_enabled",
    "flow_model_loaded",
    "flow_corrected_landmarks",
    "flow_corrected_ratio",
    "flow_threshold",
    "flow_fallback_used",
)


def default_flow_metrics(
    *,
    enabled: bool,
    loaded: bool = False,
    threshold: float = DEFAULT_LANDMARK_FLOW_THRESHOLD,
    fallback_used: bool = False,
    corrected_landmarks: int = 0,
    total_slots: int = 0,
) -> Dict[str, float]:
    total_slots = max(1, int(total_slots))
    return {
        "flow_model_enabled": float(bool(enabled)),
        "flow_model_loaded": float(bool(loaded)),
        "flow_corrected_landmarks": float(corrected_landmarks),
        "flow_corrected_ratio": float(corrected_landmarks) / float(total_slots),
        "flow_threshold": float(threshold),
        "flow_fallback_used": float(bool(fallback_used)),
    }


def landmark_flow_model_path(model_path: Optional[Path] = None) -> Path:
    if model_path is not None:
        return Path(model_path).expanduser()
    env_path = os.environ.get("LANDMARK_FLOW_MODEL")
    if env_path:
        return Path(env_path).expanduser()
    return DEFAULT_LANDMARK_FLOW_MODEL


def landmark_flow_enabled(model_path: Optional[Path] = None, enabled: Optional[bool] = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    env_value = os.environ.get("LANDMARK_FLOW_ENABLED")
    if env_value is not None:
        return _env_bool(env_value)
    return landmark_flow_model_path(model_path).exists()


def landmark_flow_threshold(default: float = DEFAULT_LANDMARK_FLOW_THRESHOLD) -> float:
    env_value = os.environ.get("LANDMARK_FLOW_THRESHOLD")
    if not env_value:
        return float(default)
    try:
        return float(env_value)
    except ValueError:
        return float(default)


def create_landmark_flow_corrector(
    model_path: Optional[Path] = None,
    *,
    threshold: Optional[float] = None,
    enabled: Optional[bool] = None,
) -> Tuple[Optional["FlowLandmarkCorrector"], Dict[str, float]]:
    resolved_threshold = landmark_flow_threshold() if threshold is None else float(threshold)
    resolved_path = landmark_flow_model_path(model_path)
    should_enable = landmark_flow_enabled(resolved_path, enabled=enabled)
    if not should_enable:
        return None, default_flow_metrics(enabled=False, threshold=resolved_threshold)
    if not resolved_path.exists():
        return None, default_flow_metrics(enabled=True, threshold=resolved_threshold, fallback_used=True)
    try:
        corrector = FlowLandmarkCorrector.from_checkpoint(resolved_path, threshold=resolved_threshold)
    except Exception:
        return None, default_flow_metrics(enabled=True, threshold=resolved_threshold, fallback_used=True)
    return corrector, default_flow_metrics(enabled=True, loaded=True, threshold=resolved_threshold)


class FlowLandmarkCorrector:
    """Apply a trained rectified-flow model to low-confidence x/y landmarks only."""

    def __init__(
        self,
        model: Any,
        landmark_order: Iterable[str],
        *,
        threshold: float = DEFAULT_LANDMARK_FLOW_THRESHOLD,
        window_size: int = DEFAULT_LANDMARK_FLOW_WINDOW,
        inference_steps: int = DEFAULT_LANDMARK_FLOW_STEPS,
        torch_module: Any = None,
        device: Optional[str] = None,
    ) -> None:
        if torch_module is None:
            import torch as torch_module  # type: ignore[no-redef]

        self.model = model
        self.landmark_order = [str(name) for name in landmark_order]
        self.threshold = float(threshold)
        self.window_size = max(1, int(window_size))
        self.inference_steps = max(1, int(inference_steps))
        self.torch = torch_module
        self.device = device or ("cuda" if self.torch.cuda.is_available() else "cpu")
        self._live_frames: Deque[PoseFrame] = deque(maxlen=self.window_size)
        if hasattr(self.model, "to"):
            self.model.to(self.device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: Path,
        threshold: float = DEFAULT_LANDMARK_FLOW_THRESHOLD,
        *,
        inference_steps: Optional[int] = None,
        device: Optional[str] = None,
    ) -> "FlowLandmarkCorrector":
        import torch
        from landmark_flow.model import LandmarkFlowModel

        checkpoint = torch.load(Path(path).expanduser(), map_location=device or "cpu")
        metadata = checkpoint.get("metadata") or {}
        landmark_order = metadata.get("landmark_order") or metadata.get("landmark_names")
        if not landmark_order:
            raise ValueError("Landmark-flow checkpoint is missing metadata['landmark_order'].")
        model_config = dict(metadata.get("model_config") or checkpoint.get("model_config") or {})
        model_config.setdefault("num_landmarks", len(landmark_order))
        if isinstance(model_config.get("dilations"), list):
            model_config["dilations"] = tuple(model_config["dilations"])
        model = LandmarkFlowModel(**model_config)
        state_dict = checkpoint.get("model_state") or checkpoint.get("model_state_dict") or checkpoint.get("state_dict")
        if state_dict is None:
            raise ValueError("Landmark-flow checkpoint is missing model weights.")
        model.load_state_dict(state_dict)
        steps = int(inference_steps or metadata.get("inference_steps") or DEFAULT_LANDMARK_FLOW_STEPS)
        window_size = int(metadata.get("window_size") or DEFAULT_LANDMARK_FLOW_WINDOW)
        return cls(
            model,
            landmark_order,
            threshold=threshold,
            window_size=window_size,
            inference_steps=steps,
            torch_module=torch,
            device=device,
        )

    def correct_sequence(self, sequence: PoseSequence) -> Tuple[PoseSequence, Dict[str, float]]:
        frames = _clone_frames(sequence.frames)
        if not frames or not self.landmark_order:
            metrics = default_flow_metrics(
                enabled=True,
                loaded=True,
                threshold=self.threshold,
                total_slots=len(frames) * len(self.landmark_order),
            )
            return _sequence_like(sequence, frames, metrics), metrics

        condition = self._condition_for_frames(frames)
        prediction = self._predict_xy(condition)
        corrected_count = self._apply_prediction(frames, prediction, prediction_index=slice(None))
        total_slots = len(frames) * len(self.landmark_order)
        metrics = default_flow_metrics(
            enabled=True,
            loaded=True,
            threshold=self.threshold,
            corrected_landmarks=corrected_count,
            total_slots=total_slots,
        )
        return _sequence_like(sequence, frames, metrics), metrics

    def correct_live_frame(self, frame: PoseFrame) -> Tuple[PoseFrame, Dict[str, float]]:
        current = _clone_frame(frame)
        self._live_frames.append(current)
        frames = list(self._live_frames)
        if len(frames) < self.window_size and frames:
            pad_count = self.window_size - len(frames)
            frames = [_clone_frame(frames[0]) for _ in range(pad_count)] + frames

        condition = self._condition_for_frames(frames)
        prediction = self._predict_xy(condition)
        corrected = _clone_frame(current)
        corrected_count = self._apply_prediction([corrected], prediction, prediction_index=-1)
        metrics = default_flow_metrics(
            enabled=True,
            loaded=True,
            threshold=self.threshold,
            corrected_landmarks=corrected_count,
            total_slots=len(self.landmark_order),
        )
        return corrected, metrics

    def reset(self) -> None:
        self._live_frames.clear()

    def _condition_for_frames(self, frames: List[PoseFrame]) -> np.ndarray:
        frame_count = len(frames)
        landmark_count = len(self.landmark_order)
        xy = np.full((frame_count, landmark_count, 2), np.nan, dtype=np.float32)
        visibility = np.zeros((frame_count, landmark_count), dtype=np.float32)
        low_confidence = np.ones((frame_count, landmark_count), dtype=np.float32)

        for frame_idx, frame in enumerate(frames):
            for landmark_idx, name in enumerate(self.landmark_order):
                values = frame.landmarks.get(name)
                if values is None or len(values) < 2:
                    continue
                x = float(values[0])
                y = float(values[1])
                if np.isfinite(x) and np.isfinite(y):
                    xy[frame_idx, landmark_idx] = [x, y]
                vis = float(values[3]) if len(values) > 3 and np.isfinite(float(values[3])) else 0.0
                visibility[frame_idx, landmark_idx] = np.clip(vis, 0.0, 1.0)
                low_confidence[frame_idx, landmark_idx] = 1.0 if vis < self.threshold else 0.0

        xy = _fill_missing_xy(xy)
        return np.concatenate(
            [xy, visibility[..., None], low_confidence[..., None]],
            axis=-1,
        ).astype(np.float32)

    def _predict_xy(self, condition_np: np.ndarray) -> np.ndarray:
        torch = self.torch
        condition = torch.as_tensor(condition_np[None, ...], dtype=torch.float32, device=self.device)
        estimate = condition[..., :2].clone()
        step_size = 1.0 / float(self.inference_steps)
        with torch.no_grad():
            for step_idx in range(self.inference_steps):
                t = torch.full((1,), step_idx * step_size, dtype=torch.float32, device=self.device)
                estimate = estimate + self.model(estimate, condition, t) * step_size
        prediction = estimate.detach().cpu().numpy()[0]
        return np.clip(prediction, -1.0, 2.0).astype(np.float32)

    def _apply_prediction(
        self,
        frames: List[PoseFrame],
        prediction: np.ndarray,
        *,
        prediction_index: Any,
    ) -> int:
        corrected_count = 0
        if isinstance(prediction_index, slice):
            frame_indices = range(len(frames))
        else:
            frame_indices = [int(prediction_index)]

        for out_idx, pred_idx in enumerate(frame_indices):
            frame = frames[out_idx]
            for landmark_idx, name in enumerate(self.landmark_order):
                values = frame.landmarks.get(name)
                if values is None or len(values) < 2:
                    continue
                visibility = float(values[3]) if len(values) > 3 else 1.0
                if visibility >= self.threshold:
                    continue
                padded = list(values)
                while len(padded) < 4:
                    padded.append(1.0 if len(padded) == 3 else 0.0)
                padded[0] = float(prediction[pred_idx, landmark_idx, 0])
                padded[1] = float(prediction[pred_idx, landmark_idx, 1])
                frame.landmarks[name] = padded[:4]
                corrected_count += 1
        return corrected_count


def merge_flow_metrics(metrics: Dict[str, float], flow_metrics: Dict[str, float]) -> Dict[str, float]:
    merged = dict(metrics)
    for key in FLOW_METRIC_KEYS:
        if key in flow_metrics:
            merged[key] = float(flow_metrics[key])
    return merged


def _env_bool(value: str) -> bool:
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _clone_frame(frame: PoseFrame) -> PoseFrame:
    return PoseFrame(
        timestamp=frame.timestamp,
        landmarks={name: [float(value) for value in values] for name, values in frame.landmarks.items()},
    )


def _clone_frames(frames: Iterable[PoseFrame]) -> List[PoseFrame]:
    return [_clone_frame(frame) for frame in frames]


def _sequence_like(sequence: PoseSequence, frames: List[PoseFrame], metrics: Dict[str, float]) -> PoseSequence:
    return PoseSequence(
        fps=sequence.fps,
        width=sequence.width,
        height=sequence.height,
        landmark_names=list(sequence.landmark_names),
        frames=frames,
        quality_report=sequence.quality_report,
    )


def _fill_missing_xy(xy: np.ndarray) -> np.ndarray:
    filled = xy.copy()
    frame_count = filled.shape[0]
    frame_positions = np.arange(frame_count)
    for landmark_idx in range(filled.shape[1]):
        for axis in range(2):
            values = filled[:, landmark_idx, axis]
            finite = np.isfinite(values)
            if np.all(finite):
                continue
            if np.any(finite):
                filled[:, landmark_idx, axis] = np.interp(frame_positions, frame_positions[finite], values[finite])
            else:
                filled[:, landmark_idx, axis] = 0.5
    return filled
