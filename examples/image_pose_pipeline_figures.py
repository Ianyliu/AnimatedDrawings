#!/usr/bin/env python

"""Generate poster-ready pose pipeline figures from a still image of a person."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps, UnidentifiedImageError

from animated_drawings.video_pose import (
    CausalPoseSmoother,
    CausalPoseSmootherConfig,
    LiveMediaPipePoseEstimator,
    PoseFrame,
    PoseVideoError,
    analyze_pose_frame,
    create_landmark_flow_corrector,
)
from animated_drawings.video_pose.constants import POSE_CONNECTIONS


DEFAULT_FIGURE_SIZE = 1024
LOW_CONF_COLOR = (30, 30, 230)
MID_CONF_COLOR = (25, 125, 245)
HIGH_CONF_COLOR = (85, 195, 85)
UNKNOWN_CONF_COLOR = (180, 210, 225)
DIRECT_BACKGROUND = (0, 0, 0)
OVERLAY_PADDING_BACKGROUND = (245, 245, 245)


@dataclass(frozen=True)
class CanvasTransform:
    source_width: int
    source_height: int
    scale: float
    offset_x: int
    offset_y: int


@dataclass(frozen=True)
class StageFigure:
    name: str
    frame: PoseFrame


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Estimate MediaPipe pose from a still image, run the same optional "
            "landmark-flow correction and causal smoothing used by the webcam "
            "prototype, and write separate poster panels."
        )
    )
    parser.add_argument("input_image", help="Path to a readable image of a person.")
    parser.add_argument("out_dir", help="Directory where generated figure PNGs will be written.")
    parser.add_argument(
        "--prefix",
        default=None,
        help="Output filename prefix. Defaults to the input image stem.",
    )
    parser.add_argument(
        "--figure-size",
        type=int,
        default=DEFAULT_FIGURE_SIZE,
        help="Square output size in pixels. Use 0 to preserve the input image dimensions.",
    )
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--min-tracking-confidence", type=float, default=0.5)
    parser.add_argument(
        "--landmark-flow-model",
        default=None,
        help="Optional landmark-flow checkpoint for confidence-aware correction.",
    )
    parser.add_argument("--no-landmark-flow", action="store_true", help="Disable landmark-flow correction.")
    parser.add_argument(
        "--landmark-flow-threshold",
        type=float,
        default=0.5,
        help="Visibility threshold below which landmark-flow may correct x/y landmarks.",
    )
    parser.add_argument(
        "--high-confidence-threshold",
        type=float,
        default=0.8,
        help="Visibility threshold used for high-confidence drawing color.",
    )
    parser.add_argument("--smoothing-alpha", type=float, default=0.35, help="Causal smoother EMA alpha.")
    parser.add_argument(
        "--line-thickness",
        type=int,
        default=0,
        help="Pose line thickness in pixels. Defaults to an automatic size.",
    )
    parser.add_argument(
        "--joint-radius",
        type=int,
        default=0,
        help="Pose joint radius in pixels. Defaults to an automatic size.",
    )
    parser.add_argument(
        "--manifest",
        default="pose_pipeline_figures.json",
        help="JSON manifest filename to write in the output directory. Use an empty string to skip.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_image).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(exist_ok=True, parents=True)

    image_bgr = _read_image_bgr(input_path)
    raw_frame = _estimate_pose_frame(image_bgr, args)
    status = analyze_pose_frame(raw_frame, visibility_threshold=float(args.landmark_flow_threshold))
    if not raw_frame.landmarks:
        raise SystemExit(f"No human pose was detected in {input_path}.")

    corrected_frame, flow_metrics = _correct_pose(raw_frame, args)
    smoothed_frame = _smooth_pose(corrected_frame, args)
    figures = [
        StageFigure("raw", raw_frame),
        StageFigure("corrected", corrected_frame),
        StageFigure("smoothed", smoothed_frame),
    ]

    prefix = args.prefix or input_path.stem
    outputs = _write_figures(image_bgr, figures, out_dir, prefix, args)
    manifest_path = _write_manifest(out_dir, args.manifest, input_path, outputs, status, flow_metrics)

    for label, path in outputs.items():
        print(f"Wrote {label}: {path}")
    if manifest_path is not None:
        print(f"Wrote manifest: {manifest_path}")


def _read_image_bgr(path: Path) -> npt.NDArray[np.uint8]:
    try:
        with Image.open(path) as image:
            rgb = ImageOps.exif_transpose(image).convert("RGB")
    except (OSError, UnidentifiedImageError) as exc:
        raise SystemExit(f"Could not read image: {path}") from exc

    return cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)


def _estimate_pose_frame(frame_bgr: npt.NDArray[np.uint8], args) -> PoseFrame:
    try:
        with LiveMediaPipePoseEstimator(
            model_complexity=args.model_complexity,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            static_image_mode=True,
        ) as estimator:
            return estimator.estimate_frame(frame_bgr, timestamp=0.0)
    except PoseVideoError as exc:
        raise SystemExit(str(exc)) from exc


def _correct_pose(frame: PoseFrame, args) -> Tuple[PoseFrame, Dict[str, float]]:
    flow_corrector, flow_metrics = create_landmark_flow_corrector(
        Path(args.landmark_flow_model).expanduser() if args.landmark_flow_model else None,
        threshold=float(args.landmark_flow_threshold),
        enabled=False if args.no_landmark_flow else None,
    )
    if flow_corrector is None:
        return _clone_frame(frame), flow_metrics
    return flow_corrector.correct_live_frame(frame)


def _smooth_pose(frame: PoseFrame, args) -> PoseFrame:
    smoother = CausalPoseSmoother(
        CausalPoseSmootherConfig(
            alpha=float(args.smoothing_alpha),
            visibility_threshold=float(args.landmark_flow_threshold),
        )
    )
    return smoother.process(frame)


def _write_figures(
    image_bgr: npt.NDArray[np.uint8],
    figures: Iterable[StageFigure],
    out_dir: Path,
    prefix: str,
    args,
) -> Dict[str, str]:
    outputs: Dict[str, str] = {}
    for figure in figures:
        for mode in ("overlay", "direct"):
            canvas = draw_pose_figure(
                image_bgr,
                figure.frame,
                mode=mode,
                figure_size=int(args.figure_size),
                low_confidence_threshold=float(args.landmark_flow_threshold),
                high_confidence_threshold=float(args.high_confidence_threshold),
                line_thickness=int(args.line_thickness),
                joint_radius=int(args.joint_radius),
            )
            path = out_dir / f"{prefix}_{figure.name}_{mode}.png"
            if not cv2.imwrite(str(path), canvas):
                raise SystemExit(f"Could not write output image: {path}")
            outputs[f"{figure.name}_{mode}"] = str(path)
    return outputs


def draw_pose_figure(
    image_bgr: npt.NDArray[np.uint8],
    frame: PoseFrame,
    *,
    mode: str,
    figure_size: int = DEFAULT_FIGURE_SIZE,
    low_confidence_threshold: float = 0.5,
    high_confidence_threshold: float = 0.8,
    line_thickness: int = 0,
    joint_radius: int = 0,
) -> npt.NDArray[np.uint8]:
    if mode not in {"overlay", "direct"}:
        raise ValueError("mode must be 'overlay' or 'direct'.")

    canvas, transform = _build_canvas(image_bgr, mode=mode, figure_size=figure_size)
    min_dim = max(1, min(canvas.shape[:2]))
    line_thickness = line_thickness if line_thickness > 0 else max(3, int(round(min_dim * 0.008)))
    joint_radius = joint_radius if joint_radius > 0 else max(5, int(round(min_dim * 0.012)))
    outline = max(2, int(round(line_thickness * 0.75)))

    for start_name, end_name in POSE_CONNECTIONS:
        start = frame.landmarks.get(start_name)
        end = frame.landmarks.get(end_name)
        if start is None or end is None or not _coords_finite(start) or not _coords_finite(end):
            continue
        start_pixel = _landmark_pixel(start, transform)
        end_pixel = _landmark_pixel(end, transform)
        color = _connection_color(start, end, low_confidence_threshold, high_confidence_threshold)
        if mode == "overlay":
            cv2.line(canvas, start_pixel, end_pixel, (255, 255, 255), line_thickness + outline, cv2.LINE_AA)
        cv2.line(canvas, start_pixel, end_pixel, color, line_thickness, cv2.LINE_AA)

    for values in frame.landmarks.values():
        if not _coords_finite(values):
            continue
        point = _landmark_pixel(values, transform)
        color = _confidence_color(values, low_confidence_threshold, high_confidence_threshold)
        if mode == "overlay":
            cv2.circle(canvas, point, joint_radius + outline, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, point, joint_radius, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, point, joint_radius, (20, 20, 20), max(1, line_thickness // 3), cv2.LINE_AA)

    return canvas


def _build_canvas(
    image_bgr: npt.NDArray[np.uint8],
    *,
    mode: str,
    figure_size: int,
) -> Tuple[npt.NDArray[np.uint8], CanvasTransform]:
    source_h, source_w = image_bgr.shape[:2]
    if figure_size <= 0:
        if mode == "overlay":
            canvas = image_bgr.copy()
        else:
            canvas = np.full((source_h, source_w, 3), DIRECT_BACKGROUND, dtype=np.uint8)
        return canvas, CanvasTransform(source_w, source_h, 1.0, 0, 0)

    size = max(1, int(figure_size))
    scale = min(size / max(1, source_w), size / max(1, source_h))
    out_w = max(1, int(round(source_w * scale)))
    out_h = max(1, int(round(source_h * scale)))
    offset_x = (size - out_w) // 2
    offset_y = (size - out_h) // 2
    fill = OVERLAY_PADDING_BACKGROUND if mode == "overlay" else DIRECT_BACKGROUND
    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    if mode == "overlay":
        resized = cv2.resize(image_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        canvas[offset_y : offset_y + out_h, offset_x : offset_x + out_w] = resized
    return canvas, CanvasTransform(source_w, source_h, scale, offset_x, offset_y)


def _landmark_pixel(values: List[float], transform: CanvasTransform) -> Tuple[int, int]:
    x = float(values[0]) * float(transform.source_width - 1)
    y = float(values[1]) * float(transform.source_height - 1)
    return (
        int(round(transform.offset_x + x * transform.scale)),
        int(round(transform.offset_y + y * transform.scale)),
    )


def _confidence_color(values: List[float], low_threshold: float, high_threshold: float) -> Tuple[int, int, int]:
    if len(values) < 4 or not np.isfinite(float(values[3])):
        return UNKNOWN_CONF_COLOR
    visibility = float(values[3])
    if visibility < low_threshold:
        return LOW_CONF_COLOR
    if visibility < high_threshold:
        return MID_CONF_COLOR
    return HIGH_CONF_COLOR


def _connection_color(
    start: List[float],
    end: List[float],
    low_threshold: float,
    high_threshold: float,
) -> Tuple[int, int, int]:
    start_visibility = float(start[3]) if len(start) >= 4 and np.isfinite(float(start[3])) else 0.0
    end_visibility = float(end[3]) if len(end) >= 4 and np.isfinite(float(end[3])) else 0.0
    weakest = min(start_visibility, end_visibility)
    return _confidence_color([0.0, 0.0, 0.0, weakest], low_threshold, high_threshold)


def _coords_finite(values: List[float]) -> bool:
    return len(values) >= 2 and np.isfinite(float(values[0])) and np.isfinite(float(values[1]))


def _clone_frame(frame: PoseFrame) -> PoseFrame:
    return PoseFrame(
        timestamp=frame.timestamp,
        landmarks={name: [float(value) for value in values] for name, values in frame.landmarks.items()},
    )


def _write_manifest(
    out_dir: Path,
    manifest_name: str,
    input_path: Path,
    outputs: Dict[str, str],
    status,
    flow_metrics: Dict[str, float],
) -> Optional[Path]:
    if not manifest_name:
        return None
    manifest_path = out_dir / manifest_name
    payload = {
        "input_image": str(input_path),
        "outputs": outputs,
        "pose_status": {
            "state": status.state,
            "message": status.message,
            "missing_landmarks": list(status.missing_landmarks),
        },
        "flow_metrics": {name: float(value) for name, value in flow_metrics.items()},
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return manifest_path


if __name__ == "__main__":
    main()
