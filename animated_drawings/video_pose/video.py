"""Video helpers for the pose estimation pipeline."""

from __future__ import annotations

import json
import shutil
import subprocess
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
    duration_seconds: Optional[float] = None

    @property
    def duration(self) -> float:
        if self.duration_seconds is not None and self.duration_seconds > 0:
            return self.duration_seconds
        if self.fps <= 0:
            return 0.0
        return self.frame_count / self.fps


def read_video_metadata(video_path: Path) -> VideoMetadata:
    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise PoseVideoError(f"Could not open video: {video_path}")

        raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        probe = _probe_video(video_path)
        duration = _first_positive_float(probe.get("format_duration"), probe.get("stream_duration"))
        fps = _select_metadata_fps(
            raw_fps=raw_fps,
            frame_count=frame_count,
            probed_duration=duration,
            probed_fps=_parse_frame_rate(str(probe.get("avg_frame_rate") or "")),
        )

        return VideoMetadata(
            fps=fps,
            frame_count=frame_count,
            width=width or int(probe.get("width") or 0),
            height=height or int(probe.get("height") or 0),
            duration_seconds=duration,
        )
    finally:
        cap.release()


def validate_video_duration(video_path: Path, max_seconds: int) -> VideoMetadata:
    metadata = read_video_metadata(video_path)
    if metadata.duration > max_seconds + 0.05:
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

    return transcode_to_browser_mp4(output_path)


def transcode_to_browser_mp4(
    input_path: Path,
    output_path: Optional[Path] = None,
    *,
    strict: bool = False,
    timeout: Optional[float] = None,
) -> Path:
    """Convert an MP4 to browser-friendly H.264 when ffmpeg is available.

    The historical helper was best-effort and returned the source video if
    ffmpeg was missing or failed. Product-facing browser previews need stricter
    behavior, so strict mode raises a PoseVideoError instead of returning a
    potentially unsupported file.
    """

    input_path = Path(input_path)
    output_path = Path(output_path or input_path)
    output_path.parent.mkdir(exist_ok=True, parents=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if strict:
            raise PoseVideoError("ffmpeg is required to prepare a browser-playable video.")
        if output_path != input_path:
            shutil.copy2(input_path, output_path)
        return output_path

    tmp_path = output_path
    replace_input = input_path.resolve() == output_path.resolve()
    if replace_input:
        tmp_path = output_path.with_name(f".{output_path.stem}.browser_tmp.mp4")

    commands = [
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ],
        [
            ffmpeg,
            "-y",
            "-i",
            str(input_path),
            "-an",
            "-c:v",
            "h264_videotoolbox",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp_path),
        ],
    ]

    failures = []
    for command in commands:
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as e:
            failures.append(f"ffmpeg timed out after {timeout:g}s")
            if tmp_path != input_path and tmp_path.exists():
                tmp_path.unlink()
            continue
        if result.returncode == 0 and tmp_path.exists() and tmp_path.stat().st_size > 0:
            if replace_input:
                tmp_path.replace(output_path)
            return output_path
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        failures.append(detail or f"ffmpeg exited with code {result.returncode}")

    if strict:
        detail = failures[-1] if failures else "unknown ffmpeg error"
        raise PoseVideoError(f"ffmpeg transcode failed. {detail}")
    if output_path != input_path:
        shutil.copy2(input_path, output_path)
    return output_path


def _probe_video(video_path: Path) -> dict:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {}

    result = subprocess.run(
        [
            ffprobe,
            "-hide_banner",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate,r_frame_rate,duration,width,height:format=duration",
            "-of",
            "json",
            str(video_path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return {}

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}

    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = data.get("format") or {}
    return {
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "stream_duration": stream.get("duration"),
        "format_duration": fmt.get("duration"),
        "width": stream.get("width"),
        "height": stream.get("height"),
    }


def _select_metadata_fps(
    raw_fps: float,
    frame_count: int,
    probed_duration: Optional[float],
    probed_fps: Optional[float],
) -> float:
    if _is_reasonable_fps(raw_fps):
        return raw_fps
    if probed_duration and probed_duration > 0 and frame_count > 0:
        derived_fps = frame_count / probed_duration
        if _is_reasonable_fps(derived_fps):
            return derived_fps
    if probed_fps and _is_reasonable_fps(probed_fps):
        return probed_fps
    return 30.0


def _is_reasonable_fps(fps: float) -> bool:
    return 1.0 <= fps <= 120.0


def _parse_frame_rate(value: str) -> Optional[float]:
    if not value or value == "0/0":
        return None
    try:
        if "/" in value:
            numerator, denominator = value.split("/", 1)
            denominator_f = float(denominator)
            if denominator_f == 0:
                return None
            return float(numerator) / denominator_f
        return float(value)
    except ValueError:
        return None


def _first_positive_float(*values) -> Optional[float]:
    for value in values:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


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
