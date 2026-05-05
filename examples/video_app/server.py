from __future__ import annotations

import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import hashlib
import hmac
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import requests
import yaml
from flask import Flask, abort, current_app, jsonify, make_response, render_template, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from werkzeug.datastructures import FileStorage
from werkzeug.utils import secure_filename

from animated_drawings.video_pose import (
    PoseVideoError,
    VideoDurationError,
    available_pose_estimators,
    build_motion_from_video,
    write_motion_config_for_bvh,
)
from animated_drawings.video_pose.constants import DEFAULT_MAX_SECONDS
from animated_drawings.video_pose.video import transcode_to_browser_mp4, validate_video_duration

try:
    from examples.video_app.diagnostics import diagnostics_payload
except ImportError:  # pragma: no cover - used when launched as examples/video_app.py
    from video_app.diagnostics import diagnostics_payload


logger = logging.getLogger(__name__)

APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
DEFAULT_OUTPUT_ROOT = APP_DIR / "outputs"
MEDIA_DIR = REPO_ROOT / "media"
CHARACTERS_DIR = EXAMPLES_DIR / "characters"
MOTION_CFG_DIR = EXAMPLES_DIR / "config/motion"
RETARGET_CFG_DIR = EXAMPLES_DIR / "config/retarget"
MEDIAPIPE_RETARGET_CFG = RETARGET_CFG_DIR / "mediapipe_pfp.yaml"
DEFAULT_RETARGET_CFG = RETARGET_CFG_DIR / "fair1_ppf.yaml"
SESSION_COOKIE_NAME = "animated_drawings_video_session"

VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".webm", ".avi", ".mkv"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
IMAGE_FORMATS = {"JPEG", "MPO", "PNG", "WEBP"}
BVH_EXTENSIONS = {".bvh"}
CHARACTER_PREVIEW_FILES = {"texture.png", "joint_overlay.png", "image.png", "original_img.png"}
DEMO_MEDIA = {"garlic.gif": MEDIA_DIR / "garlic.gif"}

RETARGET_CFG_BY_BVH = {
    "cxk_mediapipe.bvh": MEDIAPIPE_RETARGET_CFG,
    "motion.bvh": MEDIAPIPE_RETARGET_CFG,
    "uploaded_motion.bvh": MEDIAPIPE_RETARGET_CFG,
    "jumping_jacks.bvh": RETARGET_CFG_DIR / "cmu1_pfp.yaml",
    "walk-cycle.bvh": RETARGET_CFG_DIR / "walk_cycle_pfp.yaml",
}

SAMPLE_ASSETS = [
    {
        "id": "starter",
        "label": "Try sample",
        "description": "Dab motion with the first bundled character.",
        "motion_cfg": MOTION_CFG_DIR / "dab.yaml",
        "character_cfg": CHARACTERS_DIR / "char1/char_cfg.yaml",
    },
    {
        "id": "wave",
        "label": "Try another",
        "description": "Wave motion with the second bundled character.",
        "motion_cfg": MOTION_CFG_DIR / "wave_hello.yaml",
        "character_cfg": CHARACTERS_DIR / "char2/char_cfg.yaml",
    },
]


class AppError(ValueError):
    def __init__(self, code: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class LocalRateLimiter:
    """In-memory sliding-window limiter for the local single-host app."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, list[float]] = {}

    def check(self, rules: list[tuple[str, int, float]], now: Optional[float] = None) -> tuple[bool, float]:
        now = now or time.time()
        with self._lock:
            for key, limit, window_seconds in rules:
                events = self._pruned(key, now, window_seconds)
                if len(events) >= limit:
                    retry_after = window_seconds - (now - events[0])
                    return False, max(1.0, retry_after)
            for key, _, window_seconds in rules:
                events = self._pruned(key, now, window_seconds)
                events.append(now)
                self._events[key] = events
        return True, 0.0

    def _pruned(self, key: str, now: float, window_seconds: float) -> list[float]:
        cutoff = now - window_seconds
        events = [event for event in self._events.get(key, []) if event >= cutoff]
        self._events[key] = events
        return events


@dataclass
class JobRecord:
    id: str
    kind: str
    session_id: str
    status: str = "queued"
    progress: int = 0
    message: str = "Queued."
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[dict[str, Any]] = None
    error: Optional[dict[str, str]] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "status_url": f"/api/jobs/{self.id}",
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if self.result is not None:
            payload["result"] = self.result
        if self.error is not None:
            payload["error"] = self.error
        return payload


ProgressCallback = Callable[[int, str], None]
JobWork = Callable[[ProgressCallback], dict[str, Any]]


class JobManager:
    """Small in-process job store for the local single-host app."""

    def __init__(self, max_workers: int = 1, run_inline: bool = False) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, JobRecord] = {}
        self._executor = None if run_inline else ThreadPoolExecutor(
            max_workers=max(1, max_workers),
            thread_name_prefix="video-app-job",
        )

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    def submit(
        self,
        kind: str,
        session_id: str,
        work: JobWork,
        message: str,
        job_id: Optional[str] = None,
    ) -> JobRecord:
        job = JobRecord(
            id=job_id or self.new_job_id(),
            kind=kind,
            session_id=session_id,
            message=message,
        )
        with self._lock:
            self._jobs[job.id] = job

        if self._executor is None:
            self._run_job(job.id, work)
        else:
            self._executor.submit(self._run_job, job.id, work)
        return self.get(job.id) or job

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._jobs.get(job_id)

    def pending_count(self) -> int:
        with self._lock:
            return sum(1 for job in self._jobs.values() if job.status in {"queued", "running"})

    def _run_job(self, job_id: str, work: JobWork) -> None:
        self._update(job_id, status="running", progress=1, message="Running...")

        def update(progress: int, message: str) -> None:
            self._update(job_id, progress=progress, message=message)

        try:
            result = work(update)
        except Exception as e:  # pragma: no cover - exercised through endpoint tests
            logger.exception("Video app job %s failed", job_id)
            self._update(
                job_id,
                status="failed",
                error=_error_payload(e),
                message=_error_payload(e)["message"],
            )
            return

        self._update(
            job_id,
            status="completed",
            progress=100,
            message="Complete.",
            result=result,
        )

    def _update(self, job_id: str, **changes: Any) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            for key, value in changes.items():
                if key == "progress" and value is not None:
                    value = max(0, min(100, int(value)))
                setattr(job, key, value)
            job.updated_at = time.time()


def create_app(output_root: Optional[Path] = None, job_manager: Optional[JobManager] = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(APP_DIR / "templates"),
        static_folder=str(APP_DIR / "static"),
        static_url_path="/static",
    )
    app.config["OUTPUT_ROOT"] = Path(output_root or DEFAULT_OUTPUT_ROOT).resolve()
    app.config["MAX_CONTENT_LENGTH"] = _env_int("VIDEO_APP_MAX_UPLOAD_MB", 200) * 1024 * 1024
    app.config["VIDEO_APP_MAX_SECONDS"] = _env_int("VIDEO_APP_MAX_SECONDS", DEFAULT_MAX_SECONDS)
    app.config["VIDEO_APP_MAX_VIDEO_WIDTH"] = _env_int("VIDEO_APP_MAX_VIDEO_WIDTH", 1920)
    app.config["VIDEO_APP_MAX_VIDEO_HEIGHT"] = _env_int("VIDEO_APP_MAX_VIDEO_HEIGHT", 1920)
    app.config["VIDEO_APP_MAX_VIDEO_FPS"] = _env_float("VIDEO_APP_MAX_VIDEO_FPS", 60.0)
    app.config["VIDEO_APP_MAX_IMAGE_WIDTH"] = _env_int("VIDEO_APP_MAX_IMAGE_WIDTH", 4096)
    app.config["VIDEO_APP_MAX_IMAGE_HEIGHT"] = _env_int("VIDEO_APP_MAX_IMAGE_HEIGHT", 4096)
    app.config["VIDEO_APP_MAX_BVH_MB"] = _env_int("VIDEO_APP_MAX_BVH_MB", 10)
    app.config["VIDEO_APP_RENDER_TIMEOUT"] = _env_float("VIDEO_APP_RENDER_TIMEOUT", 180.0)
    app.config["VIDEO_APP_TRANSCODE_TIMEOUT"] = _env_float("VIDEO_APP_TRANSCODE_TIMEOUT", 120.0)
    app.config["VIDEO_APP_TORCHSERVE_TIMEOUT"] = _env_float("VIDEO_APP_TORCHSERVE_TIMEOUT", 30.0)
    app.config["VIDEO_APP_OUTPUT_TTL_HOURS"] = _env_float("VIDEO_APP_OUTPUT_TTL_HOURS", 24.0)
    app.config["VIDEO_APP_CLEANUP_INTERVAL_SECONDS"] = _env_float("VIDEO_APP_CLEANUP_INTERVAL_SECONDS", 300.0)
    app.config["VIDEO_APP_RATE_WINDOW_SECONDS"] = _env_float("VIDEO_APP_RATE_WINDOW_SECONDS", 300.0)
    app.config["VIDEO_APP_SESSION_JOB_LIMIT"] = _env_int("VIDEO_APP_SESSION_JOB_LIMIT", 6)
    app.config["VIDEO_APP_IP_JOB_LIMIT"] = _env_int("VIDEO_APP_IP_JOB_LIMIT", 20)
    app.config["VIDEO_APP_MAX_PENDING_JOBS"] = _env_int("VIDEO_APP_MAX_PENDING_JOBS", 4)
    app.config["VIDEO_APP_POSE_ESTIMATOR"] = os.environ.get("VIDEO_APP_POSE_ESTIMATOR", "mediapipe")
    app.config["VIDEO_APP_RANDOM_FOREST_MODEL"] = os.environ.get("VIDEO_APP_RANDOM_FOREST_MODEL", "")
    app.config["VIDEO_APP_CATBOOST_MODEL"] = os.environ.get("VIDEO_APP_CATBOOST_MODEL", "")
    app.config["VIDEO_APP_TRUST_PROXY"] = _env_bool("VIDEO_APP_TRUST_PROXY", False)
    app.config["VIDEO_APP_SECURE_COOKIE"] = _env_bool("VIDEO_APP_SECURE_COOKIE", False)
    app.config["VIDEO_APP_REQUIRE_CSRF"] = _env_bool("VIDEO_APP_REQUIRE_CSRF", True)
    app.secret_key = os.environ.get("VIDEO_APP_SECRET_KEY") or secrets.token_hex(32)
    app.config["_OUTPUT_CLEANUP_LAST_RUN"] = 0.0
    app.config["JOB_MANAGER"] = job_manager or JobManager(max_workers=_env_int("VIDEO_APP_WORKERS", 1))
    app.config["RATE_LIMITER"] = LocalRateLimiter()

    @app.route("/")
    def index():
        _maybe_cleanup_old_sessions(app)
        session_id = _safe_session_id(request.cookies.get(SESSION_COOKIE_NAME) or uuid.uuid4().hex)
        csrf_token = _csrf_token(app, session_id)
        response = make_response(
            render_template(
                "index.html",
                max_seconds=app.config["VIDEO_APP_MAX_SECONDS"],
                session_id=session_id,
                csrf_token=csrf_token,
            )
        )
        response.set_cookie(
            SESSION_COOKIE_NAME,
            session_id,
            max_age=60 * 60 * 24,
            httponly=True,
            samesite="Lax",
            secure=bool(app.config["VIDEO_APP_SECURE_COOKIE"]),
        )
        return response

    @app.route("/api/assets")
    def assets():
        motions = []
        for motion_cfg in sorted(MOTION_CFG_DIR.glob("*.yaml")):
            motions.append(
                {
                    "id": _repo_rel(motion_cfg),
                    "name": motion_cfg.stem.replace("_", " "),
                    "retarget_cfg": _repo_rel(_retarget_for_motion(motion_cfg)),
                    "preview_url": None,
                }
            )

        characters = []
        for char_cfg in sorted(CHARACTERS_DIR.glob("*/char_cfg.yaml")):
            preview_url = _character_preview_url(char_cfg.parent.name, "texture.png")
            overlay_url = _character_preview_url(char_cfg.parent.name, "joint_overlay.png")
            characters.append(
                {
                    "id": _repo_rel(char_cfg),
                    "name": char_cfg.parent.name,
                    "preview_url": preview_url if (char_cfg.parent / "texture.png").exists() else None,
                    "joint_overlay_url": overlay_url if (char_cfg.parent / "joint_overlay.png").exists() else None,
                }
            )

        return jsonify(
            {
                "motions": motions,
                "characters": characters,
                "samples": _sample_assets(),
                "limits": _limits_payload(app),
                "demo": {"animation_url": "/demo-media/garlic.gif"},
                "pose_estimators": _pose_estimators_payload(app),
                "default_pose_estimator": app.config["VIDEO_APP_POSE_ESTIMATOR"],
            }
        )

    @app.route("/api/diagnostics")
    def diagnostics():
        return jsonify(
            diagnostics_payload(
                {
                    "torchserve_timeout": min(2.0, float(app.config["VIDEO_APP_TORCHSERVE_TIMEOUT"])),
                }
            )
        )

    @app.route("/demo-media/<filename>")
    def demo_media(filename: str):
        path = DEMO_MEDIA.get(filename)
        if path is None or not path.exists():
            abort(404)
        return send_from_directory(path.parent, path.name)

    @app.route("/example-assets/characters/<character_id>/<filename>")
    def character_preview(character_id: str, filename: str):
        safe_character_id = secure_filename(character_id)
        safe_filename = secure_filename(filename)
        if safe_character_id != character_id or safe_filename not in CHARACTER_PREVIEW_FILES:
            abort(404)
        character_dir = (CHARACTERS_DIR / safe_character_id).resolve()
        _ensure_under(character_dir, CHARACTERS_DIR)
        asset_path = character_dir / safe_filename
        if not asset_path.exists():
            abort(404)
        return send_from_directory(character_dir, safe_filename)

    @app.route("/api/jobs/<job_id>")
    def job_status(job_id: str):
        job = app.config["JOB_MANAGER"].get(job_id)
        if job is None:
            return _json_error("job_not_found", "The requested job was not found.", 404)
        if job.session_id != _current_session_id():
            return _json_error("job_not_found", "The requested job was not found.", 404)
        return jsonify({"job": job.to_dict()})

    @app.route("/api/motion/video", methods=["POST"])
    def motion_video():
        try:
            _require_csrf(app)
            session_id, session_dir = _session_from_request(app)
            uploaded = _require_upload("video")
            suffix = _validate_upload(uploaded, VIDEO_EXTENSIONS, "video")
            max_seconds = _parse_max_seconds(request.form.get("max_seconds"), app)
            pose_estimator = _parse_pose_estimator(request.form.get("pose_estimator"), app)
            job_id = app.config["JOB_MANAGER"].new_job_id()
            job_dir = _job_output_dir(session_dir, "motion_video", job_id)
            video_path = _save_upload(uploaded, job_dir / f"input_video{suffix}")
            _validate_video_file(video_path, max_seconds, app)
            _guard_heavy_job(app, session_id)

            def work(update: ProgressCallback) -> dict[str, Any]:
                update(15, "Estimating pose...")
                result = build_motion_from_video(
                    video_path,
                    job_dir,
                    max_seconds=max_seconds,
                    estimator_name=pose_estimator,
                    estimator_config=_pose_estimator_config(app),
                )
                update(85, "Preparing motion outputs...")
                return {
                    "session_id": session_id,
                    "pose_estimator": pose_estimator,
                    "motion_cfg": str(result.motion_config_path),
                    "retarget_cfg": str(MEDIAPIPE_RETARGET_CFG),
                    "bvh_url": _output_url(session_id, result.bvh_path, session_dir),
                    "overlay_url": _output_url(session_id, result.overlay_video_path, session_dir),
                    "pose_sequence_url": _output_url(session_id, result.pose_sequence_path, session_dir),
                    "quality_report": result.quality_report.to_dict() if result.quality_report else None,
                }

            job = app.config["JOB_MANAGER"].submit(
                "motion_video",
                session_id,
                work,
                "Video pose job queued.",
                job_id=job_id,
            )
            return jsonify({"job": job.to_dict()}), 202
        except Exception as e:
            return _json_exception(e)

    @app.route("/api/motion/bvh", methods=["POST"])
    def motion_bvh():
        try:
            _require_csrf(app)
            session_id, session_dir = _session_from_request(app)
            uploaded = _require_upload("bvh")
            _validate_upload(uploaded, BVH_EXTENSIONS, "BVH")
            job_id = app.config["JOB_MANAGER"].new_job_id()
            job_dir = _job_output_dir(session_dir, "motion_bvh", job_id)
            bvh_path = _save_upload(uploaded, job_dir / "uploaded_motion.bvh")
            _validate_bvh_file(bvh_path, app)
            _guard_heavy_job(app, session_id)
            motion_cfg = job_dir / "uploaded_motion.yaml"

            def work(update: ProgressCallback) -> dict[str, Any]:
                update(35, "Preparing BVH...")
                write_motion_config_for_bvh(bvh_path, motion_cfg, _read_bvh_frame_time(bvh_path))
                return {
                    "session_id": session_id,
                    "motion_cfg": str(motion_cfg),
                    "retarget_cfg": str(MEDIAPIPE_RETARGET_CFG),
                    "bvh_url": _output_url(session_id, bvh_path, session_dir),
                }

            job = app.config["JOB_MANAGER"].submit(
                "motion_bvh",
                session_id,
                work,
                "BVH job queued.",
                job_id=job_id,
            )
            return jsonify({"job": job.to_dict()}), 202
        except Exception as e:
            return _json_exception(e)

    @app.route("/api/drawing", methods=["POST"])
    def drawing():
        try:
            _require_csrf(app)
            session_id, session_dir = _session_from_request(app)
            character_cfg = request.form.get("character_cfg")
            uploaded = request.files.get("drawing")

            if character_cfg:
                char_cfg_path = _resolve_bundled_character_cfg(character_cfg)
                overlay_src = _ensure_joint_overlay(char_cfg_path, char_cfg_path.parent / "joint_overlay.png")
                preview_path = _copy_to_session(overlay_src, session_dir / "drawing_joint_overlay.png")
                return jsonify(
                    {
                        "session_id": session_id,
                        "character_cfg": str(char_cfg_path),
                        "joint_overlay_url": _output_url(session_id, preview_path, session_dir),
                    }
                )

            if uploaded is None or uploaded.filename == "":
                raise AppError("missing_upload", "Expected a bundled character or drawing upload.")

            suffix = _validate_upload(uploaded, IMAGE_EXTENSIONS, "drawing")
            job_id = app.config["JOB_MANAGER"].new_job_id()
            job_dir = _job_output_dir(session_dir, "drawing_upload", job_id)
            drawing_path = _save_upload(uploaded, job_dir / f"uploaded_drawing{suffix}")
            _validate_image_file(drawing_path, app)
            annotation_path = _write_annotation_image(drawing_path, job_dir / "annotation_input.png")
            _guard_heavy_job(app, session_id)
            char_dir = job_dir / "character"

            def work(update: ProgressCallback) -> dict[str, Any]:
                update(15, "Estimating drawing joints...")
                _image_to_annotations(
                    annotation_path,
                    char_dir,
                    timeout=app.config["VIDEO_APP_TORCHSERVE_TIMEOUT"],
                )
                char_cfg_path = char_dir / "char_cfg.yaml"
                overlay_path = _ensure_joint_overlay(char_cfg_path, char_dir / "joint_overlay.png")
                return {
                    "session_id": session_id,
                    "character_cfg": str(char_cfg_path),
                    "joint_overlay_url": _output_url(session_id, overlay_path, session_dir),
                }

            job = app.config["JOB_MANAGER"].submit(
                "drawing_upload",
                session_id,
                work,
                "Drawing job queued.",
                job_id=job_id,
            )
            return jsonify({"job": job.to_dict()}), 202
        except Exception as e:
            return _json_exception(e)

    @app.route("/api/render", methods=["POST"])
    def render_animation():
        try:
            _require_csrf(app)
            session_id, session_dir = _session_from_request(app)
            payload = request.get_json(silent=True) or {}
            character_cfg = _resolve_character_cfg_for_render(_required_payload(payload, "character_cfg"), app)
            motion_cfg = _resolve_motion_cfg_for_render(_required_payload(payload, "motion_cfg"), app)
            retarget_cfg = _resolve_retarget_cfg_for_render(
                payload.get("retarget_cfg") or str(_retarget_for_motion(motion_cfg)),
                app,
            )
            _guard_heavy_job(app, session_id)
            job_id = app.config["JOB_MANAGER"].new_job_id()
            job_dir = _job_output_dir(session_dir, "render", job_id)

            def work(update: ProgressCallback) -> dict[str, Any]:
                raw_output_video = job_dir / "animated_drawing_raw.mp4"
                output_video = job_dir / "animated_drawing.mp4"
                mvc_cfg = job_dir / "render_mvc.yaml"
                update(10, "Preparing render...")
                _write_mvc_cfg(character_cfg, motion_cfg, retarget_cfg, raw_output_video, mvc_cfg)
                update(30, "Rendering animation...")
                _run_render(mvc_cfg, timeout=app.config["VIDEO_APP_RENDER_TIMEOUT"])
                update(85, "Preparing browser video...")
                transcode_to_browser_mp4(
                    raw_output_video,
                    output_video,
                    strict=True,
                    timeout=app.config["VIDEO_APP_TRANSCODE_TIMEOUT"],
                )
                return {
                    "session_id": session_id,
                    "animation_url": _output_url(session_id, output_video, session_dir),
                    "mvc_cfg": str(mvc_cfg),
                }

            job = app.config["JOB_MANAGER"].submit(
                "render",
                session_id,
                work,
                "Render job queued.",
                job_id=job_id,
            )
            return jsonify({"job": job.to_dict()}), 202
        except Exception as e:
            return _json_exception(e)

    @app.route("/outputs/<session_id>/<path:filename>")
    def outputs(session_id: str, filename: str):
        safe_session_id = _safe_session_id(session_id)
        if safe_session_id != _current_session_id():
            return _json_error("output_not_found", "The requested output was not found.", 404)
        session_dir = (app.config["OUTPUT_ROOT"] / safe_session_id).resolve()
        _ensure_under(session_dir, app.config["OUTPUT_ROOT"])
        response = send_from_directory(session_dir, filename)
        response.headers["Cache-Control"] = "no-store, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response

    return app


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _sample_assets() -> list[dict[str, Any]]:
    samples = []
    for sample in SAMPLE_ASSETS:
        motion_cfg = sample["motion_cfg"]
        character_cfg = sample["character_cfg"]
        character_id = character_cfg.parent.name
        samples.append(
            {
                "id": sample["id"],
                "label": sample["label"],
                "description": sample["description"],
                "motion_cfg": _repo_rel(motion_cfg),
                "retarget_cfg": _repo_rel(_retarget_for_motion(motion_cfg)),
                "character_cfg": _repo_rel(character_cfg),
                "character_preview_url": _character_preview_url(character_id, "texture.png"),
                "joint_overlay_url": _character_preview_url(character_id, "joint_overlay.png"),
            }
        )
    return samples


def _limits_payload(app: Flask) -> dict[str, Any]:
    return {
        "max_upload_mb": int(app.config["MAX_CONTENT_LENGTH"] / (1024 * 1024)),
        "max_seconds": int(app.config["VIDEO_APP_MAX_SECONDS"]),
        "max_video_width": int(app.config["VIDEO_APP_MAX_VIDEO_WIDTH"]),
        "max_video_height": int(app.config["VIDEO_APP_MAX_VIDEO_HEIGHT"]),
        "max_video_fps": float(app.config["VIDEO_APP_MAX_VIDEO_FPS"]),
        "max_image_width": int(app.config["VIDEO_APP_MAX_IMAGE_WIDTH"]),
        "max_image_height": int(app.config["VIDEO_APP_MAX_IMAGE_HEIGHT"]),
        "max_bvh_mb": int(app.config["VIDEO_APP_MAX_BVH_MB"]),
        "session_job_limit": int(app.config["VIDEO_APP_SESSION_JOB_LIMIT"]),
        "ip_job_limit": int(app.config["VIDEO_APP_IP_JOB_LIMIT"]),
        "rate_window_seconds": int(app.config["VIDEO_APP_RATE_WINDOW_SECONDS"]),
        "max_pending_jobs": int(app.config["VIDEO_APP_MAX_PENDING_JOBS"]),
        "accepted_extensions": {
            "video": sorted(VIDEO_EXTENSIONS),
            "image": sorted(IMAGE_EXTENSIONS),
            "bvh": sorted(BVH_EXTENSIONS),
        },
    }


def _pose_estimator_config(app: Flask) -> dict[str, str]:
    return {
        "random_forest_model": str(app.config.get("VIDEO_APP_RANDOM_FOREST_MODEL") or ""),
        "catboost_model": str(app.config.get("VIDEO_APP_CATBOOST_MODEL") or ""),
    }


def _pose_estimators_payload(app: Flask) -> list[dict[str, Any]]:
    return available_pose_estimators(_pose_estimator_config(app))


def _parse_pose_estimator(raw_value: Optional[str], app: Flask) -> str:
    estimator = (raw_value or app.config["VIDEO_APP_POSE_ESTIMATOR"] or "mediapipe").strip().lower()
    known = {item["id"]: item for item in _pose_estimators_payload(app)}
    if estimator not in known:
        raise AppError("invalid_pose_estimator", f"Unknown pose estimator: {estimator}.")
    if estimator != "mediapipe" and not known[estimator]["available"]:
        raise AppError(
            "pose_estimator_unavailable",
            f"{known[estimator]['name']} is not configured or unavailable. Check diagnostics.",
        )
    return estimator


def _character_preview_url(character_id: str, filename: str) -> str:
    return f"/example-assets/characters/{character_id}/{filename}"


def _guard_heavy_job(app: Flask, session_id: str) -> None:
    max_pending = int(app.config["VIDEO_APP_MAX_PENDING_JOBS"])
    if app.config["JOB_MANAGER"].pending_count() >= max_pending:
        raise AppError(
            "server_busy",
            "The local render queue is full. Wait for the current jobs to finish and try again.",
            429,
        )

    window = float(app.config["VIDEO_APP_RATE_WINDOW_SECONDS"])
    allowed, retry_after = app.config["RATE_LIMITER"].check(
        [
            (f"session:{session_id}", int(app.config["VIDEO_APP_SESSION_JOB_LIMIT"]), window),
            (f"ip:{_client_ip()}", int(app.config["VIDEO_APP_IP_JOB_LIMIT"]), window),
        ]
    )
    if not allowed:
        raise AppError(
            "rate_limited",
            f"Too many jobs started recently. Try again in {int(round(retry_after))} seconds.",
            429,
        )


def _client_ip() -> str:
    forwarded_for = request.headers.get("X-Forwarded-For", "")
    if current_app.config.get("VIDEO_APP_TRUST_PROXY") and forwarded_for:
        return forwarded_for.split(",", 1)[0].strip() or "unknown"
    return request.remote_addr or "local"


def _session_from_request(app: Flask) -> tuple[str, Path]:
    _maybe_cleanup_old_sessions(app)
    session_id = _request_session_id()
    safe_session_id = _safe_session_id(session_id or uuid.uuid4().hex)
    session_dir = (app.config["OUTPUT_ROOT"] / safe_session_id).resolve()
    _ensure_under(session_dir, app.config["OUTPUT_ROOT"])
    session_dir.mkdir(exist_ok=True, parents=True)
    return safe_session_id, session_dir


def _request_session_id() -> Optional[str]:
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        return session_id
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id")
    if not session_id:
        session_id = request.form.get("session_id") or request.args.get("session_id")
    return session_id


def _current_session_id() -> str:
    session_id = _request_session_id()
    return _safe_session_id(session_id) if session_id else ""


def _csrf_token(app: Flask, session_id: str) -> str:
    secret = str(app.secret_key or "").encode("utf-8")
    return hmac.new(secret, _safe_session_id(session_id).encode("utf-8"), hashlib.sha256).hexdigest()


def _require_csrf(app: Flask) -> None:
    if not app.config["VIDEO_APP_REQUIRE_CSRF"]:
        return
    session_id = _request_session_id()
    if not session_id:
        raise AppError("csrf_failed", "Missing session cookie.", 403)
    expected = _csrf_token(app, session_id)
    token = request.headers.get("X-CSRF-Token") or request.form.get("csrf_token")
    if request.is_json and not token:
        payload = request.get_json(silent=True) or {}
        token = payload.get("csrf_token")
    if not token or not hmac.compare_digest(str(token), expected):
        raise AppError("csrf_failed", "Request verification failed. Refresh the page and try again.", 403)


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", session_id)
    return cleaned[:64] or "default"


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _output_url(session_id: str, path: Optional[Path], session_dir: Path) -> Optional[str]:
    if path is None:
        return None
    rel_path = path.resolve().relative_to(session_dir.resolve()).as_posix()
    version = path.stat().st_mtime_ns if path.exists() else time.time_ns()
    return f"/outputs/{session_id}/{rel_path}?v={version}"


def _job_output_dir(session_dir: Path, kind: str, job_id: str) -> Path:
    job_dir = (session_dir / "jobs" / f"{kind}_{job_id}").resolve()
    _ensure_under(job_dir, session_dir)
    job_dir.mkdir(exist_ok=True, parents=True)
    return job_dir


def _resolve_bundled_character_cfg(raw_path: str) -> Path:
    path = _resolve_allowed_path(raw_path, [CHARACTERS_DIR])
    if path.name != "char_cfg.yaml":
        raise AppError("invalid_character", "Character config must be a char_cfg.yaml file.")
    return path


def _resolve_character_cfg_for_render(raw_path: str, app: Flask) -> Path:
    path = _resolve_allowed_path(raw_path, [CHARACTERS_DIR, app.config["OUTPUT_ROOT"]])
    if path.name != "char_cfg.yaml":
        raise AppError("invalid_character", "Character config must be a char_cfg.yaml file.")
    return path


def _resolve_motion_cfg_for_render(raw_path: str, app: Flask) -> Path:
    path = _resolve_allowed_path(raw_path, [MOTION_CFG_DIR, app.config["OUTPUT_ROOT"]])
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise AppError("invalid_motion", "Motion config must be a YAML file.")
    return path


def _resolve_retarget_cfg_for_render(raw_path: str, app: Flask) -> Path:
    path = _resolve_allowed_path(raw_path, [RETARGET_CFG_DIR, app.config["OUTPUT_ROOT"]])
    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise AppError("invalid_retarget", "Retarget config must be a YAML file.")
    return path


def _resolve_allowed_path(raw_path: str, roots: list[Path]) -> Path:
    if not raw_path:
        raise AppError("missing_path", "Expected a config path.")
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    for root in roots:
        root = root.resolve()
        if _is_under(resolved, root):
            return resolved
    raise AppError("path_not_allowed", "The requested file is outside the allowed app workspace.")


def _ensure_under(path: Path, root: Path) -> None:
    if not _is_under(path.resolve(), root.resolve()):
        raise AppError("path_not_allowed", "Path is outside the expected app workspace.")


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _copy_to_session(src: Path, dst: Path) -> Path:
    dst.parent.mkdir(exist_ok=True, parents=True)
    shutil.copy2(src, dst)
    return dst


def _require_upload(field_name: str) -> FileStorage:
    uploaded = request.files.get(field_name)
    if uploaded is None or uploaded.filename == "":
        raise AppError("missing_upload", f"Expected a {field_name} file.")
    return uploaded


def _validate_upload(uploaded: FileStorage, allowed_extensions: set[str], label: str) -> str:
    filename = secure_filename(uploaded.filename or "")
    suffix = Path(filename).suffix.lower()
    if not filename or suffix not in allowed_extensions:
        accepted = ", ".join(sorted(allowed_extensions))
        raise AppError("unsupported_file_type", f"Upload a supported {label} file ({accepted}).")
    return suffix


def _save_upload(uploaded: FileStorage, dst: Path) -> Path:
    dst.parent.mkdir(exist_ok=True, parents=True)
    uploaded.save(dst)
    if not dst.exists() or dst.stat().st_size == 0:
        raise AppError("empty_upload", "Uploaded file was empty.")
    return dst


def _parse_max_seconds(raw_value: Optional[str], app: Flask) -> int:
    configured_max = int(app.config["VIDEO_APP_MAX_SECONDS"])
    try:
        max_seconds = int(raw_value or configured_max)
    except (TypeError, ValueError):
        raise AppError("invalid_duration_limit", "Video duration limit must be a whole number.")
    if max_seconds < 1 or max_seconds > configured_max:
        raise AppError(
            "invalid_duration_limit",
            f"Video duration limit must be between 1 and {configured_max} seconds.",
        )
    return max_seconds


def _validate_video_metadata(metadata, app: Flask) -> None:
    if metadata.width <= 0 or metadata.height <= 0:
        raise AppError("invalid_video", "Could not read video dimensions.")
    max_width = int(app.config["VIDEO_APP_MAX_VIDEO_WIDTH"])
    max_height = int(app.config["VIDEO_APP_MAX_VIDEO_HEIGHT"])
    max_fps = float(app.config["VIDEO_APP_MAX_VIDEO_FPS"])
    if metadata.width > max_width or metadata.height > max_height:
        raise AppError(
            "video_too_large",
            f"Video resolution must be at most {max_width}x{max_height}.",
        )
    if metadata.fps > max_fps:
        raise AppError("video_fps_too_high", f"Video frame rate must be at most {max_fps:g} FPS.")


def _validate_video_file(video_path: Path, max_seconds: int, app: Flask) -> None:
    metadata = validate_video_duration(video_path, max_seconds)
    _validate_video_metadata(metadata, app)


def _validate_image_file(image_path: Path, app: Flask) -> None:
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except (OSError, UnidentifiedImageError) as e:
        raise AppError("invalid_image", "Upload a readable PNG, JPEG, or WebP drawing.") from e

    if image_format not in IMAGE_FORMATS:
        raise AppError("invalid_image", "Upload a readable PNG, JPEG, or WebP drawing.")

    max_width = int(app.config["VIDEO_APP_MAX_IMAGE_WIDTH"])
    max_height = int(app.config["VIDEO_APP_MAX_IMAGE_HEIGHT"])
    if width > max_width or height > max_height:
        raise AppError("image_too_large", f"Drawing image must be at most {max_width}x{max_height}.")


def _write_annotation_image(image_path: Path, output_path: Path) -> Path:
    output_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        with Image.open(image_path) as image:
            image = ImageOps.exif_transpose(image)
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                image = image.convert("RGBA")
                background = Image.new("RGBA", image.size, (255, 255, 255, 255))
                background.alpha_composite(image)
                image = background.convert("RGB")
            else:
                image = image.convert("RGB")
            image.save(output_path, format="PNG")
    except (OSError, UnidentifiedImageError) as e:
        raise AppError("invalid_image", "Upload a readable PNG, JPEG, or WebP drawing.") from e
    return output_path


def _validate_bvh_file(bvh_path: Path, app: Flask) -> None:
    max_bytes = int(app.config["VIDEO_APP_MAX_BVH_MB"]) * 1024 * 1024
    if bvh_path.stat().st_size > max_bytes:
        raise AppError("bvh_too_large", f"BVH file must be at most {app.config['VIDEO_APP_MAX_BVH_MB']} MB.")
    try:
        contents = bvh_path.read_text(encoding="utf-8")
    except UnicodeDecodeError as e:
        raise AppError("invalid_bvh", "BVH file must be readable text.") from e
    if "HIERARCHY" not in contents or "MOTION" not in contents:
        raise AppError("invalid_bvh", "BVH file must include HIERARCHY and MOTION sections.")


def _required_payload(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not value:
        raise AppError("missing_field", f"Expected '{key}' in the request body.")
    return str(value)


def _image_to_annotations(image_path: Path, out_dir: Path, timeout: float) -> None:
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    from image_to_annotations import image_to_annotations

    image_to_annotations(str(image_path), str(out_dir), request_timeout=timeout)


def _ensure_joint_overlay(char_cfg_path: Path, overlay_path: Path) -> Path:
    if overlay_path.exists():
        return overlay_path

    with char_cfg_path.open("r") as f:
        char_cfg = yaml.safe_load(f)

    texture_path = char_cfg_path.parent / "texture.png"
    image = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise AppError("invalid_character", "Could not read the generated character texture.")
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    for joint in char_cfg["skeleton"]:
        x, y = [int(round(v)) for v in joint["loc"]]
        cv2.circle(image, (x, y), 5, (0, 0, 0), 3)
        cv2.putText(image, joint["name"], (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, 2)

    overlay_path.parent.mkdir(exist_ok=True, parents=True)
    cv2.imwrite(str(overlay_path), image)
    return overlay_path


def _retarget_for_motion(motion_cfg: Path) -> Path:
    try:
        with motion_cfg.open("r") as f:
            cfg = yaml.safe_load(f) or {}
        bvh_name = Path(cfg.get("filepath", "")).name
        return RETARGET_CFG_BY_BVH.get(bvh_name, DEFAULT_RETARGET_CFG)
    except OSError:
        return DEFAULT_RETARGET_CFG


def _read_bvh_frame_time(bvh_path: Path) -> float:
    with bvh_path.open("r") as f:
        for line in f:
            if line.strip().startswith("Frame Time:"):
                return float(line.split(":", 1)[1].strip())
    return 1.0 / 30.0


def _write_mvc_cfg(
    character_cfg: Path,
    motion_cfg: Path,
    retarget_cfg: Path,
    output_video: Path,
    mvc_cfg: Path,
) -> None:
    cfg = {
        "scene": {
            "ADD_FLOOR": False,
            "ADD_AD_RETARGET_BVH": False,
            "ANIMATED_CHARACTERS": [
                {
                    "character_cfg": str(character_cfg),
                    "motion_cfg": str(motion_cfg),
                    "retarget_cfg": str(retarget_cfg),
                }
            ],
        },
        "view": {
            "WINDOW_DIMENSIONS": [500, 500],
            "CAMERA_POS": [0.0, 0.7, 2.4],
            "CAMERA_FWD": [0.0, 0.5, 2.4],
            "DRAW_AD_RIG": False,
            "DRAW_AD_TXTR": True,
            "DRAW_AD_COLOR": False,
            "DRAW_AD_MESH_LINES": False,
            "USE_MESA": False,
        },
        "controller": {
            "MODE": "video_render",
            "OUTPUT_VIDEO_PATH": str(output_video),
            "OUTPUT_VIDEO_CODEC": "mp4v",
        },
    }
    mvc_cfg.parent.mkdir(exist_ok=True, parents=True)
    with mvc_cfg.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_render(mvc_cfg: Path, timeout: Optional[float] = None) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "animated_drawings.render", str(mvc_cfg)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(f"Animation render failed with exit code {result.returncode}.\n{detail}")


def _maybe_cleanup_old_sessions(app: Flask) -> None:
    now = time.time()
    last_run = float(app.config.get("_OUTPUT_CLEANUP_LAST_RUN", 0.0))
    interval = float(app.config["VIDEO_APP_CLEANUP_INTERVAL_SECONDS"])
    if interval > 0 and now - last_run < interval:
        return
    app.config["_OUTPUT_CLEANUP_LAST_RUN"] = now
    _cleanup_old_sessions(app.config["OUTPUT_ROOT"], float(app.config["VIDEO_APP_OUTPUT_TTL_HOURS"]), now=now)


def _cleanup_old_sessions(output_root: Path, ttl_hours: float, now: Optional[float] = None) -> None:
    if ttl_hours <= 0 or not output_root.exists():
        return
    output_root = output_root.resolve()
    cutoff = (now or time.time()) - (ttl_hours * 60 * 60)
    for child in output_root.iterdir():
        try:
            if not child.is_dir() or child.is_symlink():
                continue
            resolved = child.resolve()
            if not _is_under(resolved, output_root):
                continue
            if _latest_mtime(resolved) < cutoff:
                shutil.rmtree(resolved)
        except OSError:
            logger.exception("Failed to clean old video app output: %s", child)


def _latest_mtime(path: Path) -> float:
    latest = path.stat().st_mtime
    for child in path.rglob("*"):
        try:
            latest = max(latest, child.stat().st_mtime)
        except OSError:
            continue
    return latest


def _json_error(code: str, message: str, status_code: int):
    return jsonify({"error": {"code": code, "message": message}}), status_code


def _json_exception(error: Exception):
    if isinstance(error, AppError):
        return _json_error(error.code, error.message, error.status_code)
    if isinstance(error, VideoDurationError):
        return _json_error("video_too_long", str(error), 400)
    if isinstance(error, PoseVideoError):
        payload = _error_payload(error)
        return _json_error(payload["code"], payload["message"], 400)
    logger.exception("Video app request failed")
    payload = _error_payload(error)
    return _json_error(payload["code"], payload["message"], 500)


def _error_payload(error: Exception) -> dict[str, str]:
    if isinstance(error, AppError):
        return {"code": error.code, "message": error.message}
    if isinstance(error, VideoDurationError):
        return {"code": "video_too_long", "message": str(error)}
    if isinstance(error, subprocess.TimeoutExpired):
        return {"code": "timeout", "message": "The operation timed out. Try a smaller input or try again."}
    if isinstance(error, requests.Timeout):
        return {"code": "torchserve_timeout", "message": "Drawing analysis timed out. Check TorchServe and try again."}
    if isinstance(error, requests.RequestException):
        return {"code": "torchserve_unavailable", "message": "Drawing analysis service is unavailable."}
    if isinstance(error, PoseVideoError):
        message = str(error)
        if "Could not open video" in message:
            return {"code": "invalid_video", "message": "Upload a readable video file."}
        if "No human pose" in message:
            return {"code": "pose_not_found", "message": "No human pose was detected in the video."}
        if "ffmpeg" in message.lower() or "transcode" in message.lower() or "browser" in message.lower():
            return {"code": "browser_video_failed", "message": "Could not prepare a browser-playable video."}
        return {"code": "video_pose_failed", "message": "Could not estimate pose from the video."}
    if isinstance(error, AssertionError):
        return {"code": "drawing_detection_failed", "message": _drawing_error_message(str(error))}
    if isinstance(error, RuntimeError) and "Animation render failed" in str(error):
        return {"code": "render_failed", "message": "Animation rendering failed."}
    return {"code": "internal_error", "message": "Something went wrong while processing the request."}


def _drawing_error_message(message: str) -> str:
    if "humanoid" in message.lower():
        return "Could not detect a single drawn humanoid in the image."
    if "skeleton" in message.lower():
        return "Could not detect a usable character skeleton in the drawing."
    return "Could not analyze the drawing."


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5060, debug=False, threaded=True)
