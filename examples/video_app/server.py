from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Optional

import cv2
import yaml
from flask import Flask, jsonify, render_template, request, send_from_directory

from animated_drawings.video_pose import build_motion_from_video, write_motion_config_for_bvh
from animated_drawings.video_pose.constants import DEFAULT_MAX_SECONDS
from animated_drawings.video_pose.types import PoseVideoError


APP_DIR = Path(__file__).resolve().parent
EXAMPLES_DIR = APP_DIR.parent
REPO_ROOT = EXAMPLES_DIR.parent
DEFAULT_OUTPUT_ROOT = APP_DIR / "outputs"
MEDIAPIPE_RETARGET_CFG = REPO_ROOT / "examples/config/retarget/mediapipe_pfp.yaml"
DEFAULT_RETARGET_CFG = REPO_ROOT / "examples/config/retarget/fair1_ppf.yaml"
RETARGET_CFG_BY_BVH = {
    "cxk_mediapipe.bvh": MEDIAPIPE_RETARGET_CFG,
    "motion.bvh": MEDIAPIPE_RETARGET_CFG,
    "uploaded_motion.bvh": MEDIAPIPE_RETARGET_CFG,
    "jumping_jacks.bvh": REPO_ROOT / "examples/config/retarget/cmu1_pfp.yaml",
    "walk-cycle.bvh": REPO_ROOT / "examples/config/retarget/walk_cycle_pfp.yaml",
}


def create_app(output_root: Optional[Path] = None) -> Flask:
    app = Flask(
        __name__,
        template_folder=str(APP_DIR / "templates"),
        static_folder=str(APP_DIR / "static"),
        static_url_path="/static",
    )
    app.config["OUTPUT_ROOT"] = Path(output_root or DEFAULT_OUTPUT_ROOT).resolve()
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

    @app.route("/")
    def index():
        return render_template("index.html", max_seconds=DEFAULT_MAX_SECONDS)

    @app.route("/api/assets")
    def assets():
        motions = []
        for motion_cfg in sorted((EXAMPLES_DIR / "config/motion").glob("*.yaml")):
            motions.append(
                {
                    "id": _repo_rel(motion_cfg),
                    "name": motion_cfg.stem.replace("_", " "),
                    "retarget_cfg": _repo_rel(_retarget_for_motion(motion_cfg)),
                }
            )

        characters = []
        for char_cfg in sorted((EXAMPLES_DIR / "characters").glob("*/char_cfg.yaml")):
            characters.append(
                {
                    "id": _repo_rel(char_cfg),
                    "name": char_cfg.parent.name,
                }
            )

        return jsonify({"motions": motions, "characters": characters})

    @app.route("/api/motion/video", methods=["POST"])
    def motion_video():
        try:
            session_id, session_dir = _session_from_request(app)
            uploaded = request.files.get("video")
            if uploaded is None or uploaded.filename == "":
                return _json_error("Expected a video file.", 400)

            suffix = Path(uploaded.filename).suffix or ".webm"
            video_path = session_dir / f"input_video{suffix}"
            uploaded.save(video_path)

            max_seconds = int(request.form.get("max_seconds", DEFAULT_MAX_SECONDS))
            result = build_motion_from_video(video_path, session_dir, max_seconds=max_seconds)
            return jsonify(
                {
                    "session_id": session_id,
                    "motion_cfg": str(result.motion_config_path),
                    "retarget_cfg": str(MEDIAPIPE_RETARGET_CFG),
                    "bvh_url": _output_url(session_id, result.bvh_path, session_dir),
                    "overlay_url": _output_url(session_id, result.overlay_video_path, session_dir),
                    "pose_sequence_url": _output_url(session_id, result.pose_sequence_path, session_dir),
                }
            )
        except Exception as e:
            return _json_exception(e)

    @app.route("/api/motion/bvh", methods=["POST"])
    def motion_bvh():
        try:
            session_id, session_dir = _session_from_request(app)
            uploaded = request.files.get("bvh")
            if uploaded is None or uploaded.filename == "":
                return _json_error("Expected a BVH file.", 400)

            bvh_path = session_dir / "uploaded_motion.bvh"
            uploaded.save(bvh_path)
            motion_cfg = session_dir / "uploaded_motion.yaml"
            write_motion_config_for_bvh(bvh_path, motion_cfg, _read_bvh_frame_time(bvh_path))
            return jsonify(
                {
                    "session_id": session_id,
                    "motion_cfg": str(motion_cfg),
                    "retarget_cfg": str(MEDIAPIPE_RETARGET_CFG),
                    "bvh_url": _output_url(session_id, bvh_path, session_dir),
                }
            )
        except Exception as e:
            return _json_exception(e)

    @app.route("/api/drawing", methods=["POST"])
    def drawing():
        try:
            session_id, session_dir = _session_from_request(app)
            character_cfg = request.form.get("character_cfg")
            uploaded = request.files.get("drawing")

            if character_cfg:
                char_cfg_path = _resolve_allowed_path(character_cfg, app.config["OUTPUT_ROOT"])
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
                return _json_error("Expected a bundled character or drawing upload.", 400)

            suffix = Path(uploaded.filename).suffix or ".png"
            drawing_path = session_dir / f"uploaded_drawing{suffix}"
            uploaded.save(drawing_path)
            char_dir = session_dir / "character"
            _image_to_annotations(drawing_path, char_dir)
            char_cfg_path = char_dir / "char_cfg.yaml"
            overlay_path = _ensure_joint_overlay(char_cfg_path, char_dir / "joint_overlay.png")
            return jsonify(
                {
                    "session_id": session_id,
                    "character_cfg": str(char_cfg_path),
                    "joint_overlay_url": _output_url(session_id, overlay_path, session_dir),
                }
            )
        except Exception as e:
            return _json_exception(e)

    @app.route("/api/render", methods=["POST"])
    def render_animation():
        try:
            session_id, session_dir = _session_from_request(app)
            payload = request.get_json(force=True)
            character_cfg = _resolve_allowed_path(payload["character_cfg"], app.config["OUTPUT_ROOT"])
            motion_cfg = _resolve_allowed_path(payload["motion_cfg"], app.config["OUTPUT_ROOT"])
            retarget_cfg = _resolve_allowed_path(
                payload.get("retarget_cfg") or str(_retarget_for_motion(motion_cfg)),
                app.config["OUTPUT_ROOT"],
            )

            output_video = session_dir / "animated_drawing.mp4"
            mvc_cfg = session_dir / "render_mvc.yaml"
            _write_mvc_cfg(character_cfg, motion_cfg, retarget_cfg, output_video, mvc_cfg)
            _run_render(mvc_cfg)
            return jsonify(
                {
                    "session_id": session_id,
                    "animation_url": _output_url(session_id, output_video, session_dir),
                    "mvc_cfg": str(mvc_cfg),
                }
            )
        except Exception as e:
            return _json_exception(e)

    @app.route("/outputs/<session_id>/<path:filename>")
    def outputs(session_id: str, filename: str):
        safe_session_id = _safe_session_id(session_id)
        session_dir = (app.config["OUTPUT_ROOT"] / safe_session_id).resolve()
        _ensure_under(session_dir, app.config["OUTPUT_ROOT"])
        return send_from_directory(session_dir, filename)

    return app


def _session_from_request(app: Flask) -> tuple[str, Path]:
    session_id = request.form.get("session_id")
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        session_id = payload.get("session_id", session_id)
    safe_session_id = _safe_session_id(session_id or "default")
    session_dir = (app.config["OUTPUT_ROOT"] / safe_session_id).resolve()
    _ensure_under(session_dir, app.config["OUTPUT_ROOT"])
    session_dir.mkdir(exist_ok=True, parents=True)
    return safe_session_id, session_dir


def _safe_session_id(session_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_-]", "", session_id)
    return cleaned[:64] or "default"


def _repo_rel(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def _output_url(session_id: str, path: Optional[Path], session_dir: Path) -> Optional[str]:
    if path is None:
        return None
    rel_path = path.resolve().relative_to(session_dir.resolve()).as_posix()
    return f"/outputs/{session_id}/{rel_path}"


def _resolve_allowed_path(raw_path: str, output_root: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = REPO_ROOT / path
    resolved = path.resolve()
    if _is_under(resolved, REPO_ROOT) or _is_under(resolved, output_root):
        return resolved
    raise ValueError(f"Path is outside the app workspace: {raw_path}")


def _ensure_under(path: Path, root: Path) -> None:
    if not _is_under(path.resolve(), root.resolve()):
        raise ValueError(f"Path is outside expected root: {path}")


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


def _image_to_annotations(image_path: Path, out_dir: Path) -> None:
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    from image_to_annotations import image_to_annotations

    image_to_annotations(str(image_path), str(out_dir))


def _ensure_joint_overlay(char_cfg_path: Path, overlay_path: Path) -> Path:
    if overlay_path.exists():
        return overlay_path

    with char_cfg_path.open("r") as f:
        char_cfg = yaml.safe_load(f)

    texture_path = char_cfg_path.parent / "texture.png"
    image = cv2.imread(str(texture_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read character texture: {texture_path}")
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
    with mvc_cfg.open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _run_render(mvc_cfg: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "animated_drawings.render", str(mvc_cfg)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = "\n".join(part for part in (result.stdout, result.stderr) if part).strip()
        raise RuntimeError(f"Animation render failed with exit code {result.returncode}.\n{detail}")


def _json_error(message: str, status_code: int):
    return jsonify({"error": message}), status_code


def _json_exception(error: Exception):
    traceback.print_exc()
    status_code = 400 if isinstance(error, (PoseVideoError, ValueError)) else 500
    return jsonify({"error": str(error)}), status_code


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5060, debug=False, threaded=False)
