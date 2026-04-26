import os
import time
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from PIL import Image

pytest.importorskip("flask")

from animated_drawings.video_pose import MotionBuildResult, PoseVideoError
from examples.video_app import server


def _url_path(url: str) -> str:
    return url.split("?", 1)[0]


def _test_app(tmp_path: Path):
    return server.create_app(output_root=tmp_path, job_manager=server.JobManager(run_inline=True))


def _completed_job_result(client, payload: dict) -> dict:
    assert "job" in payload
    response = client.get(payload["job"]["status_url"])
    assert response.status_code == 200
    job = response.get_json()["job"]
    assert job["status"] == "completed"
    return job["result"]


def _failed_job(client, payload: dict) -> dict:
    assert "job" in payload
    response = client.get(payload["job"]["status_url"])
    assert response.status_code == 200
    job = response.get_json()["job"]
    assert job["status"] == "failed"
    return job


def _png_bytes(width: int = 16, height: int = 16) -> BytesIO:
    data = BytesIO()
    Image.new("RGB", (width, height), "white").save(data, format="PNG")
    data.seek(0)
    return data


def _fake_motion_result(video_path, out_dir, max_seconds):
    pose = out_dir / "pose_sequence.json"
    overlay = out_dir / "pose_overlay.mp4"
    bvh = out_dir / "motion.bvh"
    motion = out_dir / "motion.yaml"
    pose.write_text("{}")
    overlay.write_bytes(b"overlay")
    bvh.write_text("HIERARCHY\nMOTION\n")
    motion.write_text("filepath: motion.bvh\n")
    return MotionBuildResult(pose, overlay, bvh, motion)


def test_video_app_assets_endpoint(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.get("/api/assets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["motions"]
    assert payload["characters"]
    assert payload["samples"]
    assert payload["limits"]["max_seconds"] == 10
    assert payload["limits"]["accepted_extensions"]["video"]
    assert payload["demo"]["animation_url"] == "/demo-media/garlic.gif"


def test_video_app_index_issues_session_cookie(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.get("/")

    assert response.status_code == 200
    assert server.SESSION_COOKIE_NAME in response.headers["Set-Cookie"]
    assert b"window.APP_SESSION_ID" in response.data
    assert b"Start creating" in response.data
    assert b"Files are processed locally" in response.data
    assert b'id="workflow"' in response.data


def test_demo_media_route_is_allowlisted(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.get("/demo-media/garlic.gif")
    missing = client.get("/demo-media/not_allowed.gif")

    assert response.status_code == 200
    assert response.content_type == "image/gif"
    assert missing.status_code == 404


def test_video_app_bundled_drawing_endpoint(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/drawing",
        data={
            "session_id": "test-session",
            "character_cfg": "examples/characters/char1/char_cfg.yaml",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["character_cfg"].endswith("examples/characters/char1/char_cfg.yaml")
    assert _url_path(payload["joint_overlay_url"]).endswith("drawing_joint_overlay.png")


def test_video_app_video_motion_endpoint_returns_completed_job(tmp_path: Path, monkeypatch):
    def fake_validate_video_duration(video_path, max_seconds):
        assert video_path.name == "input_video.mp4"
        assert max_seconds == 10
        return SimpleNamespace(width=32, height=32, fps=30.0, duration=1.0)

    monkeypatch.setattr(server, "validate_video_duration", fake_validate_video_duration)
    monkeypatch.setattr(server, "build_motion_from_video", _fake_motion_result)
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/motion/video",
        data={
            "session_id": "video-session",
            "video": (BytesIO(b"fake video"), "clip.mp4"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    result = _completed_job_result(client, response.get_json())
    assert result["motion_cfg"].endswith("motion.yaml")
    assert "/jobs/motion_video_" in _url_path(result["overlay_url"])
    assert _url_path(result["overlay_url"]).endswith("pose_overlay.mp4")


def test_video_app_bvh_endpoint_returns_completed_job(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/motion/bvh",
        data={
            "session_id": "bvh-session",
            "bvh": (BytesIO(b"HIERARCHY\nMOTION\nFrames: 1\nFrame Time: 0.0333333\n"), "motion.bvh"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    result = _completed_job_result(client, response.get_json())
    assert result["motion_cfg"].endswith("uploaded_motion.yaml")
    assert _url_path(result["bvh_url"]).endswith("uploaded_motion.bvh")


def test_video_app_uploaded_drawing_endpoint_returns_completed_job(tmp_path: Path, monkeypatch):
    def fake_image_to_annotations(image_path, out_dir, timeout):
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "char_cfg.yaml").write_text(yaml.safe_dump({"skeleton": [], "height": 1, "width": 1}))
        Image.new("RGB", (1, 1), "white").save(out_dir / "joint_overlay.png")

    monkeypatch.setattr(server, "_image_to_annotations", fake_image_to_annotations)
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/drawing",
        data={
            "session_id": "drawing-session",
            "drawing": (_png_bytes(), "drawing.png"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    result = _completed_job_result(client, response.get_json())
    assert result["character_cfg"].endswith("character/char_cfg.yaml")
    assert _url_path(result["joint_overlay_url"]).endswith("character/joint_overlay.png")


def test_video_app_render_endpoint_returns_completed_job(tmp_path: Path, monkeypatch):
    def fake_run_render(mvc_cfg_path, timeout=None):
        with open(mvc_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        Path(cfg["controller"]["OUTPUT_VIDEO_PATH"]).write_bytes(b"mp4")

    def fake_transcode(input_path, output_path=None, strict=False, timeout=None):
        output_path = Path(output_path or input_path)
        output_path.write_bytes(Path(input_path).read_bytes())
        return output_path

    monkeypatch.setattr(server, "_run_render", fake_run_render)
    monkeypatch.setattr(server, "transcode_to_browser_mp4", fake_transcode)
    app = _test_app(tmp_path)
    client = app.test_client()

    assets = client.get("/api/assets").get_json()
    sample = assets["samples"][0]

    response = client.post(
        "/api/render",
        json={
            "session_id": "render-session",
            "character_cfg": sample["character_cfg"],
            "motion_cfg": sample["motion_cfg"],
            "retarget_cfg": sample["retarget_cfg"],
        },
    )

    assert response.status_code == 202
    result = _completed_job_result(client, response.get_json())
    animation_path = _url_path(result["animation_url"])
    assert "/jobs/render_" in animation_path
    assert animation_path.endswith("animated_drawing.mp4")
    output_rel = animation_path.split("/outputs/render-session/", 1)[1]
    assert (tmp_path / "render-session" / output_rel).exists()


def test_video_app_failed_job_exposes_public_error(tmp_path: Path, monkeypatch):
    def fake_validate_video_duration(video_path, max_seconds):
        return SimpleNamespace(width=32, height=32, fps=30.0, duration=1.0)

    def fake_build_motion_from_video(video_path, out_dir, max_seconds):
        raise PoseVideoError("No human pose was detected in the video.")

    monkeypatch.setattr(server, "validate_video_duration", fake_validate_video_duration)
    monkeypatch.setattr(server, "build_motion_from_video", fake_build_motion_from_video)
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/motion/video",
        data={
            "session_id": "failed-session",
            "video": (BytesIO(b"fake video"), "clip.mp4"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 202
    job = _failed_job(client, response.get_json())
    assert job["error"]["code"] == "pose_not_found"
    assert job["error"]["message"] == "No human pose was detected in the video."


def test_video_app_rate_limit_returns_429(tmp_path: Path, monkeypatch):
    def fake_validate_video_duration(video_path, max_seconds):
        return SimpleNamespace(width=32, height=32, fps=30.0, duration=1.0)

    monkeypatch.setattr(server, "validate_video_duration", fake_validate_video_duration)
    monkeypatch.setattr(server, "build_motion_from_video", _fake_motion_result)
    app = _test_app(tmp_path)
    app.config["VIDEO_APP_SESSION_JOB_LIMIT"] = 1
    client = app.test_client()

    first = client.post(
        "/api/motion/video",
        data={"session_id": "limited-session", "video": (BytesIO(b"fake video"), "clip.mp4")},
        content_type="multipart/form-data",
    )
    second = client.post(
        "/api/motion/video",
        data={"session_id": "limited-session", "video": (BytesIO(b"fake video"), "clip.mp4")},
        content_type="multipart/form-data",
    )

    assert first.status_code == 202
    assert second.status_code == 429
    assert second.get_json()["error"]["code"] == "rate_limited"


def test_video_app_queue_guard_returns_429(tmp_path: Path, monkeypatch):
    def fake_validate_video_duration(video_path, max_seconds):
        return SimpleNamespace(width=32, height=32, fps=30.0, duration=1.0)

    monkeypatch.setattr(server, "validate_video_duration", fake_validate_video_duration)
    app = _test_app(tmp_path)
    app.config["VIDEO_APP_MAX_PENDING_JOBS"] = 0
    client = app.test_client()

    response = client.post(
        "/api/motion/video",
        data={"session_id": "busy-session", "video": (BytesIO(b"fake video"), "clip.mp4")},
        content_type="multipart/form-data",
    )

    assert response.status_code == 429
    assert response.get_json()["error"]["code"] == "server_busy"


def test_video_app_rejects_unsupported_upload_type(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/motion/bvh",
        data={
            "session_id": "bad-upload",
            "bvh": (BytesIO(b"not bvh"), "motion.txt"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "unsupported_file_type"


def test_video_app_rejects_malformed_bvh(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/motion/bvh",
        data={
            "session_id": "bad-bvh",
            "bvh": (BytesIO(b"not a bvh"), "motion.bvh"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_bvh"


def test_video_app_rejects_image_extension_mismatch(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/drawing",
        data={
            "session_id": "bad-image",
            "drawing": (_png_bytes(), "drawing.jpg"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "file_type_mismatch"


def test_video_app_rejects_unreadable_video(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/motion/video",
        data={
            "session_id": "bad-video",
            "video": (BytesIO(b"not a video"), "clip.mp4"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "invalid_video"


def test_video_app_render_rejects_disallowed_path(tmp_path: Path):
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/render",
        json={
            "session_id": "bad-render",
            "character_cfg": "/etc/passwd",
            "motion_cfg": "examples/config/motion/dab.yaml",
            "retarget_cfg": "examples/config/retarget/fair1_ppf.yaml",
        },
    )

    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "path_not_allowed"


def test_output_files_are_not_cached(tmp_path: Path):
    session_dir = tmp_path / "cache-session"
    session_dir.mkdir()
    (session_dir / "clip.mp4").write_bytes(b"mp4")
    app = _test_app(tmp_path)
    client = app.test_client()

    response = client.get("/outputs/cache-session/clip.mp4")

    assert response.status_code == 200
    assert "no-store" in response.headers["Cache-Control"]


def test_cleanup_old_sessions_removes_only_expired_session_dirs(tmp_path: Path):
    now = time.time()
    old_session = tmp_path / "old-session"
    fresh_session = tmp_path / "fresh-session"
    unrelated_file = tmp_path / "notes.txt"
    old_session.mkdir()
    fresh_session.mkdir()
    (old_session / "clip.mp4").write_bytes(b"old")
    (fresh_session / "clip.mp4").write_bytes(b"fresh")
    unrelated_file.write_text("keep")

    old_time = now - (48 * 60 * 60)
    os.utime(old_session / "clip.mp4", (old_time, old_time))
    os.utime(old_session, (old_time, old_time))

    server._cleanup_old_sessions(tmp_path, ttl_hours=24, now=now)

    assert not old_session.exists()
    assert fresh_session.exists()
    assert unrelated_file.exists()
