from io import BytesIO
from pathlib import Path

import pytest
import yaml

pytest.importorskip("flask")

from animated_drawings.video_pose import MotionBuildResult
from examples.video_app import server


def test_video_app_assets_endpoint(tmp_path: Path):
    app = server.create_app(output_root=tmp_path)
    client = app.test_client()

    response = client.get("/api/assets")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["motions"]
    assert payload["characters"]


def test_video_app_bundled_drawing_endpoint(tmp_path: Path):
    app = server.create_app(output_root=tmp_path)
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
    assert payload["joint_overlay_url"].endswith("drawing_joint_overlay.png")


def test_video_app_video_motion_endpoint_is_mockable(tmp_path: Path, monkeypatch):
    def fake_build_motion_from_video(video_path, out_dir, max_seconds):
        pose = out_dir / "pose_sequence.json"
        overlay = out_dir / "pose_overlay.mp4"
        bvh = out_dir / "motion.bvh"
        motion = out_dir / "motion.yaml"
        pose.write_text("{}")
        overlay.write_bytes(b"overlay")
        bvh.write_text("HIERARCHY\n")
        motion.write_text("filepath: motion.bvh\n")
        return MotionBuildResult(pose, overlay, bvh, motion)

    monkeypatch.setattr(server, "build_motion_from_video", fake_build_motion_from_video)
    app = server.create_app(output_root=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/motion/video",
        data={
            "session_id": "video-session",
            "video": (BytesIO(b"fake video"), "clip.mp4"),
        },
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["motion_cfg"].endswith("motion.yaml")
    assert payload["overlay_url"].endswith("pose_overlay.mp4")


def test_video_app_render_endpoint_is_mockable(tmp_path: Path, monkeypatch):
    def fake_run_render(mvc_cfg_path):
        with open(mvc_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        Path(cfg["controller"]["OUTPUT_VIDEO_PATH"]).write_bytes(b"mp4")

    monkeypatch.setattr(server, "_run_render", fake_run_render)
    app = server.create_app(output_root=tmp_path)
    client = app.test_client()

    response = client.post(
        "/api/render",
        json={
            "session_id": "render-session",
            "character_cfg": "examples/characters/char1/char_cfg.yaml",
            "motion_cfg": "examples/config/motion/dab.yaml",
            "retarget_cfg": "examples/config/retarget/fair1_ppf.yaml",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["animation_url"].endswith("animated_drawing.mp4")
    assert (tmp_path / "render-session" / "animated_drawing.mp4").exists()
