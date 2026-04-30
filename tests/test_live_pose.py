import importlib.util
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import animated_drawings.video_pose.live as live_helpers
from animated_drawings.config import RetargetConfig
from animated_drawings.video_pose import (
    CausalPoseSmoother,
    CausalPoseSmootherConfig,
    LivePoseRetargeter,
    PoseFrame,
    analyze_pose_frame,
    compose_live_dashboard,
    draw_pose_overlay,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RETARGET_CFG = REPO_ROOT / "examples/config/retarget/mediapipe_pfp.yaml"


def test_live_retargeter_produces_finite_orientations():
    cfg = RetargetConfig(str(RETARGET_CFG))
    retargeter = LivePoseRetargeter(cfg)

    assert retargeter.update_pose(_pose_frame(0.0))
    orientations, depths, root = retargeter.get_retargeted_frame_data(0.0)

    assert set(cfg.char_joint_bvh_joints_mapping).issubset(orientations)
    assert all(np.isfinite(value) for value in orientations.values())
    assert all(np.isfinite(value) for value in depths.values())
    assert np.isfinite(root).all()


def test_live_retargeter_holds_last_pose_when_landmarks_missing():
    cfg = RetargetConfig(str(RETARGET_CFG))
    retargeter = LivePoseRetargeter(cfg)
    retargeter.update_pose(_pose_frame(0.0))
    before = retargeter.get_retargeted_frame_data(0.0)

    assert not retargeter.update_pose(PoseFrame(timestamp=0.1, landmarks={}))
    after = retargeter.get_retargeted_frame_data(0.1)

    assert after[0] == before[0]
    assert after[1] == before[1]
    np.testing.assert_allclose(after[2], before[2])


def test_live_retargeter_locked_root_returns_character_start_location():
    cfg = RetargetConfig(str(RETARGET_CFG))
    retargeter = LivePoseRetargeter(cfg, root_mode="locked")
    retargeter.update_pose(_pose_frame(0.0, root_shift=0.2))

    _, _, root = retargeter.get_retargeted_frame_data(0.0)

    np.testing.assert_allclose(root, np.array(cfg.char_start_loc, dtype=np.float32))


def test_live_retargeter_hip_root_tracks_hip_center_after_reference_frame():
    cfg = RetargetConfig(str(RETARGET_CFG))
    retargeter = LivePoseRetargeter(cfg, root_mode="hip")
    retargeter.update_pose(_pose_frame(0.0, root_shift=0.0))
    _, _, first_root = retargeter.get_retargeted_frame_data(0.0)

    retargeter.update_pose(_pose_frame(0.1, root_shift=0.08))
    _, _, moved_root = retargeter.get_retargeted_frame_data(0.1)

    assert moved_root[0] - first_root[0] == pytest.approx(0.08, abs=1e-6)


def test_causal_pose_smoother_smooths_and_repairs_low_confidence_landmarks():
    smoother = CausalPoseSmoother(CausalPoseSmootherConfig(alpha=0.25, visibility_threshold=0.35))
    smoother.process(PoseFrame(timestamp=0.0, landmarks={"LEFT_WRIST": [0.0, 0.0, 0.0, 1.0]}))

    smoothed = smoother.process(PoseFrame(timestamp=0.1, landmarks={"LEFT_WRIST": [1.0, 1.0, 1.0, 1.0]}))

    assert smoothed.landmarks["LEFT_WRIST"][:3] == pytest.approx([0.25, 0.25, 0.25])

    repaired = smoother.process(PoseFrame(timestamp=0.2, landmarks={"LEFT_WRIST": [0.9, 0.9, 0.9, 0.01]}))

    assert repaired.landmarks["LEFT_WRIST"][:3] == pytest.approx([0.25, 0.25, 0.25])


def test_pose_tracking_status_reports_full_pose():
    status = analyze_pose_frame(_pose_frame(0.0))

    assert status.state == "tracking"
    assert status.missing_landmarks == ()
    assert "Tracking" in status.message


def test_pose_tracking_status_reports_partial_full_body_guidance():
    frame = _pose_frame(0.0)
    del frame.landmarks["LEFT_ANKLE"]
    del frame.landmarks["RIGHT_KNEE"]

    status = analyze_pose_frame(frame)

    assert status.state == "partial"
    assert "full body" in status.message
    assert "ankles" in status.message
    assert "knees" in status.message
    assert status.missing_landmarks == ("RIGHT_KNEE", "LEFT_ANKLE")


def test_pose_tracking_status_reports_lost_pose():
    status = analyze_pose_frame(PoseFrame(timestamp=0.0, landmarks={}))

    assert status.state == "lost"
    assert "No pose" in status.message


def test_draw_pose_overlay_modifies_image():
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    status = analyze_pose_frame(_pose_frame(0.0))

    overlay = draw_pose_overlay(frame, _pose_frame(0.0), status)

    assert overlay.shape == frame.shape
    assert int(overlay.sum()) > 0


def test_compose_live_dashboard_dimensions():
    camera = np.zeros((120, 160, 3), dtype=np.uint8)
    animation = np.full((80, 80, 3), 255, dtype=np.uint8)
    status = analyze_pose_frame(_pose_frame(0.0))

    dashboard = compose_live_dashboard(camera, animation, status, pane_size=240)

    assert dashboard.shape == (454, 534, 3)


def test_compose_live_dashboard_draws_upload_tile_and_status():
    camera = np.zeros((120, 160, 3), dtype=np.uint8)
    animation = np.full((80, 80, 3), 255, dtype=np.uint8)
    status = analyze_pose_frame(_pose_frame(0.0))

    dashboard = compose_live_dashboard(
        camera,
        animation,
        status,
        pane_size=240,
        active_figure="char2",
        upload_status="Analyzing sketch.png...",
        figures=["char1", "char2", "uploaded sketch"],
        active_figure_index=1,
    )
    left, top, right, bottom = live_helpers.live_dashboard_upload_rect(240)
    upload_tile = dashboard[top:bottom, left:right]

    assert upload_tile.shape[0] > 0
    assert upload_tile.shape[1] > 0
    assert int(upload_tile.std()) > 0


def test_live_dashboard_text_fitting_ellipsizes_long_labels():
    fitted = live_helpers._ellipsize_text("a very long uploaded drawing name", 80, 0.5, 1)

    assert fitted.endswith("...")
    assert live_helpers._text_size(fitted, 0.5, 1)[0] <= 80


def test_webcam_figure_discovery_lists_bundled_characters():
    webcam = _webcam_module()

    options = webcam._discover_bundled_figures()

    assert [option.name for option in options] == ["char1", "char2", "char3", "char4", "char5", "char6"]
    assert all(option.bundled for option in options)
    assert all(option.character_cfg.name == "char_cfg.yaml" for option in options)


def test_webcam_custom_character_is_added_to_figure_options(tmp_path: Path):
    webcam = _webcam_module()
    custom_dir = tmp_path / "custom_figure"
    custom_dir.mkdir()
    custom_cfg = custom_dir / "char_cfg.yaml"
    custom_cfg.write_text("height: 1\nwidth: 1\nskeleton: []\n", encoding="utf-8")

    options, active_index = webcam._figure_options_for_character(str(custom_cfg))

    assert active_index == 0
    assert options[0].name == "custom_figure"
    assert options[0].character_cfg == custom_cfg.resolve()
    assert not options[0].bundled
    assert [option.name for option in options[1:]] == ["char1", "char2", "char3", "char4", "char5", "char6"]


def test_webcam_bundled_character_starts_on_matching_figure():
    webcam = _webcam_module()

    options, active_index = webcam._figure_options_for_character("examples/characters/char2/char_cfg.yaml")

    assert options[active_index].name == "char2"
    assert all(option.bundled for option in options)


def test_webcam_dashboard_key_requests_figure_switches():
    webcam = _webcam_module()
    figure_state = webcam.FigureState(
        options=[
            webcam.FigureOption("char1", Path("char1/char_cfg.yaml")),
            webcam.FigureOption("char2", Path("char2/char_cfg.yaml")),
            webcam.FigureOption("char3", Path("char3/char_cfg.yaml")),
        ],
        active_index=1,
        live_retargeter=SimpleNamespace(reset_root_reference=lambda: None),
    )
    state = webcam.RunState()
    smoother = SimpleNamespace(reset=lambda: None)

    webcam._handle_dashboard_key(ord("3"), state, smoother, figure_state.live_retargeter, figure_state)
    assert figure_state.pending_index == 2

    figure_state.pending_index = None
    webcam._handle_dashboard_key(ord("["), state, smoother, figure_state.live_retargeter, figure_state)
    assert figure_state.pending_index == 0

    figure_state.pending_index = None
    webcam._handle_dashboard_key(ord("]"), state, smoother, figure_state.live_retargeter, figure_state)
    assert figure_state.pending_index == 2

    webcam._handle_dashboard_key(ord("u"), state, smoother, figure_state.live_retargeter, figure_state)
    assert state.upload_requested

    webcam._handle_dashboard_key(ord("C"), state, smoother, figure_state.live_retargeter, figure_state)
    assert state.choose_character_requested


def test_webcam_dashboard_mouse_requests_upload():
    webcam = _webcam_module()
    state = webcam.RunState()
    left, top, right, bottom = webcam.live_dashboard_upload_rect(240)

    webcam._handle_dashboard_mouse(
        webcam.cv2.EVENT_LBUTTONUP,
        (left + right) // 2,
        (top + bottom) // 2,
        0,
        {"state": state, "pane_size": 240},
    )

    assert state.upload_requested


def test_webcam_upload_picker_adds_generated_character(tmp_path: Path, monkeypatch):
    webcam = _webcam_module()
    drawing_path = tmp_path / "My Sketch.png"
    Image.new("RGB", (12, 12), "white").save(drawing_path)

    def fake_image_to_annotations(image_path, out_dir, timeout):
        out_dir.mkdir(exist_ok=True, parents=True)
        (out_dir / "char_cfg.yaml").write_text("height: 12\nwidth: 12\nskeleton: []\n", encoding="utf-8")
        Image.new("RGBA", (12, 12), "white").save(out_dir / "texture.png")
        Image.new("L", (12, 12), 255).save(out_dir / "mask.png")

    monkeypatch.setattr(webcam, "_pick_upload_path", lambda: drawing_path)
    monkeypatch.setattr(webcam, "_image_to_annotations", fake_image_to_annotations)
    figure_state = _figure_state_for_webcam(webcam)
    args = SimpleNamespace(upload_output_dir=tmp_path / "uploads", torchserve_timeout=0.1, max_image_size=4096)

    with ThreadPoolExecutor(max_workers=1) as executor:
        upload_state = webcam.UploadState(executor=executor)
        webcam._start_upload_from_picker(upload_state, figure_state, args)
        assert upload_state.future is not None
        upload_state.future.result(timeout=5)
        webcam._poll_upload_job(upload_state, figure_state)

    assert figure_state.options[-1].name == "my-sketch"
    assert figure_state.options[-1].character_cfg.name == "char_cfg.yaml"
    assert figure_state.pending_index == len(figure_state.options) - 1
    assert upload_state.status_message == "Added my-sketch."


def test_webcam_choose_existing_character_accepts_config_and_folder(tmp_path: Path):
    webcam = _webcam_module()
    character_dir = _write_minimal_character_dir(tmp_path / "existing_figure")

    option_from_dir = webcam._figure_option_for_existing_character(character_dir)
    option_from_cfg = webcam._figure_option_for_existing_character(character_dir / "char_cfg.yaml")

    assert option_from_dir.character_cfg == character_dir / "char_cfg.yaml"
    assert option_from_cfg.character_cfg == character_dir / "char_cfg.yaml"
    assert option_from_dir.name == "existing_figure"


def test_webcam_choose_existing_character_picker_switches_to_selected(tmp_path: Path, monkeypatch):
    webcam = _webcam_module()
    character_dir = _write_minimal_character_dir(tmp_path / "picked_figure")
    figure_state = _figure_state_for_webcam(webcam)
    upload_state = webcam.UploadState(executor=ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(webcam, "_pick_existing_character_path", lambda: character_dir)

    try:
        webcam._add_existing_character_from_picker(upload_state, figure_state)
    finally:
        upload_state.executor.shutdown(wait=False, cancel_futures=True)

    assert figure_state.options[-1].name == "picked_figure"
    assert figure_state.pending_index == len(figure_state.options) - 1


def test_webcam_upload_rejects_invalid_or_unreadable_images(tmp_path: Path):
    webcam = _webcam_module()
    text_path = tmp_path / "drawing.txt"
    text_path.write_text("not an image", encoding="utf-8")
    bad_png = tmp_path / "bad.png"
    bad_png.write_bytes(b"not a png")

    with pytest.raises(webcam.UploadError, match="supported drawing"):
        webcam._validate_image_upload(text_path, 4096)
    with pytest.raises(webcam.UploadError, match="readable PNG"):
        webcam._validate_image_upload(bad_png, 4096)


def test_webcam_file_dialog_uses_osascript_on_macos(monkeypatch):
    webcam = _webcam_module()
    calls = []

    def fake_run(command, capture_output, text, timeout):
        calls.append(command)
        return SimpleNamespace(returncode=0, stdout="/tmp/drawing.png\n", stderr="")

    monkeypatch.setattr(webcam.sys, "platform", "darwin")
    monkeypatch.setattr(webcam.subprocess, "run", fake_run)

    selected, used_dialog = webcam._open_file_dialog("Choose file", [("All files", "*")])

    assert selected == Path("/tmp/drawing.png")
    assert used_dialog
    assert calls[0][0] == "osascript"
    assert "choose file" in calls[0][2]


def test_webcam_directory_dialog_handles_macos_cancel(monkeypatch):
    webcam = _webcam_module()

    def fake_run(command, capture_output, text, timeout):
        return SimpleNamespace(returncode=1, stdout="", stderr="User canceled.")

    monkeypatch.setattr(webcam.sys, "platform", "darwin")
    monkeypatch.setattr(webcam.subprocess, "run", fake_run)

    selected, used_dialog = webcam._open_directory_dialog("Choose folder")

    assert selected is None
    assert used_dialog


def test_webcam_to_animation_help_does_not_open_camera():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples/webcam_to_animation.py"), "--help"],
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "webcam" in result.stdout.lower()
    assert "--no-overlay" in result.stdout
    assert "--list-figures" in result.stdout


def test_webcam_to_animation_list_figures_does_not_open_camera():
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples/webcam_to_animation.py"), "--list-figures"],
        capture_output=True,
        cwd=REPO_ROOT,
        text=True,
        timeout=10,
    )

    assert result.returncode == 0
    assert "Available bundled figures" in result.stdout
    assert "1. char1" in result.stdout
    assert "6. char6" in result.stdout
    assert "Could not open webcam" not in result.stderr


def _pose_frame(timestamp: float, root_shift: float = 0.0) -> PoseFrame:
    landmarks = {
        "NOSE": [0.50 + root_shift, 0.18, 0.0, 1.0],
        "LEFT_SHOULDER": [0.38 + root_shift, 0.34, 0.0, 1.0],
        "RIGHT_SHOULDER": [0.62 + root_shift, 0.34, 0.0, 1.0],
        "LEFT_ELBOW": [0.30 + root_shift, 0.48, 0.0, 1.0],
        "RIGHT_ELBOW": [0.70 + root_shift, 0.48, 0.0, 1.0],
        "LEFT_WRIST": [0.26 + root_shift, 0.62, 0.0, 1.0],
        "RIGHT_WRIST": [0.74 + root_shift, 0.62, 0.0, 1.0],
        "LEFT_HIP": [0.43 + root_shift, 0.62, 0.0, 1.0],
        "RIGHT_HIP": [0.57 + root_shift, 0.62, 0.0, 1.0],
        "LEFT_KNEE": [0.42 + root_shift, 0.78, 0.0, 1.0],
        "RIGHT_KNEE": [0.58 + root_shift, 0.78, 0.0, 1.0],
        "LEFT_ANKLE": [0.41 + root_shift, 0.94, 0.0, 1.0],
        "RIGHT_ANKLE": [0.59 + root_shift, 0.94, 0.0, 1.0],
    }
    return PoseFrame(timestamp=timestamp, landmarks=landmarks)


def _webcam_module():
    module_name = "webcam_to_animation_under_test"
    module_path = REPO_ROOT / "examples/webcam_to_animation.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _figure_state_for_webcam(webcam):
    return webcam.FigureState(
        options=[webcam.FigureOption("char1", Path("char1/char_cfg.yaml"))],
        active_index=0,
        live_retargeter=SimpleNamespace(reset_root_reference=lambda: None),
    )


def _write_minimal_character_dir(character_dir: Path) -> Path:
    character_dir.mkdir()
    (character_dir / "char_cfg.yaml").write_text("height: 8\nwidth: 8\nskeleton: []\n", encoding="utf-8")
    Image.new("RGBA", (8, 8), "white").save(character_dir / "texture.png")
    Image.new("L", (8, 8), 255).save(character_dir / "mask.png")
    return character_dir
