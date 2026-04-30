from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

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

    assert dashboard.shape == (64 + 240 + 48 + 16, 16 * 2 + 240 * 2 + 16, 3)


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
