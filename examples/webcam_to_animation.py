#!/usr/bin/env python

"""Drive an Animated Drawing from live webcam pose without writing BVH files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import time

import cv2
import glfw
import numpy as np
import numpy.typing as npt
from OpenGL import GL

from animated_drawings.config import CharacterConfig, RetargetConfig, ViewConfig
from animated_drawings.model.animated_drawing import AnimatedDrawing
from animated_drawings.model.scene import Scene
from animated_drawings.video_pose.live import (
    CausalPoseSmoother,
    LiveMediaPipePoseEstimator,
    LivePoseRetargeter,
    PoseTrackingStatus,
    analyze_pose_frame,
    camera_error_status,
    compose_live_dashboard,
    draw_pose_overlay,
    paused_status,
)
from animated_drawings.view.window_view import WindowView


EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_CHARACTER = EXAMPLES_DIR / "characters/char1/char_cfg.yaml"
DEFAULT_RETARGET = EXAMPLES_DIR / "config/retarget/mediapipe_pfp.yaml"


@dataclass
class RunState:
    paused: bool = False
    quit: bool = False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a webcam and drive a bundled Animated Drawing directly from live MediaPipe pose."
    )
    parser.add_argument("--character", default=str(DEFAULT_CHARACTER), help="Character char_cfg.yaml to animate.")
    parser.add_argument("--retarget", default=str(DEFAULT_RETARGET), help="MediaPipe-compatible retarget YAML.")
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index. Defaults to 0.")
    parser.add_argument("--mirror", dest="mirror", action="store_true", default=True, help="Mirror webcam input.")
    parser.add_argument("--no-mirror", dest="mirror", action="store_false", help="Do not mirror webcam input.")
    parser.add_argument("--root-mode", choices=("locked", "hip"), default="locked", help="Character root behavior.")
    parser.add_argument(
        "--depth-mode",
        choices=("flat", "mediapipe-z"),
        default="flat",
        help="Layer ordering depth source.",
    )
    parser.add_argument("--window-size", type=int, default=500, help="Square renderer window size.")
    parser.add_argument("--draw-rig", action="store_true", help="Show the character rig overlay.")
    parser.add_argument("--no-overlay", action="store_true", help="Hide the webcam pose overlay.")
    parser.add_argument("--model-complexity", type=int, choices=(0, 1, 2), default=1, help="MediaPipe model complexity.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retarget_cfg = RetargetConfig(args.retarget)
    live_retargeter = LivePoseRetargeter(
        retarget_cfg,
        root_mode=args.root_mode,
        depth_mode=args.depth_mode,
    )
    character = AnimatedDrawing(
        CharacterConfig(args.character),
        retarget_cfg,
        motion_cfg=None,
        retargeter=live_retargeter,
    )
    scene = Scene(SimpleNamespace(add_floor=False, add_ad_retarget_bvh=False, animated_characters=[]))
    scene.add_child(character)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open webcam index {args.camera}.")

    view = None
    try:
        view = _create_hidden_view(args)
        state = RunState()
        smoother = CausalPoseSmoother()
        print("Running live webcam dashboard. Press Space to pause, R to reset root, Q/Esc to quit.")

        with LiveMediaPipePoseEstimator(model_complexity=args.model_complexity) as estimator:
            _run_loop(
                cap,
                scene,
                view,
                estimator,
                smoother,
                live_retargeter,
                state,
                mirror=args.mirror,
                pane_size=args.window_size,
                show_overlay=not args.no_overlay,
            )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if view is not None:
            view.cleanup()


def _run_loop(
    cap,
    scene: Scene,
    view: WindowView,
    estimator: LiveMediaPipePoseEstimator,
    smoother: CausalPoseSmoother,
    live_retargeter: LivePoseRetargeter,
    state: RunState,
    *,
    mirror: bool,
    pane_size: int,
    show_overlay: bool,
) -> None:
    start_time = time.time()
    prev_time = start_time
    last_pose_frame = None
    last_camera_frame = np.full((pane_size, pane_size, 3), 228, dtype=np.uint8)
    status = PoseTrackingStatus(state="lost", message="Waiting for webcam frame.")
    window_name = "Animated Drawings Live Webcam"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    while not state.quit and not glfw.window_should_close(view.win):
        now = time.time()
        delta_t = now - prev_time
        prev_time = now

        if not state.paused:
            ok, frame = cap.read()
            if ok:
                if mirror:
                    frame = cv2.flip(frame, 1)
                last_camera_frame = frame
                pose_frame = estimator.estimate_frame(frame, timestamp=now - start_time)
                last_pose_frame = pose_frame
                status = analyze_pose_frame(pose_frame)
                smoothed = smoother.process(pose_frame)
                live_retargeter.update_pose(smoothed)
            else:
                status = camera_error_status()
        else:
            status = paused_status()

        scene.progress_time(delta_t)
        scene.update_transforms()
        animation_frame = _render_animation_frame(view, scene)
        camera_frame = last_camera_frame
        if show_overlay and last_pose_frame is not None:
            camera_frame = draw_pose_overlay(last_camera_frame, last_pose_frame, status)
        dashboard = compose_live_dashboard(
            camera_frame,
            animation_frame,
            status,
            pane_size=pane_size,
            paused=state.paused,
        )
        cv2.imshow(window_name, dashboard)
        _handle_dashboard_key(cv2.waitKey(1), state, smoother, live_retargeter)
        glfw.poll_events()

    cv2.destroyWindow(window_name)


def _handle_dashboard_key(
    key: int,
    state: RunState,
    smoother: CausalPoseSmoother,
    live_retargeter: LivePoseRetargeter,
) -> None:
    if key < 0:
        return
    key = key & 0xFF
    if key in (27, ord("q"), ord("Q")):
        state.quit = True
    elif key == ord(" "):
        state.paused = not state.paused
    elif key in (ord("r"), ord("R")):
        live_retargeter.reset_root_reference()
        smoother.reset()


def _create_hidden_view(args) -> WindowView:
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    view = WindowView(_view_config(args))
    glfw.window_hint(glfw.VISIBLE, glfw.TRUE)
    glfw.set_window_title(view.win, "Animated Drawings Hidden Renderer")
    return view


def _render_animation_frame(view: WindowView, scene: Scene) -> npt.NDArray[np.uint8]:
    width, height = view.get_framebuffer_size()
    view.clear_window()
    view.render(scene)
    GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
    frame = np.empty((height, width, 4), dtype=np.uint8)
    GL.glReadPixels(0, 0, width, height, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, frame)
    view.swap_buffers()
    return frame[::-1, :, :3].copy()


def _view_config(args) -> ViewConfig:
    return ViewConfig(
        {
            "CLEAR_COLOR": [1.0, 1.0, 1.0, 0.0],
            "BACKGROUND_IMAGE": None,
            "WINDOW_DIMENSIONS": [args.window_size, args.window_size],
            "DRAW_AD_RIG": bool(args.draw_rig),
            "DRAW_AD_TXTR": True,
            "DRAW_AD_COLOR": False,
            "DRAW_AD_MESH_LINES": False,
            "USE_MESA": False,
            "CAMERA_POS": [0.0, 0.7, 2.0],
            "CAMERA_FWD": [0.0, 0.5, 2.0],
        }
    )


if __name__ == "__main__":
    main()
