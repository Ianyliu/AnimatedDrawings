#!/usr/bin/env python

"""Drive an Animated Drawing from live webcam pose without writing BVH files."""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace
import time
from typing import Optional

import cv2
import glfw
import numpy as np
import numpy.typing as npt
from OpenGL import GL
from PIL import Image, UnidentifiedImageError

from animated_drawings.config import CharacterConfig, RetargetConfig, ViewConfig
from animated_drawings.model.animated_drawing import AnimatedDrawing
from animated_drawings.model.scene import Scene
from animated_drawings.utils import resolve_ad_filepath
from animated_drawings.video_pose.live import (
    CausalPoseSmoother,
    LiveMediaPipePoseEstimator,
    LivePoseRetargeter,
    PoseTrackingStatus,
    analyze_pose_frame,
    camera_error_status,
    compose_live_dashboard,
    draw_pose_overlay,
    live_dashboard_upload_rect,
    paused_status,
)
from animated_drawings.view.window_view import WindowView


EXAMPLES_DIR = Path(__file__).resolve().parent
CHARACTERS_DIR = EXAMPLES_DIR / "characters"
DEFAULT_CHARACTER = EXAMPLES_DIR / "characters/char1/char_cfg.yaml"
DEFAULT_RETARGET = EXAMPLES_DIR / "config/retarget/mediapipe_pfp.yaml"
DEFAULT_UPLOAD_OUTPUT_DIR = EXAMPLES_DIR / "webcam_uploads"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@dataclass(frozen=True)
class FigureOption:
    name: str
    character_cfg: Path
    bundled: bool = True


@dataclass
class RunState:
    paused: bool = False
    quit: bool = False
    upload_requested: bool = False
    choose_character_requested: bool = False


@dataclass
class UploadState:
    executor: ThreadPoolExecutor
    future: Optional[Future[FigureOption]] = None
    status_message: Optional[str] = None

    @property
    def busy(self) -> bool:
        return self.future is not None and not self.future.done()


@dataclass
class FigureState:
    options: list[FigureOption]
    active_index: int
    live_retargeter: LivePoseRetargeter
    pending_index: Optional[int] = None

    @property
    def active_option(self) -> FigureOption:
        return self.options[self.active_index]


class UploadError(ValueError):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Open a webcam and drive a bundled Animated Drawing directly from live MediaPipe pose."
    )
    parser.add_argument("--character", default=str(DEFAULT_CHARACTER), help="Character char_cfg.yaml to animate.")
    parser.add_argument("--list-figures", action="store_true", help="List bundled figures and exit.")
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
    parser.add_argument(
        "--upload-output-dir",
        default=str(DEFAULT_UPLOAD_OUTPUT_DIR),
        help="Directory for generated webcam drawing uploads.",
    )
    parser.add_argument(
        "--torchserve-timeout",
        type=float,
        default=30.0,
        help="Seconds to wait for drawing analysis service calls.",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=4096,
        help="Maximum width or height for uploaded drawing images.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.list_figures:
        _print_figure_options(_discover_bundled_figures())
        return

    figure_options, active_index = _figure_options_for_character(args.character)
    character, live_retargeter = _build_live_character(figure_options[active_index], args)
    figure_state = FigureState(
        options=figure_options,
        active_index=active_index,
        live_retargeter=live_retargeter,
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
        _print_startup_controls(figure_state)

        with LiveMediaPipePoseEstimator(model_complexity=args.model_complexity) as estimator:
            _run_loop(
                cap,
                scene,
                view,
                estimator,
                smoother,
                figure_state,
                state,
                args,
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
    figure_state: FigureState,
    state: RunState,
    args,
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
    upload_state = UploadState(executor=ThreadPoolExecutor(max_workers=1, thread_name_prefix="webcam-upload"))
    cv2.setMouseCallback(window_name, _handle_dashboard_mouse, {"state": state, "pane_size": pane_size})

    try:
        while not state.quit and not glfw.window_should_close(view.win):
            _poll_upload_job(upload_state, figure_state)
            live_retargeter = figure_state.live_retargeter
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
                active_figure=figure_state.active_option.name,
                controls=_dashboard_controls_text(len(figure_state.options)),
                upload_status=upload_state.status_message,
                figures=[option.name for option in figure_state.options],
                active_figure_index=figure_state.active_index,
            )
            cv2.imshow(window_name, dashboard)
            _handle_dashboard_key(cv2.waitKey(1), state, smoother, live_retargeter, figure_state)
            picker_opened = _handle_upload_requests(state, upload_state, figure_state, args)
            if picker_opened:
                prev_time = time.time()
            _apply_pending_figure_switch(scene, figure_state, smoother, args)
            glfw.poll_events()
    finally:
        upload_state.executor.shutdown(wait=False, cancel_futures=True)

    cv2.destroyWindow(window_name)


def _handle_dashboard_key(
    key: int,
    state: RunState,
    smoother: CausalPoseSmoother,
    live_retargeter: LivePoseRetargeter,
    figure_state: Optional[FigureState] = None,
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
    elif key in (ord("u"), ord("U")):
        state.upload_requested = True
    elif key in (ord("c"), ord("C")):
        state.choose_character_requested = True
    elif figure_state is not None:
        switch_index = _figure_switch_index_for_key(key, figure_state.active_index, len(figure_state.options))
        if switch_index is not None:
            figure_state.pending_index = switch_index


def _handle_dashboard_mouse(event: int, x: int, y: int, flags: int, userdata) -> None:
    del flags
    if event != cv2.EVENT_LBUTTONUP or not isinstance(userdata, dict):
        return
    state = userdata.get("state")
    pane_size = int(userdata.get("pane_size") or 0)
    if not isinstance(state, RunState) or pane_size <= 0:
        return
    if _point_in_rect((x, y), live_dashboard_upload_rect(pane_size)):
        state.upload_requested = True


def _handle_upload_requests(
    state: RunState,
    upload_state: UploadState,
    figure_state: FigureState,
    args,
) -> bool:
    picker_opened = False
    if state.upload_requested:
        state.upload_requested = False
        picker_opened = True
        _start_upload_from_picker(upload_state, figure_state, args)
    if state.choose_character_requested:
        state.choose_character_requested = False
        picker_opened = True
        _add_existing_character_from_picker(upload_state, figure_state)
    return picker_opened


def _start_upload_from_picker(upload_state: UploadState, figure_state: FigureState, args) -> None:
    if upload_state.busy:
        upload_state.status_message = "Still analyzing the current drawing."
        return

    selected_path = _pick_upload_path()
    if selected_path is None:
        upload_state.status_message = None
        return
    _start_upload_for_path(selected_path, upload_state, figure_state, args)


def _add_existing_character_from_picker(upload_state: UploadState, figure_state: FigureState) -> None:
    selected_path = _pick_existing_character_path()
    if selected_path is None:
        upload_state.status_message = None
        return

    try:
        option = _figure_option_for_existing_character(selected_path)
    except Exception as e:
        upload_state.status_message = _friendly_upload_error(e)
        print(upload_state.status_message)
        return

    index = _append_figure_option(figure_state, option)
    figure_state.pending_index = index
    upload_state.status_message = f"Added {option.name}."
    print(f"Added custom figure {_figure_short_label(index, figure_state.options[index])}.")


def _start_upload_for_path(
    selected_path: Path,
    upload_state: UploadState,
    figure_state: FigureState,
    args,
) -> None:
    if upload_state.busy:
        upload_state.status_message = "Still analyzing the current drawing."
        return

    selected_path = Path(selected_path).expanduser()
    if selected_path.is_dir() or selected_path.name == "char_cfg.yaml":
        try:
            option = _figure_option_for_existing_character(selected_path)
        except Exception as e:
            upload_state.status_message = _friendly_upload_error(e)
            print(upload_state.status_message)
            return
        index = _append_figure_option(figure_state, option)
        figure_state.pending_index = index
        upload_state.status_message = f"Added {option.name}."
        print(f"Added custom figure {_figure_short_label(index, figure_state.options[index])}.")
        return

    try:
        _validate_image_upload(selected_path, int(args.max_image_size))
    except Exception as e:
        upload_state.status_message = _friendly_upload_error(e)
        print(upload_state.status_message)
        return

    upload_state.status_message = f"Analyzing {selected_path.name}..."
    upload_state.future = upload_state.executor.submit(_rig_uploaded_drawing, selected_path.resolve(), args)
    print(f"Analyzing uploaded drawing {selected_path}.")


def _poll_upload_job(upload_state: UploadState, figure_state: FigureState) -> None:
    future = upload_state.future
    if future is None or not future.done():
        return

    upload_state.future = None
    try:
        option = future.result()
    except Exception as e:
        upload_state.status_message = _friendly_upload_error(e)
        print(upload_state.status_message)
        return

    index = _append_figure_option(figure_state, option)
    figure_state.pending_index = index
    upload_state.status_message = f"Added {option.name}."
    print(f"Added uploaded figure {_figure_short_label(index, figure_state.options[index])}.")


def _rig_uploaded_drawing(image_path: Path, args) -> FigureOption:
    output_root = Path(args.upload_output_dir).expanduser().resolve()
    output_root.mkdir(exist_ok=True, parents=True)
    slug = _safe_slug(image_path.stem) or "drawing"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_dir = output_root / f"{timestamp}_{slug}"
    character_dir = job_dir / "character"
    job_dir.mkdir(exist_ok=False, parents=True)

    source_path = job_dir / f"source{image_path.suffix.lower()}"
    shutil.copy2(image_path, source_path)
    _image_to_annotations(source_path, character_dir, timeout=float(args.torchserve_timeout))
    char_cfg_path = _resolve_existing_character_cfg(character_dir)
    return FigureOption(name=slug, character_cfg=char_cfg_path, bundled=False)


def _image_to_annotations(image_path: Path, out_dir: Path, timeout: float) -> None:
    if str(EXAMPLES_DIR) not in sys.path:
        sys.path.insert(0, str(EXAMPLES_DIR))
    from image_to_annotations import image_to_annotations

    image_to_annotations(str(image_path), str(out_dir), request_timeout=timeout)


def _figure_option_for_existing_character(source_path: Path) -> FigureOption:
    char_cfg_path = _resolve_existing_character_cfg(source_path)
    return FigureOption(name=char_cfg_path.parent.name, character_cfg=char_cfg_path, bundled=False)


def _resolve_existing_character_cfg(source_path: Path) -> Path:
    path = Path(source_path).expanduser()
    if path.is_dir():
        path = path / "char_cfg.yaml"
    if path.name != "char_cfg.yaml":
        raise UploadError("Choose a char_cfg.yaml file or a character folder.")
    path = path.resolve()
    if not path.exists():
        raise UploadError("Could not find char_cfg.yaml.")
    if not (path.parent / "texture.png").exists() or not (path.parent / "mask.png").exists():
        raise UploadError("Character folder must contain texture.png and mask.png.")
    return path


def _validate_image_upload(image_path: Path, max_image_size: int) -> None:
    if not image_path.exists() or not image_path.is_file():
        raise UploadError("Choose an existing drawing image.")
    suffix = image_path.suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        accepted = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise UploadError(f"Upload a supported drawing file ({accepted}) or choose char_cfg.yaml.")
    try:
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format
            image.verify()
    except (OSError, UnidentifiedImageError) as e:
        raise UploadError("Upload a readable PNG, JPEG, or WebP drawing.") from e

    expected_formats = {
        ".png": "PNG",
        ".jpg": "JPEG",
        ".jpeg": "JPEG",
        ".webp": "WEBP",
    }
    if image_format != expected_formats.get(suffix):
        raise UploadError(f"Drawing file content must match the {suffix} extension.")
    if width > max_image_size or height > max_image_size:
        raise UploadError(f"Drawing image must be at most {max_image_size}x{max_image_size}.")


def _append_figure_option(figure_state: FigureState, option: FigureOption) -> int:
    character_cfg = option.character_cfg.resolve()
    for idx, existing in enumerate(figure_state.options):
        if existing.character_cfg.resolve() == character_cfg:
            return idx
    figure_state.options.append(option)
    return len(figure_state.options) - 1


def _pick_upload_path() -> Optional[Path]:
    selected, used_dialog = _open_file_dialog(
        title="Choose a drawing or char_cfg.yaml",
        filetypes=[
            ("Drawings and character configs", "*.png *.jpg *.jpeg *.webp *.yaml"),
            ("Drawing images", "*.png *.jpg *.jpeg *.webp"),
            ("Character config", "char_cfg.yaml"),
            ("All files", "*"),
        ],
    )
    if selected is not None or used_dialog:
        return selected
    return _prompt_for_path("Path to drawing image or char_cfg.yaml (blank to cancel): ")


def _pick_existing_character_path() -> Optional[Path]:
    selected, used_dialog = _open_file_dialog(
        title="Choose an existing char_cfg.yaml",
        filetypes=[("Character config", "char_cfg.yaml"), ("YAML", "*.yaml"), ("All files", "*")],
    )
    if selected is not None:
        return selected
    directory, used_directory_dialog = _open_directory_dialog(title="Choose a character folder")
    if directory is not None or used_dialog or used_directory_dialog:
        return directory
    return _prompt_for_path("Path to char_cfg.yaml or character folder (blank to cancel): ")


def _open_file_dialog(title: str, filetypes: list[tuple[str, str]]) -> tuple[Optional[Path], bool]:
    if sys.platform == "darwin":
        return _open_macos_path_dialog(title, choose_folder=False)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None, False

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
        return Path(selected) if selected else None, True
    except Exception:
        return None, False
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _open_directory_dialog(title: str) -> tuple[Optional[Path], bool]:
    if sys.platform == "darwin":
        return _open_macos_path_dialog(title, choose_folder=True)

    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None, False

    root = None
    try:
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        return Path(selected) if selected else None, True
    except Exception:
        return None, False
    finally:
        if root is not None:
            try:
                root.destroy()
            except Exception:
                pass


def _open_macos_path_dialog(title: str, *, choose_folder: bool) -> tuple[Optional[Path], bool]:
    escaped_title = _applescript_string(title)
    command = "choose folder" if choose_folder else "choose file"
    script = f'POSIX path of ({command} with prompt "{escaped_title}")'
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, False

    if result.returncode != 0:
        if "User canceled" in result.stderr:
            return None, True
        return None, False

    selected = result.stdout.strip()
    return (Path(selected) if selected else None), True


def _applescript_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _prompt_for_path(prompt: str) -> Optional[Path]:
    try:
        raw = input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None
    return Path(raw).expanduser() if raw else None


def _friendly_upload_error(error: Exception) -> str:
    if isinstance(error, UploadError):
        return str(error)

    message = str(error)
    lowered = message.lower()
    if "torchserve" in lowered or "failed to get" in lowered or "connection" in lowered:
        return "Drawing analysis needs TorchServe. Start it and try again."
    if "humanoid" in lowered:
        return "Could not detect a single drawn humanoid in the image."
    if "skeleton" in lowered:
        return "Could not detect a usable character skeleton in the drawing."
    if "timeout" in lowered:
        return "Drawing analysis timed out. Try a smaller image or try again."
    return "Could not add drawing. Check the file and try again."


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_-]+", "-", value.strip().lower())
    return slug.strip("-_")[:48]


def _point_in_rect(point: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    x, y = point
    left, top, right, bottom = rect
    return left <= x <= right and top <= y <= bottom


def _discover_bundled_figures() -> list[FigureOption]:
    return [
        FigureOption(name=char_cfg.parent.name, character_cfg=char_cfg.resolve(), bundled=True)
        for char_cfg in sorted(CHARACTERS_DIR.glob("*/char_cfg.yaml"))
        if re.fullmatch(r"char\d+", char_cfg.parent.name)
    ]


def _figure_options_for_character(character_arg: str) -> tuple[list[FigureOption], int]:
    bundled = _discover_bundled_figures()
    selected_path = resolve_ad_filepath(character_arg, "character cfg")

    for idx, option in enumerate(bundled):
        if option.character_cfg == selected_path:
            return bundled, idx

    custom = FigureOption(name=selected_path.parent.name, character_cfg=selected_path, bundled=False)
    return [custom, *bundled], 0


def _build_live_character(option: FigureOption, args) -> tuple[AnimatedDrawing, LivePoseRetargeter]:
    retarget_cfg = RetargetConfig(args.retarget)
    live_retargeter = LivePoseRetargeter(
        retarget_cfg,
        root_mode=args.root_mode,
        depth_mode=args.depth_mode,
    )
    character = AnimatedDrawing(
        CharacterConfig(str(option.character_cfg)),
        retarget_cfg,
        motion_cfg=None,
        retargeter=live_retargeter,
    )
    return character, live_retargeter


def _apply_pending_figure_switch(
    scene: Scene,
    figure_state: FigureState,
    smoother: CausalPoseSmoother,
    args,
) -> None:
    pending_index = figure_state.pending_index
    figure_state.pending_index = None
    if pending_index is None or pending_index == figure_state.active_index:
        return

    option = figure_state.options[pending_index]
    try:
        character, live_retargeter = _build_live_character(option, args)
    except Exception as e:
        print(f"Could not switch to figure {option.name}: {e}")
        return

    _replace_scene_character(scene, character)
    figure_state.active_index = pending_index
    figure_state.live_retargeter = live_retargeter
    smoother.reset()
    live_retargeter.reset_root_reference()
    print(f"Switched figure to {_figure_short_label(pending_index, option)}.")


def _replace_scene_character(scene: Scene, character: AnimatedDrawing) -> None:
    children = scene.get_children()
    children[:] = [character]
    character.set_parent(scene)
    scene.dirty_bit = True


def _figure_switch_index_for_key(key: int, active_index: int, option_count: int) -> Optional[int]:
    if key < 0 or option_count <= 0:
        return None

    key = key & 0xFF
    if ord("1") <= key <= ord("9"):
        requested_index = key - ord("1")
        return requested_index if requested_index < option_count else None
    if key == ord("["):
        return (active_index - 1) % option_count
    if key == ord("]"):
        return (active_index + 1) % option_count
    return None


def _dashboard_controls_text(figure_count: int) -> str:
    max_shortcut = min(figure_count, 9)
    figure_keys = f"1-{max_shortcut} select" if max_shortcut > 1 else "1 select"
    return f"Space pause   R reset   U upload   C choose rig   [/] figure   {figure_keys}   Q/Esc quit"


def _print_figure_options(options: list[FigureOption]) -> None:
    print("Available bundled figures:")
    for idx, option in enumerate(options):
        print(f"  {_figure_short_label(idx, option)}  {_repo_relative_path(option.character_cfg)}")


def _print_startup_controls(figure_state: FigureState) -> None:
    print(f"Running live webcam dashboard with figure {_figure_short_label(figure_state.active_index, figure_state.active_option)}.")
    print("Figures:")
    for idx, option in enumerate(figure_state.options):
        marker = "*" if idx == figure_state.active_index else " "
        print(f" {marker} {_figure_short_label(idx, option)}")
    print(
        "Controls: Space pause/resume, R reset pose, U upload drawing, "
        "C choose character rig, [/] switch figure, 1-9 select figure, Q/Esc quit."
    )


def _figure_short_label(index: int, option: FigureOption) -> str:
    return f"{index + 1}. {option.name}"


def _repo_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(EXAMPLES_DIR.parent))
    except ValueError:
        return str(path)


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
