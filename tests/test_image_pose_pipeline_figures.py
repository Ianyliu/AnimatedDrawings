import importlib.util
from pathlib import Path
import sys

import numpy as np

from animated_drawings.video_pose import PoseFrame


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_draw_pose_figure_writes_overlay_and_direct_panels():
    module = _image_pose_figures_module()
    image = np.full((120, 80, 3), 255, dtype=np.uint8)
    frame = PoseFrame(
        timestamp=0.0,
        landmarks={
            "NOSE": [0.50, 0.12, 0.0, 0.95],
            "LEFT_SHOULDER": [0.35, 0.30, 0.0, 0.90],
            "RIGHT_SHOULDER": [0.65, 0.30, 0.0, 0.90],
            "LEFT_ELBOW": [0.22, 0.48, 0.0, 0.70],
            "RIGHT_ELBOW": [0.78, 0.48, 0.0, 0.70],
            "LEFT_WRIST": [0.14, 0.67, 0.0, 0.40],
            "RIGHT_WRIST": [0.86, 0.67, 0.0, 0.40],
            "LEFT_HIP": [0.42, 0.58, 0.0, 0.95],
            "RIGHT_HIP": [0.58, 0.58, 0.0, 0.95],
            "LEFT_KNEE": [0.40, 0.78, 0.0, 0.95],
            "RIGHT_KNEE": [0.60, 0.78, 0.0, 0.95],
            "LEFT_ANKLE": [0.38, 0.96, 0.0, 0.95],
            "RIGHT_ANKLE": [0.62, 0.96, 0.0, 0.95],
        },
    )

    overlay = module.draw_pose_figure(image, frame, mode="overlay", figure_size=256)
    direct = module.draw_pose_figure(image, frame, mode="direct", figure_size=256)

    assert overlay.shape == (256, 256, 3)
    assert direct.shape == (256, 256, 3)
    assert int(np.abs(overlay.astype(np.int16) - 255).sum()) > 0
    assert int(direct.sum()) > 0


def _image_pose_figures_module():
    module_name = "image_pose_pipeline_figures_under_test"
    module_path = REPO_ROOT / "examples/image_pose_pipeline_figures.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
