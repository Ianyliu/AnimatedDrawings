"""Constants shared by the video pose pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


DEFAULT_MAX_SECONDS = 10

MEDIAPIPE_REQUIRED_LANDMARKS = [
    "NOSE",
    "LEFT_SHOULDER",
    "RIGHT_SHOULDER",
    "LEFT_ELBOW",
    "RIGHT_ELBOW",
    "LEFT_WRIST",
    "RIGHT_WRIST",
    "LEFT_HIP",
    "RIGHT_HIP",
    "LEFT_KNEE",
    "RIGHT_KNEE",
    "LEFT_ANKLE",
    "RIGHT_ANKLE",
]

POSE_CONNECTIONS = [
    ("LEFT_SHOULDER", "RIGHT_SHOULDER"),
    ("LEFT_SHOULDER", "LEFT_ELBOW"),
    ("LEFT_ELBOW", "LEFT_WRIST"),
    ("RIGHT_SHOULDER", "RIGHT_ELBOW"),
    ("RIGHT_ELBOW", "RIGHT_WRIST"),
    ("LEFT_SHOULDER", "LEFT_HIP"),
    ("RIGHT_SHOULDER", "RIGHT_HIP"),
    ("LEFT_HIP", "RIGHT_HIP"),
    ("LEFT_HIP", "LEFT_KNEE"),
    ("LEFT_KNEE", "LEFT_ANKLE"),
    ("RIGHT_HIP", "RIGHT_KNEE"),
    ("RIGHT_KNEE", "RIGHT_ANKLE"),
    ("NOSE", "LEFT_SHOULDER"),
    ("NOSE", "RIGHT_SHOULDER"),
]


@dataclass(frozen=True)
class BvhJointSpec:
    name: str
    offset: Tuple[float, float, float]
    children: Tuple["BvhJointSpec", ...] = ()
    end_offset: Optional[Tuple[float, float, float]] = None


BVH_ROOT = BvhJointSpec(
    "Hip",
    (0.0, 0.0, 0.0),
    (
        BvhJointSpec(
            "RightHip",
            (-106.99298858642578, 0.0, 0.0),
            (
                BvhJointSpec(
                    "RightKnee",
                    (0.0, 0.0, -388.80108642578125),
                    (
                        BvhJointSpec(
                            "RightAnkle",
                            (0.0, 0.0, -365.5960693359375),
                            end_offset=(0.0, -146.23843383789062, 0.0),
                        ),
                    ),
                ),
            ),
        ),
        BvhJointSpec(
            "LeftHip",
            (106.99298858642578, 0.0, 0.0),
            (
                BvhJointSpec(
                    "LeftKnee",
                    (0.0, 0.0, -388.80108642578125),
                    (
                        BvhJointSpec(
                            "LeftAnkle",
                            (0.0, 0.0, -365.5960693359375),
                            end_offset=(0.0, -146.23843383789062, 0.0),
                        ),
                    ),
                ),
            ),
        ),
        BvhJointSpec(
            "Spine",
            (0.0, 0.0, 233.8563690185547),
            (
                BvhJointSpec(
                    "Thorax",
                    (0.0, 0.0, 233.8563690185547),
                    (
                        BvhJointSpec(
                            "Neck",
                            (0.0, 0.0, 57.029170989990234),
                            end_offset=(0.0, 0.0, 22.81166648864746),
                        ),
                        BvhJointSpec(
                            "LeftShoulder",
                            (160.2506561279297, 0.0, 0.0),
                            (
                                BvhJointSpec(
                                    "LeftElbow",
                                    (232.46737670898438, 0.0, 0.0),
                                    (
                                        BvhJointSpec(
                                            "LeftWrist",
                                            (223.72117614746094, 0.0, 0.0),
                                            end_offset=(89.48846435546875, 0.0, 0.0),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                        BvhJointSpec(
                            "RightShoulder",
                            (-160.2506561279297, 0.0, 0.0),
                            (
                                BvhJointSpec(
                                    "RightElbow",
                                    (-232.46737670898438, 0.0, 0.0),
                                    (
                                        BvhJointSpec(
                                            "RightWrist",
                                            (-223.72117614746094, 0.0, 0.0),
                                            end_offset=(-89.48846435546875, 0.0, 0.0),
                                        ),
                                    ),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    ),
)

BVH_ROTATION_JOINTS = [
    "Hip",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "Spine",
    "Thorax",
    "Neck",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
]

BVH_CONTROL_BONES: Dict[str, Tuple[str, str, Tuple[float, float, float], Optional[str]]] = {
    "RightHip": ("RightHip", "RightKnee", (0.0, 0.0, -388.80108642578125), None),
    "RightKnee": ("RightKnee", "RightAnkle", (0.0, 0.0, -365.5960693359375), "RightHip"),
    "LeftHip": ("LeftHip", "LeftKnee", (0.0, 0.0, -388.80108642578125), None),
    "LeftKnee": ("LeftKnee", "LeftAnkle", (0.0, 0.0, -365.5960693359375), "LeftHip"),
    "Spine": ("Spine", "Thorax", (0.0, 0.0, 233.8563690185547), None),
    "Thorax": ("Thorax", "Neck", (0.0, 0.0, 57.029170989990234), "Spine"),
    "LeftShoulder": ("LeftShoulder", "LeftElbow", (232.46737670898438, 0.0, 0.0), "Thorax"),
    "LeftElbow": ("LeftElbow", "LeftWrist", (223.72117614746094, 0.0, 0.0), "LeftShoulder"),
    "RightShoulder": ("RightShoulder", "RightElbow", (-232.46737670898438, 0.0, 0.0), "Thorax"),
    "RightElbow": ("RightElbow", "RightWrist", (-223.72117614746094, 0.0, 0.0), "RightShoulder"),
}

MEDIAPIPE_MOTION_CFG = {
    "start_frame_idx": 0,
    "end_frame_idx": None,
    "groundplane_joint": "LeftAnkle",
    "forward_perp_joint_vectors": [
        ["LeftShoulder", "RightShoulder"],
        ["LeftHip", "RightHip"],
    ],
    "scale": 0.001,
    "up": "+z",
}
