"""Shared landmark definitions for the flow corrector."""

LANDMARK_NAMES = [
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

BVH_ALIASES = {
    "NOSE": ["Head", "HeadTop_End", "End Site"],
    "LEFT_SHOULDER": ["LeftShoulder", "LeftArm", "LeftCollar"],
    "RIGHT_SHOULDER": ["RightShoulder", "RightArm", "RightCollar"],
    "LEFT_ELBOW": ["LeftForeArm", "LeftElbow"],
    "RIGHT_ELBOW": ["RightForeArm", "RightElbow"],
    "LEFT_WRIST": ["LeftHand", "LeftWrist"],
    "RIGHT_WRIST": ["RightHand", "RightWrist"],
    "LEFT_HIP": ["LeftUpLeg", "LeftHip"],
    "RIGHT_HIP": ["RightUpLeg", "RightHip"],
    "LEFT_KNEE": ["LeftLeg", "LeftKnee"],
    "RIGHT_KNEE": ["RightLeg", "RightKnee"],
    "LEFT_ANKLE": ["LeftFoot", "LeftAnkle"],
    "RIGHT_ANKLE": ["RightFoot", "RightAnkle"],
}

CORRUPTION_TYPE_TO_ID = {
    "random": 1,
    "short_span": 2,
    "medium_span": 3,
    "long_span": 4,
    "whole_limb": 5,
    "spike": 6,
    "swap": 7,
    "high_conf_wrong": 8,
}

SPAN_BUCKET_TO_ID = {
    "none": 0,
    "short": 1,
    "medium": 2,
    "long": 3,
}
