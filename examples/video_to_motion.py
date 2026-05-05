#!/usr/bin/env python

"""Convert an uploaded or recorded video into an Animated Drawings motion config."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from animated_drawings.video_pose import build_motion_from_video
from animated_drawings.video_pose.constants import DEFAULT_MAX_SECONDS


def parse_args():
    parser = argparse.ArgumentParser(description="Estimate pose from a video and write BVH motion files.")
    parser.add_argument("input_video", help="Path to the input video.")
    parser.add_argument("out_dir", help="Directory where generated motion files will be written.")
    parser.add_argument(
        "--max-seconds",
        type=int,
        default=DEFAULT_MAX_SECONDS,
        help="Maximum accepted video duration. Defaults to 10 seconds.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    result = build_motion_from_video(
        Path(args.input_video),
        Path(args.out_dir),
        max_seconds=args.max_seconds,
    )
    print(f"Wrote pose JSON: {result.pose_sequence_path}")
    print(f"Wrote overlay video: {result.overlay_video_path}")
    print(f"Wrote BVH: {result.bvh_path}")
    print(f"Wrote motion config: {result.motion_config_path}")


if __name__ == "__main__":
    main()
