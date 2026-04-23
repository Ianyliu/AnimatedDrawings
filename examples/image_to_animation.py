# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from pathlib import Path
import logging
from pkg_resources import resource_filename


def image_to_animation(img_fn: str, char_anno_dir: str, motion_cfg_fn: str, retarget_cfg_fn: str):
    """
    Given the image located at img_fn, create annotation files needed for animation.
    Then create animation from those animations and motion cfg and retarget cfg.
    """
    from image_to_annotations import image_to_annotations
    from annotations_to_animation import annotations_to_animation

    # create the annotations
    image_to_annotations(img_fn, char_anno_dir)

    # create the animation
    annotations_to_animation(char_anno_dir, motion_cfg_fn, retarget_cfg_fn)


def parse_args():
    """Parse CLI arguments for the image-to-animation example pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate character annotations from an input image and then render "
            "an animation using the selected motion and retarget configs."
        )
    )
    parser.add_argument(
        "img_fn",
        help="Path to the input character image.",
    )
    parser.add_argument(
        "char_anno_dir",
        help="Output directory where generated annotation files will be written.",
    )
    parser.add_argument(
        "motion_cfg_fn",
        nargs="?",
        default=resource_filename(__name__, 'config/motion/dab.yaml'),
        help="Optional motion config YAML. Defaults to config/motion/dab.yaml.",
    )
    parser.add_argument(
        "retarget_cfg_fn",
        nargs="?",
        default=resource_filename(__name__, 'config/retarget/fair1_ppf.yaml'),
        help=(
            "Optional retarget config YAML. Defaults to "
            "config/retarget/fair1_ppf.yaml."
        ),
    )
    return parser.parse_args()


def main():
    """Run the example pipeline from CLI arguments and default config fallbacks."""
    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.DEBUG)

    args = parse_args()

    image_to_animation(
        args.img_fn,
        args.char_anno_dir,
        args.motion_cfg_fn,
        args.retarget_cfg_fn,
    )


if __name__ == '__main__':
    main()
