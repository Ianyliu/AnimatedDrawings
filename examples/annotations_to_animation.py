# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import animated_drawings.render
import logging
from pathlib import Path
import yaml


EXAMPLES_DIR = Path(__file__).resolve().parent
DEFAULT_MOTION_CFG = EXAMPLES_DIR / 'config/motion/dab.yaml'
DEFAULT_RETARGET_CFG = EXAMPLES_DIR / 'config/retarget/fair1_ppf.yaml'
DEFAULT_RETARGET_CFG_BY_BVH = {
    'jumping_jacks.bvh': EXAMPLES_DIR / 'config/retarget/cmu1_pfp.yaml',
    'walk-cycle.bvh': EXAMPLES_DIR / 'config/retarget/walk_cycle_pfp.yaml',
}


def annotations_to_animation(char_anno_dir: str, motion_cfg_fn: str, retarget_cfg_fn: str):
    """
    Given a path to a directory with character annotations, a motion configuration file, and a retarget configuration file,
    creates an animation and saves it to {annotation_dir}/video.png
    """

    # package character_cfg_fn, motion_cfg_fn, and retarget_cfg_fn
    animated_drawing_dict = {
        'character_cfg': str(Path(char_anno_dir, 'char_cfg.yaml').resolve()),
        'motion_cfg': str(Path(motion_cfg_fn).resolve()),
        'retarget_cfg': str(Path(retarget_cfg_fn).resolve())
    }

    # create mvc config
    mvc_cfg = {
        'scene': {'ANIMATED_CHARACTERS': [animated_drawing_dict]},  # add the character to the scene
        'controller': {
            'MODE': 'video_render',  # 'video_render' or 'interactive'
            'OUTPUT_VIDEO_PATH': str(Path(char_anno_dir, 'video.gif').resolve())}  # set the output location
    }

    # write the new mvc config file out
    output_mvc_cfn_fn = str(Path(char_anno_dir, 'mvc_cfg.yaml'))
    with open(output_mvc_cfn_fn, 'w') as f:
        yaml.dump(dict(mvc_cfg), f)

    # render the video
    animated_drawings.render.start(output_mvc_cfn_fn)


def get_default_retarget_cfg(motion_cfg_fn: str) -> str:
    """Return the bundled retarget config that matches a known motion config."""
    try:
        with open(motion_cfg_fn, 'r') as f:
            motion_cfg = yaml.safe_load(f) or {}
    except OSError:
        return str(DEFAULT_RETARGET_CFG)

    bvh_filepath = motion_cfg.get('filepath')
    if not isinstance(bvh_filepath, str):
        return str(DEFAULT_RETARGET_CFG)

    bvh_name = Path(bvh_filepath).name
    return str(DEFAULT_RETARGET_CFG_BY_BVH.get(bvh_name, DEFAULT_RETARGET_CFG))


def parse_args():
    """Parse CLI arguments for the annotations-to-animation example pipeline."""
    parser = argparse.ArgumentParser(
        description=(
            "Render an animation from an annotated character directory and a "
            "selected motion config."
        )
    )
    parser.add_argument(
        "char_anno_dir",
        help="Directory containing char_cfg.yaml and the generated annotation assets.",
    )
    parser.add_argument(
        "motion_cfg_fn",
        nargs="?",
        default=str(DEFAULT_MOTION_CFG),
        help="Optional motion config YAML. Defaults to config/motion/dab.yaml.",
    )
    parser.add_argument(
        "retarget_cfg_fn",
        nargs="?",
        default=None,
        help=(
            "Optional retarget config YAML. If omitted, a bundled retarget "
            "config is selected to match the chosen motion config when "
            "available."
        ),
    )
    return parser.parse_args()


def main():
    """Run the animation renderer from CLI arguments and default config fallbacks."""

    log_dir = Path('./logs')
    log_dir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=f'{log_dir}/log.txt', level=logging.DEBUG)

    args = parse_args()
    retarget_cfg_fn = args.retarget_cfg_fn or get_default_retarget_cfg(args.motion_cfg_fn)

    annotations_to_animation(args.char_anno_dir, args.motion_cfg_fn, retarget_cfg_fn)


if __name__ == '__main__':
    main()
