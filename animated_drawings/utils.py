# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from importlib import resources
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

TOLERANCE = 10**-5
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parent


def package_resource_path(*parts: str) -> Path:
    """Return a filesystem path for a bundled animated_drawings resource."""

    return Path(str(resources.files("animated_drawings").joinpath(*parts)))


def resolve_ad_filepath(file_name: str, file_type: str) -> Path:
    """
    Given input filename, attempts to find the file, first by relative to cwd,
    then by absolute, then relative to the package and repository roots.
    If not found, raises a FileNotFoundError indicating which file_type it is.
    """
    path = Path(file_name).expanduser()
    candidates = [path]
    if not path.is_absolute():
        candidates.extend(
            [
                Path.cwd() / path,
                PACKAGE_ROOT / path,
                REPO_ROOT / path,
            ]
        )

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    msg = f"Could not find the {file_type} specified: {file_name}"
    logging.critical(msg)
    raise FileNotFoundError(msg)


def read_background_image(file_name: str) -> npt.NDArray[np.uint8]:
    """
    Given path to input image file, opens it, flips it based on EXIF tags, if present, and returns image with proper orientation.
    """
    # Check the file path
    file_path = resolve_ad_filepath(file_name, 'background_image')

    # Open the image and rotate as needed depending upon exif tag
    image = Image.open(str(file_path))
    image = ImageOps.exif_transpose(image)

    # Convert to numpy array and flip rightside up
    image_np = np.asarray(image)
    image_np = cv2.flip(image_np, 0)

    # Ensure we have RGBA
    if len(image_np.shape) == 3 and image_np.shape[-1] == 3:  # if RGB
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2RGBA)
    if len(image_np.shape) == 2:  # if grayscale
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGBA)

    return image_np.astype(np.uint8)
