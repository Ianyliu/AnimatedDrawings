#!/usr/bin/env bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
UV_BIN="$(command -v uv || true)"
PYTHON_BIN="${VENV_DIR}/bin/python"
TORCHSERVE_BIN="${VENV_DIR}/bin/torchserve"

require_command() {
	local command_name="$1"
	local install_hint="$2"

	if ! command -v "${command_name}" >/dev/null 2>&1; then
		echo "${command_name} could not be found on PATH."
		echo "${install_hint}"
		exit 1
	fi
}

get_java_major_version() {
	if ! command -v java >/dev/null 2>&1; then
		echo 0
		return
	fi

	local raw_version
	raw_version="$(java -version 2>&1 | head -n 1 | awk -F '\"' '{print $2}')"
	if [[ "${raw_version}" == 1.* ]]; then
		echo "${raw_version#1.}" | cut -d. -f1
	else
		echo "${raw_version}" | cut -d. -f1
	fi
}

sanitize_native_build_env() {
	# Conda shell flags leak Homebrew/Conda include and rpath settings into native extensions.
	unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PREFIX CONDA_PREFIX_1 CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_SHLVL
	unset DYLD_FALLBACK_LIBRARY_PATH DYLD_FRAMEWORK_PATH DYLD_LIBRARY_PATH DYLD_INSERT_LIBRARIES
	unset LIBRARY_PATH CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH OBJC_INCLUDE_PATH
	unset CFLAGS CPPFLAGS CXXFLAGS LDFLAGS
}

download_model() {
	local url="$1"
	local output="$2"

	if [[ -s "${output}" ]]; then
		echo "Already downloaded: ${output}"
		return
	fi

	curl -fL --retry 3 --connect-timeout 15 "${url}" -o "${output}"
}

if [[ "$(uname -s)" != "Darwin" ]]; then
	echo "This setup script is for local macOS TorchServe installs."
	exit 1
fi

if [[ "$(uname -m)" != "arm64" ]]; then
	echo "Warning: this script is optimized for Apple Silicon. Continuing on $(uname -m)."
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
	echo "Expected a uv-managed virtual environment at ${VENV_DIR}."
	echo "Create it from the repository root with:"
	echo "  uv python install 3.9"
	echo "  uv venv --python 3.9 .venv"
	echo "  uv pip install -e ."
	exit 1
fi

if [[ -z "${UV_BIN}" ]]; then
	echo "uv could not be found on PATH."
	echo "Install it with: brew install uv"
	exit 1
fi

require_command curl "curl ships with macOS. Reinstall the Xcode command line tools if it is missing."

if ! "${PYTHON_BIN}" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] == (3, 9) else 1)
PY
then
	echo "Local TorchServe setup requires Python 3.9 for the pinned OpenMMLab stack."
	echo "Recreate .venv from the repository root with:"
	echo "  uv python install 3.9"
	echo "  uv venv --python 3.9 .venv"
	echo "  uv pip install -e ."
	exit 1
fi

if "${PYTHON_BIN}" - <<'PY'
import sys
version = sys.version.lower()
prefix = sys.prefix.lower()
base_prefix = sys.base_prefix.lower()
raise SystemExit(0 if any(token in version or token in prefix or token in base_prefix for token in ("conda", "miniconda", "anaconda")) else 1)
PY
then
	echo "The virtual environment is using a Conda-backed Python interpreter."
	echo "That breaks the native xtcocotools extension used by local TorchServe on macOS."
	echo "Recreate .venv from the repository root with:"
	echo "  rm -rf .venv"
	echo "  uv python install 3.9"
	echo "  uv venv --python 3.9 .venv"
	echo "  uv pip install -e ."
	exit 1
fi

cd "${SCRIPT_DIR}"

sanitize_native_build_env

JAVA_MAJOR_VERSION="$(get_java_major_version)"
if (( JAVA_MAJOR_VERSION < 11 )); then
	echo "Java 11+ is required for TorchServe; found Java ${JAVA_MAJOR_VERSION}."
	echo "Installing Homebrew openjdk@17 and using it for this setup."
	require_command brew "Install Homebrew from https://brew.sh/, then rerun this script."
	brew install openjdk@17
	export JAVA_HOME="$(brew --prefix openjdk@17)/libexec/openjdk.jdk/Contents/Home"
	export PATH="${JAVA_HOME}/bin:${PATH}"
	JAVA_MAJOR_VERSION="$(get_java_major_version)"
fi

if (( JAVA_MAJOR_VERSION < 11 )); then
	echo "Unable to configure a Java 11+ runtime."
	exit 1
fi

echo "*** Installing packages"
# Bootstrap build tooling in the target venv because chumpy's build expects pip
# and xtcocotools needs a stable numpy/Cython toolchain available in-env.
"${PYTHON_BIN}" -m ensurepip --upgrade --default-pip
"${UV_BIN}" pip install --python "${PYTHON_BIN}" -U pip "setuptools<81" wheel "numpy==1.23.3" "cython>=0.27.3,<3"
# Make the repo package importable inside .venv without re-resolving all project deps.
"${UV_BIN}" pip install --python "${PYTHON_BIN}" --no-deps -e "${REPO_ROOT}"

if [[ ! -d xtcocoapi ]]; then
	git clone https://github.com/jin-s13/xtcocoapi.git
fi
cd xtcocoapi
"${PYTHON_BIN}" - <<'PY'
from pathlib import Path
import shutil

for name in ("build", "dist", "xtcocotools.egg-info", "xtcocotools/_mask.c"):
    path = Path(name)
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
PY
"${PYTHON_BIN}" -m cython xtcocotools/_mask.pyx -o xtcocotools/_mask.c
"${UV_BIN}" pip install --python "${PYTHON_BIN}" --no-build-isolation --no-deps --refresh --reinstall .
cd ..
"${PYTHON_BIN}" - <<'PY'
import xtcocotools._mask as mask
print("Verified xtcocotools._mask:", mask.__file__)
PY

"${UV_BIN}" pip install --python "${PYTHON_BIN}" --no-build-isolation-package chumpy -U openmim torch==1.13.0 torchserve mmdet==2.27.0 mmpose==0.29.0 numpy==1.23.3 platformdirs requests==2.31.0 scipy==1.10.0 tomli tqdm==4.64.1 scikit-image scikit-learn shapely glfw==2.5.5 PyOpenGL==3.1.6
# openmim still imports pkg_resources, so force a setuptools version that still ships it.
"${UV_BIN}" pip install --python "${PYTHON_BIN}" "setuptools<81"
"${PYTHON_BIN}" -m mim install mmcv-full==1.7.0
"${PYTHON_BIN}" - <<'PY'
import animated_drawings
import glfw
import mmcv
import mmdet
import mmpose
import OpenGL
import shapely
import skimage
import sklearn
print(
    "Verified imports:",
    animated_drawings.__file__,
    skimage.__version__,
    sklearn.__version__,
    shapely.__version__,
    getattr(OpenGL, "__version__", "unknown"),
    getattr(glfw, "__version__", "unknown"),
    mmcv.__version__,
    mmdet.__version__,
    mmpose.__version__,
)
PY

echo "*** Downloading models"
mkdir -p ./model-store
download_model \
	"https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_detector.mar" \
	"./model-store/drawn_humanoid_detector.mar"
download_model \
	"https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_pose_estimator.mar" \
	"./model-store/drawn_humanoid_pose_estimator.mar"

echo "*** Now run torchserve:"
echo "export JAVA_HOME=\"${JAVA_HOME:-$(/usr/libexec/java_home 2>/dev/null || true)}\""
echo "export PATH=\"\${JAVA_HOME}/bin:\${PATH}\""
echo "${TORCHSERVE_BIN} --start --disable-token-auth --ts-config config.local.properties --foreground"
