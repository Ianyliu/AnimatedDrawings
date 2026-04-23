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

if [[ ! -x "${PYTHON_BIN}" ]]; then
	echo "Expected a uv-managed virtual environment at ${VENV_DIR}."
	echo "Create it from the repository root with:"
	echo "  uv venv .venv"
	echo "  uv pip install -e ."
	exit 1
fi

if [[ -z "${UV_BIN}" ]]; then
	echo "uv could not be found on PATH."
	exit 1
fi

cd "${SCRIPT_DIR}"

JAVA_MAJOR_VERSION="$(get_java_major_version)"
if (( JAVA_MAJOR_VERSION < 11 )); then
	echo "Java 11+ is required for TorchServe; found Java ${JAVA_MAJOR_VERSION}."
	echo "Installing Homebrew openjdk@17 and using it for this setup."
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
if [[ ! -d xtcocoapi ]]; then
	git clone https://github.com/jin-s13/xtcocoapi.git
fi
cd xtcocoapi
"${UV_BIN}" pip install --python "${PYTHON_BIN}" -r requirements.txt
"${UV_BIN}" pip install --python "${PYTHON_BIN}" .
cd ..

# Bootstrap build tooling in the target venv because chumpy's build expects pip.
"${PYTHON_BIN}" -m ensurepip --upgrade --default-pip
"${UV_BIN}" pip install --python "${PYTHON_BIN}" -U pip "setuptools<81" wheel
"${UV_BIN}" pip install --python "${PYTHON_BIN}" --no-build-isolation-package chumpy -U openmim torch==1.13.0 torchserve mmdet==2.27.0 mmpose==0.29.0 numpy==1.23.3 requests==2.31.0 scipy==1.10.0 tqdm==4.64.1
# openmim still imports pkg_resources, so force a setuptools version that still ships it.
"${UV_BIN}" pip install --python "${PYTHON_BIN}" "setuptools<81"
"${PYTHON_BIN}" -m mim install mmcv-full==1.7.0

echo "*** Downloading models"
mkdir -p ./model-store
curl -L https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_detector.mar -o ./model-store/drawn_humanoid_detector.mar
curl -L https://github.com/facebookresearch/AnimatedDrawings/releases/download/v0.0.1/drawn_humanoid_pose_estimator.mar -o ./model-store/drawn_humanoid_pose_estimator.mar

echo "*** Now run torchserve:"
echo "export JAVA_HOME=\"${JAVA_HOME:-$(/usr/libexec/java_home 2>/dev/null || true)}\""
echo "export PATH=\"\${JAVA_HOME}/bin:\${PATH}\""
echo "${TORCHSERVE_BIN} --start --ts-config config.local.properties --foreground"
