from __future__ import annotations

import importlib.util
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests


@dataclass(frozen=True)
class DiagnosticCheck:
    id: str
    label: str
    status: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "label": self.label,
            "status": self.status,
            "message": self.message,
        }


def run_diagnostics(config: dict[str, Any] | None = None) -> list[DiagnosticCheck]:
    config = config or {}
    checks = [
        _module_check("flask", "Flask"),
        _module_check("mediapipe", "MediaPipe", warning=True),
        _module_check("sklearn", "scikit-learn"),
        _module_check("catboost", "CatBoost", warning=True),
        _binary_check("ffmpeg", "ffmpeg"),
        _binary_check("ffprobe", "ffprobe"),
        _opencv_video_check(),
        _renderer_check(),
        _torchserve_check(timeout=float(config.get("torchserve_timeout", 2.0))),
    ]
    return checks


def diagnostics_payload(config: dict[str, Any] | None = None) -> dict[str, Any]:
    checks = run_diagnostics(config)
    has_error = any(check.status == "error" for check in checks)
    has_warning = any(check.status == "warning" for check in checks)
    status = "error" if has_error else "warning" if has_warning else "ok"
    return {
        "status": status,
        "checks": [check.to_dict() for check in checks],
    }


def format_diagnostics(checks: list[DiagnosticCheck]) -> str:
    lines = ["Video app diagnostics:"]
    for check in checks:
        marker = {"ok": "OK", "warning": "WARN", "error": "ERR"}.get(check.status, check.status.upper())
        lines.append(f"  [{marker}] {check.label}: {check.message}")
    return "\n".join(lines)


def _module_check(module_name: str, label: str, warning: bool = False) -> DiagnosticCheck:
    if importlib.util.find_spec(module_name) is not None:
        return DiagnosticCheck(module_name, label, "ok", "available")
    status = "warning" if warning else "error"
    return DiagnosticCheck(module_name, label, status, "not installed")


def _binary_check(binary_name: str, label: str) -> DiagnosticCheck:
    path = shutil.which(binary_name)
    if path:
        return DiagnosticCheck(binary_name, label, "ok", path)
    return DiagnosticCheck(binary_name, label, "warning", "not found on PATH; browser MP4 preparation may fail")


def _opencv_video_check() -> DiagnosticCheck:
    try:
        with tempfile.TemporaryDirectory() as tmp:
            video_path = Path(tmp) / "probe.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (16, 16))
            if not writer.isOpened():
                return DiagnosticCheck("opencv_video", "OpenCV video", "warning", "could not open MP4 writer")
            writer.write(np.zeros((16, 16, 3), dtype=np.uint8))
            writer.release()
            cap = cv2.VideoCapture(str(video_path))
            ok = cap.isOpened()
            cap.release()
            if not ok:
                return DiagnosticCheck("opencv_video", "OpenCV video", "warning", "could not read generated MP4")
        return DiagnosticCheck("opencv_video", "OpenCV video", "ok", "read/write available")
    except Exception as e:  # pragma: no cover - platform-dependent
        return DiagnosticCheck("opencv_video", "OpenCV video", "warning", str(e))


def _renderer_check() -> DiagnosticCheck:
    if importlib.util.find_spec("animated_drawings.render") is not None:
        return DiagnosticCheck("renderer", "Renderer", "ok", "module available")
    return DiagnosticCheck("renderer", "Renderer", "error", "animated_drawings.render could not be imported")


def _torchserve_check(timeout: float) -> DiagnosticCheck:
    try:
        response = requests.get("http://127.0.0.1:8080/ping", timeout=timeout)
    except requests.RequestException:
        return DiagnosticCheck(
            "torchserve",
            "TorchServe",
            "warning",
            "not reachable; bundled characters still work, drawing uploads need TorchServe",
        )
    if response.ok:
        return DiagnosticCheck("torchserve", "TorchServe", "ok", "healthy")
    return DiagnosticCheck("torchserve", "TorchServe", "warning", f"returned HTTP {response.status_code}")
