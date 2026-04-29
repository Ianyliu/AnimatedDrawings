# TODO

This file tracks known risks and cleanup items that should be addressed before treating this repository as a maintained production service.

## Security

- Re-enable TorchServe token auth before exposing it beyond `127.0.0.1`. The local macOS workflow currently uses `--disable-token-auth` for compatibility with the example scripts.
- Add malware scanning or process sandboxing for uploaded videos, images, BVH files, and ffmpeg/OpenCV processing if the video app is ever deployed for untrusted users.
- Replace remaining `assert False` validation paths with explicit exceptions. Python can remove `assert` statements under optimized execution.
- Run `pip-audit` or an equivalent dependency audit regularly. The TorchServe/OpenMMLab stack pins older ML packages that may require security backports or isolation.
- Add authentication, persistent rate limits, and durable job storage before running the Flask video app as anything other than a local single-user tool.
- Keep Docker TorchServe ports private by default. `torchserve/config.properties` binds to `0.0.0.0` for container use, so host port publishing exposes it to the network.

## Reliability

- Add a lightweight smoke test that builds a fresh `.venv`, imports `animated_drawings`, verifies bundled package resources, and launches `examples/video_app.py --check` on Apple Silicon.
- Add an integration test for `torchserve/setup_macos.sh` in a clean macOS runner or documented manual release checklist.
- Make generated output cleanup configurable per session and test behavior with large, interrupted render jobs.
- Replace the in-process video app job manager with a persistent queue if concurrent users or long-running renders are required.
- Add stricter validation around generated TorchServe model responses, including keypoint array shape, bounding-box ranges, and confidence thresholds.

## Packaging And Repo Hygiene

- Stop tracking generated frontend bundles in `examples/fixer_app/dist` or document the exact build command and Node version used to regenerate them.
- Decide whether `torchserve/xtcocoapi` should remain vendored. If it stays, record the upstream commit and local patch policy.
- Consider adding a formatter/linter profile once the legacy rendering code has been separated from actively maintained code paths.
- Keep `uv.lock` refreshed after dependency changes and verify it on both Apple Silicon and Linux.
