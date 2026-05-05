# Conditional Flow Landmark Corrector Plan

## Summary
Build a trainable **temporal inpainting postprocessor** that reconstructs corrupted landmark spans before the existing `PoseToBvhConverter`. The model will be a lightweight **conditional rectified-flow network** over landmark sequences, trained on synthetic 2D projections from mocap/BVH data.

The revised goal is not to make MediaPipe itself more confident. MediaPipe already emits landmarks every frame, but some spans can be unreliable. V1 treats low-confidence/corrupted spans as an inpainting problem:

```text
clean BVH-derived landmark motion
      -> synthetic span corruption
      -> model reconstructs clean x/y trajectory over corrupted spans
```

The model is useful only if it reconstructs corrupted spans better than deterministic baselines such as previous-valid and linear interpolation.

Final runtime pipeline:

```text
video -> MediaPipe PoseSequence
      -> span-aware landmark corrector for unreliable joints
      -> existing PoseToBvhConverter
      -> existing retarget/render
```

Use CMU mocap BVH as the default data source because it is public, BVH-based, and close to the repo’s motion format. Sources: [CMU Mocap](https://mocap.cs.cmu.edu/), [CMU BVH conversion notes](https://sites.google.com/a/cgspeed.com/cgspeed/motion-capture/the-motionbuilder-friendly-bvh-conversion-release-of-cmus-motion-capture-database), optional mirror [una-dinosauria/cmu-mocap](https://github.com/una-dinosauria/cmu-mocap).

## Data Plan
- Acquire data:
  - Default: download CMU BVH files from the GitHub mirror `una-dinosauria/cmu-mocap` or a user-provided CMU BVH directory on the server.
  - Do not crawl the CMU website directly; if using official CMU/cgspeed downloads, download the listed zip archives intentionally.
  - Also include existing repo BVHs under `examples/bvh` as smoke-test data.
- Convert BVH to canonical 13-landmark sequences:
  - Target landmarks: `NOSE`, shoulders, elbows, wrists, hips, knees, ankles.
  - Use flexible BVH alias mapping:
    - hips: `LeftUpLeg/RightUpLeg`, `LeftHip/RightHip`
    - knees: `LeftLeg/RightLeg`, `LeftKnee/RightKnee`
    - ankles: `LeftFoot/RightFoot`, `LeftAnkle/RightAnkle`
    - shoulders: `LeftArm/RightArm`, `LeftShoulder/RightShoulder`
    - elbows: `LeftForeArm/RightForeArm`, `LeftElbow/RightElbow`
    - wrists: `LeftHand/RightHand`, `LeftWrist/RightWrist`
    - nose: `Head` if available, otherwise extrapolate from `Neck -> Head`, otherwise skip clip.
  - Skip BVHs whose required body joints cannot be mapped; write a rejected-files report.
- Generate synthetic 2D training data:
  - Run BVH forward kinematics, sample clips of `T=31` frames. Use stride as a data-volume knob:
    - `stride=120` for quick CMU subset experiments.
    - `stride=30-60` for larger training runs.
    - `stride=15` for dense overlapping clips when runtime/storage are acceptable.
  - Apply random orthographic cameras: yaw `[-70, 70]`, pitch `[-15, 15]`, roll `[-8, 8]`, random scale/translation, normalize coordinates to roughly `[0, 1]`.
  - Clean target `y_clean`: `[T, 13, 2]` x/y landmarks. Treat these BVH-derived targets as fully trusted ground truth.
  - Corrupted input `x_corrupt`: clean x/y plus `visibility` and `low_conf_mask`, using synthetic span-corruption errors:
    - Mild Gaussian jitter on all non-corrupted joints.
    - Deliberate contiguous corrupted spans, not only independent per-frame noise.
    - Span lengths bucketed for evaluation:
      - short: `3-8` frames
      - medium: `9-16` frames
      - long: `17-30` frames, especially for harder experiments
    - Whole-limb corrupted spans, e.g. shoulder/elbow/wrist or hip/knee/ankle together.
    - Coordinate drift or large noise within corrupted spans to simulate unreliable landmark tracks.
    - Occasional one-frame spikes.
    - Occasional left/right swap bursts.
    - Optional high-confidence wrong spans as a separate stress-test split, because real estimators can sometimes be confidently wrong.
  - For corrupted spans, lower synthetic visibility and set `low_conf_mask=1`; for clean regions, use `visibility=1` and `low_conf_mask=0`.
  - Split by BVH file, not by clip: `80% train`, `10% val`, `10% test`.

## Model And Training
- Implement a PyTorch conditional rectified-flow model:
  - Input condition: corrupted landmarks `[x, y, visibility, low_conf_mask]` with shape `[B, T, 13, 4]`.
  - Flow state: current landmark estimate `[B, T, 13, 2]`.
  - Network: temporal residual TCN/U-Net style model with hidden size `128`, 6 residual blocks, kernel size `3`, dilations `[1, 2, 4, 8, 4, 2]`, GELU, dropout `0.1`, sinusoidal time embedding.
  - Output: velocity field `[B, T, 13, 2]`.
- Train rectified flow from corrupted to clean:
  - Sample `t ~ Uniform(0, 1)`.
  - `y_t = (1 - t) * x_corrupt_xy + t * y_clean`.
  - Target velocity `v = y_clean - x_corrupt_xy`.
  - Loss: masked SmoothL1/L1 on corrupted span joints plus small full-sequence smoothness loss.
  - Primary supervision is on `low_conf_mask == 1`; uncorrupted regions are context and should not dominate the loss.
  - Always include interpolation baseline; model must beat it on held-out corrupted-span reconstruction before runtime integration is considered successful.
- Track an optional second baseline/model:
  - Linear interpolation only.
  - Interpolation-residual TCN, where the model predicts `clean_xy - interpolated_xy`.
  - Use this if pure flow continues to lose to interpolation, because it directly asks the model to improve the baseline rather than rediscover it.
- Defaults:
  - `epochs=50`, `batch_size=128`, `lr=1e-3`, AdamW, cosine decay, gradient clip `1.0`.
  - Save best checkpoint by validation masked L1.
  - Export artifact as `landmark_flow_corrector.pt` plus `metadata.json` containing landmark order, window size, normalization, threshold, model config, and metrics.

## Runtime Integration
- Integrate as a new optional postprocessor, not a new pose estimator.
- Add config/env controls:
  - `VIDEO_APP_LANDMARK_CORRECTOR_MODEL=/path/to/landmark_flow_corrector.pt`
  - `VIDEO_APP_LANDMARK_CORRECTOR_ENABLED=true|false`
  - `VIDEO_APP_LANDMARK_CORRECTOR_THRESHOLD=0.35`
- Runtime behavior:
  - If no model path is configured, keep current `DefaultPosePostprocessor`.
  - If enabled, run current MediaPipe estimator, then flow corrector on sliding windows.
  - Blend conservatively:
    - if `visibility >= threshold`, keep MediaPipe x/y;
    - if `visibility < threshold`, replace x/y with corrected x/y;
    - preserve MediaPipe z/visibility by default.
    - optionally add a separate `repaired` flag rather than pretending MediaPipe was originally confident.
  - Run existing smoothing/root/foot checks afterward so current quality metrics remain comparable.
- Failure behavior:
  - If model load or inference fails, log warning and fall back to current rule-based postprocessor.

## Evaluation
- Offline landmark reconstruction:
  - Compare `previous valid`, `linear interpolation`, current rule-based repair, and flow model.
  - Metrics: masked-span L1, masked-span L2/RMSE, PCK@0.02 image units, jitter/acceleration, bone-length variance.
  - Report metrics by corruption type and span length:
    - random points
    - short spans
    - medium spans
    - long spans
    - whole-limb spans
    - left/right swaps
    - high-confidence wrong spans if enabled
  - Report interpolation metrics in the same table. Interpolation is a serious baseline; if it wins, use it.
- Pipeline evaluation:
  - Run `video -> MediaPipe -> postprocessor -> BVH` on a fixed small video set.
  - Compare current postprocessor vs flow model on:
    - `repaired_landmark_ratio`
    - root jumps
    - foot stabilizations
    - BVH file validity
    - visual overlay inspection
    - final rendered animation smoke test.
- Acceptance criteria:
  - Flow model beats linear interpolation by at least `10%` masked-span L1 and RMSE on held-out synthetic test clips, or beats it in the specific hard-span buckets where interpolation is known to fail.
  - PCK@0.02 is higher than interpolation on corrupted spans.
  - No regression in BVH generation smoke tests.
  - Runtime fallback works when model file is absent.
  - At least one end-to-end demo video renders successfully with and without the model.

## Assumptions
- The server has Python/PyTorch-capable training resources.
- CMU BVH or equivalent mocap data can be downloaded or placed on the server.
- V1 corrects only 2D x/y landmarks; current converter mostly ignores landmark z, so z refinement is out of scope.
- V1 does not calibrate or increase MediaPipe confidence. It repairs coordinates over unreliable spans and may mark those landmarks as repaired for downstream logic.
- V1 uses the 13 body landmarks needed by the current converter, not all 33 MediaPipe landmarks.
