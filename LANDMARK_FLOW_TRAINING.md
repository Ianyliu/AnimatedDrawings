# Landmark Flow Training

This adds a standalone training path for the `PLAN_CV.md` landmark corrector. It does not change the existing animation runtime yet.

## 1. Prepare Data

The default data source is the local CMU mirror:

```text
cmu-mocap/data
```

By default the Slurm data job uses `MAX_FILES=400`, `STRIDE=30`, and span corruption up to `MAX_SPAN=30`.

Prepare the default CMU subset:

```bash
sbatch scripts/slurm/prepare_landmark_flow_data.sbatch
```

Use all local CMU BVHs:

```bash
MAX_FILES=all sbatch scripts/slurm/prepare_landmark_flow_data.sbatch
```

Use a different BVH directory:

```bash
BVH_ROOT=/path/to/bvh-folder MAX_FILES=all sbatch scripts/slurm/prepare_landmark_flow_data.sbatch
```

Generate fewer overlapping clips from the same BVH files by raising stride:

```bash
STRIDE=60 sbatch scripts/slurm/prepare_landmark_flow_data.sbatch
```

Generate more overlapping clips by lowering stride:

```bash
STRIDE=15 sbatch scripts/slurm/prepare_landmark_flow_data.sbatch
```

Tune corruption difficulty:

```bash
MIN_SPAN=8 MAX_SPAN=30 WHOLE_LIMB_PROB=0.5 SWAP_PROB=0.25 sbatch scripts/slurm/prepare_landmark_flow_data.sbatch
```

Outputs go to `data/landmark_flow/{train,val,test}.npz` plus `metadata.json`. The metadata includes accepted and rejected BVH files.

## 2. Train

```bash
sbatch scripts/slurm/train_landmark_flow.sbatch
```

Useful overrides:

```bash
EPOCHS=10 BATCH_SIZE=64 sbatch scripts/slurm/train_landmark_flow.sbatch
```

Model/hyperparameter overrides:

```bash
HIDDEN_SIZE=256 DROPOUT=0.05 LR=3e-4 EPOCHS=100 sbatch scripts/slurm/train_landmark_flow.sbatch
```

The exported model is:

```text
outputs/landmark_flow/landmark_flow_corrector.pt
outputs/landmark_flow/metadata.json
```

## 3. Evaluate

```bash
sbatch scripts/slurm/evaluate_landmark_flow.sbatch
```

For CPU-only evaluation:

```bash
sbatch scripts/slurm/evaluate_landmark_flow_cpu.sbatch
```

The key acceptance check is whether `masked_l1` and `masked_rmse` beat the matching interpolation metrics by at least 10 percent on the test split or on the hard corruption buckets.

## 4. Inference / Real-time
Running the real-time animation with flow-based correction model, threshold as 0.5
```bash
uv run python examples/webcam_to_animation.py \
  --landmark-flow-model outputs/landmark_flow/landmark_flow_corrector.pt \
  --landmark-flow-threshold 0.5
```
Running the real-time animation without flow-based correction model
```bash
uv run python examples/webcam_to_animation.py --no-landmark-flow
```

## Notes

- If your cluster requires modules or a Conda environment, add the relevant `module load ...` or `conda activate ...` lines near the top of each `.sbatch` file.
- The Slurm scripts run through `uv`, so missing Python packages are resolved in the job environment.
- Data-prep dependencies live in `requirements-landmark-flow-data.txt`; training/eval dependencies live in `requirements-landmark-flow.txt`.
- The bundled CMU mirror currently has 2,548 BVH files under `cmu-mocap/data`.
