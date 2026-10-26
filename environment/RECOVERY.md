# RL Environment Recovery

This repository contains the portable engineering assets for the VERL RL experiments:

- `patches/002-ngrpo-algorithm.patch`: complete NGRPO patch for the validated OpenPangu image.
- `patch_tools/patch_verl_ngrpo.py`: runtime NGRPO patcher with backup and restore.
- `patches/001-rollout-diagnostics.patch`: rollout evidence patch.
- `tools/verified_math/`: verified math data preparation, preflight, audit, summary, and short-run launcher.
- `assets/data/math_chat_concise_answer/`: the small train/test parquet inputs used by the verified math run.
- `notebooks/` and `src/rl_diagnostics/`: Jupyter and browser diagnostics.

## What is intentionally not committed

Model weights and Docker images are external dependencies. The current server references:

- Model: Qwen3-8B, mounted inside the container as `/mnt/hzy/qwen7b`.
- Model host path used here: `/mnt/local/pangu_rl/models/qwen3-8b`.
- Image: `openpangu-72b-2512-postrain:v1-zkx-pangu-mathfix`.
- Image ID: `sha256:ea9d28f13a103f7016f3cee624826a6186b022a588ae9d208f27264111b61d7a`.
- Original image archive SHA256: see `environment/artifacts.sha256`.

The image archive and model weights are too large for a normal Git repository. Obtain them from the artifact store, verify their hashes, then mount them using the paths expected by the launcher.

## Rebuild the local toolkit

```bash
git clone <repository>
cd rl-training-toolkit
python3 -m pip install -e .
```

The runtime patch and dashboard have no third-party runtime dependency beyond the packages already present in the OpenPangu image. The optional Jupyter interface uses the host's existing Jupyter installation.

## Apply runtime patches inside the container

```bash
python3 patch_tools/patch_verl_ngrpo.py apply \
  --verl-root /workspace/verl \
  --backup-dir /workspace/verl_ngrpo_backup
python3 patch_tools/patch_verl_ngrpo.py check --verl-root /workspace/verl
```

For the validated image, the complete static patch can also be applied with:

```bash
patch --batch --forward --directory=/workspace/verl -p1 \
  < patches/002-ngrpo-algorithm.patch
```

## Run the verified math experiment

The launcher is designed to run inside the VERL container and requires only environment variables:

```bash
export MODEL_PATH=/mnt/hzy/qwen7b
export DATA_DIR=/mnt/zkx/assets/data/math_chat_concise_answer
export RUN_ROOT=/mnt/zkx/runs
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3
export TOTAL_STEPS=5
bash tools/verified_math/run_verified_math.sh
```

The dashboard reads `RUN_ROOT` and a state directory outside Git:

```bash
rl-dashboard \
  --runs-root /mnt/local/transformer_files/zkx/runs \
  --state-dir /mnt/local/transformer_files/zkx/experiment_registry \
  --host 0.0.0.0 --port 8765
```

## Move experiment evidence to another machine

Create a portable archive without checkpoints or model weights:

```bash
python3 tools/experiment_bundle.py export \
  --runs-root /mnt/local/transformer_files/zkx/runs \
  --output /mnt/local/transformer_files/zkx/exports/rl-experiments.tar.gz \
  --all
```

After transferring the archive and cloning this repository, restore and serve it:

```bash
python3 tools/experiment_bundle.py import \
  --archive ~/Downloads/rl-experiments.tar.gz \
  --output ~/rl-experiment-runs
rl-dashboard --runs-root ~/rl-experiment-runs \
  --state-dir ~/rl-experiment-state --host 127.0.0.1 --port 8765
```
