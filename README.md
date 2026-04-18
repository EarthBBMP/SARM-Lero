# SARM Reward Model
Trains a SARM reward model on `cueng/so101_demo_bowl` and outputs
`sarm_progress.parquet` to use as the reward signal in RLT training.

Reference: https://huggingface.co/docs/lerobot/sarm

---

## First time setup

```
python setup.py
```

This will install all packages and download the dataset (~2.56 GB).

---

## Run

```
python run_pipeline.py
```

Or run one step at a time:

```
python run_pipeline.py --step train
python run_pipeline.py --step visualize
python run_pipeline.py --step progress
```

---

## What each step does

**train** - Trains the SARM model. Uses `single_stage` mode so no
annotation is needed. Progress is linear 0 to 1 over each episode.
Saves checkpoint to `outputs/train/sarm/`.

**visualize** - Plots predicted progress curves for 5 episodes.
Progress should rise toward 1.0 by end of episode. Saves plots
to `outputs/sarm_viz/`.

**progress** - Runs the model on all 221,896 frames and saves
`outputs/sarm_progress.parquet`.

---

## Output

It has these columns:

| Column | Description |
|---|---|
| index | global frame index |
| episode_index | episode number |
| frame_index | frame within episode |
| progress_sparse | reward score 0.0 to 1.0 |

Use it in RLT training like this:

```
lerobot-train \
  --dataset.repo_id=cueng/so101_demo_bowl \
  --policy.type=smolvla \
  --use_rabc=true \
  --rabc_head_mode=sparse \
  --rabc_kappa=0.01 \
  --output_dir=outputs/train/smolvla_rabc \
  --steps=40000
```