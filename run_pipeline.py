# run_pipeline.py
# SARM Reward Model Pipeline
# Author: Earth ADME #1
#
# Trains a reward model on cueng/so101_demo_bowl and outputs
# sarm_progress.parquet to use as the reward signal in RLT training.
#
# Reference: https://huggingface.co/docs/lerobot/sarm
# Paper: https://arxiv.org/abs/2509.25358
#
# Usage:
#   python run_pipeline.py                  <- run everything
#   python run_pipeline.py --step train
#   python run_pipeline.py --step visualize
#   python run_pipeline.py --step progress

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


# Config

DATASET_REPO_ID = "cueng/so101_demo_bowl"

DATASET_ROOT = os.path.join(
    os.path.expanduser("~"),
    ".cache", "huggingface", "lerobot", "cueng", "so101_demo_bowl"
)

OUTPUT_DIR   = "outputs/train/sarm"
VIZ_DIR      = "outputs/sarm_viz"
PROGRESS_DIR = "outputs"

IMAGE_KEY  = "observation.images.top"
STATE_KEY  = "observation.state"
STEPS      = 20000
BATCH_SIZE = 32
NUM_VIZ    = 5


# Helpers

def run_cmd(cmd, label):
    print("\n" + "=" * 55)
    print(" ", label)
    print(" ", " ".join(str(c) for c in cmd))
    print("=" * 55 + "\n")
    result = subprocess.run([str(c) for c in cmd])
    if result.returncode != 0:
        print(f"\nFailed: {label}")
        print("Fix the error above then re-run with --step <name>")
        sys.exit(result.returncode)
    print(f"\nDone: {label}\n")


def get_sarm_script(name):
    import lerobot
    path = Path(lerobot.__file__).parent / "policies" / "sarm" / name
    if not path.exists():
        print(f"Cannot find {name}. Run setup.py first.")
        sys.exit(1)
    return path


def check_env():
    print("Checking environment...")
    ok = True

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print("  lerobot: OK")
    except ImportError:
        print("  lerobot: missing -- run setup.py")
        ok = False

    try:
        import torch
        if torch.cuda.is_available():
            print(f"  torch {torch.__version__}: GPU {torch.cuda.get_device_name(0)}")
        else:
            print(f"  torch {torch.__version__}: no GPU (will be slow)")
    except ImportError:
        print("  torch: missing -- run setup.py")
        ok = False

    try:
        import numpy as np
        if int(np.__version__.split(".")[0]) >= 2:
            print(f"  numpy {np.__version__}: too new, run: pip install 'numpy<2.0'")
            ok = False
        else:
            print(f"  numpy {np.__version__}: OK")
    except ImportError:
        print("  numpy: missing -- run setup.py")
        ok = False

    if not ok:
        print("\nSome packages are missing. Run: python setup.py")
        sys.exit(1)

    print("All good, starting pipeline...\n")


# Steps

def train():
    cmd = [
        "lerobot-train",
        f"--dataset.repo_id={DATASET_REPO_ID}",
        f"--dataset.root={DATASET_ROOT}",
        "--policy.type=sarm",
        "--policy.annotation_mode=single_stage",
        f"--policy.image_key={IMAGE_KEY}",
        f"--policy.state_key={STATE_KEY}",
        "--policy.n_obs_steps=8",
        "--policy.frame_gap=30",
        f"--output_dir={OUTPUT_DIR}",
        f"--batch_size={BATCH_SIZE}",
        f"--steps={STEPS}",
        "--num_workers=0",
        "--policy.push_to_hub=false",
        "--wandb.enable=false",
    ]

    out = Path(OUTPUT_DIR)
    if (out / "train_config.json").exists():
        cmd.append("--resume=true")
        print("Resuming from checkpoint...")
    elif out.exists():
        shutil.rmtree(out)
        print("Cleared old output dir, starting fresh...")
    else:
        print("Starting fresh training...")

    run_cmd(cmd, "Train SARM reward model")


def visualize():
    Path(VIZ_DIR).mkdir(parents=True, exist_ok=True)
    script = get_sarm_script("compute_rabc_weights.py")

    cmd = [
        sys.executable, script,
        f"--dataset-repo-id={DATASET_REPO_ID}",
        f"--reward-model-path={OUTPUT_DIR}",
        "--visualize-only",
        f"--num-visualizations={NUM_VIZ}",
        "--head-mode=sparse",
        f"--output-dir={VIZ_DIR}",
    ]

    run_cmd(cmd, "Visualize predictions")
    print(f"Plots saved to: {VIZ_DIR}/")
    print("Progress should rise toward 1.0 by end of each episode.")


def compute_progress():
    Path(PROGRESS_DIR).mkdir(parents=True, exist_ok=True)
    script = get_sarm_script("compute_rabc_weights.py")

    cmd = [
        sys.executable, script,
        f"--dataset-repo-id={DATASET_REPO_ID}",
        f"--reward-model-path={OUTPUT_DIR}",
        "--head-mode=sparse",
        f"--num-visualizations={NUM_VIZ}",
        f"--output-dir={PROGRESS_DIR}",
    ]

    run_cmd(cmd, "Compute progress values for all frames")
    print(f"Saved: {PROGRESS_DIR}/sarm_progress.parquet")


def print_summary():
    print("=" * 50)
    print("Pipeline complete!")
    print()
    print("Outputs:")
    print(f"  Model:    {OUTPUT_DIR}/")
    print(f"  Plots:    {VIZ_DIR}/")
    print(f"  Reward:   {PROGRESS_DIR}/sarm_progress.parquet")
    print()
    print("=" * 50)


# Main

def main():
    parser = argparse.ArgumentParser(description="SARM reward model pipeline")
    parser.add_argument(
        "--step",
        choices=["all", "train", "visualize", "progress"],
        default="all",
        help="Which step to run (default: all)"
    )
    args = parser.parse_args()

    check_env()

    if args.step in ("all", "train"):
        train()

    if args.step in ("all", "visualize"):
        visualize()

    if args.step in ("all", "progress"):
        compute_progress()

    if args.step == "all":
        print_summary()


if __name__ == "__main__":
    main()