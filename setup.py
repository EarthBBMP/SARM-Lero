import subprocess
import sys
import os
from pathlib import Path


def run(cmd):
    print("  running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("Failed. See error above.")
        sys.exit(1)


def install_packages():
    print("\n[1/3] Installing packages...")

    # Check if torch has GPU support, reinstall if not
    try:
        import torch
        if torch.cuda.is_available():
            print("  torch already has GPU support:", torch.cuda.get_device_name(0))
        else:
            print("  torch has no GPU support, reinstalling with CUDA...")
            run([sys.executable, "-m", "pip", "uninstall", "torch", "torchvision", "-y"])
            run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
                 "--index-url", "https://download.pytorch.org/whl/cu118"])
    except ImportError:
        print("  torch not found, installing with CUDA support...")
        run([sys.executable, "-m", "pip", "install", "torch", "torchvision",
             "--index-url", "https://download.pytorch.org/whl/cu118"])

    # Install lerobot with sarm extras
    print("  installing lerobot[sarm]...")
    run([sys.executable, "-m", "pip", "install",
         "lerobot[sarm] @ git+https://github.com/huggingface/lerobot.git"])

    # Fix version conflicts
    print("  fixing package versions...")
    run([sys.executable, "-m", "pip", "install",
         "numpy<2.0",
         "huggingface_hub>=0.23.2,<1.0",
         "diffusers>=0.27.2,<0.32.0",
         "transformers>=4.44.0,<5.0",
         "accelerate"])

    print("  done.")


def download_dataset():
    print("\n[2/3] Downloading dataset...")

    save_path = os.path.join(
        os.path.expanduser("~"),
        ".cache", "huggingface", "lerobot", "cueng", "so101_demo_bowl"
    )

    # Skip if already downloaded
    if (Path(save_path) / "meta" / "info.json").exists():
        print("  dataset already downloaded at:", save_path)
        return

    print("  downloading cueng/so101_demo_bowl (about 2.56 GB)...")
    print("  this will take a few minutes...")

    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id="cueng/so101_demo_bowl",
        repo_type="dataset",
        local_dir=save_path,
        ignore_patterns=["*.md", ".gitattributes"],
    )

    print("  dataset saved to:", save_path)


def check_gpu():
    print("\n[3/3] Checking GPU...")
    import torch
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU found: {torch.cuda.get_device_name(0)} ({vram:.1f} GB)")
    else:
        print("  No GPU detected. Training will be slow on CPU.")
        print("  Make sure NVIDIA drivers are installed.")


def main():
    print("=" * 45)
    print("SARM Setup")
    print("=" * 45)

    install_packages()
    download_dataset()
    check_gpu()

    print("\n" + "=" * 45)
    print("Setup done! Now run: python run_pipeline.py")
    print("=" * 45)


if __name__ == "__main__":
    main()
