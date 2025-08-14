#!/usr/bin/env python3
"""
Script to download the specified models for BabyLM evaluation.
"""

import os
import subprocess
import sys
from huggingface_hub import snapshot_download
import argparse

def install_dependencies():
    """Install additional dependencies for specific models"""
    print("Installing additional dependencies...")

    # Install ai2-olmo for the DataDecide model (which uses hf_olmo architecture)
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ai2-olmo"])
        print("Successfully installed ai2-olmo")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to install ai2-olmo: {e}")
        print("This may cause issues with the DataDecide model")

    # Upgrade transformers to latest version
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "transformers"])
        print("Successfully upgraded transformers")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to upgrade transformers: {e}")

def download_model(model_id, local_dir):
    """Download a model from Hugging Face Hub"""
    print(f"Downloading {model_id}...")
    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=False
        )
        print(f"Successfully downloaded {model_id} to {local_dir}")
    except Exception as e:
        print(f"Error downloading {model_id}: {e}")
        raise

def main():
    parser = argparse.ArgumentParser(description="Download models for BabyLM evaluation")
    parser.add_argument("--models_dir", default="./models", help="Directory to save models")
    parser.add_argument("--install_deps", action="store_true", help="Install additional dependencies")
    args = parser.parse_args()

    # Install dependencies if requested
    if args.install_deps:
        install_dependencies()

    # Create models directory
    os.makedirs(args.models_dir, exist_ok=True)

    # Models to download
    models = [
        "microsoft/bitnet-b1.58-2B-4T",
        "allenai/DataDecide-dolma1_7-no-math-code-14M",
        "Xuandong/HPD-TinyBERT-F128"
    ]

    for model_id in models:
        # Create safe directory name
        model_name = model_id.replace("/", "--")
        local_dir = os.path.join(args.models_dir, model_name)

        try:
            download_model(model_id, local_dir)
        except Exception as e:
            print(f"Failed to download {model_id}. You may need to:")
            print("1. Check your internet connection")
            print("2. Verify the model exists on Hugging Face")
            print("3. Install additional dependencies with --install_deps")
            continue

    print("\nModel download completed!")
    print("\nNote: If you encounter model loading issues, try:")
    print("1. Run with --install_deps flag to install additional dependencies")
    print("2. Upgrade transformers: pip install --upgrade transformers")
    print("3. Install ai2-olmo: pip install ai2-olmo")

if __name__ == "__main__":
    main()
