#!/usr/bin/env python3
"""
Script to download the specified models for BabyLM evaluation.
"""

import os
from huggingface_hub import snapshot_download
import argparse

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
    args = parser.parse_args()

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
        download_model(model_id, local_dir)

    print("All models downloaded successfully!")

if __name__ == "__main__":
    main()
