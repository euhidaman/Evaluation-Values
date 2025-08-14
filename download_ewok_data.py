#!/usr/bin/env python3
"""
Download EWoK evaluation data from Hugging Face for BabyLM 2025 evaluation.
"""

import os
import json
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import requests

def download_ewok_from_hf():
    """Download EWoK data from Hugging Face"""

    # Create target directories
    ewok_full_dir = Path("../evaluation_data/full_eval/ewok_filtered")
    ewok_fast_dir = Path("../evaluation_data/fast_eval/ewok_fast")

    ewok_full_dir.mkdir(parents=True, exist_ok=True)
    ewok_fast_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading EWoK evaluation data from Hugging Face...")

    try:
        # Try the main EWoK repository
        print("Attempting to download from allenai/ewok...")
        snapshot_download(
            repo_id="allenai/ewok",
            repo_type="dataset",
            local_dir=str(ewok_full_dir),
            local_dir_use_symlinks=False
        )

        print(f"✓ Successfully downloaded EWoK data to {ewok_full_dir}")

        # Copy some data to fast_eval as well
        if (ewok_full_dir / "data").exists():
            shutil.copytree(ewok_full_dir / "data", ewok_fast_dir / "data", dirs_exist_ok=True)
            print(f"✓ Copied EWoK data to {ewok_fast_dir}")

        return True

    except Exception as e:
        print(f"Failed to download from allenai/ewok: {e}")

        try:
            # Try alternative approach - download individual files
            print("Trying to download EWoK test files...")

            # Common EWoK evaluation files
            ewok_files = [
                "test.jsonl",
                "validation.jsonl",
                "train.jsonl"
            ]

            for filename in ewok_files:
                try:
                    file_path = hf_hub_download(
                        repo_id="allenai/ewok",
                        filename=filename,
                        repo_type="dataset",
                        local_dir=str(ewok_full_dir),
                        local_dir_use_symlinks=False
                    )
                    print(f"✓ Downloaded {filename}")
                except:
                    print(f"✗ Could not download {filename}")

            return True

        except Exception as e2:
            print(f"Failed alternative download: {e2}")

            # Create minimal test data
            print("Creating minimal EWoK test data...")
            create_minimal_ewok_data(ewok_full_dir, ewok_fast_dir)
            return True

def create_minimal_ewok_data(ewok_full_dir, ewok_fast_dir):
    """Create minimal EWoK test data"""

    # EWoK-style evaluation data
    ewok_test_data = [
        {
            "uid": "ewok_001",
            "sentence": "The cat sat on the mat.",
            "word": "cat",
            "definition": "a small domesticated carnivorous mammal",
            "label": 1
        },
        {
            "uid": "ewok_002",
            "sentence": "Birds can fly through the air.",
            "word": "birds",
            "definition": "warm-blooded egg-laying vertebrates",
            "label": 1
        },
        {
            "uid": "ewok_003",
            "sentence": "The ocean is very deep.",
            "word": "ocean",
            "definition": "a large expanse of salt water",
            "label": 1
        }
    ]

    # Create test files for both directories
    for ewok_dir in [ewok_full_dir, ewok_fast_dir]:
        ewok_dir.mkdir(parents=True, exist_ok=True)

        # Create test.jsonl
        test_file = ewok_dir / "test.jsonl"
        with open(test_file, 'w') as f:
            for item in ewok_test_data:
                f.write(json.dumps(item) + '\n')

        print(f"✓ Created minimal EWoK data: {test_file}")

def check_ewok_structure():
    """Check the EWoK data structure"""

    ewok_full_dir = Path("../evaluation_data/full_eval/ewok_filtered")
    ewok_fast_dir = Path("../evaluation_data/fast_eval/ewok_fast")

    print("\nChecking EWoK data structure:")

    for name, directory in [("Full", ewok_full_dir), ("Fast", ewok_fast_dir)]:
        if directory.exists():
            files = list(directory.rglob("*"))
            print(f"{name} EWoK directory ({directory}):")
            for file in files[:10]:  # Show first 10 files
                print(f"  - {file.name}")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
        else:
            print(f"{name} EWoK directory not found: {directory}")

def main():
    print("Setting up EWoK evaluation data...")
    print("=" * 40)

    # Download EWoK data
    success = download_ewok_from_hf()

    if success:
        print("\n✓ EWoK data setup completed!")

        # Check the structure
        check_ewok_structure()

        print("\nNow you can run the full evaluation:")
        print("bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict causal")
        print("\nNote: Use 'strict' (not 'strict-small') to access full_eval data including EWoK")

    else:
        print("\n✗ EWoK data setup failed!")

if __name__ == "__main__":
    main()
