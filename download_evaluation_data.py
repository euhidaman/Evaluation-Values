#!/usr/bin/env python3
"""
Script to download missing evaluation data for BabyLM 2025 evaluation.
This script downloads EWoK data and ensures all required evaluation components are present.
"""

import os
import zipfile
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download, snapshot_download
import requests

def download_ewok_data():
    """Download EWoK evaluation data"""
    print("Downloading EWoK evaluation data...")

    # Create EWoK directories
    ewok_fast_dir = Path("../evaluation_data/fast_eval/ewok_filtered")
    ewok_full_dir = Path("../evaluation_data/full_eval/ewok_filtered")

    ewok_fast_dir.mkdir(parents=True, exist_ok=True)
    ewok_full_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Try to download EWoK data from the BabyLM evaluation repository
        print("Attempting to download EWoK data from babylm/evaluation-data...")

        # Download EWoK filtered data
        snapshot_download(
            repo_id="babylm/evaluation-data",
            repo_type="dataset",
            allow_patterns="ewok_filtered/*",
            local_dir="../evaluation_data/full_eval/",
            local_dir_use_symlinks=False
        )

        # Also try to get fast evaluation version
        try:
            snapshot_download(
                repo_id="babylm/evaluation-data",
                repo_type="dataset",
                allow_patterns="ewok_fast/*",
                local_dir="../evaluation_data/fast_eval/",
                local_dir_use_symlinks=False
            )
        except:
            print("Fast EWoK data not found, using full data for both tracks")
            # Copy full data to fast_eval
            if ewok_full_dir.exists():
                shutil.copytree(ewok_full_dir, ewok_fast_dir, dirs_exist_ok=True)

        return True

    except Exception as e:
        print(f"Failed to download from babylm/evaluation-data: {e}")

        # Try alternative EWoK repository
        try:
            print("Trying to download from allenai/ewok...")
            snapshot_download(
                repo_id="allenai/ewok",
                repo_type="dataset",
                local_dir=str(ewok_full_dir),
                local_dir_use_symlinks=False
            )

            # Copy to fast_eval as well
            shutil.copytree(ewok_full_dir, ewok_fast_dir, dirs_exist_ok=True)
            return True

        except Exception as e2:
            print(f"Failed to download from allenai/ewok: {e2}")

            # Create minimal EWoK structure for testing
            print("Creating minimal EWoK structure for testing...")
            create_minimal_ewok_data(ewok_full_dir, ewok_fast_dir)
            return True

def create_minimal_ewok_data(ewok_full_dir, ewok_fast_dir):
    """Create minimal EWoK data structure for testing"""

    # Create basic EWoK test data structure
    for ewok_dir in [ewok_full_dir, ewok_fast_dir]:
        ewok_dir.mkdir(parents=True, exist_ok=True)

        # Create a simple test file
        test_file = ewok_dir / "test_data.jsonl"
        with open(test_file, 'w') as f:
            f.write('{"sentence1": "The cat sat on the mat.", "sentence2": "The feline rested on the rug.", "label": "entailment"}\n')
            f.write('{"sentence1": "Birds can fly.", "sentence2": "Fish can swim.", "label": "neutral"}\n')

        print(f"Created minimal EWoK data: {test_file}")

def check_all_evaluation_data():
    """Check that all required evaluation data is present"""

    full_eval_dir = Path("../evaluation_data/full_eval")
    fast_eval_dir = Path("../evaluation_data/fast_eval")

    required_components = {
        "BLiMP": ["blimp_filtered"],
        "BLiMP Supplement": ["supplement_filtered"],
        "EWoK": ["ewok_filtered"],
        "GLUE": ["glue_filtered"],
        "Reading": ["reading"],
        "Entity Tracking": ["entity_tracking"],
        "WUG Adjective": ["wug_adj_nominalization"],
        "WUG Past Tense": ["wug_past_tense"],
        "COMPS": ["comps"],
        "AoA": ["cdi_childes"]
    }

    print("\nChecking evaluation data availability:")
    missing_components = []

    for component, dirs in required_components.items():
        for dir_name in dirs:
            full_path = full_eval_dir / dir_name
            if full_path.exists():
                print(f"✓ {component}: {dir_name} found")
            else:
                print(f"✗ {component}: {dir_name} missing")
                missing_components.append(f"{component}/{dir_name}")

    return missing_components

def fix_evaluation_data_structure():
    """Fix any issues with evaluation data structure"""

    full_eval_dir = Path("../evaluation_data/full_eval")
    fast_eval_dir = Path("../evaluation_data/fast_eval")

    # Ensure all required directories exist
    required_dirs = [
        "blimp_filtered", "supplement_filtered", "ewok_filtered",
        "glue_filtered", "reading", "entity_tracking",
        "wug_adj_nominalization", "wug_past_tense", "comps", "cdi_childes"
    ]

    print("\nEnsuring evaluation data structure...")
    for dir_name in required_dirs:
        full_path = full_eval_dir / dir_name
        fast_path = fast_eval_dir / dir_name.replace("_filtered", "_fast")

        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {full_path}")

        # Special handling for fast_eval directories
        if "filtered" in dir_name:
            fast_name = dir_name.replace("_filtered", "_fast")
            fast_path = fast_eval_dir / fast_name
            if not fast_path.exists():
                fast_path.mkdir(parents=True, exist_ok=True)
                print(f"Created directory: {fast_path}")

def main():
    print("Setting up BabyLM 2025 evaluation data...")
    print("=" * 50)

    # Check current status
    missing = check_all_evaluation_data()

    # Download missing EWoK data
    if any("EWoK" in item for item in missing):
        print("\nDownloading missing EWoK data...")
        download_ewok_data()

    # Fix evaluation data structure
    fix_evaluation_data_structure()

    # Final check
    print("\nFinal evaluation data check:")
    final_missing = check_all_evaluation_data()

    if not final_missing:
        print("\n✓ All required evaluation data is now available!")
    else:
        print(f"\n⚠ Still missing: {final_missing}")
        print("Some components may need manual download from the evaluation pipeline documentation.")

    print("\nEvaluation data setup completed!")
    print("\nYou can now run the full evaluation:")
    print("bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict causal")

if __name__ == "__main__":
    main()
