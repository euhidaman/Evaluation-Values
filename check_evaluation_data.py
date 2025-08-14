#!/usr/bin/env python3
"""
Script to download missing evaluation data for BabyLM 2025 evaluation.
"""

import os
import requests
import zipfile
from pathlib import Path

def download_file(url, filepath):
    """Download a file from URL to filepath"""
    print(f"Downloading {url}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(filepath, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"Downloaded to {filepath}")

def extract_zip(zip_path, extract_to):
    """Extract a zip file"""
    print(f"Extracting {zip_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted to {extract_to}")

def ensure_evaluation_data():
    """Ensure all required evaluation data is available"""

    # Base paths
    eval_data_dir = Path("../evaluation_data")
    fast_eval_dir = eval_data_dir / "fast_eval"
    full_eval_dir = eval_data_dir / "full_eval"

    # Create directories if they don't exist
    fast_eval_dir.mkdir(parents=True, exist_ok=True)
    full_eval_dir.mkdir(parents=True, exist_ok=True)

    print("Checking evaluation data availability...")

    # Check if critical directories exist
    required_dirs = {
        "fast_eval": [
            "blimp_fast", "ewok_fast", "supplement_fast",
            "entity_tracking_fast", "wug_adj_nominalization",
            "wug_past_tense", "reading"
        ],
        "full_eval": [
            "blimp_filtered", "ewok_filtered", "supplement_filtered",
            "entity_tracking", "wug_adj_nominalization", "wug_past_tense",
            "comps", "glue_filtered", "reading", "cdi_childes"
        ]
    }

    missing_data = []

    for eval_type, dirs in required_dirs.items():
        base_dir = eval_data_dir / eval_type
        for dir_name in dirs:
            if not (base_dir / dir_name).exists():
                missing_data.append(f"{eval_type}/{dir_name}")

    if missing_data:
        print("Missing evaluation data directories:")
        for item in missing_data:
            print(f"  - {item}")
        print("\nNOTE: Some evaluation data may need to be downloaded separately.")
        print("Please check the evaluation-pipeline-2025 repository for data download instructions.")
        print("The evaluation scripts will skip missing components gracefully.")
    else:
        print("All required evaluation data directories are present.")

    # Create AoA data directory structure if missing
    aoa_dir = full_eval_dir / "cdi_childes"
    if not aoa_dir.exists():
        aoa_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created AoA data directory: {aoa_dir}")

        # Create a placeholder file with instructions
        placeholder_file = aoa_dir / "README_DOWNLOAD_NEEDED.txt"
        with open(placeholder_file, 'w') as f:
            f.write("""
CDI-CHILDES data for Age of Acquisition evaluation is missing.

To download this data:
1. Check the evaluation-pipeline-2025 repository documentation
2. Look for AoA evaluation data download instructions
3. The expected file should be: cdi_childes.json

Without this data, the AoA evaluation will be skipped.
""")

if __name__ == "__main__":
    ensure_evaluation_data()
    print("Evaluation data check completed!")
