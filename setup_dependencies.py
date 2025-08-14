#!/usr/bin/env python3
"""
Setup script to install all required dependencies for BabyLM 2025 evaluation.
This script handles the specific requirements for different model architectures.
"""

import subprocess
import sys
import os

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    print("BabyLM 2025 Evaluation Setup")
    print("============================")

    # Step 1: Upgrade pip
    run_command("python -m pip install --upgrade pip", "Upgrading pip")

    # Step 2: Install base requirements from evaluation pipeline
    pipeline_requirements = "../evaluation-pipeline-2025/requirements.txt"
    if os.path.exists(pipeline_requirements):
        run_command(f"pip install -r {pipeline_requirements}", "Installing base evaluation pipeline requirements")
    else:
        print(f"Warning: {pipeline_requirements} not found, installing core packages manually")
        core_packages = [
            "transformers>=4.51.3",
            "torch>=2.7.0",
            "datasets>=3.6.0",
            "scikit-learn>=1.6.1",
            "numpy>=2.2.5",
            "pandas>=2.2.3",
            "statsmodels>=0.14.4",
            "nltk>=3.9.1",
            "wandb>=0.19.11",
            "huggingface-hub"
        ]
        for package in core_packages:
            run_command(f"pip install {package}", f"Installing {package}")

    # Step 3: Install model-specific dependencies
    print("\nInstalling model-specific dependencies...")

    # For DataDecide model (OLMo-based)
    success = run_command("pip install ai2-olmo", "Installing ai2-olmo for DataDecide model")
    if not success:
        print("Trying alternative installation for ai2-olmo...")
        run_command("pip install git+https://github.com/allenai/OLMo.git", "Installing OLMo from source")

    # For BitNet model (if special requirements)
    run_command("pip install accelerate", "Installing accelerate for model loading")
    run_command("pip install bitsandbytes", "Installing bitsandbytes for quantization support")

    # Step 4: Upgrade transformers to latest version
    run_command("pip install --upgrade transformers", "Upgrading transformers to latest version")

    # Step 5: Install additional utilities
    run_command("pip install einops", "Installing einops for tensor operations")

    print("\n" + "="*50)
    print("Setup completed!")
    print("\nNext steps:")
    print("1. Download models: python download_models.py --install_deps")
    print("2. Check evaluation data: python check_evaluation_data.py")
    print("3. Run evaluations: bash run_all_evaluations.sh")
    print("\nIf you still encounter model loading issues:")
    print("- For DataDecide model: pip install git+https://github.com/allenai/OLMo.git")
    print("- For transformers issues: pip install git+https://github.com/huggingface/transformers.git")

if __name__ == "__main__":
    main()
