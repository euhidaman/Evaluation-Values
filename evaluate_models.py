#!/usr/bin/env python3
"""
BabyLM 2025 Text-Only Model Evaluation Script
Evaluates three specified models on the text-only strict track.
"""

import os
import subprocess
import sys
from pathlib import Path

# Model configurations
MODELS = [
    {
        "name": "TinyLLama-v0-5M-F16",
        "hf_path": "mofosyne/TinyLLama-v0-5M-F16-llamafile",
        "architecture": "causal",
        "revision": "main",
        "model_type": "gguf",
        "gguf_file": "TinyLLama-4.6M-v0.0-F16.gguf",
        "compatible": True
    },
    {
        "name": "bitnet-b1.58-2B-4T",
        "hf_path": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "architecture": "causal",
        "revision": "main",
        "model_type": "gguf",
        "gguf_file": "ggml-model-i2_s.gguf",
        "compatible": False,
        "note": "GGUF uses unsupported quantization type (np.uint32(36)). May work with specialized GGUF tools."
    },
    {
        "name": "DataDecide-dolma1_7-no-math-code-14M",
        "hf_path": "allenai/DataDecide-dolma1_7-no-math-code-14M",
        "architecture": "causal",
        "revision": "main",
        "model_type": "olmo",
        "requires_trust_remote_code": True,
        "compatible": False,
        "note": "Requires newer transformers version with hf_olmo support. Consider upgrading transformers or using AI2 OLMo library."
    }
]

# Evaluation configurations
EVAL_TYPES = {
    "fast": {
        "script": "./eval_zero_shot_fast.sh",
        "data_dir": "../evaluation_data/fast_eval"
    },
    "full": {
        "script": "./eval_zero_shot.sh",
        "data_dir": "../evaluation_data/full_eval"
    },
    "finetune": {
        "script": "./eval_finetuning.sh",
        "data_dir": "../evaluation_data/full_eval"
    }
}

def run_command(cmd, description):
    """Run a shell command with error handling."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: {description} failed!")
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")
            return False
        else:
            print(f"SUCCESS: {description} completed!")
            if result.stdout:
                print(f"OUTPUT: {result.stdout}")
            return True
    except Exception as e:
        print(f"EXCEPTION: {description} failed with exception: {e}")
        return False

def evaluate_model(model_config, eval_type="fast"):
    """Evaluate a single model with specified evaluation type."""
    model_name = model_config["name"]
    model_path = model_config["hf_path"]
    architecture = model_config["architecture"]
    revision = model_config["revision"]

    print(f"\n🚀 Starting {eval_type} evaluation for {model_name}")

    # Check if model is known to be incompatible
    if not model_config.get("compatible", True):
        print(f"⚠️  WARNING: {model_name} has known compatibility issues:")
        print(f"   {model_config.get('note', 'Unknown compatibility issue')}")
        print(f"   Attempting evaluation anyway - the evaluation pipeline may handle it differently...")

    if eval_type == "fast":
        cmd = f"{EVAL_TYPES[eval_type]['script']} {model_path} {revision} {architecture} {EVAL_TYPES[eval_type]['data_dir']}"
    elif eval_type == "full":
        cmd = f"{EVAL_TYPES[eval_type]['script']} {model_path} {architecture} {EVAL_TYPES[eval_type]['data_dir']}"
    elif eval_type == "finetune":
        cmd = f"{EVAL_TYPES[eval_type]['script']} {model_path}"
    else:
        print(f"Unknown evaluation type: {eval_type}")
        return False

    success = run_command(cmd, f"{eval_type} evaluation for {model_name}")

    if success:
        print(f"✅ {eval_type} evaluation completed for {model_name}")
    else:
        print(f"❌ {eval_type} evaluation failed for {model_name}")
        if not model_config.get("compatible", True):
            print(f"   This failure may be due to the known compatibility issue mentioned above.")

    return success

def main():
    """Main evaluation function."""
    print("🎯 BabyLM 2025 Text-Only Model Evaluation")
    print("=" * 60)

    # Check if we're in the right directory
    if not os.path.exists("evaluation_pipeline"):
        print("ERROR: evaluation_pipeline directory not found!")
        print("Please run this script from the evaluation-pipeline-2025 root directory.")
        sys.exit(1)

    # Check if evaluation data exists
    if not os.path.exists("../evaluation_data"):
        print("ERROR: evaluation_data directory not found!")
        print("Please ensure evaluation_data is downloaded and placed in the parent directory.")
        sys.exit(1)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Track evaluation results
    results = {}

    # Run evaluations for each model
    for model in MODELS:
        model_name = model["name"]
        results[model_name] = {}

        print(f"\n📊 Evaluating model: {model_name}")
        print(f"   HuggingFace path: {model['hf_path']}")
        print(f"   Architecture: {model['architecture']}")

        # Run fast evaluation (for quick testing)
        results[model_name]["fast"] = evaluate_model(model, "fast")

        # Run full evaluation (comprehensive)
        results[model_name]["full"] = evaluate_model(model, "full")

        # Run fine-tuning evaluation (GLUE tasks)
        results[model_name]["finetune"] = evaluate_model(model, "finetune")

    # Print summary
    print("\n" + "=" * 60)
    print("📋 EVALUATION SUMMARY")
    print("=" * 60)

    for model_name, evals in results.items():
        print(f"\n{model_name}:")
        for eval_type, success in evals.items():
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"  {eval_type.capitalize()}: {status}")

    print(f"\n📁 Results saved in: {os.path.abspath('results')}")
    print("\n🎉 Evaluation pipeline completed!")

if __name__ == "__main__":
    main()
