#!/usr/bin/env python3
"""
BabyLM 2025 Compatible Model Evaluation Script
Evaluates only the models that are confirmed to be compatible.
"""

import os
import subprocess
import sys
from pathlib import Path

# Only compatible models
COMPATIBLE_MODELS = [
    {
        "name": "TinyLLama-v0-5M-F16",
        "hf_path": "mofosyne/TinyLLama-v0-5M-F16-llamafile",
        "architecture": "causal",
        "revision": "main",
        "model_type": "gguf",
        "gguf_file": "TinyLLama-4.6M-v0.0-F16.gguf",
        "description": "Small GGUF model - fully compatible"
    }
]

# Models with known issues (for reference)
PROBLEMATIC_MODELS = [
    {
        "name": "bitnet-b1.58-2B-4T",
        "hf_path": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "issue": "GGUF uses unsupported quantization type (np.uint32(36))",
        "potential_solution": "May require specialized BitNet tools or different GGUF version"
    },
    {
        "name": "DataDecide-dolma1_7-no-math-code-14M",
        "hf_path": "allenai/DataDecide-dolma1_7-no-math-code-14M",
        "issue": "hf_olmo architecture not supported in current transformers",
        "potential_solution": "Install AI2 OLMo library: pip install ai2-olmo"
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
    print(f"   Description: {model_config['description']}")

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

    return success

def main():
    """Main evaluation function."""
    print("🎯 BabyLM 2025 Compatible Model Evaluation")
    print("=" * 60)
    print("This script evaluates only models confirmed to be compatible.")
    print()

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

    # Show status of all models
    print("📋 MODEL STATUS SUMMARY")
    print("=" * 60)

    print("\n✅ COMPATIBLE MODELS (will be evaluated):")
    for model in COMPATIBLE_MODELS:
        print(f"  • {model['name']}: {model['description']}")

    print("\n❌ PROBLEMATIC MODELS (skipped):")
    for model in PROBLEMATIC_MODELS:
        print(f"  • {model['name']}")
        print(f"    Issue: {model['issue']}")
        print(f"    Potential solution: {model['potential_solution']}")

    print("\n" + "=" * 60)

    # Track evaluation results
    results = {}

    # Run evaluations for compatible models only
    for model in COMPATIBLE_MODELS:
        model_name = model["name"]
        results[model_name] = {}

        print(f"\n📊 Evaluating model: {model_name}")
        print(f"   HuggingFace path: {model['hf_path']}")
        print(f"   Architecture: {model['architecture']}")
        print(f"   Type: {model['model_type']}")

        # Run fast evaluation (for quick testing)
        print(f"\n🏃 Running FAST evaluation...")
        results[model_name]["fast"] = evaluate_model(model, "fast")

        # Run full evaluation (comprehensive)
        print(f"\n🔍 Running FULL evaluation...")
        results[model_name]["full"] = evaluate_model(model, "full")

        # Run fine-tuning evaluation (GLUE tasks)
        print(f"\n🎯 Running FINETUNE evaluation...")
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

    # Calculate success rate
    total_evaluations = sum(len(evals) for evals in results.values())
    successful_evaluations = sum(sum(evals.values()) for evals in results.values())
    success_rate = (successful_evaluations / total_evaluations * 100) if total_evaluations > 0 else 0

    print(f"\n📊 Overall Success Rate: {successful_evaluations}/{total_evaluations} ({success_rate:.1f}%)")
    print(f"\n📁 Results saved in: {os.path.abspath('results')}")

    if success_rate == 100:
        print("\n🎉 All compatible models evaluated successfully!")
    elif success_rate > 0:
        print(f"\n⚠️  Some evaluations failed. Check the output above for details.")
    else:
        print(f"\n❌ All evaluations failed. Check your setup and try again.")

    print("\n💡 Next Steps:")
    print("   • Check the results/ directory for detailed evaluation outputs")
    print("   • For problematic models, try the suggested solutions above")
    print("   • Consider adding more compatible models to the evaluation")

if __name__ == "__main__":
    main()
