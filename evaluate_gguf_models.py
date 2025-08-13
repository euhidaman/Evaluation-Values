#!/usr/bin/env python3
"""
BabyLM 2025 GGUF Model Evaluation Script
Evaluates GGUF models directly using HuggingFace transformers native GGUF support.
"""

import os
import subprocess
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Model configurations for GGUF evaluation
MODELS = [
    {
        "name": "TinyLLama-v0-5M-F16",
        "hf_path": "mofosyne/TinyLLama-v0-5M-F16-llamafile",
        "architecture": "causal",
        "revision": "main",
        "model_type": "gguf"
    },
    {
        "name": "bitnet-b1.58-2B-4T",
        "hf_path": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "architecture": "causal",
        "revision": "main",
        "model_type": "gguf"
    },
    {
        "name": "DataDecide-dolma1_7-no-math-code-14M",
        "hf_path": "allenai/DataDecide-dolma1_7-no-math-code-14M",
        "architecture": "causal",
        "revision": "main",
        "model_type": "standard"
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

def test_gguf_model_loading(model_config):
    """Test if a GGUF model can be loaded with native transformers support."""
    print(f"\n🔍 Testing GGUF model: {model_config['name']}")
    print(f"   HuggingFace path: {model_config['hf_path']}")

    try:
        # Check if model has GGUF files
        from huggingface_hub import hf_hub_download, list_repo_files

        files = list_repo_files(model_config['hf_path'], revision=model_config['revision'])
        gguf_files = [f for f in files if f.endswith('.gguf')]

        print(f"   📁 GGUF files found: {gguf_files}")

        if not gguf_files:
            print(f"   ⚠️  No GGUF files found, treating as standard model")
            return test_standard_model_loading(model_config)

        # Try to load tokenizer with GGUF support
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['hf_path'],
            revision=model_config['revision'],
            trust_remote_code=True,
            gguf_file=gguf_files[0] if len(gguf_files) == 1 else None
        )
        print(f"   ✅ GGUF tokenizer loaded successfully")

        # Try to load model with GGUF support
        model = AutoModelForCausalLM.from_pretrained(
            model_config['hf_path'],
            revision=model_config['revision'],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True,
            gguf_file=gguf_files[0] if len(gguf_files) == 1 else None
        )
        print(f"   ✅ GGUF model loaded successfully")
        print(f"   📊 Model size: ~{model.num_parameters() / 1e6:.1f}M parameters")

        # Test a simple forward pass
        test_input = tokenizer("Hello world", return_tensors="pt")
        with torch.no_grad():
            output = model(**test_input)
        print(f"   ✅ GGUF model forward pass successful")

        # Clean up
        del model, tokenizer, test_input, output
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True, "GGUF compatible"

    except Exception as e:
        print(f"   ❌ GGUF loading failed: {str(e)}")
        print(f"   🔄 Trying standard loading method...")
        return test_standard_model_loading(model_config)

def test_standard_model_loading(model_config):
    """Test standard HuggingFace model loading."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['hf_path'],
            revision=model_config['revision'],
            trust_remote_code=True
        )
        print(f"   ✅ Standard tokenizer loaded successfully")

        model = AutoModelForCausalLM.from_pretrained(
            model_config['hf_path'],
            revision=model_config['revision'],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",
            low_cpu_mem_usage=True
        )
        print(f"   ✅ Standard model loaded successfully")
        print(f"   📊 Model size: ~{model.num_parameters() / 1e6:.1f}M parameters")

        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True, "Standard compatible"

    except Exception as e:
        print(f"   ❌ Standard loading failed: {str(e)}")
        return False, str(e)

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

    # Set environment variables for GGUF support if needed
    env_vars = ""
    if model_config.get("model_type") == "gguf":
        env_vars = "TRANSFORMERS_VERBOSITY=info "

    if eval_type == "fast":
        cmd = f"{env_vars}{EVAL_TYPES[eval_type]['script']} {model_path} {revision} {architecture} {EVAL_TYPES[eval_type]['data_dir']}"
    elif eval_type == "full":
        cmd = f"{env_vars}{EVAL_TYPES[eval_type]['script']} {model_path} {architecture} {EVAL_TYPES[eval_type]['data_dir']}"
    elif eval_type == "finetune":
        cmd = f"{env_vars}{EVAL_TYPES[eval_type]['script']} {model_path}"
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
    print("🎯 BabyLM 2025 GGUF Model Evaluation")
    print("=" * 60)

    # Check transformers version for GGUF support
    import transformers
    print(f"📦 Transformers version: {transformers.__version__}")

    # Check for GGUF support
    try:
        from transformers.utils import is_gguf_available
        if is_gguf_available():
            print("✅ Native GGUF support available")
        else:
            print("⚠️  GGUF support not available, install with: pip install gguf")
    except ImportError:
        print("⚠️  GGUF utilities not found")

    # Check if we have GPU available
    if torch.cuda.is_available():
        print(f"🎯 GPU Available: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  No GPU detected - evaluation will be slow")

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

    # Test model compatibility first
    print("\n" + "=" * 60)
    print("🔬 TESTING MODEL COMPATIBILITY")
    print("=" * 60)

    compatible_models = []
    for model in MODELS:
        if model.get("model_type") == "gguf":
            is_compatible, message = test_gguf_model_loading(model)
        else:
            is_compatible, message = test_standard_model_loading(model)

        if is_compatible:
            model['status'] = message
            compatible_models.append(model)
            print(f"   ✅ {model['name']}: {message}")
        else:
            print(f"   ❌ {model['name']}: {message}")

    if not compatible_models:
        print("❌ No compatible models found!")
        sys.exit(1)

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Track evaluation results
    results = {}

    # Run evaluations for each compatible model
    print("\n" + "=" * 60)
    print("🚀 STARTING EVALUATIONS")
    print("=" * 60)

    for model in compatible_models:
        model_name = model["name"]
        results[model_name] = {}

        print(f"\n📊 Evaluating model: {model_name}")
        print(f"   HuggingFace path: {model['hf_path']}")
        print(f"   Architecture: {model['architecture']}")
        print(f"   Status: {model['status']}")

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
    print("\n🎉 GGUF evaluation pipeline completed!")

if __name__ == "__main__":
    main()
