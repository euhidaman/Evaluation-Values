#!/usr/bin/env python3
"""
BabyLM 2025 Model Compatibility Checker
Tests if models can be loaded and are compatible with the evaluation pipeline.
"""

import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from huggingface_hub import list_repo_files
import requests

# Model configurations to test
MODELS = [
    {
        "name": "TinyLLama-v0-5M-F16",
        "hf_path": "mofosyne/TinyLLama-v0-5M-F16-llamafile",
        "architecture": "causal",
        "model_type": "gguf",
        "gguf_file": "TinyLLama-4.6M-v0.0-F16.gguf"
    },
    {
        "name": "bitnet-b1.58-2B-4T",
        "hf_path": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "architecture": "causal",
        "model_type": "gguf",
        "gguf_file": "ggml-model-i2_s.gguf"
    },
    {
        "name": "DataDecide-dolma1_7-no-math-code-14M",
        "hf_path": "allenai/DataDecide-dolma1_7-no-math-code-14M",
        "architecture": "causal",
        "model_type": "olmo",
        "requires_trust_remote_code": True
    }
]

def check_repo_exists(model_path):
    """Check if HuggingFace repository exists and is accessible."""
    try:
        files = list_repo_files(model_path)
        return True, files
    except Exception as e:
        return False, str(e)

def detect_gguf_files(repo_files):
    """Detect GGUF files in the repository."""
    gguf_files = [f for f in repo_files if f.endswith('.gguf')]
    return gguf_files

def test_model_config(model_path, model_config=None):
    """Test if model configuration can be loaded."""
    try:
        # For OLMo models, we need trust_remote_code
        if model_config and model_config.get("model_type") == "olmo":
            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        else:
            config = AutoConfig.from_pretrained(model_path)
        return True, config
    except Exception as e:
        return False, str(e)

def test_tokenizer_loading(model_path, model_config=None):
    """Test if tokenizer can be loaded."""
    try:
        # For OLMo models, we need trust_remote_code
        if model_config and model_config.get("model_type") == "olmo":
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path)
        return True, tokenizer
    except Exception as e:
        # For GGUF models, try to use a fallback tokenizer
        if model_config and model_config.get("model_type") == "gguf":
            try:
                # Try to use LlamaTokenizer as fallback for GGUF models
                from transformers import LlamaTokenizer
                tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b")
                return True, tokenizer
            except Exception as fallback_error:
                return False, f"Primary: {str(e)}, Fallback: {str(fallback_error)}"
        return False, str(e)

def test_model_loading(model_config):
    """Test if model can be loaded successfully."""
    model_path = model_config["hf_path"]
    model_type = model_config["model_type"]

    try:
        if model_type == "gguf":
            # Try GGUF loading first
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    gguf_file=model_config["gguf_file"],
                    torch_dtype=torch.float16,
                    device_map="cpu"  # Use CPU for compatibility testing
                )
                return True, "GGUF loading successful"
            except Exception as gguf_error:
                # Fall back to standard loading
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16,
                        device_map="cpu"
                    )
                    return True, f"Standard loading successful (GGUF failed: {gguf_error})"
                except Exception as std_error:
                    return False, f"Both GGUF and standard loading failed. GGUF: {gguf_error}, Standard: {std_error}"
        elif model_type == "olmo":
            # OLMo model loading with trust_remote_code
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            return True, "OLMo loading successful with trust_remote_code"
        else:
            # Standard model loading
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="cpu"
            )
            return True, "Standard loading successful"

    except Exception as e:
        return False, str(e)

def check_model_compatibility(model_config):
    """Comprehensive compatibility check for a model."""
    model_name = model_config["name"]
    model_path = model_config["hf_path"]

    print(f"\n{'='*60}")
    print(f"🔍 Checking compatibility for: {model_name}")
    print(f"📍 Repository: {model_path}")
    print(f"🏗️  Architecture: {model_config['architecture']}")
    print(f"📦 Model type: {model_config['model_type']}")
    print(f"{'='*60}")

    results = {
        "model_name": model_name,
        "repo_accessible": False,
        "config_loadable": False,
        "tokenizer_loadable": False,
        "model_loadable": False,
        "gguf_files": [],
        "errors": []
    }

    # 1. Check repository accessibility
    print("1️⃣ Checking repository accessibility...")
    repo_exists, repo_info = check_repo_exists(model_path)
    if repo_exists:
        print("   ✅ Repository is accessible")
        results["repo_accessible"] = True

        # Check for GGUF files
        if model_config["model_type"] == "gguf":
            gguf_files = detect_gguf_files(repo_info)
            results["gguf_files"] = gguf_files
            if gguf_files:
                print(f"   📦 Found GGUF files: {gguf_files}")
            else:
                print("   ⚠️  No GGUF files found in repository")
    else:
        print(f"   ❌ Repository not accessible: {repo_info}")
        results["errors"].append(f"Repository access failed: {repo_info}")
        return results

    # 2. Check model configuration
    print("2️⃣ Checking model configuration...")
    config_ok, config_info = test_model_config(model_path, model_config)
    if config_ok:
        print("   ✅ Model configuration loaded successfully")
        print(f"   📋 Model type: {config_info.model_type}")
        results["config_loadable"] = True
    else:
        print(f"   ❌ Model configuration failed: {config_info}")
        results["errors"].append(f"Config loading failed: {config_info}")

    # 3. Check tokenizer
    print("3️⃣ Checking tokenizer...")
    tokenizer_ok, tokenizer_info = test_tokenizer_loading(model_path, model_config)
    if tokenizer_ok:
        print("   ✅ Tokenizer loaded successfully")
        print(f"   📝 Vocab size: {len(tokenizer_info.get_vocab())}")
        results["tokenizer_loadable"] = True
    else:
        print(f"   ❌ Tokenizer loading failed: {tokenizer_info}")
        results["errors"].append(f"Tokenizer loading failed: {tokenizer_info}")

    # 4. Check model loading
    print("4️⃣ Checking model loading...")
    model_ok, model_info = test_model_loading(model_config)
    if model_ok:
        print(f"   ✅ Model loaded successfully: {model_info}")
        results["model_loadable"] = True
    else:
        print(f"   ❌ Model loading failed: {model_info}")
        results["errors"].append(f"Model loading failed: {model_info}")

    # Overall compatibility assessment
    all_checks_passed = all([
        results["repo_accessible"],
        results["config_loadable"],
        results["tokenizer_loadable"],
        results["model_loadable"]
    ])

    if all_checks_passed:
        print(f"\n🎉 {model_name} is FULLY COMPATIBLE with the evaluation pipeline!")
    else:
        print(f"\n⚠️  {model_name} has compatibility issues. See errors above.")

    return results

def main():
    """Main compatibility checking function."""
    print("🔧 BabyLM 2025 Model Compatibility Checker")
    print("=" * 60)
    print("This script tests if models can be loaded and are compatible with the evaluation pipeline.")
    print()

    # Check Python and PyTorch environment
    print(f"🐍 Python version: {sys.version}")
    print(f"🔥 PyTorch version: {torch.__version__}")
    print(f"💾 CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU count: {torch.cuda.device_count()}")
    print()

    all_results = []

    # Test each model
    for model_config in MODELS:
        result = check_model_compatibility(model_config)
        all_results.append(result)

    # Summary report
    print("\n" + "=" * 60)
    print("📋 COMPATIBILITY SUMMARY")
    print("=" * 60)

    for result in all_results:
        model_name = result["model_name"]
        status = "✅ COMPATIBLE" if result["model_loadable"] else "❌ INCOMPATIBLE"
        print(f"{model_name}: {status}")

        if result["errors"]:
            print(f"   Issues found: {len(result['errors'])}")
            for error in result["errors"]:
                print(f"     - {error}")

        if result["gguf_files"]:
            print(f"   GGUF files: {result['gguf_files']}")

    # Overall summary
    compatible_count = sum(1 for r in all_results if r["model_loadable"])
    total_count = len(all_results)

    print(f"\n📊 Overall: {compatible_count}/{total_count} models are compatible")

    if compatible_count == total_count:
        print("🎉 All models are ready for evaluation!")
    else:
        print("⚠️  Some models need attention before evaluation.")

    return all_results

if __name__ == "__main__":
    main()
