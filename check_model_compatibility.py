#!/usr/bin/env python3
"""
Pre-evaluation setup script to check model compatibility and download models.
"""

import os
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Model configurations with compatibility checks
MODELS = [
    {
        "name": "TinyLLama-v0-5M-F16",
        "hf_path": "mofosyne/TinyLLama-v0-5M-F16-llamafile",
        "architecture": "causal",
        "revision": "main",
        "status": "checking"
    },
    {
        "name": "bitnet-b1.58-2B-4T",
        "hf_path": "microsoft/bitnet-b1.58-2B-4T-gguf",
        "architecture": "causal",
        "revision": "main",
        "status": "gguf_format",
        "note": "GGUF format - may need conversion"
    },
    {
        "name": "DataDecide-dolma1_7-no-math-code-14M",
        "hf_path": "allenai/DataDecide-dolma1_7-no-math-code-14M",
        "architecture": "causal",
        "revision": "main",
        "status": "checking"
    }
]

def check_model_compatibility(model_config):
    """Check if a model is compatible with HuggingFace transformers."""
    print(f"\n🔍 Checking model: {model_config['name']}")
    print(f"   HuggingFace path: {model_config['hf_path']}")

    try:
        # Try to load tokenizer first (lightweight check)
        tokenizer = AutoTokenizer.from_pretrained(
            model_config['hf_path'],
            revision=model_config['revision'],
            trust_remote_code=True
        )
        print(f"   ✅ Tokenizer loaded successfully")

        # Try to load model config (without downloading weights)
        model = AutoModelForCausalLM.from_pretrained(
            model_config['hf_path'],
            revision=model_config['revision'],
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="cpu",  # Don't load to GPU yet
            low_cpu_mem_usage=True
        )
        print(f"   ✅ Model structure loaded successfully")
        print(f"   📊 Model size: ~{model.num_parameters() / 1e6:.1f}M parameters")

        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        return True, "Compatible"

    except Exception as e:
        print(f"   ❌ Error loading model: {str(e)}")
        return False, str(e)

def check_alternative_models():
    """Suggest alternative models if some are incompatible."""
    print("\n🔄 Checking for alternative compatible models...")

    alternative_models = [
        {
            "name": "TinyLlama-1.1B-Chat",
            "hf_path": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "architecture": "causal",
            "note": "Reliable small model"
        },
        {
            "name": "microsoft-DialoGPT-small",
            "hf_path": "microsoft/DialoGPT-small",
            "architecture": "causal",
            "note": "Small Microsoft model"
        },
        {
            "name": "gpt2",
            "hf_path": "gpt2",
            "architecture": "causal",
            "note": "Standard baseline model"
        }
    ]

    compatible_alternatives = []
    for model in alternative_models:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model['hf_path'], trust_remote_code=True)
            compatible_alternatives.append(model)
            print(f"   ✅ Alternative available: {model['name']}")
            del tokenizer
        except:
            print(f"   ❌ Alternative not available: {model['name']}")

    return compatible_alternatives

def main():
    """Main model compatibility checking function."""
    print("🚀 BabyLM 2025 Model Compatibility Checker")
    print("=" * 60)

    # Check if we have GPU available
    if torch.cuda.is_available():
        print(f"🎯 GPU Available: {torch.cuda.get_device_name()}")
        print(f"📊 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    else:
        print("⚠️  No GPU detected - evaluation will be slow")

    # Check each model
    compatible_models = []
    incompatible_models = []

    for model in MODELS:
        is_compatible, message = check_model_compatibility(model)

        if is_compatible:
            compatible_models.append(model)
        else:
            model['error'] = message
            incompatible_models.append(model)

    # Print summary
    print("\n" + "=" * 60)
    print("📋 COMPATIBILITY SUMMARY")
    print("=" * 60)

    if compatible_models:
        print(f"\n✅ Compatible Models ({len(compatible_models)}):")
        for model in compatible_models:
            print(f"   • {model['name']}")

    if incompatible_models:
        print(f"\n❌ Incompatible Models ({len(incompatible_models)}):")
        for model in incompatible_models:
            print(f"   • {model['name']}: {model['error']}")

        # Suggest alternatives
        alternatives = check_alternative_models()
        if alternatives:
            print(f"\n🔄 Suggested Alternative Models:")
            for alt in alternatives:
                print(f"   • {alt['name']} ({alt['hf_path']}) - {alt['note']}")

    print(f"\n🎉 Compatibility check completed!")
    return len(compatible_models) > 0

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
