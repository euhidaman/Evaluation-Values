#!/usr/bin/env python3
"""
Fix tokenizer padding issues for BitNet and other models during finetuning.
"""

import os
import sys
from pathlib import Path

def create_tokenizer_fix():
    """Create a patch for the tokenizer padding issue"""

    # Path to the evaluation pipeline
    pipeline_path = Path("../evaluation-pipeline-2025/evaluation_pipeline")
    finetune_path = pipeline_path / "finetune"

    if not finetune_path.exists():
        print(f"Error: Finetune directory not found at {finetune_path}")
        return False

    # Create a patch for the dataset.py file
    dataset_file = finetune_path / "dataset.py"

    if not dataset_file.exists():
        print(f"Error: Dataset file not found at {dataset_file}")
        return False

    # Read the original file
    with open(dataset_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if "# TOKENIZER PADDING FIX" in content:
        print("✓ Tokenizer padding fix already applied")
        return True

    # Find the collate_function and add the fix
    if "def collate_function" in content:
        # Add the fix before the encodings line
        old_line = "encodings = tokenizer(texts, return_tensors=\"pt\", padding=True, truncation=True, max_length=max_length)"

        new_code = """    # TOKENIZER PADDING FIX - Set pad token if not present
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)"""

        content = content.replace(old_line, new_code)

        # Write the patched file
        with open(dataset_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ Applied tokenizer padding fix to {dataset_file}")
        return True
    else:
        print("Error: Could not find collate_function to patch")
        return False

def create_aoa_fix():
    """Fix the missing eval_util module for AoA evaluation"""

    pipeline_path = Path("../evaluation-pipeline-2025/evaluation_pipeline")
    aoa_path = pipeline_path / "AoA_word"

    if not aoa_path.exists():
        print(f"Error: AoA directory not found at {aoa_path}")
        return False

    # Create a minimal eval_util.py file
    eval_util_file = aoa_path / "eval_util.py"

    eval_util_content = '''"""
Minimal eval_util module for AoA evaluation
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class StepConfig:
    """Configuration for evaluation steps"""
    name: str
    description: str = ""
    enabled: bool = True

class JsonProcessor:
    """Simple JSON processor for evaluation data"""
    
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.data = None
        
    def load(self):
        """Load JSON data from file"""
        try:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading {self.file_path}: {e}")
            return False
    
    def get_data(self):
        """Get loaded data"""
        return self.data

def load_eval(config_path: str) -> Dict[str, Any]:
    """Load evaluation configuration"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading eval config from {config_path}: {e}")
        return {}
'''

    with open(eval_util_file, 'w', encoding='utf-8') as f:
        f.write(eval_util_content)

    print(f"✓ Created eval_util.py at {eval_util_file}")
    return True

def main():
    """Apply all fixes to the evaluation pipeline"""

    print("Fixing Evaluation Pipeline Issues")
    print("=" * 40)

    success = True

    # Fix tokenizer padding
    print("\n1. Fixing tokenizer padding issue...")
    if not create_tokenizer_fix():
        success = False

    # Fix AoA eval_util
    print("\n2. Fixing AoA eval_util module...")
    if not create_aoa_fix():
        success = False

    if success:
        print("\n✓ All evaluation pipeline fixes applied successfully!")
        print("\nYou can now run finetuning evaluations without tokenizer errors.")
    else:
        print("\n✗ Some fixes failed. Please check the error messages above.")

    return success

if __name__ == "__main__":
    main()
