#!/usr/bin/env python3
"""
Fix tokenizer padding issues for BitNet and other models during finetuning.
This script patches the evaluation pipeline from the Evaluation-Values repository.
"""

import os
import sys
import shutil
from pathlib import Path

def apply_tokenizer_fix():
    """Apply tokenizer padding fix to the evaluation pipeline"""

    print("Applying tokenizer padding fix...")

    # Path to the evaluation pipeline
    pipeline_path = Path("../evaluation-pipeline-2025/evaluation_pipeline")
    finetune_path = pipeline_path / "finetune"

    if not finetune_path.exists():
        print(f"Error: Finetune directory not found at {finetune_path}")
        return False

    # Target file to patch
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

    # Find and replace the tokenizer call
    old_pattern = 'encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)'

    new_pattern = '''        # TOKENIZER PADDING FIX - Set pad token if not present
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            else:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        
        encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)'''

    if old_pattern in content:
        # Create backup
        backup_file = dataset_file.with_suffix('.py.backup')
        shutil.copy2(dataset_file, backup_file)
        print(f"✓ Created backup: {backup_file}")

        # Apply fix
        content = content.replace(old_pattern, new_pattern)

        # Write the patched file
        with open(dataset_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ Applied tokenizer padding fix to {dataset_file}")
        return True
    else:
        print("Error: Could not find the tokenizer call pattern to patch")
        print("The evaluation pipeline might have been updated")
        return False

def apply_aoa_fix():
    """Create the missing eval_util module for AoA evaluation"""

    print("Applying AoA eval_util fix...")

    pipeline_path = Path("../evaluation-pipeline-2025/evaluation_pipeline")
    aoa_path = pipeline_path / "AoA_word"

    if not aoa_path.exists():
        print(f"Error: AoA directory not found at {aoa_path}")
        return False

    # Create eval_util.py file
    eval_util_file = aoa_path / "eval_util.py"

    # Check if already exists
    if eval_util_file.exists():
        print("✓ eval_util.py already exists")
        return True

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

def apply_sentence_zero_shot_fix():
    """Fix the UID KeyError in sentence zero shot evaluation"""

    print("Applying sentence zero shot UID fix...")

    pipeline_path = Path("../evaluation-pipeline-2025/evaluation_pipeline")
    sentence_path = pipeline_path / "sentence_zero_shot"

    if not sentence_path.exists():
        print(f"Error: Sentence zero shot directory not found at {sentence_path}")
        return False

    # Target file to patch
    run_file = sentence_path / "run.py"

    if not run_file.exists():
        print(f"Error: Run file not found at {run_file}")
        return False

    # Read the original file
    with open(run_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already patched
    if "# UID KEYERROR FIX" in content:
        print("✓ UID KeyError fix already applied")
        return True

    # Find and replace the problematic line with proper indentation
    old_pattern = '            average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())'

    new_pattern = '''            # UID KEYERROR FIX - Handle missing UID key
            if "UID" in accuracy and accuracy["UID"]:
                average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())
            else:
                # Use first available key if UID is missing
                available_keys = [k for k in accuracy.keys() if isinstance(accuracy[k], dict) and accuracy[k]]
                if available_keys:
                    key = available_keys[0]
                    average_accuracies[temp] = sum(accuracy[key].values()) / len(accuracy[key].values())
                else:
                    average_accuracies[temp] = 0.0'''

    if old_pattern in content:
        # Create backup
        backup_file = run_file.with_suffix('.py.backup')
        shutil.copy2(run_file, backup_file)
        print(f"✓ Created backup: {backup_file}")

        # Apply fix
        content = content.replace(old_pattern, new_pattern)

        # Write the patched file
        with open(run_file, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"✓ Applied UID KeyError fix to {run_file}")
        return True
    else:
        print("Warning: Could not find the UID KeyError pattern to patch")
        print("The file content might have changed. Attempting alternative fix...")

        # Alternative pattern with different spacing
        alt_pattern = 'average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())'

        if alt_pattern in content:
            # Find the exact indentation by looking at surrounding lines
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if alt_pattern.strip() in line:
                    # Get the indentation from the current line
                    indent = len(line) - len(line.lstrip())

                    # Create properly indented replacement
                    new_lines = [
                        ' ' * indent + '# UID KEYERROR FIX - Handle missing UID key',
                        ' ' * indent + 'if "UID" in accuracy and accuracy["UID"]:',
                        ' ' * (indent + 4) + 'average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())',
                        ' ' * indent + 'else:',
                        ' ' * (indent + 4) + '# Use first available key if UID is missing',
                        ' ' * (indent + 4) + 'available_keys = [k for k in accuracy.keys() if isinstance(accuracy[k], dict) and accuracy[k]]',
                        ' ' * (indent + 4) + 'if available_keys:',
                        ' ' * (indent + 8) + 'key = available_keys[0]',
                        ' ' * (indent + 8) + 'average_accuracies[temp] = sum(accuracy[key].values()) / len(accuracy[key].values())',
                        ' ' * (indent + 4) + 'else:',
                        ' ' * (indent + 8) + 'average_accuracies[temp] = 0.0'
                    ]

                    # Replace the line
                    lines[i] = '\n'.join(new_lines)

                    # Create backup
                    backup_file = run_file.with_suffix('.py.backup')
                    shutil.copy2(run_file, backup_file)
                    print(f"✓ Created backup: {backup_file}")

                    # Write the fixed content
                    with open(run_file, 'w', encoding='utf-8') as f:
                        f.write('\n'.join(lines))

                    print(f"✓ Applied UID KeyError fix to {run_file}")
                    return True

        print("Error: Could not find any pattern to fix")
        return False

def cleanup_fixes():
    """Remove applied fixes and restore backups"""

    print("Cleaning up applied fixes...")

    # Files to clean up
    files_to_restore = [
        "../evaluation-pipeline-2025/evaluation_pipeline/finetune/dataset.py",
        "../evaluation-pipeline-2025/evaluation_pipeline/sentence_zero_shot/run.py"
    ]

    files_to_remove = [
        "../evaluation-pipeline-2025/evaluation_pipeline/AoA_word/eval_util.py"
    ]

    # Restore from backups
    for file_path in files_to_restore:
        file_path = Path(file_path)
        backup_path = file_path.with_suffix('.py.backup')

        if backup_path.exists():
            shutil.copy2(backup_path, file_path)
            backup_path.unlink()
            print(f"✓ Restored {file_path} from backup")

    # Remove created files
    for file_path in files_to_remove:
        file_path = Path(file_path)
        if file_path.exists():
            file_path.unlink()
            print(f"✓ Removed {file_path}")

def main():
    """Apply all fixes to the evaluation pipeline"""

    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_fixes()
        print("\n✓ All fixes cleaned up!")
        return

    print("Fixing Evaluation Pipeline Issues")
    print("=" * 40)

    success = True

    # Fix tokenizer padding
    print("\n1. Fixing tokenizer padding issue...")
    if not apply_tokenizer_fix():
        success = False

    # Fix AoA eval_util
    print("\n2. Fixing AoA eval_util module...")
    if not apply_aoa_fix():
        success = False

    # Fix sentence zero shot UID error
    print("\n3. Fixing sentence zero shot UID KeyError...")
    if not apply_sentence_zero_shot_fix():
        success = False

    if success:
        print("\n✓ All evaluation pipeline fixes applied successfully!")
        print("\nFixes applied:")
        print("- Tokenizer padding for finetuning tasks")
        print("- AoA eval_util module creation")
        print("- Sentence zero shot UID KeyError handling")
        print("\nYou can now run finetuning evaluations without errors.")
        print("\nTo clean up fixes later, run: python fix_tokenizer_padding.py cleanup")
    else:
        print("\n✗ Some fixes failed. Please check the error messages above.")

    return success

if __name__ == "__main__":
    main()
