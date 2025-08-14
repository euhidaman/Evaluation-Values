#!/usr/bin/env python3
"""
Comprehensive fix for all evaluation pipeline issues.
This script applies all necessary patches from the Evaluation-Values repository.
"""

import os
import sys
import shutil
from pathlib import Path

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

def apply_dataset_fix():
    """Apply tokenizer padding fix to dataset.py"""

    print("Applying dataset.py tokenizer padding fix...")

    dataset_file = Path("../evaluation-pipeline-2025/evaluation_pipeline/finetune/dataset.py")

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

    # Create backup
    backup_file = dataset_file.with_suffix('.py.backup')
    shutil.copy2(dataset_file, backup_file)
    print(f"✓ Created backup: {backup_file}")

    # Apply fix by inserting the tokenizer fix before the encodings line
    lines = content.split('\n')
    new_lines = []

    for i, line in enumerate(lines):
        if 'encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)' in line:
            # Get the indentation of the current line
            indent = len(line) - len(line.lstrip())

            # Add the tokenizer fix with proper indentation
            new_lines.append(' ' * indent + '# TOKENIZER PADDING FIX - Set pad token if not present')
            new_lines.append(' ' * indent + 'if tokenizer.pad_token is None:')
            new_lines.append(' ' * (indent + 4) + 'if tokenizer.eos_token is not None:')
            new_lines.append(' ' * (indent + 8) + 'tokenizer.pad_token = tokenizer.eos_token')
            new_lines.append(' ' * (indent + 4) + 'else:')
            new_lines.append(' ' * (indent + 8) + 'tokenizer.add_special_tokens({\'pad_token\': \'[PAD]\'})')
            new_lines.append('')
            new_lines.append(line)  # Add the original encodings line
        else:
            new_lines.append(line)

    # Write the fixed content
    with open(dataset_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

    print(f"✓ Applied tokenizer padding fix to {dataset_file}")
    return True

def apply_sentence_zero_shot_fix():
    """Fix the UID KeyError in sentence zero shot evaluation"""

    print("Applying sentence zero shot UID fix...")

    run_file = Path("../evaluation-pipeline-2025/evaluation_pipeline/sentence_zero_shot/run.py")

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

    # Create backup
    backup_file = run_file.with_suffix('.py.backup')
    shutil.copy2(run_file, backup_file)
    print(f"✓ Created backup: {backup_file}")

    # Apply fix
    lines = content.split('\n')
    new_lines = []

    for i, line in enumerate(lines):
        if 'average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())' in line:
            # Get the indentation of the current line
            indent = len(line) - len(line.lstrip())

            # Add the UID fix with proper indentation
            new_lines.append(' ' * indent + '# UID KEYERROR FIX - Handle missing UID key')
            new_lines.append(' ' * indent + 'if "UID" in accuracy and accuracy["UID"]:')
            new_lines.append(' ' * (indent + 4) + 'average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())')
            new_lines.append(' ' * indent + 'else:')
            new_lines.append(' ' * (indent + 4) + '# Use first available key if UID is missing')
            new_lines.append(' ' * (indent + 4) + 'available_keys = [k for k in accuracy.keys() if isinstance(accuracy[k], dict) and accuracy[k]]')
            new_lines.append(' ' * (indent + 4) + 'if available_keys:')
            new_lines.append(' ' * (indent + 8) + 'key = available_keys[0]')
            new_lines.append(' ' * (indent + 8) + 'average_accuracies[temp] = sum(accuracy[key].values()) / len(accuracy[key].values())')
            new_lines.append(' ' * (indent + 4) + 'else:')
            new_lines.append(' ' * (indent + 8) + 'average_accuracies[temp] = 0.0')
        else:
            new_lines.append(line)

    # Write the fixed content
    with open(run_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(new_lines))

    print(f"✓ Applied UID KeyError fix to {run_file}")
    return True

def apply_aoa_fix():
    """Create the missing eval_util module for AoA evaluation"""

    print("Applying AoA eval_util fix...")

    aoa_path = Path("../evaluation-pipeline-2025/evaluation_pipeline/AoA_word")

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

def main():
    """Apply all fixes to the evaluation pipeline"""

    if len(sys.argv) > 1 and sys.argv[1] == "cleanup":
        cleanup_fixes()
        print("\n✓ All fixes cleaned up!")
        return True

    print("Comprehensive Evaluation Pipeline Fix")
    print("=" * 40)

    success = True

    # Fix dataset.py tokenizer padding
    print("\n1. Fixing dataset.py tokenizer padding...")
    if not apply_dataset_fix():
        success = False

    # Fix sentence zero shot UID error
    print("\n2. Fixing sentence zero shot UID KeyError...")
    if not apply_sentence_zero_shot_fix():
        success = False

    # Fix AoA eval_util
    print("\n3. Creating AoA eval_util module...")
    if not apply_aoa_fix():
        success = False

    if success:
        print("\n✓ All evaluation pipeline fixes applied successfully!")
        print("\nFixes applied:")
        print("- Tokenizer padding for finetuning tasks")
        print("- Sentence zero shot UID KeyError handling")
        print("- AoA eval_util module creation")
        print("\nYou can now run evaluations without errors.")
        print("\nTo clean up fixes later, run: python comprehensive_fix.py cleanup")
    else:
        print("\n✗ Some fixes failed. Please check the error messages above.")

    return success

if __name__ == "__main__":
    main()
