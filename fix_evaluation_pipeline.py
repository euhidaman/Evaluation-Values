#!/usr/bin/env python3
"""
Patch script to fix the KeyError: 'UID' issue in the evaluation pipeline.
This script modifies the run.py file to handle empty evaluation results gracefully.
"""

import os
import shutil
from pathlib import Path

def patch_evaluation_pipeline():
    """Apply the fix to the evaluation pipeline run.py file"""

    # Path to the evaluation pipeline file
    run_file = Path("/workspace/evaluation-pipeline-2025/evaluation_pipeline/sentence_zero_shot/run.py")

    if not run_file.exists():
        print(f"Error: {run_file} not found!")
        return False

    # Create backup
    backup_file = run_file.with_suffix('.py.backup')
    shutil.copy2(run_file, backup_file)
    print(f"Created backup: {backup_file}")

    # Read the current file
    with open(run_file, 'r') as f:
        content = f.read()

    # Find the problematic code and replace it
    old_code = '''    # Average accuracies
    average_accuracies = {}
    if args.task != "entity_tracking":
        for temp, accuracy in accuracies.items():
            average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())'''

    new_code = '''    # Average accuracies
    average_accuracies = {}
    if args.task != "entity_tracking":
        for temp, accuracy in accuracies.items():
            # Handle case where no results were generated (empty dataset)
            if "UID" not in accuracy or not accuracy["UID"]:
                print(f"Warning: No results found for temperature {temp} in task {args.task}. Setting accuracy to 0.")
                average_accuracies[temp] = 0.0
            else:
                average_accuracies[temp] = sum(accuracy["UID"].values()) / len(accuracy["UID"].values())'''

    # Also fix the entity_tracking section
    old_entity_code = '''    else:
        splits = ["regular", "ambiref", "move_contents"]
        for temp, subdomain_dict in accuracies.items():
            split_accs = []
            split_dict = subdomain_dict["UID"]'''

    new_entity_code = '''    else:
        splits = ["regular", "ambiref", "move_contents"]
        for temp, subdomain_dict in accuracies.items():
            # Handle case where no results were generated for entity_tracking
            if "UID" not in subdomain_dict or not subdomain_dict["UID"]:
                print(f"Warning: No results found for temperature {temp} in entity_tracking task. Setting accuracy to 0.")
                average_accuracies[temp] = 0.0
                continue
                
            split_accs = []
            split_dict = subdomain_dict["UID"]'''

    # Apply patches
    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✓ Applied fix for non-entity_tracking tasks")
    else:
        print("Warning: Could not find exact match for non-entity_tracking code")

    if old_entity_code in content:
        content = content.replace(old_entity_code, new_entity_code)
        print("✓ Applied fix for entity_tracking task")
    else:
        print("Warning: Could not find exact match for entity_tracking code")

    # Write the patched file
    with open(run_file, 'w') as f:
        f.write(content)

    print(f"✓ Patched {run_file}")
    print("The evaluation pipeline should now handle empty datasets gracefully.")
    return True

if __name__ == "__main__":
    print("Applying patch to fix KeyError: 'UID' in evaluation pipeline...")
    success = patch_evaluation_pipeline()
    if success:
        print("\n✓ Patch applied successfully!")
        print("You can now re-run the BitNet evaluation:")
        print("bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict-small causal")
    else:
        print("\n✗ Patch failed!")
