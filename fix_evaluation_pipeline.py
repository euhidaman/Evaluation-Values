#!/usr/bin/env python3
"""
Patch script to fix the KeyError: 'UID' issue in the evaluation pipeline.
This script modifies the run.py file to handle empty evaluation results gracefully
and adds JSON result storage functionality.
"""

import os
import shutil
import json
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

    # Add JSON result storage functionality
    old_save_section = '''    # Save predictions
    if args.save_predictions:
        save_predictions(args, predictions, best_temp)'''

    new_save_section = '''    # Save detailed results to JSON
    results_data = {
        "model_name": args.model_name,
        "task": args.task,
        "dataset": dataset,
        "backend": args.backend,
        "revision_name": revision_name,
        "best_temperature": best_temp,
        "best_accuracy": best_acc,
        "all_temperatures_results": {
            str(temp): acc for temp, acc in average_accuracies.items()
        },
        "detailed_accuracies": {
            str(temp): {
                subdomain: {str(k): float(v) for k, v in subdomain_accs.items()} 
                for subdomain, subdomain_accs in temp_accs.items()
            } for temp, temp_accs in accuracies.items()
        }
    }
    
    # Save results to JSON file
    results_json_path = args.output_path / "detailed_results.json"
    with open(results_json_path, 'w') as f:
        json.dump(results_data, f, indent=2)
    print(f"✓ Detailed results saved to: {results_json_path}")
    
    # Save predictions
    if args.save_predictions:
        save_predictions(args, predictions, best_temp)'''

    # Apply patches
    patches_applied = 0

    if old_code in content:
        content = content.replace(old_code, new_code)
        print("✓ Applied fix for non-entity_tracking tasks")
        patches_applied += 1
    else:
        print("Warning: Could not find exact match for non-entity_tracking code")

    if old_entity_code in content:
        content = content.replace(old_entity_code, new_entity_code)
        print("✓ Applied fix for entity_tracking task")
        patches_applied += 1
    else:
        print("Warning: Could not find exact match for entity_tracking code")

    if old_save_section in content:
        content = content.replace(old_save_section, new_save_section)
        print("✓ Added JSON result storage functionality")
        patches_applied += 1
    else:
        print("Warning: Could not find exact match for save section")

    # Add import for json at the top of the file if not already present
    if "import json" not in content:
        import_section = "import pathlib\nimport json\nimport argparse"
        if "import pathlib" in content:
            content = content.replace("import pathlib", import_section)
            print("✓ Added json import")
            patches_applied += 1

    # Write the patched file
    with open(run_file, 'w') as f:
        f.write(content)

    print(f"✓ Patched {run_file}")
    print(f"Applied {patches_applied} patches total")
    print("The evaluation pipeline will now:")
    print("  - Handle empty datasets gracefully")
    print("  - Save detailed results to detailed_results.json in each task's output directory")
    return True

if __name__ == "__main__":
    print("Applying patch to fix KeyError: 'UID' and add JSON result storage...")
    success = patch_evaluation_pipeline()
    if success:
        print("\n✓ Patch applied successfully!")
        print("You can now re-run the BitNet evaluation:")
        print("bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict-small causal")
        print("\nResults will be saved to:")
        print("  - Text reports: results/.../best_temperature_report.txt")
        print("  - JSON data: results/.../detailed_results.json")
    else:
        print("\n✗ Patch failed!")
