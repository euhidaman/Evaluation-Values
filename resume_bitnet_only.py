#!/usr/bin/env python3
"""
Resume evaluation script for BitNet model only.
This script checks what's already been completed for BitNet and runs only the remaining tasks.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def check_bitnet_completed_evaluations():
    """Check what BitNet evaluations have already been completed"""

    results_file = Path("evaluation_results.json")
    results_dir = Path("results")

    bitnet_model = "microsoft--bitnet-b1.58-2B-4T"
    completed_tasks = {
        'strict': {'zero_shot': [], 'finetuning': [], 'reading': []},
        'strict-small': {'zero_shot': [], 'finetuning': [], 'reading': []}
    }

    print("Checking BitNet evaluation status...")
    print("=" * 50)

    # Check if results file exists
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print("Found existing evaluation_results.json")

            if "model_results" in data and bitnet_model in data["model_results"]:
                model_data = data["model_results"][bitnet_model]

                for track_name in ['strict', 'strict-small']:
                    if track_name in model_data:
                        track_data = model_data[track_name]
                        print(f"\nBitNet - {track_name} track:")

                        # Check zero-shot tasks
                        if "zero_shot" in track_data:
                            for task_name, task_data in track_data["zero_shot"].items():
                                if task_data.get("status") == "completed":
                                    completed_tasks[track_name]['zero_shot'].append(task_name)
                                    print(f"  ✓ Zero-shot: {task_name}")

                        # Check finetuning tasks
                        if "finetuning" in track_data:
                            for task_name, task_data in track_data["finetuning"].items():
                                if task_data.get("status") == "completed":
                                    completed_tasks[track_name]['finetuning'].append(task_name)
                                    print(f"  ✓ Finetuning: {task_name}")

                        # Check reading tasks
                        if "reading" in track_data:
                            for task_name, task_data in track_data["reading"].items():
                                if task_data.get("status") == "completed":
                                    completed_tasks[track_name]['reading'].append(task_name)
                                    print(f"  ✓ Reading: {task_name}")

        except Exception as e:
            print(f"Error reading evaluation_results.json: {e}")
    else:
        print("No evaluation_results.json found - checking results directory")

    # Also check results directory structure for BitNet
    bitnet_results_dir = results_dir / bitnet_model
    if bitnet_results_dir.exists():
        print(f"\nFound BitNet results directory:")

        for track_name in ['strict', 'strict-small']:
            track_dir = bitnet_results_dir / track_name
            if track_dir.exists():
                print(f"\nBitNet - {track_name} track (from log files):")

                # Check for log files to determine completed tasks
                for log_file in track_dir.glob("*.log"):
                    task_name = log_file.stem

                    # Categorize tasks
                    if "finetune_" in task_name:
                        task_category = 'finetuning'
                        clean_name = task_name.replace("finetune_", "").replace("_output", "")
                    elif task_name in ["reading_output"]:
                        task_category = 'reading'
                        clean_name = "reading_tasks"
                    else:
                        task_category = 'zero_shot'
                        clean_name = task_name.replace("_output", "")

                    if clean_name not in completed_tasks[track_name][task_category]:
                        completed_tasks[track_name][task_category].append(clean_name)
                        print(f"  ✓ {task_category.capitalize()}: {clean_name}")

    return completed_tasks

def get_bitnet_remaining_tasks():
    """Get list of BitNet tasks that still need to be completed"""

    # Define all expected tasks
    all_tasks = {
        'zero_shot': [
            'blimp',
            'blimp_supplement',
            'ewok',
            'entity_tracking',
            'wug_adj',
            'wug_past',
            'comps',
            'aoa'
        ],
        'finetuning': [
            'boolq',
            'multirc',
            'rte',
            'wsc',
            'mrpc',
            'qqp',
            'mnli'
        ],
        'reading': [
            'reading_tasks'
        ]
    }

    completed_tasks = check_bitnet_completed_evaluations()
    remaining_tasks = {
        'strict': {'zero_shot': [], 'finetuning': [], 'reading': []},
        'strict-small': {'zero_shot': [], 'finetuning': [], 'reading': []}
    }

    print(f"\n" + "=" * 50)
    print("REMAINING BITNET TASKS TO COMPLETE")
    print("=" * 50)

    for track in ['strict', 'strict-small']:
        print(f"\nBitNet - {track} track:")

        for task_type, task_list in all_tasks.items():
            # Skip finetuning for strict-small track
            if task_type == 'finetuning' and track == 'strict-small':
                continue

            for task in task_list:
                if task not in completed_tasks[track][task_type]:
                    remaining_tasks[track][task_type].append(task)

        # Print remaining tasks for this track
        total_remaining = (len(remaining_tasks[track]['zero_shot']) +
                         len(remaining_tasks[track]['finetuning']) +
                         len(remaining_tasks[track]['reading']))

        if total_remaining > 0:
            for task_type, tasks in remaining_tasks[track].items():
                if tasks:
                    print(f"  {task_type.capitalize()}: {', '.join(tasks)}")
        else:
            print(f"  ✓ COMPLETED - All tasks done!")

    return remaining_tasks

def create_bitnet_resume_script(remaining_tasks):
    """Create a script to resume BitNet evaluation from where it left off"""

    script_content = """#!/bin/bash

# Resume BabyLM 2025 BitNet Evaluation
# This script continues BitNet evaluation from where it left off

echo "Resuming BabyLM 2025 BitNet Model Evaluation"
echo "==========================================="

cd /workspace/Evaluation-Values

# Apply comprehensive fixes first
echo "Applying evaluation pipeline fixes..."
python comprehensive_fix.py

"""

    evaluation_count = 0

    for track, task_types in remaining_tasks.items():
        total_tasks = sum(len(tasks) for tasks in task_types.values())

        if total_tasks > 0:
            evaluation_count += 1

            script_content += f"""
echo ""
echo "Resuming BitNet evaluation on {track} track"
echo "Tasks remaining: {total_tasks}"
echo "Backend: causal"
bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T {track} causal

if [ $? -eq 0 ]; then
    echo "✓ Successfully completed {track} evaluation for BitNet"
else
    echo "✗ Failed {track} evaluation for BitNet"
fi
"""

    if evaluation_count == 0:
        script_content += """
echo "🎉 BitNet evaluation is already completed!"
echo "No remaining tasks to run."

# Show final summary
python -c "
from realtime_results_saver import get_summary
summary = get_summary()
print()
print('BITNET EVALUATION SUMMARY')
print('=' * 30)
if 'microsoft--bitnet-b1.58-2B-4T' in summary.get('models_evaluated', []):
    print('✓ BitNet evaluation completed')
else:
    print('? BitNet evaluation status unknown')
print()
print('✓ Results saved to: evaluation_results.json')
"
"""
    else:
        script_content += f"""
echo ""
echo "Completed BitNet resume script - {evaluation_count} track(s) processed"
echo "Final results available in: evaluation_results.json"
"""

    # Write the script
    with open("resume_bitnet.sh", "w", encoding='utf-8') as f:
        f.write(script_content)

    os.chmod("resume_bitnet.sh", 0o755)
    print(f"\n✓ Created resume_bitnet.sh script")
    print(f"✓ {evaluation_count} BitNet track(s) will be processed")

    return evaluation_count

def main():
    """Main function to check BitNet status and create resume script"""

    print("BabyLM 2025 BitNet Evaluation Resume Tool")
    print("=" * 45)

    # Check what's been completed for BitNet
    remaining_tasks = get_bitnet_remaining_tasks()

    # Create resume script
    remaining_count = create_bitnet_resume_script(remaining_tasks)

    print(f"\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)

    if remaining_count > 0:
        print("To continue your BitNet evaluation, run:")
        print("bash resume_bitnet.sh")
        print()
        print("This will:")
        print("- Apply all necessary fixes")
        print("- Run only the remaining BitNet evaluations")
        print("- Save results incrementally to evaluation_results.json")
        print("- Skip any BitNet tasks that are already completed")
    else:
        print("🎉 BitNet evaluation is complete!")
        print("Check evaluation_results.json for BitNet results")

if __name__ == "__main__":
    main()
