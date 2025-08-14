#!/usr/bin/env python3
"""
Resume evaluation script - continues from where previous evaluations left off.
This script checks what's already been completed and runs only the remaining tasks.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

def check_completed_evaluations():
    """Check what evaluations have already been completed"""

    results_file = Path("evaluation_results.json")
    results_dir = Path("results")

    completed_tasks = {}

    # Check if results file exists
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            print("Found existing evaluation_results.json")
            print("=" * 50)

            if "model_results" in data:
                for model_name, model_data in data["model_results"].items():
                    print(f"\nModel: {model_name}")
                    completed_tasks[model_name] = {}

                    for track_name, track_data in model_data.items():
                        print(f"  Track: {track_name}")
                        completed_tasks[model_name][track_name] = {
                            'zero_shot': [],
                            'finetuning': [],
                            'reading': []
                        }

                        # Check zero-shot tasks
                        if "zero_shot" in track_data:
                            for task_name, task_data in track_data["zero_shot"].items():
                                if task_data.get("status") == "completed":
                                    completed_tasks[model_name][track_name]['zero_shot'].append(task_name)
                                    print(f"    ✓ Zero-shot: {task_name}")

                        # Check finetuning tasks
                        if "finetuning" in track_data:
                            for task_name, task_data in track_data["finetuning"].items():
                                if task_data.get("status") == "completed":
                                    completed_tasks[model_name][track_name]['finetuning'].append(task_name)
                                    print(f"    ✓ Finetuning: {task_name}")

                        # Check reading tasks
                        if "reading" in track_data:
                            for task_name, task_data in track_data["reading"].items():
                                if task_data.get("status") == "completed":
                                    completed_tasks[model_name][track_name]['reading'].append(task_name)
                                    print(f"    ✓ Reading: {task_name}")

        except Exception as e:
            print(f"Error reading evaluation_results.json: {e}")
    else:
        print("No evaluation_results.json found - will check results directory")

    # Also check results directory structure
    if results_dir.exists():
        print(f"\nFound results directory with the following structure:")
        for model_dir in results_dir.iterdir():
            if model_dir.is_dir():
                print(f"\nModel: {model_dir.name}")
                if model_dir.name not in completed_tasks:
                    completed_tasks[model_dir.name] = {}

                for track_dir in model_dir.iterdir():
                    if track_dir.is_dir():
                        print(f"  Track: {track_dir.name}")
                        if track_dir.name not in completed_tasks[model_dir.name]:
                            completed_tasks[model_dir.name][track_dir.name] = {
                                'zero_shot': [],
                                'finetuning': [],
                                'reading': []
                            }

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

                            if clean_name not in completed_tasks[model_dir.name][track_dir.name][task_category]:
                                completed_tasks[model_dir.name][track_dir.name][task_category].append(clean_name)
                                print(f"    ✓ {task_category.capitalize()}: {clean_name}")

    return completed_tasks

def get_remaining_tasks():
    """Get list of tasks that still need to be completed"""

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

    # Expected models and tracks
    expected_models = [
        'microsoft--bitnet-b1.58-2B-4T',
        'allenai--DataDecide-dolma1_7-no-math-code-14M',
        'Xuandong--HPD-TinyBERT-F128'
    ]

    expected_tracks = ['strict', 'strict-small']

    completed_tasks = check_completed_evaluations()
    remaining_tasks = {}

    print(f"\n" + "=" * 50)
    print("REMAINING TASKS TO COMPLETE")
    print("=" * 50)

    for model in expected_models:
        remaining_tasks[model] = {}

        for track in expected_tracks:
            remaining_tasks[model][track] = {
                'zero_shot': [],
                'finetuning': [],
                'reading': []
            }

            # Get completed tasks for this model/track
            completed_for_track = {}
            if model in completed_tasks and track in completed_tasks[model]:
                completed_for_track = completed_tasks[model][track]
            else:
                completed_for_track = {'zero_shot': [], 'finetuning': [], 'reading': []}

            # Find remaining tasks
            for task_type, task_list in all_tasks.items():
                # Skip finetuning for strict-small track
                if task_type == 'finetuning' and track == 'strict-small':
                    continue

                for task in task_list:
                    if task not in completed_for_track.get(task_type, []):
                        remaining_tasks[model][track][task_type].append(task)

            # Print remaining tasks for this model/track
            total_remaining = (len(remaining_tasks[model][track]['zero_shot']) +
                             len(remaining_tasks[model][track]['finetuning']) +
                             len(remaining_tasks[model][track]['reading']))

            if total_remaining > 0:
                print(f"\n{model} - {track} track:")
                for task_type, tasks in remaining_tasks[model][track].items():
                    if tasks:
                        print(f"  {task_type.capitalize()}: {', '.join(tasks)}")
            else:
                print(f"\n{model} - {track} track: ✓ COMPLETED")

    return remaining_tasks

def create_resume_script(remaining_tasks):
    """Create a script to resume evaluation from where it left off"""

    script_content = """#!/bin/bash

# Resume BabyLM 2025 Evaluations
# This script continues evaluation from where it left off

echo "Resuming BabyLM 2025 Model Evaluations"
echo "======================================"

cd /workspace/Evaluation-Values

# Apply comprehensive fixes first
echo "Applying evaluation pipeline fixes..."
python comprehensive_fix.py

"""

    # Define model backends
    model_backends = {
        'microsoft--bitnet-b1.58-2B-4T': 'causal',
        'allenai--DataDecide-dolma1_7-no-math-code-14M': 'causal',
        'Xuandong--HPD-TinyBERT-F128': 'masked'
    }

    evaluation_count = 0

    for model, tracks in remaining_tasks.items():
        for track, task_types in tracks.items():
            total_tasks = sum(len(tasks) for tasks in task_types.values())

            if total_tasks > 0:
                evaluation_count += 1
                backend = model_backends.get(model, 'causal')

                script_content += f"""
echo ""
echo "Resuming evaluation: {model} on {track} track"
echo "Tasks remaining: {total_tasks}"
echo "Backend: {backend}"
bash evaluate_model.sh ./models/{model} {model} {track} {backend}

if [ $? -eq 0 ]; then
    echo "✓ Successfully completed {track} evaluation for {model}"
else
    echo "✗ Failed {track} evaluation for {model}"
fi
"""

    if evaluation_count == 0:
        script_content += """
echo "🎉 All evaluations are already completed!"
echo "No remaining tasks to run."

# Show final summary
python -c "
from realtime_results_saver import get_summary
summary = get_summary()
print()
print('EVALUATION SUMMARY')
print('=' * 30)
print(f'Total models: {summary[\"total_models\"]}')
print(f'Total tasks: {summary[\"total_tasks\"]}')
print(f'Completed tasks: {summary[\"completed_tasks\"]}')
print(f'Success rate: {summary[\"progress_percentage\"]:.1f}%')
print()
print('✓ All results saved to: evaluation_results.json')
"
"""
    else:
        script_content += f"""
echo ""
echo "Completed resume script - {evaluation_count} model/track combinations processed"
echo "Final results available in: evaluation_results.json"
"""

    # Write the script
    with open("resume_evaluation.sh", "w", encoding='utf-8') as f:
        f.write(script_content)

    os.chmod("resume_evaluation.sh", 0o755)
    print(f"\n✓ Created resume_evaluation.sh script")
    print(f"✓ {evaluation_count} model/track combinations will be processed")

    return evaluation_count

def main():
    """Main function to check status and create resume script"""

    print("BabyLM 2025 Evaluation Resume Tool")
    print("=" * 40)

    # Check what's been completed
    remaining_tasks = get_remaining_tasks()

    # Create resume script
    remaining_count = create_resume_script(remaining_tasks)

    print(f"\n" + "=" * 50)
    print("NEXT STEPS")
    print("=" * 50)

    if remaining_count > 0:
        print("To continue your evaluations, run:")
        print("bash resume_evaluation.sh")
        print()
        print("This will:")
        print("- Apply all necessary fixes")
        print("- Run only the remaining evaluations")
        print("- Save results incrementally to evaluation_results.json")
        print("- Skip any tasks that are already completed")
    else:
        print("🎉 All evaluations are complete!")
        print("Check evaluation_results.json for final results")

if __name__ == "__main__":
    main()
