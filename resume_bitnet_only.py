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

# Initialize real-time results saving
python -c "
from realtime_results_saver import start_model_evaluation
start_model_evaluation('microsoft--bitnet-b1.58-2B-4T', 'strict')
print('✓ Results tracking initialized for BitNet strict track')
"

MODEL_NAME="microsoft--bitnet-b1.58-2B-4T"
MODEL_ABS_PATH="/workspace/Evaluation-Values/models/microsoft--bitnet-b1.58-2B-4T"
BACKEND="causal"

# Helper function to save task results
save_task_result() {
    local track=$1
    local task_type=$2
    local task_name=$3
    local output_file=$4
    
    cd /workspace/Evaluation-Values
    python -c "
import sys
from realtime_results_saver import save_task_result

# Read output file
try:
    with open('$output_file', 'r', encoding='utf-8') as f:
        output_text = f.read()
except:
    output_text = ''

save_task_result('$MODEL_NAME', '$track', '$task_type', '$task_name', {}, output_text)
print('✓ Saved $task_type results for $task_name')
"
}

# BitNet model setup
echo "Setting up BitNet model files..."
python bitnet_wrapper.py

cleanup_bitnet() {
    echo "Cleaning up BitNet temporary files..."
    cd /workspace/Evaluation-Values
    python bitnet_wrapper.py cleanup
}
trap cleanup_bitnet EXIT

# Set evaluation data directory
EVAL_DIR="../evaluation_data/full_eval"

"""

    # Check which tasks need to be run for each track
    for track in ['strict', 'strict-small']:
        track_tasks = remaining_tasks[track]
        total_tasks = sum(len(tasks) for tasks in track_tasks.values())

        if total_tasks > 0:
            script_content += f"""
echo ""
echo "Processing BitNet {track} track - {total_tasks} tasks remaining"
echo "=" * 60

# Create results directory for {track} track
RESULTS_DIR="/workspace/Evaluation-Values/results/$MODEL_NAME/{track}"
mkdir -p "$RESULTS_DIR"

cd ../evaluation-pipeline-2025

"""

            # Add zero-shot tasks
            if track_tasks['zero_shot']:
                script_content += f"""
echo "Running remaining zero-shot evaluations for {track} track..."
"""

                for task in track_tasks['zero_shot']:
                    if task == 'blimp':
                        script_content += f"""
echo "Running BLiMP evaluation..."
OUTPUT_FILE="$RESULTS_DIR/blimp_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \\
    --model_path_or_name "$MODEL_ABS_PATH" \\
    --backend $BACKEND \\
    --task blimp \\
    --data_path "${{EVAL_DIR}}/blimp_filtered" \\
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"
save_task_result "{track}" "zero_shot" "blimp" "$OUTPUT_FILE"
"""
                    elif task == 'blimp_supplement':
                        script_content += f"""
echo "Running BLiMP Supplement evaluation..."
OUTPUT_FILE="$RESULTS_DIR/blimp_supplement_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \\
    --model_path_or_name "$MODEL_ABS_PATH" \\
    --backend $BACKEND \\
    --task blimp \\
    --data_path "${{EVAL_DIR}}/supplement_filtered" \\
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"
save_task_result "{track}" "zero_shot" "blimp_supplement" "$OUTPUT_FILE"
"""
                    elif task == 'ewok':
                        script_content += f"""
echo "Running EWoK evaluation..."
OUTPUT_FILE="$RESULTS_DIR/ewok_output.log"
if [ -d "${{EVAL_DIR}}/ewok_filtered" ] && [ "$(ls -A ${{EVAL_DIR}}/ewok_filtered 2>/dev/null)" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \\
        --model_path_or_name "$MODEL_ABS_PATH" \\
        --backend $BACKEND \\
        --task ewok \\
        --data_path "${{EVAL_DIR}}/ewok_filtered" \\
        --save_predictions 2>&1 | tee "$OUTPUT_FILE"
else
    echo "EWoK data not found or empty, skipping..." | tee "$OUTPUT_FILE"
fi
cd /workspace/Evaluation-Values
save_task_result "{track}" "zero_shot" "ewok" "$OUTPUT_FILE"
cd ../evaluation-pipeline-2025
"""
                    elif task == 'entity_tracking':
                        script_content += f"""
echo "Running Entity Tracking evaluation..."
OUTPUT_FILE="$RESULTS_DIR/entity_tracking_output.log"
if [ -d "${{EVAL_DIR}}/entity_tracking" ] && [ "$(ls -A ${{EVAL_DIR}}/entity_tracking 2>/dev/null)" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \\
        --model_path_or_name "$MODEL_ABS_PATH" \\
        --backend $BACKEND \\
        --task entity_tracking \\
        --data_path "${{EVAL_DIR}}/entity_tracking" \\
        --save_predictions 2>&1 | tee "$OUTPUT_FILE"
else
    echo "Entity Tracking data not found or empty, skipping..." | tee "$OUTPUT_FILE"
fi
cd /workspace/Evaluation-Values
save_task_result "{track}" "zero_shot" "entity_tracking" "$OUTPUT_FILE"
cd ../evaluation-pipeline-2025
"""
                    elif task == 'wug_adj':
                        script_content += f"""
echo "Running WUG Adjective Nominalization evaluation..."
OUTPUT_FILE="$RESULTS_DIR/wug_adj_output.log"
if [ -d "${{EVAL_DIR}}/wug_adj_nominalization" ] && [ "$(ls -A ${{EVAL_DIR}}/wug_adj_nominalization 2>/dev/null)" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \\
        --model_path_or_name "$MODEL_ABS_PATH" \\
        --backend $BACKEND \\
        --task wug_adj \\
        --data_path "${{EVAL_DIR}}/wug_adj_nominalization" \\
        --save_predictions 2>&1 | tee "$OUTPUT_FILE"
else
    echo "WUG Adjective Nominalization data not found or empty, skipping..." | tee "$OUTPUT_FILE"
fi
cd /workspace/Evaluation-Values
save_task_result "{track}" "zero_shot" "wug_adj" "$OUTPUT_FILE"
cd ../evaluation-pipeline-2025
"""
                    elif task == 'wug_past':
                        script_content += f"""
echo "Running WUG Past Tense evaluation..."
OUTPUT_FILE="$RESULTS_DIR/wug_past_output.log"
if [ -d "${{EVAL_DIR}}/wug_past_tense" ] && [ "$(ls -A ${{EVAL_DIR}}/wug_past_tense 2>/dev/null)" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \\
        --model_path_or_name "$MODEL_ABS_PATH" \\
        --backend $BACKEND \\
        --task wug_past \\
        --data_path "${{EVAL_DIR}}/wug_past_tense" \\
        --save_predictions 2>&1 | tee "$OUTPUT_FILE"
else
    echo "WUG Past Tense data not found or empty, skipping..." | tee "$OUTPUT_FILE"
fi
cd /workspace/Evaluation-Values
save_task_result "{track}" "zero_shot" "wug_past" "$OUTPUT_FILE"
cd ../evaluation-pipeline-2025
"""
                    elif task == 'aoa':
                        script_content += f"""
echo "Running Age of Acquisition evaluation..."
OUTPUT_FILE="$RESULTS_DIR/aoa_output.log"
if [ -f "${{EVAL_DIR}}/cdi_childes/cdi_childes.json" ]; then
    python -m evaluation_pipeline.AoA_word.run \\
        --model_name "$MODEL_ABS_PATH" \\
        --backend $BACKEND \\
        --track_name {track} \\
        --word_path "${{EVAL_DIR}}/cdi_childes/cdi_childes.json" \\
        --output_dir "results" 2>&1 | tee "$OUTPUT_FILE"
else
    echo "CDI data not found, skipping AoA evaluation..." | tee "$OUTPUT_FILE"
fi
cd /workspace/Evaluation-Values
save_task_result "{track}" "zero_shot" "aoa" "$OUTPUT_FILE"
cd ../evaluation-pipeline-2025
"""

            # Add reading tasks
            if track_tasks['reading']:
                script_content += f"""
echo "Running reading evaluation for {track} track..."
OUTPUT_FILE="$RESULTS_DIR/reading_output.log"
if [ -f "${{EVAL_DIR}}/reading/reading_data.csv" ]; then
    python -m evaluation_pipeline.reading.run \\
        --model_path_or_name "$MODEL_ABS_PATH" \\
        --backend $BACKEND \\
        --data_path "${{EVAL_DIR}}/reading/reading_data.csv" 2>&1 | tee "$OUTPUT_FILE"
else
    echo "Reading data not found, skipping..." | tee "$OUTPUT_FILE"
fi
cd /workspace/Evaluation-Values
save_task_result "{track}" "reading" "reading_tasks" "$OUTPUT_FILE"
cd ../evaluation-pipeline-2025
"""

            # Add finetuning tasks (only for strict track)
            if track == 'strict' and track_tasks['finetuning']:
                script_content += f"""
echo "Running finetuning evaluations for {track} track..."
echo "Applying comprehensive fix before finetuning..."
cd /workspace/Evaluation-Values
python comprehensive_fix.py
cd ../evaluation-pipeline-2025

# GLUE task parameters
LR=3e-5
BSZ=32
BIG_BSZ=16
MAX_EPOCHS=10
WSC_EPOCHS=30
SEED=42
"""

                for task in track_tasks['finetuning']:
                    if task in ['boolq', 'multirc']:
                        script_content += f"""
echo "Finetuning on {task}..."
OUTPUT_FILE="$RESULTS_DIR/finetune_{task}_output.log"
python -m evaluation_pipeline.finetune.run \\
    --model_name_or_path "$MODEL_ABS_PATH" \\
    --train_data "${{EVAL_DIR}}/glue_filtered/{task}.train.jsonl" \\
    --valid_data "${{EVAL_DIR}}/glue_filtered/{task}.valid.jsonl" \\
    --predict_data "${{EVAL_DIR}}/glue_filtered/{task}.valid.jsonl" \\
    --task "{task}" \\
    --num_labels 2 \\
    --batch_size $BIG_BSZ \\
    --learning_rate $LR \\
    --num_epochs $MAX_EPOCHS \\
    --sequence_length 512 \\
    --results_dir "results" \\
    --save \\
    --save_dir "models" \\
    --metrics accuracy f1 mcc \\
    --metric_for_valid accuracy \\
    --seed $SEED \\
    --verbose 2>&1 | tee "$OUTPUT_FILE"
save_task_result "{track}" "finetuning" "{task}" "$OUTPUT_FILE"
"""
                    elif task == 'wsc':
                        script_content += f"""
echo "Finetuning on WSC..."
OUTPUT_FILE="$RESULTS_DIR/finetune_wsc_output.log"
python -m evaluation_pipeline.finetune.run \\
    --model_name_or_path "$MODEL_ABS_PATH" \\
    --train_data "${{EVAL_DIR}}/glue_filtered/wsc.train.jsonl" \\
    --valid_data "${{EVAL_DIR}}/glue_filtered/wsc.valid.jsonl" \\
    --predict_data "${{EVAL_DIR}}/glue_filtered/wsc.valid.jsonl" \\
    --task "wsc" \\
    --num_labels 2 \\
    --batch_size $BSZ \\
    --learning_rate $LR \\
    --num_epochs $WSC_EPOCHS \\
    --sequence_length 512 \\
    --results_dir "results" \\
    --save \\
    --save_dir "models" \\
    --metrics accuracy f1 mcc \\
    --metric_for_valid accuracy \\
    --seed $SEED \\
    --verbose 2>&1 | tee "$OUTPUT_FILE"
save_task_result "{track}" "finetuning" "wsc" "$OUTPUT_FILE"
"""
                    elif task in ['mrpc', 'qqp']:
                        script_content += f"""
echo "Finetuning on {task}..."
OUTPUT_FILE="$RESULTS_DIR/finetune_{task}_output.log"
python -m evaluation_pipeline.finetune.run \\
    --model_name_or_path "$MODEL_ABS_PATH" \\
    --train_data "${{EVAL_DIR}}/glue_filtered/{task}.train.jsonl" \\
    --valid_data "${{EVAL_DIR}}/glue_filtered/{task}.valid.jsonl" \\
    --predict_data "${{EVAL_DIR}}/glue_filtered/{task}.valid.jsonl" \\
    --task "{task}" \\
    --num_labels 2 \\
    --batch_size $BSZ \\
    --learning_rate $LR \\
    --num_epochs $MAX_EPOCHS \\
    --sequence_length 512 \\
    --results_dir "results" \\
    --save \\
    --save_dir "models" \\
    --metrics accuracy f1 mcc \\
    --metric_for_valid f1 \\
    --seed $SEED \\
    --verbose 2>&1 | tee "$OUTPUT_FILE"
save_task_result "{track}" "finetuning" "{task}" "$OUTPUT_FILE"
"""
                    elif task in ['rte', 'mnli']:
                        num_labels = 3 if task == 'mnli' else 2
                        script_content += f"""
echo "Finetuning on {task}..."
OUTPUT_FILE="$RESULTS_DIR/finetune_{task}_output.log"
python -m evaluation_pipeline.finetune.run \\
    --model_name_or_path "$MODEL_ABS_PATH" \\
    --train_data "${{EVAL_DIR}}/glue_filtered/{task}.train.jsonl" \\
    --valid_data "${{EVAL_DIR}}/glue_filtered/{task}.valid.jsonl" \\
    --predict_data "${{EVAL_DIR}}/glue_filtered/{task}.valid.jsonl" \\
    --task "{task}" \\
    --num_labels {num_labels} \\
    --batch_size $BSZ \\
    --learning_rate $LR \\
    --num_epochs $MAX_EPOCHS \\
    --sequence_length 512 \\
    --results_dir "results" \\
    --save \\
    --save_dir "models" \\
    --metrics accuracy f1 mcc \\
    --metric_for_valid accuracy \\
    --seed $SEED \\
    --verbose 2>&1 | tee "$OUTPUT_FILE"
save_task_result "{track}" "finetuning" "{task}" "$OUTPUT_FILE"
"""

            script_content += f"""
# Mark {track} track as complete
cd /workspace/Evaluation-Values
python -c "
from realtime_results_saver import complete_model_evaluation
complete_model_evaluation('$MODEL_NAME', '{track}', '$BACKEND')
print('✓ Completed BitNet {track} track evaluation')
"
"""

    # Check if no tasks remain
    total_remaining = sum(sum(len(tasks) for tasks in track_tasks.values()) for track_tasks in remaining_tasks.values())

    if total_remaining == 0:
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
print('✓ BitNet evaluation completed')
print()
print('✓ Results saved to: evaluation_results.json')
"
"""
    else:
        script_content += f"""
echo ""
echo "✓ Completed BitNet resume script - all remaining tasks processed"
echo "✓ Results continuously saved to: evaluation_results.json"
echo "✓ Individual task results saved after each evaluation"
"""

    # Write the script
    with open("resume_bitnet.sh", "w", encoding='utf-8') as f:
        f.write(script_content)

    os.chmod("resume_bitnet.sh", 0o755)
    print(f"\n✓ Created resume_bitnet.sh script")
    print(f"✓ Will process {total_remaining} remaining BitNet tasks")
    print("✓ Results will be saved after each individual evaluation")

    return total_remaining

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
