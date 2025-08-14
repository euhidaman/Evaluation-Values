#!/bin/bash

# BabyLM 2025 Evaluation Script for Text-Only Models
# This script evaluates models on both strict and strict-small tracks

MODEL_DIR=$1
MODEL_NAME=$2
TRACK=$3  # "strict" or "strict-small"
BACKEND=${4:-"causal"}  # default to causal for decoder models

if [ -z "$MODEL_DIR" ] || [ -z "$MODEL_NAME" ] || [ -z "$TRACK" ]; then
    echo "Usage: $0 <model_dir> <model_name> <track> [backend]"
    echo "  model_dir: Path to the downloaded model directory"
    echo "  model_name: Name for organizing results (e.g., bitnet-b1.58-2B-4T)"
    echo "  track: 'strict' or 'strict-small'"
    echo "  backend: 'causal' (default), 'enc_dec', or 'masked'"
    exit 1
fi

# Convert to absolute path
MODEL_ABS_PATH=$(realpath "$MODEL_DIR")

# Check if model directory exists
if [ ! -d "$MODEL_ABS_PATH" ]; then
    echo "Error: Model directory does not exist: $MODEL_ABS_PATH"
    exit 1
fi

# Check if model has required files
if [ ! -f "$MODEL_ABS_PATH/config.json" ]; then
    echo "Error: config.json not found in model directory: $MODEL_ABS_PATH"
    echo "Please ensure the model was downloaded correctly."
    exit 1
fi

# Initialize real-time results saving
echo "Initializing evaluation for $MODEL_NAME on $TRACK track..."
cd /workspace/Evaluation-Values
python -c "
from realtime_results_saver import start_model_evaluation
start_model_evaluation('$MODEL_NAME', '$TRACK')
print('✓ Results tracking initialized')
"

# Helper function to save task results
save_task_result() {
    local task_type=$1
    local task_name=$2
    local output_file=$3

    if [ -f "$output_file" ]; then
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

save_task_result('$MODEL_NAME', '$TRACK', '$task_type', '$task_name', {}, output_text)
"
        cd ../evaluation-pipeline-2025
    else
        # Save with empty output if file doesn't exist
        cd /workspace/Evaluation-Values
        python -c "
from realtime_results_saver import save_task_result
save_task_result('$MODEL_NAME', '$TRACK', '$task_type', '$task_name', {}, '')
"
        cd ../evaluation-pipeline-2025
    fi
}

# Special setup for BitNet model
if [[ "$MODEL_NAME" == *"bitnet"* ]] || [[ "$MODEL_DIR" == *"bitnet"* ]]; then
    echo "Detected BitNet model, setting up custom files..."
    cd /workspace/Evaluation-Values
    python bitnet_wrapper.py
    if [ $? -ne 0 ]; then
        echo "Error: Failed to setup BitNet model files"
        exit 1
    fi

    # Set up cleanup on exit
    cleanup_bitnet() {
        echo "Cleaning up BitNet temporary files..."
        cd /workspace/Evaluation-Values
        python bitnet_wrapper.py cleanup
        # Mark evaluation as complete
        python -c "
from realtime_results_saver import complete_model_evaluation
complete_model_evaluation('$MODEL_NAME', '$TRACK', '$BACKEND')
"
    }
    trap cleanup_bitnet EXIT
fi

# Set evaluation data directory - ALWAYS use full_eval for complete data
EVAL_DIR="../evaluation_data/full_eval"
if [ "$TRACK" = "strict-small" ]; then
    echo "Running STRICT-SMALL track evaluation (using full eval data, limited tests)"
else
    echo "Running STRICT track evaluation (full eval)"
fi

# Create results directory
RESULTS_DIR="./results/$MODEL_NAME/$TRACK"
mkdir -p "$RESULTS_DIR"

# Change to evaluation pipeline directory BEFORE creating the results directory path
cd ../evaluation-pipeline-2025

# Update RESULTS_DIR to absolute path now that we've changed directories
RESULTS_DIR="/workspace/Evaluation-Values/results/$MODEL_NAME/$TRACK"
mkdir -p "$RESULTS_DIR"

echo "Evaluating model: $MODEL_NAME"
echo "Track: $TRACK"
echo "Backend: $BACKEND"
echo "Model directory: $MODEL_ABS_PATH"
echo "Evaluation data: $EVAL_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Set backend for reading task
if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

echo "Starting zero-shot evaluations..."

# BLiMP evaluation
echo "Running BLiMP evaluation..."
OUTPUT_FILE="$RESULTS_DIR/blimp_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND \
    --task blimp \
    --data_path "${EVAL_DIR}/blimp_filtered" \
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"

save_task_result "zero_shot" "blimp" "$OUTPUT_FILE"

# BLiMP Supplement evaluation
echo "Running BLiMP Supplement evaluation..."
OUTPUT_FILE="$RESULTS_DIR/blimp_supplement_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND \
    --task blimp \
    --data_path "${EVAL_DIR}/supplement_filtered" \
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"

save_task_result "zero_shot" "blimp_supplement" "$OUTPUT_FILE"

# EWoK evaluation
echo "Running EWoK evaluation..."
OUTPUT_FILE="$RESULTS_DIR/ewok_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND \
    --task ewok \
    --data_path "${EVAL_DIR}/ewok_filtered" \
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"

save_task_result "zero_shot" "ewok" "$OUTPUT_FILE"

# Entity Tracking evaluation
echo "Running Entity Tracking evaluation..."
OUTPUT_FILE="$RESULTS_DIR/entity_tracking_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND \
    --task entity_tracking \
    --data_path "${EVAL_DIR}/entity_tracking" \
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"

save_task_result "zero_shot" "entity_tracking" "$OUTPUT_FILE"

# WUG Adjective Nominalization evaluation
echo "Running WUG Adjective Nominalization evaluation..."
OUTPUT_FILE="$RESULTS_DIR/wug_adj_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND \
    --task wug_adj \
    --data_path "${EVAL_DIR}/wug_adj_nominalization" \
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"

save_task_result "zero_shot" "wug_adj" "$OUTPUT_FILE"

# WUG Past Tense evaluation
echo "Running WUG Past Tense evaluation..."
OUTPUT_FILE="$RESULTS_DIR/wug_past_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND \
    --task wug_past \
    --data_path "${EVAL_DIR}/wug_past_tense" \
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"

save_task_result "zero_shot" "wug_past" "$OUTPUT_FILE"

# COMPS evaluation
echo "Running COMPS evaluation..."
OUTPUT_FILE="$RESULTS_DIR/comps_output.log"
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND \
    --task comps \
    --data_path "${EVAL_DIR}/comps" \
    --save_predictions 2>&1 | tee "$OUTPUT_FILE"

save_task_result "zero_shot" "comps" "$OUTPUT_FILE"

# Reading evaluation
echo "Running Reading evaluation..."
OUTPUT_FILE="$RESULTS_DIR/reading_output.log"
python -m evaluation_pipeline.reading.run \
    --model_path_or_name "$MODEL_ABS_PATH" \
    --backend $BACKEND_READ \
    --data_path "${EVAL_DIR}/reading/reading_data.csv" 2>&1 | tee "$OUTPUT_FILE"

save_task_result "reading" "reading_tasks" "$OUTPUT_FILE"

echo "Zero-shot evaluations completed!"

# Run finetuning evaluations (only for strict track, skip for strict-small due to padding issues)
if [ "$TRACK" = "strict" ]; then
    echo "Starting finetuning evaluations..."
    echo "Note: Applying tokenizer padding fix..."

    # Apply tokenizer fix
    cd /workspace/Evaluation-Values
    python fix_tokenizer_padding.py
    cd ../evaluation-pipeline-2025

    # GLUE tasks finetuning
    LR=3e-5
    BSZ=32
    BIG_BSZ=16
    MAX_EPOCHS=10
    WSC_EPOCHS=30
    SEED=42

    for task in boolq multirc; do
        echo "Finetuning on $task..."
        OUTPUT_FILE="$RESULTS_DIR/finetune_${task}_output.log"
        python -m evaluation_pipeline.finetune.run \
            --model_name_or_path "$MODEL_ABS_PATH" \
            --train_data "${EVAL_DIR}/glue_filtered/$task.train.jsonl" \
            --valid_data "${EVAL_DIR}/glue_filtered/$task.valid.jsonl" \
            --predict_data "${EVAL_DIR}/glue_filtered/$task.valid.jsonl" \
            --task "$task" \
            --num_labels 2 \
            --batch_size $BIG_BSZ \
            --learning_rate $LR \
            --num_epochs $MAX_EPOCHS \
            --sequence_length 512 \
            --results_dir "results" \
            --save \
            --save_dir "models" \
            --metrics accuracy f1 mcc \
            --metric_for_valid accuracy \
            --seed $SEED \
            --verbose 2>&1 | tee "$OUTPUT_FILE"

        save_task_result "finetuning" "$task" "$OUTPUT_FILE"
    done

    # RTE task
    echo "Finetuning on RTE..."
    OUTPUT_FILE="$RESULTS_DIR/finetune_rte_output.log"
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_ABS_PATH" \
        --train_data "${EVAL_DIR}/glue_filtered/rte.train.jsonl" \
        --valid_data "${EVAL_DIR}/glue_filtered/rte.valid.jsonl" \
        --predict_data "${EVAL_DIR}/glue_filtered/rte.valid.jsonl" \
        --task rte \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose 2>&1 | tee "$OUTPUT_FILE"

    save_task_result "finetuning" "rte" "$OUTPUT_FILE"

    # WSC task
    echo "Finetuning on WSC..."
    OUTPUT_FILE="$RESULTS_DIR/finetune_wsc_output.log"
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_ABS_PATH" \
        --train_data "${EVAL_DIR}/glue_filtered/wsc.train.jsonl" \
        --valid_data "${EVAL_DIR}/glue_filtered/wsc.valid.jsonl" \
        --predict_data "${EVAL_DIR}/glue_filtered/wsc.valid.jsonl" \
        --task wsc \
        --num_labels 2 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $WSC_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose 2>&1 | tee "$OUTPUT_FILE"

    save_task_result "finetuning" "wsc" "$OUTPUT_FILE"

    # MRPC and QQP tasks
    for task in mrpc qqp; do
        echo "Finetuning on $task..."
        OUTPUT_FILE="$RESULTS_DIR/finetune_${task}_output.log"
        python -m evaluation_pipeline.finetune.run \
            --model_name_or_path "$MODEL_ABS_PATH" \
            --train_data "${EVAL_DIR}/glue_filtered/$task.train.jsonl" \
            --valid_data "${EVAL_DIR}/glue_filtered/$task.valid.jsonl" \
            --predict_data "${EVAL_DIR}/glue_filtered/$task.valid.jsonl" \
            --task "$task" \
            --num_labels 2 \
            --batch_size $BSZ \
            --learning_rate $LR \
            --num_epochs $MAX_EPOCHS \
            --sequence_length 512 \
            --results_dir "results" \
            --save \
            --save_dir "models" \
            --metrics accuracy f1 mcc \
            --metric_for_valid f1 \
            --seed $SEED \
            --verbose 2>&1 | tee "$OUTPUT_FILE"

        save_task_result "finetuning" "$task" "$OUTPUT_FILE"
    done

    # MNLI task
    echo "Finetuning on MNLI..."
    OUTPUT_FILE="$RESULTS_DIR/finetune_mnli_output.log"
    python -m evaluation_pipeline.finetune.run \
        --model_name_or_path "$MODEL_ABS_PATH" \
        --train_data "${EVAL_DIR}/glue_filtered/mnli.train.jsonl" \
        --valid_data "${EVAL_DIR}/glue_filtered/mnli.valid.jsonl" \
        --predict_data "${EVAL_DIR}/glue_filtered/mnli.valid.jsonl" \
        --task mnli \
        --num_labels 3 \
        --batch_size $BSZ \
        --learning_rate $LR \
        --num_epochs $MAX_EPOCHS \
        --sequence_length 512 \
        --results_dir "results" \
        --save \
        --save_dir "models" \
        --metrics accuracy f1 mcc \
        --metric_for_valid accuracy \
        --seed $SEED \
        --verbose 2>&1 | tee "$OUTPUT_FILE"

    save_task_result "finetuning" "mnli" "$OUTPUT_FILE"

    echo "Finetuning evaluations completed!"
else
    echo "Skipping finetuning evaluations for strict-small track (zero-shot only)"
fi

# Run AoA evaluation if CDI data is available
if [ -f "${EVAL_DIR}/cdi_childes/cdi_childes.json" ]; then
    echo "Running Age of Acquisition evaluation..."

    # Apply AoA fix first
    cd /workspace/Evaluation-Values
    python fix_tokenizer_padding.py
    cd ../evaluation-pipeline-2025

    OUTPUT_FILE="$RESULTS_DIR/aoa_output.log"
    python -m evaluation_pipeline.AoA_word.run \
        --model_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --track_name $TRACK \
        --word_path "${EVAL_DIR}/cdi_childes/cdi_childes.json" \
        --output_dir "results" 2>&1 | tee "$OUTPUT_FILE"

    save_task_result "zero_shot" "aoa" "$OUTPUT_FILE"
else
    echo "CDI data not found, skipping AoA evaluation"
fi

# Mark this model evaluation as complete
cd /workspace/Evaluation-Values
python -c "
from realtime_results_saver import complete_model_evaluation
complete_model_evaluation('$MODEL_NAME', '$TRACK', '$BACKEND')
"

echo "All evaluations completed for $MODEL_NAME on $TRACK track!"
echo "Results saved to: $RESULTS_DIR"
echo "Real-time results available in: evaluation_results.json"
