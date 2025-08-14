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

# Set evaluation data directory based on track
if [ "$TRACK" = "strict-small" ]; then
    EVAL_DIR="../evaluation_data/fast_eval"
    echo "Running STRICT-SMALL track evaluation (fast eval)"
else
    EVAL_DIR="../evaluation_data/full_eval"
    echo "Running STRICT track evaluation (full eval)"
fi

# Create results directory
RESULTS_DIR="./results/$MODEL_NAME/$TRACK"
mkdir -p "$RESULTS_DIR"

echo "Evaluating model: $MODEL_NAME"
echo "Track: $TRACK"
echo "Backend: $BACKEND"
echo "Model directory: $MODEL_ABS_PATH"
echo "Evaluation data: $EVAL_DIR"
echo "Results will be saved to: $RESULTS_DIR"

# Change to evaluation pipeline directory
cd ../evaluation-pipeline-2025

# Set backend for reading task
if [[ "$BACKEND" == *"enc_dec"* ]]; then
    BACKEND_READ="enc_dec"
else
    BACKEND_READ=$BACKEND
fi

echo "Starting zero-shot evaluations..."

# BLiMP evaluation
echo "Running BLiMP evaluation..."
if [ "$TRACK" = "strict-small" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task blimp \
        --data_path "${EVAL_DIR}/blimp_fast" \
        --save_predictions \
        --revision_name $TRACK
else
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task blimp \
        --data_path "${EVAL_DIR}/blimp_filtered" \
        --save_predictions
fi

# BLiMP Supplement evaluation
echo "Running BLiMP Supplement evaluation..."
if [ "$TRACK" = "strict-small" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task blimp \
        --data_path "${EVAL_DIR}/supplement_fast" \
        --save_predictions \
        --revision_name $TRACK
else
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task blimp \
        --data_path "${EVAL_DIR}/supplement_filtered" \
        --save_predictions
fi

# EWoK evaluation
echo "Running EWoK evaluation..."
if [ "$TRACK" = "strict-small" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task ewok \
        --data_path "${EVAL_DIR}/ewok_fast" \
        --save_predictions \
        --revision_name $TRACK
else
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task ewok \
        --data_path "${EVAL_DIR}/ewok_filtered" \
        --save_predictions
fi

# Entity Tracking evaluation
echo "Running Entity Tracking evaluation..."
if [ "$TRACK" = "strict-small" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task entity_tracking \
        --data_path "${EVAL_DIR}/entity_tracking_fast" \
        --save_predictions \
        --revision_name $TRACK
else
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task entity_tracking \
        --data_path "${EVAL_DIR}/entity_tracking" \
        --save_predictions
fi

# WUG Adjective Nominalization evaluation
echo "Running WUG Adjective Nominalization evaluation..."
if [ "$TRACK" = "strict-small" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task wug_adj \
        --data_path "${EVAL_DIR}/wug_adj_nominalization" \
        --save_predictions \
        --revision_name $TRACK
else
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task wug_adj \
        --data_path "${EVAL_DIR}/wug_adj_nominalization" \
        --save_predictions
fi

# WUG Past Tense evaluation
echo "Running WUG Past Tense evaluation..."
if [ "$TRACK" = "strict-small" ]; then
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task wug_past \
        --data_path "${EVAL_DIR}/wug_past_tense" \
        --save_predictions \
        --revision_name $TRACK
else
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task wug_past \
        --data_path "${EVAL_DIR}/wug_past_tense" \
        --save_predictions
fi

# COMPS evaluation (only for full/strict track)
if [ "$TRACK" != "strict-small" ]; then
    echo "Running COMPS evaluation..."
    python -m evaluation_pipeline.sentence_zero_shot.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --task comps \
        --data_path "${EVAL_DIR}/comps" \
        --save_predictions
fi

# Reading evaluation
echo "Running Reading evaluation..."
if [ "$TRACK" = "strict-small" ]; then
    python -m evaluation_pipeline.reading.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND_READ \
        --data_path "${EVAL_DIR}/reading/reading_data.csv" \
        --revision_name $TRACK
else
    python -m evaluation_pipeline.reading.run \
        --model_path_or_name "$MODEL_ABS_PATH" \
        --backend $BACKEND_READ \
        --data_path "${EVAL_DIR}/reading/reading_data.csv"
fi

echo "Zero-shot evaluations completed!"

# Run finetuning evaluations (only for strict track)
if [ "$TRACK" != "strict-small" ]; then
    echo "Starting finetuning evaluations..."

    # GLUE tasks finetuning
    LR=3e-5
    BSZ=32
    BIG_BSZ=16
    MAX_EPOCHS=10
    WSC_EPOCHS=30
    SEED=42

    for task in boolq multirc; do
        echo "Finetuning on $task..."
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
            --verbose
    done

    # RTE task
    echo "Finetuning on RTE..."
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
        --verbose

    # WSC task
    echo "Finetuning on WSC..."
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
        --verbose

    # MRPC and QQP tasks
    for task in mrpc qqp; do
        echo "Finetuning on $task..."
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
            --verbose
    done

    # MNLI task
    echo "Finetuning on MNLI..."
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
        --verbose

    echo "Finetuning evaluations completed!"
fi

# Run AoA evaluation if CDI data is available
if [ -f "${EVAL_DIR}/cdi_childes/cdi_childes.json" ]; then
    echo "Running Age of Acquisition evaluation..."
    python -m evaluation_pipeline.AoA_word.run \
        --model_name "$MODEL_ABS_PATH" \
        --backend $BACKEND \
        --track_name $TRACK \
        --word_path "${EVAL_DIR}/cdi_childes/cdi_childes.json" \
        --output_dir "results"
else
    echo "CDI data not found, skipping AoA evaluation"
fi

echo "All evaluations completed for $MODEL_NAME on $TRACK track!"
echo "Results saved to: $RESULTS_DIR"
