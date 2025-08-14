#!/bin/bash

# Script to evaluate all three models on both strict and strict-small tracks

echo "Starting BabyLM 2025 Model Evaluations"
echo "======================================"

# Define models and their corresponding backends
declare -A models
models["microsoft--bitnet-b1.58-2B-4T"]="causal"
models["allenai--DataDecide-dolma1_7-no-math-code-14M"]="causal"
models["Xuandong--HPD-TinyBERT-F128"]="masked"

# Define tracks
tracks=("strict" "strict-small")

# Create results directory
mkdir -p results

# Get the current directory (should be Evaluation-Values)
CURRENT_DIR=$(pwd)
echo "Working directory: $CURRENT_DIR"

# Loop through each model and track
for model_dir in "${!models[@]}"; do
    backend=${models[$model_dir]}
    model_path="$CURRENT_DIR/models/$model_dir"

    echo ""
    echo "Processing model: $model_dir"
    echo "Backend: $backend"
    echo "Model path: $model_path"

    if [ ! -d "$model_path" ]; then
        echo "ERROR: Model directory not found: $model_path"
        echo "Please run download_models.py first"
        continue
    fi

    # Check if model has config.json
    if [ ! -f "$model_path/config.json" ]; then
        echo "ERROR: config.json not found in $model_path"
        echo "Please ensure the model was downloaded correctly"
        continue
    fi

    for track in "${tracks[@]}"; do
        echo ""
        echo "Evaluating $model_dir on $track track..."
        echo "----------------------------------------"

        # Run evaluation
        bash evaluate_model.sh "$model_path" "$model_dir" "$track" "$backend"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed $track evaluation for $model_dir"
        else
            echo "✗ Failed $track evaluation for $model_dir"
        fi
    done
done

echo ""
echo "All evaluations completed!"
echo "Results are saved in the ./results directory"
echo ""
echo "Results structure:"
echo "results/"
echo "├── microsoft--bitnet-b1.58-2B-4T/"
echo "│   ├── strict/"
echo "│   └── strict-small/"
echo "├── allenai--DataDecide-dolma1_7-no-math-code-14M/"
echo "│   ├── strict/"
echo "│   └── strict-small/"
echo "└── Xuandong--HPD-TinyBERT-F128/"
echo "    ├── strict/"
echo "    └── strict-small/"
