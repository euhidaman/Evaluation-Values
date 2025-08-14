#!/bin/bash

# Script to evaluate all three models on both strict and strict-small tracks

echo "Starting BabyLM 2025 Model Evaluations"
echo "======================================"

# Initialize real-time results tracking
cd /workspace/Evaluation-Values
python -c "
from realtime_results_saver import EvaluationResultsSaver
saver = EvaluationResultsSaver()
print('✓ Real-time results tracking initialized')
print('Results will be continuously saved to: evaluation_results.json')
"

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

# Track overall success
TOTAL_EVALUATIONS=0
SUCCESSFUL_EVALUATIONS=0

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

        TOTAL_EVALUATIONS=$((TOTAL_EVALUATIONS + 1))

        # Run evaluation
        bash evaluate_model.sh "$model_path" "$model_dir" "$track" "$backend"

        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed $track evaluation for $model_dir"
            SUCCESSFUL_EVALUATIONS=$((SUCCESSFUL_EVALUATIONS + 1))
        else
            echo "✗ Failed $track evaluation for $model_dir"
        fi

        # Show current progress
        echo "Progress: $SUCCESSFUL_EVALUATIONS/$TOTAL_EVALUATIONS evaluations completed"

        # Show current results summary
        python -c "
from realtime_results_saver import get_summary
summary = get_summary()
print(f'Real-time summary: {summary[\"completed_tasks\"]}/{summary[\"total_tasks\"]} tasks ({summary[\"progress_percentage\"]:.1f}%)')
"
    done
done

# Mark all evaluations as complete
cd /workspace/Evaluation-Values
python -c "
from realtime_results_saver import mark_evaluation_complete, get_summary

# Mark completion
mark_evaluation_complete()

# Show final summary
summary = get_summary()
print()
print('=' * 50)
print('FINAL EVALUATION SUMMARY')
print('=' * 50)
print(f'Total models: {summary[\"total_models\"]}')
print(f'Completed models: {summary[\"completed_models\"]}')
print(f'Total tasks: {summary[\"total_tasks\"]}')
print(f'Completed tasks: {summary[\"completed_tasks\"]}')
print(f'Success rate: {summary[\"progress_percentage\"]:.1f}%')
print(f'Last updated: {summary[\"last_updated\"]}')
print()
print('✓ All results saved to: evaluation_results.json')
"

echo ""
echo "All evaluations completed!"
echo "Final success rate: $SUCCESSFUL_EVALUATIONS/$TOTAL_EVALUATIONS"
echo ""
echo "Results are saved in the ./results directory"
echo "Real-time results available in: evaluation_results.json"
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
