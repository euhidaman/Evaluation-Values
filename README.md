# BabyLM 2025 Model Evaluation Setup

This repository contains scripts and instructions for evaluating three specific models on the BabyLM 2025 text-only tracks (strict and strict-small).

## Models to Evaluate

1. **Microsoft BitNet B1.58-2B-4T**: `microsoft/bitnet-b1.58-2B-4T`
2. **AllenAI DataDecide Dolma 14M**: `allenai/DataDecide-dolma1_7-no-math-code-14M`  
3. **HPD TinyBERT F128**: `Xuandong/HPD-TinyBERT-F128`

## Evaluation Tracks

- **Strict Track**: Full evaluation with all tasks including finetuning
- **Strict-Small Track**: Fast evaluation subset for quick testing

## Prerequisites

### 1. System Requirements
- Python 3.8+ 
- CUDA-capable GPU (recommended)
- At least 50GB free disk space for models and data
- Linux/macOS environment (Windows with WSL2 also works)

### 2. Required Repositories
Ensure you have the following repositories in your BabyLM workspace:
```
BabyLM/
├── evaluation-pipeline-2025/     # Main evaluation pipeline
├── evaluation_data/              # Evaluation datasets 
├── babylm_dataset/              # Training data (if needed)
└── Evaluation-Values/           # This repository
```

## Setup Instructions

### Step 1: Install Dependencies

Navigate to the evaluation pipeline and install requirements:
```bash
cd ../evaluation-pipeline-2025
pip install -r requirements.txt
```

Required packages include:
- transformers>=4.51.3
- torch>=2.7.0
- datasets>=3.6.0
- scikit-learn>=1.6.1
- numpy>=2.2.5
- pandas>=2.2.3
- statsmodels>=0.14.4
- nltk>=3.9.1
- wandb>=0.19.11

### Step 2: Check and Download Evaluation Data

First, check if all evaluation data is available:
```bash
cd Evaluation-Values
python check_evaluation_data.py
```

If any evaluation data is missing, you may need to download it manually. Check the evaluation-pipeline-2025 repository for specific download instructions.

### Step 3: Download Models

Download all three models from Hugging Face:
```bash
python download_models.py --models_dir ./models
```

This will create:
```
models/
├── microsoft--bitnet-b1.58-2B-4T/
├── allenai--DataDecide-dolma1_7-no-math-code-14M/
└── Xuandong--HPD-TinyBERT-F128/
```

**Note**: The BitNet model (2B parameters) will require significant download time and storage (~4GB).

## Running Evaluations

### Option 1: Evaluate All Models (Recommended)

Run comprehensive evaluation of all models on both tracks:
```bash
bash run_all_evaluations.sh
```

This will automatically:
- Evaluate each model on both strict and strict-small tracks
- Use appropriate backends for each model type
- Save results in organized directory structure

### Option 2: Evaluate Individual Models

For more control, evaluate models individually:

```bash
# BitNet model on STRICT track (RECOMMENDED - includes AoA evaluation)
bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict causal

# DataDecide model on STRICT track 
bash evaluate_model.sh ./models/allenai--DataDecide-dolma1_7-no-math-code-14M allenai--DataDecide-dolma1_7-no-math-code-14M strict causal

# TinyBERT model on STRICT track
bash evaluate_model.sh ./models/Xuandong--HPD-TinyBERT-F128 Xuandong--HPD-TinyBERT-F128 strict masked

# For faster testing only, use strict-small track (limited evaluation, no AoA):
bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict-small causal
```

**Important**: Use the **strict** track for complete evaluation including:
- ✅ Age of Acquisition (AoA) evaluation with CDI data
- ✅ All GLUE fine-tuning tasks  
- ✅ Complete datasets for all evaluation components

The **strict-small** track is only for quick testing and uses limited datasets without AoA evaluation.

## Evaluation Components

### Strict Track (Full Evaluation)
- **Zero-shot tasks**:
  - BLiMP (linguistic acceptability)
  - BLiMP Supplement
  - EWoK (semantic knowledge)
  - Entity Tracking
  - WUG Adjective Nominalization
  - WUG Past Tense Formation
  - COMPS (conceptual property knowledge)
  - Reading Time Prediction

- **Fine-tuning tasks (GLUE)**:
  - BoolQ (question answering)
  - MultiRC (reading comprehension)
  - RTE (textual entailment)
  - WSC (coreference resolution)
  - MRPC (paraphrase detection)
  - QQP (question paraphrase)
  - MNLI (natural language inference)

- **Age of Acquisition (AoA)**:
  - Correlation with human acquisition data

### Strict-Small Track (Fast Evaluation)
- Subset of zero-shot tasks with reduced datasets
- No fine-tuning evaluations
- Faster turnaround for development/testing

## Model Backend Configuration

The evaluation pipeline automatically selects appropriate backends:

- **BitNet & DataDecide models**: `causal` (decoder-only)
- **TinyBERT model**: `masked` (encoder-only)

## Expected Runtime

**Approximate evaluation times per model**:

- **Strict-Small Track**: 2-4 hours
- **Strict Track**: 8-12 hours (including fine-tuning)

**Total for all models**: 30-48 hours

Times vary based on:
- GPU specifications
- Model size
- Network speed for downloads

## Results Structure

Results will be organized as follows:
```
results/
├── microsoft--bitnet-b1.58-2B-4T/
│   ├── strict/
│   │   ├── main/
│   │   │   ├── finetune/
│   │   │   │   ├── boolq/
│   │   │   │   ├── multirc/
│   │   │   │   └── ...
│   │   │   └── zero_shot/
│   │   │       ├── causal/
│   │   │       │   ├── blimp/
│   │   │       │   ├── ewok/
│   │   │       │   └── ...
│   │   └── ...
│   └── strict-small/
│       └── strict-small/
├── allenai--DataDecide-dolma1_7-no-math-code-14M/
└── Xuandong--HPD-TinyBERT-F128/
```

Each task directory contains:
- `predictions.jsonl`: Model predictions
- `results.txt`: Evaluation metrics
- `best_temperature_report.txt`: Calibration results (for some tasks)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   ```bash
   export CUDA_VISIBLE_DEVICES=0  # Use specific GPU
   # Reduce batch size in evaluation scripts if needed
   ```

2. **Missing evaluation data**:
   - Check `../evaluation_data/` directory structure
   - Refer to evaluation-pipeline-2025 documentation
   - Some tasks will be skipped if data is unavailable

3. **Model download failures**:
   ```bash
   # Retry with specific model
   python download_models.py --models_dir ./models
   # Or download manually from Hugging Face
   ```

4. **Hugging Face authentication** (if needed):
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

### Performance Optimization

- **For faster evaluation**: Use strict-small track first
- **For GPU memory issues**: Evaluate models sequentially
- **For network issues**: Download models separately before evaluation

## File Descriptions

- `download_models.py`: Downloads all three models from Hugging Face
- `evaluate_model.sh`: Main evaluation script for single model/track
- `run_all_evaluations.sh`: Batch evaluation of all models and tracks
- `check_evaluation_data.py`: Verifies evaluation data availability

## Important Notes

1. **Model Types**: Different models use different architectures:
   - BitNet: Quantized decoder model
   - DataDecide: Standard decoder model  
   - TinyBERT: Compressed encoder model

2. **Evaluation Data**: Some evaluation components may require additional downloads not covered by this setup

3. **Results Submission**: Results from the `strict` track should be used for final BabyLM 2025 submissions

4. **Resource Requirements**: Ensure adequate disk space and memory before starting

## Support

For issues related to:
- **Evaluation pipeline**: Check evaluation-pipeline-2025 repository issues
- **Model downloads**: Check individual model pages on Hugging Face
- **This setup**: Review error logs and check file paths

## Quick Start Summary

```bash
# 1. Install dependencies
cd ../evaluation-pipeline-2025
pip install -r requirements.txt

# 2. Return to evaluation directory
cd ../Evaluation-Values

# 3. Check evaluation data
python check_evaluation_data.py

# 4. Download models
python download_models.py

# 5. Run all evaluations
bash run_all_evaluations.sh
```

Expected total runtime: 30-48 hours for complete evaluation of all models on both tracks.
