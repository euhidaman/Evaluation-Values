# BabyLM 2025 Text-Only Model Evaluation (GGUF Support)

This repository contains scripts to evaluate three specific text-based language models on the BabyLM 2025 text-only strict track evaluation suite, including **native GGUF model support**.

## Models to Evaluate

1. **TinyLLama-v0-5M-F16** - `mofosyne/TinyLLama-v0-5M-F16-llamafile` (GGUF)
2. **BitNet-b1.58-2B-4T** - `microsoft/bitnet-b1.58-2B-4T-gguf` (GGUF)
3. **DataDecide-dolma1_7-no-math-code-14M** - `allenai/DataDecide-dolma1_7-no-math-code-14M` (Standard)

## ✨ GGUF Support

This evaluation pipeline now supports **GGUF models natively** using HuggingFace transformers 4.51.3+ built-in GGUF capabilities. No conversion needed!

### GGUF-Specific Setup

1. **Install GGUF dependencies** (if not already installed):
    ```bash
    pip install gguf>=0.10.0
    ```

2. **Use the GGUF evaluation script**:
    ```bash
    # Copy the GGUF-specific script to the evaluation pipeline directory
    cp ../Evaluation-Values/evaluate_gguf_models.py .

    # Run GGUF evaluation
    python evaluate_gguf_models.py
    ```

### How GGUF Support Works

The evaluation pipeline automatically:
- **Detects GGUF files** in HuggingFace repositories
- **Uses native transformers GGUF loading** with `gguf_file` parameter
- **Falls back to standard loading** if GGUF fails
- **Tests model compatibility** before running full evaluations

### GGUF vs Standard Evaluation

| Feature | GGUF Models | Standard Models |
|---------|-------------|-----------------|
| Loading Method | Native transformers GGUF support | Standard transformers |
| Memory Usage | Optimized quantized weights | Full precision weights |
| Inference Speed | Faster (quantized) | Baseline speed |
| Model Size | Smaller on disk | Larger on disk |
| Accuracy | Slightly reduced (quantization) | Full accuracy |

## Evaluation Tasks

The evaluation includes the following text-only tasks:
- **BLiMP** (Benchmark of Linguistic Minimal Pairs)
- **EWoK** (Evaluation Without Kin)
- **Entity Tracking** 
- **WUG Tasks** (Adjective Nominalization & Past Tense)
- **Reading Comprehension**
- **Fine-tuning on GLUE tasks** (BoolQ, MultiRC, RTE)

## Directory Structure

The correct directory structure should be:

```
D:\BabyLM\
├── evaluation-pipeline-2025/          # Main evaluation pipeline
│   ├── evaluation_pipeline/
│   ├── eval_zero_shot.sh
│   ├── eval_zero_shot_fast.sh
│   ├── eval_finetuning.sh
│   └── requirements.txt
├── evaluation_data/                   # Evaluation datasets
│   ├── fast_eval/
│   └── full_eval/
└── Evaluation-Values/                 # This repository
    ├── evaluate_models.py
    ├── evaluate_gguf_models.py
    ├── check_model_compatibility.py
    └── README.md
```

## Setup Instructions

### 1. Prerequisites

Ensure you have the evaluation pipeline and data in the correct locations:

```bash
# Navigate to your BabyLM directory
cd /workspace

# Verify directory structure
ls -la
# Should show: evaluation-pipeline-2025/, evaluation_data/, Evaluation-Values/
```

### 2. Setup Python Environment

```bash
# Navigate to the evaluation pipeline directory
cd evaluation-pipeline-2025

# Install dependencies (if not already installed)
pip install -r requirements.txt

# Install additional GGUF support
pip install gguf>=0.10.0
```

### 3. Fix NLTK Dependencies

The NLTK `punkt_tab` error has been fixed in the evaluation pipeline. The fix automatically downloads required NLTK data when running EWoK evaluations.

### 4. Test Model Compatibility

Before running evaluations, test if models can be loaded:

```bash
# Copy the compatibility checker to the evaluation pipeline directory
cp ../Evaluation-Values/check_model_compatibility.py .

# Run compatibility check
python check_model_compatibility.py
```

## Usage

### Option 1: Standard Evaluation (All Models)

```bash
# Copy the standard evaluation script
cp ../Evaluation-Values/evaluate_models.py .

# Run all evaluations (fast, full, finetune)
python evaluate_models.py
```

### Option 2: GGUF-Focused Evaluation

```bash
# Copy the GGUF evaluation script
cp ../Evaluation-Values/evaluate_gguf_models.py .

# Run GGUF-optimized evaluations
python evaluate_gguf_models.py
```

### Option 3: Manual Evaluation

Run evaluations manually for individual models:

```bash
# Fast evaluation (quick test)
./eval_zero_shot_fast.sh mofosyne/TinyLLama-v0-5M-F16-llamafile main causal ../evaluation_data/fast_eval

# Full evaluation (comprehensive)
./eval_zero_shot.sh mofosyne/TinyLLama-v0-5M-F16-llamafile causal ../evaluation_data/full_eval

# Fine-tuning evaluation (GLUE tasks)
./eval_finetuning.sh mofosyne/TinyLLama-v0-5M-F16-llamafile
```

## Troubleshooting

### NLTK Error Fixed

The original error:
```
LookupError: Resource punkt_tab not found.
```

Has been fixed by adding automatic NLTK data downloads to the evaluation pipeline. The fix is located in:
- `evaluation_pipeline/ewok/dl_and_filter.py`

### Common Issues

1. **Path Issues**: Ensure you're running scripts from the `evaluation-pipeline-2025` directory
2. **Missing Data**: Verify `../evaluation_data/` exists and contains `fast_eval/` and `full_eval/`
3. **GGUF Loading Fails**: The scripts automatically fall back to standard model loading
4. **Memory Issues**: GGUF models use less memory; try those first for limited GPU setups

### Model-Specific Notes

- **TinyLLama-v0-5M-F16**: GGUF format, very small model, good for testing
- **BitNet-b1.58-2B-4T**: GGUF format, larger quantized model
- **DataDecide-dolma1_7-no-math-code-14M**: Standard format, requires more memory

## Files in This Repository

- `evaluate_models.py`: Standard evaluation script for all three models
- `evaluate_gguf_models.py`: GGUF-optimized evaluation with compatibility testing
- `check_model_compatibility.py`: Tests model loading and compatibility
- `README.md`: This documentation

## Results

Results will be saved in the `results/` directory within the evaluation pipeline, with separate folders for each evaluation type and model.

## Support

For issues with:
- **Evaluation pipeline**: Check the main evaluation-pipeline-2025 repository
- **Model loading**: Use the compatibility checker script
- **GGUF support**: Ensure transformers>=4.51.3 and gguf>=0.10.0
- **Path issues**: Verify the directory structure matches the layout above
