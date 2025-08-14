# BabyLM 2025 Model Evaluation

This repository contains scripts and configurations to evaluate three models on the BabyLM 2025 competition's **strict** and **strict-small** tracks for text-only evaluations.

## Models to Evaluate

1. **Microsoft BitNet B1.58-2B-4T** - `microsoft/bitnet-b1.58-2B-4T`
2. **AllenAI DataDecide Dolma1.7 14M** - `allenai/DataDecide-dolma1_7-no-math-code-14M`  
3. **Xuandong HPD-TinyBERT-F128** - `Xuandong/HPD-TinyBERT-F128`

## Evaluation Tasks

### Zero-Shot Evaluations (Both Tracks)
- **BLiMP** - Acceptability judgments for syntax
- **BLiMP Supplement** - Additional syntactic evaluations
- **EWoK (core 1.0)** - World knowledge evaluations
- **Entity Tracking** - Coreference and entity understanding
- **WUG Adjective Nominalization** - Derivational morphology
- **WUG Past Tense** - Inflectional morphology
- **COMPS** - Compositionality evaluations
- **Reading** - Eye tracking and self-paced reading metrics
- **AoA (Age of Acquisition)** - Word acquisition patterns

### Finetuning Evaluations (Strict Track Only)
- **GLUE Suite**: BoolQ, MNLI, MRPC, MultiRC, QQP, RTE, WSC

## Quick Start

### 1. Set Up Environment

```bash
# Clone the evaluation pipeline (if not already present)
git clone https://github.com/babylm/evaluation-pipeline-2025.git

# Install dependencies
cd Evaluation-Values
python setup_dependencies.py
```

### 2. Download Models and Data

```bash
# Download all three models
python download_models.py

# Download evaluation data
python download_evaluation_data.py

# Set up EWoK data (missing from standard datasets)
python setup_ewok_data.py

# Apply evaluation pipeline fixes
python fix_tokenizer_padding.py
```

### 3. Run Evaluations

#### Option A: Evaluate All Models (Recommended)
```bash
# Run all models on both tracks
bash run_all_evaluations.sh
```

#### Option B: Evaluate Individual Models
```bash
# BitNet model - strict track (full evaluation with finetuning)
bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict causal

# BitNet model - strict-small track (zero-shot only)
bash evaluate_model.sh ./models/microsoft--bitnet-b1.58-2B-4T microsoft--bitnet-b1.58-2B-4T strict-small causal

# AllenAI model - strict track
bash evaluate_model.sh ./models/allenai--DataDecide-dolma1_7-no-math-code-14M allenai--DataDecide-dolma1_7-no-math-code-14M strict causal

# HPD-TinyBERT model - strict track (masked language model)
bash evaluate_model.sh ./models/Xuandong--HPD-TinyBERT-F128 Xuandong--HPD-TinyBERT-F128 strict masked
```

### 4. Collect Results

```bash
# Aggregate all results into JSON format
python collect_results.py
```

## File Structure

```
Evaluation-Values/
├── models/                          # Downloaded model weights (not in git)
│   ├── microsoft--bitnet-b1.58-2B-4T/
│   ├── allenai--DataDecide-dolma1_7-no-math-code-14M/
│   └── Xuandong--HPD-TinyBERT-F128/
├── results/                         # Evaluation results (created during evaluation)
│   ├── microsoft--bitnet-b1.58-2B-4T/
│   │   ├── strict/
│   │   └── strict-small/
│   ├── allenai--DataDecide-dolma1_7-no-math-code-14M/
│   └── Xuandong--HPD-TinyBERT-F128/
├── download_models.py               # Downloads all three models
├── download_evaluation_data.py      # Downloads evaluation datasets
├── setup_ewok_data.py              # Sets up missing EWoK data
├── fix_tokenizer_padding.py        # Fixes tokenizer issues for finetuning
├── evaluate_model.sh               # Main evaluation script
├── run_all_evaluations.sh          # Evaluates all models
├── collect_results.py              # Aggregates results to JSON
├── configuration_bitnet.py         # BitNet model configuration
├── modeling_bitnet.py              # BitNet model implementation
├── bitnet_wrapper.py               # BitNet setup utility
└── evaluation_results.json         # Final aggregated results
```

## Model-Specific Notes

### BitNet B1.58-2B-4T
- **Type**: Causal language model with 1.58-bit quantization
- **Backend**: `causal`
- **Special handling**: Requires custom configuration and modeling files
- **Issues fixed**: Model loading, quantization configuration, attention bias

### AllenAI DataDecide Dolma1.7 14M
- **Type**: Small causal language model based on OLMo
- **Backend**: `causal`
- **Issues fixed**: Model type recognition, transformers compatibility

### HPD-TinyBERT-F128
- **Type**: Masked language model (BERT-style)
- **Backend**: `masked`
- **Note**: Sentence transformer architecture with pooling layers

## Evaluation Tracks

### Strict Track
- **Full evaluation** with both zero-shot and finetuning tasks
- **All datasets** from `evaluation_data/full_eval/`
- **GLUE finetuning** on 7 tasks (BoolQ, MNLI, MRPC, MultiRC, QQP, RTE, WSC)
- **Complete metrics** including reading tasks and AoA

### Strict-Small Track  
- **Zero-shot only** (no finetuning due to tokenizer padding constraints)
- **Same datasets** from `evaluation_data/full_eval/` but limited task scope
- **Faster evaluation** suitable for quick testing

## Data Sources

- **BLiMP & Supplement**: Included in evaluation pipeline
- **Entity Tracking**: Included in evaluation pipeline  
- **WUG Tasks**: Included in evaluation pipeline
- **COMPS**: Included in evaluation pipeline
- **Reading**: Included in evaluation pipeline
- **GLUE**: Filtered versions included in evaluation pipeline
- **CDI/Childes**: For AoA evaluation (included)
- **EWoK**: Downloaded from `ewok-core/ewok-core-1.0` on Hugging Face

## Fixed Issues

1. **EWoK Data Missing**: Added automatic download from Hugging Face
2. **Tokenizer Padding**: Fixed for BitNet and other models during finetuning  
3. **BitNet Model Loading**: Added custom configuration and modeling files
4. **AoA eval_util**: Created missing module for Age of Acquisition evaluation
5. **Data Paths**: Ensured all evaluations use `full_eval` for complete data
6. **Model Compatibility**: Fixed transformers version issues for all models

## Results Format

Results are automatically saved in JSON format with the following structure:

```json
{
  "evaluation_summary": {
    "total_models": 3,
    "models_evaluated": ["microsoft--bitnet-b1.58-2B-4T", "..."],
    "tracks": ["strict", "strict-small"],
    "evaluation_date": "2025-08-14T..."
  },
  "model_results": {
    "model_name": {
      "track_name": {
        "zero_shot": { "task_name": { "scores": "..." } },
        "finetuning": { "task_name": { "scores": "..." } },
        "reading": { "metrics": "..." }
      }
    }
  }
}
```

## Cloud Deployment Commands

After pushing this repository to your cloud environment:

```bash
# 1. Set up environment
python setup_dependencies.py

# 2. Download everything
python download_models.py
python download_evaluation_data.py  
python setup_ewok_data.py

# 3. Apply fixes
python fix_tokenizer_padding.py

# 4. Run evaluations
bash run_all_evaluations.sh

# 5. Collect results
python collect_results.py
```

The entire evaluation suite will run automatically and save all results in JSON format for easy analysis and submission.

## Troubleshooting

### BitNet Model Issues
- Ensure `configuration_bitnet.py` and `modeling_bitnet.py` are in the main directory
- The `bitnet_wrapper.py` automatically sets up symlinks during evaluation

### Memory Issues
- Use strict-small track for faster evaluation with limited resources
- BitNet model will show GPU warnings but works on CPU

### Data Issues  
- Run `setup_ewok_data.py` if EWoK evaluations fail
- Ensure `evaluation_data/full_eval/` contains all required datasets

### Tokenizer Issues
- Run `fix_tokenizer_padding.py` before finetuning evaluations
- This sets pad_token = eos_token for models without padding tokens
