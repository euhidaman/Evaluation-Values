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

## Setup Instructions

### 1. Clone the Evaluation Pipeline

```bash
# Clone the official evaluation pipeline
git clone https://github.com/babylm/evaluation-pipeline-2025.git
cd evaluation-pipeline-2025
```

### 2. Setup Evaluation Data

The evaluation data should already be available in the parent directory as `../evaluation_data/`. 
The directory structure should look like this:

```
BabyLM/
├── evaluation-pipeline-2025/
│   ├── evaluation_pipeline/
│   ├── eval_zero_shot.sh
│   └── ...
├── evaluation_data/
│   ├── fast_eval/
│   └── full_eval/
└── Evaluation-Values/
    ├── evaluate_models.py
    └── README.md
```

If evaluation_data is not present, download it from OSF:
```bash
# From the BabyLM parent directory
wget -O evaluation_data.zip "https://osf.io/ryjfm/download"
unzip evaluation_data.zip
```

### 3. Setup Python Environment

```bash
# Create virtual environment
python -m venv babylm_eval
source babylm_eval/bin/activate  # On Windows: babylm_eval\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### 4. Setup HuggingFace Access

```bash
# Login to HuggingFace (required for model access)
huggingface-cli login
# Enter your HuggingFace token when prompted
```

### 5. Download EWoK Data (Special Setup)

```bash
# EWoK requires special access - run this after HF login
python -m evaluation_pipeline.ewok.dl_and_filter

# For fast EWoK data (password: BabyLM2025)
cd evaluation_data/fast_eval
unzip ewok_fast.zip
# Enter password: BabyLM2025
cd ../..
```

## Running Evaluations

### Option 1: Automated Evaluation Script

Copy the `evaluate_models.py` script to the evaluation-pipeline-2025 directory and run:

```bash
python evaluate_models.py
```

This will automatically run all evaluations (fast, full, and fine-tuning) for all three models.

### Option 2: Manual Evaluation Commands

#### Fast Evaluation (Quick Testing)

```bash
# TinyLLama-v0-5M-F16
./eval_zero_shot_fast.sh mofosyne/TinyLLama-v0-5M-F16-llamafile main causal

# BitNet-b1.58-2B-4T
./eval_zero_shot_fast.sh microsoft/bitnet-b1.58-2B-4T-gguf main causal

# DataDecide-dolma1_7-no-math-code-14M
./eval_zero_shot_fast.sh allenai/DataDecide-dolma1_7-no-math-code-14M main causal
```

#### Full Evaluation (Comprehensive)

```bash
# TinyLLama-v0-5M-F16
./eval_zero_shot.sh mofosyne/TinyLLama-v0-5M-F16-llamafile causal

# BitNet-b1.58-2B-4T
./eval_zero_shot.sh microsoft/bitnet-b1.58-2B-4T-gguf causal

# DataDecide-dolma1_7-no-math-code-14M
./eval_zero_shot.sh allenai/DataDecide-dolma1_7-no-math-code-14M causal
```

#### Fine-tuning Evaluation (GLUE Tasks)

```bash
# TinyLLama-v0-5M-F16
./eval_finetuning.sh mofosyne/TinyLLama-v0-5M-F16-llamafile

# BitNet-b1.58-2B-4T
./eval_finetuning.sh microsoft/bitnet-b1.58-2B-4T-gguf

# DataDecide-dolma1_7-no-math-code-14M
./eval_finetuning.sh allenai/DataDecide-dolma1_7-no-math-code-14M
```

## Expected Results Structure

After running evaluations, results will be stored in the `results/` directory:

```
results/
├── TinyLLama-v0-5M-F16-llamafile/
│   ├── main/
│   │   ├── finetune/
│   │   │   ├── boolq/
│   │   │   ├── multirc/
│   │   │   └── rte/
│   │   └── zero_shot/
│   │       └── causal/
│   │           ├── blimp/
│   │           ├── ewok/
│   │           ├── entity_tracking/
│   │           ├── wug_adj/
│   │           ├── wug_past/
│   │           └── reading/
├── bitnet-b1.58-2B-4T-gguf/
│   └── ...
└── DataDecide-dolma1_7-no-math-code-14M/
    └── ...
```

## Individual Task Commands

If you want to run specific tasks manually:

### Zero-shot Tasks

```bash
# BLiMP evaluation
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name MODEL_PATH \
    --backend causal \
    --task blimp \
    --data_path evaluation_data/fast_eval/blimp_fast \
    --save_predictions \
    --revision_name main

# EWoK evaluation
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name MODEL_PATH \
    --backend causal \
    --task ewok \
    --data_path evaluation_data/fast_eval/ewok_fast \
    --save_predictions \
    --revision_name main

# Entity Tracking
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name MODEL_PATH \
    --backend causal \
    --task entity_tracking \
    --data_path evaluation_data/fast_eval/entity_tracking_fast \
    --save_predictions \
    --revision_name main

# WUG Adjective Nominalization
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name MODEL_PATH \
    --backend causal \
    --task wug_adj \
    --data_path evaluation_data/fast_eval/wug_adj_nominalization \
    --save_predictions \
    --revision_name main

# WUG Past Tense
python -m evaluation_pipeline.sentence_zero_shot.run \
    --model_path_or_name MODEL_PATH \
    --backend causal \
    --task wug_past \
    --data_path evaluation_data/fast_eval/wug_past_tense \
    --save_predictions \
    --revision_name main

# Reading Task
python -m evaluation_pipeline.reading.run \
    --model_path_or_name MODEL_PATH \
    --backend causal \
    --data_path evaluation_data/fast_eval/reading/reading_data.csv \
    --revision_name main
```

### Fine-tuning Tasks

```bash
# BoolQ
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path MODEL_PATH \
    --train_data evaluation_data/full_eval/glue_filtered/boolq.train.jsonl \
    --valid_data evaluation_data/full_eval/glue_filtered/boolq.valid.jsonl \
    --predict_data evaluation_data/full_eval/glue_filtered/boolq.valid.jsonl \
    --task boolq \
    --num_labels 2 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 10 \
    --sequence_length 512 \
    --results_dir results \
    --save \
    --save_dir models \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed 42 \
    --verbose

# MultiRC
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path MODEL_PATH \
    --train_data evaluation_data/full_eval/glue_filtered/multirc.train.jsonl \
    --valid_data evaluation_data/full_eval/glue_filtered/multirc.valid.jsonl \
    --predict_data evaluation_data/full_eval/glue_filtered/multirc.valid.jsonl \
    --task multirc \
    --num_labels 2 \
    --batch_size 16 \
    --learning_rate 3e-5 \
    --num_epochs 10 \
    --sequence_length 512 \
    --results_dir results \
    --save \
    --save_dir models \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed 42 \
    --verbose

# RTE
python -m evaluation_pipeline.finetune.run \
    --model_name_or_path MODEL_PATH \
    --train_data evaluation_data/full_eval/glue_filtered/rte.train.jsonl \
    --valid_data evaluation_data/full_eval/glue_filtered/rte.valid.jsonl \
    --predict_data evaluation_data/full_eval/glue_filtered/rte.valid.jsonl \
    --task rte \
    --num_labels 2 \
    --batch_size 32 \
    --learning_rate 3e-5 \
    --num_epochs 10 \
    --sequence_length 512 \
    --results_dir results \
    --save \
    --save_dir models \
    --metrics accuracy f1 mcc \
    --metric_for_valid accuracy \
    --seed 42
```

## Hardware Requirements

- **GPU**: A100 (recommended) or similar high-memory GPU
- **RAM**: 32GB+ recommended
- **Storage**: 50GB+ free space for models and evaluation data

## Troubleshooting

### Common Issues

1. **HuggingFace Authentication**: Ensure you're logged in with `huggingface-cli login`
2. **EWoK Access**: Make sure you have approval for EWoK dataset on HuggingFace
3. **Memory Issues**: Reduce batch sizes if encountering OOM errors
4. **Permission Errors**: Ensure shell scripts are executable with `chmod +x *.sh`

### Performance Optimization

- Use `CUDA_VISIBLE_DEVICES=0` to specify GPU
- Set `export TRANSFORMERS_CACHE=/path/to/cache` for model caching
- Use gradient checkpointing for memory efficiency

## Expected Runtime

- **Fast Evaluation**: ~30-60 minutes per model
- **Full Evaluation**: ~2-4 hours per model  
- **Fine-tuning**: ~1-2 hours per model per task

## Collecting Results

After all evaluations complete, results will be in JSON format in the `results/` directory. You can use the built-in collation script:

```bash
python -m evaluation_pipeline.collate_preds
```

This will generate summary statistics and performance metrics for all evaluated models.

## Notes

- All models are evaluated as causal language models
- The evaluation uses the text-only strict track configuration
- Results include both zero-shot and fine-tuned performance metrics
- The pipeline automatically handles model loading and tokenization
