# Fine-Tuning Pipeline for Qwen2-VL on CircuitSense

## Overview

This document describes the fine-tuning pipeline to train Qwen2-VL on circuit understanding tasks using LoRA/QLoRA with HuggingFace Transformers.

## Target Configuration

- **Model**: Qwen2-VL-7B-Instruct (or 72B variant)
- **Method**: LoRA/QLoRA for parameter-efficient fine-tuning
- **Tasks**: Circuit Q&A and symbolic equation derivation
- **Framework**: HuggingFace Transformers + PEFT
- **Trainable Parameters**: ~161M (1.9% of 8.4B total)

## Directory Structure

```
training/
├── configs/
│   └── qwen2vl_lora.yaml        # Training hyperparameters
├── data/
│   ├── __init__.py              # Module exports
│   ├── dataset.py               # CircuitSense dataset (generated data)
│   ├── hf_dataset.py            # HuggingFace dataset loader
│   └── collator.py              # Data collator for batching
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
├── inference.py                 # Inference/demo script
└── requirements_training.txt    # Training dependencies
```

## Data Sources

### Option 1: HuggingFace Dataset (Recommended)

Use the pre-built CircuitSense dataset from HuggingFace: `armanakbari4/CircuitSense`

**Download dataset:**
```bash
# Download a subset (e.g., 100 samples for testing)
PYTHONPATH=. python training/data/hf_dataset.py \
  --output_dir datasets/circuitsense_hf \
  --max_samples 100

# Download full dataset (~7800 samples)
PYTHONPATH=. python training/data/hf_dataset.py \
  --output_dir datasets/circuitsense_hf
```

The HuggingFace dataset contains:
- Circuit diagram images (PNG)
- Questions about circuit analysis
- Answers with explanations
- Multiple difficulty levels (Perception, Analysis, Design)

### Option 2: Generated Dataset

Generate synthetic circuit data locally:
```bash
# Generate circuits
PYTHONPATH=. python main.py --note training_data --gen_num 1000 --symbolic

# Derive equations (requires lcapy)
PYTHONPATH=. python main.py --note training_data --skip_generation --skip_visualization --derive_equations
```

## Quick Start

### 1. Install Dependencies

```bash
pip install transformers peft accelerate bitsandbytes tqdm pillow pyyaml sympy
pip install huggingface_hub  # For HF dataset
```

### 2. Download Dataset

```bash
# Download 100 samples for testing
HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONPATH=. python training/data/hf_dataset.py \
  --output_dir datasets/circuitsense_hf \
  --max_samples 100
```

### 3. Run Training

```bash
HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONPATH=. python training/train.py \
  --config training/configs/qwen2vl_lora.yaml \
  --data_dir datasets/circuitsense_hf \
  --dataset_type huggingface \
  --output_dir outputs/circuitsense_finetune \
  --epochs 3 \
  --no_wandb
```

### 4. Run Inference

```bash
# Interactive mode
PYTHONPATH=. python training/inference.py \
  --model outputs/circuitsense_finetune/final_model \
  --interactive

# Single image prediction
PYTHONPATH=. python training/inference.py \
  --model outputs/circuitsense_finetune/final_model \
  --image path/to/circuit.png \
  --question "What is the transfer function of this circuit?"
```

## Configuration

### Training Config (`training/configs/qwen2vl_lora.yaml`)

```yaml
# Model configuration
model:
  name: "Qwen/Qwen2-VL-7B-Instruct"
  torch_dtype: "bfloat16"
  use_flash_attention: false  # Set true if flash_attn installed

# Quantization (QLoRA)
quantization:
  enabled: true
  bits: 4
  quant_type: "nf4"
  double_quant: true

# LoRA configuration
lora:
  r: 64
  alpha: 128
  dropout: 0.05
  target_modules: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Training configuration
training:
  output_dir: "outputs/qwen2vl-circuitsense"
  num_epochs: 3
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2e-4
  gradient_checkpointing: true

# Data configuration
data:
  data_dir: "datasets/circuitsense_hf"
  dataset_type: "huggingface"  # or "generated"
  train_split: 0.9
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to YAML config file |
| `--data_dir` | Dataset directory |
| `--dataset_type` | `huggingface` or `generated` |
| `--output_dir` | Output directory for model |
| `--epochs` | Number of training epochs |
| `--batch_size` | Per-device batch size |
| `--learning_rate` | Learning rate |
| `--no_quantization` | Disable QLoRA (use standard LoRA) |
| `--no_wandb` | Disable Weights & Biases logging |

## Dataset Format

### HuggingFace Dataset Structure

After downloading, the dataset has this structure:
```
datasets/circuitsense_hf/
├── images/                    # Circuit diagram images
│   ├── sample_1.png
│   └── ...
├── labels.json               # Image paths and Q&A data
└── circuit_qa_dataset.json   # Detailed Q&A metadata
```

### Generated Dataset Structure

```
datasets/training_data/
├── images/                    # Generated circuit images
├── labels.json               # Circuit metadata
├── symbolic_equations.json   # Derived transfer functions
└── circuit_qa_dataset.json   # Q&A pairs
```

## Evaluation

The evaluation pipeline supports three task types with specialized metrics for each.

### Task Types

| Task Type | Description | Metrics |
|-----------|-------------|---------|
| **Equations** | Transfer function derivation | LaTeX exact match, SymPy equivalence |
| **Q&A (Voltage)** | Node voltage calculations | Numerical accuracy (±1%), unit matching |
| **Q&A (Current)** | Component current calculations | Numerical accuracy (±1%), unit matching |

### Running Evaluation

```bash
# Evaluate on HuggingFace dataset
HF_HUB_ENABLE_HF_TRANSFER=0 PYTHONPATH=. python training/evaluate.py \
  --model outputs/circuitsense_finetune/final_model \
  --data_dir datasets/circuitsense_hf \
  --no_flash_attention \
  --output results/eval_results.json

# Evaluate on generated dataset with equations
PYTHONPATH=. python training/evaluate.py \
  --model outputs/circuitsense_finetune/final_model \
  --data_dir datasets/training_data \
  --task_type equations \
  --no_flash_attention

# Evaluate only Q&A tasks
PYTHONPATH=. python training/evaluate.py \
  --model outputs/circuitsense_finetune/final_model \
  --data_dir datasets/circuitsense_hf \
  --task_type qa \
  --max_samples 100 \
  --no_flash_attention
```

### Evaluation Arguments

| Argument | Description |
|----------|-------------|
| `--model` | Path to fine-tuned model (required) |
| `--data_dir` | Path to evaluation dataset (required) |
| `--task_type` | `equations`, `qa`, or `both` (default: both) |
| `--output` | JSON file for detailed results |
| `--batch_size` | Batch size for inference (default: 1) |
| `--max_samples` | Limit number of samples to evaluate |
| `--no_flash_attention` | Disable flash attention (use if not installed) |

### Evaluation Metrics

**For Equation Tasks:**
- `equation_exact_accuracy`: Normalized LaTeX string match
- `equation_sympy_accuracy`: Mathematical equivalence via SymPy

**For Q&A Tasks:**
- `qa_numerical_accuracy`: Value within ±1% tolerance
- `qa_unit_accuracy`: Correct unit extraction (V, A, Ω, etc.)

### Output Format

The `--output` flag saves detailed results as JSON:
```json
{
  "metrics": {
    "equation_exact_accuracy": 0.85,
    "equation_sympy_accuracy": 0.92,
    "qa_numerical_accuracy": 0.78,
    "qa_unit_accuracy": 0.95
  },
  "predictions": [
    {
      "circuit_id": "circuit_001",
      "task_type": "equation",
      "prediction": "H(s) = 1/(1 + sRC)",
      "ground_truth": "H(s) = 1/(1 + sRC)",
      "exact_match": true,
      "sympy_match": true
    }
  ]
}
```

### Evaluation Functions

The evaluation utilities can also be used programmatically:

```python
from training.evaluate import (
    normalize_latex,
    check_sympy_equivalence,
    extract_numerical_answer,
    compute_numerical_accuracy,
)

# Equation evaluation
pred_norm = normalize_latex("$\\frac{1}{1+sRC}$")
is_equiv = check_sympy_equivalence("$x+y$", "$y+x$")

# Q&A evaluation
value, unit = extract_numerical_answer("The voltage is 5.5 V")
is_correct = compute_numerical_accuracy(5.55, 5.5, tolerance=0.01)
```

## Hardware Requirements

| Configuration | GPU Memory | Batch Size |
|--------------|------------|------------|
| QLoRA (4-bit) | ~16 GB | 1-2 |
| LoRA (fp16) | ~32 GB | 1-2 |
| Full fine-tune | ~80 GB | 1 |

**Tested on:** NVIDIA RTX 5090 (33.7 GB)

## Troubleshooting

### Common Issues

1. **`HF_HUB_ENABLE_HF_TRANSFER` error**
   ```bash
   # Disable fast transfer
   HF_HUB_ENABLE_HF_TRANSFER=0 python ...
   ```

2. **Flash Attention not installed**
   - Set `use_flash_attention: false` in config
   - Or install: `pip install flash-attn`

3. **Out of Memory**
   - Reduce `batch_size` to 1
   - Enable `gradient_checkpointing: true`
   - Use QLoRA (`quantization.enabled: true`)

4. **No training samples**
   - Check `train_split` ratio (use 1.0 for small datasets)
   - Verify dataset files exist in `data_dir`

5. **SymPy LaTeX parsing warnings**
   - Install ANTLR4: `pip install antlr4-python3-runtime`
   - SymPy equivalence still works but falls back to string comparison

## Model Output

After training, the output directory contains:
```
outputs/circuitsense_finetune/final_model/
├── adapter_config.json       # LoRA configuration
├── adapter_model.safetensors # LoRA weights (~645 MB)
├── tokenizer.json            # Tokenizer
├── preprocessor_config.json  # Image processor config
└── ...
```

## Merging LoRA Weights

To merge LoRA weights into the base model for faster inference:
```bash
PYTHONPATH=. python training/inference.py \
  --model outputs/circuitsense_finetune/final_model \
  --merge outputs/merged_model
```

## Dataset Comparison

### Task Coverage by Dataset

| Task Type | HuggingFace Dataset | Generated Dataset |
|-----------|---------------------|-------------------|
| **Equations** (transfer functions) | ❌ Not available | ✅ Available via `symbolic_equations.json` |
| **Q&A (Voltage/Current)** | ✅ Available | ✅ Available via `circuit_qa_dataset.json` |
| **Circuit Images** | ✅ PNG format | ✅ JPG format |
| **Difficulty Levels** | ✅ Perception/Analysis/Design | ❌ Not categorized |

### HuggingFace Dataset (`armanakbari4/CircuitSense`)

**Contents:**
- ~7800 circuit Q&A samples
- Three difficulty levels: Perception, Analysis, Design
- Questions about voltage, current, resistance, transfer functions
- Circuit diagram images (PNG)

**Limitations:**
- No symbolic equation derivation tasks
- Answers may be simplified (some placeholder values in subsets)
- No structured `symbolic_equations.json` file

**Best for:** Q&A training and evaluation, large-scale training

### Generated Dataset

**Contents:**
- Synthetic circuits with known ground truth
- Transfer function derivation (`symbolic_equations.json`)
- Voltage/current Q&A pairs (`circuit_qa_dataset.json`)
- Circuit images (JPG)

**Limitations:**
- Requires local generation (slower)
- Needs lcapy for equation derivation
- Circuit complexity depends on generation parameters

**Best for:** Equation tasks, controlled experiments, custom circuit types

### Testing Summary

During pipeline testing, we used:

| Test | Dataset Used | Purpose |
|------|--------------|---------|
| Training smoke test | HuggingFace (10 samples) | Verify training pipeline |
| Equation evaluation | Generated (1 sample) | Test SymPy equivalence |
| Q&A evaluation | HuggingFace (3 samples) | Test numerical accuracy |
| Unit tests | Mock data | Test utility functions |

### Recommendations

1. **For Q&A training**: Use HuggingFace dataset (larger, more diverse)
2. **For equation training**: Use generated dataset (has `symbolic_equations.json`)
3. **For mixed training**: Combine both datasets
4. **For evaluation**: Match dataset to task type being evaluated

## Key Design Decisions

1. **HuggingFace Trainer over TRL SFTTrainer**: Better compatibility with custom PyTorch datasets
2. **QLoRA by default**: Memory efficient, enables training on consumer GPUs
3. **Dual dataset support**: HuggingFace for convenience, generated for custom circuits
4. **Conversation format**: Qwen2-VL native chat template for optimal performance
