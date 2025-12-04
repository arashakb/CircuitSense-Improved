# Testing Fine-Tuned Model on CircuitSense Benchmark

## Your Fine-Tuned Model Location

Your fine-tuned LoRA adapter is saved at:
```
outputs/circuitsense_finetune/final_model/
```

This is a LoRA adapter (not a merged model), which means:
- The adapter weights are only ~646 MB (instead of full 14GB+ model)
- It will automatically load the base model (Qwen/Qwen2-VL-7B-Instruct) and apply the adapter
- The evaluation script handles this automatically

## Testing Your Fine-Tuned Model

### 1. Quick Test (1-5 questions)

```bash
conda activate circuitsense
cd /home/arash/CircuitSense-Improved

PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
    --model outputs/circuitsense_finetune/final_model \
    --dataset_path datasets/symbolic_level15_27 \
    --mode full \
    --max_questions 5 \
    --no_flash_attention
```

### 2. Inference Only (Generate responses for later evaluation)

```bash
PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
    --model outputs/circuitsense_finetune/final_model \
    --dataset_path datasets/symbolic_level15_27 \
    --mode inference \
    --max_questions 50 \
    --no_flash_attention
```

### 3. Evaluation Only (Evaluate existing saved responses)

```bash
PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
    --model outputs/circuitsense_finetune/final_model \
    --dataset_path datasets/symbolic_level15_27 \
    --mode evaluation \
    --max_questions 50 \
    --no_flash_attention
```

### 4. Full Evaluation (All questions)

```bash
PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
    --model outputs/circuitsense_finetune/final_model \
    --dataset_path datasets/symbolic_level15_27 \
    --mode full \
    --max_questions 600 \
    --no_flash_attention
```

## Comparing Base vs Fine-Tuned Models

To compare performance before and after fine-tuning:

### Step 1: Test Base Model (if not already done)

```bash
PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
    --model Qwen/Qwen2-VL-7B-Instruct \
    --dataset_path datasets/symbolic_level15_27 \
    --mode full \
    --max_questions 50 \
    --no_flash_attention
```

Results saved to: `results/results_qwen2-vl-7b-instruct_TIMESTAMP.csv`

### Step 2: Test Fine-Tuned Model

```bash
PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
    --model outputs/circuitsense_finetune/final_model \
    --dataset_path datasets/symbolic_level15_27 \
    --mode full \
    --max_questions 50 \
    --no_flash_attention
```

Results saved to: `results/results_circuitsense_finetune_TIMESTAMP.csv`

### Step 3: Compare Results

Compare the accuracy from both CSV files:
- Base model accuracy: Check the CSV file for base model
- Fine-tuned accuracy: Check the CSV file for fine-tuned model
- Improvement: Fine-tuned accuracy - Base accuracy

## Model Loading Details

The script automatically detects LoRA adapters by checking for `adapter_config.json`. When it finds one:

1. Loads the base model from HuggingFace (`Qwen/Qwen2-VL-7B-Instruct`)
2. Loads your LoRA adapter weights
3. Combines them for inference

This is handled automatically - you just need to provide the path to the `final_model` directory.

## Notes

- **First run**: The base model will be downloaded from HuggingFace if not cached (takes time)
- **Flash Attention**: Use `--no_flash_attention` if flash-attn is not installed
- **Model Tag**: Responses will be saved with tag `circuitsense_finetune` (from the model path)
- **Results Location**: All results saved to `results/` directory

## Example Output

After running evaluation, you'll see:

```
======================================================================
EVALUATION RESULTS SUMMARY
======================================================================
Total Questions: 50
Evaluated (excl. skipped): 48
Correct Answers: 32
Accuracy: 66.67%
Total Time: 1320.5 seconds
Average Time per Question: 26.4 seconds
```

Then check the CSV file for detailed results per question.

