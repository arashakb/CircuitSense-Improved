#!/usr/bin/env python3
"""Evaluation script for fine-tuned Qwen2-VL on CircuitSense.

This script evaluates model performance on both symbolic equation
derivation and circuit Q&A tasks.

Usage:
    PYTHONPATH=. python training/evaluate.py --model outputs/qwen2vl-circuitsense/final_model --data_dir datasets/test_data
"""

import argparse
import json
import logging
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

from training.data import CircuitSenseDataset, CircuitSenseEvalCollator

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_latex(latex: str) -> str:
    """Normalize LaTeX string for comparison.

    Removes whitespace and common formatting variations.
    """
    if not latex:
        return ""

    # Remove $ delimiters
    latex = latex.strip("$")

    # Remove whitespace
    latex = re.sub(r"\s+", "", latex)

    # Normalize common LaTeX commands
    latex = latex.replace(r"\cdot", "*")
    latex = latex.replace(r"\times", "*")
    latex = latex.replace(r"\frac", "frac")

    return latex.lower()


def check_sympy_equivalence(pred: str, target: str) -> bool:
    """Check if two expressions are mathematically equivalent using SymPy.

    Args:
        pred: Predicted expression
        target: Target expression

    Returns:
        True if expressions are equivalent
    """
    try:
        import sympy
        from sympy.parsing.latex import parse_latex

        # Extract LaTeX from potential surrounding text
        pred_match = re.search(r"\$([^$]+)\$", pred)
        target_match = re.search(r"\$([^$]+)\$", target)

        pred_latex = pred_match.group(1) if pred_match else pred
        target_latex = target_match.group(1) if target_match else target

        # Parse expressions
        pred_expr = parse_latex(pred_latex)
        target_expr = parse_latex(target_latex)

        # Check equivalence
        diff = sympy.simplify(pred_expr - target_expr)
        return diff == 0

    except Exception:
        # If parsing fails, fall back to string comparison
        return False


def extract_numerical_answer(text: str) -> tuple[float | None, str | None]:
    """Extract numerical answer and unit from text.

    Args:
        text: Response text

    Returns:
        Tuple of (value, unit) or (None, None) if extraction fails
    """
    # Pattern for scientific notation with unit
    sci_pattern = r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*([VAΩFHmunpkMG]?[VAΩFHs]?)"

    match = re.search(sci_pattern, text)
    if match:
        try:
            value = float(match.group(1))
            unit = match.group(2) if match.group(2) else None
            return value, unit
        except ValueError:
            pass

    return None, None


def compute_numerical_accuracy(
    pred: float | None,
    target: float | None,
    tolerance: float = 0.01,
) -> bool:
    """Check if prediction is within tolerance of target.

    Args:
        pred: Predicted value
        target: Target value
        tolerance: Relative tolerance (default 1%)

    Returns:
        True if within tolerance
    """
    if pred is None or target is None:
        return False

    if target == 0:
        return abs(pred) < 1e-10

    relative_error = abs(pred - target) / abs(target)
    return relative_error <= tolerance


class CircuitSenseEvaluator:
    """Evaluator for CircuitSense fine-tuned models."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        """Initialize evaluator.

        Args:
            model_path: Path to fine-tuned model
            device: Device to use
            torch_dtype: Torch dtype for inference
            use_flash_attention: Whether to use flash attention
        """
        self.device = device
        self.model_path = Path(model_path)

        logger.info(f"Loading model from: {model_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
        )

        # Check if this is a LoRA model or merged model
        adapter_config = self.model_path / "adapter_config.json"

        model_kwargs = {
            "torch_dtype": torch_dtype,
            "trust_remote_code": True,
            "device_map": "auto",
        }

        if use_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        if adapter_config.exists():
            # Load base model + adapter
            logger.info("Loading LoRA adapter model...")
            with open(adapter_config) as f:
                config = json.load(f)

            base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2-VL-7B-Instruct")

            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_name,
                **model_kwargs,
            )
            self.model = PeftModel.from_pretrained(base_model, model_path)
        else:
            # Load merged model
            logger.info("Loading merged model...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path,
                **model_kwargs,
            )

        self.model.eval()

    @torch.no_grad()
    def generate(
        self,
        batch: dict,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
    ) -> list[str]:
        """Generate responses for a batch.

        Args:
            batch: Batch from data collator
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample

        Returns:
            List of generated responses
        """
        # Move inputs to device
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)

        # Handle image inputs
        pixel_values = batch.get("pixel_values")
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device)

        image_grid_thw = batch.get("image_grid_thw")
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        # Generate
        outputs = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode outputs (only the generated part)
        generated_ids = outputs[:, input_ids.shape[1]:]
        responses = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return responses

    def evaluate(
        self,
        dataset: CircuitSenseDataset,
        batch_size: int = 1,
        max_samples: int | None = None,
    ) -> dict:
        """Evaluate model on dataset.

        Args:
            dataset: Dataset to evaluate
            batch_size: Batch size for evaluation
            max_samples: Maximum samples to evaluate (None for all)

        Returns:
            Dictionary with evaluation metrics
        """
        from torch.utils.data import DataLoader

        collator = CircuitSenseEvalCollator(
            processor=self.processor,
            max_length=2048,
        )

        # Limit samples if specified
        if max_samples is not None and max_samples < len(dataset):
            indices = list(range(max_samples))
            dataset = torch.utils.data.Subset(dataset, indices)

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        results = {
            "equation": {"correct": 0, "total": 0, "sympy_correct": 0},
            "qa": {"correct": 0, "total": 0, "unit_correct": 0},
            "predictions": [],
        }

        logger.info(f"Evaluating on {len(dataset)} samples...")

        for batch in tqdm(dataloader, desc="Evaluating"):
            metadata = batch.pop("metadata")

            # Generate predictions
            predictions = self.generate(batch)

            # Evaluate each prediction
            for pred, meta in zip(predictions, metadata):
                task_type = meta["task_type"]
                ground_truth = meta["ground_truth"]

                result = {
                    "circuit_id": meta["circuit_id"],
                    "task_type": task_type,
                    "prediction": pred,
                    "ground_truth": ground_truth,
                }

                if task_type == "equation":
                    results["equation"]["total"] += 1

                    # Exact match (normalized)
                    pred_norm = normalize_latex(pred)
                    target_norm = normalize_latex(ground_truth)
                    exact_match = pred_norm == target_norm

                    if exact_match:
                        results["equation"]["correct"] += 1

                    # SymPy equivalence
                    sympy_match = check_sympy_equivalence(pred, ground_truth)
                    if sympy_match:
                        results["equation"]["sympy_correct"] += 1

                    result["exact_match"] = exact_match
                    result["sympy_match"] = sympy_match

                elif task_type == "qa":
                    results["qa"]["total"] += 1

                    # Extract numerical values
                    pred_val, pred_unit = extract_numerical_answer(pred)
                    target_val, target_unit = extract_numerical_answer(ground_truth)

                    # Check numerical accuracy
                    num_correct = compute_numerical_accuracy(pred_val, target_val)
                    if num_correct:
                        results["qa"]["correct"] += 1

                    # Check unit
                    if pred_unit and target_unit and pred_unit == target_unit:
                        results["qa"]["unit_correct"] += 1

                    result["numerical_correct"] = num_correct
                    result["predicted_value"] = pred_val
                    result["target_value"] = target_val

                results["predictions"].append(result)

        # Compute final metrics
        metrics = {}

        if results["equation"]["total"] > 0:
            metrics["equation_exact_accuracy"] = (
                results["equation"]["correct"] / results["equation"]["total"]
            )
            metrics["equation_sympy_accuracy"] = (
                results["equation"]["sympy_correct"] / results["equation"]["total"]
            )

        if results["qa"]["total"] > 0:
            metrics["qa_numerical_accuracy"] = (
                results["qa"]["correct"] / results["qa"]["total"]
            )
            metrics["qa_unit_accuracy"] = (
                results["qa"]["unit_correct"] / results["qa"]["total"]
            )

        results["metrics"] = metrics
        return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate fine-tuned Qwen2-VL on CircuitSense"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Path to evaluation dataset directory",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="both",
        choices=["equations", "qa", "both"],
        help="Task type to evaluate",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for detailed results (JSON)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to evaluate",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable flash attention",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Initialize evaluator
    evaluator = CircuitSenseEvaluator(
        model_path=args.model,
        use_flash_attention=not args.no_flash_attention,
    )

    # Load dataset
    dataset = CircuitSenseDataset(
        data_dir=args.data_dir,
        task_type=args.task_type,
        split="val",  # Use validation split for evaluation
        train_ratio=0.0,  # Use all data as validation
    )

    # Run evaluation
    results = evaluator.evaluate(
        dataset=dataset,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
    )

    # Print metrics
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)

    metrics = results["metrics"]
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")

    print("=" * 50)

    # Save detailed results if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
