#!/usr/bin/env python3
"""Evaluate Qwen2-VL-7B-Instruct on the CircuitSense symbolic equations benchmark.

This script evaluates Qwen2-VL models (base or fine-tuned) on the symbolic equations
dataset without using APIs. It loads the model locally and reuses the equation comparison
logic from benchmark_symbolic_equations.py.

Usage:
    # Evaluate base model
    PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
        --model Qwen/Qwen2-VL-7B-Instruct \
        --dataset_path datasets/symbolic_level15_27 \
        --mode inference \
        --max_questions 10

    # Evaluate fine-tuned model (LoRA adapter)
    PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
        --model outputs/circuitsense_finetune/final_model \
        --dataset_path datasets/symbolic_level15_27 \
        --mode full \
        --max_questions 50

    # Evaluate only (on existing saved responses)
    PYTHONPATH=. python evaluation/evaluate_qwen2vl.py \
        --model outputs/circuitsense_finetune/final_model \
        --dataset_path datasets/symbolic_level15_27 \
        --mode full \
        --max_questions 50
"""

import argparse
import json
import logging
import os
import re
import signal
import sys
import time
from functools import wraps
from multiprocessing import Process, Queue
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Import equation comparison logic from the existing benchmark
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation.benchmark_symbolic_equations import (
    SymbolicEquationBenchmark,
    GLOBAL_COMPARE_TIMEOUT_SECONDS,
    STRATEGY_TIMEOUT_SECONDS,
    timeout,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class Qwen2VLEvaluator:
    """Evaluator for Qwen2-VL models on symbolic equations benchmark."""

    def __init__(
        self,
        model_path: str,
        dataset_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
        temperature: float = 0.1,
        max_new_tokens: int = 1024,
    ):
        """Initialize evaluator.

        Args:
            model_path: Path to model (HuggingFace model ID or local path)
            dataset_path: Path to symbolic equations dataset
            device: Device to use
            torch_dtype: Torch dtype for inference
            use_flash_attention: Whether to use flash attention
            temperature: Sampling temperature
            max_new_tokens: Maximum tokens to generate
        """
        self.device = device
        self.model_path = Path(model_path)
        self.dataset_path = Path(dataset_path)
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens

        logger.info(f"Loading model from: {model_path}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            str(self.model_path) if self.model_path.exists() else model_path,
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

            base_model_name = config.get(
                "base_model_name_or_path", "Qwen/Qwen2-VL-7B-Instruct"
            )

            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                base_model_name,
                **model_kwargs,
            )
            self.model = PeftModel.from_pretrained(base_model, str(self.model_path))
        else:
            # Load merged model or base model
            logger.info("Loading full model...")
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_path if isinstance(model_path, str) and not self.model_path.exists() else str(self.model_path),
                **model_kwargs,
            )

        self.model.eval()
        logger.info("Model loaded successfully!")

        # Initialize equation benchmark for comparison logic
        self.equation_benchmark = SymbolicEquationBenchmark(
            dataset_path=str(self.dataset_path), init_model=False
        )

        # Model tag for saving responses
        self.model_tag = self._get_model_tag()

    def _get_model_tag(self) -> str:
        """Generate a filesystem-friendly tag for this model."""
        model_str = str(self.model_path)
        # Extract model name from path
        if "/" in model_str:
            parts = model_str.split("/")
            tag = parts[-1] if parts[-1] else parts[-2]
        else:
            tag = model_str

        # Sanitize
        tag = re.sub(r"[^A-Za-z0-9._-]", "_", tag).lower()
        return tag

    @torch.no_grad()
    def predict(self, image_path: Path, prompt: str) -> str:
        """Generate prediction for a single image and prompt.

        Args:
            image_path: Path to image
            prompt: Question/prompt text

        Returns:
            Generated response
        """
        # Load image
        image = Image.open(image_path).convert("RGB")

        # Format message for Qwen2-VL
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Apply chat template
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # Process inputs
        inputs = self.processor(
            text=[text],
            images=[[image]],
            padding=True,
            return_tensors="pt",
        )

        # Move to device
        inputs = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in inputs.items()
        }

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            do_sample=self.temperature > 0,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode (only generated tokens)
        generated_ids = outputs[:, inputs["input_ids"].shape[1] :]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return response

    def load_question_data(self, question_dir: str) -> Dict:
        """Load all files for a single question."""
        base_path = self.dataset_path / question_dir

        # Load question text
        with open(base_path / f"{question_dir}_question.txt", "r") as f:
            question = f.read().strip()

        # Load correct answer (symbolic equation)
        with open(base_path / f"{question_dir}_ta.txt", "r") as f:
            correct_answer = f.read().strip()

        # Load image
        image_path = base_path / f"{question_dir}_image.png"
        if not image_path.exists():
            raise FileNotFoundError(f"Could not find image for question {question_dir}")

        return {
            "question": question,
            "correct_answer": correct_answer,
            "image_path": str(image_path),
        }

    def create_prompt(self, question: str) -> str:
        """Create prompt for the model."""
        prompt = f"""You are an expert electrical engineer specializing in circuit analysis. Analyze the circuit diagram and solve for the requested symbolic expression.

**Task:** {question}

**Critical Instructions:**
1. Use EXACT component labels as shown in the circuit (e.g., R1, R2, C1, C2, L1, not generic R, C, L)
2. For Laplace domain, use lowercase 's' as the complex frequency variable
3. Use standard impedances: R for resistors, 1/(s*C) for capacitors, s*L for inductors
4. For op-amps: Apply virtual short (V+ = V-) if in negative feedback, use Ad for gain if specified
5. Simplify the final expression to match standard form

**Response Format:**
You MUST structure your response exactly as follows:

<think>
[Show all your reasoning and intermediate steps here]
- Identify all components and their labels from the circuit
- Define impedances for each component
- Apply appropriate circuit analysis method (nodal, mesh, etc.)
- Write out all equations step by step
- Show algebraic manipulation
- Simplify to get the final form
</think>

<answer>
[Only the final symbolic equation here in the form: H(s) = expression, or Vn1(s) = expression, etc.]
</answer>

Use * for multiplication, / for division, and parentheses for grouping. Keep the answer in simplest form."""
        return prompt

    def save_response_to_file(self, question_dir: str, response_text: str) -> None:
        """Save the model's response to a text file in the question folder."""
        try:
            output_path = self.dataset_path / question_dir / f"{question_dir}_{self.model_tag}.txt"
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response_text)
            logger.info(f"Saved response to: {output_path}")
        except Exception as e:
            logger.warning(f"Could not save response to file: {str(e)}")

    def load_response_from_file(self, question_dir: str) -> str:
        """Load the model's response from a text file in the question folder."""
        try:
            response_path = (
                self.dataset_path / question_dir / f"{question_dir}_{self.model_tag}.txt"
            )
            if not response_path.exists():
                raise FileNotFoundError(
                    f"No saved response found for {question_dir} ({self.model_tag})"
                )

            with open(response_path, "r", encoding="utf-8") as f:
                response_text = f.read()
            return response_text
        except Exception as e:
            logger.error(f"Error loading response for {question_dir}: {str(e)}")
            return ""

    def evaluate_question(self, question_data: Dict, question_dir: str) -> Dict:
        """Evaluate a single question."""
        prompt = self.create_prompt(question_data["question"])

        try:
            # Generate response
            response_text = self.predict(
                Path(question_data["image_path"]), prompt
            )

            # Save the full response
            self.save_response_to_file(question_dir, response_text)

            logger.info("\n" + "=" * 60)
            logger.info("FULL RESPONSE:")
            logger.info("-" * 60)
            logger.info(response_text)
            logger.info("=" * 60)

            # Extract equation and thinking process
            model_answer, thinking = (
                self.equation_benchmark.extract_equation_from_response(response_text)
            )

            # Display extracted components
            if thinking:
                logger.info("\nTHINKING PROCESS (first 500 chars):")
                logger.info("-" * 60)
                logger.info(
                    thinking[:500] + "..." if len(thinking) > 500 else thinking
                )
                logger.info("-" * 60)

            logger.info(f"\nEXTRACTED EQUATION: {model_answer}")
            logger.info(f"EXPECTED ANSWER:    {question_data['correct_answer']}")

            # Compare with correct answer using SymPy with a hard global timeout
            is_correct, comparison_details = (
                self.equation_benchmark.compare_with_global_timeout(
                    model_answer, question_data["correct_answer"]
                )
            )

            logger.info(f"\nIS CORRECT: {is_correct}")
            logger.info(f"DETAILS: {comparison_details}")

            return {
                "model_answer": model_answer,
                "correct_answer": question_data["correct_answer"],
                "is_correct": is_correct,
                "comparison_details": comparison_details,
                "thinking_process": thinking,
                "full_response": response_text,
            }

        except Exception as e:
            logger.error(f"Error evaluating question: {str(e)}")
            # Save error information to file as well
            error_text = f"Error occurred: {str(e)}\n\nNo response generated."
            self.save_response_to_file(question_dir, error_text)

            return {
                "error": str(e),
                "model_answer": None,
                "correct_answer": question_data["correct_answer"],
                "is_correct": False,
                "thinking_process": None,
            }

    def run_inference_only(self, max_questions: int = 600) -> pd.DataFrame:
        """Run inference only and save responses to files."""
        # Get all question directories
        question_dirs = [
            d
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
            and d.startswith("q")
        ]

        # Sort by question number
        question_dirs.sort(key=lambda x: int(x[1:]))

        # Limit to first max_questions
        question_dirs = question_dirs[:max_questions]
        logger.info(
            f"Running inference on {len(question_dirs)} questions: {question_dirs[:10]}{'...' if len(question_dirs) > 10 else ''}"
        )

        start_time = time.time()
        inference_results = []

        for i, q_dir in enumerate(question_dirs):
            logger.info("\n" + "=" * 70)
            logger.info(f"INFERENCE {q_dir} ({i+1}/{len(question_dirs)})")
            logger.info(f"Elapsed time: {time.time() - start_time:.1f} seconds")
            logger.info("=" * 70)

            try:
                # Check if response already exists
                response_path = (
                    self.dataset_path
                    / q_dir
                    / f"{q_dir}_{self.model_tag}.txt"
                )
                if response_path.exists():
                    logger.info(f"Response already exists for {q_dir}, skipping...")
                    inference_results.append(
                        {
                            "question_number": q_dir,
                            "status": "already_exists",
                            "processing_time": time.time() - start_time,
                        }
                    )
                    continue

                # Load question data
                question_data = self.load_question_data(q_dir)
                logger.info(f"Question: {question_data['question']}")

                # Generate prompt and response
                prompt = self.create_prompt(question_data["question"])
                response_text = self.predict(
                    Path(question_data["image_path"]), prompt
                )

                # Save the response to file
                self.save_response_to_file(q_dir, response_text)
                logger.info(f"✓ Generated and saved response for {q_dir}")

                inference_results.append(
                    {
                        "question_number": q_dir,
                        "status": "completed",
                        "processing_time": time.time() - start_time,
                        "response_length": len(response_text),
                    }
                )

            except Exception as e:
                logger.error(f"✗ ERROR generating response for {q_dir}: {str(e)}")
                # Save error information to file
                error_text = (
                    f"Error occurred during inference: {str(e)}\n\nNo response generated."
                )
                self.save_response_to_file(q_dir, error_text)

                inference_results.append(
                    {
                        "question_number": q_dir,
                        "status": "error",
                        "error": str(e),
                        "processing_time": time.time() - start_time,
                    }
                )

            # Save intermediate results every 10 questions
            if (i + 1) % 10 == 0:
                temp_df = pd.DataFrame(inference_results)
                output_dir = Path("results")
                output_dir.mkdir(exist_ok=True)
                temp_df.to_csv(
                    output_dir / f"temp_inference_{self.model_tag}_{i+1}.csv",
                    index=False,
                )
                logger.info(
                    f"Intermediate inference results saved to temp_inference_{self.model_tag}_{i+1}.csv"
                )

        # Convert results to DataFrame
        df = pd.DataFrame(inference_results)

        # Calculate metrics
        total_questions = len(df)
        completed = (
            (df["status"] == "completed").sum() if "status" in df.columns else 0
        )
        already_exists = (
            (df["status"] == "already_exists").sum()
            if "status" in df.columns
            else 0
        )
        errors = (df["status"] == "error").sum() if "status" in df.columns else 0
        total_time = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("INFERENCE RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Questions: {total_questions}")
        logger.info(f"Completed: {completed}")
        logger.info(f"Already Existed: {already_exists}")
        logger.info(f"Errors: {errors}")
        logger.info(f"Total Time: {total_time:.1f} seconds")
        if total_questions > 0:
            logger.info(
                f"Average Time per Question: {total_time/total_questions:.1f} seconds"
            )

        return df

    def run_evaluation_only(self, max_questions: int = 600) -> pd.DataFrame:
        """Run evaluation only on existing saved responses."""
        # Get all question directories
        question_dirs = [
            d
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
            and d.startswith("q")
        ]

        # Sort by question number
        question_dirs.sort(key=lambda x: int(x[1:]))

        # Limit to first max_questions
        question_dirs = question_dirs[:max_questions]
        logger.info(
            f"Evaluating {len(question_dirs)} questions: {question_dirs[:10]}{'...' if len(question_dirs) > 10 else ''}"
        )

        start_time = time.time()
        results = []

        for i, q_dir in enumerate(question_dirs):
            logger.info("\n" + "=" * 70)
            logger.info(f"EVALUATION {q_dir} ({i+1}/{len(question_dirs)})")
            logger.info(f"Elapsed time: {time.time() - start_time:.1f} seconds")
            logger.info("=" * 70)

            try:
                # Load question data
                question_data = self.load_question_data(q_dir)
                logger.info(f"Question: {question_data['question']}")

                # Load existing response
                response_text = self.load_response_from_file(q_dir)
                if not response_text:
                    logger.warning(f"✗ No saved response found for {q_dir}")
                    results.append(
                        {
                            "question_number": q_dir,
                            "question_text": question_data["question"],
                            "error": "No saved response found",
                            "is_correct": False,
                            "processing_time": time.time() - start_time,
                        }
                    )
                    continue

                # Extract equation and thinking process
                model_answer, thinking = (
                    self.equation_benchmark.extract_equation_from_response(
                        response_text
                    )
                )

                # Display extracted components
                if thinking:
                    logger.info("\nTHINKING PROCESS (first 200 chars):")
                    logger.info("-" * 40)
                    logger.info(
                        thinking[:200] + "..."
                        if len(thinking) > 200
                        else thinking
                    )
                    logger.info("-" * 40)

                logger.info(f"\nEXTRACTED EQUATION: {model_answer}")
                logger.info(f"EXPECTED ANSWER:    {question_data['correct_answer']}")

                # Compare with correct answer using SymPy with a hard global timeout
                is_correct, comparison_details = (
                    self.equation_benchmark.compare_with_global_timeout(
                        model_answer, question_data["correct_answer"]
                    )
                )

                logger.info(f"\nIS CORRECT: {is_correct}")
                logger.info(f"DETAILS: {comparison_details}")

                result = {
                    "question_number": q_dir,
                    "question_text": question_data["question"],
                    "model_answer": model_answer,
                    "correct_answer": question_data["correct_answer"],
                    "is_correct": is_correct,
                    "comparison_details": comparison_details,
                    "thinking_process": thinking,
                    "processing_time": time.time() - start_time,
                }

                results.append(result)

                status = (
                    "✓ CORRECT"
                    if is_correct is True
                    else ("– SKIPPED" if pd.isna(is_correct) else "✗ INCORRECT")
                )
                logger.info(f"\n{status} - {q_dir}")

            except Exception as e:
                logger.error(f"✗ ERROR evaluating {q_dir}: {str(e)}")
                # Try to include question text if readable
                qt = None
                try:
                    qt = (
                        self.dataset_path
                        / q_dir
                        / f"{q_dir}_question.txt"
                    ).read_text(encoding="utf-8", errors="ignore").strip()
                except Exception:
                    qt = None
                results.append(
                    {
                        "question_number": q_dir,
                        "question_text": qt,
                        "error": str(e),
                        "is_correct": False,
                        "processing_time": time.time() - start_time,
                    }
                )

            # Save intermediate results every 10 questions
            if (i + 1) % 10 == 0:
                temp_df = pd.DataFrame(results)
                output_dir = Path("results")
                output_dir.mkdir(exist_ok=True)
                temp_df.to_csv(
                    output_dir / f"temp_evaluation_{self.model_tag}_{i+1}.csv",
                    index=False,
                )
                logger.info(
                    f"Intermediate evaluation results saved to temp_evaluation_{self.model_tag}_{i+1}.csv"
                )

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Calculate metrics (exclude skipped/None from accuracy)
        total_questions = len(df)
        evaluated = (
            df["is_correct"].notna().sum() if "is_correct" in df.columns else 0
        )
        correct_answers = (
            (df["is_correct"] == True).sum() if "is_correct" in df.columns else 0
        )
        accuracy = (correct_answers / evaluated) if evaluated > 0 else 0
        total_time = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("EVALUATION RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Questions: {total_questions}")
        logger.info(f"Evaluated (excl. skipped): {evaluated}")
        logger.info(f"Correct Answers: {correct_answers}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Total Time: {total_time:.1f} seconds")
        if total_questions > 0:
            logger.info(
                f"Average Time per Question: {total_time/total_questions:.1f} seconds"
            )

        # Grouped accuracy by question type (transfer vs other)
        if "question_text" in df.columns:
            try:
                df["is_transfer"] = df["question_text"].str.contains(
                    r"transfer function", case=False, na=False
                )
                for group_name, group_mask in [
                    ("TRANSFER GROUP", df["is_transfer"] == True),
                    ("OTHER GROUP", df["is_transfer"] == False),
                ]:
                    gdf = df[group_mask]
                    g_total = len(gdf)
                    g_evaluated = gdf["is_correct"].notna().sum()
                    g_correct = (gdf["is_correct"] == True).sum()
                    g_accuracy = (g_correct / g_evaluated) if g_evaluated > 0 else 0
                    logger.info(f"\n[{group_name}]")
                    logger.info(f"  Total: {g_total}")
                    logger.info(f"  Evaluated: {g_evaluated}")
                    logger.info(f"  Correct: {g_correct}")
                    logger.info(f"  Accuracy: {g_accuracy:.2%}")
            except Exception as e:
                logger.warning(f"Failed to compute grouped accuracy: {str(e)}")

        # Show detailed results
        logger.info(f"\nDetailed Results:")
        logger.info("-" * 70)
        for _, row in df.iterrows():
            ic = row.get("is_correct", None)
            status = (
                "✓"
                if ic is True
                else ("–" if pd.isna(ic) else "✗")
            )
            details_val = row.get("comparison_details", None)
            if details_val is None or (
                isinstance(details_val, float) and pd.isna(details_val)
            ):
                details_str = "N/A"
            else:
                details_str = str(details_val)
            if len(details_str) > 50:
                details_str = details_str[:47] + "..."
            logger.info(f"{status} {row['question_number']}: {details_str}")

        return df

    def run_benchmark(self, max_questions: int = 3) -> pd.DataFrame:
        """Run the full benchmark (inference + evaluation)."""
        # Get all question directories
        question_dirs = [
            d
            for d in os.listdir(self.dataset_path)
            if os.path.isdir(os.path.join(self.dataset_path, d))
            and d.startswith("q")
        ]

        # Sort by question number
        question_dirs.sort(key=lambda x: int(x[1:]))

        # Limit to first max_questions
        question_dirs = question_dirs[:max_questions]
        logger.info(
            f"Processing first {len(question_dirs)} questions: {question_dirs}"
        )

        start_time = time.time()
        results = []

        for i, q_dir in enumerate(question_dirs):
            logger.info("\n" + "=" * 70)
            logger.info(f"QUESTION {q_dir} ({i+1}/{len(question_dirs)})")
            logger.info(f"Elapsed time: {time.time() - start_time:.1f} seconds")
            logger.info("=" * 70)

            try:
                # Load question data
                question_data = self.load_question_data(q_dir)
                logger.info(f"Question: {question_data['question']}")

                # Evaluate question
                result = self.evaluate_question(question_data, q_dir)
                result["question_number"] = q_dir
                result["question_text"] = question_data["question"]
                result["processing_time"] = time.time() - start_time

                results.append(result)

                ic = result["is_correct"]
                status = (
                    "✓ CORRECT"
                    if ic is True
                    else ("– SKIPPED" if pd.isna(ic) else "✗ INCORRECT")
                )
                logger.info(f"\n{status} - {q_dir}")

            except Exception as e:
                logger.error(f"✗ ERROR processing {q_dir}: {str(e)}")
                results.append(
                    {
                        "question_number": q_dir,
                        "error": str(e),
                        "is_correct": False,
                        "processing_time": time.time() - start_time,
                    }
                )

            # Save intermediate results every 5 questions
            if (i + 1) % 5 == 0:
                temp_df = pd.DataFrame(results)
                output_dir = Path("results")
                output_dir.mkdir(exist_ok=True)
                temp_df.to_csv(
                    output_dir / f"temp_results_{self.model_tag}_{i+1}.csv",
                    index=False,
                )
                logger.info(
                    f"Intermediate results saved to temp_results_{self.model_tag}_{i+1}.csv"
                )

        # Convert results to DataFrame
        df = pd.DataFrame(results)

        # Calculate metrics (exclude skipped/None from accuracy)
        total_questions = len(df)
        evaluated = (
            df["is_correct"].notna().sum() if "is_correct" in df.columns else 0
        )
        correct_answers = (
            (df["is_correct"] == True).sum() if "is_correct" in df.columns else 0
        )
        accuracy = (correct_answers / evaluated) if evaluated > 0 else 0
        total_time = time.time() - start_time

        logger.info("\n" + "=" * 70)
        logger.info("BENCHMARK RESULTS SUMMARY")
        logger.info("=" * 70)
        logger.info(f"Total Questions: {total_questions}")
        logger.info(f"Evaluated (excl. skipped): {evaluated}")
        logger.info(f"Correct Answers: {correct_answers}")
        logger.info(f"Accuracy: {accuracy:.2%}")
        logger.info(f"Total Time: {total_time:.1f} seconds")
        if total_questions > 0:
            logger.info(
                f"Average Time per Question: {total_time/total_questions:.1f} seconds"
            )

        # Grouped accuracy by question type (transfer vs other)
        if "question_text" in df.columns:
            try:
                df["is_transfer"] = df["question_text"].str.contains(
                    r"transfer function", case=False, na=False
                )
                for group_name, group_mask in [
                    ("TRANSFER GROUP", df["is_transfer"] == True),
                    ("OTHER GROUP", df["is_transfer"] == False),
                ]:
                    gdf = df[group_mask]
                    g_total = len(gdf)
                    g_evaluated = gdf["is_correct"].notna().sum()
                    g_correct = (gdf["is_correct"] == True).sum()
                    g_accuracy = (g_correct / g_evaluated) if g_evaluated > 0 else 0
                    logger.info(f"\n[{group_name}]")
                    logger.info(f"  Total: {g_total}")
                    logger.info(f"  Evaluated: {g_evaluated}")
                    logger.info(f"  Correct: {g_correct}")
                    logger.info(f"  Accuracy: {g_accuracy:.2%}")
            except Exception as e:
                logger.warning(f"Failed to compute grouped accuracy: {str(e)}")

        # Show detailed results
        logger.info(f"\nDetailed Results:")
        logger.info("-" * 70)
        for _, row in df.iterrows():
            ic = row.get("is_correct", None)
            status = (
                "✓"
                if ic is True
                else ("–" if pd.isna(ic) else "✗")
            )
            details_val = row.get("comparison_details", None)
            if details_val is None or (
                isinstance(details_val, float) and pd.isna(details_val)
            ):
                details_str = "N/A"
            else:
                details_str = str(details_val)
            if len(details_str) > 50:
                details_str = details_str[:47] + "..."
            logger.info(f"{status} {row['question_number']}: {details_str}")

        return df


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen2-VL on CircuitSense symbolic equations benchmark"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model (HuggingFace model ID or local path)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to symbolic equations dataset",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["inference", "evaluation", "full"],
        default="full",
        help="Mode: inference (only generate), evaluation (only evaluate), full (both)",
    )
    parser.add_argument(
        "--max_questions",
        type=int,
        default=600,
        help="Maximum number of questions to process",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype",
    )
    parser.add_argument(
        "--no_flash_attention",
        action="store_true",
        help="Disable flash attention",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=1024,
        help="Maximum new tokens to generate",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, inference will be slow!")
    else:
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Initialize evaluator
    torch_dtype = getattr(torch, args.torch_dtype)
    evaluator = Qwen2VLEvaluator(
        model_path=args.model,
        dataset_path=args.dataset_path,
        device=args.device,
        torch_dtype=torch_dtype,
        use_flash_attention=not args.no_flash_attention,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # Run based on mode
    if args.mode == "inference":
        results_df = evaluator.run_inference_only(max_questions=args.max_questions)
        output_file = (
            output_dir
            / f"inference_{evaluator.model_tag}_{timestamp}.csv"
        )
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nInference results saved to {output_file}")

    elif args.mode == "evaluation":
        results_df = evaluator.run_evaluation_only(max_questions=args.max_questions)
        output_file = (
            output_dir
            / f"evaluation_{evaluator.model_tag}_{timestamp}.csv"
        )
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nEvaluation results saved to {output_file}")

    else:  # full
        results_df = evaluator.run_benchmark(max_questions=args.max_questions)
        output_file = (
            output_dir / f"results_{evaluator.model_tag}_{timestamp}.csv"
        )
        results_df.to_csv(output_file, index=False)
        logger.info(f"\nBenchmark results saved to {output_file}")


if __name__ == "__main__":
    main()

