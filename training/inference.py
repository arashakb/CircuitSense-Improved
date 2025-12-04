#!/usr/bin/env python3
"""Inference script for fine-tuned Qwen2-VL on CircuitSense.

Supports interactive mode, single image inference, and batch processing.

Usage:
    # Interactive mode
    PYTHONPATH=. python training/inference.py --model outputs/qwen2vl-circuitsense/final_model --interactive

    # Single image
    PYTHONPATH=. python training/inference.py --model outputs/qwen2vl-circuitsense/final_model --image circuit.jpg --question "What is the transfer function?"

    # Batch inference
    PYTHONPATH=. python training/inference.py --model outputs/qwen2vl-circuitsense/final_model --data_dir datasets/test_data --output predictions.json
"""

import argparse
import json
import logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CircuitSenseInference:
    """Inference class for CircuitSense fine-tuned models."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        use_flash_attention: bool = True,
    ):
        """Initialize inference engine.

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
        logger.info("Model loaded successfully!")

    @torch.no_grad()
    def predict(
        self,
        image: str | Path | Image.Image,
        question: str,
        max_new_tokens: int = 512,
        temperature: float = 0.1,
        do_sample: bool = False,
    ) -> str:
        """Generate prediction for a single image and question.

        Args:
            image: Path to image or PIL Image
            question: Question to ask about the circuit
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to sample

        Returns:
            Generated response
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")

        # Format message
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question},
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
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        # Generate
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=do_sample,
            pad_token_id=self.processor.tokenizer.pad_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
        )

        # Decode (only generated tokens)
        generated_ids = outputs[:, inputs["input_ids"].shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return response

    def interactive(self):
        """Run interactive inference mode."""
        print("\n" + "=" * 60)
        print("CircuitSense Interactive Inference")
        print("=" * 60)
        print("Commands:")
        print("  - Enter image path and question to get prediction")
        print("  - Type 'quit' or 'exit' to stop")
        print("  - Type 'help' for example questions")
        print("=" * 60 + "\n")

        example_questions = [
            "Derive the transfer function for this circuit.",
            "What is the voltage at node 1?",
            "Calculate the current through R1.",
            "What type of circuit is this?",
            "List all components in this circuit.",
        ]

        while True:
            # Get image path
            image_path = input("\nImage path (or 'quit'): ").strip()

            if image_path.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if image_path.lower() == "help":
                print("\nExample questions:")
                for i, q in enumerate(example_questions, 1):
                    print(f"  {i}. {q}")
                continue

            # Check if image exists
            if not Path(image_path).exists():
                print(f"Error: Image not found: {image_path}")
                continue

            # Get question
            question = input("Question: ").strip()

            if not question:
                print("Error: Please enter a question")
                continue

            # Generate prediction
            try:
                print("\nGenerating response...")
                response = self.predict(image_path, question)
                print(f"\nResponse: {response}")
            except Exception as e:
                print(f"Error during inference: {e}")

    def batch_predict(
        self,
        data_dir: str | Path,
        output_file: str | Path | None = None,
        task_type: str = "both",
        max_samples: int | None = None,
    ) -> list[dict]:
        """Run batch prediction on a dataset.

        Args:
            data_dir: Path to dataset directory
            output_file: Optional output file for predictions
            task_type: Task type to predict
            max_samples: Maximum samples to process

        Returns:
            List of prediction results
        """
        from training.data import CircuitSenseDataset

        # Load dataset
        dataset = CircuitSenseDataset(
            data_dir=data_dir,
            task_type=task_type,
            split="val",
            train_ratio=0.0,  # Use all data
        )

        if max_samples is not None:
            dataset.samples = dataset.samples[:max_samples]

        results = []

        logger.info(f"Running batch prediction on {len(dataset)} samples...")

        for i in tqdm(range(len(dataset)), desc="Predicting"):
            sample = dataset[i]

            image_path = sample["image_path"]
            question = sample["messages"][0]["content"][1]["text"]
            ground_truth = sample["messages"][1]["content"][0]["text"]

            try:
                prediction = self.predict(image_path, question)
            except Exception as e:
                logger.warning(f"Error predicting sample {i}: {e}")
                prediction = f"ERROR: {e}"

            results.append({
                "circuit_id": sample["circuit_id"],
                "task_type": sample["task_type"],
                "image_path": image_path,
                "question": question,
                "prediction": prediction,
                "ground_truth": ground_truth,
            })

        # Save results if output file specified
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

            logger.info(f"Predictions saved to: {output_path}")

        return results


def merge_lora_weights(
    model_path: str,
    output_path: str,
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Merge LoRA weights into base model for faster inference.

    Args:
        model_path: Path to LoRA adapter model
        output_path: Path to save merged model
    """
    logger.info(f"Merging LoRA weights from: {model_path}")

    model_path = Path(model_path)
    adapter_config = model_path / "adapter_config.json"

    if not adapter_config.exists():
        logger.error("No adapter_config.json found. Is this a LoRA model?")
        return

    with open(adapter_config) as f:
        config = json.load(f)

    base_model_name = config.get("base_model_name_or_path", "Qwen/Qwen2-VL-7B-Instruct")

    # Load base model
    logger.info(f"Loading base model: {base_model_name}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        base_model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
        device_map="auto",
    )

    # Load LoRA adapter
    logger.info("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, str(model_path))

    # Merge weights
    logger.info("Merging weights...")
    model = model.merge_and_unload()

    # Save merged model
    logger.info(f"Saving merged model to: {output_path}")
    model.save_pretrained(output_path)

    # Also save processor
    processor = AutoProcessor.from_pretrained(str(model_path), trust_remote_code=True)
    processor.save_pretrained(output_path)

    logger.info("Merge complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Inference with fine-tuned Qwen2-VL on CircuitSense"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to fine-tuned model",
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--interactive",
        action="store_true",
        help="Run interactive inference mode",
    )
    mode_group.add_argument(
        "--merge",
        type=str,
        default=None,
        help="Merge LoRA weights and save to specified path",
    )

    # Single prediction
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to single image for prediction",
    )
    parser.add_argument(
        "--question",
        type=str,
        default=None,
        help="Question for single prediction",
    )

    # Batch prediction
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset directory for batch prediction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for batch predictions",
    )
    parser.add_argument(
        "--task_type",
        type=str,
        default="both",
        choices=["equations", "qa", "both"],
        help="Task type for batch prediction",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples for batch prediction",
    )

    # Inference options
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
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

    # Handle merge mode
    if args.merge:
        merge_lora_weights(args.model, args.merge)
        return

    # Initialize inference engine
    engine = CircuitSenseInference(
        model_path=args.model,
        use_flash_attention=not args.no_flash_attention,
    )

    # Interactive mode
    if args.interactive:
        engine.interactive()
        return

    # Single prediction mode
    if args.image and args.question:
        response = engine.predict(
            image=args.image,
            question=args.question,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"\nResponse: {response}")
        return

    # Batch prediction mode
    if args.data_dir:
        results = engine.batch_predict(
            data_dir=args.data_dir,
            output_file=args.output,
            task_type=args.task_type,
            max_samples=args.max_samples,
        )

        # Print summary
        print(f"\nProcessed {len(results)} samples")
        if args.output:
            print(f"Results saved to: {args.output}")
        return

    # No mode specified
    print("Please specify a mode:")
    print("  --interactive         Run interactive mode")
    print("  --image + --question  Single prediction")
    print("  --data_dir            Batch prediction")
    print("  --merge <output_path> Merge LoRA weights")


if __name__ == "__main__":
    main()
