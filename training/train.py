#!/usr/bin/env python3
"""Main training script for Qwen2-VL fine-tuning on CircuitSense.

This script implements LoRA/QLoRA fine-tuning for Qwen2-VL using
HuggingFace Transformers, PEFT, and TRL libraries.

Usage:
    PYTHONPATH=. python training/train.py --config training/configs/qwen2vl_lora.yaml
    PYTHONPATH=. python training/train.py --config training/configs/qwen2vl_lora.yaml --data_dir datasets/my_data
"""

import argparse
import logging
import os
from datetime import datetime
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import Trainer

from training.data import (
    CircuitSenseDataset,
    CircuitSenseCollator,
    CircuitSenseHFDataset,
    QAFolderDataset,
    load_qa_folder_data,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def merge_config_with_args(config: dict, args: argparse.Namespace) -> dict:
    """Merge command line arguments with config file."""
    # Override config with command line args if provided
    if args.data_dir:
        config["data"]["data_dir"] = args.data_dir
    if args.output_dir:
        config["training"]["output_dir"] = args.output_dir
    if args.epochs:
        config["training"]["num_epochs"] = args.epochs
    if args.batch_size:
        config["training"]["per_device_train_batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.no_quantization:
        config["quantization"]["enabled"] = False
    if args.wandb_project:
        config["logging"]["project_name"] = args.wandb_project
    if args.no_wandb:
        config["logging"]["report_to"] = "none"
    if hasattr(args, 'dataset_type') and args.dataset_type:
        config["data"]["dataset_type"] = args.dataset_type

    return config


def setup_quantization(config: dict) -> BitsAndBytesConfig | None:
    """Setup quantization configuration for QLoRA."""
    quant_config = config.get("quantization", {})

    if not quant_config.get("enabled", False):
        return None

    compute_dtype = getattr(torch, quant_config.get("compute_dtype", "bfloat16"))

    return BitsAndBytesConfig(
        load_in_4bit=quant_config.get("bits", 4) == 4,
        load_in_8bit=quant_config.get("bits", 4) == 8,
        bnb_4bit_quant_type=quant_config.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=quant_config.get("double_quant", True),
        bnb_4bit_compute_dtype=compute_dtype,
    )


def setup_lora(config: dict) -> LoraConfig:
    """Setup LoRA configuration."""
    lora_config = config.get("lora", {})

    return LoraConfig(
        r=lora_config.get("r", 64),
        lora_alpha=lora_config.get("alpha", 128),
        lora_dropout=lora_config.get("dropout", 0.05),
        target_modules=lora_config.get("target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )


def load_model_and_processor(config: dict):
    """Load Qwen2-VL model and processor."""
    model_config = config.get("model", {})
    model_name = model_config.get("name", "Qwen/Qwen2-VL-7B-Instruct")

    logger.info(f"Loading model: {model_name}")

    # Setup quantization
    bnb_config = setup_quantization(config)

    # Determine torch dtype
    torch_dtype = getattr(torch, model_config.get("torch_dtype", "bfloat16"))

    # Load processor
    if AutoProcessor is None:
        raise ImportError(
            "AutoProcessor not available. Please upgrade transformers to >= 4.30.0 "
            "or ensure Qwen2VLProcessor is available."
        )
    
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=model_config.get("trust_remote_code", True),
    )

    # Load model
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": model_config.get("trust_remote_code", True),
        "device_map": "auto",
    }

    if bnb_config is not None:
        model_kwargs["quantization_config"] = bnb_config

    if model_config.get("use_flash_attention", True):
        model_kwargs["attn_implementation"] = "flash_attention_2"

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        **model_kwargs,
    )

    # Prepare model for k-bit training if using quantization
    if bnb_config is not None:
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=False,  # Enable after LoRA setup
        )

    # Setup LoRA BEFORE enabling gradient checkpointing
    lora_config = setup_lora(config)
    model = get_peft_model(model, lora_config)

    # Enable gradient checkpointing AFTER LoRA setup (if requested)
    if config.get("training", {}).get("gradient_checkpointing", True):
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        elif hasattr(model, "model") and hasattr(model.model, "gradient_checkpointing_enable"):
            model.model.gradient_checkpointing_enable()

    # Enable input gradients for vision-language models
    # This allows gradients to flow through even when vision encoder is frozen
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        elif hasattr(model, "model") and hasattr(model.model, "enable_input_require_grads"):
            model.model.enable_input_require_grads()
    except Exception as e:
        logger.warning(f"Could not enable input require grads (may be OK): {e}")

    # Ensure model is in training mode
    model.train()
    
    # Verify LoRA adapters are properly set up and trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable_params == 0:
        logger.error("No trainable parameters found! LoRA adapters may not be properly configured.")
        raise RuntimeError("No trainable parameters in model. Check LoRA configuration.")

    # Print trainable parameters
    model.print_trainable_parameters()
    
    logger.info(f"Model training mode: {model.training}")
    logger.info(f"Total trainable parameters: {trainable_params:,}")

    return model, processor


def load_datasets(config: dict):
    """Load training and validation datasets.

    Supports three dataset types:
    - "qa_folder": Auto-detected QA folder structure (q1/, q2/, etc.)
    - "generated": Use CircuitSenseDataset for locally generated data
    - "huggingface": Use CircuitSenseHFDataset for HuggingFace data
    """
    data_config = config.get("data", {})
    data_dir = Path(data_config.get("data_dir", "datasets/training_data"))
    dataset_type = data_config.get("dataset_type", "auto")  # Default to auto-detect

    logger.info(f"Loading datasets from: {data_dir} (type: {dataset_type})")

    # Auto-detect QA folder structure if type is "auto"
    if dataset_type == "auto":
        # Check if directory contains q* folders (QA folder structure)
        q_folders = [
            d
            for d in data_dir.iterdir()
            if d.is_dir() and d.name.startswith("q") and d.name[1:].isdigit()
        ]
        if q_folders:
            logger.info(f"Auto-detected QA folder structure ({len(q_folders)} question folders)")
            dataset_type = "qa_folder"
        # Check if labels.json exists (generated structure)
        elif (data_dir / "labels.json").exists():
            logger.info("Auto-detected generated dataset structure")
            dataset_type = "generated"
        else:
            logger.warning("Could not auto-detect dataset type, defaulting to 'generated'")
            dataset_type = "generated"

    if dataset_type == "qa_folder":
        # Use QA folder dataset loader
        train_dataset, val_dataset = load_qa_folder_data(
            data_dir=data_dir,
            train_ratio=data_config.get("train_split", 0.9),
            seed=config.get("training", {}).get("seed", 42),
        )
    elif dataset_type == "huggingface":
        # Use HuggingFace dataset loader
        train_dataset = CircuitSenseHFDataset(
            data_dir=data_dir,
            split="train",
            train_ratio=data_config.get("train_split", 0.9),
        )

        val_dataset = CircuitSenseHFDataset(
            data_dir=data_dir,
            split="val",
            train_ratio=data_config.get("train_split", 0.9),
        )
    else:
        # Use original generated dataset loader
        train_dataset = CircuitSenseDataset(
            data_dir=data_dir,
            task_type=data_config.get("task_type", "both"),
            split="train",
            train_ratio=data_config.get("train_split", 0.9),
            augment_questions=data_config.get("augment_questions", True),
        )

        val_dataset = CircuitSenseDataset(
            data_dir=data_dir,
            task_type=data_config.get("task_type", "both"),
            split="val",
            train_ratio=data_config.get("train_split", 0.9),
            augment_questions=False,  # No augmentation for validation
        )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")

    return train_dataset, val_dataset


def setup_training_arguments(config: dict) -> TrainingArguments:
    """Setup HuggingFace training arguments."""
    train_config = config.get("training", {})
    log_config = config.get("logging", {})

    # Generate run name if not provided
    run_name = log_config.get("run_name")
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"circuitsense_{timestamp}"

    output_dir = train_config.get("output_dir", "outputs/qwen2vl-circuitsense")

    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_config.get("num_epochs", 3),
        per_device_train_batch_size=train_config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=train_config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 8),
        learning_rate=train_config.get("learning_rate", 2e-4),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_ratio=train_config.get("warmup_ratio", 0.03),
        lr_scheduler_type=train_config.get("lr_scheduler_type", "cosine"),
        gradient_checkpointing=train_config.get("gradient_checkpointing", True),
        optim=train_config.get("optim", "adamw_torch"),
        fp16=train_config.get("fp16", False),
        bf16=train_config.get("bf16", True),
        max_grad_norm=train_config.get("max_grad_norm", 1.0),
        seed=train_config.get("seed", 42),
        # Logging
        report_to=log_config.get("report_to", "wandb"),
        run_name=run_name,
        logging_steps=log_config.get("logging_steps", 10),
        # Checkpointing
        save_strategy=log_config.get("save_strategy", "steps"),
        save_steps=log_config.get("save_steps", 500),
        save_total_limit=log_config.get("save_total_limit", 3),
        # Evaluation
        eval_strategy=log_config.get("eval_strategy", "steps"),
        eval_steps=log_config.get("eval_steps", 500),
        load_best_model_at_end=log_config.get("load_best_model_at_end", True),
        metric_for_best_model=log_config.get("metric_for_best_model", "eval_loss"),
        # Other
        remove_unused_columns=False,  # Required for custom datasets
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
    )


def train(config: dict):
    """Main training function."""
    # Load model and processor
    model, processor = load_model_and_processor(config)

    # Load datasets
    train_dataset, val_dataset = load_datasets(config)

    if len(train_dataset) == 0:
        logger.error("No training samples found! Check your data directory.")
        return

    # Setup data collator
    collator = CircuitSenseCollator(
        processor=processor,
        max_length=config.get("training", {}).get("max_seq_length", 2048),
    )

    # Setup training arguments
    training_args = setup_training_arguments(config)

    # Setup wandb if enabled
    if training_args.report_to == "wandb":
        try:
            import wandb
            wandb.init(
                project=config.get("logging", {}).get("project_name", "circuitsense-finetune"),
                name=training_args.run_name,
                config=config,
            )
        except ImportError:
            logger.warning("wandb not installed, disabling wandb logging")
            training_args.report_to = "none"

    # Initialize trainer (using standard Trainer for PyTorch Dataset compatibility)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        data_collator=collator,
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    final_model_path = Path(training_args.output_dir) / "final_model"
    logger.info(f"Saving final model to: {final_model_path}")
    trainer.save_model(str(final_model_path))

    # Save processor
    processor.save_pretrained(str(final_model_path))

    logger.info("Training complete!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen2-VL on CircuitSense dataset"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/qwen2vl_lora.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Override data directory from config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--no_quantization",
        action="store_true",
        help="Disable quantization (use standard LoRA instead of QLoRA)",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Override wandb project name",
    )
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Disable wandb logging",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Resume training from checkpoint",
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["generated", "huggingface"],
        default=None,
        help="Dataset type: 'generated' for local data, 'huggingface' for HF data",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load and merge config
    logger.info(f"Loading config from: {args.config}")
    config = load_config(args.config)
    config = merge_config_with_args(config, args)

    # Set environment variables
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Check CUDA availability
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, training will be slow!")
    else:
        logger.info(f"CUDA available: {torch.cuda.device_count()} GPU(s)")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Run training
    train(config)


if __name__ == "__main__":
    main()
