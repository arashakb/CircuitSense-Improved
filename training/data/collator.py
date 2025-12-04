"""Data collator for Qwen2-VL fine-tuning on CircuitSense.

This module provides a custom data collator that handles:
- Loading and processing images through Qwen2-VL processor
- Converting conversations to model input format
- Batching with proper padding
"""

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image


@dataclass
class CircuitSenseCollator:
    """Data collator for CircuitSense fine-tuning with Qwen2-VL.

    Handles image loading, text tokenization, and batch creation
    for the Qwen2-VL model.

    Args:
        processor: Qwen2-VL processor (AutoProcessor)
        max_length: Maximum sequence length for tokenization
        padding: Padding strategy ("max_length" or "longest")
        return_tensors: Return type ("pt" for PyTorch)
    """

    processor: Any
    max_length: int = 2048
    padding: str = "longest"
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collate a batch of samples.

        Args:
            features: List of samples from CircuitSenseDataset

        Returns:
            Batched inputs ready for the model
        """
        # Prepare texts and images for the processor
        texts = []
        images_list = []

        for feature in features:
            messages = feature["messages"]

            # Load image
            image_path = feature["image_path"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                # Create a blank image as fallback
                image = Image.new("RGB", (384, 384), color="white")

            images_list.append([image])

            # Format conversation for processor
            # Qwen2-VL expects a specific chat template format
            text = self._format_conversation(messages)
            texts.append(text)

        # Process batch through Qwen2-VL processor
        # Note: truncation=False to avoid image token mismatch errors
        # Long sequences will be handled by the model's max_position_embeddings
        batch = self.processor(
            text=texts,
            images=images_list,
            padding=self.padding,
            max_length=self.max_length,
            truncation=False,  # Disable truncation to avoid image token mismatch
            return_tensors=self.return_tensors,
        )

        # Create labels for language modeling
        # Labels are the same as input_ids but with padding tokens set to -100
        if "input_ids" in batch:
            labels = batch["input_ids"].clone()
            if self.processor.tokenizer.pad_token_id is not None:
                labels[labels == self.processor.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch

    def _format_conversation(self, messages: list[dict[str, Any]]) -> str:
        """Format conversation messages for Qwen2-VL.

        Args:
            messages: List of message dictionaries with role and content

        Returns:
            Formatted conversation string
        """
        # Use the processor's chat template if available
        if hasattr(self.processor, "apply_chat_template"):
            # Convert our format to the expected format
            formatted_messages = []
            for msg in messages:
                content_parts = []
                for part in msg["content"]:
                    if part["type"] == "image":
                        content_parts.append({"type": "image"})
                    elif part["type"] == "text":
                        content_parts.append({"type": "text", "text": part["text"]})
                formatted_messages.append({
                    "role": msg["role"],
                    "content": content_parts,
                })

            return self.processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

        # Fallback: manual formatting
        conversation = ""
        for msg in messages:
            role = msg["role"]
            text_content = ""
            has_image = False

            for part in msg["content"]:
                if part["type"] == "text":
                    text_content = part["text"]
                elif part["type"] == "image":
                    has_image = True

            if role == "user":
                if has_image:
                    conversation += f"<|im_start|>user\n<image>\n{text_content}<|im_end|>\n"
                else:
                    conversation += f"<|im_start|>user\n{text_content}<|im_end|>\n"
            elif role == "assistant":
                conversation += f"<|im_start|>assistant\n{text_content}<|im_end|>\n"

        return conversation


@dataclass
class CircuitSenseEvalCollator:
    """Data collator for evaluation (without labels).

    Similar to CircuitSenseCollator but only prepares the user message
    for generation, without including the assistant response.

    Args:
        processor: Qwen2-VL processor
        max_length: Maximum sequence length
    """

    processor: Any
    max_length: int = 2048

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch for evaluation/inference.

        Args:
            features: List of samples

        Returns:
            Batched inputs ready for generation
        """
        texts = []
        images_list = []
        metadata_list = []

        for feature in features:
            messages = feature["messages"]

            # Load image
            image_path = feature["image_path"]
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception:
                image = Image.new("RGB", (384, 384), color="white")

            images_list.append([image])

            # Only include user message for generation
            user_message = messages[0]
            text = self._format_user_message(user_message)
            texts.append(text)

            # Keep metadata for evaluation
            metadata_list.append({
                "circuit_id": feature.get("circuit_id"),
                "task_type": feature.get("task_type"),
                "ground_truth": messages[1]["content"][0]["text"],
                "metadata": feature.get("metadata", {}),
            })

        # Process batch
        # Note: truncation=False to avoid image token mismatch errors
        batch = self.processor(
            text=texts,
            images=images_list,
            padding="longest",
            max_length=self.max_length,
            truncation=False,  # Disable truncation to avoid image token mismatch
            return_tensors="pt",
        )

        batch["metadata"] = metadata_list
        return batch

    def _format_user_message(self, message: dict[str, Any]) -> str:
        """Format user message for generation."""
        if hasattr(self.processor, "apply_chat_template"):
            content_parts = []
            for part in message["content"]:
                if part["type"] == "image":
                    content_parts.append({"type": "image"})
                elif part["type"] == "text":
                    content_parts.append({"type": "text", "text": part["text"]})

            formatted_messages = [{"role": "user", "content": content_parts}]

            return self.processor.apply_chat_template(
                formatted_messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        # Fallback
        text_content = ""
        for part in message["content"]:
            if part["type"] == "text":
                text_content = part["text"]

        return f"<|im_start|>user\n<image>\n{text_content}<|im_end|>\n<|im_start|>assistant\n"
