"""CircuitSense Dataset for Qwen2-VL fine-tuning.

This module provides dataset classes for loading CircuitSense data
and converting it to the conversation format expected by Qwen2-VL.
"""

import json
import random
from pathlib import Path
from typing import Any, Optional

from torch.utils.data import Dataset


# Question templates for symbolic equation tasks
EQUATION_QUESTION_TEMPLATES = [
    "Derive the transfer function for this circuit.",
    "What is the transfer function of this circuit?",
    "Calculate the transfer function H(s) for this circuit.",
    "Analyze this circuit and provide its transfer function.",
    "Find the transfer function relating output to input for this circuit.",
]

# Question templates for circuit Q&A tasks (voltage)
VOLTAGE_QUESTION_TEMPLATES = [
    "What is the voltage at node {node}?",
    "Calculate the voltage at node {node}.",
    "Find the voltage measured at node {node}.",
    "Determine the voltage at node {node} in this circuit.",
]

# Question templates for circuit Q&A tasks (current)
CURRENT_QUESTION_TEMPLATES = [
    "What is the current through {component}?",
    "Calculate the current flowing through {component}.",
    "Find the current in {component}.",
    "Determine the current through {component} in this circuit.",
]


def load_labels(labels_file: Path) -> dict[str, Any]:
    """Load circuit labels from JSON file."""
    with open(labels_file, "r") as f:
        return json.load(f)


def load_equations(equations_file: Path) -> dict[str, Any]:
    """Load symbolic equations from JSON file."""
    with open(equations_file, "r") as f:
        data = json.load(f)
    # Index by circuit_id for easy lookup
    if "results" in data:
        return {item["circuit_id"]: item for item in data["results"]}
    return data


def load_qa_data(qa_file: Path) -> list[dict[str, Any]]:
    """Load Q&A dataset from JSON file."""
    with open(qa_file, "r") as f:
        data = json.load(f)
    return data.get("questions", [])


class CircuitSenseDataset(Dataset):
    """Dataset for CircuitSense fine-tuning with Qwen2-VL.

    Supports two task types:
    1. Symbolic equation derivation (transfer functions)
    2. Circuit Q&A (voltage/current questions)

    Args:
        data_dir: Path to the dataset directory (e.g., datasets/training_data/)
        task_type: One of "equations", "qa", or "both"
        split: One of "train" or "val"
        train_ratio: Ratio of data to use for training (default: 0.9)
        seed: Random seed for reproducibility
        augment_questions: Whether to use question template augmentation
    """

    def __init__(
        self,
        data_dir: str | Path,
        task_type: str = "both",
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        augment_questions: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.task_type = task_type
        self.split = split
        self.train_ratio = train_ratio
        self.augment_questions = augment_questions

        # Set random seed for reproducible splits
        random.seed(seed)

        # Load data
        self.samples = self._load_and_prepare_data()

        # Split into train/val
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        # TRL SFTTrainer compatibility - set column_names to None
        # This tells TRL to infer columns from the first sample
        self.column_names = None

    def _load_and_prepare_data(self) -> list[dict[str, Any]]:
        """Load all data sources and prepare samples."""
        samples = []

        # Load labels (required)
        labels_file = self.data_dir / "labels.json"
        if not labels_file.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_file}")
        labels = load_labels(labels_file)

        # Load equations if needed
        equations = {}
        if self.task_type in ["equations", "both"]:
            equations_file = self.data_dir / "symbolic_equations.json"
            if equations_file.exists():
                equations = load_equations(equations_file)

        # Load Q&A data if needed
        qa_data = []
        if self.task_type in ["qa", "both"]:
            qa_file = self.data_dir / "circuit_qa_dataset.json"
            if qa_file.exists():
                qa_data = load_qa_data(qa_file)

        # Create equation samples
        if self.task_type in ["equations", "both"]:
            for circuit_id, eq_data in equations.items():
                if circuit_id not in labels:
                    continue

                label_data = labels[circuit_id]
                image_path = self.data_dir / label_data.get("image", f"images/{circuit_id}.jpg")

                if not image_path.exists():
                    continue

                # Get transfer functions
                transfer_functions = eq_data.get("transfer_functions", {})
                for tf_name, tf_value in transfer_functions.items():
                    if tf_value:
                        samples.append({
                            "type": "equation",
                            "circuit_id": circuit_id,
                            "image_path": str(image_path),
                            "question": self._get_equation_question(),
                            "answer": f"The transfer function is: ${tf_value}$",
                            "metadata": {
                                "tf_name": tf_name,
                                "raw_answer": tf_value,
                            }
                        })

        # Create Q&A samples
        if self.task_type in ["qa", "both"]:
            for qa_item in qa_data:
                circuit_id = qa_item.get("circuit_id")
                if circuit_id not in labels:
                    continue

                label_data = labels[circuit_id]
                image_path = self.data_dir / label_data.get("image", f"images/{circuit_id}.jpg")

                if not image_path.exists():
                    continue

                if not qa_item.get("has_answer", False):
                    continue

                samples.append({
                    "type": "qa",
                    "circuit_id": circuit_id,
                    "image_path": str(image_path),
                    "question": qa_item["question"],
                    "answer": qa_item["answer_formatted"],
                    "metadata": {
                        "measurement_type": qa_item.get("measurement_type"),
                        "unit": qa_item.get("unit"),
                        "raw_answer": qa_item.get("answer"),
                    }
                })

        return samples

    def _get_equation_question(self) -> str:
        """Get a question for equation derivation task."""
        if self.augment_questions:
            return random.choice(EQUATION_QUESTION_TEMPLATES)
        return EQUATION_QUESTION_TEMPLATES[0]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample in Qwen2-VL conversation format.

        Returns:
            Dictionary with:
            - messages: List of conversation turns
            - image_path: Path to the circuit image
            - metadata: Additional information for evaluation
        """
        sample = self.samples[idx]

        # Build conversation in Qwen2-VL format
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": sample["image_path"]},
                    {"type": "text", "text": sample["question"]},
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["answer"]},
                ],
            },
        ]

        return {
            "messages": messages,
            "image_path": sample["image_path"],
            "task_type": sample["type"],
            "circuit_id": sample["circuit_id"],
            "metadata": sample.get("metadata", {}),
        }


def load_circuitsense_data(
    data_dir: str | Path,
    task_type: str = "both",
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[CircuitSenseDataset, CircuitSenseDataset]:
    """Load CircuitSense data and return train/val datasets.

    Args:
        data_dir: Path to the dataset directory
        task_type: One of "equations", "qa", or "both"
        train_ratio: Ratio of data for training
        seed: Random seed

    Returns:
        Tuple of (train_dataset, val_dataset)
    """
    train_dataset = CircuitSenseDataset(
        data_dir=data_dir,
        task_type=task_type,
        split="train",
        train_ratio=train_ratio,
        seed=seed,
    )

    val_dataset = CircuitSenseDataset(
        data_dir=data_dir,
        task_type=task_type,
        split="val",
        train_ratio=train_ratio,
        seed=seed,
    )

    return train_dataset, val_dataset


if __name__ == "__main__":
    # Test the dataset
    import sys

    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "datasets/grid_v11_240831"

    print(f"Loading data from: {data_dir}")
    train_ds, val_ds = load_circuitsense_data(data_dir)

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples: {len(val_ds)}")

    if len(train_ds) > 0:
        print("\nSample item:")
        sample = train_ds[0]
        print(f"  Task type: {sample['task_type']}")
        print(f"  Circuit ID: {sample['circuit_id']}")
        print(f"  Image path: {sample['image_path']}")
        print(f"  Question: {sample['messages'][0]['content'][1]['text']}")
        print(f"  Answer: {sample['messages'][1]['content'][0]['text'][:100]}...")
