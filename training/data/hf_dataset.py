"""HuggingFace CircuitSense Dataset Loader.

This module provides a dataset class for loading the CircuitSense dataset
from HuggingFace (armanakbari4/CircuitSense) and converting it to the
format expected by our Qwen2-VL fine-tuning pipeline.

Usage:
    from training.data.hf_dataset import CircuitSenseHFDataset, download_circuitsense_hf

    # Download a subset
    download_circuitsense_hf(output_dir="datasets/circuitsense_hf", max_samples=100)

    # Load for training
    dataset = CircuitSenseHFDataset("datasets/circuitsense_hf", split="train")
"""

import json
import os
import random
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download, hf_hub_download, list_repo_files
from torch.utils.data import Dataset
from PIL import Image
from tqdm import tqdm


def download_circuitsense_hf(
    output_dir: str = "datasets/circuitsense_hf",
    max_samples: int | None = None,
    repo_id: str = "armanakbari4/CircuitSense",
    seed: int = 42,
) -> dict:
    """Download CircuitSense dataset from HuggingFace.

    The dataset has a hierarchical structure:
    - Domain (Perception/Analysis/Design)
      - Level folders
        - Question folders (q#_question.txt, q#_image.png, q#_ta.txt, q#-a.txt, q#_mc.txt)

    Args:
        output_dir: Directory to save the dataset
        max_samples: Maximum number of samples to download (None for all)
        repo_id: HuggingFace repository ID
        seed: Random seed for sampling

    Returns:
        Dictionary with dataset statistics
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    images_dir = output_path / "images"
    images_dir.mkdir(exist_ok=True)

    print(f"Listing files from {repo_id}...")

    # List all files in the repository
    all_files = list_repo_files(repo_id, repo_type="dataset")

    # Find all question folders by looking for image files
    image_files = [f for f in all_files if f.endswith('_image.png')]

    print(f"Found {len(image_files)} question samples")

    # Sample if max_samples specified
    if max_samples is not None and max_samples < len(image_files):
        random.seed(seed)
        image_files = random.sample(image_files, max_samples)
        print(f"Sampled {max_samples} samples")

    # Download and process each sample
    samples = []

    for img_file in tqdm(image_files, desc="Downloading samples"):
        try:
            # Parse the path to get question info
            # Format: Domain/Level/q#/q#_image.png
            parts = img_file.rsplit('/', 1)
            folder_path = parts[0] if len(parts) > 1 else ""
            img_name = parts[-1]

            # Extract question prefix (e.g., "q1" from "q1_image.png")
            q_prefix = img_name.replace('_image.png', '')

            # Construct paths for related files
            question_file = f"{folder_path}/{q_prefix}_question.txt"
            answer_file = f"{folder_path}/{q_prefix}_a.txt"  # Note: underscore not hyphen

            # Download the files
            local_img_path = hf_hub_download(
                repo_id=repo_id,
                filename=img_file,
                repo_type="dataset",
                local_dir=str(output_path / "raw"),
            )

            # Try to download question and answer
            try:
                local_question_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=question_file,
                    repo_type="dataset",
                    local_dir=str(output_path / "raw"),
                )
                with open(local_question_path, 'r') as f:
                    question = f.read().strip()
            except Exception:
                question = "Analyze this circuit diagram."

            try:
                local_answer_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=answer_file,
                    repo_type="dataset",
                    local_dir=str(output_path / "raw"),
                )
                with open(local_answer_path, 'r') as f:
                    answer = f.read().strip()
            except Exception:
                continue  # Skip samples without answers

            # Copy image to images directory with unique name
            sample_id = img_file.replace('/', '_').replace('_image.png', '')
            new_img_name = f"{sample_id}.png"
            new_img_path = images_dir / new_img_name

            # Copy the image
            import shutil
            shutil.copy(local_img_path, new_img_path)

            samples.append({
                "id": sample_id,
                "image": f"images/{new_img_name}",
                "question": question,
                "answer": answer,
                "source_path": img_file,
            })

        except Exception as e:
            print(f"Warning: Failed to process {img_file}: {e}")
            continue

    # Save labels.json in our expected format
    labels = {}
    qa_data = {"questions": []}

    for sample in samples:
        sample_id = sample["id"]
        labels[sample_id] = {
            "image": sample["image"],
            "question": sample["question"],
            "answer": sample["answer"],
        }

        qa_data["questions"].append({
            "circuit_id": sample_id,
            "question": sample["question"],
            "answer": sample["answer"],
            "answer_formatted": sample["answer"],
            "has_answer": True,
            "measurement_type": "general",
        })

    # Save labels.json
    with open(output_path / "labels.json", 'w') as f:
        json.dump(labels, f, indent=2)

    # Save circuit_qa_dataset.json
    qa_data["metadata"] = {
        "total_circuits": len(samples),
        "total_questions": len(samples),
        "source": repo_id,
    }
    with open(output_path / "circuit_qa_dataset.json", 'w') as f:
        json.dump(qa_data, f, indent=2)

    print(f"\nDataset saved to {output_path}")
    print(f"  - {len(samples)} samples")
    print(f"  - labels.json")
    print(f"  - circuit_qa_dataset.json")
    print(f"  - images/")

    return {
        "total_samples": len(samples),
        "output_dir": str(output_path),
    }


class CircuitSenseHFDataset(Dataset):
    """Dataset for CircuitSense from HuggingFace.

    This dataset loads Q&A pairs from the downloaded HuggingFace dataset
    and formats them for Qwen2-VL fine-tuning.

    Args:
        data_dir: Path to the downloaded dataset directory
        split: One of "train" or "val"
        train_ratio: Ratio of data to use for training
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.train_ratio = train_ratio

        random.seed(seed)

        # Load data
        self.samples = self._load_data()

        # Split into train/val
        random.shuffle(self.samples)
        split_idx = int(len(self.samples) * train_ratio)
        if split == "train":
            self.samples = self.samples[:split_idx]
        else:
            self.samples = self.samples[split_idx:]

        # TRL/Trainer compatibility
        self.column_names = None

    def _load_data(self) -> list[dict[str, Any]]:
        """Load data from labels.json."""
        labels_file = self.data_dir / "labels.json"
        if not labels_file.exists():
            raise FileNotFoundError(f"labels.json not found in {self.data_dir}")

        with open(labels_file, 'r') as f:
            labels = json.load(f)

        samples = []
        for sample_id, data in labels.items():
            image_path = self.data_dir / data["image"]
            if not image_path.exists():
                continue

            samples.append({
                "id": sample_id,
                "image_path": str(image_path),
                "question": data["question"],
                "answer": data["answer"],
            })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get a sample in Qwen2-VL conversation format."""
        sample = self.samples[idx]

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
            "task_type": "qa",
            "circuit_id": sample["id"],
            "metadata": {},
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download CircuitSense dataset from HuggingFace")
    parser.add_argument("--output_dir", type=str, default="datasets/circuitsense_hf",
                        help="Output directory for the dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to download")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for sampling")

    args = parser.parse_args()

    stats = download_circuitsense_hf(
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        seed=args.seed,
    )

    print(f"\nDownloaded {stats['total_samples']} samples to {stats['output_dir']}")

    # Test loading
    print("\nTesting dataset loading...")
    dataset = CircuitSenseHFDataset(args.output_dir, split="train", train_ratio=0.9)
    print(f"Train samples: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Sample keys: {list(sample.keys())}")
        print(f"Question: {sample['messages'][0]['content'][1]['text'][:100]}...")
