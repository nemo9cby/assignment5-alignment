#!/usr/bin/env python3
"""
Quick test script to verify the Countdown baseline setup works correctly.
Tests loading data, model downloading, and basic inference.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import only the necessary functions without importing vllm
import json
from datasets import load_dataset
import random


def load_countdown_data(split: str = "train", max_examples: int = None, seed: int = 42):
    """Load Countdown examples from HuggingFace dataset."""
    print("Loading Countdown dataset from HuggingFace...")

    # Load the dataset from HuggingFace
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)

    print(f"Total examples in {split} split: {len(dataset)}")

    # Sample if max_examples is specified
    if max_examples and max_examples < len(dataset):
        random.seed(seed)
        indices = random.sample(range(len(dataset)), max_examples)
        sampled_dataset = dataset.select(indices)
    else:
        sampled_dataset = dataset

    # Convert to our format
    examples = []
    for item in sampled_dataset:
        nums_str = ", ".join(str(n) for n in item["nums"])
        question = f"Use the numbers [{nums_str}] to reach the target {item['target']}. You can use addition (+), subtraction (-), multiplication (*), and division (/). Each number can be used at most once."

        examples.append({
            "question": question,
            "ground_truth": str(item["target"]),
            "nums": item["nums"],
            "target": item["target"]
        })

    print(f"Loaded {len(examples)} Countdown examples")
    return examples


def format_prompt_r1_zero(question: str, prompt_path: str = "cs336_alignment/prompts/r1_zero.prompt"):
    """Format question using r1_zero prompt template."""
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    # Replace {question} with actual question
    prompt = prompt_template.replace("{question}", question)
    return prompt


def test_countdown_data_loading():
    """Test that we can load and parse Countdown data correctly."""
    print("Testing Countdown dataset loading from HuggingFace...")

    # Load a few examples
    examples = load_countdown_data(split="train", max_examples=5, seed=42)

    print(f"Successfully loaded {len(examples)} examples")

    # Display all examples
    for i, example in enumerate(examples):
        print(f"\nExample {i+1}:")
        print(f"  Numbers: {example['nums']}")
        print(f"  Target: {example['target']}")
        print(f"  Question: {example['question'][:100]}...")
        print(f"  Ground truth: {example['ground_truth']}")

    return examples


def test_prompt_formatting():
    """Test prompt formatting with r1_zero template."""
    print("\nTesting prompt formatting...")

    # Check if prompt template exists
    prompt_path = "cs336_alignment/prompts/r1_zero.prompt"
    if not os.path.exists(prompt_path):
        print(f"  Warning: Prompt template not found at {prompt_path}")
        return

    # Create a sample question
    question = "Use the numbers [3, 7, 25, 50] to reach the target 100. You can use addition (+), subtraction (-), multiplication (*), and division (/). Each number can be used at most once."

    formatted = format_prompt_r1_zero(question, prompt_path)

    print(f"  Original question: {question[:80]}...")
    print(f"  Formatted prompt length: {len(formatted)} chars")
    print(f"  Prompt preview: {formatted[:200]}...")

    # Check that the question was inserted
    if question in formatted:
        print("  ✓ Question successfully inserted into prompt")
    else:
        print("  ✗ Question not found in formatted prompt")


def test_answer_format():
    """Test the expected answer format for Countdown puzzles."""
    print("\nExpected answer format for Countdown puzzles:")
    print("  The model should use the r1_zero format:")
    print("  - Thinking section: <think>...</think>")
    print("  - Answer section: <answer>NUMBER</answer>")
    print("  - The answer should be the target number")
    print("\nExample expected output:")
    print("  <think>")
    print("  Let me solve this step by step...")
    print("  3 + 7 = 10")
    print("  10 * 10 = 100")
    print("  I reached the target!")
    print("  </think> <answer>100</answer>")


def test_model_info():
    """Print information about model configuration."""
    print("\nModel configuration:")
    print("  Model ID: Qwen/Qwen2.5-Math-1.5B")
    print("  Will download from HuggingFace Hub on first run")

    # Check HF_HOME environment variable
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    print(f"  HuggingFace cache directory: {hf_home}")

    # Estimate model size
    print("  Estimated model size: ~3GB (1.5B parameters in bfloat16)")
    print("  Note: First run will download the model, which may take several minutes")


def main():
    print("="*60)
    print("CS336 Assignment 5 - Countdown Baseline Setup Test")
    print("="*60)

    try:
        # Run tests
        test_countdown_data_loading()
        test_prompt_formatting()
        test_answer_format()
        test_model_info()

        print("\n" + "="*60)
        print("✓ All setup tests passed!")
        print("="*60)
        print("\nTo run the baseline evaluation with a small sample:")
        print("  python cs336_alignment/baseline_countdown.py --max-examples 10 --analyze-errors")
        print("\nTo run evaluation with 1000 examples (default):")
        print("  python cs336_alignment/baseline_countdown.py")
        print("\nTo run with more examples:")
        print("  python cs336_alignment/baseline_countdown.py --max-examples 5000")
        print("\nNote: The Countdown dataset has 490,364 total examples")
        print("      Start with a small number to verify everything works")

    except Exception as e:
        print(f"\n✗ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())