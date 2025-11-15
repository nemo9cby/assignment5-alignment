"""
Zero-shot Countdown dataset baseline evaluation for CS336 Assignment 5.
This module establishes baseline performance of Qwen 2.5 Math 1.5B on the Countdown dataset.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import argparse
from datetime import datetime
import random

import torch
from vllm import LLM, SamplingParams
from tqdm import tqdm
import pandas as pd
from datasets import load_dataset

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


def load_countdown_data(split: str = "train", max_examples: Optional[int] = None, seed: int = 42) -> List[Dict]:
    """Load Countdown examples from HuggingFace dataset.

    Args:
        split: Dataset split to use (currently only 'train' is available)
        max_examples: Maximum number of examples to load
        seed: Random seed for sampling examples

    Returns:
        List of examples with 'question' and 'ground_truth' keys
    """
    print("Loading Countdown dataset from HuggingFace...")

    # Load the dataset from HuggingFace
    dataset = load_dataset("Jiayi-Pan/Countdown-Tasks-3to4", split=split)

    print(f"Total examples in {split} split: {len(dataset)}")

    # Sample if max_examples is specified
    if max_examples and max_examples < len(dataset):
        # Set seed for reproducibility
        random.seed(seed)
        indices = random.sample(range(len(dataset)), max_examples)
        sampled_dataset = dataset.select(indices)
    else:
        sampled_dataset = dataset

    # Convert to our format
    examples = []
    for item in sampled_dataset:
        # Format the question as a Countdown puzzle
        nums_str = ", ".join(str(n) for n in item["nums"])
        question = f"Use the numbers [{nums_str}] to reach the target {item['target']}. You can use addition (+), subtraction (-), multiplication (*), and division (/). Each number can be used at most once."

        examples.append({
            "question": question,
            "ground_truth": str(item["target"]),  # Convert to string for consistency
            "nums": item["nums"],
            "target": item["target"]
        })

    print(f"Loaded {len(examples)} Countdown examples")
    return examples


def format_prompt_r1_zero(question: str, prompt_path: str = "cs336_alignment/prompts/r1_zero.prompt") -> str:
    """Format question using r1_zero prompt template."""
    with open(prompt_path, 'r') as f:
        prompt_template = f.read()

    # Replace {question} with actual question
    prompt = prompt_template.replace("{question}", question)
    return prompt


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn,
    prompts: List[str],
    ground_truths: List[str],
    eval_sampling_params: SamplingParams,
    output_dir: Optional[str] = None,
    save_interval: int = 100
) -> Dict:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.

    Args:
        vllm_model: vLLM model for generation
        reward_fn: Function to compute rewards from responses
        prompts: List of formatted prompts
        ground_truths: List of ground truth answers
        eval_sampling_params: Sampling parameters for generation
        output_dir: Directory to save results
        save_interval: Save intermediate results every N examples

    Returns:
        Dictionary with evaluation metrics and results
    """
    assert len(prompts) == len(ground_truths), "Prompts and ground truths must have same length"

    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    print(f"Generating responses for {len(prompts)} prompts...")

    # Generate responses using vLLM
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    # Process outputs and compute rewards
    results = []
    total_format_reward = 0.0
    total_answer_reward = 0.0
    total_reward = 0.0

    # Categories for analysis
    category_counts = {
        "correct_format_and_answer": 0,  # format=1, answer=1
        "correct_format_wrong_answer": 0,  # format=1, answer=0
        "wrong_format": 0  # format=0
    }

    print("Computing rewards...")
    for i, output in enumerate(tqdm(outputs)):
        prompt = output.prompt
        generated_text = output.outputs[0].text
        ground_truth = ground_truths[i]

        # Compute rewards
        reward_dict = reward_fn(generated_text, ground_truth)

        # Update statistics
        total_format_reward += reward_dict["format_reward"]
        total_answer_reward += reward_dict["answer_reward"]
        total_reward += reward_dict["reward"]

        # Categorize
        if reward_dict["format_reward"] == 1.0 and reward_dict["answer_reward"] == 1.0:
            category_counts["correct_format_and_answer"] += 1
        elif reward_dict["format_reward"] == 1.0 and reward_dict["answer_reward"] == 0.0:
            category_counts["correct_format_wrong_answer"] += 1
        elif reward_dict["format_reward"] == 0.0:
            category_counts["wrong_format"] += 1

        # Store result
        results.append({
            "index": i,
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "format_reward": reward_dict["format_reward"],
            "answer_reward": reward_dict["answer_reward"],
            "reward": reward_dict["reward"]
        })

        # Save intermediate results
        if output_dir and (i + 1) % save_interval == 0:
            intermediate_path = os.path.join(output_dir, f"results_intermediate_{i+1}.json")
            with open(intermediate_path, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved intermediate results to {intermediate_path}")

    n_examples = len(prompts)

    # Compute metrics
    metrics = {
        "n_examples": n_examples,
        "avg_format_reward": total_format_reward / n_examples,
        "avg_answer_reward": total_answer_reward / n_examples,
        "avg_total_reward": total_reward / n_examples,
        "accuracy": total_answer_reward / n_examples,  # Same as avg_answer_reward for binary rewards
        "correct_format_and_answer": category_counts["correct_format_and_answer"],
        "correct_format_wrong_answer": category_counts["correct_format_wrong_answer"],
        "wrong_format": category_counts["wrong_format"],
    }

    print("\n" + "="*50)
    print("EVALUATION METRICS:")
    print("="*50)
    print(f"Number of examples: {metrics['n_examples']}")
    print(f"Accuracy (answer reward): {metrics['accuracy']:.3f}")
    print(f"Average format reward: {metrics['avg_format_reward']:.3f}")
    print(f"Average total reward: {metrics['avg_total_reward']:.3f}")
    print(f"\nCategory breakdown:")
    print(f"  Correct format and answer: {metrics['correct_format_and_answer']} ({metrics['correct_format_and_answer']/n_examples*100:.1f}%)")
    print(f"  Correct format, wrong answer: {metrics['correct_format_wrong_answer']} ({metrics['correct_format_wrong_answer']/n_examples*100:.1f}%)")
    print(f"  Wrong format: {metrics['wrong_format']} ({metrics['wrong_format']/n_examples*100:.1f}%)")
    print("="*50 + "\n")

    # Save final results
    if output_dir:
        # Save detailed results
        results_path = os.path.join(output_dir, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)

        # Save metrics
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)

        # Save as CSV for easier analysis
        df = pd.DataFrame(results)
        csv_path = os.path.join(output_dir, "results.csv")
        df.to_csv(csv_path, index=False)

        print(f"Results saved to {output_dir}")

    return {
        "metrics": metrics,
        "results": results
    }


def analyze_errors(results: List[Dict], n_samples: int = 10):
    """Analyze and print examples of different error categories."""

    format_errors = [r for r in results if r["format_reward"] == 0.0]
    answer_errors = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 0.0]
    correct_examples = [r for r in results if r["format_reward"] == 1.0 and r["answer_reward"] == 1.0]

    print("\n" + "="*50)
    print(f"ERROR ANALYSIS (showing up to {n_samples} examples):")
    print("="*50)

    # Show some correct examples first
    if correct_examples:
        print(f"\nCORRECT EXAMPLES ({len(correct_examples)} total):")
        print("-"*50)
        for i, result in enumerate(correct_examples[:min(3, n_samples//3)]):
            print(f"\nExample {i+1}:")
            # Extract the question from the prompt
            if "Use the numbers" in result['prompt']:
                question_part = result['prompt'].split("Use the numbers")[1].split("\n")[0]
                print(f"Question: Use the numbers{question_part}")
            # Extract the answer part
            if "<answer>" in result['generated_text'] and "</answer>" in result['generated_text']:
                answer_text = result['generated_text'].split("<answer>")[-1].split("</answer>")[0]
                print(f"Model answer: {answer_text}")
            print(f"Ground truth: {result['ground_truth']}")

    if format_errors:
        print(f"\nFORMAT ERRORS ({len(format_errors)} total):")
        print("-"*50)
        for i, result in enumerate(format_errors[:n_samples]):
            print(f"\nExample {i+1}:")
            print(f"Generated text (last 200 chars): ...{result['generated_text'][-200:]}")
            print(f"Ground truth: {result['ground_truth']}")

    if answer_errors:
        print(f"\nANSWER ERRORS (format correct) ({len(answer_errors)} total):")
        print("-"*50)
        for i, result in enumerate(answer_errors[:n_samples]):
            print(f"\nExample {i+1}:")
            # Extract the question from the prompt
            if "Use the numbers" in result['prompt']:
                question_part = result['prompt'].split("Use the numbers")[1].split("\n")[0]
                print(f"Question: Use the numbers{question_part}")
            # Extract the answer part
            if "<answer>" in result['generated_text'] and "</answer>" in result['generated_text']:
                answer_text = result['generated_text'].split("<answer>")[-1].split("</answer>")[0]
                print(f"Model answer: {answer_text}")
            else:
                print(f"Model answer: Could not extract")
            print(f"Ground truth: {result['ground_truth']}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5 Math 1.5B zero-shot on Countdown dataset")

    # Model configuration - Use HuggingFace model ID to download automatically
    parser.add_argument("--model-path", type=str,
                       default="Qwen/Qwen2.5-Math-1.5B",
                       help="HuggingFace model ID or local path to model")

    # Use local HF cache directory if specified, otherwise use default
    parser.add_argument("--cache-dir", type=str,
                       default=None,
                       help="Directory to cache HuggingFace models (defaults to HF_HOME)")

    # Data configuration - Use Countdown dataset from HuggingFace
    parser.add_argument("--prompt-path", type=str,
                       default="cs336_alignment/prompts/r1_zero.prompt",
                       help="Path to prompt template")

    parser.add_argument("--output-dir", type=str,
                       default=f"results/baseline_countdown_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                       help="Directory to save results")

    parser.add_argument("--max-examples", type=int, default=1000,
                       help="Maximum number of examples to evaluate (default: 1000)")

    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for sampling examples")

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0,
                       help="Sampling temperature")
    parser.add_argument("--max-tokens", type=int, default=1024,
                       help="Maximum tokens to generate")

    # Hardware configuration
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85,
                       help="GPU memory utilization for vLLM")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device to use for vLLM")

    # Analysis options
    parser.add_argument("--analyze-errors", action="store_true",
                       help="Print error analysis")

    args = parser.parse_args()

    # Load data
    print("Loading Countdown dataset from HuggingFace...")
    countdown_data = load_countdown_data(
        split="train",
        max_examples=args.max_examples,
        seed=args.seed
    )

    # Extract questions and ground truths
    questions = [ex["question"] for ex in countdown_data]
    ground_truths = [ex["ground_truth"] for ex in countdown_data]

    # Show a few examples
    print(f"\nExample questions:")
    for i in range(min(3, len(questions))):
        print(f"  {i+1}. {questions[i]}")
        print(f"     Answer: {ground_truths[i]}")

    # Format prompts
    print("\nFormatting prompts...")
    prompts = [format_prompt_r1_zero(q, args.prompt_path) for q in questions]

    # Initialize vLLM with HuggingFace model
    print(f"\nInitializing vLLM with model {args.model_path}...")

    # Use HF_HOME or cache_dir for model downloads
    if args.cache_dir:
        cache_dir = args.cache_dir
        print(f"Using specified cache directory: {cache_dir}")
    else:
        # Use HF_HOME environment variable
        cache_dir = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
        print(f"Using HF_HOME cache directory: {cache_dir}")

    print("Note: The model will be downloaded from HuggingFace if not cached locally.")

    # Prepare vLLM kwargs
    vllm_kwargs = {
        "model": args.model_path,
        "dtype": torch.bfloat16,
        "enable_prefix_caching": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,  # Qwen models may require this
        "download_dir": cache_dir,  # Always set download_dir to use proper cache
    }

    # Add device configuration
    if args.device != "cuda":
        vllm_kwargs["device"] = args.device

    llm = LLM(**vllm_kwargs)

    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=1.0,
        max_tokens=args.max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,  # We want to include </answer> in output
        min_tokens=4,  # Prevent empty strings
    )

    # Run evaluation
    print(f"\nStarting evaluation with {len(prompts)} examples...")
    evaluation_results = evaluate_vllm(
        vllm_model=llm,
        reward_fn=r1_zero_reward_fn,
        prompts=prompts,
        ground_truths=ground_truths,
        eval_sampling_params=sampling_params,
        output_dir=args.output_dir
    )

    # Analyze errors if requested
    if args.analyze_errors:
        analyze_errors(evaluation_results["results"], n_samples=10)

    return evaluation_results


if __name__ == "__main__":
    main()