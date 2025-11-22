#!/usr/bin/env python3
"""
SFT training script for GSM8K dataset
Based on CS336 Assignment 5 requirements
"""

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch
import json
from tqdm import tqdm
import argparse
import os
from typing import List, Dict, Any

# Import your utilities
from cs336_alignment.utils import (
    tokenize_prompt_and_output,
    get_response_log_probs,
    sft_microbatch_train_step,
    log_generations,
)


def load_gsm8k_sft_data(filepath: str) -> List[Dict[str, Any]]:
    """Load GSM8K SFT data from jsonl file with conversation format"""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            item = json.loads(line)
            # Extract prompt and response from conversation format
            messages = item['messages']

            # Combine system and user messages as prompt
            prompt_parts = []
            for msg in messages:
                if msg['role'] == 'system':
                    prompt_parts.append(f"System: {msg['content']}")
                elif msg['role'] == 'user':
                    prompt_parts.append(f"User: {msg['content']}")
                elif msg['role'] == 'assistant':
                    response = f"Assistant: {msg['content']}"
                    break

            prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"

            # The response is the assistant's message content
            data.append({
                'prompt': prompt,
                'response': msg['content'],  # Just the assistant's response
                'ground_truth': item['metadata'].get('ground_truth', ''),
                'example_idx': item['metadata'].get('example_idx', -1)
            })

    return data


def init_vllm(model_id: str, device: str, seed: int, gpu_memory_utilization: float = 0.85):
    """Initialize vLLM with patches for our setup"""
    vllm_set_random_seed(seed)

    # Patches to make vLLM work on a single GPU in our multi-GPU setup
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None
    )

    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: torch.nn.Module, llm: LLM):
    """Copy policy weights into vLLM instance for evaluation"""
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


def evaluate_on_validation(
    vllm_model: LLM,
    val_data: List[Dict],
    sampling_params: SamplingParams,
    num_samples: int = 100
) -> Dict[str, float]:
    """Evaluate model on validation set"""
    # Sample validation examples
    import random
    eval_samples = random.sample(val_data, min(num_samples, len(val_data)))

    # Extract prompts
    prompts = [sample['prompt'] for sample in eval_samples]
    ground_truths = [sample['ground_truth'] for sample in eval_samples]

    # Generate responses
    outputs = vllm_model.generate(prompts, sampling_params)

    # Calculate metrics
    correct = 0
    format_correct = 0

    for i, output in enumerate(outputs):
        generated_text = output.outputs[0].text

        # Check format (has answer tags)
        if '<answer>' in generated_text and '</answer>' in generated_text:
            format_correct += 1

            # Extract answer
            try:
                answer_start = generated_text.index('<answer>') + len('<answer>')
                answer_end = generated_text.index('</answer>')
                predicted_answer = generated_text[answer_start:answer_end].strip()

                # Simple string match for correctness
                if predicted_answer == ground_truths[i]:
                    correct += 1
            except:
                pass

    accuracy = correct / len(eval_samples) if eval_samples else 0
    format_accuracy = format_correct / len(eval_samples) if eval_samples else 0

    return {
        'accuracy': accuracy,
        'format_accuracy': format_accuracy,
        'num_evaluated': len(eval_samples)
    }


def train_sft(
    model_path: str = "gpt2",  # Default to small model for testing
    train_data_path: str = "/shared_workspace_mfs/boyuan/assignment5-alignment/data/gsm8k_sft_final/sft_conversations.jsonl",
    val_data_path: str = None,  # If None, use part of train for validation
    output_dir: str = "/shared_workspace_mfs/boyuan/assignment5-alignment/models/sft_gsm8k",
    n_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation_steps: int = 8,
    learning_rate: float = 1e-5,
    max_grad_norm: float = 1.0,
    eval_every: int = 50,
    save_every: int = 100,
    log_every: int = 10,
    device_policy: str = "cuda:0",
    device_vllm: str = "cuda:1",
    use_wandb: bool = True,
    seed: int = 42,
):
    """Main SFT training loop"""

    # Set random seed
    torch.manual_seed(seed)

    # Initialize wandb if requested
    if use_wandb:
        wandb.init(project="cs336-sft-gsm8k", config={
            "model": model_path,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": batch_size * gradient_accumulation_steps,
            "learning_rate": learning_rate,
            "n_epochs": n_epochs,
            "max_grad_norm": max_grad_norm,
        })

        # Setup metrics tracking
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # Load model and tokenizer
    print(f"Loading model and tokenizer from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2" if "gpt2" not in model_path else "eager",
    ).to(device_policy)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Important for batch generation

    # Load data
    print(f"Loading training data from {train_data_path}...")
    train_data = load_gsm8k_sft_data(train_data_path)
    print(f"Loaded {len(train_data)} training examples")

    # Split validation if no separate validation provided
    if val_data_path:
        val_data = load_gsm8k_sft_data(val_data_path)
    else:
        # Use 10% for validation
        val_size = len(train_data) // 10
        val_data = train_data[:val_size]
        train_data = train_data[val_size:]
        print(f"Split data: {len(train_data)} train, {len(val_data)} validation")

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Initialize vLLM for evaluation (only if we have 2 GPUs)
    vllm_model = None
    if torch.cuda.device_count() > 1 and device_vllm != device_policy:
        print(f"Initializing vLLM on {device_vllm} for evaluation...")
        vllm_model = init_vllm(
            model_id=model_path,
            device=device_vllm,
            seed=seed,
            gpu_memory_utilization=0.85
        )
    else:
        print("Using single GPU - vLLM evaluation disabled")

    # Evaluation sampling parameters
    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True
    )

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Training loop
    train_step = 0
    eval_step = 0

    print("\nStarting training...")
    for epoch in range(n_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"{'='*50}")

        # Shuffle training data
        import random
        random.shuffle(train_data)

        # Process in batches
        for batch_start in tqdm(range(0, len(train_data), batch_size), desc="Training"):
            batch_end = min(batch_start + batch_size, len(train_data))
            batch = train_data[batch_start:batch_end]

            # Skip incomplete batches if they're too small
            if len(batch) < 1:
                continue

            # Extract prompts and responses
            prompts = [item["prompt"] for item in batch]
            responses = [item["response"] for item in batch]

            # Tokenize
            tokenized = tokenize_prompt_and_output(prompts, responses, tokenizer)
            input_ids = tokenized["input_ids"].to(device_policy)
            labels = tokenized["labels"].to(device_policy)
            response_mask = tokenized["response_mask"].to(device_policy)

            # Forward pass to get log probabilities (with gradients)
            model.train()
            result = get_response_log_probs(
                model=model,
                input_ids=input_ids,
                labels=labels,
                return_token_entropy=True
            )

            # Training step with backward pass
            # Normalize by the average number of response tokens per example
            num_response_tokens = response_mask.sum(dim=1).mean().item()
            loss, metadata = sft_microbatch_train_step(
                policy_log_probs=result["log_probs"],
                response_mask=response_mask,
                gradient_accumulation_steps=gradient_accumulation_steps,
                normalize_constant=num_response_tokens
            )

            # Gradient clipping and optimizer step
            if (train_step + 1) % gradient_accumulation_steps == 0:
                # Clip gradients
                if max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                # Optimizer step
                optimizer.step()
                optimizer.zero_grad()

            # Logging
            if train_step % log_every == 0:
                avg_entropy = result["token_entropy"].mean().item() if result["token_entropy"] is not None else 0
                response_lengths = response_mask.sum(dim=1).float().mean().item()

                # Log basic metrics
                if use_wandb:
                    wandb.log({
                        "train/loss": metadata["loss"],
                        "train/entropy": avg_entropy,
                        "train/response_length": response_lengths,
                        "train/normalized_tokens": num_response_tokens,
                        "train_step": train_step
                    })
                else:
                    print(f"Step {train_step}: Loss={metadata['loss']:.4f}, Entropy={avg_entropy:.4f}, Response Len={response_lengths:.1f}")

                # Log sample generations every 10x log_every steps
                if train_step % (log_every * 10) == 0 and train_step > 0:
                    # Get a few samples for generation logging
                    sample_batch = batch[:min(2, len(batch))]  # Take up to 2 samples
                    sample_prompts = [item["prompt"] for item in sample_batch]
                    sample_ground_truths = [item.get("ground_truth", "") for item in sample_batch]

                    # Generate and log
                    gen_results = log_generations(
                        model=model,
                        tokenizer=tokenizer,
                        prompts=sample_prompts,
                        ground_truths=sample_ground_truths if sample_ground_truths[0] else None,
                        max_new_tokens=256,
                        temperature=1.0,
                        do_sample=False
                    )

                    # Log to wandb or print
                    if use_wandb:
                        # Create a table for wandb
                        table = wandb.Table(columns=["Prompt", "Generation", "Ground Truth"])
                        for i in range(len(sample_prompts)):
                            table.add_data(
                                sample_prompts[i][:200] + "...",
                                gen_results["generations"][i][:300] + "...",
                                sample_ground_truths[i] if sample_ground_truths else "N/A"
                            )
                        wandb.log({"train/generations": table, "train_step": train_step})
                    else:
                        print(f"\n--- Sample Generation at Step {train_step} ---")
                        print(f"Prompt: {sample_prompts[0][:200]}...")
                        print(f"Generation: {gen_results['generations'][0][:300]}...")
                        if sample_ground_truths and sample_ground_truths[0]:
                            print(f"Ground Truth: {sample_ground_truths[0]}")
                        print("---\n")

            # Periodic evaluation
            if vllm_model and train_step % eval_every == 0 and train_step > 0:
                print(f"\nEvaluating at step {train_step}...")
                model.eval()

                # Copy weights to vLLM
                load_policy_into_vllm_instance(model, vllm_model)

                # Run evaluation
                eval_results = evaluate_on_validation(
                    vllm_model,
                    val_data,
                    sampling_params,
                    num_samples=50  # Evaluate on 50 examples
                )

                print(f"Validation Accuracy: {eval_results['accuracy']:.2%}")
                print(f"Format Accuracy: {eval_results['format_accuracy']:.2%}")

                if use_wandb:
                    wandb.log({
                        "eval/accuracy": eval_results['accuracy'],
                        "eval/format_accuracy": eval_results['format_accuracy'],
                        "eval_step": eval_step
                    })

                eval_step += 1

            # Periodic saving
            if train_step % save_every == 0 and train_step > 0:
                checkpoint_dir = os.path.join(output_dir, f"checkpoint-{train_step}")
                print(f"\nSaving checkpoint to {checkpoint_dir}")
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)

            train_step += 1

    # Save final model
    print(f"\nSaving final model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Final evaluation
    if vllm_model:
        print("\nRunning final evaluation...")
        model.eval()
        load_policy_into_vllm_instance(model, vllm_model)
        final_results = evaluate_on_validation(
            vllm_model,
            val_data,
            sampling_params,
            num_samples=min(200, len(val_data))
        )
        print(f"Final Validation Accuracy: {final_results['accuracy']:.2%}")
        print(f"Final Format Accuracy: {final_results['format_accuracy']:.2%}")

    if use_wandb:
        wandb.finish()

    print("\nTraining complete!")


def main():
    parser = argparse.ArgumentParser(description="Train SFT on GSM8K dataset")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-Math-1.5B",
                       help="Model name/path (HuggingFace hub ID or local path)")
    parser.add_argument("--train-data", type=str,
                       default="/shared_workspace_mfs/boyuan/assignment5-alignment/data/gsm8k_full/sft_conversations.jsonl",
                       help="Path to training data")
    parser.add_argument("--output-dir", type=str,
                       default="/shared_workspace_mfs/boyuan/assignment5-alignment/models/sft_gsm8k",
                       help="Output directory for model")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=8, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--eval-every", type=int, default=50, help="Evaluate every N steps")
    parser.add_argument("--save-every", type=int, default=100, help="Save checkpoint every N steps")
    parser.add_argument("--log-every", type=int, default=10, help="Log metrics every N steps")
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--device-policy", type=str, default="cuda:0",
                       help="Device for training policy model")
    parser.add_argument("--device-vllm", type=str, default="cuda:1",
                       help="Device for vLLM evaluation (if available)")

    args = parser.parse_args()

    # Check available devices
    num_devices = torch.cuda.device_count()
    print(f"Available CUDA devices: {num_devices}")

    # Set devices based on availability
    device_policy = args.device_policy
    device_vllm = args.device_vllm if num_devices > 1 else args.device_policy

    # If only one device available, use it for both
    if num_devices == 1:
        device_policy = "cuda:0"
        device_vllm = "cuda:0"
        print("Only 1 GPU available - using same device for training and evaluation")

    print(f"Using devices: Training on {device_policy}, vLLM on {device_vllm}")

    train_sft(
        model_path=args.model,
        train_data_path=args.train_data,
        output_dir=args.output_dir,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_every=args.eval_every,
        save_every=args.save_every,
        log_every=args.log_every,
        device_policy=device_policy,
        device_vllm=device_vllm,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()