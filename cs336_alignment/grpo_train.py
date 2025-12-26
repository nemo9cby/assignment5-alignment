"""
GRPO (Group Relative Policy Optimization) Training Loop

Based on Algorithm 3 from the CS336 Assignment 5 PDF:
1. Sample batch of questions from dataset
2. Set old policy = current policy
3. Sample G outputs per question from old policy
4. Compute rewards and group-normalized advantages
5. For each train step: update policy using GRPO objective
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel
from vllm import LLM, SamplingParams
from vllm.model_executor import set_random_seed as vllm_set_random_seed

# Import your implemented utilities
from cs336_alignment.utils import (
    compute_group_normalized_rewards,
    get_response_log_probs,
    tokenize_prompt_and_output,
    grpo_microbatch_train_step,
    masked_mean,
)

# Import the reward function
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model paths
    model_path: str = "/data/a5-alignment/models/Qwen2.5-Math-1.5B"
    output_dir: str = "/data/yourusername/grpo_output"

    # Dataset paths
    train_data_path: str = "/data/a5-alignment/MATH/train.jsonl"
    val_data_path: str = "/data/a5-alignment/MATH/validation.jsonl"
    prompt_template_path: str = "cs336_alignment/prompts/r1_zero.prompt"

    # Training hyperparameters
    n_grpo_steps: int = 200
    learning_rate: float = 1e-5
    weight_decay: float = 0.0
    betas: tuple[float, float] = (0.9, 0.95)
    max_grad_norm: float = 1.0  # Gradient clipping

    # Rollout configuration
    rollout_batch_size: int = 256  # Total responses per GRPO step
    group_size: int = 8            # Responses per question (G in the paper)

    # Sampling parameters for rollouts
    sampling_temperature: float = 1.0
    sampling_top_p: float = 1.0
    sampling_min_tokens: int = 4   # Avoid empty responses
    sampling_max_tokens: int = 1024
    stop_strings: list[str] = field(default_factory=lambda: ["</answer>"])

    # Training batch configuration
    epochs_per_rollout_batch: int = 1    # 1 = on-policy, >1 = off-policy
    train_batch_size: int = 256          # Examples per optimizer step
    gradient_accumulation_steps: int = 128  # microbatch_size = train_batch_size / this

    # Loss configuration
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"] = "reinforce_with_baseline"
    use_std_normalization: bool = True   # Normalize by group std (original GRPO vs Dr. GRPO)
    advantage_eps: float = 1e-6          # Epsilon for advantage normalization
    cliprange: float = 0.2               # For grpo_clip loss type

    # vLLM configuration
    gpu_memory_utilization: float = 0.85
    policy_device: str = "cuda:0"
    vllm_device: str = "cuda:1"

    # Logging and evaluation
    eval_every_n_steps: int = 10
    log_every_n_steps: int = 1
    save_every_n_steps: int = 50
    n_eval_examples: int = 1024  # Number of validation examples to evaluate

    # Reproducibility
    seed: int = 42

    # Wandb
    use_wandb: bool = True
    wandb_project: str = "cs336-grpo"
    wandb_run_name: str | None = None

    @property
    def n_prompts_per_rollout_batch(self) -> int:
        """Number of unique prompts per rollout batch."""
        return self.rollout_batch_size // self.group_size

    @property
    def micro_train_batch_size(self) -> int:
        """Size of each microbatch."""
        return self.train_batch_size // self.gradient_accumulation_steps

    def validate(self):
        """Validate configuration consistency."""
        assert self.rollout_batch_size % self.group_size == 0, \
            "rollout_batch_size must be divisible by group_size"
        assert self.train_batch_size % self.gradient_accumulation_steps == 0, \
            "train_batch_size must be divisible by gradient_accumulation_steps"
        assert self.train_batch_size >= self.group_size, \
            "train_batch_size must be >= group_size"
        if self.loss_type == "grpo_clip":
            assert self.epochs_per_rollout_batch > 1 or self.train_batch_size < self.rollout_batch_size, \
                "grpo_clip requires off-policy setting (multiple epochs or smaller train batch)"


# =============================================================================
# vLLM Initialization Helpers
# =============================================================================

def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85
) -> LLM:
    """
    Initialize vLLM for inference on a separate GPU.

    Based on TRL's monkeypatch approach to place vLLM on a specific device.
    """
    vllm_set_random_seed(seed)

    # Monkeypatch from TRL to place vLLM on desired device
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


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copy policy weights into the vLLM instance for generation.

    This allows us to generate from the current policy without reloading.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


# =============================================================================
# Data Loading
# =============================================================================

def load_dataset(path: str) -> list[dict]:
    """Load JSONL dataset."""
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def load_prompt_template(path: str) -> str:
    """Load prompt template from file."""
    with open(path, "r") as f:
        return f.read()


def format_prompt(question: str, template: str) -> str:
    """Format a question using the prompt template."""
    return template.format(question=question)


def sample_prompts(
    examples: list[dict],
    n_prompts: int,
    prompt_template: str,
    rng: torch.Generator | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Sample a batch of prompts from the dataset.

    This is the first step in GRPO - sampling questions from the training set
    and formatting them as prompts for the language model.

    Args:
        examples: List of dataset examples, each with 'problem' and 'answer' keys
        n_prompts: Number of prompts to sample
        prompt_template: Template string with {question} placeholder
        rng: Optional random generator for reproducibility

    Returns:
        questions: List of raw question strings (n_prompts)
        ground_truths: List of ground truth answers (n_prompts)
        prompts: List of formatted prompt strings (n_prompts)
    """
    # Sample random indices
    n_examples = len(examples)

    if rng is not None:
        # Use the provided generator for reproducibility
        indices = torch.randint(0, n_examples, (n_prompts,), generator=rng).tolist()
    else:
        indices = torch.randint(0, n_examples, (n_prompts,)).tolist()

    # Extract questions and ground truths
    questions = []
    ground_truths = []
    prompts = []

    for idx in indices:
        example = examples[idx]
        question = example['problem']  # MATH dataset uses 'problem' key
        answer = example['answer']     # MATH dataset uses 'answer' key

        questions.append(question)
        ground_truths.append(answer)
        prompts.append(format_prompt(question, prompt_template))

    return questions, ground_truths, prompts


# =============================================================================
# Rollout Generation
# =============================================================================

def generate_rollouts(
    llm: LLM,
    prompts: list[str],
    ground_truths: list[str],
    group_size: int,
    sampling_params: SamplingParams,
) -> tuple[list[str], list[str], list[str]]:
    """
    Generate rollouts for a batch of prompts.

    Args:
        llm: vLLM instance
        prompts: List of formatted prompts (n_prompts)
        ground_truths: List of ground truth answers (n_prompts)
        group_size: Number of responses per prompt
        sampling_params: vLLM sampling parameters

    Returns:
        repeated_prompts: Prompts repeated group_size times (n_prompts * group_size)
        rollout_responses: Generated responses (n_prompts * group_size)
        repeated_ground_truths: Ground truths repeated (n_prompts * group_size)
    """
    # TODO: Implement rollout generation
    # 1. Generate group_size responses for each prompt using llm.generate()
    # 2. Flatten the outputs
    # 3. Repeat prompts and ground_truths to match
    pass


# =============================================================================
# Training Step Components
# =============================================================================

def compute_old_log_probs(
    policy: PreTrainedModel,
    prompts: list[str],
    responses: list[str],
    tokenizer,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute log probabilities from the OLD policy (before gradient updates).

    These are used for importance sampling in off-policy training.
    Should be called with torch.inference_mode() - no gradients needed.

    Returns:
        old_log_probs: (batch_size, seq_length)
        input_ids: (batch_size, seq_length)
        response_mask: (batch_size, seq_length)
    """
    # TODO: Implement
    # 1. Tokenize prompts and responses
    # 2. Get log probs using get_response_log_probs with no gradients
    pass


def train_on_rollout_batch(
    policy: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    tokenizer,
    prompts: list[str],
    responses: list[str],
    advantages: torch.Tensor,
    raw_rewards: torch.Tensor,
    old_log_probs: torch.Tensor | None,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    config: GRPOConfig,
) -> dict[str, float]:
    """
    Perform training updates on a rollout batch.

    For on-policy (epochs=1): single pass through the batch
    For off-policy (epochs>1): multiple passes, using old_log_probs for importance weighting

    Returns:
        Dictionary of training metrics
    """
    # TODO: Implement the inner training loop
    # For each epoch:
    #   Shuffle the rollout batch
    #   For each microbatch:
    #     1. Get current policy log probs
    #     2. Call grpo_microbatch_train_step
    #   After gradient_accumulation_steps microbatches:
    #     1. Clip gradients
    #     2. optimizer.step()
    #     3. optimizer.zero_grad()
    pass


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    llm: LLM,
    val_examples: list[dict],
    prompt_template: str,
    reward_fn: Callable,
    sampling_params: SamplingParams,
    n_examples: int | None = None,
) -> dict[str, float]:
    """
    Evaluate the policy on validation set.

    Returns:
        Dictionary with evaluation metrics (accuracy, format_reward, etc.)
    """
    # TODO: Implement evaluation
    # 1. Sample n_examples from validation set
    # 2. Format prompts
    # 3. Generate responses
    # 4. Compute rewards
    # 5. Return aggregated metrics
    pass


# =============================================================================
# Main Training Loop
# =============================================================================

def grpo_train_loop(config: GRPOConfig | None = None):
    """
    Main GRPO training loop implementing Algorithm 3 from the paper.

    Overview:
    1. Initialize policy model and vLLM instance
    2. For each GRPO step:
       a. Sample batch of questions
       b. Generate G rollouts per question
       c. Compute rewards and advantages
       d. Train policy on rollouts (possibly multiple epochs)
    3. Periodically evaluate and save checkpoints
    """
    if config is None:
        config = GRPOConfig()
    config.validate()

    # Set random seeds
    torch.manual_seed(config.seed)

    # ==========================================================================
    # Setup: Load model, tokenizer, optimizer
    # ==========================================================================

    print(f"Loading policy model from {config.model_path}")
    # TODO: Load policy model with bfloat16 and flash attention
    policy = None

    print(f"Loading tokenizer from {config.model_path}")
    # TODO: Load tokenizer
    tokenizer = None

    print(f"Initializing optimizer")
    # TODO: Initialize AdamW optimizer
    optimizer = None

    # ==========================================================================
    # Setup: Initialize vLLM for generation
    # ==========================================================================

    print(f"Initializing vLLM on {config.vllm_device}")
    # TODO: Initialize vLLM
    llm = None

    # ==========================================================================
    # Setup: Load data and prompt template
    # ==========================================================================

    print(f"Loading training data from {config.train_data_path}")
    train_examples = load_dataset(config.train_data_path)
    print(f"Loaded {len(train_examples)} training examples")

    print(f"Loading validation data from {config.val_data_path}")
    val_examples = load_dataset(config.val_data_path)
    print(f"Loaded {len(val_examples)} validation examples")

    print(f"Loading prompt template from {config.prompt_template_path}")
    prompt_template = load_prompt_template(config.prompt_template_path)

    # ==========================================================================
    # Setup: Sampling parameters
    # ==========================================================================

    rollout_sampling_params = SamplingParams(
        temperature=config.sampling_temperature,
        top_p=config.sampling_top_p,
        max_tokens=config.sampling_max_tokens,
        min_tokens=config.sampling_min_tokens,
        stop=config.stop_strings,
        include_stop_str_in_output=True,
        n=config.group_size,  # Generate G responses per prompt
        seed=config.seed,
    )

    eval_sampling_params = SamplingParams(
        temperature=config.sampling_temperature,
        top_p=config.sampling_top_p,
        max_tokens=config.sampling_max_tokens,
        min_tokens=config.sampling_min_tokens,
        stop=config.stop_strings,
        include_stop_str_in_output=True,
        seed=config.seed,
    )

    # ==========================================================================
    # Setup: Wandb logging
    # ==========================================================================

    if config.use_wandb:
        import wandb
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=config.__dict__,
        )
        # Define custom x-axes for train vs eval metrics
        wandb.define_metric("train_step")
        wandb.define_metric("eval_step")
        wandb.define_metric("train/*", step_metric="train_step")
        wandb.define_metric("eval/*", step_metric="eval_step")

    # ==========================================================================
    # Main Training Loop
    # ==========================================================================

    print(f"Starting GRPO training for {config.n_grpo_steps} steps")

    train_step = 0
    eval_step = 0

    for grpo_step in range(config.n_grpo_steps):

        # ======================================================================
        # Step 1: Sample batch of questions
        # ======================================================================

        # Sample n_prompts_per_rollout_batch questions from train set
        batch_questions, batch_ground_truths, batch_prompts = sample_prompts(
            examples=train_examples,
            n_prompts=config.n_prompts_per_rollout_batch,
            prompt_template=prompt_template,
            rng=None,  # Could pass a generator for reproducibility
        )

        print(f"Step {grpo_step}: Sampled {len(batch_prompts)} prompts")

        # ======================================================================
        # Step 2: Load current policy weights into vLLM
        # ======================================================================

        # TODO: Copy policy weights to vLLM for generation
        # load_policy_into_vllm_instance(policy, llm)

        # ======================================================================
        # Step 3: Generate G rollouts per question
        # ======================================================================

        # TODO: Generate rollouts using vLLM
        # repeated_prompts, rollout_responses, repeated_ground_truths = generate_rollouts(...)

        # ======================================================================
        # Step 4: Compute rewards and group-normalized advantages
        # ======================================================================

        # TODO: Compute rewards and advantages
        # advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(...)

        # ======================================================================
        # Step 5: Compute old log probs (for off-policy training)
        # ======================================================================

        # TODO: If off-policy, compute old_log_probs with no gradients
        # old_log_probs, input_ids, response_mask = compute_old_log_probs(...)

        # ======================================================================
        # Step 6: Train on rollout batch
        # ======================================================================

        # TODO: Perform training updates
        # train_metrics = train_on_rollout_batch(...)

        train_step += 1

        # ======================================================================
        # Step 7: Logging
        # ======================================================================

        if grpo_step % config.log_every_n_steps == 0:
            # TODO: Log training metrics
            pass

        # ======================================================================
        # Step 8: Evaluation
        # ======================================================================

        if grpo_step % config.eval_every_n_steps == 0:
            print(f"Evaluating at step {grpo_step}...")

            # TODO: Evaluate on validation set
            # eval_metrics = evaluate(...)

            eval_step += 1

        # ======================================================================
        # Step 9: Save checkpoint
        # ======================================================================

        if grpo_step % config.save_every_n_steps == 0 and grpo_step > 0:
            # TODO: Save model checkpoint
            pass

    # ==========================================================================
    # Final save and cleanup
    # ==========================================================================

    print("Training complete!")
    # TODO: Save final model

    if config.use_wandb:
        wandb.finish()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    # You can use typer or argparse for CLI argument parsing
    # For now, run with default config
    config = GRPOConfig()
    grpo_train_loop(config)
