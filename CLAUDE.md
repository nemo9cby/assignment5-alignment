# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

CS 336 Spring 2025 Assignment 5: Alignment - Educational assignment teaching LLM alignment techniques including SFT, Expert Iteration, GRPO, DPO, and optional RLHF/Safety components.

## Development Commands

### Setup and Installation
```bash
# Install dependencies (two-step process due to flash-attn compilation)
uv sync --no-install-package flash-attn
uv sync
```

### Testing
```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_sft.py -v

# Run single test
uv run pytest tests/test_sft.py::test_tokenize_prompt_and_output -v

# Create submission package (runs tests and creates zip)
./test_and_make_submission.sh
```

## Code Architecture

### Implementation Structure
Students implement functions in `tests/adapters.py` that connect to the test suite. The assignment uses **snapshot testing** - comparing outputs against pre-recorded `.npz` files in `tests/_snapshots/`.

### Key Components

**Core Alignment Implementations** (in `tests/adapters.py`):
- **SFT Functions**: `run_tokenize_prompt_and_output()`, `run_get_response_log_probs()`, `run_compute_entropy()`, `run_sft_microbatch_train_step()`
- **GRPO Functions**: `run_compute_group_normalized_rewards()`, `run_compute_naive_policy_gradient_loss()`, `run_compute_grpo_clip_loss()`, `run_grpo_microbatch_train_step()`
- **DPO Functions**: `run_compute_per_instance_dpo_loss()`
- **Utility Functions**: `run_masked_mean()`, `run_masked_normalize()`

**Test Organization**:
- `test_sft.py`: 8 tests for supervised fine-tuning
- `test_grpo.py`: 13+ tests for group relative policy optimization
- `test_dpo.py`: Direct preference optimization tests
- `test_data.py`: Dataset packing and batching tests
- `test_metrics.py`: Response parsing tests (MMLU, GSM8K)

**Data and Models**:
- `data/`: Contains GSM8K, AlpacaEval, MMLU benchmarks
- `tests/fixtures/`: Test models (tiny-gpt2) and sample data
- `cs336_alignment/prompts/`: Prompt templates for different tasks

### Testing Architecture

Tests use pytest fixtures from `conftest.py` providing:
- Pre-computed tensors (logits, rewards, masks)
- Model/tokenizer loading utilities
- Deterministic dummy reward functions
- Standard batch configurations (batch_size=2, seq_length=10, group_size=4)

Snapshot files verify numerical correctness - implementations must match expected outputs exactly.

## Important Implementation Notes

1. **Masking Operations**: Many functions require careful attention to masking (response_mask, attention_mask). Always verify shapes match expectations.

2. **Log Probability Computation**: When computing log probabilities from logits, shift tokens appropriately (logits[:-1] predicts tokens[1:]).

3. **Reward Normalization**: GRPO requires normalizing rewards within groups while handling masking correctly.

4. **Loss Functions**: All loss functions should return per-instance losses (not averaged) for the test suite to verify.

5. **Tokenizer Padding**: The tokenizer uses left-padding by default. Handle padding tokens carefully in computations.

## Common Pitfalls

- Flash-attn compilation issues: Install in two steps as shown above
- Snapshot mismatches: Ensure numerical precision matches (use float32 consistently)
- Masking errors: Double-check mask broadcasting and element-wise operations
- Index shifting: Careful with causal language modeling's shifted indices

## Assignment Resources

- Main handout: `cs336_spring2025_assignment5_alignment.pdf`
- Optional safety/RLHF: `cs336_spring2025_assignment5_supplement_safety_rlhf.pdf`
- References in code comments link to DeepSeekMath and DeepSeek-R1 papers