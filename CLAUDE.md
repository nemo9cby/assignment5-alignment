# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Purpose

CS 336 Spring **2026** Assignment 5: Reasoning RL (synced with upstream `stanford-cs336/assignment5-alignment`) - Educational assignment teaching LLM post-training: GRPO and its variants (Dr-GRPO, MaxRL, RFT, GSPO), RLVR on GSM8K with Olmo-2-1B, plus optional SFT/DPO/safety components.

This fork additionally contains an interactive course in `course/` (in Chinese) that structures the assignment into verifiable modules. See `course/README.md`.

## Development Commands

### Setup and Installation
```bash
# CPU-only local dev (flash-attn/vllm live in the `gpu` extra now)
uv sync

# GPU environments (Modal images do this automatically)
uv sync --extra gpu
```

### Testing
```bash
# Run all tests (26 tests; all NotImplementedError before implementation)
uv run pytest

# Main required suite (19 tests)
uv run pytest tests/test_grpo.py -v

# Single test
uv run pytest "tests/test_grpo.py::test_grpo_train_step_off_policy[gspo]" -v

# Course progress dashboard / per-module gates
uv run python course/check.py
uv run python course/check.py 3
```

## Code Architecture

### Implementation Structure
Students implement functions in `tests/adapters.py` that connect to the test suite. The assignment uses **snapshot testing** - comparing outputs against pre-recorded `.npz` files in `tests/_snapshots/`. Real logic should live in `cs336_alignment/`; adapters only forward calls.

### Key Components

**Core Adapters** (in `tests/adapters.py`, 2026 interface):
- `run_tokenize_prompt_and_output()`, `run_get_response_log_probs()`
- `run_compute_rollout_rewards()`, `run_compute_group_normalized_rewards()` (baseline × advantage_normalizer switches cover GRPO/Dr-GRPO/RFT/MaxRL)
- `run_compute_policy_gradient_loss()` (`importance_reweighting_method`: none/noclip/grpo/gspo)
- `run_aggregate_loss_across_microbatch()` (sequence vs constant normalization)
- `run_grpo_train_step()` (full step: gradient accumulation + grad clip + optimizer step)
- Optional: `get_packed_sft_dataset()`, `run_iterate_batches()`, `run_parse_mmlu_response()`, `run_parse_gsm8k_response()`, `run_compute_per_instance_dpo_loss()`

**Test Organization**:
- `test_grpo.py`: 19 required tests (tokenization → losses → full train steps with variants)
- `test_data.py`, `test_dpo.py`, `test_metrics.py`: optional supplement (SFT packing, DPO, response parsing)
- Note: 2025's `test_sft.py` was removed upstream; the 2025 implementations are preserved in `course/reference/adapters_2025_reference.py`

**Infra provided by upstream**:
- `cs336_alignment/vllm_utils.py`: vLLM server lifecycle + NCCL weight sync
- `cs336_alignment/modal_utils.py`: Modal job submission (requires setting SUNET_ID; defaults to B200:2 - downscale for personal accounts)
- `cs336_alignment/drgrpo_grader.py`: `r1_zero_reward_fn` verifier (format + answer rewards)
- `course/modal/`: lightweight Modal scripts for the course (smoke test, GSM8K baseline eval)

**Data and Models**:
- `data/gsm8k/`: train (7473) / test (1319) JSONL; ground-truth answer appears after `#### `
- `tests/fixtures/`: tiny-gpt2 models, toy word-level tokenizers (see conftest)
- `cs336_alignment/prompts/r1_zero.prompt`: main RLVR prompt template

## Important Implementation Notes

1. **Masking**: `response_mask` aligns with `labels` (already shifted), 1 on response tokens only.
2. **Shifting**: `input_ids = seq[:-1]`, `labels = seq[1:]`; do NOT shift again inside log-prob computation.
3. **Advantages**: reshape rewards to `(n_prompts, group_size)`; same-prompt rollouts are adjacent.
4. **Gradient accumulation**: divide each microbatch loss by accumulation steps; clip once after all backwards; `zero_grad(set_to_none=True)` after step.
5. **Precision**: keep float32; snapshots compare numerically (rtol=1e-4 typical).

## Assignment Resources

- Main handout: `cs336_spring2026_assignment5_alignment.pdf`
- Optional safety/RLHF: `cs336_spring2026_assignment5_supplement_safety_rlhf.pdf`
- Interactive course: `course/README.md` (modules 00-09 + interview question set)

## Style
- Act as a coach like Josh Waitzkin, focuses on LLM/PyTorch foundamentals and progress steps by step
- Guide toward solutions rather than writing assignment implementations wholesale; the learning is the point
