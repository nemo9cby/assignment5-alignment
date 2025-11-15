Based on the assignment requirements, here's a structured plan for implementing the LLM alignment techniques:

  Phase 1: Zero-shot Baseline (Section 3)

  Goal: Establish baseline performance of Qwen 2.5 Math 1.5B on MATH dataset

  Components to implement:
  1. Model and Data Loading
    - Load Qwen 2.5 Math 1.5B Base from /data/a5-alignment/models/Qwen2.5-Math-1.5B
    - Load MATH validation data from /data/a5-alignment/MATH/validation.jsonl
    - Set up vLLM for efficient inference
  2. Evaluation Pipeline
    - Format questions using r1_zero prompt template
    - Generate responses with vLLM (temp=1.0, max_tokens=1024, stop at </answer>)
    - Parse responses and compute rewards using r1_zero_reward_fn
    - Track format rewards vs answer rewards

  Phase 2: Supervised Fine-tuning (Section 4)

  Goal: Fine-tune model on reasoning traces from DeepSeek R1

  Components to implement:
  1. Helper Methods (tested functions):
    - tokenize_prompt_and_output() - Tokenize Q&A pairs with response masks
    - compute_entropy() - Calculate per-token entropy
    - get_response_log_probs() - Get token log probabilities from model
    - masked_normalize() - Normalize tensors with masking
    - sft_microbatch_train_step() - Single microbatch update
  2. SFT Training Loop
    - Load SFT data from /data/a5-alignment/MATH/sft.jsonl
    - Implement gradient accumulation for memory efficiency
    - Set up dual-GPU: one for policy, one for vLLM evaluation
    - Periodic evaluation on validation set
    - Log generations for monitoring

  Phase 3: Expert Iteration (Section 5)

  Goal: Bootstrap reasoning by filtering self-generated correct solutions

  Algorithm components:
  1. Sample G outputs per question from current policy
  2. Compute rewards for each output
  3. Filter to keep only correct outputs
  4. Run SFT on filtered dataset
  5. Iterate for n_ei_steps

  Key hyperparameters: G (rollouts per question), batch size, epochs

  Phase 4: GRPO Implementation (Sections 6-7)

  Goal: Use policy gradients with verified rewards

  Components to implement:
  1. Reward Processing
    - compute_group_normalized_rewards() - Normalize rewards within groups
    - Handle both std normalization and mean-only normalization
  2. Policy Gradient Losses
    - compute_naive_policy_gradient_loss() - Basic REINFORCE loss
    - compute_grpo_clip_loss() - PPO-style clipped loss
    - compute_policy_gradient_loss() - Wrapper for different loss types
  3. Utility Functions
    - masked_mean() - Average with masking
    - grpo_microbatch_train_step() - GRPO training step
  4. GRPO Training Loop
    - Rollout generation with old policy
    - Group normalization of rewards
    - Multiple gradient steps per rollout batch (off-policy)
    - Clipping for stability

  Phase 5: Experiments (Section 8)

  Ablations and hyperparameter tuning:
  1. Learning rate sweep
  2. Baseline comparison (no baseline vs normalized rewards)
  3. Length normalization (masked_mean vs masked_normalize)
  4. Group std normalization ablation
  5. On-policy vs off-policy comparison
  6. Clipping ablation
  7. Prompt ablation (r1_zero vs question_only)

  Phase 6: Leaderboard (Section 9)

  Goal: Maximize MATH validation accuracy within 4 GPU hours
  - Optimize hyperparameters
  - Consider system optimizations (torch.compile, mixed precision)
  - Curriculum learning or data filtering strategies

  🔧 Implementation Strategy

  Testing Approach:

  - Implement each function to pass tests in tests/adapters.py
  - Use snapshot testing for numerical verification
  - Test functions individually before integration

  Key Technical Considerations:

  1. Memory Management
    - Use gradient accumulation (suggested: 128 steps)
    - Load models in bfloat16 with FlashAttention-2
    - GPU memory utilization: 0.85
  2. Multi-GPU Setup
    - Policy model on one GPU
    - vLLM inference on another GPU
    - Use patches for vLLM to work correctly
  3. Logging & Monitoring
    - Use wandb for experiment tracking
    - Log train/validation metrics separately
    - Monitor entropy, response length, clip fractions
    - Save example generations periodically

  Critical Implementation Details:

  - Tokenizer uses left-padding by default
  - Careful with index shifting for causal LM (logits[:-1] predicts tokens[1:])
  - Response masks must exclude prompt and padding tokens
  - Gradient clipping value: 1.0
  - Stop generation at second </answer> tag

  💭 Discussion Points

  Before we start implementing:

  1. Priority Order: Should we implement in the order presented (baseline → SFT → EI → GRPO) or start with shared components first?
  2. Testing Strategy: Do you want to implement test adapters first to ensure correctness, or build the main training loops and then adapt to tests?
  3. Resource Management: The assignment uses 2 H100 GPUs. How should we handle development/debugging on potentially limited resources?
  4. Code Organization: Should we create separate modules for each technique (sft.py, grpo.py, etc.) or organize by functionality (losses.py, training.py, etc.)?
  5. Evaluation Frequency: The assignment suggests evaluating every 5-10 steps. Should we make this configurable?
  6. Checkpoint Strategy: How often should we save model checkpoints during training?
