# 模块 00 · 环境搭建 + 全局地图

## 这一关的目标

1. 本地 CPU 环境跑通，26 个测试全部收集成功、全部以 `NotImplementedError` 失败（这就是你的"全红基线"）。
2. 建好 Modal 账号（模块 9 用，现在只花 5 分钟注册）。
3. 在脑子里装一张后训练全局地图：知道接下来每个模块在图上的位置。

## 1. 环境

```bash
# CPU 环境只需要这一步(2026 版把 flash-attn/vllm 移进了 gpu extra，本地不用装)
uv sync

# 全红基线
uv run pytest tests/ -q
# 期望: 26 failed, 全部 NotImplementedError
```

Modal（GPU 部分才用，先注册好）：

```bash
uv run modal setup   # 浏览器登录，新账号每月送 $30 免费额度
```

### ✅ 过关验证

```bash
uv run python course/check.py 0
```

## 2. 全局地图：后训练在做什么

一个基座模型（pretrained LM）只会"续写"。后训练（post-training）把它变成"做事"的模型，主流流水线是三段：

```
预训练 LM ──SFT──▶ 会按格式回答 ──RL(HF/VR)──▶ 回答得更好
                                │
                                ├─ RLHF: 奖励来自 reward model(人类偏好) —— 对话/写作类能力
                                ├─ RLVR: 奖励来自 verifier(可自动判对错) —— 数学/代码/推理类能力 ← 本课主线
                                └─ DPO:  跳过 RL，直接在偏好对上做监督式优化 —— 模块 8
```

本作业（CS336 2026 A5）的主线是 **RLVR**：用 GSM8K 数学题训练 Olmo-2-1B，
奖励函数就是"答案对不对 + 格式对不对"（`cs336_alignment/drgrpo_grader.py::r1_zero_reward_fn`），
优化算法是 **GRPO** 及其 2024–2025 年的一系列变体（Dr-GRPO、MaxRL、GSPO、RFT）。

一次 RLVR 迭代的数据流（把这张图背下来，后面每个模块就是其中一个箭头）：

```
                 ┌────────────────────────────────────────────────┐
                 ▼                                                │
  问题 batch ─▶ 采样(vLLM): 每题 G 条回答 ─▶ 奖励 r(可验证) ─▶ 组内归一化 ─▶ 优势 A
                                                                  │
  新策略 π_θ ◀─ optimizer.step ◀─ 聚合 loss ◀─ 策略梯度 loss(θ) ◀──┘
                     (模块6)        (模块5)        (模块4)      (模块3)
  其中 loss 需要 log π_θ(response|prompt) ── 模块1(tokenize+mask) + 模块2(logprobs)
```

## 3. 2025 → 2026 变了什么

你在 2025 版上已经写过 SFT 和大半个 GRPO（保留在 `course/reference/adapters_2025_reference.py`）。2026 版的主要变化：

| 方面 | 2025 | 2026 |
|---|---|---|
| 任务 | Qwen-2.5-Math-1.5B × MATH | **Olmo-2-1B × GSM8K**（更小更便宜） |
| 必做测试 | test_sft.py + test_grpo.py | 只剩 `test_grpo.py`（SFT 挪进选做支线） |
| 损失接口 | `compute_naive_pg_loss` / `compute_grpo_clip_loss` 分开 | 统一成 `run_compute_policy_gradient_loss(importance_reweighting_method=...)`，新增 **GSPO** |
| 优势接口 | `normalize_by_std: bool` | `baseline` × `advantage_normalizer` 两个正交开关，覆盖 GRPO / **Dr-GRPO / MaxRL / RFT** |
| 聚合 | `masked_mean` / `masked_normalize` | `run_aggregate_loss_across_microbatch(loss_normalization="sequence"|"constant")` |
| 训练步 | microbatch train step | `run_grpo_train_step` 直接管理**整个梯度累积循环** + grad clip |
| 基建 | 自己拼 | 官方给了 `vllm_utils.py`（vLLM server + NCCL 权重同步）和 `modal_utils.py` |

对照旧实现读一遍新接口 `tests/adapters.py`，列出"哪些函数能直接搬、哪些要重构"——这是本模块的热身练习。

## 4. 读代码顺序（30 分钟）

1. `tests/adapters.py` —— 12 个 `run_*` 函数的 docstring，就是全部规格。
2. `tests/test_grpo.py` —— 每个测试怎么调用 adapter、比对什么。
3. `tests/conftest.py` —— fixtures：`tokenizer`（15 个词的玩具 word-level tokenizer）、`model`（tiny-gpt2）、`tiny_train_model`（8 维单层 GPT-2，专供训练步测试）、各种预置张量的 shape。
4. `cs336_alignment/drgrpo_grader.py` 的 `r1_zero_reward_fn` —— 奖励怎么算的（只读接口和返回值）。

## 自测题

1. 为什么 RLVR 用 GSM8K 这类任务，而不是开放式对话？
2. 快照测试(snapshot test)相比"自己肉眼看输出"的验证方式，本质优势是什么？

<details><summary>参考答案</summary>

1. 因为奖励可以由程序自动验证（答案数值对/错），不需要训练 reward model，奖励无噪声、不可被"讨好"——这是 2024 后推理模型（DeepSeek-R1 等）能力跃升的关键配方。开放式对话的奖励只能来自人类偏好建模，会引入 reward hacking 空间。
2. 快照测试把"正确"定义为与参考实现**数值逐元素一致**（含 rtol/atol），能抓住肉眼看不出来的错误（差一个 shift、mask 少乘一位、均值分母错了）。这也是本课"每一步可验证"的基石。
</details>

---
下一站：[模块 01 · Tokenize prompt/response + response mask](01_tokenize_and_mask.md)
