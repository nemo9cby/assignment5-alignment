# 后训练快速入门：交互式课程（基于 CS336 2026 A5）

> 目标：从零到能独立实现并跑通一个 **RLVR（可验证奖励强化学习）后训练全流程**。
> 方法论（借用 Josh Waitzkin 的学习观）：把大目标拆成最小可掌握的单元，**每一步都有一个客观的"过关信号"**，
> 学会一个单元再进入下一个。不追求快，追求每一步都站稳。

## 这门课的交互方式

1. **每个模块 = 一篇讲义 + 一个可运行的关卡（Gate）**。关卡就是仓库自带的快照测试（snapshot tests）：
   你的实现输出必须和预先录制的 `.npz` 数值完全对上，红变绿即过关。
2. **进度面板**：任何时候运行

   ```bash
   uv run python course/check.py
   ```

   会打印每个模块的过关状态和下一步建议。只跑某一关（带详细报错）：

   ```bash
   uv run python course/check.py 3
   ```

3. **GPU 部分全部走 [Modal](https://modal.com)**（模块 9），本地只需要 CPU。
   Modal 的产物（冒烟测试 / baseline 评测 / 训练结果）会写成 JSON 落回 `course/artifacts/`，
   `check.py` 靠这些文件判定 GPU 关卡是否通过——同样可验证。
4. 每篇讲义末尾有**自测题**（答案折叠在 `<details>` 里），先答后看。

## 路线图

| 模块 | 主题 | 关卡（红 → 绿） | 预计用时 |
|---|---|---|---|
| [00](00_setup.md) | 环境搭建 + 全局地图 | 26 个测试全部收集成功、全部 `NotImplementedError` | 0.5h |
| [01](01_tokenize_and_mask.md) | Tokenize prompt/response + response mask | `test_tokenize_prompt_and_output` | 1–2h |
| [02](02_logprobs_entropy.md) | 条件 log-prob 与 token 熵 | `test_get_response_log_probs` | 1–2h |
| [03](03_rewards_advantages.md) | 可验证奖励 + 组内归一化优势（GRPO / Dr-GRPO / MaxRL / RFT） | 4 个 rewards/advantages 测试 | 2h |
| [04](04_policy_gradient.md) | 策略梯度损失：on-policy → 重要性采样 → PPO/GRPO clip → GSPO | 3 个 policy-gradient 测试 | 2–3h |
| [05](05_loss_aggregation.md) | 微批聚合与两种损失归一化 | 2 个 aggregate 测试 | 1h |
| [06](06_grpo_train_step.md) | 完整 GRPO 训练步（含梯度累积、变体、off-policy） | 8 个 train-step 测试 | 2–3h |
| [07](07_eval_and_parsing.md) | 评测与答案解析（GSM8K / MMLU） | `tests/test_metrics.py` 4 个测试 | 1h |
| [08](08_sft_dpo.md) | 支线：SFT 数据打包 + DPO | `test_data` + `test_dpo` 共 3 个测试 | 2–3h |
| [09](09_modal_gpu.md) | GPU 实战（Modal）：vLLM rollout → GSM8K baseline → GRPO 训练 | 3 个 artifact 关卡 | 3h+ |
| [10](interview_questions.md) | 结业：AI Lab 面试题自测 | 全部能脱稿回答 | 2h |

模块 01–06 是主线（对应 2026 版必做部分 `tests/test_grpo.py`），07 起为评测与支线。
**主线做完，你就拥有一个通过全部快照测试的 GRPO 实现**；模块 9 把它放到真 GPU 上对 Olmo-2-1B × GSM8K 做端到端训练。

## 你要写代码的地方

所有实现写在你自己的模块里（建议 `cs336_alignment/` 下，你 2025 版的 `utils.py`、`grpo_train.py` 可以继续演化），
然后在 `tests/adapters.py` 里把 `run_*` 函数接到你的实现上。**adapters 只做转接，不写逻辑**。

> ⚠️ 2026 版对 2025 版接口做了大改（详见 [00_setup.md](00_setup.md#2025--2026-变了什么)）。
> 你 2025 年写的实现完整保留在 [`course/reference/adapters_2025_reference.py`](reference/adapters_2025_reference.py)，
> 很多函数稍作改造就能复用——这本身就是第一批练习。

## 心法（每个模块都适用）

1. **先读测试，再读讲义，最后写代码。** 测试是最精确的规格说明。
2. **先在 toy 张量上手算一遍**（batch=2, seq=4 的小例子），再写向量化实现。
3. 卡住超过 30 分钟：把中间张量的 shape 和前几个值打印出来，与手算对照——几乎所有 bug 都是 shape、mask、shift 三类之一。
4. 绿了之后问自己一句："如果面试官让我不看代码重写，我能写出来吗？"——不能就重写一遍。
