# 模块 03 · 可验证奖励 + 组内归一化优势

## 这一步在地图上的位置

RLVR 的"R"来自这里：每条 rollout 被 verifier 打一个标量奖励；然后 GRPO 家族的灵魂操作——
**用同一道题的 G 条回答互为 baseline**，把奖励变成优势(advantage)。没有 critic 网络，
这正是 GRPO 相对 PPO 的工程简化。

## 第一关：`run_compute_rollout_rewards`

纯粹的"跑批 + 记账"：

- 对每个 `(response, ground_truth)` 调 `reward_fn`，取出 `"reward"` 拼成 `(rollout_batch_size,)` 张量；
- metadata 至少记录 mean total reward 和 mean format reward——真训练时这两条曲线是你最先看的仪表。

顺手读一下真实的 verifier `cs336_alignment/drgrpo_grader.py::r1_zero_reward_fn`：
格式奖励（`</think> <answer>...</answer>` 结构完整）和答案奖励（数学等价判定，sympy/latex 归一化）分开给分。

## 第二关：`run_compute_group_normalized_rewards`

把 `(rollout_batch_size,)` 的奖励 reshape 成 `(n_prompts, group_size)`，按组做两个**正交**的操作：

```
A = (r - baseline(组)) / normalizer(组)

baseline:             "mean" → 减组内均值        | "none" → 不减
advantage_normalizer: "std"  → 除组内 std + eps  | "mean" → 除组内均值 | "none" → 不除
```

四种组合就是四个算法（这就是 2026 版接口设计的漂亮之处——论文里四个名字，代码里两个开关）：

| 算法 | baseline | normalizer | 出处/动机 |
|---|---|---|---|
| GRPO | mean | std | DeepSeekMath：组内 z-score |
| Dr-GRPO | mean | none | 去掉 ÷std：难度归一化引入偏差（全对/全错的题 std≈0，优势被 eps 放大或不稳定；接近 0.5 正确率的题被压小） |
| RFT | none | none | 拒绝采样微调视角：奖励直接当权重 |
| MaxRL | mean | mean | 除以组均值：按"期望正确率"缩放 |

实现提示：

- `rewards.view(-1, group_size)` 后沿 `dim=-1` 求均值/std，`keepdim=True` 方便广播，最后 `view(-1)` 摊平回去。
- std 用无偏还是有偏？测试用例 `[1,0,0,1]`, group_size=2 两种算出来不一样——先想清楚 torch 的默认行为（`torch.std` 默认 `correction=1`），跑一次测试让快照告诉你答案。这种"用测试当 oracle"的小实验是快照测试的正当用法。
- `advantage_eps` 加在分母上防零除，只在 `"std"` 模式用。

## ✅ 过关验证

```bash
uv run python course/check.py 3
```

## 手算练习（写代码前做）

`raw_rewards = [1, 0, 0, 1]`，group_size=2 ⇒ 两组 `[1,0]`、`[0,1]`。
手算四种模式下的 advantages，写在纸上，再跑测试对答案：

- GRPO（÷std）: 每组均值 0.5，std(有偏)=0.5 ⇒ `[+1, -1, -1, +1]`（若无偏 std≈0.707 ⇒ ±0.707…哪个能过测试？）
- Dr-GRPO: `[+0.5, -0.5, -0.5, +0.5]`
- RFT: `[1, 0, 0, 1]`
- MaxRL: `[+1, -1, -1, +1]`（÷0.5）

## 自测题

1. 为什么组内减均值是一个合法的 baseline（不改变策略梯度的期望）？
2. 一个组 G 条回答全对（r 全为 1），GRPO 和 Dr-GRPO 各给出什么优势？哪个行为更合理？
3. 为什么 GRPO 可以不要 PPO 的 value network？代价是什么？

<details><summary>参考答案</summary>

1. 策略梯度定理中减去不依赖动作的 baseline 不改变期望：`E[∇log π(a) · b] = b·∇E[Σπ(a)] = b·∇1 = 0`。组内均值是对同一 prompt 的其他样本统计量，对单条样本近似独立（严格说 leave-one-out 才无偏，GRPO 忽略这点）。
2. 全对时 r−mean = 0：两者优势都是 0（Dr-GRPO 直接是 0；GRPO 是 0/(0+eps)=0），这题没有梯度——合理，因为没有对比信息。但若组内"几乎全对"（如 7/8），GRPO 的 ÷std 会把小差异放大成大优势，对"太简单/太难"的题给出不成比例的梯度，这是 Dr-GRPO 论文指出的难度偏差(difficulty bias)。
3. 用同题多样本的经验均值代替学出来的 V(s)。省一个与策略同规模的 critic（显存、训练不稳定性都省了）。代价：每个 prompt 必须采 G 条（推理开销 ×G）；baseline 方差比训练好的 critic 高，且只有序列级(outcome)信号，没有 token 级 credit assignment。
</details>

---
下一站：[模块 04 · 策略梯度损失](04_policy_gradient.md)
