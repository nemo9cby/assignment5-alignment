# 模块 05 · 微批聚合与损失归一化

## 这一步在地图上的位置

模块 4 产出 `(B, T)` 的 per-token loss，优化器需要一个标量。
"怎么把带 mask 的矩阵平均成标量"听起来是琐事，实际上**不同的归一化 = 不同的优化目标**，
它决定长回答和短回答谁的梯度大——这是 2025 年 Dr-GRPO/DAPO 论文吵的核心问题之一。

## 规格（`run_aggregate_loss_across_microbatch`）

**`"sequence"`（GRPO 原版：先句内平均，再句间平均）**

```
loss = mean_i ( Σ_t loss[i,t]·mask[i,t] / Σ_t mask[i,t] )
```

每条序列先除以**自己的**响应长度 → 每条序列在 batch 中权重相等。
后果：长回答里每个 token 的梯度被 1/|y_i| 稀释——答错且啰嗦的序列，单 token 惩罚反而小，
这被指出会催生"越错越长"的病态激励。

**`"constant"`（Dr-GRPO 风格：总和除以常数）**

```
loss = Σ_{i,t} loss[i,t]·mask[i,t] / (B · C)
```

C 是训练前固定的常数（如最大生成长度），**不随本批实际长度变**。每个 token 权重相等，
去掉长度偏差。注意除的是 `batch_size × normalization_constant`——想想为什么还要除 B
（提示：微批 loss 之后要跨微批平均，见模块 6；让每条序列贡献 `Σ/C`，batch 内求 mean）。

> 具体分母形式先按自己的推导写，跑测试用快照校准（C=42、B=2 的用例能区分 `B·C` 和 `C`）。
> 又一次"以测试为 oracle 做受控实验"。

## 实现提示

- 就是一个带 mask 的加权平均，注意 `mask` 转 float 再乘，避免 bool 张量做除法。
- 别用 `torch.masked_select`——形状信息丢了之后没法按序列归一化。
- 返回标量 tensor（`.mean()`/`.sum()` 的结果），**不要 `.item()`**——测试之外它还要 `backward()`。

## ✅ 过关验证

```bash
uv run python course/check.py 5
```

## 自测题

1. 两条序列，response 长度 10 和 1000，各 token loss 相同。`"sequence"` 模式下两条序列单个 token 的梯度之比是多少？`"constant"` 模式呢？
2. 为什么 `"constant"` 的 C 必须在整个训练中固定，而不能用"本 batch 最长长度"？

<details><summary>参考答案</summary>

1. `"sequence"`：每序列内除以自身长度，单 token 梯度比为 100:1（短序列 token 梯度大 100 倍）。`"constant"`：所有 token 同权，1:1（但长序列整体贡献是短的 100 倍）。
2. 若 C 随 batch 变化，梯度尺度在训练过程中漂移，等价于隐式的、由数据长度决定的学习率调度；不同 batch 之间目标函数不再一致，也破坏梯度累积的正确性（各微批的分母不同就不能简单相加平均）。固定 C 保证目标是良定义的 `E[Σ_t loss_t]/C`。
</details>

---
下一站：[模块 06 · 完整 GRPO 训练步](06_grpo_train_step.md)
