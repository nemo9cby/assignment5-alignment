# 模块 02 · 条件 log-prob 与 token 熵

## 这一步在地图上的位置

策略梯度 loss 的核心量是 `log π_θ(response | prompt)`，它等于 response 每个 token 的
**条件 log 概率**之和。这一模块实现：给定模型、`input_ids`、`labels`，取出每个位置上
"label token 的 log 概率"，并（可选）计算每个位置的**下一 token 分布熵**——训练时监控
熵是判断"模型是否还在探索"的第一仪表。

## 规格（`run_get_response_log_probs`）

```
logits = model(input_ids).logits          # (B, T, V)
log_probs[i, t] = log softmax(logits[i, t])[labels[i, t]]     # (B, T)
token_entropy[i, t] = H(softmax(logits[i, t]))                # (B, T), 可选
```

注意：模块 1 已经做过 shift，这里 `input_ids` 和 `labels` 天生对齐，**不要再 shift 一次**。

## 三个数值细节（面试也爱考）

1. **先 log_softmax 再取值，不要 softmax 之后再 log**。`log(softmax(x))` 在概率接近 0 时下溢成 `-inf`；`log_softmax` 内部用 log-sum-exp 技巧数值稳定。
2. **取出 label 对应概率用 `gather`**：
   `log_probs = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)`
   逐位置索引的 for 循环也对，但慢两个数量级。
3. **熵的稳定算法**：`H = -Σ p·log p`。直接算 `p * log p` 在 p→0 时是 `0 * -inf = nan`。
   用 `H = logsumexp(logits) - Σ softmax(logits)·logits`（推导见自测题），或
   `-(log_p.exp() * log_p).sum(-1)` 也可（`exp(log_p)` 处的 0 乘有限数没问题——想想为什么这里 log_p 不是 -inf 时才安全，测试的 tiny 模型不会触发极端值，但要知道边界在哪）。

pad 位置怎么办？——这一层**不管 mask**：对所有位置都算，交给下游用 `response_mask` 筛。
分层设计让每个函数只做一件事，这是这套代码的架构原则。

## ✅ 过关验证

```bash
uv run python course/check.py 2
```

## 常见翻车点

- 又 shift 了一次（logits[:, :-1] 对 labels[:, 1:]）→ 数值全部错位。
- 用 float16 算 softmax → 快照精度对不上。保持 float32。
- 熵按错误的维度求和（应沿 vocab 维 `dim=-1`）。
- 返回 dict 的 key 拼错：必须是 `"log_probs"` 和 `"token_entropy"`。

## 自测题

1. 推导：为什么 `H = logsumexp(z) - Σ softmax(z)·z`（z 为 logits）？
2. RLVR 训练里"熵坍缩(entropy collapse)"指什么？为什么它预示训练将失去增益？
3. 这个函数在训练步里会被调用几次？（提示：想想 on-policy 需要什么、off-policy 需要什么。）

<details><summary>参考答案</summary>

1. `p_i = e^{z_i}/Z`，`log p_i = z_i - log Z`，其中 `log Z = logsumexp(z)`。
   `H = -Σ p_i log p_i = -Σ p_i (z_i - log Z) = log Z - Σ p_i z_i`（用了 `Σ p_i = 1`）。
2. 策略分布的平均 token 熵快速降到接近 0：模型对每题几乎只采样一种回答。组内 G 条回答趋同 → 组内奖励方差为 0 → 优势为 0 → 梯度消失，训练停滞；同时探索停止，pass@k 不再提升。常见对策：熵正则/更高采样温度、clip-higher（DAPO）、控制 KL、早停。
3. on-policy 一次（当前策略打分，带梯度）；off-policy 两次角色——旧策略的 `old_log_probs` 在 rollout 时以 `no_grad` 算一次并缓存，当前策略每个微批前向一次（带梯度）。
</details>

---
下一站：[模块 03 · 可验证奖励 + 组内优势](03_rewards_advantages.md)
