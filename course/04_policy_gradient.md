# 模块 04 · 策略梯度损失：on-policy → 重要性采样 → clip → GSPO

## 这一步在地图上的位置

优势 A 已经有了（模块 3），当前策略的 per-token log-prob 也有了（模块 2）。
现在把它们变成**每个 token 的损失**。`run_compute_policy_gradient_loss` 用一个
`importance_reweighting_method` 开关覆盖四代方法——这是全课理论密度最高的一小时。

## 逐级推导（别跳，面试必考）

**Level 0 · `"none"`（on-policy vanilla PG / REINFORCE with baseline）**

```
per_token_loss[i, t] = -A_i · log π_θ(x_t | x_<t)
```

- A 的 shape 是 `(B,)` 或 `(B,1)`，log-probs 是 `(B,T)`：广播乘。序列级优势摊给每个 token——GRPO 家族没有 token 级 credit assignment。
- 负号：优化器做梯度下降，我们要最大化 `E[A·log π]`。

**Level 1 · `"noclip"`（重要性采样，无剪切）**

rollout 是旧策略 π_old 采的，但参数已经更新过几步（off-policy）。修正分布错位：

```
ratio[i, t] = exp(policy_log_probs - old_log_probs)      # π_θ / π_old, per token
loss = -A · ratio
```

注意梯度只流经 `policy_log_probs`；`old_log_probs` 是常数（rollout 时缓存的，本身就 no-grad）。

**Level 2 · `"grpo"`（PPO/GRPO-style token 级剪切）**

ratio 无界 → 一次更新可能把策略拉飞。PPO 的答案是悲观下界：

```
loss = -min( ratio · A,  clip(ratio, 1-ε, 1+ε) · A )
```

理解 `min` 而不是"直接 clip"：对 A>0，超过 1+ε 的 ratio 不再给额外奖励（防过冲）；
对 A<0，ratio 掉到 1-ε 以下时取**未剪切**项（坏动作的惩罚不封顶）。两侧都取悲观值。

metadata 里记录 clip 触发比例（clip fraction）——训练时这条曲线告诉你策略更新的"步幅"。

**Level 3 · `"gspo"`（GSPO：序列级剪切，Qwen 2025）**

GRPO 的 ratio 是 per-token 的：长序列上 token ratio 的乘积方差爆炸，且 token 级剪切与
序列级奖励错位（MoE 模型上尤其不稳）。GSPO 把重要性比率定义在**序列级、做长度归一化**：

```
s_i = exp( mean_{t∈response}(policy_log_probs - old_log_probs) )   # 几何平均 ratio
loss[i, t] = -min( s_i · A_i,  clip(s_i, 1-ε, 1+ε) · A_i )         # 整条序列同一个 s_i
```

- `mean` 只在 response token 上取——这就是这个函数签名里 `response_mask` 的用途：
  `(Σ mask·Δlogp) / Σ mask`，再广播回 `(B, T)`。
- 想一想：为什么用几何平均（log 空间算术平均）而不是把 token ratio 直接连乘？

## 实现提示

- 一个函数四个分支，共享骨架：先算 `coef`（1、ratio 或 s_i），再 `-min(coef·A, clip(coef)·A)`（`"none"`/`"noclip"` 无 clip 分支）。
- `"none"` 时的 `coef·A` 里没有 `log π`？——注意 Level 0 的形式不同：`-A·log π` 本身，不是 ratio 形式。别把四个分支硬套一个公式。
- clip 用 `torch.clamp(ratio, 1-ε, 1+ε)`；min 用 `torch.minimum`（逐元素）。
- 全程不要 `detach` `policy_log_probs`；`old_log_probs` 进来时已无梯度。

## ✅ 过关验证

```bash
uv run python course/check.py 4
```

## 自测题

1. 证明 `"none"` 模式是 `"noclip"` 在 π_θ = π_old 处的一阶等价（梯度相同）。
2. ε=0.1 时，某 token ratio=1.3、A>0，梯度是多少？ratio=0.7、A<0 呢？
3. 为什么 GSPO 的序列 ratio 要做长度归一化（÷|y|）？不做会怎样？

<details><summary>参考答案</summary>

1. `∇(-A·ratio) = -A·ratio·∇log π_θ`。在 π_θ=π_old 处 ratio=1，梯度为 `-A·∇log π_θ`，与 `∇(-A·log π_θ)` 相同。所以第一步微批更新两者一致，差异随策略偏离累积。
2. ratio=1.3 > 1+ε，A>0：min 取剪切项 `clip·A`，clip 后是常数 → **梯度为 0**（该 token 停止推高）。ratio=0.7 < 1−ε 且 A<0：min 取**未剪切**项 `ratio·A`（更负），梯度非零 → 坏 token 继续被压。这个不对称正是"悲观下界"的含义。
3. 不归一化时 `Σ_t Δlogp` 的方差随长度线性增长，长回答的 ratio 指数爆炸/塌缩，clip 几乎必然触发，长度成为混杂因子。几何平均让不同长度序列的 ratio 落在可比尺度上，ε 才有统一含义。
</details>

---
下一站：[模块 05 · 微批聚合与损失归一化](05_loss_aggregation.md)
