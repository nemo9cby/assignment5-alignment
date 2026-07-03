# 模块 06 · 完整 GRPO 训练步（主线大 Boss）

## 这一步在地图上的位置

把模块 1–5 的零件装配成 `run_grpo_train_step`：**给定一批 rollout，完成一次 optimizer 更新**。
8 个快照测试会用一个 8 维玩具 GPT-2 走完真实的 forward/backward/step，
逐参数比对更新后的权重——装配中任何一环错一点，参数就对不上。这是主线的期末考。

## 装配清单（按数据流顺序）

```
1. tokenize:   repeated_prompts × rollout_responses → input_ids, labels, response_mask   (模块1)
2. rewards:    reward_fn × responses × ground_truths → raw_rewards                        (模块3)
3. advantages: group normalize (baseline, advantage_normalizer)                           (模块3)
4. 微批循环 for k in range(gradient_accumulation_steps):
     切片第 k 个微批 (batch 均分)
     policy_log_probs = 前向(model, 微批)                                                 (模块2)
     per_token_loss  = policy_gradient_loss(A, log_probs, method, old_logp, ε, mask)      (模块4)
     microbatch_loss = aggregate(per_token_loss, mask, loss_normalization, C)             (模块5)
     (microbatch_loss / gradient_accumulation_steps).backward()
5. grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)   # 若 max_grad_norm 非 None
6. optimizer.step(); optimizer.zero_grad(set_to_none=True)
7. 返回 (loss, metadata)
```

## 容易翻车的装配细节（每一条都对应测试的一种失败方式）

- **微批切法**：`rollout_batch_size` 均分成 `gradient_accumulation_steps` 份，**按顺序切片**（测试 B=4、accum=2 → 每微批 2 条）。advantages / old_log_probs 也要切同样的片。
- **÷ accumulation steps**：每个微批 loss 除以微批数再 backward，这样累积起来的梯度等于"全批平均"。忘除 → 梯度大 N 倍 → 参数快照对不上。
- **先全部 backward，再统一 clip + step 一次**——不是每个微批 step 一次。
- **tokenize 一次还是每微批一次？** 先对全批 tokenize 再切片（保证 padding 长度一致，old_log_probs 的 T 维才能对齐——测试给的 old_log_probs 是按全批 max_len 构造的）。
- **step 之后 zero_grad**：测试最后断言 `param.grad is None`，用 `set_to_none=True`。
- **返回的 loss**：报账用（已 detach 与否看你的口味，测试只比数值）；跨微批怎么合成一个数，想想什么统计量对得上"整批平均 loss"。
- 熵、clip fraction、grad norm 放进 metadata——真训练时的仪表盘（测试不比对 metadata，但别偷懒，模块 9 要用）。

## 变体矩阵（8 个测试在测什么）

| 测试 | 开关组合 | 验证的知识点 |
|---|---|---|
| `standard_on_policy` | 默认: mean/std/sequence, method=none | 基础装配 |
| `variants[grpo_constant]` | mean/std + constant(C=32) | 聚合切换 |
| `variants[dr_grpo]` | mean/none + constant | Dr-GRPO 配方 |
| `variants[rft]` | none/none + constant | RFT 配方 |
| `variants[maxrl]` | mean/mean + constant | MaxRL 配方 |
| `off_policy[noclip]` | + old_log_probs | IS ratio 接线 |
| `off_policy[grpo]` | + cliprange | token 级 clip |
| `off_policy[gspo]` | + cliprange, 序列级 | GSPO + mask 传递 |

建议通关顺序：先让 `standard_on_policy` 绿（装配正确性），再逐个开开关——每个新绿的测试
都精确告诉你哪个开关接对了。**一次只拧一个开关**，这是调试的基本功。

## ✅ 过关验证

```bash
uv run python course/check.py 6      # 全部 8 个
# 单个: uv run pytest "tests/test_grpo.py::test_grpo_train_step_off_policy[gspo]" -v
```

通关后跑一次全量主线回归：`uv run pytest tests/test_grpo.py -q` 应该 **19 passed**。

## 自测题

1. 梯度累积在数学上等价于什么？什么情况下它和"直接大 batch"**不**等价？
2. 为什么 `clip_grad_norm_` 要在所有微批 backward 完成后做，而不是每个微批做一次？
3. 测试断言 step 后 `param.grad is None`，工程上为什么偏好 `set_to_none=True` 而不是置零？

<details><summary>参考答案</summary>

1. 等价于用 N 倍大的 batch 算一次平均梯度再 step（loss 线性可加时严格成立）。不等价的情况：模型 forward 有跨样本的 batch 统计量（如 BatchNorm）、loss 归一化依赖整批统计（比如分母用了"本批总响应 token 数"——各微批分母不同就不能直接平均，这正是模块 5 `"constant"` 设计的动机之一）。
2. clip 的对象是"整个 batch 的累积梯度"的范数。逐微批 clip 相当于对部分和分别限幅，最终梯度既不是无 clip 的平均，也不是对全量的 clip，目标被扭曲且与 accumulation steps 数耦合。
3. 置零保留了梯度张量的内存并让下次 backward 做"加法"，`set_to_none` 释放张量，下次 backward 直接赋值：省显存、少一次 kernel，还能暴露"忘了 backward 就 step"之类的 bug（会直接报 None 而不是静默用 0）。
</details>

---
下一站：[模块 07 · 评测与答案解析](07_eval_and_parsing.md)
