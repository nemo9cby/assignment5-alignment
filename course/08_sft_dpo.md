# 模块 08 · 支线：SFT 数据打包 + DPO

> 支线但不冷门：**SFT 数据管线和 DPO 是面试出现频率最高的两个后训练话题**（比 GRPO 还高，
> 因为几乎所有 lab 都在用）。主线赶进度可以先跳过，结业前建议补上。

## 第一关：`get_packed_sft_dataset`（`tests/test_data.py`）

指令微调的数据端：把变长的 (prompt, response) 文档流变成**定长训练样本**。

```
每条样本 → 模板渲染(见 cs336_alignment/prompts_safety/alpaca_sft.prompt) → tokenize
所有文档 tokens 首尾相接(文档间用 eos 分隔) → 切成 seq_length 的等长块 → 每块一个训练样本
input_ids = 块[:-1] 对应的定长版本; labels = 左移一位
```

要点：

- **packing vs padding**：padding 浪费算力（大量 pad token 不产生学习信号），packing 把利用率拉满，代价是"一个样本可能横跨两篇文档 + 注意力跨文档泄漏"。这个 trade-off 会考（生产系统还会用 document masking / sequence packing with block-diagonal attention 消除泄漏）。
- `shuffle=True` 是**打乱文档顺序再 packing**，不是打乱切好的块（读测试怎么断言的）。
- 数据是 `tests/fixtures/sft_sample.jsonl`，期望输出直接给了 `tokenized_sft_sample.json`——
  逐 token 比对，模板哪里多个空格都会红，**用期望文件反推模板拼接细节**（含 eos 放哪）。
- tokenizer 是真的 Llama-3（只有 tokenizer 文件没有权重），注意 `add_special_tokens` 与 bos 行为。

`run_iterate_batches`：一个 epoch 的批迭代器。`torch.utils.data.DataLoader(dataset, batch_size, shuffle)` 就够——知道什么时候**不**造轮子也是基本功。

## 第二关：DPO（`tests/test_dpo.py`）

RLHF 三件套（reward model + rollout + PPO）太重，DPO 把"从偏好学习"变成一个监督损失：

```
L = -log σ( β·[ (log π(y_w|x) - log π_ref(y_w|x)) - (log π(y_l|x) - log π_ref(y_l|x)) ] )
```

`run_compute_per_instance_dpo_loss` 单实例版：

1. 把 prompt 分别与 chosen/rejected 拼接 tokenize（复用模块 1 的思路；本测试用 Alpaca 模板拼接 prompt/response，读 `tests/fixtures` 的 tokenizer 词表——`### Instruction:`/`### Response:` 都在里面，是提示）。**加上 eos**：response 的结束也是要学的行为。
2. 四次前向：π 和 π_ref 各对 chosen/rejected 打分（response token 上的 log-prob 之和，复用模块 2）。`π_ref` 前向包在 `torch.no_grad()` 里。
3. 套公式。`torch.nn.functional.logsigmoid` 比 `log(sigmoid(x))` 稳定。

理解检查：β 前面那两个差是**隐式奖励** `r(x,y) = β·log(π/π_ref)`——DPO 论文的核心变换是
"最优策略与奖励一一对应，于是奖励建模可以直接参数化为策略"。这句话要能展开讲（见面试题）。

## ✅ 过关验证

```bash
uv run python course/check.py 8
```

## 自测题

1. DPO 里 reference model 起什么作用？把 π_ref 去掉（设 log π_ref ≡ 0）会发生什么？
2. 为什么 DPO 训练常观察到 chosen 和 rejected 的 log-prob **双双下降**？这是 bug 吗？
3. packing 时不做跨文档注意力隔离，为什么实践里往往"也还行"？

<details><summary>参考答案</summary>

1. π_ref 是 KL 锚点：DPO 由"最大化奖励 + β·KL(π‖π_ref) 约束"推导而来，隐式奖励是相对 π_ref 的对数比。去掉后目标退化为无约束地拉大 chosen/rejected 的绝对 log-prob 差，模型会漂离初始分布、流畅度崩坏（相当于 β·KL 正则消失）。
2. 损失只关心**差值** margin = logp_w − logp_l（相对 π_ref），margin 变大的同时两者绝对值可以都降（把概率质量挪到偏好对之外的序列上）。不算 bug，但过度时伤害生成质量，这是 DPO 已知痛点，衍生出 IPO、DPOP、加 SFT 正则项等修法。
3. 文档边界有 eos 分隔，模型能学会"看到 eos 重置上下文"；且预训练本身就是这么做的，模型已适应。但在小模型/短训练/强格式任务上泄漏影响会放大，所以现代框架（如带 flash-attn 的 varlen/block-diagonal attention）提供隔离选项。
</details>

---
下一站：[模块 09 · GPU 实战（Modal）](09_modal_gpu.md)
