# 模块 01 · Tokenize prompt/response + response mask

## 为什么这一步是地基

RL 微调的损失只作用在 **response 的 token** 上：prompt 是条件，padding 是噪声，都不能进 loss。
这一模块产出三个对齐好的张量——`input_ids`、`labels`、`response_mask`——后面所有模块都建立在
"mask 与 labels 严格对齐"这个约定上。差一位，模块 2–6 的所有数值全错。

## 规格（读 `tests/adapters.py::run_tokenize_prompt_and_output`）

输入：`prompt_strs`、`output_strs`（各 batch 条）、`tokenizer`。
对每条样本，把 prompt 和 output 分别 tokenize 后拼接，然后：

```
拼接序列:   [p1, p2, p3, o1, o2, <pad>, <pad>]          长度 = max(prompt+output 长度)
input_ids:  [p1, p2, p3, o1, o2, <pad>]                 去掉最后一个 token
labels:     [p2, p3, o1, o2, <pad>, <pad>]              去掉第一个 token（左移一位）
response_mask:[0,  0,  1,  1,  0,   0 ]                 与 labels 对齐，response 处为 1
```

关键点：

- **因果 LM 的 shift**：位置 t 的 logits 预测的是 t+1 的 token。所以 `labels = 序列[1:]`，`input_ids = 序列[:-1]`，两者长度都是 `max_len - 1`。
- **mask 对齐 labels 而不是 input_ids**：`response_mask[i][t] == 1` 当且仅当 `labels[i][t]` 是 response 的 token。第一个 response token 出现在 labels 的 `len(prompt) - 1` 位置——自己在纸上推一遍为什么减 1。
- **padding 补右边**，pad 位置 mask 为 0。用 `tokenizer.pad_token_id` 填充。

## 建议的实现步骤

1. 先只处理 batch 中的一条：`prompt_ids = tokenizer.encode(prompt)`，同理 output，注意用 `add_special_tokens=False`（对照测试的玩具 tokenizer 想想为什么）。
2. 手算测试用例：conftest 里 `prompt_strs = output_strs = ["Hello, world!", ...]`，tokenizer 是 15 个词的 word-level 玩具词表（`Hello`→3、`world`→4 …，标点会变成 `<unk>`=2）。把第一条样本的三个张量手写出来。
3. 向量化成 batch：先求 `max_len`，建全 `pad_token_id` 的矩阵再逐条填入，比拼 `torch.nn.utils.rnn.pad_sequence` 更不容易错。
4. 你 2025 版已经写过这个函数（见 `course/reference/adapters_2025_reference.py`），对照检查接口差异后迁移。

## ✅ 过关验证

```bash
uv run python course/check.py 1
# 等价于: uv run pytest tests/test_grpo.py::test_tokenize_prompt_and_output -v
```

## 常见翻车点

- mask 从 `len(prompt)` 开始而不是 `len(prompt) - 1`（忘了 labels 已经左移）。
- 对拼接后的字符串整体 tokenize（`prompt + output` 再 encode）——边界 token 可能与分别 encode 不同，规格要求分别 encode 再拼 id。
- pad 位置的 labels 随便填了别的值：本测试比对 labels 全量数值，pad 处也要是 `pad_token_id`。

## 自测题

1. 为什么 `input_ids` 要把最后一个 token 切掉？留着会怎样？
2. 假如下游用这个 mask 算 SFT loss，把 prompt 位置也设成 1 会发生什么？训练还能收敛吗？

<details><summary>参考答案</summary>

1. 最后一个 token 的 logits 预测的是"序列之后的下一个 token"，而我们没有它的目标（label），算了也用不上；切掉让 `input_ids` 和 `labels` 一一对应，避免下游到处写 `[:-1]`/`[1:]`。
2. loss 会包含"预测 prompt 自身"的项，模型把容量花在建模问题分布上。通常还是能收敛（这就是普通的语言建模），但相当于混入了无监督目标，response 上的有效学习信号被稀释；在 RL 场景更糟——prompt token 的 log-prob 会进入策略梯度，等于优化了一个错误的目标。
</details>

---
下一站：[模块 02 · 条件 log-prob 与 token 熵](02_logprobs_entropy.md)
