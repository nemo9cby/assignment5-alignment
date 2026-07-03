# 模块 09 · GPU 实战（Modal）：从冒烟测试到端到端 GRPO

## 这一步在地图上的位置

主线测试全绿只说明"数学对了"。这一模块把你的实现放到真 GPU 上，对 **Olmo-2-1B × GSM8K**
完成一次端到端 RLVR：测基线 → 训练 → 再测 → 看到 held-out 准确率真实上涨。
三个关卡各产出一个 JSON artifact 落回 `course/artifacts/`，`check.py 9` 负责验收。

Modal 心智模型（5 分钟）：本地 Python 文件定义 `app` + 容器镜像 + 函数；
`modal run file.py` 时**本地入口在你机器上跑，被 `@app.function(gpu=...)` 装饰的函数在云端容器里跑**，
参数/返回值自动序列化往返。所以"云端算，结果写回本地 JSON"天然成立——这正是我们验证闭环的接口。

前置：`uv run modal setup` 登录过（模块 0）；新账号每月 $30 免费额度，本模块全部关卡花费 < $10。

## 关卡 1 · 冒烟测试（几分钱）

```bash
uv run modal run course/modal/smoke_test.py
```

读一遍 `course/modal/smoke_test.py`（40 行）：镜像怎么声明、函数怎么标 GPU、
入口怎么把云端返回值写进 `course/artifacts/modal_smoke.json`。之后你写任何 Modal 任务都是这个骨架。

## 关卡 2 · zero-shot 基线（< $1）

```bash
uv run modal run course/modal/gsm8k_baseline.py          # 200 题, A10G, 几分钟
```

产出 `gsm8k_baseline.json`。做三件事，别只看一个数：

1. 看 `accuracy` 和 `format_accuracy` 的**差距**——基座模型不守 `<answer>` 格式是常态，
   这正是 RLVR 初期奖励主要来自格式的原因（模块 7 的论点，这里亲眼确认）。
2. 看 `samples` 里的 5 条原始输出，感受失败模式。
3. （选做）`--model-id allenai/OLMo-2-0425-1B-SFT` 再跑一次，对比 SFT 过的模型格式服从性。

## 关卡 3 · 端到端 GRPO（大关卡，$5 左右）

这一关**没有现成脚本**——写训练脚本正是 2026 版作业的主体（handout §GRPO）。
你已有全部零件，装配蓝图：

```
scripts/grpo.py (你来写, 参考你 2025 年的 cs336_alignment/grpo_train.py):
  1. 加载 Olmo-2-1B (HF, bf16, gradient checkpointing 可选)
  2. vLLM 拉起采样引擎 —— 直接用官方 cs336_alignment/vllm_utils.py:
     VLLMServer(server 生命周期) + NCCL 权重同步(训练几步后把新权重推给 vLLM)
  3. 循环 n_grpo_steps 次:
       采一批题 → 每题 group_size 条 rollout (温度 1.0, 训练要探索, 别用 greedy)
       → r1_zero_reward_fn 打分 → 你的 run_grpo_train_step (梯度累积在里面)
       → 每步记录: reward均值/format奖励/熵/clip fraction/grad norm
  4. 训练前后各评一次 held-out (复用关卡 2 的评测逻辑, greedy)
  5. 把结果写进 course/artifacts/grpo_train.json
```

起步超参（handout 推荐区间的保守版，先小后大）：
`group_size=8`，`rollout_batch_size=256`（32 题 × 8），lr `1e-5`（AdamW），
`cliprange=0.2`，50 步起测，单 GPU（A100/H100 80GB 更稳；显存不够就减 batch + 梯度累积）。

提交到 Modal 的两条路：

- **官方路径**：`cs336_alignment/modal_utils.py`（在 `SUNET_ID` 填任意标识即可用）——
  注意它默认 `GPU = "B200:2"` 和 wandb secret，个人账号建议改成 `"A100-80GB"` 并按需去掉 secret；
- **课程路径**：照抄 `gsm8k_baseline.py` 的骨架，把 `evaluate` 换成你的训练函数，训练+评测放同一个远端函数里做完，返回 dict 落地。

artifact 格式（`check.py` 验收 `final_accuracy > baseline_accuracy`）：

```json
{
  "baseline_accuracy": 0.03,
  "final_accuracy": 0.18,
  "n_grpo_steps": 50,
  "model": "allenai/OLMo-2-0425-1B",
  "notes": "group_size=8, lr=1e-5, ..."
}
```

### 训练时盯这四条曲线（出问题的先后顺序）

| 曲线 | 健康形态 | 病态及含义 |
|---|---|---|
| train reward 均值 | 稳步上升 | 骤升到顶: 疑似 reward hacking / 数据泄漏 |
| token 熵 | 缓慢下降 | 断崖下跌: 熵坍缩, 探索死亡(模块 2 自测题) |
| clip fraction | 个位数百分比 | 持续 >20%: 策略每步被拉太远, 降 lr 或减 off-policy 步数 |
| 生成长度 | 平稳/缓变 | 单调暴涨: "sequence" 归一化的长度病(模块 5), 换 constant |

## ✅ 过关验证

```bash
uv run python course/check.py 9
```

三个 artifact 齐、且 `final > baseline` 即通关。**准确率涨幅不设及格线**——50 步能涨几个点就说明
管线全通了；想复现 handout 量级的提升（数十个点），加步数、按 handout 扫超参、跑 4 个种子。

## 自测题

1. 为什么 rollout 采样温度用 1.0，评测却用 greedy？反过来会怎样？
2. 训练进程和 vLLM 各占一份模型权重，还要 NCCL 同步——为什么不直接用训练模型 `model.generate()` 采样，省掉整套 vLLM？
3. 你的 GRPO 是 on-policy 的（每批 rollout 只更新一次）还是 off-policy 的（一批 rollout 更新多个 epoch）？各自 clip 的作用还剩多少？

<details><summary>参考答案</summary>

1. 训练需要组内多样性：温度太低 G 条 rollout 趋同 → 组内方差为 0 → 优势为 0 → 没梯度。评测要可复现的"最优行为"，greedy（或固定低温）消除采样方差，否则准确率带噪声没法比较。反过来：greedy 采样训练直接学不动；高温评测则指标虚且不可复现。
2. HF 的 `generate` 是逐 token 的朴素自回归，没有 PagedAttention/continuous batching，采 256×8 条 rollout 比 vLLM 慢一个数量级以上——RLVR 的时间瓶颈在采样端。代价就是双份权重 + 同步复杂度，这是当代 RL 基建的标准取舍（verl、OpenRLHF 都是这个架构）。
3. 每批只更新一次时接近 on-policy，ratio≈1，clip 几乎不触发（保险丝）；一批数据训多个 epoch/多个微批 step 时是真 off-policy，ratio 漂移，clip 成为主要的稳定机制。意识到自己训练循环处在哪个 regime，才能解释 clip fraction 曲线。
</details>

---
结业站：[AI Lab 面试题自测](interview_questions.md)
