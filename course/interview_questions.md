# 结业 · AI Lab 后训练面试题集

> 全部题目严格出自本课程覆盖的内容（模块号标在题后）。用法：
> **先脱稿口述/手写作答，再开折叠对答案**。面试的及格线不是"知道"，是"能推导、能写、能诊断"。
> 建议两轮：通关主线后做一遍标记盲区，结业前限时（模拟面试，每题 3–10 分钟）再做一遍。

---

## A. 概念与判断（热身，每题 ~3 分钟）

**A1.** SFT、RLHF、RLVR、DPO 各解决什么问题？给一个新任务（比如"让模型写更好的 SQL"），你按什么决策树选方法？（模块 0/8）

<details><summary>参考答案</summary>

SFT：注入格式/行为分布，需要示范数据；RLHF：优化不可程序化验证的偏好（有用性、语气），需要偏好数据 + reward model；RLVR：优化可自动验证的能力（对/错可判），只需 verifier；DPO：RLHF 的轻量替代，直接在偏好对上做监督式优化，省 RM 和 rollout 基建。
决策树：输出能程序化判对错吗（SQL 可以——能执行、能比对结果）→ 能则 RLVR 优先（奖励无噪声、无 RM 被 hack 的空间）；不能 → 有大量偏好数据和基建吗 → 有则 RLHF/在线方法，没有则 DPO 起步；模型连基本格式都不会 → 先 SFT 再谈 RL（冷启动）。
</details>

**A2.** 为什么 2024 年后推理模型（R1 类）的配方是"RLVR 为主、SFT 为辅"而不是继续堆 SFT 数据？（模块 0/3）

<details><summary>参考答案</summary>

SFT 是模仿学习，上限是示范数据的质量与覆盖，且教师分布之外无信号；RLVR 让模型探索自己的解法分布，只要 verifier 判对就强化，能超出示范者水平（发现新解法、更长的推理链）。数学/代码恰好 verifier 便宜可靠。同时 SFT 仍有价值：冷启动格式与基本推理模式，让 RL 初期有非零奖励可爬（否则全 0 奖励组内无方差，没有梯度——GRPO 的冷启动问题）。
</details>

**A3.** GRPO 相比 PPO 去掉了什么、付出了什么代价？什么场景下你反而会选 PPO？（模块 3 自测题的进阶版）

<details><summary>参考答案</summary>

去掉：critic/value network（用组内均值当 baseline）、GAE。代价：每题必须采 G 条（推理开销）、baseline 方差更高、只有 outcome 级信号没有 token 级 credit assignment。选 PPO 的场景：奖励是稠密/过程性的（PRM 逐步打分，value function 能利用）、单 prompt 采多条太贵、或环境是多步交互（agent 任务）状态价值有意义。
</details>

**A4.** 解释 "on-policy" 与 "off-policy" 在 LLM RL 语境下的具体含义。哪些工程决策会把你从 on-policy 推向 off-policy？（模块 4/9）

<details><summary>参考答案</summary>

on-policy：产生 rollout 的策略 == 正在被更新的策略（每批数据只更新一步就重采）。off-policy：rollout 来自旧参数——只要 (a) 一批 rollout 做多个 epoch/多次微批更新，或 (b) 异步基建里采样 worker 的权重滞后于训练进程，就是 off-policy。推向 off-policy 的动机全是效率：采样贵（想榨干每批 rollout）、异步流水线(采样和训练重叠)、大 rollout batch 拆小微批。代价是分布错位，需要重要性采样 + clip 保稳。
</details>

---

## B. 数学推导（白板题，每题 ~8 分钟）

**B1.** 从 `J(θ) = E_{y~π_θ}[R(y)]` 出发推导 REINFORCE 梯度估计，说明 baseline 为什么无偏，并写出 GRPO 的组内优势公式。（模块 3/4）

<details><summary>参考答案</summary>

log-derivative trick：∇J = ∇∫π_θ(y)R(y) = ∫π_θ(y)∇log π_θ(y)R(y) = E[R·∇log π_θ(y)]。
baseline：E[b·∇log π_θ(y)] = b·∇∫π_θ = b·∇1 = 0，故 E[(R−b)·∇log π] 仍无偏，方差在 b≈E[R] 附近最小。
GRPO：对同一 prompt 采 G 条，A_i = (r_i − mean(r_{1..G})) / (std(r_{1..G}) + ε)，序列级优势均摊到每个 response token：loss = −A_i·Σ_t log π_θ(y_t|x, y_<t)。
</details>

**B2.** 写出 PPO clip 目标并解释：A>0 且 ratio > 1+ε 时梯度是什么？A<0 且 ratio < 1−ε 时呢？为什么两侧行为不对称却都是"悲观"的？（模块 4）

<details><summary>参考答案</summary>

L = E[min(ρA, clip(ρ, 1−ε, 1+ε)A)]，ρ = π_θ/π_old。
A>0, ρ>1+ε：min 取剪切项（常数）→ 梯度 0，好动作的强化封顶。A<0, ρ<1−ε：clip 项 = (1−ε)A 大于未剪切项 ρA（都为负时 ρA 更小），min 取**未剪切**项 → 梯度非零，坏动作惩罚不封顶。两侧都取"对目标更保守/更小"的值，即目标的悲观下界——防止一次更新对局部估计过度自信。
</details>

**B3.** GSPO 的序列级重要性比率 `s_i = exp(mean_t Δlogp_t)` 与 GRPO 的 token 级比率相比，方差性质差在哪？推一下长度 |y| 进入方差的方式。（模块 4）

<details><summary>参考答案</summary>

真正的序列比率是 Π_t ρ_t = exp(Σ_t Δlogp_t)：若每 token 的 Δlogp 近独立、方差 σ²，指数上的和方差为 |y|σ²，序列比率是对数正态的，方差随长度**指数级**爆炸——长序列上要么溢出要么归零，clip 形同虚设。GRPO 用 per-token ρ_t 回避了连乘，但代价是每个 token 被独立 clip，与序列级的奖励信号错位（一个 token 被 clip 掉，其他 token 梯度照常，更新方向被"部分裁剪"扭曲）。GSPO 取几何平均：指数上是 mean = Σ/|y|，方差 σ²/|y| 随长度**下降**，s_i 稳定且长度可比，clip 阈值 ε 对长短序列意义一致；整条序列共享一个 s_i，clip 决策也是序列级的，与 outcome 奖励对齐。
</details>

**B4.** 从 "max_π E[r(x,y)] − β·KL(π‖π_ref)" 出发，说明 DPO 如何把 reward model 消掉（关键两步即可），并解释 β 的角色。（模块 8）

<details><summary>参考答案</summary>

步骤 1：该目标的闭式最优解 π*(y|x) ∝ π_ref(y|x)·exp(r(x,y)/β)。反解 r：r(x,y) = β·log(π*(y|x)/π_ref(y|x)) + β·log Z(x)。
步骤 2：把 r 代入 Bradley-Terry 偏好模型 P(y_w ≻ y_l) = σ(r_w − r_l)，配分函数 log Z(x) 相消，得 P = σ(β[log(π/π_ref)(y_w) − log(π/π_ref)(y_l)])。对数据集做 MLE 即 DPO 损失——奖励被"重参数化"成了策略本身。
β：KL 约束强度 = 隐式奖励的尺度。β 大 → 紧贴 π_ref、更新保守；β 小 → margin 主导、容易漂离参考分布损害流畅度。
</details>

**B5.** Dr-GRPO 指出 GRPO 的两个偏差：÷std 和 sequence 级长度归一化。分别用一个具体数值例子说明各自的病理。（模块 3/5）

<details><summary>参考答案</summary>

÷std（难度偏差）：G=8，简单题 7 对 1 错，r=[1×7, 0]，mean=0.875，std≈0.33，错那条 A≈−2.65；中等题 4 对 4 错时 std=0.5，错的 A=−1。同样"答错一条"，接近全对/全错的题给出的梯度反而大得多——策略被推向在极端难度题上过度更新。
长度归一化（长度偏差）：sequence 模式下每条序列除自身长度。两条都答错（A<0）的序列，100 token 的每 token 惩罚是 10 token 的 1/10——"错得越长，单 token 代价越小"，模型学会用啰嗦稀释惩罚；答对时则反过来激励简短。Dr-GRPO 用固定常数 C 归一化消除该耦合。
</details>

---

## C. 编码题（限时手写/上机，每题 10–15 分钟）

**C1.**（上机，10 分钟）实现 `masked_mean(x, mask, dim=None)`：对 mask=1 的位置求均值，支持任意维度。再说明：RL loss 聚合里 `masked_mean` 沿 batch+seq 一起平均 vs 先按序列平均再按 batch 平均，什么时候数值不同？（模块 5）

<details><summary>参考答案</summary>

```python
def masked_mean(x, mask, dim=None):
    mask = mask.to(x.dtype)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1e-8)
```

两者在各序列 response 长度不等时不同：整体平均按 token 数加权（长序列权重大），逐序列平均给每条序列等权。这正是 constant vs sequence 归一化之争的最小版本。
</details>

**C2.**（上机，15 分钟）给定 `logits (B,T,V)` 和 `labels (B,T)`，写出数值稳定的 per-token log-prob 和熵。禁止显式 for 循环。（模块 2）

<details><summary>参考答案</summary>

```python
logp_all = F.log_softmax(logits, dim=-1)                       # log-sum-exp 内置, 稳定
logp = logp_all.gather(-1, labels.unsqueeze(-1)).squeeze(-1)    # (B,T)
entropy = -(logp_all.exp() * logp_all).sum(-1)                 # 或 logsumexp - Σp·z
```

考点：不写 `log(softmax())`；gather 的 dim/unsqueeze；熵沿 vocab 维求和。
</details>

**C3.**（找 bug，8 分钟）下面的 GRPO 优势计算有三个 bug，找出来：

```python
def compute_advantages(rewards, group_size, eps=1e-6):
    groups = rewards.view(group_size, -1)          # (G, n_prompts)
    mean = groups.mean(dim=0)
    std = groups.std(dim=0)
    advantages = (groups - mean) / (std + eps)
    return advantages                               # 直接返回
```

<details><summary>参考答案</summary>

1. reshape 错误：rollout 的排布是"同一题的 G 条相邻"，应 `view(-1, group_size)`（n_prompts, G）再沿 dim=-1 统计；`view(group_size, -1)` 把不同题的回答混进同一组（且不等价于转置）。
2. 广播错位：`groups - mean` 需要 `keepdim=True`（或按修正后的形状沿 dim=-1, keepdim=True），否则在错的维度上广播。
3. 没有摊平回 `(rollout_batch_size,)`，且顺序必须与输入一致（`view(-1)` 于 (n_prompts, G) 布局恰好还原）。
   追问点：`torch.std` 默认无偏（correction=1），G 小时和有偏差异明显，实现要和参考约定一致。
</details>

**C4.**（设计 + 编码，15 分钟）写出一个 GRPO microbatch 训练步的伪代码，要求支持梯度累积和 grad clip，并指出"哪一行除以了 accumulation steps、为什么"。（模块 6）

<details><summary>参考答案</summary>

见模块 6 装配清单。关键行：`(microbatch_loss / n_accum).backward()`——backward 做的是梯度**累加**，除以 N 后累加起来才等于全批平均梯度；不除则有效学习率放大 N 倍，且改变 clip 阈值的语义。clip 必须在全部微批 backward 之后、step 之前做一次（对累积后的总梯度限幅）。
</details>

---

## D. 调试与实战诊断（情景题，每题 ~8 分钟）

**D1.** RLVR 训练到 200 步：train reward 从 0.2 涨到 0.75，held-out 准确率纹丝不动。给出排查顺序。（模块 7/9）

<details><summary>参考答案</summary>

按便宜到贵排查：(1) 肉眼读 20 条 rollout——reward hacking / verifier 漏洞（格式骗分、答案枚举）？(2) 格式分 vs 答案分分解——涨的是不是全是 format_reward？(3) 熵曲线——熵坍缩后模型只是反复采自己会的题（train 分布收窄，能力没变）；(4) train/held-out 是否泄漏或分布差异过大；(5) 评测端 bug：解析器/解码参数不一致（训练温度 1.0 评测也 1.0 导致噪声淹没提升）。
</details>

**D2.** 训练 loss 突然 NaN。列出 LLM RL 里最常见的四个来源和对应检查。（模块 2/4/6）

<details><summary>参考答案</summary>

(1) ratio 爆炸：exp(logp − old_logp) 中差值过大（off-policy 太狠/权重同步 bug）——查 Δlogp 的 max，clip 前先 clamp log-ratio（如 ±20）；(2) 熵/log 计算的 0·(−inf)：查是否 `log(softmax)` 或 p·log p 裸算；(3) std 除零：整组同奖励时 (r−mean)/(std+eps)=0 没事，但自己写的归一化若 eps 加错位置（如 `std(r+eps)`）会出 NaN；(4) fp16/bf16 溢出（logits 上限）或 grad 爆炸——查 grad norm 曲线、clip 是否生效、lr 是否过大。通用手段：`torch.autograd.set_detect_anomaly(True)` 定位第一个 NaN 的 op。
</details>

**D3.** clip fraction 从 5% 爬到 40%，同时 KL(π‖π_old) 增大、reward 开始震荡。发生了什么？给三个处置选项及各自代价。（模块 4/9）

<details><summary>参考答案</summary>

诊断：策略每个 rollout batch 内被更新得太远（有效步长过大），大量 token 落在 clip 区，梯度信号被裁剪扭曲，优化在"跳来跳去"。处置：(1) 降 lr——最直接，代价是训练变慢；(2) 减少每批 rollout 的更新次数/epoch（更 on-policy）——稳定，但采样成本占比上升；(3) 增大 rollout batch 或 group size——梯度方差降低，同样多花采样钱；(附加) 收紧 ε 治标，可能加剧梯度稀疏。
</details>

**D4.** 你把 `loss_normalization` 从 `"sequence"` 换成 `"constant"` 后，同样的 lr 下训练发散了。为什么这不奇怪？该怎么公平对比两种归一化？（模块 5）

<details><summary>参考答案</summary>

两种归一化的 loss 尺度不同：sequence 模式每 token 梯度 ~1/(B·|y_i|)，constant 模式 ~1/(B·C)。若实际生成长度 |y| ≫ C 或 ≪ C，梯度尺度差好几倍，等价于换了学习率。公平对比要么按平均长度重标 lr（lr' = lr·C/E[|y|] 量级），要么对两种设置各自扫 lr 取最优再比最终指标——论文里比较归一化方案不控制有效步长的结论都要打折扣。
</details>

---

## E. 系统与开放设计（senior 向，每题 ~15 分钟）

**E1.** 设计一个单机 8×H100 的 RLVR 训练系统（1B–8B 模型）：画出组件图，说明权重同步方案、采样与训练的时间重叠策略、以及三处最可能的瓶颈。（模块 9）

<details><summary>参考答案</summary>

组件：训练进程（FSDP/DDP 若干卡）+ vLLM 推理引擎（其余卡）+ verifier 池（CPU 并行，数学判定用超时保护——sympy 会卡死）+ 数据/日志。权重同步：训练若干步后广播新权重到 vLLM（NCCL 直传显存，如本仓库 vllm_utils 的做法；或 checkpoint+reload，慢但简单）。重叠：异步流水线——vLLM 用第 k 版权重采下一批时，训练进程消化上一批（代价：off-policy 滞后一步，需要 IS+clip 兜底）。瓶颈：(1) 采样吞吐（通常最大头，长 CoT 时更甚）→ 卡数分配要偏向推理；(2) 权重同步的停顿（大模型逐层广播、vLLM 显存重排）；(3) verifier 延迟长尾（个别样本 sympy 卡 30s——必须超时+进程池隔离）。加分：提到 GPU 分配比例随生成长度动态调整、rollout 结果队列化。
</details>

**E2.** 老板说："给我们的 code agent 上 RLVR。"奖励怎么设计？列出你会防的三种 reward hacking 和对应机制。（模块 3/7）

<details><summary>参考答案</summary>

奖励：单测通过率为主信号（可执行验证），叠加格式/可编译门槛（不过直接 0），惩罚项慎加。防 hack：(1) 模型直接改/删测试或硬编码期望输出 → 测试文件只读、沙箱隔离、用模型看不见的 held-out 测试打分；(2) 过拟合可见测试（针对具体断言写 if-else）→ 隐藏测试集 + 变异测试（mutation testing）抽查；(3) 退化解（空 diff、try/except 吞错误骗"不崩溃"分）→ 奖励要求净通过数提升而不是"无失败"，diff 规模下限/静态检查。通用：定期人工抽读高奖励样本——所有自动指标都会被 Goodhart。
</details>

**E3.** 只有 5000 条人工偏好对预算，产品要"回答更有帮助"。DPO 直训、训 RM 再 RLHF、还是先做数据增广？给出你的方案与理由，并说明如何验证收益是真的。（模块 7/8）

<details><summary>参考答案</summary>

推荐：DPO（或 IPO/变体）直训 + 增广。理由：5000 对训 RM 容易过拟合，RM 的误差在 RL 中会被策略主动放大（对着 RM 的弱点优化）；DPO 把同样的数据直接花在策略上，基建成本低一个数量级。增广：用现有强模型对同 prompt 生成回答对，人工只标注/校验（标注比撰写便宜数倍）；或 RLAIF 造弱标签、人工对齐校准子集。验证：held-out 偏好对上的 win rate + 盲测 A/B（人评，防 LLM-judge 偏置：位置随机化、长度控制——DPO 常把"更长"当"更好"学进去）+ 通用能力回归测试（MMLU/GSM8K 不掉点），三者都过才算真收益。
</details>

**E4.**（结合你的实现讲）从这门课的 26 个快照测试出发：如果让你为团队的 RL 训练库设计测试体系，快照测试够吗？你会加哪几层？（全课）

<details><summary>参考答案</summary>

快照测试锁定"数值与参考实现一致"，抓回归极好，但有盲区：参考实现本身的错它测不出、超参组合覆盖有限、分布式/异步路径没覆盖。加层：(1) 性质测试（property-based）：优势的组内均值≈0、mask 外梯度必须为 0、ratio=1 时 grpo==noclip==none 的梯度一致性（模块 4 自测题 1 直接变成测试）；(2) 梯度检查：有限差分 vs autograd；(3) 端到端小模型收敛测试：tiny 模型 + 可作弊的玩具任务，N 步内 reward 必须过阈值（抓"数值对但学不动"的 bug）；(4) 分布式一致性：单卡 vs 多卡 vs 梯度累积的参数逐位比对；(5) 性能回归：采样吞吐/显存水位报警。
</details>

---

## 评分参考（自测用）

- **junior 及格线**：A 全对；B1、B2 能推；C 全部限时完成；D 至少给出合理排查顺序。
- **senior 及格线**：以上全部 + B3–B5 能推 + D 每题给出机制级解释 + E 至少两题有完整方案并能被追问三层。
- 任何一题卡壳 → 回对应模块重做，那就是你的下一个训练点。不用沮丧：**发现盲区正是这套题存在的意义**。
