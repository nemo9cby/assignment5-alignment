"""模块 09 关卡 2: 在 Modal GPU 上用 vLLM 评测 zero-shot GSM8K 基线。

对 Olmo-2-1B 用 r1_zero prompt 做 greedy 解码, 用作业自带的
r1_zero_reward_fn 打分, 结果落回 course/artifacts/gsm8k_baseline.json。

用法:
    uv run modal run course/modal/gsm8k_baseline.py                  # 默认 200 题
    uv run modal run course/modal/gsm8k_baseline.py --n 1319         # 全量 test 集
    uv run modal run course/modal/gsm8k_baseline.py --model-id allenai/OLMo-2-0425-1B-SFT

费用参考: A10G 上 200 题约几分钟, 远低于 $1。
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ARTIFACT = REPO_ROOT / "course" / "artifacts" / "gsm8k_baseline.json"

app = modal.App("course-a5-gsm8k-baseline")

# 评测镜像: vllm(拉起推理) + 作业的 grader 及其数学判定依赖
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "vllm==0.19.1",
        "math-verify[antlr4-13-2]>=0.7.0",
        "pylatexenc>=2.10",
    )
    .add_local_dir(REPO_ROOT / "cs336_alignment", "/root/cs336_alignment")
)


@app.function(image=image, gpu="A10G", timeout=3600)
def evaluate(model_id: str, prompts: list[str], ground_truths: list[str]) -> dict:
    from vllm import LLM, SamplingParams

    llm = LLM(model=model_id, gpu_memory_utilization=0.85, max_model_len=2048)
    params = SamplingParams(
        temperature=0.0,  # 评测用 greedy, 保证可复现
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    outputs = llm.generate(prompts, params)

    from cs336_alignment.drgrpo_grader import r1_zero_reward_fn

    n_correct = n_format = 0
    samples = []
    for out, gt in zip(outputs, ground_truths):
        response = out.outputs[0].text
        scores = r1_zero_reward_fn(response, gt)
        n_correct += scores["answer_reward"]
        n_format += scores["format_reward"]
        if len(samples) < 5:
            samples.append({"response": response[-500:], "gt": gt, **scores})

    n = len(prompts)
    return {
        "model": model_id,
        "n_examples": n,
        "accuracy": n_correct / n,
        "format_accuracy": n_format / n,
        "samples": samples,
    }


@app.local_entrypoint()
def main(n: int = 200, model_id: str = "allenai/OLMo-2-0425-1B") -> None:
    prompt_template = (REPO_ROOT / "cs336_alignment" / "prompts" / "r1_zero.prompt").read_text()
    examples = [
        json.loads(line)
        for line in (REPO_ROOT / "data" / "gsm8k" / "test.jsonl").read_text().splitlines()[:n]
    ]
    prompts = [prompt_template.format(question=ex["question"]) for ex in examples]
    # GSM8K 的标准答案在 "#### " 之后
    ground_truths = [ex["answer"].split("####")[-1].strip() for ex in examples]

    result = evaluate.remote(model_id, prompts, ground_truths)

    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(
        f"✅ {model_id} zero-shot GSM8K: accuracy={result['accuracy']:.3f} "
        f"format={result['format_accuracy']:.3f} (n={result['n_examples']})"
    )
    print(f"artifact 已写入 {ARTIFACT}")
