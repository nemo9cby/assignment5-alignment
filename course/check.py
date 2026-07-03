#!/usr/bin/env python3
"""交互式课程进度检查器。

用法:
    uv run python course/check.py          # 总进度面板
    uv run python course/check.py 3        # 详细运行模块 3 的关卡 (pytest -v)
    uv run python course/check.py --list   # 列出每个模块的关卡

设计: 每个模块对应若干"关卡"(gate)。关卡有三种:
  - collect:  测试套件能被 pytest 完整收集(环境 OK 的信号)
  - tests:    指定的快照测试通过
  - artifact: course/artifacts/ 下存在 Modal 任务落回的 JSON(GPU 关卡)
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT / "course" / "artifacts"

GRPO = "tests/test_grpo.py"


@dataclass
class Module:
    num: int
    title: str
    doc: str
    tests: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    collect_expected: int = 0  # >0 表示这是"环境"关卡
    optional: bool = False


MODULES: list[Module] = [
    Module(0, "环境搭建 + 全局地图", "course/00_setup.md", collect_expected=26),
    Module(1, "Tokenize prompt/response + mask", "course/01_tokenize_and_mask.md",
           tests=[f"{GRPO}::test_tokenize_prompt_and_output"]),
    Module(2, "条件 log-prob 与 token 熵", "course/02_logprobs_entropy.md",
           tests=[f"{GRPO}::test_get_response_log_probs"]),
    Module(3, "可验证奖励 + 组内优势", "course/03_rewards_advantages.md",
           tests=[
               f"{GRPO}::test_compute_rollout_rewards",
               f"{GRPO}::test_compute_group_normalized_rewards_grpo",
               f"{GRPO}::test_compute_group_normalized_rewards_drgrpo",
               f"{GRPO}::test_compute_group_normalized_rewards_maxrl",
           ]),
    Module(4, "策略梯度损失 (PG → clip → GSPO)", "course/04_policy_gradient.md",
           tests=[
               f"{GRPO}::test_compute_policy_gradient_loss_on_policy",
               f"{GRPO}::test_compute_policy_gradient_loss_off_policy",
               f"{GRPO}::test_compute_policy_gradient_loss_off_policy_gspo",
           ]),
    Module(5, "微批聚合与损失归一化", "course/05_loss_aggregation.md",
           tests=[
               f"{GRPO}::test_aggregate_loss_across_microbatch_sequence",
               f"{GRPO}::test_aggregate_loss_across_microbatch_constant",
           ]),
    Module(6, "完整 GRPO 训练步", "course/06_grpo_train_step.md",
           tests=[
               f"{GRPO}::test_grpo_train_step_standard_on_policy",
               f"{GRPO}::test_grpo_train_step_variants_on_policy[grpo_constant]",
               f"{GRPO}::test_grpo_train_step_variants_on_policy[dr_grpo]",
               f"{GRPO}::test_grpo_train_step_variants_on_policy[rft]",
               f"{GRPO}::test_grpo_train_step_variants_on_policy[maxrl]",
               f"{GRPO}::test_grpo_train_step_off_policy[noclip]",
               f"{GRPO}::test_grpo_train_step_off_policy[grpo]",
               f"{GRPO}::test_grpo_train_step_off_policy[gspo]",
           ]),
    Module(7, "评测与答案解析", "course/07_eval_and_parsing.md",
           tests=[
               "tests/test_metrics.py::test_parse_mmlu_response",
               "tests/test_metrics.py::test_parse_mmlu_response_unknown",
               "tests/test_metrics.py::test_parse_gsm8k_response",
               "tests/test_metrics.py::test_parse_gsm8k_response_unknown",
           ]),
    Module(8, "支线: SFT 数据打包 + DPO", "course/08_sft_dpo.md", optional=True,
           tests=[
               "tests/test_data.py::test_packed_sft_dataset",
               "tests/test_data.py::test_iterate_batches",
               "tests/test_dpo.py::test_per_instance_dpo_loss",
           ]),
    Module(9, "GPU 实战 (Modal)", "course/09_modal_gpu.md",
           artifacts=["modal_smoke.json", "gsm8k_baseline.json", "grpo_train.json"]),
]


def run_pytest_collect() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "--collect-only", "-q",
         "-p", "no:cacheprovider"],
        cwd=ROOT, capture_output=True, text=True,
    )
    n = sum(1 for line in proc.stdout.splitlines() if "::" in line)
    ok = proc.returncode in (0, 1) and n >= 26
    return ok, f"收集到 {n} 个测试" + ("" if ok else "(应 ≥ 26, 环境或语法有问题)")


def run_pytest_gates(node_ids: list[str]) -> dict[str, bool]:
    """一次 pytest 跑完所有关卡测试, 用 junitxml 拿到每个测试的结果。"""
    if not node_ids:
        return {}
    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        xml_path = f.name
    subprocess.run(
        [sys.executable, "-m", "pytest", *node_ids, "-q", "--tb=no",
         "-p", "no:cacheprovider", f"--junitxml={xml_path}"],
        cwd=ROOT, capture_output=True, text=True,
    )
    results: dict[str, bool] = {}
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError:
        return {nid: False for nid in node_ids}
    for case in tree.iter("testcase"):
        name = case.get("name", "")
        failed = any(child.tag in ("failure", "error") for child in case)
        skipped = any(child.tag == "skipped" for child in case)
        results[name] = not failed and not skipped
    # junitxml 的 name 不含文件路径, 按测试名(含参数)匹配回 node id
    out = {}
    for nid in node_ids:
        test_name = nid.split("::", 1)[1]
        out[nid] = results.get(test_name, False)
    return out


def check_artifact(name: str) -> tuple[bool, str]:
    path = ARTIFACTS_DIR / name
    if not path.exists():
        return False, f"{name} 不存在(见模块 9 讲义)"
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False, f"{name} 不是合法 JSON"
    if name == "modal_smoke.json":
        gpu = data.get("gpu_name")
        return bool(gpu), f"GPU: {gpu}" if gpu else "缺少 gpu_name 字段"
    if name == "gsm8k_baseline.json":
        acc = data.get("accuracy")
        n = data.get("n_examples", "?")
        if acc is None:
            return False, "缺少 accuracy 字段"
        return True, f"zero-shot 基线 accuracy={acc:.3f} (n={n})"
    if name == "grpo_train.json":
        base, final = data.get("baseline_accuracy"), data.get("final_accuracy")
        if base is None or final is None:
            return False, "需要 baseline_accuracy 和 final_accuracy 字段"
        improved = final > base
        msg = f"accuracy {base:.3f} → {final:.3f}"
        return improved, msg + ("" if improved else "(尚未超过基线)")
    return True, "存在"


BAR_W = 24


def render(modules: list[Module]) -> int:
    all_tests = [t for m in modules for t in m.tests]
    print("正在运行关卡测试(首次约 1–2 分钟, 需要加载 tiny 模型)...\n")
    test_results = run_pytest_gates(all_tests)
    collect_ok, collect_msg = run_pytest_collect()

    total_gates = passed_gates = 0
    first_open: Module | None = None
    print("═" * 64)
    print("  后训练快速入门 · 进度面板")
    print("═" * 64)
    for m in modules:
        if m.collect_expected:
            gates = [(collect_ok, collect_msg)]
        elif m.artifacts:
            gates = [check_artifact(a) for a in m.artifacts]
        else:
            gates = [(test_results.get(t, False), t.split("::", 1)[1]) for t in m.tests]
        n_pass = sum(1 for ok, _ in gates if ok)
        n = len(gates)
        done = n_pass == n
        icon = "✅" if done else ("🟡" if n_pass else "🔴")
        opt = "(支线)" if m.optional else ""
        print(f" {icon} {m.num:02d} {m.title}{opt}  [{n_pass}/{n}]")
        if not done:
            for ok, desc in gates:
                if not ok:
                    print(f"      ✗ {desc}")
            if first_open is None and not m.optional:
                first_open = m
        if not m.optional:
            total_gates += n
            passed_gates += n_pass
    print("─" * 64)
    frac = passed_gates / total_gates if total_gates else 0
    filled = round(frac * BAR_W)
    print(f" 主线进度 |{'█' * filled}{'░' * (BAR_W - filled)}| {passed_gates}/{total_gates} gates")
    if first_open is None:
        print("\n 🎉 主线全部通关！去 course/interview_questions.md 做结业自测吧。")
    else:
        print(f"\n 👉 下一步: 读 {first_open.doc}")
        print(f"    过关验证: uv run python course/check.py {first_open.num}")
    print("═" * 64)
    return 0 if first_open is None else 1


def run_module_verbose(m: Module) -> int:
    print(f"—— 模块 {m.num:02d}: {m.title} ——  讲义: {m.doc}\n")
    if m.collect_expected:
        ok, msg = run_pytest_collect()
        print(("✅ " if ok else "🔴 ") + msg)
        return 0 if ok else 1
    if m.artifacts:
        code = 0
        for a in m.artifacts:
            ok, msg = check_artifact(a)
            print(("✅ " if ok else "🔴 ") + f"{a}: {msg}")
            code |= 0 if ok else 1
        return code
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *m.tests, "-v", "-x",
         "-p", "no:cacheprovider"],
        cwd=ROOT,
    )
    return proc.returncode


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--list":
        for m in MODULES:
            print(f"{m.num:02d} {m.title} -> {m.doc}")
            for t in m.tests:
                print(f"    {t}")
            for a in m.artifacts:
                print(f"    artifact: course/artifacts/{a}")
        return 0
    if args:
        try:
            num = int(args[0])
            module = next(m for m in MODULES if m.num == num)
        except (ValueError, StopIteration):
            print(f"未知模块: {args[0]} (可选 0–{MODULES[-1].num})")
            return 2
        return run_module_verbose(module)
    return render(MODULES)


if __name__ == "__main__":
    sys.exit(main())
