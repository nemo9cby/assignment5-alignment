"""模块 09 关卡 1: Modal 冒烟测试。

在 Modal 上申请一块 GPU, 跑一次真实的 CUDA 矩阵乘, 把结果落回本地
course/artifacts/modal_smoke.json —— check.py 靠它判定过关。

用法:
    uv run modal run course/modal/smoke_test.py
    uv run modal run course/modal/smoke_test.py --gpu L40S   # 换 GPU 型号
"""

from __future__ import annotations

import json
from pathlib import Path

import modal

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "modal_smoke.json"

app = modal.App("course-a5-smoke-test")

image = modal.Image.debian_slim(python_version="3.12").pip_install("torch")


@app.function(image=image, gpu="T4", timeout=600)
def gpu_check() -> dict:
    import torch

    assert torch.cuda.is_available(), "container 里没有可用 GPU"
    device = torch.device("cuda")
    a = torch.randn(1024, 1024, device=device)
    b = torch.randn(1024, 1024, device=device)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    c = a @ b
    end.record()
    torch.cuda.synchronize()
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "matmul_checksum": float(c.sum().item()),
        "matmul_ms": float(start.elapsed_time(end)),
    }


@app.local_entrypoint()
def main() -> None:
    result = gpu_check.remote()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    ARTIFACT.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"✅ GPU: {result['gpu_name']}, matmul {result['matmul_ms']:.2f} ms")
    print(f"artifact 已写入 {ARTIFACT}")
    print("运行 `uv run python course/check.py 9` 查看关卡状态")
