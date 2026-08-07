"""Small shared helpers for Chapter 01 GPU labs.

The functions here handle timing, environment capture, simple reference
quantization, and sanitized result serialization. Experiment-specific logic
stays in each notebook.
"""

from __future__ import annotations

import json
import platform
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("This lab requires a CUDA-capable GPU.")
    return torch.device("cuda")


def environment_record() -> dict[str, Any]:
    device = require_cuda()
    props = torch.cuda.get_device_properties(device)
    return {
        "gpu": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "gpu_memory_gib": round(props.total_memory / 2**30, 3),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
    }


def cuda_benchmark(
    fn: Callable[[], Any], *, warmup: int = 5, repeats: int = 20
) -> dict[str, Any]:
    require_cuda()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))
    ordered = sorted(samples)
    p90_index = min(len(ordered) - 1, max(0, int(0.9 * len(ordered)) - 1))
    return {
        "warmup": warmup,
        "repeats": repeats,
        "median_ms": round(statistics.median(samples), 6),
        "p90_ms": round(ordered[p90_index], 6),
        "samples_ms": [round(x, 6) for x in samples],
    }


def symmetric_quantize(
    tensor: torch.Tensor, *, bits: int = 4, group_size: int | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reference symmetric fake quantization along the final dimension."""
    if bits < 2 or bits > 8:
        raise ValueError("bits must be in [2, 8]")
    qmax = 2 ** (bits - 1) - 1
    qmin = -(2 ** (bits - 1))
    original_shape = tensor.shape
    width = original_shape[-1]
    group_size = group_size or width
    if width % group_size:
        raise ValueError("final dimension must be divisible by group_size")
    grouped = tensor.float().reshape(*original_shape[:-1], width // group_size, group_size)
    scale = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / qmax
    quantized = torch.round(grouped / scale).clamp(qmin, qmax).to(torch.int8)
    dequantized = (quantized.float() * scale).reshape(original_shape)
    return quantized.reshape(original_shape), scale.squeeze(-1), dequantized


def error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    ref = reference.float()
    cand = candidate.float()
    diff = cand - ref
    cosine = torch.nn.functional.cosine_similarity(
        ref.reshape(1, -1), cand.reshape(1, -1), dim=1
    ).item()
    return {
        "mae": round(diff.abs().mean().item(), 8),
        "rmse": round(diff.square().mean().sqrt().item(), 8),
        "max_abs": round(diff.abs().max().item(), 8),
        "cosine": round(cosine, 8),
    }


def base_result(lesson: int, evidence_label: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "lesson": lesson,
        "evidence_label": evidence_label,
        "executed_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "environment": environment_record(),
    }


def save_result(result: dict[str, Any], lesson_dir: Path) -> Path:
    artifact_dir = lesson_dir / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / "rtx5090-result.json"
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
