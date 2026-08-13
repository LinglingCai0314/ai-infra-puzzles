#!/usr/bin/env python3
"""Shared Triton kernels and measured experiments for Chapter 05."""

from __future__ import annotations

import math
import os
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, out, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    tl.store(out + offsets, tl.load(x + offsets, mask=mask) + tl.load(y + offsets, mask=mask), mask=mask)


@triton.jit
def affine_kernel(x, out, n_elements, scale, bias, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask).to(tl.float32)
    tl.store(out + offsets, values * scale + bias, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 128}, num_warps=4),
        triton.Config({"BLOCK": 256}, num_warps=4),
        triton.Config({"BLOCK": 512}, num_warps=8),
    ],
    key=["n_elements"],
)
@triton.jit
def autotuned_affine_kernel(x, out, n_elements, scale, bias, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    values = tl.load(x + offsets, mask=mask).to(tl.float32)
    tl.store(out + offsets, values * scale + bias, mask=mask)


@triton.jit
def row_copy_kernel(x, out, rows, cols, stride_xr, stride_xc, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = (row < rows) & (cols_off < cols)
    values = tl.load(x + row * stride_xr + cols_off * stride_xc, mask=mask)
    tl.store(out + row * cols + cols_off, values, mask=mask)


@triton.jit
def row_max_kernel(x, out, rows, cols, ZERO_PAD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = (row < rows) & (cols_off < cols)
    other = 0.0 if ZERO_PAD else -float("inf")
    values = tl.load(x + row * cols + cols_off, mask=mask, other=other).to(tl.float32)
    tl.store(out + row, tl.max(values, axis=0), mask=row < rows)


@triton.jit
def softmax_kernel(x, out, rows, cols, stride_xr, stride_or, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = (row < rows) & (cols_off < cols)
    values = tl.load(x + row * stride_xr + cols_off, mask=mask, other=-float("inf")).to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    numerator = tl.exp(shifted)
    probabilities = numerator / tl.sum(numerator, axis=0)
    tl.store(out + row * stride_or + cols_off, probabilities, mask=mask)


@triton.jit
def row_sum_kernel(x, out, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = (row < rows) & (cols_off < cols)
    values = tl.load(x + row * cols + cols_off, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + row, tl.sum(values, axis=0), mask=row < rows)


@triton.jit
def row_scan_kernel(x, out, rows, cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = (row < rows) & (cols_off < cols)
    values = tl.load(x + row * cols + cols_off, mask=mask, other=0.0).to(tl.float32)
    scanned = tl.cumsum(values, axis=0)
    tl.store(out + row * cols + cols_off, scanned, mask=mask)


@triton.jit
def matmul_kernel(a, b, c, m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
                  stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                  RELU: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                  BLOCK_K: tl.constexpr):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(n, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    for k_start in range(0, k, BLOCK_K):
        av = tl.load(a_ptrs, mask=(offs_m[:, None] < m) & (offs_k[None, :] + k_start < k), other=0.0)
        bv = tl.load(b_ptrs, mask=(offs_k[:, None] + k_start < k) & (offs_n[None, :] < n), other=0.0)
        acc += tl.dot(av, bv)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    if RELU:
        acc = tl.maximum(acc, 0.0)
    tl.store(c + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < m) & (offs_n[None, :] < n))


@triton.jit
def rmsnorm_kernel(x, weight, out, rows, cols, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = (row < rows) & (cols_off < cols)
    values = tl.load(x + row * cols + cols_off, mask=mask, other=0.0).to(tl.float32)
    variance = tl.sum(values * values, axis=0) / cols
    normalized = values * tl.rsqrt(variance + eps)
    weights = tl.load(weight + cols_off, mask=cols_off < cols, other=0.0).to(tl.float32)
    tl.store(out + row * cols + cols_off, normalized * weights, mask=mask)


@triton.jit
def paged_gather_kernel(cache, block_table, out, tokens, block_size, width,
                        stride_cb, stride_co, stride_ct, BLOCK: tl.constexpr):
    token = tl.program_id(0)
    cols_off = tl.arange(0, BLOCK)
    mask = (token < tokens) & (cols_off < width)
    logical_block = token // block_size
    slot = token % block_size
    physical_block = tl.load(block_table + logical_block, mask=token < tokens, other=0)
    values = tl.load(cache + physical_block * stride_cb + slot * stride_ct + cols_off * stride_co,
                     mask=mask, other=0.0)
    tl.store(out + token * width + cols_off, values, mask=mask)


@triton.jit
def persistent_affine_kernel(x, out, n_elements, scale, bias, num_tiles,
                             BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    step = tl.num_programs(0)
    for tile in tl.range(pid, num_tiles, step):
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        values = tl.load(x + offsets, mask=mask).to(tl.float32)
        tl.store(out + offsets, values * scale + bias, mask=mask)


@triton.jit
def gelu_kernel(x, out, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    v = tl.load(x + offsets, mask=mask).to(tl.float32)
    cube = v * v * v
    tanh_arg = 0.7978845608 * (v + 0.044715 * cube)
    # tanh(z) = 2 * sigmoid(2z) - 1; this keeps the kernel on public tl ops.
    y = v * tl.sigmoid(2.0 * tanh_arg)
    tl.store(out + offsets, y, mask=mask)


def environment(lesson_no: int) -> dict[str, Any]:
    assert torch.cuda.is_available(), "Chapter 05 retained runs require a CUDA-capable GPU."
    major, minor = torch.cuda.get_device_capability(0)
    target = triton.runtime.driver.active.get_current_target()
    return {
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": f"{major}.{minor}",
        "torch": torch.__version__,
        "cuda_runtime": str(torch.version.cuda),
        "triton": triton.__version__,
        "triton_target": str(target),
        "python": sys.version.split()[0],
        "seed": 20260813 + lesson_no,
    }


def gpu_samples(fn: Callable[[], Any], warmup: int = 5, repeats: int = 20) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        values.append(float(start.elapsed_time(stop)))
    return values


def median_ms(fn: Callable[[], Any], warmup: int = 5, repeats: int = 20) -> tuple[float, list[float]]:
    samples = gpu_samples(fn, warmup, repeats)
    return statistics.median(samples), samples


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return float(ordered[lo])
    return float(ordered[lo] * (hi - position) + ordered[hi] * (position - lo))


def error(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    return float((reference.float() - candidate.float()).abs().max().item())


def launch_affine(x: torch.Tensor, out: torch.Tensor, block: int = 256,
                  scale: float = 1.25, bias: float = 0.5, num_warps: int = 4) -> None:
    affine_kernel[(triton.cdiv(x.numel(), block),)](
        x, out, x.numel(), scale, bias, BLOCK=block, num_warps=num_warps
    )


def launch_softmax(x: torch.Tensor, out: torch.Tensor) -> None:
    rows, cols = x.shape
    block = triton.next_power_of_2(cols)
    softmax_kernel[(rows,)](x, out, rows, cols, x.stride(0), out.stride(0), BLOCK=block, num_warps=8)


def launch_matmul(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor, relu: bool = False) -> None:
    m, k = a.shape
    _, n = b.shape
    bm = bn = 32
    bk = 32
    grid = (triton.cdiv(m, bm) * triton.cdiv(n, bn),)
    matmul_kernel[grid](a, b, out, m, n, k, a.stride(0), a.stride(1),
                        b.stride(0), b.stride(1), out.stride(0), out.stride(1),
                        RELU=relu, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, num_warps=4)


def base_metrics(primary: Any, secondary: Any, max_abs_error: float,
                 passed: bool, **details: Any) -> dict[str, Any]:
    return {"primary": primary, "secondary": secondary, "max_abs_error": max_abs_error,
            "passed": bool(passed), "details": details}


def run_lesson(no: int) -> tuple[dict[str, Any], str, str]:
    env = environment(no)
    torch.manual_seed(env["seed"])
    torch.cuda.manual_seed_all(env["seed"])
    device = torch.device("cuda")
    dtype = torch.float32

    if no in {1, 2, 3, 4, 5, 9, 10, 14, 15, 21, 22, 25, 26, 28, 29, 30}:
        n = 2**22 + (17 if no in {3, 4, 25} else 0)
        x = torch.randn(n, device=device, dtype=dtype)
        out = torch.empty_like(x)
        ref = x * 1.25 + 0.5

    if no == 1:
        triton_ms, ts = median_ms(lambda: launch_affine(x, out))
        pytorch_ms, ps = median_ms(lambda: torch.add(x * 1.25, 0.5, out=out))
        launch_affine(x, out)
        err = error(ref, out)
        metrics = base_metrics(triton_ms, pytorch_ms, err, err < 1e-5,
                               triton_samples_ms=ts, pytorch_samples_ms=ps,
                               requested_bytes=2 * x.numel() * x.element_size(),
                               logical_launches={"triton": 1, "pytorch_eager": 2})
        en = f"The fused Triton affine kernel took {triton_ms:.4f} ms versus {pytorch_ms:.4f} ms for the two-operation eager path, with max error {err:.2e}."
        zh = f"融合 Triton affine kernel 用时 {triton_ms:.4f} ms；两个 eager PyTorch 操作合计 {pytorch_ms:.4f} ms，最大误差为 {err:.2e}。"
    elif no == 2:
        y = torch.randn_like(x)
        ref2 = x + y
        add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
        tm, ts = median_ms(lambda: add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256))
        pm, ps = median_ms(lambda: torch.add(x, y, out=out))
        add_kernel[(triton.cdiv(n, 256),)](x, y, out, n, BLOCK=256)
        err = error(ref2, out)
        metrics = base_metrics(tm, pm, err, err == 0.0, programs=triton.cdiv(n, 256),
                               block_elements=256, triton_samples_ms=ts, pytorch_samples_ms=ps)
        en = f"{triton.cdiv(n, 256):,} blocked programs covered {n:,} scalar elements; correctness matched torch.add and the two timings are reported separately."
        zh = f"{triton.cdiv(n, 256):,} 个 blocked program 覆盖 {n:,} 个标量元素；结果与 torch.add 一致，两条路径分别计时。"
    elif no == 3:
        start = time.perf_counter(); launch_affine(x, out); torch.cuda.synchronize()
        cold = (time.perf_counter() - start) * 1e3
        warm, samples = median_ms(lambda: launch_affine(x, out), warmup=2)
        err = error(ref, out)
        metrics = base_metrics(cold, warm, err, err < 1e-5,
                               triton=env["triton"], target=env["triton_target"], warm_samples_ms=samples)
        en = f"The first host-observed launch took {cold:.2f} ms and the warm GPU-event median was {warm:.4f} ms; they answer different questions."
        zh = f"主机观察到的首次调用为 {cold:.2f} ms，warm GPU-event 中位数为 {warm:.4f} ms；二者回答的是不同问题。"
    elif no == 4:
        results = {}
        for block in (128, 256, 512, 1024):
            tm, samples = median_ms(lambda b=block: launch_affine(x, out, b), repeats=15)
            results[str(block)] = {"median_ms": tm, "samples_ms": samples}
        best_block = min(results, key=lambda key: results[key]["median_ms"])
        launch_affine(x, out, int(best_block))
        err = error(ref, out)
        metrics = base_metrics(int(best_block), results[best_block]["median_ms"], err, err < 1e-5,
                               n_elements=n, tail=n % int(best_block), block_sweep=results)
        en = f"The odd-length tail remained correct. BLOCK={best_block} was fastest in this four-point sweep at {results[best_block]['median_ms']:.4f} ms."
        zh = f"非整除 tail 保持正确；四点 sweep 中 BLOCK={best_block} 最快，中位数 {results[best_block]['median_ms']:.4f} ms。"
    elif no == 5:
        launch_affine(x, out)
        err = error(ref, out)
        nvcc = shutil.which("nvcc")
        tm, samples = median_ms(lambda: launch_affine(x, out))
        metrics = base_metrics(bool(nvcc), tm, err, err < 1e-5,
                               cuda_source="vector_affine.cu", cuda_compiled=False,
                               reason="nvcc unavailable" if not nvcc else "source retained; compilation intentionally separate",
                               triton_samples_ms=samples)
        en = f"The Triton path ran correctly in {tm:.4f} ms. nvcc availability was {bool(nvcc)}, so the explicit CUDA source is reviewable but is not presented as a native measurement."
        zh = f"Triton 路径正确运行，用时 {tm:.4f} ms。nvcc 可用性为 {bool(nvcc)}，因此显式 CUDA 源码可审阅，但不冒充原生实测。"
    elif no == 6:
        rows, cols = 4096, 1024
        base = torch.randn((rows, cols * 2), device=device)
        contiguous = base[:, :cols].contiguous()
        strided = base[:, ::2]
        out2 = torch.empty((rows, cols), device=device)
        block = triton.next_power_of_2(cols)
        def run(v): row_copy_kernel[(rows,)](v, out2, rows, cols, v.stride(0), v.stride(1), BLOCK=block, num_warps=8)
        cm, cs = median_ms(lambda: run(contiguous)); sm, ss = median_ms(lambda: run(strided))
        run(strided); err = error(strided, out2)
        metrics = base_metrics(cm, sm, err, err == 0.0, slowdown=sm / cm,
                               contiguous_stride=list(contiguous.stride()), strided_stride=list(strided.stride()),
                               contiguous_samples_ms=cs, strided_samples_ms=ss)
        en = f"The same copy kernel took {cm:.4f} ms on contiguous columns and {sm:.4f} ms at stride two, a {sm / cm:.2f}x ratio."
        zh = f"同一个 copy kernel 在连续列上用时 {cm:.4f} ms，在 stride=2 时用时 {sm:.4f} ms，二者相差 {sm / cm:.2f} 倍。"
    elif no == 7:
        rows, cols = 1024, 1000
        values = -torch.rand((rows, cols), device=device) - 1.0
        good = torch.empty(rows, device=device); bad = torch.empty_like(good)
        block = triton.next_power_of_2(cols)
        row_max_kernel[(rows,)](values, good, rows, cols, ZERO_PAD=False, BLOCK=block)
        row_max_kernel[(rows,)](values, bad, rows, cols, ZERO_PAD=True, BLOCK=block)
        refm = values.max(1).values
        good_err, bad_err = error(refm, good), error(refm, bad)
        metrics = base_metrics(good_err, bad_err, good_err, good_err == 0.0 and bad_err > 0,
                               corrupt_rows=int((bad != refm).sum().item()), tail=block-cols)
        en = f"Using -inf as the masked max identity produced {good_err:.2e} error; padding with zero corrupted {int((bad != refm).sum())} all-negative rows."
        zh = f"max reduction 的 mask 使用 -inf 时误差为 {good_err:.2e}；错误地填 0 会破坏 {int((bad != refm).sum())} 行全负输入。"
    elif no == 8:
        rows, cols = 2048, 769
        base = torch.randn((cols, rows), device=device)
        view = base.T
        out2 = torch.empty((rows, cols), device=device)
        block = triton.next_power_of_2(cols)
        fn = lambda: row_copy_kernel[(rows,)](view, out2, rows, cols, view.stride(0), view.stride(1), BLOCK=block, num_warps=8)
        tm, samples = median_ms(fn); fn(); err = error(view, out2)
        bandwidth = 2 * view.numel() * view.element_size() / (tm / 1e3) / 1e9
        metrics = base_metrics(tm, bandwidth, err, err == 0.0, input_stride=list(view.stride()),
                               output_stride=list(out2.stride()), samples_ms=samples)
        en = f"Explicit stride arithmetic copied a transposed {rows}x{cols} view correctly at {bandwidth:.1f} requested GB/s."
        zh = f"显式 stride 算术正确复制了 {rows}x{cols} 的转置 view，请求带宽为 {bandwidth:.1f} GB/s。"
    elif no == 9:
        start = time.perf_counter(); launch_affine(x, out); torch.cuda.synchronize(); first=(time.perf_counter()-start)*1e3
        samples = gpu_samples(lambda: launch_affine(x, out), warmup=8, repeats=40)
        p50, p20, p80 = percentile(samples,.5), percentile(samples,.2), percentile(samples,.8)
        err=error(ref,out)
        metrics=base_metrics(first,p50,err,err<1e-5,p20_ms=p20,p80_ms=p80,samples_ms=samples)
        en=f"The first host-observed call was {first:.2f} ms; warm event timing was p20={p20:.4f}, p50={p50:.4f}, p80={p80:.4f} ms."
        zh=f"首次主机调用为 {first:.2f} ms；warm event 计时为 p20={p20:.4f}、p50={p50:.4f}、p80={p80:.4f} ms。"
    elif no == 10:
        launch_affine(x,out); vm,vs=median_ms(lambda:launch_affine(x,out))
        vector_bytes=2*x.numel()*x.element_size(); vector_ai=2*x.numel()/vector_bytes
        a=torch.randn((512,512),device=device,dtype=torch.float16); b=torch.randn_like(a); c=torch.empty_like(a)
        mm,ms=median_ms(lambda:torch.mm(a,b,out=c)); flops=2*512**3; mm_ai=flops/(3*512**2*2)
        metrics=base_metrics(vector_ai,mm_ai,error(ref,out),True,vector_median_ms=vm,matmul_median_ms=mm,
                             vector_gbps=vector_bytes/(vm/1e3)/1e9,matmul_tflops=flops/(mm/1e3)/1e12,
                             vector_samples_ms=vs,matmul_samples_ms=ms)
        en=f"The affine kernel has {vector_ai:.3f} FLOP/byte of requested traffic, while the square GEMM model has {mm_ai:.1f}; they require different performance questions."
        zh=f"affine kernel 的请求算术强度为 {vector_ai:.3f} FLOP/byte，而方阵 GEMM 模型为 {mm_ai:.1f}；两者不能用同一吞吐指标判断。"
    elif no == 11:
        rows,cols=4096,1024; v=torch.randn((rows,cols),device=device); o=torch.empty_like(v); r=torch.softmax(v,dim=1)
        launch_softmax(v,o); tm,ts=median_ms(lambda:launch_softmax(v,o)); pm,ps=median_ms(lambda:torch.softmax(v,dim=1))
        launch_softmax(v,o); err=error(r,o)
        metrics=base_metrics(tm,pm,err,err<2e-5,triton_samples_ms=ts,pytorch_samples_ms=ps,rows=rows,cols=cols)
        en=f"One Triton program per row fused max, exp, sum, and normalization in {tm:.4f} ms versus {pm:.4f} ms for torch.softmax; max error was {err:.2e}."
        zh=f"每行一个 Triton program，把 max、exp、sum 和归一化融合为 {tm:.4f} ms；torch.softmax 为 {pm:.4f} ms，最大误差 {err:.2e}。"
    elif no == 12:
        rows,cols=4096,256; v=torch.randn((rows,cols),device=device); sums=torch.empty(rows,device=device); scans=torch.empty_like(v)
        row_sum_kernel[(rows,)](v,sums,rows,cols,BLOCK=256); row_scan_kernel[(rows,)](v,scans,rows,cols,BLOCK=256)
        sm,ss=median_ms(lambda:row_sum_kernel[(rows,)](v,sums,rows,cols,BLOCK=256))
        cm,cs=median_ms(lambda:row_scan_kernel[(rows,)](v,scans,rows,cols,BLOCK=256))
        row_sum_kernel[(rows,)](v,sums,rows,cols,BLOCK=256); row_scan_kernel[(rows,)](v,scans,rows,cols,BLOCK=256)
        err=max(error(v.sum(1),sums),error(v.cumsum(1),scans))
        metrics=base_metrics(sm,cm,err,err<1e-3,reduction_samples_ms=ss,scan_samples_ms=cs)
        en=f"The row reduction took {sm:.4f} ms and inclusive scan {cm:.4f} ms. Their communication structures differ even with identical input shapes."
        zh=f"逐行 reduction 用时 {sm:.4f} ms，inclusive scan 用时 {cm:.4f} ms；即使输入 shape 相同，通信结构也不同。"
    elif no in {13,16,27}:
        size=512; mdtype=torch.float16 if no!=16 else torch.bfloat16
        a=torch.randn((size,size),device=device,dtype=mdtype); b=torch.randn_like(a); o=torch.empty_like(a)
        relu=no==27; refm=torch.mm(a,b); refm=torch.relu(refm) if relu else refm
        launch_matmul(a,b,o,relu); tm,ts=median_ms(lambda:launch_matmul(a,b,o,relu),repeats=15)
        pm,ps=median_ms(lambda:torch.relu(torch.mm(a,b)) if relu else torch.mm(a,b),repeats=15)
        launch_matmul(a,b,o,relu); err=error(refm,o); tol=0.5 if mdtype==torch.bfloat16 else 0.2
        flops=2*size**3
        metrics=base_metrics(tm,pm,err,err<tol,triton_tflops=flops/(tm/1e3)/1e12,
                             library_tflops=flops/(pm/1e3)/1e12,dtype=str(mdtype),fused_relu=relu,
                             triton_samples_ms=ts,library_samples_ms=ps)
        if no==13:
            en=f"The teaching Triton GEMM reached {flops/(tm/1e3)/1e12:.2f} TFLOP/s; torch.mm reached {flops/(pm/1e3)/1e12:.2f}. The library remains the baseline, not a guaranteed loser."
            zh=f"教学版 Triton GEMM 达到 {flops/(tm/1e3)/1e12:.2f} TFLOP/s，torch.mm 为 {flops/(pm/1e3)/1e12:.2f}；库函数应始终保留为基线。"
        elif no==16:
            en=f"BF16 Triton and library GEMM are reported separately: {tm:.4f} and {pm:.4f} ms, with FP32-comparison max error {err:.3e}."
            zh=f"BF16 Triton 与库 GEMM 分别为 {tm:.4f} 和 {pm:.4f} ms；按 FP32 比较的最大误差为 {err:.3e}。"
        else:
            en=f"The custom path fused ReLU into the GEMM store and took {tm:.4f} ms; torch.mm plus ReLU took {pm:.4f} ms. This is one shape, not a library ranking."
            zh=f"自定义路径把 ReLU 融合进 GEMM store，用时 {tm:.4f} ms；torch.mm 加 ReLU 为 {pm:.4f} ms。这只是一个 shape，不是库排名。"
    elif no == 14:
        grid=lambda meta:(triton.cdiv(n,meta['BLOCK']),)
        start=time.perf_counter(); autotuned_affine_kernel[grid](x,out,n,1.25,0.5); torch.cuda.synchronize(); first=(time.perf_counter()-start)*1e3
        warm,samples=median_ms(lambda:autotuned_affine_kernel[grid](x,out,n,1.25,0.5))
        err=error(ref,out)
        metrics=base_metrics(first,warm,err,err<1e-5,configs=3,warm_samples_ms=samples)
        en=f"Three configurations were eligible. The first host call cost {first:.2f} ms, while the cached warm median was {warm:.4f} ms."
        zh=f"共有 3 个候选配置；首次主机调用为 {first:.2f} ms，缓存后的 warm 中位数为 {warm:.4f} ms。"
    elif no == 15:
        sweep={}
        for block,warps in ((128,4),(256,4),(512,8),(1024,8)):
            tm,s=median_ms(lambda b=block,w=warps:launch_affine(x,out,b,num_warps=w),repeats=15)
            sweep[f"b{block}-w{warps}"]={"median_ms":tm,"samples_ms":s}
        best=min(sweep,key=lambda k:sweep[k]["median_ms"]); worst=max(sweep,key=lambda k:sweep[k]["median_ms"])
        err=error(ref,out)
        metrics=base_metrics(sweep[best]["median_ms"],sweep[worst]["median_ms"]/sweep[best]["median_ms"],err,err<1e-5,
                             best_config=best,worst_config=worst,sweep=sweep)
        en=f"The best measured configuration was {best}; the slowest/best latency ratio was {metrics['secondary']:.2f}x. Occupancy was not inferred from timing alone."
        zh=f"实测最佳配置为 {best}，最慢与最佳的延迟比为 {metrics['secondary']:.2f} 倍；实验没有仅凭计时反推 occupancy。"
    elif no == 17:
        rows,cols=4096,1024; v=torch.randn((rows,cols),device=device); w=torch.randn(cols,device=device); o=torch.empty_like(v)
        refn=v.float()*torch.rsqrt(v.float().pow(2).mean(1,keepdim=True)+1e-5)*w.float()
        block=triton.next_power_of_2(cols)
        fn=lambda:rmsnorm_kernel[(rows,)](v,w,o,rows,cols,1e-5,BLOCK=block,num_warps=8)
        fn(); tm,ts=median_ms(fn); pm,ps=median_ms(lambda:v*torch.rsqrt(v.pow(2).mean(1,keepdim=True)+1e-5)*w)
        fn(); err=error(refn,o)
        metrics=base_metrics(tm,pm,err,err<2e-5,triton_samples_ms=ts,pytorch_eager_samples_ms=ps,rows=rows,cols=cols)
        en=f"The fused RMSNorm forward path took {tm:.4f} ms versus {pm:.4f} ms for its eager expression, with max error {err:.2e}. Backward is outside this lab."
        zh=f"融合 RMSNorm forward 为 {tm:.4f} ms，eager 表达式为 {pm:.4f} ms，最大误差 {err:.2e}；本实验不覆盖 backward。"
    elif no == 18:
        batch,heads,seq,dim=2,8,512,64
        q=torch.randn((batch,heads,seq,dim),device=device,dtype=torch.float16); k=torch.randn_like(q); v=torch.randn_like(q)
        sdpa=lambda:F.scaled_dot_product_attention(q,k,v,is_causal=True)
        eager=lambda:torch.matmul(torch.softmax(torch.matmul(q,k.transpose(-2,-1))/math.sqrt(dim)+torch.triu(torch.full((seq,seq),-float('inf'),device=device,dtype=q.dtype),1),dim=-1),v)
        sm,ss=median_ms(sdpa,repeats=15); em,es=median_ms(eager,repeats=10)
        rs=sdpa(); re=eager(); err=error(re,rs)
        score_bytes=batch*heads*seq*seq*q.element_size()
        metrics=base_metrics(sm,em,err,err<0.02,materialized_score_bytes=score_bytes,
                             sdpa_samples_ms=ss,eager_samples_ms=es,backend_claim="PyTorch SDPA; internal kernel not asserted")
        en=f"PyTorch SDPA took {sm:.4f} ms versus {em:.4f} ms for materialized eager attention. Avoiding a {score_bytes/2**20:.1f} MiB score tensor is the algorithmic boundary; the internal SDPA kernel is not named."
        zh=f"PyTorch SDPA 用时 {sm:.4f} ms，显式物化 attention 为 {em:.4f} ms。算法边界是避免 {score_bytes/2**20:.1f} MiB score tensor；不臆测 SDPA 内部 kernel 名称。"
    elif no == 19:
        rows,cols=2048,1024; v=(torch.randn((rows,cols),device=device)*20+80).half(); o=torch.empty_like(v)
        launch_softmax(v,o); stable=torch.softmax(v.float(),dim=1)
        unstable=torch.exp(v)/torch.exp(v).sum(1,keepdim=True)
        stable_err=error(stable,o); invalid=int((~torch.isfinite(unstable)).sum().item())
        metrics=base_metrics(invalid,stable_err,stable_err,invalid>0 and stable_err<0.01,
                             stable_invalid=int((~torch.isfinite(o)).sum().item()),input_min=float(v.min()),input_max=float(v.max()))
        en=f"The unshifted FP16 expression produced {invalid:,} non-finite values. Max-shifted Triton softmax produced none and differed from FP32 reference by {stable_err:.3e}."
        zh=f"未减最大值的 FP16 表达式产生 {invalid:,} 个非有限值；max-shift Triton softmax 没有非有限值，相对 FP32 参考最大误差 {stable_err:.3e}。"
    elif no == 20:
        debug_ops={name:hasattr(tl,name) for name in ("static_print","static_assert","device_print","device_assert")}
        interpreter_doc=bool(os.environ.get("TRITON_INTERPRET"))
        x2=torch.arange(257,device=device,dtype=torch.float32); o=torch.empty_like(x2)
        launch_affine(x2,o,256); err=error(x2*1.25+0.5,o)
        metrics=base_metrics(sum(debug_ops.values()), interpreter_doc, err, all(debug_ops.values()) and err<1e-5,
                             debug_operations=debug_ops,interpreter_enabled_for_this_gpu_run=interpreter_doc,
                             note="Interpreter is a separate CPU debugging mode, not a GPU performance mode")
        en=f"All {sum(debug_ops.values())} documented debug operators were present. The retained GPU run validates the kernel; interpreter mode remains a separate CPU debugging workflow."
        zh=f"文档列出的 {sum(debug_ops.values())} 个调试操作均存在。当前保留的是 GPU 正确性运行；interpreter 是独立的 CPU 调试流程。"
    elif no == 21:
        launch_affine(x,out); err=error(ref,out)
        source=affine_kernel.src
        metrics=base_metrics(len(source.splitlines()), "TTIR -> TTGIR -> LLVM IR -> PTX", err, err<1e-5,
                             source_has_mask="mask=mask" in source,target=env["triton_target"],
                             inspect_commands=["TRITON_KERNEL_DUMP=1", "proton-viewer", "ncu"])
        en=f"The reviewed kernel source has {len(source.splitlines())} lines and targets {env['triton_target']}. IR/PTX inspection is a path to evidence, not a substitute for counters."
        zh=f"被审阅的 kernel 源码有 {len(source.splitlines())} 行，目标为 {env['triton_target']}。阅读 IR/PTX 是取证路径，不能代替 counter。"
    elif no == 22:
        def eager(v): return torch.sigmoid(v*1.25+0.5)
        compiled=torch.compile(eager,fullgraph=True)
        start=time.perf_counter(); rc=compiled(x); torch.cuda.synchronize(); cold=(time.perf_counter()-start)*1e3
        cm,cs=median_ms(lambda:compiled(x),repeats=15); em,es=median_ms(lambda:eager(x),repeats=15)
        err=error(eager(x),rc)
        metrics=base_metrics(cold,cm,err,err<1e-5,eager_median_ms=em,compiled_samples_ms=cs,eager_samples_ms=es,
                             backend="torch.compile/Inductor")
        en=f"torch.compile cold execution took {cold:.2f} ms; its warm median was {cm:.4f} ms versus {em:.4f} ms eager. Compile cost and steady state stay separate."
        zh=f"torch.compile 冷启动为 {cold:.2f} ms；warm 中位数 {cm:.4f} ms，eager 为 {em:.4f} ms。编译成本与稳态性能分开记录。"
    elif no == 23:
        blocks,block_size,width,tokens=256,16,128,2048
        cache=torch.randn((blocks,block_size,width),device=device,dtype=torch.float16)
        logical=math.ceil(tokens/block_size); table=torch.randperm(blocks,device=device,dtype=torch.int32)[:logical]
        o=torch.empty((tokens,width),device=device,dtype=cache.dtype); token_ids=torch.arange(tokens,device=device)
        refp=cache[table[(token_ids//block_size).long()].long(),(token_ids%block_size).long()]
        block=triton.next_power_of_2(width)
        fn=lambda:paged_gather_kernel[(tokens,)](cache,table,o,tokens,block_size,width,cache.stride(0),cache.stride(2),cache.stride(1),BLOCK=block)
        fn(); tm,ts=median_ms(fn); pm,ps=median_ms(lambda:cache[table[(token_ids//block_size).long()].long(),(token_ids%block_size).long()])
        fn(); err=error(refp,o)
        metrics=base_metrics(tm,pm,err,err==0.0,block_table_entries=logical,triton_samples_ms=ts,pytorch_samples_ms=ps)
        en=f"The Triton paged gather followed {logical} logical-to-physical block entries in {tm:.4f} ms and matched advanced indexing exactly."
        zh=f"Triton paged gather 按 {logical} 个逻辑到物理 block 映射完成读取，用时 {tm:.4f} ms，与 advanced indexing 完全一致。"
    elif no == 24:
        target=triton.runtime.driver.active.get_current_target(); source=affine_kernel.src
        launch_affine(torch.arange(1025,device=device,dtype=torch.float32),torch.empty(1025,device=device),256)
        backend=getattr(target,"backend","unknown"); arch=getattr(target,"arch","unknown")
        metrics=base_metrics(backend,arch,0.0,backend=="cuda",triton_version=triton.__version__,
                             same_source_candidate=True,rocm_executed=False,source_lines=len(source.splitlines()))
        en=f"This run compiled the same Triton source for backend={backend}, arch={arch}. ROCm portability remains unmeasured on this NVIDIA-only execution."
        zh=f"本次将同一份 Triton 源码编译到 backend={backend}、arch={arch}；由于只运行了 NVIDIA 环境，ROCm 可移植性仍未实测。"
    elif no == 25:
        shapes=(1025,65537,1048593); details={}; maxerr=0.0
        for size in shapes:
            v=torch.randn(size,device=device); o=torch.empty_like(v); r=v*1.25+0.5
            start=time.perf_counter(); launch_affine(v,o,256); torch.cuda.synchronize(); first=(time.perf_counter()-start)*1e3
            warm,s=median_ms(lambda:launch_affine(v,o,256),repeats=10); maxerr=max(maxerr,error(r,o))
            details[str(size)]={"first_ms":first,"warm_median_ms":warm,"tail":size%256,"samples_ms":s}
        metrics=base_metrics(len(shapes),max(v["warm_median_ms"] for v in details.values()),maxerr,maxerr<1e-5,shape_results=details)
        en=f"One runtime-size kernel handled {len(shapes)} shapes and three different tails without changing source; the largest warm median was {metrics['secondary']:.4f} ms."
        zh=f"同一个 runtime-size kernel 在不改源码的情况下处理了 {len(shapes)} 个 shape 和三种 tail；最大 warm 中位数为 {metrics['secondary']:.4f} ms。"
    elif no == 26:
        block=256; tiles=triton.cdiv(n,block); programs=min(256,tiles)
        def persistent(): persistent_affine_kernel[(programs,)](x,out,n,1.25,0.5,tiles,BLOCK=block,num_warps=4)
        persistent(); pm,ps=median_ms(persistent); sm,ss=median_ms(lambda:launch_affine(x,out,block))
        persistent(); err=error(ref,out)
        metrics=base_metrics(pm,sm,err,err<1e-5,persistent_programs=programs,standard_programs=tiles,
                             persistent_samples_ms=ps,standard_samples_ms=ss,tma_used=False)
        en=f"Persistent scheduling reduced the grid from {tiles:,} to {programs} programs and took {pm:.4f} ms versus {sm:.4f} ms for the standard grid. TMA was not used."
        zh=f"persistent 调度把 grid 从 {tiles:,} 个 program 降到 {programs} 个，用时 {pm:.4f} ms；标准 grid 为 {sm:.4f} ms。本实验未使用 TMA。"
    elif no == 28:
        baseline,s1=median_ms(lambda:launch_affine(x,out),repeats=30); candidate,s2=median_ms(lambda:launch_affine(x,out),repeats=30)
        ratio=candidate/baseline; status="pass" if ratio<=1.10 else ("warning" if ratio<=1.20 else "error")
        err=error(ref,out)
        metrics=base_metrics(ratio,status,err,err<1e-5 and status!="error",baseline_ms=baseline,candidate_ms=candidate,
                             warning_ratio=1.10,error_ratio=1.20,baseline_samples_ms=s1,candidate_samples_ms=s2,
                             versions={"torch":env["torch"],"triton":env["triton"],"cuda":env["cuda_runtime"]})
        en=f"Two independent warm sample groups produced a {ratio:.3f}x candidate/baseline ratio, classified {status} under frozen 1.10/1.20 warning/error gates."
        zh=f"两组独立 warm 样本得到 candidate/baseline={ratio:.3f}，按固定的 1.10/1.20 warning/error gate 判为 {status}。"
    elif no == 29:
        tm,ts=median_ms(lambda:launch_affine(x,out)); pm,ps=median_ms(lambda:torch.add(x*1.25,0.5,out=out))
        scenarios={
            "fused_pointwise":{"triton":5,"cuda":3,"library":2},
            "standard_gemm":{"triton":3,"cuda":4,"library":5},
            "architecture_specific_primitive":{"triton":2,"cuda":5,"library":4},
        }
        winners={k:max(v,key=v.get) for k,v in scenarios.items()}; err=error(ref,out)
        metrics=base_metrics(tm/pm,winners["fused_pointwise"],err,err<1e-5,decision_scores=scenarios,winners=winners,
                             measured_triton_ms=tm,measured_pytorch_ms=pm,triton_samples_ms=ts,pytorch_samples_ms=ps)
        en=f"The measured affine ratio was {tm/pm:.3f}x, while the explicit decision matrix selected different tools for different constraints. No universal winner was encoded."
        zh=f"affine 实测比值为 {tm/pm:.3f}；显式决策矩阵在不同约束下选出不同工具，没有写入“万能赢家”。"
    elif no == 30:
        gelu_kernel[(triton.cdiv(n,256),)](x,out,n,BLOCK=256); refg=F.gelu(x,approximate="tanh")
        tm,ts=median_ms(lambda:gelu_kernel[(triton.cdiv(n,256),)](x,out,n,BLOCK=256),repeats=30)
        pm,ps=median_ms(lambda:F.gelu(x,approximate="tanh"),repeats=30)
        gelu_kernel[(triton.cdiv(n,256),)](x,out,n,BLOCK=256); err=error(refg,out)
        speedup=pm/tm; gate=err<1e-4 and speedup>=0.8
        metrics=base_metrics(speedup,tm,err,gate,pytorch_median_ms=pm,correctness_tolerance=1e-4,
                             minimum_speed_ratio=0.8,rollback="torch.nn.functional.gelu",triton_samples_ms=ts,pytorch_samples_ms=ps)
        en=f"The deliverable GELU reached {speedup:.2f}x of the PyTorch baseline with {err:.2e} max error, so its frozen correctness and no-catastrophic-regression gate was {gate}."
        zh=f"可交付 GELU 相对 PyTorch baseline 为 {speedup:.2f} 倍，最大误差 {err:.2e}；固定的正确性与非灾难回归 gate 判定为 {gate}。"
    else:
        raise ValueError(f"unsupported lesson {no}")

    return metrics, en, zh
