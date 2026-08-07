#!/usr/bin/env python3
"""Self-contained CUDA experiment cells for Chapter 02 notebooks."""

from __future__ import annotations


ENV_CODE = r'''
from pathlib import Path
import copy, gzip, hashlib, importlib.util, io, json, math, random, shutil, statistics, sys
import torch
import torch.nn as nn
import torch.nn.functional as F

assert torch.cuda.is_available(), "This lab requires a CUDA-capable GPU."
DEVICE = torch.device("cuda")
SEED = 20260808 + LESSON_NO
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cuda.matmul.allow_tf32 = False

gpu_name = torch.cuda.get_device_name(0)
major, minor = torch.cuda.get_device_capability(0)
ENV = {
    "gpu": gpu_name,
    "compute_capability": f"{major}.{minor}",
    "torch": torch.__version__,
    "cuda_runtime": str(torch.version.cuda),
    "python": sys.version.split()[0],
    "seed": SEED,
}
print(json.dumps(ENV, indent=2))

def percentile(values, q):
    ordered = sorted(float(v) for v in values)
    if not ordered:
        return float("nan")
    position = (len(ordered) - 1) * q
    lo, hi = math.floor(position), math.ceil(position)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] * (hi - position) + ordered[hi] * (position - lo)

def cuda_times(fn, warmup=6, repeats=24):
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        samples = []
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            samples.append(float(start.elapsed_time(end)))
    return samples

def timing_summary(samples):
    return {
        "median_ms": float(statistics.median(samples)),
        "p95_ms": float(percentile(samples, 0.95)),
        "p99_ms": float(percentile(samples, 0.99)),
        "samples_ms": [float(x) for x in samples],
    }

def count_params(module):
    return int(sum(p.numel() for p in module.parameters()))

def zero_fraction(tensor):
    return float((tensor == 0).float().mean().item())

def magnitude_mask(tensor, sparsity):
    flat = tensor.detach().abs().flatten()
    prune_count = int(round(flat.numel() * float(sparsity)))
    prune_count = min(max(prune_count, 0), flat.numel())
    mask = torch.ones_like(flat)
    if prune_count:
        idx = torch.topk(flat, prune_count, largest=False).indices
        mask[idx] = 0
    return mask.view_as(tensor)

def exact_2_4_mask(weight):
    assert weight.shape[-1] % 4 == 0
    groups = weight.detach().abs().reshape(*weight.shape[:-1], -1, 4)
    keep = torch.topk(groups, 2, dim=-1, largest=True).indices
    mask = torch.zeros_like(groups)
    mask.scatter_(-1, keep, 1)
    return mask.reshape_as(weight)

def compliance_2_4(weight):
    groups = weight.detach().reshape(*weight.shape[:-1], -1, 4)
    return float(((groups != 0).sum(dim=-1) == 2).float().mean().item())

def tensor_metrics(reference, candidate):
    ref = reference.float()
    cand = candidate.float()
    delta = cand - ref
    return {
        "rmse": float(torch.sqrt(torch.mean(delta.square())).item()),
        "mae": float(torch.mean(delta.abs()).item()),
        "max_error": float(delta.abs().max().item()),
        "cosine": float(F.cosine_similarity(ref.flatten(), cand.flatten(), dim=0).item()),
    }

def spearman(a, b):
    a = torch.as_tensor(a, dtype=torch.float64)
    b = torch.as_tensor(b, dtype=torch.float64)
    ra = torch.empty_like(a)
    rb = torch.empty_like(b)
    ra[torch.argsort(a)] = torch.arange(a.numel(), dtype=torch.float64)
    rb[torch.argsort(b)] = torch.arange(b.numel(), dtype=torch.float64)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb / (ra.norm() * rb.norm() + 1e-12)).item())
'''.strip()


EXPERIMENTS: dict[int, str] = {
    1: r'''
dtype = torch.bfloat16
batch, in_features, out_features = 32, 2048, 2048
x = torch.randn(batch, in_features, device=DEVICE, dtype=dtype)
dense = nn.Linear(in_features, out_features, bias=False, device=DEVICE, dtype=dtype).eval()
masked = copy.deepcopy(dense)
mask = magnitude_mask(masked.weight, 0.50)
with torch.no_grad():
    masked.weight.mul_(mask)
keep = torch.arange(out_features // 2, device=DEVICE)
narrow = nn.Linear(in_features, keep.numel(), bias=False, device=DEVICE, dtype=dtype).eval()
with torch.no_grad():
    narrow.weight.copy_(dense.weight[keep])

dense_t = timing_summary(cuda_times(lambda: dense(x)))
masked_t = timing_summary(cuda_times(lambda: masked(x)))
narrow_t = timing_summary(cuda_times(lambda: narrow(x)))
metrics = {
    "dense_parameters": count_params(dense),
    "masked_parameters": count_params(masked),
    "masked_sparsity": zero_fraction(masked.weight),
    "narrow_parameters": count_params(narrow),
    "dense_output_width": out_features,
    "narrow_output_width": int(keep.numel()),
    "dense_median_ms": dense_t["median_ms"],
    "dense_p95_ms": dense_t["p95_ms"],
    "masked_median_ms": masked_t["median_ms"],
    "masked_p95_ms": masked_t["p95_ms"],
    "narrow_median_ms": narrow_t["median_ms"],
    "narrow_p95_ms": narrow_t["p95_ms"],
    "samples": {"dense": dense_t["samples_ms"], "masked": masked_t["samples_ms"], "narrow": narrow_t["samples_ms"]},
}
analysis = (
    f"The mask created {metrics['masked_sparsity']:.1%} logical sparsity but kept "
    f"{metrics['masked_parameters']:,} dense parameters and a {out_features}-wide output. "
    f"Its median was {metrics['masked_median_ms']:.6f} ms versus {metrics['dense_median_ms']:.6f} ms. "
    f"The physical candidate reduced parameters to {metrics['narrow_parameters']:,}, output width to "
    f"{keep.numel()}, and measured {metrics['narrow_median_ms']:.6f} ms. These numbers answer this CUDA "
    "operator workload only; they do not establish mobile first-frame latency."
)
''',
    2: r'''
dtype = torch.bfloat16
rows, cols, batch = 2048, 2048, 32
w = torch.randn(rows, cols, device=DEVICE, dtype=dtype)
x = torch.randn(batch, cols, device=DEVICE, dtype=dtype)
dense = w.clone()
unstructured = w * magnitude_mask(w, 0.50)
blocks = w.reshape(rows // 16, 16, cols // 16, 16).clone()
block_selector = torch.arange((rows // 16) * (cols // 16), device=DEVICE).reshape(rows // 16, cols // 16) % 2
blocks = blocks * block_selector[:, None, :, None]
block_sparse = blocks.reshape_as(w)
nm = w * exact_2_4_mask(w)
narrow = w[: rows // 2]

td = timing_summary(cuda_times(lambda: F.linear(x, dense)))
tu = timing_summary(cuda_times(lambda: F.linear(x, unstructured)))
tb = timing_summary(cuda_times(lambda: F.linear(x, block_sparse)))
tnm = timing_summary(cuda_times(lambda: F.linear(x, nm)))
tn = timing_summary(cuda_times(lambda: F.linear(x, narrow)))
metrics = {
    "unstructured_sparsity": zero_fraction(unstructured),
    "block_sparsity": zero_fraction(block_sparse),
    "nm_sparsity": zero_fraction(nm),
    "unstructured_2_4_compliance": compliance_2_4(unstructured),
    "block_2_4_compliance": compliance_2_4(block_sparse),
    "nm_compliance": compliance_2_4(nm),
    "dense_median_ms": td["median_ms"],
    "unstructured_median_ms": tu["median_ms"],
    "block_median_ms": tb["median_ms"],
    "nm_dense_path_median_ms": tnm["median_ms"],
    "narrow_median_ms": tn["median_ms"],
    "narrow_shape": list(narrow.shape),
}
analysis = (
    f"All three masks were near 50% sparse, but exact 2:4 compliance was "
    f"{metrics['nm_compliance']:.1%} versus {metrics['unstructured_2_4_compliance']:.1%} for the "
    f"unstructured mask. The ordinary dense path measured {metrics['dense_median_ms']:.6f} ms for dense, "
    f"{metrics['unstructured_median_ms']:.6f} ms for unstructured, and "
    f"{metrics['nm_dense_path_median_ms']:.6f} ms for compliant values. Only the narrow control changed "
    "the matrix shape; no sparse-kernel dispatch is inferred from these timings."
)
''',
    3: r'''
dtype = torch.bfloat16
model = nn.Sequential(
    nn.Linear(1024, 2048, device=DEVICE, dtype=dtype), nn.GELU(),
    nn.Linear(2048, 1024, device=DEVICE, dtype=dtype), nn.GELU(),
    nn.Linear(1024, 256, device=DEVICE, dtype=dtype),
).eval()

def linear_flops(module, batch):
    return int(sum(2 * batch * m.in_features * m.out_features for m in module.modules() if isinstance(m, nn.Linear)))

def run_batch(batch):
    inp = torch.randn(batch, 1024, device=DEVICE, dtype=dtype)
    torch.cuda.reset_peak_memory_stats()
    times = timing_summary(cuda_times(lambda: model(inp), warmup=8, repeats=40))
    peak = torch.cuda.max_memory_allocated() / 2**20
    return inp, times, peak

x1, t1, peak1 = run_batch(1)
x64, t64, peak64 = run_batch(64)
metrics = {
    "parameters": count_params(model),
    "batch1_flops": linear_flops(model, 1),
    "batch64_flops": linear_flops(model, 64),
    "batch1_median_ms": t1["median_ms"],
    "batch1_p95_ms": t1["p95_ms"],
    "batch1_examples_s": 1000.0 / t1["median_ms"],
    "batch64_median_ms": t64["median_ms"],
    "batch64_p95_ms": t64["p95_ms"],
    "batch64_examples_s": 64000.0 / t64["median_ms"],
    "peak_memory_mib": max(peak1, peak64),
    "samples": {"batch1": t1["samples_ms"], "batch64": t64["samples_ms"]},
}
analysis = (
    f"The frozen MLP contains {metrics['parameters']:,} parameters and {metrics['batch1_flops']:,} "
    f"leading linear FLOPs at batch 1. Batch 1 measured median/p95 "
    f"{metrics['batch1_median_ms']:.6f}/{metrics['batch1_p95_ms']:.6f} ms, while batch 64 measured "
    f"{metrics['batch64_median_ms']:.6f} ms and {metrics['batch64_examples_s']:.1f} examples/s. "
    "The batch field therefore changes the meaning of the performance baseline even though the parameters are identical."
)
''',
    4: r'''
torch.manual_seed(SEED)
n, d, classes = 1536, 32, 4
x = torch.randn(n, d, device=DEVICE)
teacher = torch.randn(d, classes, device=DEVICE)
y = (x @ teacher + 0.2 * torch.randn(n, classes, device=DEVICE)).argmax(1)
train_x, val_x = x[:1200], x[1200:]
train_y, val_y = y[:1200], y[1200:]

def make_model():
    return nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, classes)).to(DEVICE)

def accuracy(model):
    model.eval()
    with torch.inference_mode():
        return float((model(val_x).argmax(1) == val_y).float().mean().item())

def train_steps(model, steps, masks=None, lr=0.03):
    model.train(); opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.8)
    for step in range(steps):
        idx = torch.arange(step * 96, step * 96 + 96, device=DEVICE) % train_x.shape[0]
        opt.zero_grad(); loss = F.cross_entropy(model(train_x[idx]), train_y[idx]); loss.backward(); opt.step()
        if masks:
            with torch.no_grad():
                for p, mask in masks.items(): p.mul_(mask)
    return float(loss.item())

def global_masks(model, amount):
    weights = [p for name, p in model.named_parameters() if "weight" in name]
    flat = torch.cat([p.detach().abs().flatten() for p in weights])
    k = int(flat.numel() * amount)
    threshold = torch.topk(flat, k, largest=False).values.max() if k else -1
    return {p: (p.detach().abs() > threshold).to(p.dtype) for p in weights}

def apply_masks(masks):
    with torch.no_grad():
        for p, mask in masks.items(): p.mul_(mask)

dense = make_model(); train_steps(dense, 140, lr=0.05)
dense_acc = accuracy(dense)
one = copy.deepcopy(dense); one_masks = global_masks(one, 0.70); apply_masks(one_masks)
one_immediate = accuracy(one); train_steps(one, 36, one_masks, lr=0.015); one_final = accuracy(one)
gradual = copy.deepcopy(dense); gradual_masks = None
trajectory = []
for stage, amount in enumerate((0.30, 0.50, 0.70), 1):
    gradual_masks = global_masks(gradual, amount); apply_masks(gradual_masks)
    immediate = accuracy(gradual)
    train_steps(gradual, 12, gradual_masks, lr=0.015)
    trajectory.append({"stage": stage, "target": amount, "immediate_accuracy": immediate, "recovered_accuracy": accuracy(gradual)})
gradual_final = accuracy(gradual)
final_sparsity = sum((p == 0).sum().item() for p in gradual_masks) / sum(p.numel() for p in gradual_masks)
metrics = {
    "dense_accuracy": dense_acc,
    "oneshot_immediate_accuracy": one_immediate,
    "oneshot_recovered_accuracy": one_final,
    "gradual_recovered_accuracy": gradual_final,
    "final_sparsity": final_sparsity,
    "recovery_steps_per_route": 36,
    "gradual_trajectory": trajectory,
}
analysis = (
    f"The dense toy classifier reached {dense_acc:.1%}. One-shot 70% pruning changed validation accuracy "
    f"immediately to {one_immediate:.1%} and recovered to {one_final:.1%} after 36 updates. The staged route "
    f"finished at {gradual_final:.1%} with {final_sparsity:.1%} zeros under the same update budget. "
    "This isolates the support trajectory on one synthetic task; it is not a universal schedule ranking."
)
''',
    5: r'''
import torch.nn.utils.prune as prune
layer = nn.Linear(1024, 1024, bias=False, device=DEVICE)
probe = torch.randn(8, 1024, device=DEVICE)

def serialized(state):
    buffer = io.BytesIO(); torch.save(state, buffer); raw = buffer.getvalue()
    return len(raw), len(gzip.compress(raw, compresslevel=9))

dense_output = layer(probe).detach()
dense_raw, dense_gzip = serialized(layer.state_dict())
prune.l1_unstructured(layer, name="weight", amount=0.80)
hook_output = layer(probe).detach()
hook_keys = sorted(layer.state_dict().keys())
hook_raw, hook_gzip = serialized(layer.state_dict())
logical_sparsity = zero_fraction(layer.weight)
prune.remove(layer, "weight")
removed_output = layer(probe).detach()
removed_keys = sorted(layer.state_dict().keys())
removed_raw, removed_gzip = serialized(layer.state_dict())
metrics = {
    "sparsity": logical_sparsity,
    "dense_raw_bytes": dense_raw,
    "dense_gzip_bytes": dense_gzip,
    "hook_raw_bytes": hook_raw,
    "hook_gzip_bytes": hook_gzip,
    "removed_raw_bytes": removed_raw,
    "removed_gzip_bytes": removed_gzip,
    "hook_keys": hook_keys,
    "removed_keys": removed_keys,
    "remove_max_error": float((removed_output - hook_output).abs().max().item()),
    "pruning_change_rmse": tensor_metrics(dense_output, removed_output)["rmse"],
}
analysis = (
    f"The effective weight reached {logical_sparsity:.1%} sparsity. The hook checkpoint used keys "
    f"{hook_keys} and occupied {hook_raw:,} raw bytes, while `remove` restored a single key {removed_keys} "
    f"and {removed_raw:,} raw bytes. Gzip reduced the materialized payload to {removed_gzip:,} bytes. "
    f"Forward drift across `remove` was {metrics['remove_max_error']:.3e}, proving lifecycle equivalence but not sparse storage."
)
''',
    6: r'''
torch.manual_seed(SEED)
base = nn.Sequential(nn.Linear(64, 128, bias=False), nn.ReLU(), nn.Linear(128, 64, bias=False), nn.ReLU(), nn.Linear(64, 32, bias=False)).to(DEVICE).eval()
cal = torch.randn(256, 64, device=DEVICE)
held = torch.randn(256, 64, device=DEVICE) * 1.2
with torch.inference_mode(): ref_cal, ref_held = base(cal), base(held)
names = [name for name, p in base.named_parameters() if p.ndim == 2]

def clone_with_masks(mask_by_name):
    model = copy.deepcopy(base)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name in mask_by_name: p.mul_(mask_by_name[name])
    return model

sensitivities = {}
for name, p in base.named_parameters():
    if name in names:
        model = clone_with_masks({name: magnitude_mask(p, 0.30)})
        with torch.inference_mode(): sensitivities[name] = tensor_metrics(ref_cal, model(cal))["rmse"]

uniform_masks = {name: magnitude_mask(dict(base.named_parameters())[name], 0.50) for name in names}
all_values = torch.cat([dict(base.named_parameters())[name].detach().abs().flatten() for name in names])
k = int(all_values.numel() * 0.50); threshold = torch.topk(all_values, k, largest=False).values.max()
global_masks_dict = {name: (dict(base.named_parameters())[name].detach().abs() > threshold).float() for name in names}
adjusted_parts = []
for name in names:
    adjusted_parts.append((dict(base.named_parameters())[name].detach().abs() * (sensitivities[name] + 1e-6)).flatten())
adjusted = torch.cat(adjusted_parts); ath = torch.topk(adjusted, k, largest=False).values.max()
aware_masks = {name: ((dict(base.named_parameters())[name].detach().abs() * (sensitivities[name] + 1e-6)) > ath).float() for name in names}

uniform = clone_with_masks(uniform_masks); global_model = clone_with_masks(global_masks_dict); aware = clone_with_masks(aware_masks)
with torch.inference_mode():
    um = tensor_metrics(ref_held, uniform(held)); gm = tensor_metrics(ref_held, global_model(held)); am = tensor_metrics(ref_held, aware(held))
total_sparsity = sum((m == 0).sum().item() for m in aware_masks.values()) / sum(m.numel() for m in aware_masks.values())
metrics = {
    "uniform_rmse": um["rmse"], "global_rmse": gm["rmse"], "aware_rmse": am["rmse"],
    "uniform_cosine": um["cosine"], "global_cosine": gm["cosine"], "aware_cosine": am["cosine"],
    "total_sparsity": total_sparsity,
    "most_sensitive_layer": max(sensitivities, key=sensitivities.get),
    "sensitivities": sensitivities,
    "aware_layer_sparsity": {name: zero_fraction(mask) for name, mask in aware_masks.items()},
}
analysis = (
    f"All candidates used approximately {total_sparsity:.1%} global sparsity. Held-out RMSE was "
    f"{um['rmse']:.6f} for uniform, {gm['rmse']:.6f} for global magnitude, and {am['rmse']:.6f} "
    f"for the sensitivity-adjusted allocation. The calibration sweep identified "
    f"`{metrics['most_sensitive_layer']}` as most sensitive. This validates the budget experiment, not optimality of the heuristic."
)
''',
    7: r'''
dtype = torch.bfloat16
class Block(nn.Module):
    def __init__(self, c1=32):
        super().__init__(); self.conv1 = nn.Conv2d(16, c1, 3, padding=1, bias=False, dtype=dtype); self.conv2 = nn.Conv2d(c1, 24, 3, padding=1, bias=False, dtype=dtype)
    def forward(self, x): return self.conv2(F.relu(self.conv1(x)))

full = Block().to(DEVICE).eval(); masked = copy.deepcopy(full)
keep = torch.arange(0, 32, 2, device=DEVICE); remove = torch.arange(1, 32, 2, device=DEVICE)
with torch.no_grad(): masked.conv1.weight[remove] = 0
narrow = Block(c1=keep.numel()).to(DEVICE).eval()
with torch.no_grad():
    narrow.conv1.weight.copy_(full.conv1.weight[keep])
    narrow.conv2.weight.copy_(full.conv2.weight[:, keep])
x = torch.randn(16, 16, 32, 32, device=DEVICE, dtype=dtype)
with torch.inference_mode(): masked_y, narrow_y = masked(x), narrow(x)
tm = timing_summary(cuda_times(lambda: masked(x))); tn = timing_summary(cuda_times(lambda: narrow(x)))
def conv_flops(model, batch=16, h=32, w=32):
    return int(sum(2*batch*h*w*m.out_channels*(m.in_channels//m.groups)*m.kernel_size[0]*m.kernel_size[1] for m in model.modules() if isinstance(m, nn.Conv2d)))
metrics = {
    "masked_parameters": count_params(masked), "narrow_parameters": count_params(narrow),
    "masked_flops": conv_flops(masked), "narrow_flops": conv_flops(narrow),
    "flop_reduction": 1 - conv_flops(narrow)/conv_flops(masked),
    "max_error": float((masked_y.float()-narrow_y.float()).abs().max().item()),
    "masked_median_ms": tm["median_ms"], "narrow_median_ms": tn["median_ms"],
    "masked_channels": [32,24], "narrow_channels": [int(keep.numel()),24],
}
analysis = (
    f"Masking retained {metrics['masked_parameters']:,} parameters, while physical propagation reduced the block to "
    f"{metrics['narrow_parameters']:,} and analytical convolution work by {metrics['flop_reduction']:.1%}. "
    f"The copied narrow block matched the masked control within {metrics['max_error']:.3e}. Median latency changed "
    f"from {metrics['masked_median_ms']:.6f} to {metrics['narrow_median_ms']:.6f} ms on this shape."
)
''',
    8: r'''
class SlimBlock(nn.Module):
    def __init__(self, channels=24):
        super().__init__(); self.conv1=nn.Conv2d(8,channels,3,padding=1,bias=False); self.bn=nn.BatchNorm2d(channels); self.conv2=nn.Conv2d(channels,12,3,padding=1,bias=False)
    def forward(self,x): return self.conv2(F.relu(self.bn(self.conv1(x))))

full = SlimBlock().to(DEVICE).eval()
with torch.no_grad():
    full.bn.weight.copy_(torch.linspace(0.02, 1.2, 24, device=DEVICE)[torch.randperm(24, device=DEVICE)])
    full.bn.bias.zero_(); full.bn.running_mean.zero_(); full.bn.running_var.fill_(1)
keep = torch.topk(full.bn.weight.abs(), 12).indices.sort().values
remove_mask = torch.ones(24, dtype=torch.bool, device=DEVICE); remove_mask[keep] = False
masked = copy.deepcopy(full)
with torch.no_grad(): masked.bn.weight[remove_mask] = 0; masked.bn.bias[remove_mask] = 0
narrow = SlimBlock(12).to(DEVICE).eval()
with torch.no_grad():
    narrow.conv1.weight.copy_(full.conv1.weight[keep]); narrow.bn.weight.copy_(full.bn.weight[keep]); narrow.bn.bias.copy_(full.bn.bias[keep])
    narrow.bn.running_mean.copy_(full.bn.running_mean[keep]); narrow.bn.running_var.copy_(full.bn.running_var[keep]); narrow.conv2.weight.copy_(full.conv2.weight[:,keep])
x = torch.randn(12,8,32,32,device=DEVICE)
with torch.inference_mode(): ym, yn = masked(x), narrow(x)
tm=timing_summary(cuda_times(lambda: masked(x))); tn=timing_summary(cuda_times(lambda: narrow(x)))
metrics={
    "retained_channels": int(keep.numel()), "gamma_threshold": float(full.bn.weight[keep].abs().min().item()),
    "max_error": float((ym-yn).abs().max().item()), "full_parameters": count_params(full), "narrow_parameters": count_params(narrow),
    "masked_median_ms": tm["median_ms"], "narrow_median_ms": tn["median_ms"], "kept_indices": keep.tolist(),
}
analysis=(
    f"The gamma ranking retained {metrics['retained_channels']} channels above an absolute threshold of "
    f"{metrics['gamma_threshold']:.6f}. After slicing convolution and every BatchNorm state tensor, the narrow "
    f"output matched the gamma-masked control within {metrics['max_error']:.3e}. Parameters fell from "
    f"{metrics['full_parameters']:,} to {metrics['narrow_parameters']:,}; ranking quality on a real task remains unmeasured."
)
''',
    9: r'''
class Residual(nn.Module):
    def __init__(self, width=16):
        super().__init__(); self.a=nn.Conv2d(8,width,1,bias=False); self.abn=nn.BatchNorm2d(width); self.b=nn.Conv2d(8,width,3,padding=1,bias=False); self.bbn=nn.BatchNorm2d(width); self.out=nn.Conv2d(width,12,1,bias=False)
    def forward(self,x): return self.out(F.relu(self.abn(self.a(x))+self.bbn(self.b(x))))

full=Residual().to(DEVICE).eval(); x=torch.randn(4,8,20,20,device=DEVICE); keep=torch.arange(0,16,2,device=DEVICE)
bad_a=nn.Conv2d(8,8,1,bias=False,device=DEVICE)
with torch.no_grad(): bad_a.weight.copy_(full.a.weight[keep])
mismatch_captured=False; mismatch_message=""
try:
    _ = bad_a(x) + full.bbn(full.b(x))
except RuntimeError as exc:
    mismatch_captured=True; mismatch_message=str(exc).splitlines()[0]

control=copy.deepcopy(full)
remove=torch.tensor([i for i in range(16) if i not in keep.tolist()],device=DEVICE)
with torch.no_grad():
    control.a.weight[remove]=0; control.b.weight[remove]=0; control.abn.weight[remove]=0; control.abn.bias[remove]=0; control.bbn.weight[remove]=0; control.bbn.bias[remove]=0
valid=Residual(8).to(DEVICE).eval()
with torch.no_grad():
    valid.a.weight.copy_(full.a.weight[keep]); valid.b.weight.copy_(full.b.weight[keep]); valid.out.weight.copy_(full.out.weight[:,keep])
    for src,dst in ((full.abn,valid.abn),(full.bbn,valid.bbn)):
        dst.weight.copy_(src.weight[keep]); dst.bias.copy_(src.bias[keep]); dst.running_mean.copy_(src.running_mean[keep]); dst.running_var.copy_(src.running_var[keep])
with torch.inference_mode(): yc,yv=control(x),valid(x)
metrics={
    "mismatch_captured":mismatch_captured,"mismatch_message":mismatch_message,"retained_channels":int(keep.numel()),
    "valid_output_channels":int(yv.shape[1]),"valid_max_error":float((yc-yv).abs().max().item()),
    "full_parameters":count_params(full),"narrow_parameters":count_params(valid),
}
analysis=(
    f"Pruning only one additive branch produced a captured shape failure: `{mismatch_message}`. The synchronized group "
    f"retained {keep.numel()} channels across both branches, normalization state, and the consumer; it produced "
    f"{metrics['valid_output_channels']} output channels with {metrics['valid_max_error']:.3e} control drift."
)
''',
    10: r'''
torch.manual_seed(SEED)
n,d,h,c=1400,16,12,3
x=torch.randn(n,d,device=DEVICE); teacher=torch.randn(d,c,device=DEVICE); y=(x@teacher+0.3*torch.randn(n,c,device=DEVICE)).argmax(1)
train_x,cal_x,test_x=x[:900],x[900:1150],x[1150:]; train_y,cal_y,test_y=y[:900],y[900:1150],y[1150:]
model=nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,c)).to(DEVICE)
opt=torch.optim.Adam(model.parameters(),lr=0.03)
for step in range(120):
    idx=torch.arange(step*64,step*64+64,device=DEVICE)%train_x.shape[0]; opt.zero_grad(); loss=F.cross_entropy(model(train_x[idx]),train_y[idx]); loss.backward(); opt.step()
model.eval(); hidden=F.relu(model[0](cal_x)); hidden.retain_grad(); logits=model[2](hidden); cal_loss=F.cross_entropy(logits,cal_y); model.zero_grad(); cal_loss.backward()
taylor=(hidden*hidden.grad).abs().mean(0).detach()
l1=(model[0].weight.abs().sum(1)+model[2].weight.abs().sum(0)).detach()
with torch.inference_mode():
    test_hidden=F.relu(model[0](test_x)); base_logits=model[2](test_hidden); base_loss=float(F.cross_entropy(base_logits,test_y).item()); actual=[]
    for channel in range(h):
        ablated=test_hidden.clone(); ablated[:,channel]=0; actual.append(float(F.cross_entropy(model[2](ablated),test_y).item()-base_loss))
actual_t=torch.tensor(actual)
metrics={
    "l1_spearman":spearman(l1.cpu(),actual_t),"taylor_spearman":spearman(taylor.cpu(),actual_t),
    "l1_top_channel":int(torch.argmax(l1).item()),"taylor_top_channel":int(torch.argmax(taylor).item()),
    "actual_top_channel":int(torch.argmax(actual_t).item()),"baseline_loss":base_loss,
    "l1_scores":l1.cpu().tolist(),"taylor_scores":taylor.cpu().tolist(),"actual_loss_increase":actual,
}
analysis=(
    f"Against exhaustive held-out ablations, L1 ranking had Spearman {metrics['l1_spearman']:.4f} and Taylor "
    f"had {metrics['taylor_spearman']:.4f}. Their top channels were {metrics['l1_top_channel']} and "
    f"{metrics['taylor_top_channel']}, while the largest actual loss increase came from channel "
    f"{metrics['actual_top_channel']}. The result tests one local ranking on one calibration batch."
)
''',
    11: r'''
torch.manual_seed(SEED)
n,d,c=1400,24,4
x=torch.randn(n,d,device=DEVICE); tw=torch.randn(d,c,device=DEVICE); y=(x@tw+0.25*torch.randn(n,c,device=DEVICE)).argmax(1)
tx,vx=x[:1100],x[1100:]; ty,vy=y[:1100],y[1100:]
def make(): return nn.Sequential(nn.Linear(d,48),nn.ReLU(),nn.Linear(48,c)).to(DEVICE)
def acc(m):
    m.eval();
    with torch.inference_mode(): return float((m(vx).argmax(1)==vy).float().mean().item())
def weights(m): return [p for name,p in m.named_parameters() if "weight" in name]
def set_global_mask(m,rate):
    ps=weights(m); flat=torch.cat([p.detach().abs().flatten() for p in ps]); k=int(flat.numel()*rate); idx=torch.topk(flat,k,largest=False).indices; fm=torch.ones_like(flat); fm[idx]=0
    masks={}; offset=0
    for p in ps: masks[p]=fm[offset:offset+p.numel()].view_as(p); offset+=p.numel()
    with torch.no_grad():
        for p,mask in masks.items(): p.mul_(mask)
    return masks
def train(m,steps,schedule):
    opt=torch.optim.SGD(m.parameters(),lr=0.02,momentum=0.8); masks=None; rows=[]
    for step in range(steps):
        target=schedule(step,steps)
        if step==0 or step%4==0 or step==steps-1: masks=set_global_mask(m,target); rows.append({"step":step,"target":target,"accuracy_before_update":acc(m)})
        idx=torch.arange(step*80,step*80+80,device=DEVICE)%tx.shape[0]; opt.zero_grad(); loss=F.cross_entropy(m(tx[idx]),ty[idx]); loss.backward(); opt.step()
        if masks:
            with torch.no_grad():
                for p,mask in masks.items(): p.mul_(mask)
    return rows,masks
dense=make(); opt=torch.optim.Adam(dense.parameters(),lr=0.03)
for step in range(120):
    idx=torch.arange(step*64,step*64+64,device=DEVICE)%tx.shape[0]; opt.zero_grad(); loss=F.cross_entropy(dense(tx[idx]),ty[idx]); loss.backward(); opt.step()
dense_acc=acc(dense); one=copy.deepcopy(dense); gradual=copy.deepcopy(dense); steps=40
one_rows,one_masks=train(one,steps,lambda step,total:0.80)
def cubic(step,total):
    progress=step/max(total-1,1); return 0.80*(1-(1-progress)**3)
grad_rows,grad_masks=train(gradual,steps,cubic)
one_sp=sum((p==0).sum().item() for p in one_masks)/sum(p.numel() for p in one_masks); grad_sp=sum((p==0).sum().item() for p in grad_masks)/sum(p.numel() for p in grad_masks)
metrics={"dense_accuracy":dense_acc,"oneshot_final_accuracy":acc(one),"gradual_final_accuracy":acc(gradual),"target_sparsity":0.80,"oneshot_actual_sparsity":one_sp,"gradual_actual_sparsity":grad_sp,"schedule_updates":len(grad_rows),"oneshot_trajectory":one_rows,"gradual_trajectory":grad_rows}
analysis=(f"Both routes used {steps} optimizer updates and finished near 80% sparsity. One-shot validation accuracy ended at "
          f"{metrics['oneshot_final_accuracy']:.1%}, while the cubic route ended at {metrics['gradual_final_accuracy']:.1%} "
          f"after {metrics['schedule_updates']} recorded mask updates. The retained trajectories show when each support shock occurred.")
''',
    12: r'''
torch.manual_seed(SEED)
n,d=1600,24
x=torch.randn(n,d,device=DEVICE); true_w=torch.zeros(d,device=DEVICE); true_w[:6]=torch.tensor([2.0,-1.5,1.2,-0.9,0.7,0.5],device=DEVICE); y=x@true_w+0.15*torch.randn(n,device=DEVICE)
tx,vx=x[:1200],x[1200:]; ty,vy=y[:1200],y[1200:]
class Gated(nn.Module):
    def __init__(self): super().__init__(); self.logits=nn.Parameter(torch.zeros(d)); self.weight=nn.Parameter(torch.randn(d)*0.05)
    def gates(self): return torch.sigmoid(self.logits)
    def forward(self,z): return (z*self.gates())@self.weight
def train_gate(lam):
    m=Gated().to(DEVICE); opt=torch.optim.Adam(m.parameters(),lr=0.04)
    for step in range(260):
        idx=torch.arange(step*96,step*96+96,device=DEVICE)%tx.shape[0]; opt.zero_grad(); pred=m(tx[idx]); loss=F.mse_loss(pred,ty[idx])+lam*m.gates().mean(); loss.backward(); opt.step()
    return m
weak=train_gate(0.005); strong=train_gate(0.15)
thresholds=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
rows=[]
with torch.inference_mode():
    for th in thresholds:
        g=strong.gates(); mask=(g>=th).float(); pred=(vx*g*mask)@strong.weight; rows.append({"threshold":th,"active":int(mask.sum().item()),"mse":float(F.mse_loss(pred,vy).item())})
best=min(rows,key=lambda r:r["mse"]); wg=weak.gates().detach(); sg=strong.gates().detach()
metrics={"weak_active_channels":int((wg>=0.5).sum().item()),"strong_active_channels":int((sg>=0.5).sum().item()),"weak_gate_mean":float(wg.mean().item()),"strong_gate_mean":float(sg.mean().item()),"weak_val_mse":float(F.mse_loss(weak(vx),vy).item()),"strong_val_mse":float(F.mse_loss(strong(vx),vy).item()),"selected_threshold":best["threshold"],"thresholded_val_mse":best["mse"],"threshold_sweep":rows,"strong_gates":sg.cpu().tolist()}
analysis=(f"Weak regularization left {metrics['weak_active_channels']} gates above 0.5 with mean {metrics['weak_gate_mean']:.4f}; "
          f"strong regularization left {metrics['strong_active_channels']} with mean {metrics['strong_gate_mean']:.4f}. "
          f"The frozen threshold sweep selected {best['threshold']:.1f}, {best['active']} active channels, and validation MSE {best['mse']:.6f}. Continuous gates still require physical rebuilding.")
''',
    13: r'''
dtype=torch.bfloat16; rows,cols,batch=1024,1024,32
w=torch.randn(rows,cols,device=DEVICE,dtype=dtype); x=torch.randn(batch,cols,device=DEVICE,dtype=dtype)
random_mask=(torch.rand_like(w.float())>0.5).to(dtype); random_w=w*random_mask; nm_w=w*exact_2_4_mask(w)
td=timing_summary(cuda_times(lambda:F.linear(x,w))); tn=timing_summary(cuda_times(lambda:F.linear(x,nm_w)))
native_ok=False; native_message="API unavailable"
try:
    if hasattr(torch.sparse,"to_sparse_semi_structured"):
        sparse=torch.sparse.to_sparse_semi_structured(nm_w); _=F.linear(x,sparse); torch.cuda.synchronize(); native_ok=True; native_message=str(type(sparse).__name__)
    else: native_message="torch.sparse.to_sparse_semi_structured is unavailable"
except Exception as exc:
    native_message=f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
metrics={"random_compliance":compliance_2_4(random_w),"nm_compliance":compliance_2_4(nm_w),"random_sparsity":zero_fraction(random_w),"nm_sparsity":zero_fraction(nm_w),"dense_median_ms":td["median_ms"],"nm_dense_path_median_ms":tn["median_ms"],"native_conversion_succeeded":native_ok,"native_message":native_message,"shape":[rows,cols],"dtype":str(dtype)}
analysis=(f"Random 50% masking achieved {metrics['random_compliance']:.1%} local compliance, while top-2-of-4 reached "
          f"{metrics['nm_compliance']:.1%} at {metrics['nm_sparsity']:.1%} sparsity. Ordinary dense-path medians were "
          f"{td['median_ms']:.6f} and {tn['median_ms']:.6f} ms. Native semi-structured conversion success was {native_ok}; "
          f"the retained probe message is `{native_message}`.")
''',
    14: r'''
import torch.nn.utils.prune as prune
layer=nn.Linear(256,128,device=DEVICE); x=torch.randn(32,256,device=DEVICE); target=torch.randn(32,128,device=DEVICE)
prune.l1_unstructured(layer,"weight",amount=0.25)
first_sparsity=zero_fraction(layer.weight); first_params=sorted(n for n,_ in layer.named_parameters()); first_buffers=sorted(n for n,_ in layer.named_buffers())
prune.l1_unstructured(layer,"weight",amount=0.25)
iterated_sparsity=zero_fraction(layer.weight); before_output=layer(x).detach(); hooks_before=len(layer._forward_pre_hooks)
opt=torch.optim.SGD(layer.parameters(),lr=0.01); opt.zero_grad(); loss=F.mse_loss(layer(x),target); loss.backward(); opt.step(); trained_output=layer(x).detach(); post_step_sparsity=zero_fraction(layer.weight)
params_before=sorted(n for n,_ in layer.named_parameters()); buffers_before=sorted(n for n,_ in layer.named_buffers())
pre_remove=layer(x).detach(); prune.remove(layer,"weight"); post_remove=layer(x).detach()
metrics={"first_sparsity":first_sparsity,"iterated_sparsity":iterated_sparsity,"post_step_sparsity":post_step_sparsity,"first_parameters":first_params,"first_buffers":first_buffers,"parameters_before_remove":",".join(params_before),"buffers_before_remove":",".join(buffers_before),"hooks_before_remove":hooks_before,"parameters_after_remove":sorted(n for n,_ in layer.named_parameters()),"buffers_after_remove":sorted(n for n,_ in layer.named_buffers()),"hooks_after_remove":len(layer._forward_pre_hooks),"training_loss":float(loss.item()),"training_output_change":tensor_metrics(before_output,trained_output)["rmse"],"remove_max_error":float((pre_remove-post_remove).abs().max().item())}
analysis=(f"The first call produced {first_sparsity:.1%} sparsity and the second composed to {iterated_sparsity:.1%}. "
          f"Before removal, parameters were `{metrics['parameters_before_remove']}`, buffers were `{metrics['buffers_before_remove']}`, "
          f"and {hooks_before} pre-hook was active. `remove` left drift {metrics['remove_max_error']:.3e} and restored a materialized `weight` parameter.")
''',
    15: r'''
class MiniResidual(nn.Module):
    def __init__(self,width=12): super().__init__(); self.a=nn.Conv2d(4,width,1,bias=False); self.b=nn.Conv2d(4,width,3,padding=1,bias=False); self.out=nn.Conv2d(width,6,1,bias=False)
    def forward(self,x): return self.out(F.relu(self.a(x)+self.b(x)))
model=MiniResidual().to(DEVICE).eval(); example=torch.randn(1,4,12,12,device=DEVICE); idxs=[1,3,5,7]
keep=torch.tensor([i for i in range(12) if i not in idxs],device=DEVICE); manual=MiniResidual(8).to(DEVICE).eval()
with torch.no_grad(): manual.a.weight.copy_(model.a.weight[keep]); manual.b.weight.copy_(model.b.weight[keep]); manual.out.weight.copy_(model.out.weight[:,keep])
with torch.inference_mode(): manual_out=manual(example)
available=importlib.util.find_spec("torch_pruning") is not None; group_built=False; group_valid=False; group_size=0; version=None; message="torch_pruning not installed"
if available:
    try:
        import torch_pruning as tp
        version=getattr(tp,"__version__","unknown")
        dg=tp.DependencyGraph().build_dependency(model,example_inputs=example)
        group=dg.get_pruning_group(model.a,tp.prune_conv_out_channels,idxs=idxs)
        group_built=True; group_valid=bool(dg.check_pruning_group(group)); group_size=len(group); message=str(group)
    except Exception as exc: message=f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
metrics={"torch_pruning_available":available,"torch_pruning_version":version,"group_built":group_built,"group_valid":group_valid,"group_size":group_size,"manual_output_channels":int(manual_out.shape[1]),"manual_parameters":count_params(manual),"full_parameters":count_params(model),"probe_message":message[:1000]}
analysis=(f"The manual dependency control reduced the model from {metrics['full_parameters']:,} to {metrics['manual_parameters']:,} "
          f"parameters and produced {metrics['manual_output_channels']} output channels. Torch-Pruning availability was {available}; "
          f"group built/valid were {group_built}/{group_valid}. This is a bounded compatibility result when the optional package is absent.")
''',
    16: r'''
tf_available=importlib.util.find_spec("tensorflow") is not None; tfmot_available=importlib.util.find_spec("tensorflow_model_optimization") is not None
initial,final,begin,end=0.0,0.80,10,50
def schedule(step):
    if step<=begin:return initial
    if step>=end:return final
    p=(step-begin)/(end-begin); return final+(initial-final)*(1-p)**3
steps=torch.tensor([0,10,20,30,40,50,60],device=DEVICE); schedule_values=[float(schedule(int(s))) for s in steps.cpu().tolist()]
native=False; native_message="TensorFlow/TFMOT not both available"
if tf_available and tfmot_available:
    try:
        import tensorflow as tf, tensorflow_model_optimization as tfmot
        base=tf.keras.Sequential([tf.keras.layers.Input((8,)),tf.keras.layers.Dense(4)])
        wrapped=tfmot.sparsity.keras.prune_low_magnitude(base); stripped=tfmot.sparsity.keras.strip_pruning(wrapped); native=True; native_message=stripped.__class__.__name__
    except Exception as exc:native_message=f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
metrics={"tensorflow_available":tf_available,"tfmot_available":tfmot_available,"schedule_steps":steps.cpu().tolist(),"schedule_values":schedule_values,"mid_schedule_sparsity":schedule(30),"final_schedule_sparsity":schedule(50),"native_probe_executed":native,"native_message":native_message,"lifecycle_ready":bool(native)}
analysis=(f"The cubic schedule moved from {schedule_values[1]:.1%} at step 10 to {metrics['mid_schedule_sparsity']:.1%} "
          f"at step 30 and {metrics['final_schedule_sparsity']:.1%} at step 50. TensorFlow/TFMOT availability was "
          f"{tf_available}/{tfmot_available}, so native wrapper stripping executed={native}. Missing native stages remain false rather than inferred.")
''',
    17: r'''
availability={name:(importlib.util.find_spec(name) is not None) for name in ("openvino","nncf","neural_compressor")}
w=torch.randn(1024,1024,device=DEVICE,dtype=torch.bfloat16); x=torch.randn(16,1024,device=DEVICE,dtype=torch.bfloat16)
masked=w*magnitude_mask(w,0.75); narrow=w[:256]
with torch.inference_mode(): full_y=F.linear(x,w); narrow_y=F.linear(x,narrow)
metrics={"openvino_available":availability["openvino"],"nncf_available":availability["nncf"],"neural_compressor_available":availability["neural_compressor"],"logical_sparsity":zero_fraction(masked),"original_width":1024,"narrow_width":256,"width_reduction":0.75,"full_output_shape":list(full_y.shape),"narrow_output_shape":list(narrow_y.shape),"native_cpu_run":False,"native_cpu_latency_ms":None}
analysis=(f"The dense-value control reached {metrics['logical_sparsity']:.1%} logical sparsity without changing its 1024-wide "
          f"output; the physical control changed width to 256. OpenVINO/NNCF/Neural Compressor availability was "
          f"{availability['openvino']}/{availability['nncf']}/{availability['neural_compressor']}. No CPU latency is reported because no native CPU path executed.")
''',
    18: r'''
w=torch.randn(1024,1024,device=DEVICE,dtype=torch.float16); compliant=w*exact_2_4_mask(w)
trt_available=importlib.util.find_spec("tensorrt") is not None; poly_available=importlib.util.find_spec("polygraphy") is not None; trtexec=shutil.which("trtexec")
metrics={"nm_compliance":compliance_2_4(compliant),"sparsity":zero_fraction(compliant),"dtype":str(compliant.dtype),"dtype_eligible":compliant.dtype in (torch.float16,torch.int8),"tensorrt_available":trt_available,"polygraphy_available":poly_available,"trtexec_available":trtexec is not None,"trtexec_path":trtexec,"eligible_weight_data":bool(compliance_2_4(compliant)==1.0),"sparse_engine_built":False,"sparse_tactic_selected":False}
analysis=(f"The weight passed {metrics['nm_compliance']:.1%} of exact 2:4 groups at {metrics['sparsity']:.1%} sparsity and "
          f"used an eligible dtype={metrics['dtype_eligible']}. TensorRT/Polygraphy/trtexec availability was "
          f"{trt_available}/{poly_available}/{trtexec is not None}. Because no engine was built, sparse tactic selection remains false.")
''',
    19: r'''
class PrunedMultiInput(nn.Module):
    def __init__(self): super().__init__(); self.proj=nn.Linear(12,7); self.gate=nn.Linear(5,7,bias=False); self.out=nn.Linear(7,3)
    def forward(self,x,context): return self.out(F.gelu(self.proj(x)+self.gate(context)))
model=PrunedMultiInput().to(DEVICE).eval(); x=torch.randn(4,12,device=DEVICE); context=torch.randn(4,5,device=DEVICE)
with torch.inference_mode(): ref=model(x,context).cpu().numpy()
onnx_available=importlib.util.find_spec("onnx") is not None; ort_available=importlib.util.find_spec("onnxruntime") is not None
artifact_dir=Path("artifacts"); artifact_dir.mkdir(exist_ok=True); onnx_path=artifact_dir/"pruned-multi-input.onnx"
exported=checker=False; inferred_ok=False; ort_ok=False; ort_error=float("nan"); inferred_shapes=[]; message=""
try:
    torch.onnx.export(model,(x,context),onnx_path,input_names=["x","context"],output_names=["logits"],dynamic_axes={"x":{0:"batch"},"context":{0:"batch"},"logits":{0:"batch"}},opset_version=17,dynamo=False)
    exported=True
    import onnx
    loaded=onnx.load(onnx_path); onnx.checker.check_model(loaded,full_check=True); checker=True
    inferred=onnx.shape_inference.infer_shapes(loaded); inferred_ok=True
    for value in list(inferred.graph.value_info)+list(inferred.graph.output):
        dims=[d.dim_value if d.HasField("dim_value") else d.dim_param for d in value.type.tensor_type.shape.dim]
        inferred_shapes.append({"name":value.name,"dims":dims})
    if ort_available:
        import onnxruntime as ort
        session=ort.InferenceSession(str(onnx_path),providers=["CPUExecutionProvider"]); got=session.run(None,{"x":x.cpu().numpy(),"context":context.cpu().numpy()})[0]; ort_error=float(abs(got-ref).max()); ort_ok=True
except Exception as exc: message=f"{type(exc).__name__}: {str(exc).splitlines()[0]}"
metrics={"onnx_available":onnx_available,"onnxruntime_available":ort_available,"export_succeeded":exported,"checker_passed":checker,"shape_inference_passed":inferred_ok,"ort_executed":ort_ok,"ort_max_error":ort_error,"onnx_bytes":onnx_path.stat().st_size if onnx_path.exists() else 0,"inferred_shapes":inferred_shapes,"initializer_shapes":{"proj.weight":[7,12],"gate.weight":[7,5],"out.weight":[3,7]},"message":message}
analysis=(f"ONNX export/checker/shape-inference gates were {exported}/{checker}/{inferred_ok}; ONNX Runtime executed={ort_ok} "
          f"with max error {ort_error:.3e}. The graph occupied {metrics['onnx_bytes']:,} bytes and retained the physical width 7 "
          "through both input projections and the output consumer.")
''',
    20: r'''
class TinyResNet(nn.Module):
    def __init__(self,width=32):
        super().__init__(); self.stem=nn.Conv2d(3,width,3,padding=1,bias=False); self.main1=nn.Conv2d(width,width,3,padding=1,bias=False); self.main2=nn.Conv2d(width,width,3,padding=1,bias=False); self.proj=nn.Conv2d(width,width,1,bias=False); self.fc=nn.Linear(width,10)
    def forward(self,x):
        h=F.relu(self.stem(x)); h=F.relu(self.main2(F.relu(self.main1(h)))+self.proj(h)); return self.fc(h.mean((2,3)))
full=TinyResNet(32).to(DEVICE).to(torch.bfloat16).eval(); narrow=TinyResNet(16).to(DEVICE).to(torch.bfloat16).eval()
def flops(m,batch,h=32,w=32):
    total=0
    for mod in m.modules():
        if isinstance(mod,nn.Conv2d): total+=2*batch*h*w*mod.out_channels*mod.in_channels*mod.kernel_size[0]*mod.kernel_size[1]
        if isinstance(mod,nn.Linear): total+=2*batch*mod.in_features*mod.out_features
    return int(total)
x1=torch.randn(1,3,32,32,device=DEVICE,dtype=torch.bfloat16); x64=torch.randn(64,3,32,32,device=DEVICE,dtype=torch.bfloat16)
f1=timing_summary(cuda_times(lambda:full(x1))); n1=timing_summary(cuda_times(lambda:narrow(x1))); f64=timing_summary(cuda_times(lambda:full(x64))); n64=timing_summary(cuda_times(lambda:narrow(x64)))
metrics={"full_parameters":count_params(full),"narrow_parameters":count_params(narrow),"full_batch1_flops":flops(full,1),"narrow_batch1_flops":flops(narrow,1),"flop_reduction":1-flops(narrow,1)/flops(full,1),"batch1_full_median_ms":f1["median_ms"],"batch1_narrow_median_ms":n1["median_ms"],"batch64_full_median_ms":f64["median_ms"],"batch64_narrow_median_ms":n64["median_ms"],"batch64_speedup":f64["median_ms"]/n64["median_ms"]}
analysis=(f"Halving stage width reduced parameters from {metrics['full_parameters']:,} to {metrics['narrow_parameters']:,} "
          f"and analytical work by {metrics['flop_reduction']:.1%}. Batch-1 medians were {f1['median_ms']:.6f} versus "
          f"{n1['median_ms']:.6f} ms; batch-64 measured a {metrics['batch64_speedup']:.3f}x ratio. Random weights make this a systems/shape case study, not a Top-1 result.")
''',
    21: r'''
torch.manual_seed(SEED)
channels=24; samples=512
features={"large":torch.randn(samples,channels,device=DEVICE)*0.8,"medium":torch.randn(samples,channels,device=DEVICE),"small":torch.randn(samples,channels,device=DEVICE)*1.8}
weights={name:torch.randn(channels,8,device=DEVICE) for name in features}
targets={name:features[name]@weights[name] for name in features}
def select(counts):
    outputs={}; retained=0
    for name,count in counts.items():
        score=(features[name].square().mean(0).sqrt()[:,None]*weights[name].abs()).sum(1)
        keep=torch.topk(score,count).indices; outputs[name]=features[name][:,keep]@weights[name][keep]; retained+=count
    return outputs,retained
uniform,nu=select({"large":12,"medium":12,"small":12})
protected,np=select({"large":6,"medium":10,"small":20})
def errors(outputs):
    per={name:tensor_metrics(targets[name],outputs[name])["rmse"] for name in targets}
    aggregate=float(torch.sqrt(torch.mean(torch.cat([(outputs[n]-targets[n]).flatten() for n in targets]).square())).item())
    return per,aggregate
ue,ua=errors(uniform); pe,pa=errors(protected)
metrics={"uniform_aggregate_rmse":ua,"protected_aggregate_rmse":pa,"uniform_large_rmse":ue["large"],"uniform_medium_rmse":ue["medium"],"uniform_small_rmse":ue["small"],"protected_large_rmse":pe["large"],"protected_medium_rmse":pe["medium"],"protected_small_rmse":pe["small"],"uniform_worst_slice":max(ue,key=ue.get),"protected_worst_slice":max(pe,key=pe.get),"retained_channels":nu,"protected_allocation":{"large":6,"medium":10,"small":20}}
analysis=(f"Both policies retained {nu} channels across three branches. Uniform allocation produced aggregate RMSE {ua:.6f} "
          f"and small-slice RMSE {ue['small']:.6f}; protecting the high-resolution branch produced {pa:.6f} and "
          f"{pe['small']:.6f}, respectively. The per-slice table—not the aggregate alone—determines whether the risk trade is acceptable.")
''',
    22: r'''
torch.manual_seed(SEED)
class TinyBlock(nn.Module):
    def __init__(self,d=64,heads=4,ff=256):
        super().__init__(); self.d=d; self.heads=heads; self.hd=d//heads; self.ln1=nn.LayerNorm(d); self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d); self.ln2=nn.LayerNorm(d); self.fc1=nn.Linear(d,ff); self.fc2=nn.Linear(ff,d)
    def forward(self,x,head_mask=None):
        h=self.ln1(x); b,s,_=h.shape; qkv=self.qkv(h).view(b,s,3,self.heads,self.hd).permute(2,0,3,1,4); q,k,v=qkv
        attn=torch.softmax(q@k.transpose(-2,-1)/math.sqrt(self.hd),dim=-1); heads=attn@v
        if head_mask is not None: heads=heads*head_mask.view(1,-1,1,1)
        a=self.proj(heads.transpose(1,2).reshape(b,s,self.d)); x=x+a; return x+self.fc2(F.gelu(self.fc1(self.ln2(x))))
full=TinyBlock().to(DEVICE).to(torch.bfloat16).eval(); narrow=TinyBlock(ff=128).to(DEVICE).to(torch.bfloat16).eval()
with torch.no_grad():
    narrow.ln1.load_state_dict(full.ln1.state_dict()); narrow.qkv.load_state_dict(full.qkv.state_dict()); narrow.proj.load_state_dict(full.proj.state_dict()); narrow.ln2.load_state_dict(full.ln2.state_dict()); narrow.fc1.weight.copy_(full.fc1.weight[:128]); narrow.fc1.bias.copy_(full.fc1.bias[:128]); narrow.fc2.weight.copy_(full.fc2.weight[:,:128]); narrow.fc2.bias.copy_(full.fc2.bias)
x=torch.randn(8,128,64,device=DEVICE,dtype=torch.bfloat16); head_mask=torch.tensor([1,0,1,0],device=DEVICE,dtype=torch.bfloat16)
with torch.inference_mode(): ref=full(x); hm=full(x,head_mask); fn=narrow(x); skip=x
tf=timing_summary(cuda_times(lambda:full(x))); th=timing_summary(cuda_times(lambda:full(x,head_mask))); tn=timing_summary(cuda_times(lambda:narrow(x)))
metrics={"full_parameters":count_params(full),"head_mask_parameters":count_params(full),"ffn_narrow_parameters":count_params(narrow),"head_mask_rmse":tensor_metrics(ref,hm)["rmse"],"ffn_narrow_rmse":tensor_metrics(ref,fn)["rmse"],"layer_skip_rmse":tensor_metrics(ref,skip)["rmse"],"full_median_ms":tf["median_ms"],"head_mask_median_ms":th["median_ms"],"ffn_narrow_median_ms":tn["median_ms"],"sequence_length":128,"hidden_width":64,"full_ffn_width":256,"narrow_ffn_width":128}
analysis=(f"Masking two of four heads preserved {metrics['head_mask_parameters']:,} parameters and measured "
          f"{th['median_ms']:.6f} ms versus {tf['median_ms']:.6f} ms for the full block. Physical FFN narrowing "
          f"reduced parameters to {metrics['ffn_narrow_parameters']:,}, measured {tn['median_ms']:.6f} ms, and introduced "
          f"RMSE {metrics['ffn_narrow_rmse']:.6f}. Layer skipping had RMSE {metrics['layer_skip_rmse']:.6f} before recovery.")
''',
    23: r'''
torch.manual_seed(SEED)
out_features,in_features=512,512
w=torch.randn(out_features,in_features,device=DEVICE); scales=torch.logspace(-1,1,in_features,device=DEVICE); cal=torch.randn(256,in_features,device=DEVICE)*scales; held=torch.randn(256,in_features,device=DEVICE)*scales
def row_mask(score,rate=0.5):
    k=int(score.shape[1]*(1-rate)); idx=torch.topk(score,k,dim=1).indices; m=torch.zeros_like(score); m.scatter_(1,idx,1); return m
mag_score=w.abs(); act_norm=cal.square().mean(0).sqrt(); wanda_score=w.abs()*act_norm
h=(cal.T@cal)/cal.shape[0]+0.01*torch.eye(in_features,device=DEVICE); hinv=torch.linalg.inv(h); curvature_score=w.square()/(torch.diag(hinv).clamp_min(1e-8))[None,:]
mm=row_mask(mag_score); wm=row_mask(wanda_score); cm=row_mask(curvature_score)
with torch.inference_mode(): ref=F.linear(held,w); my=F.linear(held,w*mm); wy=F.linear(held,w*wm); cy=F.linear(held,w*cm)
me=tensor_metrics(ref,my); we=tensor_metrics(ref,wy); ce=tensor_metrics(ref,cy); overlap=float(((mm==1)&(wm==1)).sum().item()/max((mm==1).sum().item(),1))
metrics={"sparsity":zero_fraction(w*mm),"magnitude_rmse":me["rmse"],"wanda_rmse":we["rmse"],"curvature_rmse":ce["rmse"],"magnitude_cosine":me["cosine"],"wanda_cosine":we["cosine"],"curvature_cosine":ce["cosine"],"magnitude_wanda_overlap":overlap,"calibration_scale_min":float(scales.min().item()),"calibration_scale_max":float(scales.max().item())}
analysis=(f"At {metrics['sparsity']:.1%} sparsity, magnitude/Wanda/curvature-proxy held-out RMSE values were "
          f"{me['rmse']:.6f}/{we['rmse']:.6f}/{ce['rmse']:.6f}. Magnitude and Wanda retained-support overlap was "
          f"{overlap:.1%} under a 100x calibration feature-scale range. The curvature score is a diagonal OBS-style proxy, not SparseGPT's sequential algorithm.")
''',
    24: r'''
torch.manual_seed(SEED)
rows,cols=384,384
w=torch.randn(rows,cols,device=DEVICE)*torch.logspace(-1,1,cols,device=DEVICE)[None,:]; cal=torch.randn(256,cols,device=DEVICE); held=torch.randn(256,cols,device=DEVICE); ref=F.linear(held,w)
mask=magnitude_mask(w,0.60)
def percentile_scale(t,q=0.99): return float(torch.quantile(t.detach().abs().float().flatten(),q).item()/127.0+1e-12)
def qdq(t,scale): return torch.clamp(torch.round(t/scale),-127,127)*scale
qfirst_scale=percentile_scale(w); qfirst=qdq(w,qfirst_scale)*mask
pruned=w*mask; pfirst_scale=percentile_scale(pruned); pfirst=qdq(pruned,pfirst_scale)
qerr=tensor_metrics(ref,F.linear(held,qfirst)); perr=tensor_metrics(ref,F.linear(held,pfirst))
master=nn.Parameter(pruned.clone()); opt=torch.optim.Adam([master],lr=0.01); teacher_cal=F.linear(cal,w).detach()
for _ in range(35):
    opt.zero_grad(); pred=F.linear(cal,master*mask); loss=F.mse_loss(pred,teacher_cal); loss.backward(); opt.step()
    with torch.no_grad(): master.mul_(mask)
recovered=qdq(master*mask,percentile_scale(master*mask)); rerr=tensor_metrics(ref,F.linear(held,recovered))
metrics={"quantize_first_scale":qfirst_scale,"prune_first_scale":pfirst_scale,"quantize_first_rmse":qerr["rmse"],"prune_first_rmse":perr["rmse"],"recovered_rmse":rerr["rmse"],"quantize_first_cosine":qerr["cosine"],"prune_first_cosine":perr["cosine"],"recovered_cosine":rerr["cosine"],"final_sparsity":zero_fraction(recovered),"recovery_steps":35}
analysis=(f"Dense-calibrated quantization used scale {qfirst_scale:.6g}; recalibration after pruning used {pfirst_scale:.6g}. "
          f"Held-out RMSE was {qerr['rmse']:.6f} for quantize-then-prune and {perr['rmse']:.6f} for prune-then-quantize. "
          f"A 35-step constrained teacher recovery reached {rerr['rmse']:.6f} at {metrics['final_sparsity']:.1%} sparsity.")
''',
    25: r'''
dtype=torch.bfloat16; in_f,out_f=2048,2048
w=torch.randn(out_f,in_f,device=DEVICE,dtype=dtype); masked=w*magnitude_mask(w,0.75); narrow=w[:512]
def bench(weight,batch,repeats=80):
    x=torch.randn(batch,in_f,device=DEVICE,dtype=dtype); torch.cuda.reset_peak_memory_stats(); t=timing_summary(cuda_times(lambda:F.linear(x,weight),warmup=12,repeats=repeats)); peak=torch.cuda.max_memory_allocated()/2**20; return t,peak
d1,dp=bench(w,1); m1,mp=bench(masked,1); n1,np=bench(narrow,1); d64,_=bench(w,64,50); n64,_=bench(narrow,64,50)
metrics={"dense_p50_ms":d1["median_ms"],"dense_p95_ms":d1["p95_ms"],"dense_p99_ms":d1["p99_ms"],"masked_p50_ms":m1["median_ms"],"masked_p95_ms":m1["p95_ms"],"masked_p99_ms":m1["p99_ms"],"narrow_p50_ms":n1["median_ms"],"narrow_p95_ms":n1["p95_ms"],"narrow_p99_ms":n1["p99_ms"],"batch64_dense_ms":d64["median_ms"],"batch64_narrow_ms":n64["median_ms"],"batch64_speedup":d64["median_ms"]/n64["median_ms"],"peak_memory_mib":{"dense":dp,"masked":mp,"narrow":np},"samples":80,"raw_samples":{"dense":d1["samples_ms"],"masked":m1["samples_ms"],"narrow":n1["samples_ms"]}}
analysis=(f"At batch 1, dense p50/p99 were {d1['median_ms']:.6f}/{d1['p99_ms']:.6f} ms, the same-shape masked candidate was "
          f"{m1['median_ms']:.6f}/{m1['p99_ms']:.6f} ms, and the physical narrow candidate was {n1['median_ms']:.6f}/"
          f"{n1['p99_ms']:.6f} ms. At batch 64 the narrow/full median ratio was {metrics['batch64_speedup']:.3f}x.")
''',
    26: r'''
torch.manual_seed(SEED)
d,c=20,3; counts=[900,250,90]; centers=torch.randn(c,d,device=DEVICE)*2.0
xs=[]; ys=[]
for cls,count in enumerate(counts): xs.append(centers[cls]+torch.randn(count,d,device=DEVICE)); ys.append(torch.full((count,),cls,device=DEVICE,dtype=torch.long))
train_x=torch.cat([z[:int(len(z)*0.8)] for z in xs]); train_y=torch.cat([z[:int(len(z)*0.8)] for z in ys]); val_x=torch.cat([z[int(len(z)*0.8):] for z in xs]); val_y=torch.cat([z[int(len(z)*0.8):] for z in ys])
model=nn.Sequential(nn.Linear(d,48),nn.ReLU(),nn.Linear(48,c)).to(DEVICE); opt=torch.optim.Adam(model.parameters(),lr=0.025)
for step in range(180):
    idx=torch.randint(0,train_x.shape[0],(96,),device=DEVICE); opt.zero_grad(); loss=F.cross_entropy(model(train_x[idx]),train_y[idx]); loss.backward(); opt.step()
def evaluate(m):
    m.eval()
    with torch.inference_mode(): pred=m(val_x).argmax(1)
    cm=torch.zeros(c,c,device=DEVICE,dtype=torch.int64)
    for t,p in zip(val_y,pred): cm[t,p]+=1
    recall=(cm.diag()/cm.sum(1).clamp_min(1)).float(); return float((pred==val_y).float().mean().item()),recall,cm
dense_acc,dense_rec,dense_cm=evaluate(model); pruned=copy.deepcopy(model); masks={}
with torch.no_grad():
    for name,p in pruned.named_parameters():
        if "weight" in name: masks[p]=magnitude_mask(p,0.70); p.mul_(masks[p])
im_acc,im_rec,im_cm=evaluate(pruned); opt=torch.optim.Adam(pruned.parameters(),lr=0.008)
for step in range(70):
    idx=torch.randint(0,train_x.shape[0],(96,),device=DEVICE); opt.zero_grad(); loss=F.cross_entropy(pruned(train_x[idx]),train_y[idx]); loss.backward(); opt.step()
    with torch.no_grad():
        for p,m in masks.items(): p.mul_(m)
rec_acc,rec_rec,rec_cm=evaluate(pruned); drops=rec_rec-dense_rec; rollback=bool((dense_acc-rec_acc)>0.03 or drops.min().item()<-0.10)
metrics={"dense_accuracy":dense_acc,"pruned_immediate_accuracy":im_acc,"recovered_accuracy":rec_acc,"dense_recall":dense_rec.cpu().tolist(),"immediate_recall":im_rec.cpu().tolist(),"recovered_recall":rec_rec.cpu().tolist(),"worst_recall_drop":float(drops.min().item()),"dense_confusion":dense_cm.cpu().tolist(),"recovered_confusion":rec_cm.cpu().tolist(),"accuracy_drop_threshold":0.03,"recall_drop_threshold":-0.10,"rollback_required":rollback,"final_sparsity":sum((p==0).sum().item() for p in masks)/sum(p.numel() for p in masks)}
analysis=(f"Dense, immediate-pruned, and recovered aggregate accuracies were {dense_acc:.1%}, {im_acc:.1%}, and {rec_acc:.1%}. "
          f"The worst class-recall change after recovery was {metrics['worst_recall_drop']:.1%}; under the frozen 3-point aggregate "
          f"and 10-point recall gates, rollback_required={rollback}.")
''',
    27: r'''
def run(seed):
    torch.manual_seed(seed); config={"algorithm":"global_magnitude","shape":[256,256],"sparsity":0.75,"seed":seed,"dtype":"float32"}
    w=torch.randn(*config["shape"],device=DEVICE); x=torch.randn(16,256,device=DEVICE); mask=magnitude_mask(w,config["sparsity"]); out=F.linear(x,w*mask)
    canonical=json.dumps(config,sort_keys=True,separators=(",",":")); mask_bytes=mask.cpu().contiguous().numpy().tobytes(); out_bytes=out.detach().cpu().contiguous().numpy().tobytes()
    return {"config":config,"config_sha256":hashlib.sha256(canonical.encode()).hexdigest(),"mask_sha256":hashlib.sha256(mask_bytes).hexdigest(),"output_sha256":hashlib.sha256(out_bytes).hexdigest(),"sparsity":zero_fraction(mask)}
a=run(SEED); b=run(SEED); c=run(SEED+1)
metrics={"same_seed_config_match":a["config_sha256"]==b["config_sha256"],"same_seed_mask_match":a["mask_sha256"]==b["mask_sha256"],"same_seed_output_match":a["output_sha256"]==b["output_sha256"],"different_seed_mask_differs":a["mask_sha256"]!=c["mask_sha256"],"sparsity":a["sparsity"],"config_sha256":a["config_sha256"],"mask_sha256":a["mask_sha256"],"output_sha256":a["output_sha256"],"changed_seed_mask_sha256":c["mask_sha256"],"runs":[a,b,c]}
analysis=(f"Identical-seed runs matched config/mask/output hashes={metrics['same_seed_config_match']}/{metrics['same_seed_mask_match']}/"
          f"{metrics['same_seed_output_match']} at {metrics['sparsity']:.1%} sparsity. Changing only the seed changed the mask="
          f"{metrics['different_seed_mask_differs']}. The recorded support digest begins `{a['mask_sha256'][:12]}`.")
''',
    28: r'''
dtype=torch.bfloat16; in_f,out_f=2048,2048
w=torch.randn(out_f,in_f,device=DEVICE,dtype=dtype); masked=w*magnitude_mask(w,0.75); narrow=w[:512]
def payload_sizes(t):
    # NumPy has no native bfloat16 dtype. Reinterpret the two-byte BF16 payload
    # as uint16 so this measures the stored bits without widening to FP32.
    raw=t.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes(); return len(raw),len(gzip.compress(raw,compresslevel=9))
dr,dg=payload_sizes(w); mr,mg=payload_sizes(masked); nr,ng=payload_sizes(narrow)
def med(weight,batch):
    x=torch.randn(batch,in_f,device=DEVICE,dtype=dtype); return timing_summary(cuda_times(lambda:F.linear(x,weight),warmup=10,repeats=40))["median_ms"]
d1,m1,n1=med(w,1),med(masked,1),med(narrow,1); d64,n64=med(w,64),med(narrow,64)
metrics={"dense_raw_bytes":dr,"dense_gzip_bytes":dg,"masked_raw_bytes":mr,"masked_gzip_bytes":mg,"narrow_raw_bytes":nr,"narrow_gzip_bytes":ng,"batch1_dense_ms":d1,"batch1_masked_ms":m1,"batch1_narrow_ms":n1,"batch64_dense_ms":d64,"batch64_narrow_ms":n64,"batch64_speedup":d64/n64,"edge_runtime_measured":False,"edge_energy_measured":False,"edge_decision":"pending_native_measurement","server_decision":"narrow_candidate_supported_by_cuda_shape_probe","platform_matrix":{"edge":{"package_bytes_measured":True,"latency_measured":False,"energy_measured":False},"server":{"batch1_measured":True,"batch64_measured":True,"operator":"dense_pytorch_linear"}}}
analysis=(f"Dense/masked/narrow gzip payloads were {dg:,}/{mg:,}/{ng:,} bytes. On RTX 5090, batch-1 medians were "
          f"{d1:.6f}/{m1:.6f}/{n1:.6f} ms and the batch-64 physical-width ratio was {metrics['batch64_speedup']:.3f}x. "
          "Edge latency and energy remain unmeasured, so the edge decision is explicitly pending.")
''',
}
