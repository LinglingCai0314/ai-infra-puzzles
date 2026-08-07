# Lesson 14 — bitsandbytes 4-Bit Loading: NF4, Compute Dtype, and Nested Quantization

> **Puzzle:** Does `load_in_4bit=True` specify how the layer computes?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

`load_in_4bit=True` is not a complete numerical specification. A bitsandbytes
configuration also selects a codebook such as NF4, a compute dtype for dequantized
matrix operations, and optionally nested quantization for metadata. The loaded module
class and backend availability determine whether those settings became a real operator
or stayed configuration text.

## Predict before reading the result

1. Distinguish quantization codebook, packed storage dtype, compute dtype, and nested quantization.
2. Predict whether NF4 or uniform INT4 gives lower RMSE for normally distributed weights.
3. State what evidence would be required to label the result a native bitsandbytes run.

## 1. Start from concrete tensors and state

A bitsandbytes 4-bit configuration contains at least storage codebook (`NF4` or FP4),
compute dtype, optional double/nested quantization, and the module/backend that consumes
it.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Storage type, quantization codebook, and compute dtype are separate choices. |
| 2 | Nested quantization compresses quantization metadata; it does not turn activation compute into two-bit arithmetic. |
| 3 | Package presence and device support must be checked before claiming a bitsandbytes run. |

## 2. Derive the mechanism

NF4 assigns its 16 codes non-uniformly rather than at equal integer spacing. During a
linear operation the packed codes are dequantized or consumed by a fused path while
activations use the configured compute dtype.

Uniform INT4 places evenly spaced reconstruction levels over a selected range. NF4
instead uses a non-uniform codebook whose levels allocate more resolution where a normal
distribution has more probability mass. A stored code selects one level; matrix
multiplication still needs dequantization/scaling and a floating-point compute path.
Double or nested quantization reduces the cost of quantization constants, not the
activation arithmetic to two bits.

Codebook quality depends on the weight distribution and normalization rule. NF4 can
lower average error for bell-shaped weights while producing a larger worst-case error
near tails than a range-fitted uniform grid. The deployment decision also includes
kernel support and compute dtype stability.

## 3. Translate the theory into an experiment

**Experiment:** Compare a reference NF4 codebook with uniform INT4 on normally distributed weights and probe whether bitsandbytes is installed.

| Experimental role | Frozen definition |
|---|---|
| Baseline | uniform symmetric INT4 reconstruction of normally distributed weights |
| Candidate | reference NF4 codebook reconstruction of the same weights |
| Held constant | weight tensor, normalization, number of codes, error reference, seed |
| Measurements | RMSE/MAE/cosine/max error and bitsandbytes installation probe |
| Evidence label | `numerical-model` |

The lab isolates codebook reconstruction and separately records package presence, so a
numerical NF4 result cannot masquerade as bitsandbytes execution.

### Code walk-through

The notebook maps the same random weights through a reference NF4 codebook and a uniform
INT4 quantizer, then compares both with the original tensor. It separately checks
whether bitsandbytes is importable. Keeping these branches separate prevents a numerical
codebook experiment from masquerading as a library benchmark.

No transformers model is loaded, no `Linear4bit` module is instantiated, and no
bitsandbytes kernel is timed in the checked-in environment. The evidence label therefore
remains `numerical-model`.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| NF4 RMSE | 0.127836 |
| Uniform INT4 RMSE | 0.142396 |
| NF4 max error | 0.719360 |
| Uniform INT4 max error | 0.339060 |
| bitsandbytes installed | no |

### What the numbers mean

NF4 achieved RMSE 0.127836 and MAE 0.109566, lower than uniform INT4 at RMSE 0.142396
and MAE 0.122676 for this normal tensor. Uniform INT4 had a smaller max error, 0.339059
versus NF4's 0.719360, showing that average and tail objectives can disagree. The
environment probe reported `bitsandbytes_installed=false`.

The result supports the distribution-aware codebook intuition only. It says nothing
about native layer memory, throughput, nested-quant overhead, or task quality on this
RTX stack.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Record quantization type, compute dtype, nested-quant setting, and actual module class together.

### Acceptance and rollback gate

Capture `BitsAndBytesConfig`, package/CUDA compatibility, actual module class, storage
bytes, operator evidence, output regression, and timing.

### How this conclusion can fail

A reference codebook can differ from library normalization, block size, packing, and
scale dtype. Claiming bitsandbytes speed from it would be false. Another trap is
choosing NF4 from average RMSE while a downstream layer is sensitive to rare tail
errors.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map uniform symmetric INT4 reconstruction of normally
distributed weights and reference NF4 codebook reconstruction of the same weights back
to the derivation. Verify the printed environment, then check that weight tensor,
normalization, number of codes, error reference, seed stayed fixed. Read
RMSE/MAE/cosine/max error and bitsandbytes installation probe before applying the
acceptance gate; the artifact-writing cell retains the complete structured result from
the recorded run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/14-bitsandbytes-4bit/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Install a release compatible with the current PyTorch/CUDA stack, load one `Linear4bit`
layer, and record its actual module, storage tensors, compute dtype, output error, and
operator trace. Repeat with and without nested quantization and then with a small
model-quality suite.

## Evidence boundary

The CUDA numerical experiment isolates an algorithmic mechanism. It is not the paper's
complete implementation and does not establish a production kernel speedup.

The checked-in observation belongs to Lesson 14's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/main/quantization/bitsandbytes)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [bitsandbytes documentation](https://huggingface.co/docs/bitsandbytes/main/en/index)
