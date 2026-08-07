# Lesson 05 — Diagnosing FP16 Overflow and Gradient Scaling Failures

> **Puzzle:** When loss becomes NaN, how do we distinguish forward overflow, backward overflow, and gradient underflow?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A final NaN is the last symptom in a chain, not the diagnosis. FP16 can overflow during
forward, overflow after loss scaling during backward, or silently round tiny gradients
to zero. Each failure calls for a different response, so the first bad tensor must be
located before changing the scaler.

## Predict before reading the result

1. Predict which combinations of gradient magnitude and loss scale become zero, finite, or infinite in FP16.
2. Explain why loss scaling can rescue underflow but cannot repair a forward activation that is already Inf.
3. Choose probe locations that distinguish forward, scaled-backward, unscaled-gradient, and parameter corruption.

## 1. Start from concrete tensors and state

Diagnose four checkpoints: forward outputs, scaled loss/gradients, unscaled gradients,
and post-step parameters. A final NaN has already discarded the location of the first
failure.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Overflow creates Inf before it becomes NaN in later arithmetic. |
| 2 | Underflow silently rounds small gradients to zero. |
| 3 | Loss scaling moves gradients into a representable interval but cannot repair an already-overflowed forward pass. |

## 2. Derive the mechanism

FP16 normal values end near `6.55e4`; very small values enter a sparse subnormal region
and can become zero. Loss scaling shifts gradient magnitudes upward during storage, but
unscaling must happen before clipping and parameter updates.

With loss scale S, an exact gradient g is represented during backward as `Sg`. If g is
smaller than the FP16 subnormal range, choosing a moderate S can move it onto the
representable grid; unscale later restores its mathematical magnitude in a wider type.
If `Sg > 65504`, the scaled gradient becomes Inf. And if a forward value already
exceeded 65504, multiplying the loss later cannot reconstruct the discarded information.

This creates a feasible interval for S: large enough that important small gradients
survive, but small enough that the largest scaled gradient remains finite. Dynamic
scaling searches that interval through observed overflow. It does not guarantee that
every tiny gradient is preserved or that the forward pass is stable.

## 3. Translate the theory into an experiment

**Experiment:** Sweep synthetic gradient magnitudes and loss scales in FP16 on CUDA, counting finite, infinite, and zero gradient values.

| Experimental role | Frozen definition |
|---|---|
| Baseline | FP16 casting of four gradient magnitudes with scale 1 |
| Candidate | the same magnitudes multiplied by scales 256 and 65536 |
| Held constant | tensor size, dtype, GPU, values within each magnitude group |
| Measurements | zero fraction, finite fraction, Inf fraction, plus a separate forward-overflow probe |
| Evidence label | `pytorch-gpu` |

The CUDA sweep crosses both tiny and large magnitudes at several scales and records zero
and Inf fractions, making the failure stage observable.

### Code walk-through

The notebook sweeps a Cartesian product rather than waiting for a random training
failure. For every magnitude/scale pair it casts the scaled value to FP16 and counts
zero, finite, and infinite entries. A separate `1e5` forward probe establishes that some
damage can occur before backward begins.

Because all elements in a row share one magnitude, fractions jump cleanly between zero,
finite, and Inf. A real model would produce a distribution, but the synthetic grid makes
the representability boundaries easy to see and debug.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| 1e-8, scale 1: zero fraction | 100.0000% |
| 1e-8, scale 256: zero fraction | 0.0000% |
| 1, scale 65536: Inf fraction | 100.0000% |
| 1000, scale 256: Inf fraction | 100.0000% |
| Forward 1e5 overflowed | yes |

### What the numbers mean

At magnitude `1e-8`, scale 1 rounded every value to zero, while scales 256 and 65536
made all entries finite and non-zero. At magnitude 1, scale 65536 overflowed every
value. At magnitude 1000, scale 256 was already too large. The independent forward test
confirmed that FP16 `1e5` was non-finite.

The same tool—larger scale—therefore fixes one row and breaks another. That is the
central reason GradScaler adapts and skips unsafe optimizer steps. It is also why a
scaler change is the wrong fix for forward overflow.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> Place finiteness and zero-rate probes at forward outputs, scaled gradients, unscaled gradients, and parameters before changing the scaler policy.

### Acceptance and rollback gate

Log finite/Inf/zero fractions and the current scale. If the forward pass is already
non-finite, change the operation or dtype; if only scaled gradients overflow, adjust
scale policy.

### How this conclusion can fail

Looking only at `torch.isfinite(loss)` misses underflow because zeros are finite.
Looking only after unscale can hide where overflow began. Logging every tensor is too
expensive, so production diagnosis usually places targeted hooks at loss, selected
activations, scaled gradients, unscaled gradients, and parameters, then narrows the
search.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map FP16 casting of four gradient magnitudes with
scale 1 and the same magnitudes multiplied by scales 256 and 65536 back to the
derivation. Verify the printed environment, then check that tensor size, dtype, GPU,
values within each magnitude group stayed fixed. Read zero fraction, finite fraction,
Inf fraction, plus a separate forward-overflow probe before applying the acceptance
gate; the artifact-writing cell retains the complete structured result from the recorded
run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/05-fp16-overflow/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Instrument a small FP16 network with hooks that report min/max, zero fraction, and
finiteness at the four stages. Inject an activation spike and a tiny-gradient layer
separately. Verify that lowering the scale helps the first backward-overflow case,
raising it helps the underflow case, and neither repairs the injected forward Inf.

## Evidence boundary

The measured tensors and operations ran on CUDA through PyTorch. The result does not
name a separate production backend unless an operator trace identifies it.

The checked-in observation belongs to Lesson 05's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
