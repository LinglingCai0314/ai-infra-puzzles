# Lesson 03 — PyTorch AMP: autocast and GradScaler

> **Puzzle:** Can mixed-precision training be reduced to wrapping the forward pass in autocast?

[← Chapter 01](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

Mixed-precision training is a feedback system. Autocast chooses operation dtypes during
the forward pass, gradient scaling changes the numerical range seen by backward, and the
optimizer must only step after gradients have been checked and unscaled. Demonstrating
one BF16 activation therefore proves much less than demonstrating a complete, finite
parameter update.

## Predict before reading the result

1. Predict the dtype of model parameters and forward outputs inside BF16 autocast.
2. Predict whether GradScaler's scale should change when every gradient stays finite.
3. Name the observation that proves an optimizer update occurred rather than only a forward pass.

## 1. Start from concrete tensors and state

The AMP loop contains FP32 parameters and optimizer state, autocast-selected forward
activations, gradients, a scalar loss scale, and an optimizer update. These objects do
not all share one dtype or lifetime.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Autocast selects lower precision per eligible operation; it does not permanently convert every tensor. |
| 2 | GradScaler changes loss magnitude before backward, unscales gradients before the optimizer step, and adapts its scale. |
| 3 | The optimizer state and usually the master parameters remain higher precision. |

## 2. Derive the mechanism

If `g` is the true gradient and `S` is the loss scale, backward first produces `S·g`;
unscale restores `g` before clipping or the optimizer step. `GradScaler` skips the step
when non-finite gradients are found and adapts `S`. Autocast independently chooses
eligible forward-operation dtypes.

Let the unscaled loss be `L` and the current scale be `S`. Backward differentiates
`S·L`, producing scaled gradients `S·g`. Before the optimizer step, GradScaler divides
by S and checks for Inf/NaN. If the check passes, the optimizer consumes g; if it fails,
the step is skipped and the scale policy reacts. The ordering is semantic: clipping or
inspecting gradients before unscale changes their meaning.

Autocast is a dispatch policy, not a recursive call to `.to(bfloat16)` on the entire
model. Eligible compute-heavy operations may emit BF16 while parameters and optimizer
state remain FP32. BF16 does not usually need scaling for range in the way FP16 does,
but exercising the full scaler API is still useful because the lesson is about the
control loop and its evidence, not a single recommended dtype recipe.

## 3. Translate the theory into an experiment

**Experiment:** Train a small CUDA MLP with BF16 autocast and GradScaler while recording loss, parameter dtype, output dtype, gradient finiteness, and scale history.

| Experimental role | Frozen definition |
|---|---|
| Baseline | FP32 parameters and optimizer state outside autocast |
| Candidate | BF16 autocast forward wrapped in a complete scale/backward/step/update loop |
| Held constant | same MLP, batch, targets, optimizer, seed, and six training steps |
| Measurements | loss history, output dtype, parameter dtype, gradient finiteness, scaler value |
| Evidence label | `pytorch-gpu` |

The notebook prints parameter and output dtypes, runs the complete update loop, and
records gradient finiteness rather than stopping after one autocast forward.

### Code walk-through

The environment cell verifies CUDA and fixes the random seed. The experiment constructs
one small MLP, keeps its parameters in FP32, enters the autocast context only for
forward and loss computation, and then executes the scaler sequence. Every step records
five pieces of state so the notebook can distinguish dispatch, numerical health, and
optimization progress.

A decreasing toy loss is not a model-quality claim; it is a control-flow check. The
stronger invariants are that every recorded gradient is finite, the output is BF16 under
autocast, parameters remain FP32, and the loop reaches optimizer updates without an
error output.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Initial loss | 1.037629 |
| Final loss | 0.596515 |
| Autocast output dtype | torch.bfloat16 |
| All recorded gradients finite | yes |
| Parameter dtype | torch.float32 |
| Final scaler value | 65536.000000 |

### What the numbers mean

The six saved steps reduced loss from 1.0376294 to 0.5965154. Every output was
`torch.bfloat16`, every gradient check returned true, and parameters remained
`torch.float32`. The scale stayed at 65536 because no non-finite event forced the policy
to back off during this short run.

Taken together, those fields establish a functioning mixed-precision loop on this
PyTorch/CUDA stack. They do not establish faster training, convergence parity on a real
dataset, or the best scale-growth policy. Those require longer runs with repeated timing
and a frozen quality target.

Open [`artifacts/rtx5090-result.json`](artifacts/rtx5090-result.json) when you need
every repeated sample or a field not selected for the tutorial table.

## 5. Solve the puzzle and make a decision

> AMP is a control loop across forward, backward, unscale, step, and update—not a global dtype switch.

### Acceptance and rollback gate

Verify the order `zero_grad -> autocast forward -> scale(loss).backward -> unscale/step
-> update`, record finite gradients and scale history, and keep the loss objective
identical to the FP32 baseline.

### How this conclusion can fail

Common failures include calling `optimizer.step()` directly on scaled gradients,
clipping before `unscale_`, moving master parameters to FP16, or judging success only
from the forward dtype. A finite loss can coexist with zeroed small gradients, and a
skipped optimizer step can be invisible unless the scale and parameter update are
inspected.

## 6. Follow the theory inside the notebook

In [`lab.ipynb`](lab.ipynb), first map FP32 parameters and optimizer state outside
autocast and BF16 autocast forward wrapped in a complete scale/backward/step/update loop
back to the derivation. Verify the printed environment, then check that same MLP, batch,
targets, optimizer, seed, and six training steps stayed fixed. Read loss history, output
dtype, parameter dtype, gradient finiteness, scaler value before applying the acceptance
gate; the artifact-writing cell retains the complete structured result from the recorded
run.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-notebook.txt
jupyter lab chapters/01-mixed-precision-int4/03-pytorch-amp/lab.ipynb
```

Use **Run All** and compare the regenerated result with the checked-in artifact.

## Extend the experiment

Add a deliberately overflowing step and verify that GradScaler skips the update and
changes its scale. Then time FP32, FP16+scaler, and BF16 autocast over a longer MLP
while comparing the same validation loss trajectory. Preserve the exact optimizer, seed,
and batch order so numerical and throughput decisions are not confounded.

## Evidence boundary

The measured tensors and operations ran on CUDA through PyTorch. The result does not
name a separate production backend unless an operator trace identifies it.

The checked-in observation belongs to Lesson 03's recorded RTX 5090 environment and
controlled variables. It can explain this mechanism without establishing unmeasured
full-model quality or online-service performance. The tutorial is independently written
and does not redistribute course source files, model weights, or private infrastructure.

## References

- [PyTorch AMP documentation](https://docs.pytorch.org/docs/stable/amp.html)
- [PyTorch AMP examples](https://docs.pytorch.org/docs/stable/notes/amp_examples.html)
- [PyTorch numerical accuracy notes](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
