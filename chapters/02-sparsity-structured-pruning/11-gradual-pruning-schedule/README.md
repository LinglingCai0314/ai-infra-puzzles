# Lesson 11 — Gradual Pruning Schedules and Recovery Training

> **Puzzle:** What does a polynomial sparsity schedule control that a final target does not?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

A target sparsity says where training should end; a schedule says how abruptly the
feasible parameter set changes. With a fixed 15% recovery budget, pruning events compete
with optimization steps, so the begin step, end step, update frequency, and
learning-rate trajectory become part of the result.

For **Gradual Pruning Schedules and Recovery Training**, the engineering question is not
whether a definition can be repeated; it is whether the following claim survives a
controlled GPU test: *What does a polynomial sparsity schedule control that a final
target does not?* The lab therefore changes the mechanism described below, retains its
measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Compute the target sparsity halfway through a cubic schedule.
2. Predict which route has the largest immediate loss shock.
3. Name the schedule fields needed to reproduce a recovery trajectory.

Before opening Lesson 11's retained output, answer the first prompt— *Compute the target
sparsity halfway through a cubic schedule.*—and write one observation that would falsify
the answer. If the result is already visible, hide it and make the commitment first;
otherwise this becomes post-hoc explanation rather than a pruning experiment.

## 1. Start from concrete tensors and state

The experiment uses one frozen dense classifier, an immediate target mask, a polynomial
target function, periodic magnitude-mask updates, identical recovery updates, and a
held-out accuracy trajectory.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Final sparsity does not identify the support trajectory. |
| 2 | Pruning frequency trades adaptation time against ranking refresh. |
| 3 | Learning-rate and sparsity schedules interact. |

Lesson 11 tracks three layers through Gradual Pruning Schedules and Recovery Training:
*value state* says which entries are zero, *shape state* says which axes physically
changed, and *execution state* says which operator actually ran. The anchors above
identify where this lesson's claim lives, so a zero count cannot silently turn into a
latency claim.

## 2. Derive the mechanism

A common cubic schedule is `s(t)=s_f+(s_i-s_f)(1-(t-t0)/(t1-t0))^3` within the pruning
window. Early updates remove few weights and later changes taper as the target is
approached. The schedule does not guarantee recovery; it bounds the size and timing of
support shocks. Reapplying a newly ranked mask can remove previously useful weights,
while a fixed mask only trains survivors. Those policy choices must be frozen.

The inspectable invariant for **Gradual Pruning Schedules and Recovery Training** is
tested by: Compare one-shot and cubic gradual schedules under one initialization and
equal optimizer-step budget. Its purpose is to prevent the specific category error
behind this puzzle. An algorithmic change, a stored representation, and a runtime
observation remain separate until the candidate and measurements below connect them.

## 3. Translate the theory into an experiment

**Experiment:** Compare one-shot and cubic gradual schedules under one initialization and equal optimizer-step budget.

| Experimental role | Frozen definition |
|---|---|
| Baseline | one-shot jump to 80% sparsity at the first recovery step |
| Candidate | cubic updates from 0% to 80% over the same recovery window |
| Held constant | dense checkpoint, data order, optimizer, learning rate, total steps, target rate, and mask rule |
| Measurements | target/actual sparsity trajectory, immediate loss, final accuracy, and best accuracy |
| Evidence label | `pytorch-gpu` |

This Lesson 11 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **dense checkpoint, data order, optimizer, learning rate, total steps, target
rate, and mask rule**. That frozen condition preserves the dependency or runtime
boundary at issue; the small scale limits transfer to larger models but does not permit
the baseline and candidate to answer different questions.

### Code walk-through

The notebook records a row at each schedule update rather than only the endpoint. Both
routes clone the same baseline and execute the same number of optimizer steps. Mask
refresh and reapplication are explicit, making it possible to distinguish a schedule
failure from silent weight regrowth.

For **Gradual Pruning Schedules and Recovery Training**, the environment cell asserts
CUDA and fixes a lesson-specific seed. The experiment cell implements cubic updates from
0% to 80% over the same recovery window and records target/actual sparsity trajectory,
immediate loss, final accuracy, and best accuracy. The artifact cell serializes those
same fields. Only optional-backend import or API failures become compatibility evidence;
an error in the core comparison still fails the notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| Dense accuracy | 88.67% |
| One-shot final accuracy | 72.67% |
| Gradual final accuracy | 74.00% |
| Target sparsity | 80.00% |
| Schedule updates | 11 |

### What the numbers mean

Both routes used 40 optimizer updates and finished near 80% sparsity. One-shot
validation accuracy ended at 72.7%, while the cubic route ended at 74.0% after 11
recorded mask updates. The retained trajectories show when each support shock occurred.

Lesson 11's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **pytorch-gpu** evidence; the printed notebook payload and
the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Gradual pruning controls the sequence of support changes; its endpoint is insufficient to reproduce or judge recovery.

### Acceptance and rollback gate

Accept a schedule only when its complete trajectory, final support, quality recovery,
and training budget are recorded and meet the target card.

The gate for **Gradual Pruning Schedules and Recovery Training** is stricter than “the
code ran” because it binds this lesson's tensor or model identity, quality tolerance,
workload, runtime path, and rollback evidence. A missing optional package can settle a
compatibility question, but it cannot satisfy the native-performance decision stated
above.

### How this conclusion can fail

A schedule can appear better simply because it prunes later and spends more steps near
the dense model. Comparing endpoints without integrating the actual sparsity trajectory
hides that advantage. Small synthetic data also makes recovery unusually cheap.

## 6. Follow the theory inside the notebook

In Lesson 11's [`lab.ipynb`](lab.ipynb), first identify **one-shot jump to 80% sparsity
at the first recovery step** and **cubic updates from 0% to 80% over the same recovery
window** without running them. Next inspect the dimensions or lifecycle state that
implements the derivation. After **Run All**, verify the RTX 5090 environment and the
frozen fields before reconciling the result table with the artifact.

The reader loop for **Gradual Pruning Schedules and Recovery Training** is **predict →
execute → inspect → explain → decide**. Transferring its final number to another
architecture, workload shape, or backend requires a new run because those variables sit
outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/11-gradual-pruning-schedule/lab.ipynb
```

To reproduce **Gradual Pruning Schedules and Recovery Training**, use a PyTorch build
compiled for the target GPU and select `Run All`. Compare the measurements in the frozen
protocol with the checked-in artifact. If this lesson touches an optional toolchain,
install that named backend before claiming native execution; otherwise only the
compatibility fields are valid.

## Extend the experiment

Match candidates by area under the sparsity-time curve, sweep update frequency and
learning-rate restarts, and repeat on a downstream task metric rather than toy accuracy
alone.

For Lesson 11, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The tensors and operators executed on CUDA through PyTorch. Native sparse-kernel
identity is not inferred unless a trace or backend artifact names it.

The checked-in **Gradual Pruning Schedules and Recovery Training** observation belongs
to Lesson 11's RTX 5090 environment, shapes, seed, and protocol. It does not establish
the unmeasured task quality or platform properties named in the failure analysis. This
independently written tutorial uses the study topic as a question, without
redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [To Prune, or Not to Prune](https://arxiv.org/abs/1710.01878)
- [TensorFlow Model Optimization pruning guide](https://www.tensorflow.org/model_optimization/guide/pruning)
