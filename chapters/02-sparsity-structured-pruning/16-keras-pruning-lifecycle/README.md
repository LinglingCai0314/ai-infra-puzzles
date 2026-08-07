# Lesson 16 — TensorFlow MOT and the Keras Pruning/Export Lifecycle

> **Puzzle:** Why can training-time Keras sparsity fail to reduce a deployable TFLite artifact?

[← Chapter 02](../README.md) · [Project homepage](../../../README.md) · [Executed notebook](lab.ipynb) · [RTX 5090 result](artifacts/rtx5090-result.json)

## Why this puzzle matters

TensorFlow Model Optimization wraps Keras layers with masks, thresholds, and
pruning-step state. Export requires updating the pruning step during training, reaching
the schedule, stripping wrappers, converting, and checking the final representation.
This environment may not provide TensorFlow, so availability is an explicit result
rather than an excuse for invented output.

For **TensorFlow MOT and the Keras Pruning/Export Lifecycle**, the engineering question
is not whether a definition can be repeated; it is whether the following claim survives
a controlled GPU test: *Why can training-time Keras sparsity fail to reduce a deployable
TFLite artifact?* The lab therefore changes the mechanism described below, retains its
measured state, and names the evidence that would still be needed for deployment.

## Predict before reading the result

1. Predict the target sparsity before, during, and after a polynomial window.
2. Predict what `strip_pruning` removes and what it retains.
3. List the artifacts required before claiming a TFLite size benefit.

Before opening Lesson 16's retained output, answer the first prompt— *Predict the target
sparsity before, during, and after a polynomial window.*—and write one observation that
would falsify the answer. If the result is already visible, hide it and make the
commitment first; otherwise this becomes post-hoc explanation rather than a pruning
experiment.

## 1. Start from concrete tensors and state

The lab records TensorFlow and TFMOT package availability, evaluates the polynomial
schedule formula on CUDA, constructs the required lifecycle state machine, and runs a
tiny native strip/export probe only when dependencies exist.

### Three reasoning anchors

| # | Lesson-specific claim to keep visible |
|---:|---|
| 1 | Schedule progress depends on optimizer steps and update callbacks. |
| 2 | Stripping removes training wrappers, not necessarily dense storage. |
| 3 | TensorFlow/TFMOT availability is part of reproducible backend evidence. |

Lesson 16 tracks three layers through TensorFlow MOT and the Keras Pruning/Export
Lifecycle: *value state* says which entries are zero, *shape state* says which axes
physically changed, and *execution state* says which operator actually ran. The anchors
above identify where this lesson's claim lives, so a zero count cannot silently turn
into a latency claim.

## 2. Derive the mechanism

Polynomial decay controls target sparsity by optimizer step. Wrappers contain
training-only variables and callbacks update their step. `strip_pruning` removes
wrappers while retaining sparse weights; it does not guarantee a smaller uncompressed
format or a sparse-accelerated runtime. TFLite conversion and optional compression are
separate gates. A native experiment must therefore preserve versions, wrapper state,
stripped model, converted bytes, and output parity.

The inspectable invariant for **TensorFlow MOT and the Keras Pruning/Export Lifecycle**
is tested by: Probe the Keras pruning stack and execute the schedule/lifecycle contract
without fabricating a missing native backend. Its purpose is to prevent the specific
category error behind this puzzle. An algorithmic change, a stored representation, and a
runtime observation remain separate until the candidate and measurements below connect
them.

## 3. Translate the theory into an experiment

**Experiment:** Probe the Keras pruning stack and execute the schedule/lifecycle contract without fabricating a missing native backend.

| Experimental role | Frozen definition |
|---|---|
| Baseline | CUDA-evaluated polynomial schedule and an unstripped lifecycle state |
| Candidate | native TFMOT wrapper/strip probe when available, otherwise a bounded compatibility result |
| Held constant | environment, schedule endpoints, steps, target rate, lifecycle transitions, and seed |
| Measurements | package availability, schedule values, native probe status, and lifecycle gates |
| Evidence label | `compatibility-probe` |

This Lesson 16 comparison is deliberately small enough to rerun on a reader's GPU. Its
control is **environment, schedule endpoints, steps, target rate, lifecycle transitions,
and seed**. That frozen condition preserves the dependency or runtime boundary at issue;
the small scale limits transfer to larger models but does not permit the baseline and
candidate to answer different questions.

### Code walk-through

The notebook uses the published polynomial form to produce deterministic target rates
and checks that strip/export cannot be marked complete before training and wrapper
removal. Conditional imports keep missing packages in a structured field. No Keras
latency or TFLite size number is synthesized when the stack is absent.

For **TensorFlow MOT and the Keras Pruning/Export Lifecycle**, the environment cell
asserts CUDA and fixes a lesson-specific seed. The experiment cell implements native
TFMOT wrapper/strip probe when available, otherwise a bounded compatibility result and
records package availability, schedule values, native probe status, and lifecycle gates.
The artifact cell serializes those same fields. Only optional-backend import or API
failures become compatibility evidence; an error in the core comparison still fails the
notebook.

## 4. Read the checked-in RTX 5090 result

**Recorded environment:** NVIDIA GeForce RTX 5090; compute capability 12.0; PyTorch 2.12.0; CUDA runtime 13.0.

| Measured field | Checked-in value |
|---|---:|
| TensorFlow available | no |
| TFMOT available | no |
| Mid-schedule sparsity | 70.00% |
| Final schedule sparsity | 80.00% |
| Native probe executed | no |
| Lifecycle ready | no |

### What the numbers mean

The cubic schedule moved from 0.0% at step 10 to 70.0% at step 30 and 80.0% at step 50.
TensorFlow/TFMOT availability was False/False, so native wrapper stripping
executed=False. Missing native stages remain false rather than inferred.

Lesson 16's full [`rtx5090-result.json`](artifacts/rtx5090-result.json) retains the
arrays or diagnostic fields behind the compact selection above. For this lesson, the
interpretation is bounded by **compatibility-probe** evidence; the printed notebook
payload and the JSON were produced by the same execution.

## 5. Solve the puzzle and make a decision

> Keras pruning is a versioned train-strip-convert lifecycle; a missing native stack must remain visibly unexecuted.

### Acceptance and rollback gate

Accept a Keras pruning delivery only after native training, `UpdatePruningStep`,
`strip_pruning`, TFLite conversion, output parity, and target-device measurement all
pass.

The gate for **TensorFlow MOT and the Keras Pruning/Export Lifecycle** is stricter than
“the code ran” because it binds this lesson's tensor or model identity, quality
tolerance, workload, runtime path, and rollback evidence. A missing optional package can
settle a compatibility question, but it cannot satisfy the native-performance decision
stated above.

### How this conclusion can fail

A numerical schedule is not a TensorFlow execution. Installing TensorFlow without a
compatible GPU stack may move compute to CPU. A stripped model can retain dense tensors
with zeros, and zip compression can be confused with runtime memory savings.

## 6. Follow the theory inside the notebook

In Lesson 16's [`lab.ipynb`](lab.ipynb), first identify **CUDA-evaluated polynomial
schedule and an unstripped lifecycle state** and **native TFMOT wrapper/strip probe when
available, otherwise a bounded compatibility result** without running them. Next inspect
the dimensions or lifecycle state that implements the derivation. After **Run All**,
verify the RTX 5090 environment and the frozen fields before reconciling the result
table with the artifact.

The reader loop for **TensorFlow MOT and the Keras Pruning/Export Lifecycle** is
**predict → execute → inspect → explain → decide**. Transferring its final number to
another architecture, workload shape, or backend requires a new run because those
variables sit outside this lesson's evidence.

## Reproduce

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch jupyterlab nbclient nbformat
jupyter lab chapters/02-sparsity-structured-pruning/16-keras-pruning-lifecycle/lab.ipynb
```

This lesson's optional/native backend path requires:

```bash
pip install tensorflow tensorflow-model-optimization
```

To reproduce **TensorFlow MOT and the Keras Pruning/Export Lifecycle**, use a PyTorch
build compiled for the target GPU and select `Run All`. Compare the measurements in the
frozen protocol with the checked-in artifact. If this lesson touches an optional
toolchain, install that named backend before claiming native execution; otherwise only
the compatibility fields are valid.

## Extend the experiment

Run the notebook in a pinned TensorFlow/TFMOT environment, retain wrapper and stripped
summaries, convert to TFLite, and benchmark the exact target device.

For Lesson 16, the proposed extension is a new evidence layer rather than a replacement
for the checked-in control. Add one of its requested dimensions at a time and retain
this mechanism run, so a quality, export, operator, or service-level reversal can be
localized.

## Evidence boundary

The notebook records real package/API availability and preserves the native success or
failure state. Missing backend execution remains unmeasured.

The checked-in **TensorFlow MOT and the Keras Pruning/Export Lifecycle** observation
belongs to Lesson 16's RTX 5090 environment, shapes, seed, and protocol. It does not
establish the unmeasured task quality or platform properties named in the failure
analysis. This independently written tutorial uses the study topic as a question,
without redistributing source HTML, model weights, private paths, or infrastructure.

## References

- [TensorFlow Model Optimization pruning guide](https://www.tensorflow.org/model_optimization/guide/pruning)
- [TensorFlow strip_pruning API](https://www.tensorflow.org/model_optimization/api_docs/python/tfmot/sparsity/keras/strip_pruning)
