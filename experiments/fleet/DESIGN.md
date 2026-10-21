# Fleet trainer — design

**Goal.** Run hundreds of small, independent training runs at once on a GPU, so that a whole study (all arms × all seeds) finishes in minutes to an hour instead of overnight.

**Code:** [`src/topology_design/fleet.py`](../../src/topology_design/fleet.py). **Driver:** [`run_fleet.py`](run_fleet.py). **Tests:** [`tests/test_fleet.py`](../../tests/test_fleet.py).

## Why this shape

Our runs are tiny: about 0.4M parameters, full batch over about 1,400 examples. One run cannot occupy a GPU. Many runs stacked along a leading "run" dimension can: every matrix product becomes one batched product over all runs.

The previous batched trainer (`grokking_batched.py`) did this, but with two limitations:
- it batched only runs with the *same graph*;
- it measured each run on the CPU, one at a time. At large batch sizes that measurement loop dominates.

## What can vary within one batch

Runs are fully independent: own parameters, AdamW state, split, labels and hyperparameters. Within a batch they may differ in:
- **graph** (parent vector), via a per-run gather of the parent state. When all runs share a parent, it is a plain slice;
- **learning rate, weight decay**, and per-block lr and wd multipliers. These cover Phase 5B's per-branch time constants;
- **block output multipliers** (alphas), **beta**, **out_scale** and **init scale** (richness sweeps; Phase 5B scale matching);
- **seeds**, data splits and **labels** (the rule-plus-exceptions task, where each run has its own exceptions).

They must share: p, d, hidden, the number of blocks L, the task family, the training-set size, the horizon, the eval schedule and the branch definitions. These form `fleet.struct_key`, and the driver packs runs by it automatically.

## Exactness

- **Initialization:** each run's initial weights are copied from `grokking.build_model`, so they are bit-identical to the single-run trainer and to Studies 1–4.
- **Optimizer:** `FleetAdamW` has `torch.optim.AdamW` semantics (betas 0.9 and 0.98, eps 1e-8, decoupled decay), with per-run, per-tensor lr and wd. It is tested against torch AdamW with parameter groups.
- **Floating point:** batched and single matrix products sum in a different order. Parameters agree to about 1e-6 for the first steps and drift to about 3e-5 by step 25. Chaotic phases (collapses) therefore diverge between devices. Device validation is statistical (`validate_gpu.py`), not trajectory-exact.
- **CUDA:** TF32 is off by default, for exactness. `--allow-tf32` is faster on Ampere and later GPUs.

## Measurement on the device

Every eval runs, for all runs at once, a forward pass over the full input grid, processed in chunks of grid points (default 1,024) to bound memory regardless of batch size. It computes:
- loss and accuracy on the train, test, rule-train and exception sets;
- **per component** (embedding path and each block's direct write), the leak-free supports:
  - gen = mean test-label centered logit;
  - mem = mean train-label centered logit minus gen;
  - exception support;
  - each component's logit RMS (needed for Phase 5B scale matching);
- the test-only rule fraction, as a single pass using per-key sums;
- **ablations:** each block alone removed, plus each configured branch and all branches (accuracy, gen, mem);
- per-block gradient norms and weight norms.

Only small per-run arrays are copied to the host.

**Per-eval cost:** 1 + L + (branches + 1) grid forward passes. All of these are tested equal to `grokking.path_measure` and `grokking.measure` on the same weights.

## No per-step synchronisation

- Divergence (a non-finite loss) and early stopping are tracked in device tensors, and read only at evals.
- A diverged or finished run is frozen: its lr is set to 0, so it gets no step and no decay. Its loss is excluded, so it cannot contaminate other runs. Tested.

## Tasks

| task | definition |
|---|---|
| `modadd` | $(a+b) \bmod p$, with the same splits as Studies 1–4 |
| `modadd_exc` | as `modadd`, but a fixed random fraction `exc_frac` of the training pairs get arbitrary labels that are never the rule's. These exceptions can only be memorized. |

`modadd` is exactly the `modadd_exc` setting at `exc_frac = 0`. The group-product composition task is the next addition; the task interface (`build_task`) is where it goes.

## Not yet included (Phase 5 extensions)

These are planned on top of this core:
- Phase 5C: routing of the preconditioned update (EMA and agreement masks, norm-matched), plus group gradients via `torch.func`;
- Phase 5B: the unmasked-branch loss arm;
- Phase 5D: function-space consolidation;
- Phase 5E: incremental training sets.

The loss is computed in one place and the optimizer exposes the preconditioned update, so each is a local change.

## Running it

**Local** (CPU, or Apple GPU via MPS):

```
PYTHONPATH=src python experiments/fleet/bench_fleet.py --R 1 16 64 256
PYTHONPATH=src python experiments/fleet/run_fleet.py --grid GRID.json --out results/NAME
```

**Kaggle** (free T4 or P100), in a notebook with GPU enabled:

```
!git clone -b claude/theory-first-restructure https://github.com/rtavasso/topology-design.git
%cd topology-design
!pip -q install pytest
!PYTHONPATH=src python -m pytest -q tests/test_fleet.py
!PYTHONPATH=src python experiments/fleet/bench_fleet.py --R 16 64 256 512
!PYTHONPATH=src python experiments/fleet/validate_gpu.py
```

Kaggle's preinstalled torch is used. For a private repository, use a Kaggle secret holding a GitHub token. Results go to `/kaggle/working`; download them, or push them to a branch.

**Modal:** the same commands inside a Modal function with `gpu="T4"`, or `"L4"` for more memory. A CPU-only Modal function can instead fan out the existing per-run CPU sweep across many containers.

## Validation plan before using it for a study

1. `tests/test_fleet.py` passes on the target device. The MPS and CUDA equivalence tests run automatically when the device exists.
2. `bench_fleet.py`: run-steps per second against R, to choose `--max-batch`.
3. `validate_gpu.py`: Study 4 Stage-0 cells re-run on the device. Median grokking steps should match the CPU reference within the seed-to-seed spread.
