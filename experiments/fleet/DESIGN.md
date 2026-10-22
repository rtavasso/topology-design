# Fleet trainer — design

**Goal.** Run hundreds of small, independent training runs at once on a GPU, so that a whole study (all arms × all seeds) finishes in minutes to an hour instead of overnight.

**Code:** [`src/topology_design/fleet.py`](../../src/topology_design/fleet.py), plus [`fleet_signal.py`](../../src/topology_design/fleet_signal.py) for Phase 5A. **Driver:** [`run_fleet.py`](run_fleet.py). **Tests:** [`tests/test_fleet.py`](../../tests/test_fleet.py) and [`tests/test_fleet_p5.py`](../../tests/test_fleet_p5.py).

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
- **Floating point:**
  - In **float64**, the fleet reproduces `grokking.train_run` exactly for step lists, $t_\text{fit}$, $t_\text{grok}$ and divergence steps. Losses agree to about 1e-12 until chaotic collapse phases amplify summation-order differences (around step 1,300 in the test). Tested with mixed graphs, dense evals, stop-after-grok, compaction and a diverging run.
  - In **float32**, batched and single matrix products sum in a different order. Parameters agree to about 1e-6 for the first steps and drift to about 3e-5 by step 25, so chaotic phases diverge between devices.
  - Device validation is therefore seed-paired and statistical (`validate_gpu.py`), not trajectory-exact.
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

**Per-eval cost:** 1 + L + (branches + 1) grid forward passes, about 6× a training step on CPU.

**Tested equal to the reference** on the same weights:
- `grokking.path_measure` gen and mem;
- `grokking.measure` loss and accuracy;
- single-block and branch ablations;
- the test-only rule fraction;
- gradient norms;
- invariance to chunk size.

Exception and rule-train quantities and `part_rms` were cross-checked manually by the code review.

Sets that are empty, such as exceptions on plain `modadd`, are recorded as `null`, not 0.

## No per-step synchronisation

- Divergence (a non-finite loss) and early stopping are tracked in device tensors, and read only at evals.
- A diverged or finished run is **frozen exactly**: its parameters and optimizer moments are left unchanged, even if its gradients are NaN. Its loss is excluded, so it cannot contaminate other runs. Tested.
- **Compaction:** when the active fraction of a batch falls below one half, the batch is shrunk to its active runs (parameters, moments, hyperparameters, data). A batch's wall time therefore no longer waits on idle runs.
- The driver also sorts each batch group by learning rate, so runs of similar length share a batch.

## Tasks

| task | definition |
|---|---|
| `modadd` | $(a+b) \bmod p$, with the same splits as Studies 1–4 |
| `modadd_exc` | as `modadd`, but a fixed random fraction `exc_frac` of the training pairs get arbitrary labels that are never the rule's. These exceptions can only be memorized. |

`modadd` is exactly the `modadd_exc` setting at `exc_frac = 0`. The group-product composition task is the next addition; the task interface (`build_task`) is where it goes.

## Phase 5 extensions

All are `FleetConfig` fields, off by default. With none set, training takes the unchanged baseline path (`FleetAdamW._plain_step`), so the exactness results above still hold. Semantics follow [`../fastslow/DESIGN.md`](../fastslow/DESIGN.md). **Code:** `fleet.py`, plus `fleet_signal.py` for 5A. **Tests:** [`tests/test_fleet_p5.py`](../../tests/test_fleet_p5.py).

| study | fields | what it does |
|---|---|---|
| 5B time constants | `branch_lr_mult`, `branch_wd_mult`, `branch_alpha` | Per-branch multipliers, expanded onto each block of the branch. They multiply `block_lr_mult`, `block_wd_mult` and `alphas`. Scale matching uses `branch_alpha`. |
| 5B unmasked arm | `unmask_branch = b` | Branch b's parameters learn only from CE(z₀ + z_b), so the other branches cannot mask their error signal. All other parameters learn from the full loss. Tested against a single model. |
| 5C routing | `route` ∈ {`time`, `agree`, `rand`, `topk`, `rot`}, `route_branch`, `route_alpha`, `route_tau`, `groups` | Direction-only routing of the **preconditioned** AdamW update u, on the receiving branch only (one-sided). The routed direction replaces u, rescaled to the norm of u over that branch. Decoupled decay is unchanged. If the routed direction vanishes, u is used unchanged. |
| 5D consolidation | `consolidate` (κ), `consolidate_to`, `consolidate_from`, `consolidate_beta`, `consolidate_null` | Adds κ · MSE between the receiver's centered training logits and an EMA (rate β) of the source's. The target is detached. The null uses targets from a fixed permutation of the training examples. |
| 5A signal | `measure_signal` | At each eval, records the separation statistic S̃ for the actual update and for each mask, with matched nulls. |
| 5E arrivals | `arrive_start`, `arrive_n`, `arrive_every` | The training set grows: first `arrive_start` pairs, then `arrive_n` more every `arrive_every` steps, in a seed-fixed order. |
| 5E probes | `probe_steps`, `probe_pairs` | At the listed steps, records each unit's first-order imprint and transfer for held-out pairs. |

**The 5C routing modes:**
- `time`: the bias-corrected EMA of u (the low-frequency direction).
- `agree`: u masked to the coordinates where at least a fraction τ of the gradient signs of `groups` random disjoint example groups agree. The groups are re-drawn every step.
- `rand`: a random mask with the same density as the agreement mask.
- `topk`: the largest-|u| mask with the same density.
- `rot`: the null for `time`. It rotates u by the same angle that the EMA direction makes with it, in a random plane.

Every mode records the realized angle between the routed and the original update (`route_angle`, at evals). Group routes need per-group gradients, so they are part of the struct key: a batch is either all group-routed or not group-routed at all.

**Mixed batches.** Runs with different routing modes (other than group routes), different κ or different unmasked branches can share a batch. Each per-run choice is a mask over the run dimension.

**Extra outputs:**
- **Trajectory, at evals:**
  - `sig_*` (5A): S̃ for the time, agreement, random, top-k and rotation directions, the same for u, agreement density, and the projections of gradient, decay and update onto the gen direction;
  - `route_angle` (5C);
  - `new_acc`, `old_loss`, `n_seen` (5E).
- **Summary:** `latency` (5E). For each increment, the steps until 90% of its pairs are fitted; −1 if never.
- **Result:** `probes` (5E).

**5A leakage guard.** Test labels enter only the measured gen direction. They never enter anything used for training. A test checks that the measurement leaves every parameter's `.grad` untouched.

**Cost per feature,** relative to plain training: see `bench_p5.py`. On CPU at R = 2:
- per-branch settings and arrivals are free;
- `time` routing costs about 1.1×;
- `rot`, group routes, consolidation and the unmasked arm cost about 1.2–1.3×;
- signal measurement costs about 1.3× at eval-every-100.

GPU figures come from the Kaggle runner, which now also runs `bench_p5.py`.

## Running it

**Run identity.** `run_id` is a readable prefix plus a hash of *every* config field. Changing the horizon, width or branches can never reuse or overwrite stale results, and the driver rejects duplicate configurations.

**Summary schema.** The summary uses the same keys as `grokking.stability_summary` where they overlap: t_fit, t_grok, stable, and post_fit_min_train_acc (0 for diverged runs). It adds n_collapses and final_rule_frac_test. Trajectory records are flat, with no `paths` sub-dict, so `paths/analyze.py` would need a small adapter.

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

## Validation status

**PASSED on a Kaggle T4, 21 Oct 2026** ([results/VALIDATION.md](results/VALIDATION.md)):
- $t_\text{fit}$ matches exactly in 30 of 30 seed-paired runs;
- $t_\text{grok}$ agrees within the seed spread in all three graphs;
- tests pass on CUDA.
- Throughput is about 640 run-steps/s and compute-bound from R ≈ 16, so the default CUDA batch is now 64.

## Validation plan before using it for a study

1. `tests/test_fleet.py` passes on the target device. The MPS and CUDA equivalence tests run automatically when the device exists.
2. `bench_fleet.py`: run-steps per second against R, to choose `--max-batch`.
3. `validate_gpu.py` re-runs Study 4's Stage-1 graphs on the device (by default pk0000, pk0100 and pk0111; 10 seeds each, at their selected lrs) and compares per seed with the CPU runs in `experiments/paths/results_data/compact.json.gz`. It exits non-zero unless:
   - $t_\text{fit}$ matches exactly in at least 80% of runs (a non-chaotic phase);
   - for each graph, the median |Δ$t_\text{grok}$| is within the CPU seed IQR (or 10% of the median);
   - there is at most one grok/no-grok disagreement.

**Known costs, not yet optimized:**
- In mixed-graph batches, choosing parents stacks the earlier states (about 10% per step on CPU).
- Ablations recompute the full forward pass.
