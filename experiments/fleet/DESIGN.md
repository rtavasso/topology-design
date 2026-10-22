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

All are `FleetConfig` fields, off by default. Semantics follow [`../fastslow/DESIGN.md`](../fastslow/DESIGN.md). **Code:** `fleet.py`, plus `fleet_signal.py` for 5A and the probes. **Tests:** [`tests/test_fleet_p5.py`](../../tests/test_fleet_p5.py), each mechanism against a direct single-model computation.

**Exactness is preserved.** With no Phase-5 option set, training is the unchanged baseline path. In a batch where other runs route, track or measure, a plain run is still updated with exactly the plain AdamW arithmetic. Tested bitwise in float64:
- a plain run is bit-identical whether it shares its batch with a plain, a time-routed, a rotation-routed, a path-efficiency or a Grokfast run;
- `measure_signal` leaves the training trajectory bit-identical, including the random draws of `rand` routing. The measurement uses its own generator; routing nulls use the optimizer's generator.

| study | fields | what it does |
|---|---|---|
| 5B time constants | `branch_lr_mult`, `branch_wd_mult`, `branch_alpha` | Per-branch multipliers, expanded onto each block of the branch. They multiply `block_lr_mult`, `block_wd_mult` and `alphas`. Scale matching uses `branch_alpha`. |
| 5B unmasked arm | `unmask_branch = b` | Branch b's parameters learn only from CE(z₀ + z_b). All other parameters learn from the full loss. |
| 5C routing | `route` ∈ {`time`, `agree`, `rand`, `topk`, `rot`}, `route_branch`, `route_alpha`, `route_tau`, `groups`, `route_rot_angle` | Direction-only routing of the **preconditioned** AdamW update u, on the receiving branch only (one-sided). The routed direction replaces u, rescaled to the norm of u over that branch. Decoupled decay is unchanged. If the routed direction vanishes, u is used unchanged. |
| 5C positive control | `grokfast` (λ), `grokfast_alpha` | Uniform Grokfast-EMA on all parameters: g ← g + λ·EMA_α(g), before AdamW. |
| 5D consolidation | `consolidate` (κ), `consolidate_to`, `consolidate_from`, `consolidate_beta`, `consolidate_null` | Adds κ·‖z_to − sg(z̄_from)‖² (squared norm over classes, as in the design), averaged over present training examples. z̄ is the per-example EMA (rate β) of the source branch's centered logits, started at each run's own source logits. The null's targets come from a fixed **derangement**: always a different example. |
| 5A signal | `measure_signal` | At each eval: S̃ for g and u under every split and matched null, plus masking and the three motion projections. |
| stability | `measure_path_eff` | Cumulative path efficiency ‖Σu‖/Σ‖u‖ per branch, using u as applied. |
| 5E arrivals | `arrive_start`, `arrive_n`, `arrive_every` | The training set grows: first `arrive_start` pairs, then `arrive_n` more every `arrive_every` steps, in a seed-fixed order. |
| 5E probes | `probe_steps`, `probe_pairs`, `probe_matched` | First-order probes at the listed steps. If `probe_matched` = k > 0, also protocol-matched probes with k AdamW steps (the design uses 50). |

**The 5C routing modes:**
- `time`: the bias-corrected EMA of u, including the current step (the low-frequency direction).
- `agree`: u masked to the coordinates where at least a fraction τ of the gradient signs agree across `groups` random disjoint groups. Groups are re-drawn every step **from the present training examples**, each getting an equal share (±1).
- `rand`: a random mask with the same density as the agreement mask.
- `topk`: the largest-|u| mask with the same density.
- `rot`: the null for `time`. It rotates u toward a random orthogonal direction.
  - The design matches the angle to the one **the time-routed arm achieved**. Run the time arm first and pass its median plateau angle as `route_rot_angle` (degrees).
  - Without it, each step uses the run's own angle between u and EMA(u).

**Routing constraints:**
- Group routes need per-group gradients, so they are part of the struct key: a batch is either all group-routed or not group-routed at all.
- Group routes cannot be combined with the unmasked arm or with consolidation; the config is rejected.
- Other routing modes, κ values, unmasked branches and consolidation pairs can be mixed freely in one batch.

**Extra outputs:**
- **Trajectory, at evals:**
  - `route_angle` (5C): median angle over the steps since the last eval, which is the manipulation-check statistic. `route_angle_mean` is also recorded.
  - `sig_*` (5A):
    - S_time98, S_time998, S_agree, S_rand, S_topk, S_rot98 and S_rot998 for g;
    - S_u_time, S_u_agree, S_u_rand, S_u_topk and S_u_rot for u;
    - agree_density;
    - proj_grad (−lr·g), proj_decay (−lr·wd·θ) and proj_update (−lr·u), all in parameter units, each projected on the gen direction;
    - gen_sq;
    - mask_M with mask_proj_present / mask_proj_ablated, all from the CE gradient alone.
    - Each is per unit: branches, then the shared parameters.
  - `path_eff` (per branch).
  - `abl_branch_train_loss` / `abl_branch_test_loss`, and the same per block. These form the nonlinear tier: each branch removed (with two branches, that is the other branch alone), then all branches removed.
  - 5E:
    - `mem_new`: per-component mem attribution of the latest increment;
    - `new_acc`, `new_loss`, `old_loss`, `n_seen`.
- **Summary:** `latency` (5E). For each increment, the steps until 90% of its pairs are fitted, read every step; −1 if never.
- **Result:** `probes` (5E). Per probe step:
  - `imprint` and `transfer`, per parameter unit;
  - with `probe_matched`, also `matched_logit` and `matched_gen`. These are per function-space unit (branches, then the embedding path), as treated copy minus control copy after k steps.

**Implementation notes:**
- The 5A time splits use the EMAs as they will be after the current step. That is exactly the direction time routing applies.
- g in the separation statistics is the gradient the optimizer uses, i.e. including any consolidation or unmasked-arm term.
- Protocol-matched probes clone each run (parameters and full optimizer state), at most `max_copies` = 512 copies at a time. They never touch the original runs, which is tested. The added pair does not enter the consolidation loss.
- The first-order imprint uses the centered true-label logit, consistent with the gen and mem supports. The design writes z_y.
- **Config gotcha (5E):** `train_frac = 0.4` gives 1,392 pairs at p = 59, so 348 + 30 × 35 = 1,398 is never reached. Use `train_frac ≈ 0.4016`, or accept a last increment of 29 pairs.

**Leakage guards:**
- Test labels enter only the measured gen direction and the probes, all through `autograd.grad`, never through `p.grad`.
- The 5A measurement leaves every `.grad` untouched, and leaves training bit-identical. Both are tested.

**Not implemented (read post hoc from the recorded trajectories):**
- output-side signal masking, from the per-component gen trajectories;
- interference, from `old_loss`, at eval granularity.

**Cost per feature,** relative to plain training: run `bench_p5.py`. On CPU at R = 2 and 20 steps (noisy):
- per-branch settings, arrivals and Grokfast cost about 1.1×;
- path efficiency costs about 1.25×;
- routing, consolidation and the unmasked arm cost about 1.4–1.5×;
- group routes and signal measurement cost about 1.6×.

GPU figures come from the Kaggle runner, which also runs `bench_p5.py`.

## Running it

**Run identity.** `run_id` is a readable prefix plus a hash of *every* config field. Changing the horizon, width or branches can never reuse or overwrite stale results, and the driver rejects duplicate configurations.

**Summary schema.** The summary uses the same keys as `grokking.stability_summary` where they overlap: t_fit, t_grok, stable, and post_fit_min_train_acc (0 for diverged runs). It adds n_collapses and final_rule_frac_test. Trajectory records are flat, with no `paths` sub-dict, so `paths/analyze.py` would need a small adapter.

**Local** (CPU, or Apple GPU via MPS):

```
PYTHONPATH=src python experiments/fleet/bench_fleet.py --R 1 16 64 256
PYTHONPATH=src python experiments/fleet/run_fleet.py --grid GRID.json --out results/NAME
```

**Kaggle** (free T4 or P100), driven from your machine by [`kaggle/launch.py`](kaggle/launch.py):
- **Validation:** `python experiments/fleet/kaggle/launch.py --user NAME`. This runs the tests, the benchmarks and `validate_gpu.py`, about 25 minutes.
- **Any grid:** add `--grid path/to/grid.json`.
  - Runs that share a struct key go into one batch, and with several GPUs there is one process per GPU (`run_fleet.py --shard`).
  - Results come back as `.json.gz` in `results/kaggle_<grid>/results/`. Relaunching resumes an interrupted grid.
  - `--no-wait` pushes and exits; `--fetch` collects the results later.
  - Phase 5's stages: [../fastslow/RUNNING.md](../fastslow/RUNNING.md).

The code goes up as a private Kaggle dataset, so the repository can stay private. The job runs in a private script kernel using Kaggle's preinstalled torch.

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
