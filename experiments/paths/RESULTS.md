# Results — Study 4: what each path does

**Run:** 21 Oct 2026, natively on the researcher's MacBook Pro (10 CPU workers, commit `0aedb08`).
- 168 Stage-0 runs (8 graphs × 7 learning rates × 3 seeds) and 80 Stage-1 runs (8 graphs × 10 seeds), 0 errors.
- Scored against [PREREGISTRATION.md](PREREGISTRATION.md), committed before any run.
- Data: [`results_data/compact.json.gz`](results_data/compact.json.gz) holds per-run time-point snapshots and thinned trajectories. The automatic report is [`results_data/summary.md`](results_data/summary.md). Figures: [`roles_gen.png`](results_data/roles_gen.png), [`roles_mem.png`](results_data/roles_mem.png). Full run files stay on the Mac in `results/paths/`.

**Independent check:** a separate agent recomputed every confirmatory number from the per-run data with its own code. All verdicts reproduce. Its robustness analyses are folded in below.

## Verdicts under the preregistered rules

| | Verdict | Holm p | What it actually rests on |
|---|---|---|---|
| **H1** short paths construct, long paths memorize | **Supported** | 3e-4 | **Almost entirely one graph** (`pk0120`, unstable at a grid-edge lr). Without it, D ≈ 0.02 (direct) / 0.05 (total effect). |
| **H2** an extra short reader lowers long blocks' utility | **Not supported** | 1.0 | H2a goes the predicted way (−0.66) but is confounded by an 8× lr difference. H2b goes the opposite way (+0.49), for a structural reason built into the contrast. |
| **H3** composition delays generalization | **Supported** | 3e-4 | **Mostly the learning-rate assignment.** τ(C, delay) = 0.447 ≈ τ(C, 1/lr) = 0.445; at the shared lr 2e-3, τ = 0.02. |

The rules were applied as written. But the substance points elsewhere, and the clearest findings are ones the hypotheses did not target.

## Findings, by question

### Q-A, path roles: memorization and generalization live in the same paths

**Co-location (exploratory; the strongest structural result).** In 6 of 7 mixed graphs, the blocks that carry memorization (mem at its peak) are the blocks that later carry generalization (gen at the end), *in the same proportions*.
- **Direct attribution:** the cosine between the two block-share vectors is 0.98–0.999. The null, from permuting block labels within each run, is 0.27–0.76.
- **Total-effect attribution:** Pearson r = 0.90–0.999.
- **The trajectories show the mechanism** (figures, direct attribution):
  - while memorizing, a block's support for the true label on *unseen* inputs goes **negative**, because the memorized table pushes them toward wrong answers;
  - at grokking, the same block's support flips positive.

  So the generalizing function grows in the path that memorized. No separate path takes over.
- **Which path carries both is set by topology.** At moderate learning rates it is the block that reads the most upstream computation:
  - block 2 in `pk0100` (95% of gen, though two short readers are available);
  - block 4 in `pk0123`;
  - block 3 in `pk0121`.

  Blocks with the same parent split the work evenly. In `pk0110`, removing blocks 2 and 3 together costs exactly the sum of removing each alone ($U_{\{2,3\}} = U_{\{2\}}+U_{\{3\}}$): the wiring makes their contributions additive.
- **The one exception is `pk0120` at lr 0.016.**
  - Block 3 (depth 3) memorizes first. Then the short root reader (block 4) takes over both memorization and generalization.
  - At each run's best test-accuracy eval, the short share of gen is 0.59–0.98, against a short share of mem of about 0.2. That is a real role split, but in a graph that collapses in all 10 runs (stable 0/10, final test accuracy 0.34–0.99).

**Why H1 came out "supported."** The pooled D = 0.117 [0.101, 0.134] is mostly `pk0120` (D = 0.58 direct, 0.46 total). Per-graph D elsewhere is −0.01 to 0.09. Without `pk0120`, pooled D is 0.020 [0.010, 0.031] direct and 0.046 [0.039, 0.054] total. Its CI still excludes 0, but the size is a 2–5 point share difference. That is the preregistered "roles spread in proportion" reading.

**Destruction: none.** No component ever has negative mem. The cancellation ratio $\sum_j|\text{mem}_j|/|\text{mem}_\text{total}|$ is 1.00 at every time point in every graph. Memorization is removed by shrinking in place, never by another path cancelling it. So in these networks there is no "short path for destroying information".

**Composition is real.** Deep blocks' writes depend on what flows into them: `comp_dep_state` is 0.94–0.99 for most long blocks, and 0.57–0.86 for block 2 of `pk0121`–`pk0123`. A nominally deep block is not quietly recomputing from the embedding.
- In `pk0120`, block 2's value is 3.7. The measure exceeds 1 when the block's output on the raw embedding is larger than on its real input. This marks an off-distribution response, not a share.

**Preservation.** The embedding path's own readout carries no generalizing support (gen ≈ 0). This is structural, not learned: a sum $f(a)+g(b)$ cannot express a non-constant function of $(a+b) \bmod p$.

### Q-B, friction: learning rate decides which path carries the function

**H2 failed, and both contrasts were flawed in ways the design review missed:**
- **H2b.** Rewiring block 3 of `pk0100` to read $h_1$ makes it a *sibling* of block 2. Additive siblings split the work, so $U_{\{2\}}$ halves in `pk0110` by construction. The fairer comparison, utility of all long blocks, gives A − B = +0.05: still the wrong sign, but small.
- **H2a.** `pk0120` ran at lr 0.016 and `pk0121` at 0.002. At shared learning rates (Stage 0, 3 seeds), A − B is −0.013, −0.039, −0.047 and −0.159 at lr 2e-3, 4e-3, 8e-3 and 1.6e-2. That is consistently in the predicted direction but mostly small. It is the only friction-like signal, and it is exploratory.

**What the data show instead (exploratory, Stage 0, 3 seeds per cell).** The short paths' share of total-effect gen at the end of training (grokked runs):

| graph | independent short reader? | lr 1e-3 | 2e-3 | 4e-3 | 8e-3 | 1.6e-2 |
|---|---|---|---|---|---|---|
| `pk0120` | yes (block 4) | 0.43 | 0.49 | 0.52 | 0.68 | **0.87** |
| `pk0100` | yes (blocks 3, 4) | 0.65 | 0.64 | 0.50 | 0.50 | 0.76 |
| `pk0110` | yes (block 4) | 0.58 | 0.55 | 0.50 | 0.50 | 0.64 |
| `pk0111` | no | 0.52 | 0.51 | 0.50 | 0.50 | 0.51 |
| `pk0121` | no | 0.41 | 0.37 | 0.38 | 0.41 | 0.39 |
| `pk0122` | no | 0.35 | 0.34 | 0.35 | 0.35 | 0.29 |
| `pk0123` | no | — | 0.28 | 0.28 | 0.13 | 0.18 |

- **Where an independent short reader exists, function migrates onto it at high learning rates.** That is exactly where Study 2 found long chains unstable. Without one, allocation does not move.
- **A candidate mechanism:** instability of the long path hands the function to a stable short path. This is a friction-like effect, but driven by optimization stability, not by a shared error signal starving the long path.
- **At moderate learning rates the opposite holds.** The composed path wins even against two short readers (`pk0100`).
- *Note:* total-effect Shapley gives block 1 a large share as the root that every chain depends on, so 0.50 is roughly the floor in graphs where block 1 feeds everything.

**Gradient allocation (exploratory).** At matched train loss, short blocks receive larger gradients than long ones in most graphs (e.g., `pk0111` 0.21 vs 0.09 at loss ≤ 1). The exceptions are `pk0100` and `pk0120`. Larger gradients did not translate into carrying more function.

### Q-C, memorization vs generalization: composition costs time mainly through the learning rate

- **H3:** τ_b(C, delay) = 0.447, seed-bootstrap CI [0.38, 0.51], permutation p = 1e-4.
- **But delay × lr is roughly constant within a graph.** For example, `pk0000` gives 6.2 at lr 2.5e-4 and 6.9 at 1.6e-2. Long chains were selected at low learning rates because of their stability ceiling (Study 2), so τ(C, 1/lr) = 0.445 nearly equals τ(C, delay).
- **At a shared lr of 2e-3,** the only lr where every graph groks, τ = 0.02 (p = 0.45). At 4e-3 and 8e-3, τ is 0.15–0.18 (not significant); at 1.6e-2 it is 0.30 (p = 0.04).
- **What survives, exploratory:** the lr·wd-normalized delay. `pk0000` and `pk0120` need about 6.4–6.9 units of lr·wd·steps; the other chains need 11–19 (τ = 0.39).
- **Reading:** composition delays generalization chiefly by forcing a smaller stable step (an *optimization* claim, consistent with Study 2), plus a residual cost per unit of regularization that needs a dedicated test.
- **Other descriptors:**
  - peak memorization (mem_total) is higher in the chain graphs (35–52) than in the parallel net (21);
  - memorization persists longer in them (persistence 1,800–25,000 steps vs 420).
  - The exception both times is `pk0120` at lr 0.016 (19.5 and 392), the graph whose short reader took over.
  - Both descriptors have the same lr confound.

## Caveats

1. **Long chains are unstable at every lr that groks.** All four flagged graphs collapse repeatedly.
   - `pk0122` and `pk0123` grokked in only 4 and 5 of 10 runs, so their H1 values rest on survivors.
   - The "unstable tier" choice for `pk0120` ranked post-fit minima of 0.07–0.13. Every option had collapsed, so in effect it picked the fastest lr, at the grid edge.
2. **`pk0000`'s chosen lr (0.016) is also at the grid edge.** Its optimum may lie higher.
3. **H1 compares two times** (gen at $t_\text{end}$, mem at $t_\text{mem}$), so D mixes a split between paths with a shift over time. Co-location, which uses the same two times, avoids this problem only partly.
4. **One task with a shallow rule.** Co-location might differ for rules that need composition.
5. **Shared-lr results are 3 seeds per cell.** Stage-0 path data is not in `compact.json`; the tables above were computed on the Mac from the full Stage-0 files.
6. **Display bug,** now fixed: the automatic report printed censored medians as numbers (`pk0123` delay "500006170"). Its delay median is censored: at least 12,340, with 5 of 10 runs never grokking.

## What this changes

- **For the program's core question:** in this setting, topology does not separate memorization from generalization into different paths. It decides *which* path carries both, and that choice depends on the learning rate:
  - at moderate learning rates, the composed path wins;
  - at high learning rates, function migrates to an independent short reader where one exists.

  That is a cleaner, more falsifiable statement than any of the three hypotheses. It also bears directly on the original idea of short paths for preserving, destroying and constructing: no destruction happened, and construction happened in place.
- **Next experiments (to be preregistered fresh):**
  1. **Co-location, confirmatory:** at a shared stable lr, with the block-permutation null, 10 fresh seeds, on these graphs plus a deep-rule task.
  2. **Migration dose–response:** short-reader share vs lr in graphs with and without an independent short reader, 10 seeds, with per-path stability tracked to test the "unstable path hands off" mechanism.
  3. **Friction at matched lr:** `pk0120` vs `pk0121` at 2e-3 and 4e-3 with 10 seeds, the one small consistent signal.
