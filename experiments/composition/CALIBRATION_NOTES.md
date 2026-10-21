# Study 3 calibration notes (not evidence; used only to choose settings)

Task `addmul`: y = ((a+b) mod p)·c mod p, p = 13. Control `add3`: y = (a+b+c) mod p. One seed per cell.

**Round 1** (lr 3e-3, weight decay 1, train fraction 0.5, 15k steps):
- `add3`: the chain grokked at step 7,100. The parallel net plateaued at test accuracy 0.86.
- `addmul`: neither grokked. Test accuracy was 0.25 for the chain and 0.16 for the parallel net.
- The chain was unstable at this lr, as in Study 2: train accuracy repeatedly collapsed to 0.2–0.4.

**Round 2** (lr 1e-3, weight decay ∈ {1, 0.3}, train fraction ∈ {0.5, 0.7}, 30k steps, `addmul` only):
- No run grokked.
- Best: chain at train fraction 0.7, test 0.43 and still rising. The parallel net in the same cell reached 0.29.

**Consistent across all 8 calibration pairs (exploratory lead, one seed each):**
- `inter_frac`, the share of logit energy that is a function of (a+b, c), is higher for the chain: 0.73–0.97 vs 0.36–0.76.
- The chain's memorization measure (excluded-train accuracy) is lower: 0.46–0.65 vs 0.43–0.80.
- **Confound:** a model that is closer to generalizing automatically has higher `inter_frac`. The preregistration must compare at matched test accuracy and track `inter_frac − rule_frac`.

**Probe finding:**
- After the fix (standardized ridge, verified by tests at feature scale 0.01), s = a+b is still not linearly decodable from any residual state in any run, chance being ≈ 0.08.
- So the networks use a+b only entangled with c, not as an explicit variable in the stream. The "which block computes the intermediate" readout is therefore uninformative in this regime.

**Round 3** (Mac, 20–21 Oct; train fraction ∈ {0.8, 0.9}, lr ∈ {1e-3, 3e-4}, weight decay 1, 50k steps, 2 seeds; 16 runs):

- **Train fraction 0.9 is needed.** At 0.8 nothing grokked; the best test accuracy was 0.73 (parallel net).
- **At 0.9 the parallel net generalizes; the chain gets close but never crosses 0.95.**

  | arm, lr | seed 0 | seed 1 |
  |---|---|---|
  | pk000, 1e-3 | grok 13.7k | grok 49.2k |
  | pk000, 3e-4 | grok 43.5k | test 0.89 at 50k |
  | pk012, 1e-3 | test 0.85 | test 0.89 |
  | pk012, 3e-4 | test 0.29 | test 0.23 |

- **The chain is unstable in all 8 runs. The parallel net is stable in all 8.** At train fraction 0.9:
  - chain at lr 1e-3: 24 train-accuracy collapses per run (post-fit minimum 0.17–0.25);
  - chain at lr 3e-4: 34–39 collapses, so *lowering* the lr made it worse;
  - parallel net: 0 collapses.
- **Lower lr → more crashes contradicts a pure step-size story.** With AdamW, a lower lr also means weaker decoupled decay (lr·wd per step). Weights grow larger and sharper, so instability rises. Study 2 points the same way: weight decay 0.1 lowered every ceiling. This matches the Adam "slingshot" cycles of norm growth and loss spikes in grokking (Thilak et al. 2022, arXiv:2206.04817). **This is the lead for the Study 2 mechanism test:** track parameter norm and sharpness alongside collapses, and vary wd at fixed lr. Round 4 does the second.

**Measurement problems found in round 3** (fixed or flagged before the preregistration):

1. **The s-probe was underpowered, so the earlier "s is not decodable" conclusion is withdrawn.**
   - The ridge readout was fit on half of the *test split*: about 110 points for 129 features at train fraction 0.9, and about 330 at 0.7. So chance-level readings say nothing.
   - **Fix:** the probe now fits and evaluates on fixed random halves of the *full grid* (1,098 / 1,099 points). s is never a training label, so this is legitimate.
   - A planted circular code amid 124 distractor dimensions decodes at > 0.6 (chance 0.08). Pure noise stays < 0.15.
2. **New exact measure, `state_s_frac`.** For each residual state, the share of its energy that depends on s = a+b jointly with c beyond c alone. This is **exactly 0 for the embedding sum** h_0, because on the full grid the mean of E_a[a] + E_b[s−a] over a is constant. So any positive value means the stream has computed the sum, in whatever basis, with no probe needed. `logit_s_frac` is the same quantity for the logits.
3. **Logit-level `inter_frac` and `rule_frac` are dominated by the training fit at high train fraction.** 90% of points are fit to y, which is itself a function of (s, c). The chain's `inter_frac` of 0.97–1.00 against the parallel net's 0.85–0.97, already at step 1,000, is therefore *not* evidence of a cleaner intermediate. On this task, logit-level intermediate structure is nearly inseparable from generalization. **The preregistration's intermediate measures must be state-level** (`state_s_frac`, the probe), not logit-level.

**Round 4** (`grids/calibrate4.json`, 12 runs, about 30 min): train fraction 0.9, lr 1e-3, weight decay ∈ {1, 3, 10}, pk012 vs pk000, 2 seeds, with the fixed probe and `state_s_frac`. It asks two things:
- Does stronger decay stabilize the chain? That would be a Study 3 regime without the stability confound, and a direct test of the slingshot lead.
- Where in the stream does a+b appear?
