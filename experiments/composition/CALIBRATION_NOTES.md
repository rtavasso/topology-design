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

**Round 3** (planned on the Mac): train fraction ∈ {0.8, 0.9}, lr ∈ {1e-3, 3e-4}, 50k steps, 2 seeds. The aim is to find a regime where `addmul` generalizes within budget before writing the preregistration.
