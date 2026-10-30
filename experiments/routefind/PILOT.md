# Rung 2 pilot: the router's training signal (G3)

**Provenance.** Grid `grids/rf_pilot.json` at `705e345` (design revision 4); 27 runs, seed 100, 5,000 steps × batch 64, two T4s (Kaggle `topology-fleet-route-rf-pilot`); 2 shards of ≈ 0.55 h wall each. Records in [results_data/pilot/](results_data/pilot/); the analysis in [results_data/pilot_report.md](results_data/pilot_report.md) (`analyze.py`). Tests on the GPU: 47 passed. No run failed.

**Purpose** ([DESIGN.md](DESIGN.md) §6): pick each candidate's lr on δ = 1; a first look at the δ = 0 controls (L26). One seed; nothing here is a finding. Every outcome below is read against the predictions of §7 only to see whether the main grid can read them.

## 1. Outcomes

Shortest-correct committed accuracy on P-words at in4 / in8 at 5k; criterion = the held-out step at which every k ≤ 8 is ≥ 0.95, confirmed. Program: E (SET early, carry) / L (SET late). Determinism: sampled shortest-correct on P-words at in8. PP: shortest-correct at pp8. Distinct: distinct sampled routes per P-word query over the last 1,000 steps (16 samples).

| arm | lr | in4 / in8 | criterion | program | deterministic | PP | distinct | note |
|---|---|---|---|---|---|---|---|---|
| **IW16 δ1** | 3e-4 | 1.00 / 1.00 | 1,800 | E | 1.00 | 0.02 | 1.0 | |
| | **1e-3** | 1.00 / 1.00 | **1,600** | E | 1.00 | 0.00 | 1.0 | pp8 fails SET>PREV (F30, as EX) |
| | 3e-3 | 0.01 / 0.00 | – | – | – | – | 4.8 | learned by 500, then collapsed into all-H |
| **PF-IS16 δ1** | 3e-4 | 1.00 / 1.00 | 2,500 | L | 0.13 | 0.00 | 12.4 | |
| | **1e-3** | 0.99 / 1.00 | **1,400** | L | 0.14 | 0.00 | 12.5 | ESS / K 0.91 |
| | 3e-3 | 0.99 / 0.99 | 2,500 | L | 0.14 | 0.00 | 12.6 | |
| **TB16 δ1** | 3e-4 | 0.20 / 0.07 | – | (L) | – | – | 11.0 | |
| | **1e-3** | 0.83 / 0.76 | – | (L) | – | – | 11.3 | rising to 0.97 overall / 0.84 at k = 8 by 3,500, then flat as the lr decayed |
| | 3e-3 | 0.01 / 0.00 | – | – | – | – | 10.7 | all-H |
| **EI16 δ1** | 3e-4 | 1.00 / 1.00 | 1,300 | L | 1.00 | 0.00 | – | init beam program E (0.40) |
| | **1e-3** | 1.00 / 1.00 | **800** | L | 1.00 | 0.00 | – | |
| | 3e-3 | 0.01 / 0.00 | – | – | – | – | – | hit 1.00 but the imitated route is not shortest; all-H committed |
| **RL16 δ1** | 3e-4, 1e-3 | 0.01 / 0.00 | – | – | – | – | 1.0 | all-H; deterministic by step 200 |
| | 3e-3 | 0.00 / 0.00 | – | – | – | – | 1.0 | halts at the start |
| **ACMC16 δ1** | 3e-4, 1e-3 | 0.01 / 0.00 | – | – | – | – | 1.0 | all-H; deterministic by step 400 |
| | 3e-3 | 0.00 / 0.00 | – | – | – | – | 1.0 | never halts (+55 steps) |
| **AC16 δ1** (TD 0.9) | all | 0.01 / 0.00 | – | – | – | – | 1.0 | all-H; deterministic by step 400 |

δ = 0 controls, lr 1e-3:

| arm | in4 / in8 | criterion | program | deterministic | PP | distinct | note |
|---|---|---|---|---|---|---|---|
| **EX-AL δ0** | 0.00 / 0.00 | – | – | 1.00 | 0.00 | – | **answers 1.00 at in4, in8, none8 (all-H words) and len32, 0.86 at pp8, with exactly one extra step on every word** (§3) |
| EX-AL δ0, 3e-3 | 0.00 / 0.00 | – | – | – | – | – | answers 0.84–0.96, +6 steps |
| **PF-IS16 δ0** | 1.00 / 1.00 | 1,100 | L | 0.12 | 0.99 | 12.7 | |
| **TB16 δ0** | 1.00 / 1.00 | 3,800 | L | 0.18 | 1.00 | 11.6 | |
| RL16 δ0 | 0.00 / 0.00 | – | – | – | – | 1.0 | halts at the start |
| AC16 δ0 | 0.01 / 0.00 | – | – | – | – | 1.0 | all-H |

## 2. The lr rule

Per signal, the lr with the highest held-out shortest-correct accuracy at 5k on δ = 1; ties by the mean over the last 1,000 steps, then 1e-3.

| signal | lr | why |
|---|---|---|
| IW16 | 1e-3 | 3e-4 and 1e-3 tie at 1.00; 1e-3 reached criterion earlier (1,600 vs 1,800); 3e-3 collapsed |
| PF-IS16 | 1e-3 | all three at 0.99–1.00; 1e-3 earliest (1,400 vs 2,500) |
| TB16 | 1e-3 | 0.97 vs 0.86 / 0.84 overall; none at criterion |
| EI16 | 1e-3 | 3e-4 and 1e-3 tie; 1e-3 earlier (800 vs 1,300) |
| RL16, ACMC16, AC16 | 1e-3 | every lr at the floor; 3e-3 collapsed into halting (RL16) or never halting (ACMC16) |

Every signal keeps 1e-3. The main grid's lr values are therefore unchanged (`make_grids.py`).

## 3. What the pilot showed (one seed; to be read at the main grid)

**The reward arms collapse before they learn.** RL16, ACMC16 and AC16 become deterministic within 200–400 steps at every lr and both triggers: the 16 sampled routes per query are identical from then on, so the RLOO advantages and the critic's advantages are identically zero and nothing moves again. The policy they freeze in is all-H (criterion-set shortest-correct 0.84: every word without a P), or halting at the start at the highest lr. The actor-critic arms collapse within 200 steps of the actor switching on, after their critic warm-up. The critic's explained variance stays at 0.06–0.09.
- This is the floor condition that P4 names: **P4 is not read in this rung.** The question of contextual value needs a reward signal that keeps exploring; the pilot shows that on-policy sampling with no entropy term does not.
- The collapse also ends the P6 question for the reward arms.

**The fixed-prior posterior, sampled, learns both triggers.** PF-IS16 reached a shortest correct program on δ = 1 at every lr and on δ = 0 at 1,100 steps, in program L each time, keeping 12–13 distinct routes per query (ESS / K 0.90) throughout. TB16 learned δ = 0 (3,800 steps) and climbed to 0.97 on δ = 1 before the schedule decayed. Both are stochastic at their optimum, as derived (§5.2): sampled shortest-correct 0.12–0.18, even where the committed route is shortest-correct on 0.99–1.00.
- P2(b), the sampled fixed target not being trapped on δ = 0, looks as predicted.
- P3 (PF-IS16 over TB16) looks as predicted on this seed: 3/3 vs 0/3 lrs on δ = 1, and 1,100 vs 3,800 steps on δ = 0.
- P5's open line for PF-IS16 resolves to **L on δ = 1**, not the E of the pooled derivation. Exact PF (the main grid's check) will say whether that is the fixed point or the transient.

**EX-AL on δ = 0 found a correct program that is not the shortest: a count-shifted one.** Inferred from the endpoints, not from inspected routes:
- answers 1.00 on in4, in8, none8 and len32, deterministic;
- exactly one more step than the oracle on every stratum, including words with no P (none8: 9 vs 8; len32: 39.8 vs 38.8), so one SET per route wherever the oracle takes none;
- teacher-forced HOP accuracy 0.00 at the oracle's (c = 0) states and SET accuracy 1.00; the count-2 → PREV probe 0.75–0.81; the committed route halts, but not at the oracle's (answer, c = 0) state (HALT accuracy 0.00 there).
- Reading: "c = 0 ⇒ SET" at the start, then the oracle's rule with every decision one count higher: SET at a flagged state with c = 1, PREV at c = 2, HALT at the answer with c = 1. The PREV decision is made at a state that the init phase never labeled, which is F29's mechanism seen from the other side: the router keeps the stray-phase labels and builds the program around them, in the spare count capacity (c ≤ 2).
- Under the preregistered criterion (shortest-correct) this is "not learned", and P2(a) as stated holds. But the reading written for it — "the answer-only likelihood alone does not prevent the basin" — is too coarse: the likelihood took EX-AL out of all-H into a correct program; what it did not do is reach the shortest one, which PF-IS16 and TB16 do with the same likelihood and a fixed prior. The main grid's three EX-AL δ = 0 seeds and two δ = 1 seeds decide whether this program is systematic. If it is, the reading becomes: self-consistency keeps the init phase's labels; the likelihood decides whether a program can be built around them.

**IW16 and EI16 learn δ = 1 at two of three lrs,** in E and L respectively, both deterministic, both failing PP. IW16's pp8 failure is SET>PREV, the precedence failure F30 described for EX; EI16's is SET>HOP (L cannot read the flag at a PP arrival, as derived). EI16's learned program is L on a seed whose init beam program was E (0.40): the P5 line for EI16 is wrong on this seed. IW16 at 3e-3 learned by step 500 and then collapsed into all-H.

**TB16 is slow on δ = 1.** Its gradient is clipped on 65–85 % of steps, and its criterion curve was still rising when the cosine schedule reached 15 % of the peak (step 4,000). The main grid runs on the same schedule, so TB16's δ = 1 count will carry this confound, which review 3 (S6) anticipated. It is reported, not corrected: a longer schedule for one arm would break the matched budget.

**Timing.** 0.085–0.087 s per run-step for every arm, exact or sampled, stacked or not (a stack of 3 took 3× a stack of 1). The main grid's estimate at the EX rate holds.

## 4. Decisions for the main grid

- **lr:** 1e-3 for every signal (§2).
- **D108 — the reward arms become checks.** RL16, ACMC16 and AC16 run at 2 seeds each on δ = 1, to replicate the collapse at main-grid seeds; their δ = 0 and dose cells are dropped. This follows P4's preregistered uninterpretability clause and L27: 10/10 pilot runs of these arms collapsed by the same mechanism, so 21 further runs would be confirmatory. 93 → 72 runs, ≈ 9 T4-hours. AC16T0 stays (2 runs): it is the one cell that separates TD(0)'s halting trap from the general collapse.
- **No arm is added.** The reward arms' failure asks for an exploration term (entropy bonus or ε-mixing); that is a design change with its own CPU sanity run and review, and it is the first follow-up for risk 4, not a change to this grid.
- **Predictions are unchanged** (§7 of the design is preregistered). Two readings will need care at the main grid: P2(a) in the light of the count-shifted program, and P5's EI16 line.
