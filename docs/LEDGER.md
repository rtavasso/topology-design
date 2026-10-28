# Ledger: findings, failures, predictions, ideas

> "Rather than fighting to make our hypothesis true, we should be aware of the honest things our data says about our hypothesis. There is no pride in this. If I am wrong, let me be wrong fast." (the researcher, 22 Oct 2026)

**What this is.** One place where every claim the program makes is listed with its honest status. It is the first file to read before designing anything, and the first to update after any result.
- [history.md](history.md) records *why* decisions were made.
- This ledger records *what we currently believe and how strongly*.

**Rules**
1. **Every entry carries:** an ID, a status, its evidence (with a link) and its scope. Hypotheses also carry their prediction, falsifier and cheapest test.
2. **Statuses**
   - Findings:
     - **established**: preregistered and replicated, or derived and tested;
     - **strong lead**: preregistered, not yet replicated;
     - **suggestive**: exploratory, or few seeds.
   - For anything resting on reasoning alone:
     - **argument**: derivation or reasoning, untested;
     - **prior art**: answered or partly answered in the literature.
   - Labels for entries that leave the main lists:
     - **failed**: a prediction we made, contradicted by data;
     - **refuted**: an idea killed by data or derivation;
     - **inconclusive**: the test could not discriminate.
3. **Status changes are appended with a date, never overwritten.** Example: "suggestive → failed (2026-11-02, link)". A failed prediction is moved to *Failed predictions and refuted ideas*, never deleted.
4. **Before proposing an experiment,** check this file and [literature.md](literature.md) / [literature_review.md](literature_review.md) for whether it is already answered (lesson L2).
5. **Claim types,** as in AGENTS rule 5: expressivity, optimization or mechanism. Most of what we know so far is optimization.

## Direction

The researcher's end goal, in his words:

> "A real learner shouldn't need to walk the loss landscape, it should be able to see the minima on the horizon by reasoning over the structure of the data it has learned, which generalizes to unseen examples. All of the experiments I am interested in are related to that end-goal. But we must make small steps towards that vision to understand what is happening and why."

- **The original question:** what structure do network topologies promote during learning, and can wiring support the generalizing solution and keep the memorizing shortcut from interfering?
- **How the frame has moved (22 Oct):** from "which wiring" towards "how learning is organized":
  - the gradient as a single-path, averaging learner;
  - shared error versus separate errors;
  - search over structures versus walking;
  - topology as a hypothesis about how the data is generated.
- **Scope principle:** toy tasks isolate one piece; realism is added back step by step. A fixed dataset is not the learning system we care about in the limit.

- **How the frame moved again (24 Oct discussion, no new data):**
  - In a gradient-trained network the prior is not a separate object: it is the bias of the search (optimizer, topology, settings together). E1's predict-then-learn code length measures how well a learner's prior fits the data.
  - The horizon is short in weight space (E2) and 3 examples long in hypothesis space (E1), so "seeing the minima" points to holding hypotheses explicitly, not extrapolating a walk.
  - **Generality at the population, efficiency at the member:** a general network is data-hungry for the same reason it is general. A population of restricted members can be general overall while each learns from few examples, at a cost of about log N nats of evidence for the selection, paid in compute rather than data.
  - The open problems this exposes: where the members' structure comes from (selection across tasks, growth, compressed generators), how credit flows *between* members (responsibility by out-of-sample prediction, I20), and composition versus competition (sums share error; mixtures do not compose).
  - The two claims everything rests on: I18 (member level) and I19 (transfer).

### Update (25 Oct 2026, after the priors round)

- **The data gap is the prior, not the search (F19).**
  - A rule the network already has survives on fewer examples only about where it would be found from scratch (S ≈ 1).
  - So credit routing, competing members, growth criteria or better optimizers inside this model class can't close the gap.
- **Generic restriction (width) doesn't help (F20).** The chain's extra cost looks like prior too (F21, suggestive).
- **E2's jump result survives a matched null (F16 note),** in a narrower form.
- **What this points to:** changing the *prior*. Structure matched to the task; priors or blocks learned across a family of tasks (I19, the amortized rival I23); search over priors rather than over weights. The open question is no longer "how does the walk fail" but "what prior makes the rule cheaper than a lookup table from a few examples, and where does that prior come from".

### State of the evidence (23 Oct 2026, after E1–E4)

A short synthesis for discussion. Each line points to the entry that carries the evidence.

**What we can now say**
1. **The large inefficiency is in data, not steps.** Gradient descent needs ≈ 1,044 examples where a program-prior learner needs 3; extra steps per example barely help (F13).
2. **Memorization is actively harmful to prediction before the rule arrives.** On a stream, the network predicts new pairs confidently wrong, worse than guessing, for most of the stream; more steps per example make this worse (F14).
3. **During the plateau, the network is building features, not waiting on a readout.** The current features never contain the answer early (F15, X11).
4. **The destination is partly visible from the run's own motion, but only in the stable wiring, and only a short way ahead** (F16, X12). Acting on it skips 4–14% of the walk. Whether the *direction* matters, rather than any kick, is untested (I13).
5. **Under shared error, step size is a steep lever on which branch learns what** (F18). It moves both kinds of function in the stable wiring as preregistered, but it is not the only lever: at matched times speed splits roles transiently, the chain holds memorization regardless of speed, and the higher-decay branch carries the grokking jump (X14, I15, I16).
6. **Chain depth carries an instability that output scaling does not remove** (F17, X15). The deep chain can learn, but its learning-rate window is empty: too slow below, collapsing above.

**What the results do not say**
- Nothing about learners other than AdamW-trained networks on modular addition, apart from the one program-prior reference. "Hundreds of times more data" is relative to that prior.
- Nothing about *why* the data gap exists: whether averaging (the statistical-query argument), the shortcut (masking, H3), or the parameterization. No experiment yet separates these.
- Nothing about whether wiring can separate memorization from generalization *by design* (X10 stands; I15 is an exploratory lead).
- The data floor was measured at one learning rate and weight decay per graph. Regularization is known to move the training fraction at which grokking happens (Power et al. 2022), so "1,044" is a floor for this setting, not for gradient descent in general.
- Nothing about kernels: "speed" was the nominal step size, not each branch's effective speed.
- No random-direction control for the jumps, and no replication of the sym/asym splits beyond 10 seeds on one task.

**Open questions these results sharpen**
- Is the data gap closable inside gradient learning (perturbation credit I12, priors, structure), or is it the signature of an averaging learner (I6, H9, H10)?
- Is the drift's direction informative (I13), and why is the chain's trajectory unpredictable (I14)?
- Is there a wiring-set division of labor (I15), and is a circuit formed in one branch and copied to another (I16)?
- Since structures need their own settings (X15), how would a search over structures compare them fairly (I2, L8)?

---

## Findings

### Established or strong

**F1 — First-order equivalence of graphs at zero.** *Established (derived and tested), expressivity/optimization.*
- Every graph in the parent-selection family has the same first-order learning step at zero branch weights. Topology enters only at second order in branch weight.
- The parallel arm (I+A+B) is a single matrix trained at doubled lr.
- A sequential chain trained from identity keeps det W > 0, so it can never reverse an orientation; the parallel graph can.
- Evidence: [phase0_linear_theory.md](phase0_linear_theory.md); history, Pivot 16.

**F2 — Longer composition chains have lower stable learning-rate ceilings.** *Strong lead (preregistered, one task), optimization.*
- Across all 24 four-block graphs, τ_b(C, log lr*) = −0.836, p < 1e-4. Root readers Q have a smaller, opposite-signed effect.
- The failure mode is collapse after fitting, not divergence.
- Caveats:
  - the floor effect;
  - "stable" partly means "not yet grokked";
  - 3 seeds, one task.
- Evidence: [Study 2](../experiments/stability/RESULTS.md).
- Replicated qualitatively in Study 3 (all 8 chain runs unstable), Study 4 and the Phase 5 pilot (asym unstable at every lr tried).

**F3 — Parallel routes grok fastest on modular addition.** *Established within the task (best-vs-best, 10/10 seeds), optimization.*
- par vs seq: P = 1.00 at each arm's tuned lr.
- Reason: the rule is shallow and needs no composition.
- Evidence: [Study 1](../experiments/grokking/RESULTS.md), A3.

### Suggestive (exploratory, but robust or repeated)

**F4 — Memorization and generalization co-locate.** *Suggestive, strong; mechanism.*
- In 6 of 7 mixed graphs, the blocks carrying memorization at its peak later carry generalization, in the same proportions: cosine 0.98–0.999, against a permutation null of 0.27–0.76.
- While memorizing, a block's support for the true label on unseen inputs goes *negative*. At grokking it flips positive in the same block: the generalizing function grows in the path that memorized.
- Evidence: [Study 4](../experiments/paths/RESULTS.md), Q-A.
- Not yet tested at a shared stable lr, or on a deep-rule task.

**F5 — No path destroys another's memorization.** *Suggestive, strong; mechanism.*
- The cancellation ratio is 1.00 at every time point in every graph. Memorization is removed by shrinking in place.
- Evidence: Study 4.

**F6 — Topology decides which path carries the function; the learning rate moves it.** *Suggestive (3 seeds per cell), optimization.*
- At high lr, function migrates onto an independent short reader, and only where one exists (`pk0120`: share 0.43 → 0.87 from lr 1e-3 to 1.6e-2).
- At moderate lr, the composed path wins even against two short readers.
- Evidence: Study 4, Q-B.

**F7 — Composition's grokking delay is mostly the learning-rate assignment.** *Suggestive, optimization.*
- Within a graph, delay × lr is roughly constant. τ(C, delay) = 0.447 ≈ τ(C, 1/lr) = 0.445; at the shared lr 2e-3, τ = 0.02.
- A residual cost per unit of lr·wd·steps remains: about 6.5 for short graphs vs 11–19 for chains. Untested.
- Evidence: Study 4, Q-C.

**F8 — Chain instability does not come from large weights.** *Suggestive (2 seeds), optimization.*
- The chain collapses at about half the parameter norm of the stable parallel net.
- Stronger weight decay makes it *worse*: more collapses at wd 3, no fit at wd 10.
- Evidence: [Study 3 calibration, round 4](../experiments/composition/CALIBRATION_NOTES.md).

**F9 — End-of-run composition saturates after grokking.** *Suggestive (3 seeds), measurement.*
- c ≈ +0.95 for both branches in every arm and at every lr; R within ±0.08.
- During the plateau, composition does vary (−0.15 to +0.46).
- Evidence: [Phase 5 pilot](../experiments/fastslow/PILOT.md).

**F10 — Per-branch weight decay changes branch capacity, not only time constant.** *Suggestive (3 seeds).*
- Weight decay 2 shrinks a branch's output about 3× at lr 2e-3, and about 20× at lr 1e-3.
- Evidence: Phase 5 pilot.

**F11 — In a chain, the stream does not build the intermediate as a stage.** *Suggestive (2 seeds, chain unstable throughout).*
- On ((a+b) mod p)·c: the parallel net represents s = a+b explicitly in every block. The chain writes a joint function of (s, c) at its last block, and s is never linearly decodable.
- This cannot be read as a topology effect until the chain trains stably.
- Evidence: Study 3 calibration, round 4.

### From "diagnosing the walk" (E1, E2; 23 Oct 2026; [RESULTS](../experiments/walk/RESULTS.md), independently recomputed)

**F13 — Gradient descent needs hundreds of times more data than a program-prior learner, and more steps barely help.** *Strong (measurement, preregistered design, recomputed).*
- A Bayes mixture over short arithmetic programs pins down modular addition after **3 examples**.
- Gradient descent needs **1,044 examples** (sym) before it groks at all, even with up to 60,000 steps. asym needs between 1,044 and 1,392.
- On streams, its production point sits at that floor (≈ 1,100–1,390). Four times more steps per example lowers it only slightly.
- Scope: one task, one optimizer, two wirings.

**F14 — Under predict-then-learn scoring, gradient descent is worse than not learning for most of a stream.** *Strong (measurement).*
- After memorizing what it has seen, it predicts each new pair confidently wrong: 12–29 nats per pair against 4.1 for uniform. Its total code length is 4.5–7× uniform.
- More steps per example made this *worse* (sym, lr 1e-3: 26k → 38k nats).
- This is the anti-generalization of memorization (F4) measured as a cost.

**F15 — The readout is not the bottleneck; the features are.** *Strong (E2 P1, as predicted).* A readout re-solved on frozen features leads the network by about one eval interval at most (median relative lead 0.006).

**F16 — In the all-parallel network, the destination is partly visible from the run's own drift, and acting on it helps modestly. In the chain network, it isn't visible.** *Suggestive (exploratory split, plus preregistered P2 in sym).*
- sym: extrapolating the recent drift reaches 50% test accuracy 15–22% earlier than training, in 10/10 seeds.
- Jumping along it saves 100–400 steps (4–14%), with no more instability.
- asym: no lead; jump outcomes are chaotic.
- Not separated from "any kick helps": there was no matched random-direction control (I13).

- *2026-10-25 (priors Q2, preregistered P5 supported):* the drift jump beats a null matched in per-tensor norm change in all 4 sym cells (pooled median 137.5 steps; 9/10 or 9/9 seeds per cell).
  - So the drift's *direction* is informative, beyond "any kick helps".
  - Qualifications: the effects are small (100–475 steps on runs of about 2,900); the rescale-only arm *costs* steps, so part of the margin is the null hurting; and a plain random kick matched the drift in one cell (jump 1,000, H 1,000).
  - F16 stands in a narrower form: a short-range, informative direction.
- *2026-10-25 (the researcher's reading, adopted):* **Q2 is essentially a result about momentum, not about seeing the destination.**
  - The drift is an exponential average of the parameter updates over about 500 steps. A jump of H steps along it is a one-off, long-timescale heavy-ball extrapolation, closely related to Lookahead and Grokfast-EMA (prior art: amplifying the slow component of the gradient accelerates grokking).
  - What the nulls add: the gain comes from the averaged update *direction*, not from the norm change it carries, nor from any kick. That is the premise of momentum, now confirmed in this setting.
  - The saving (4–14% of the walk) is no larger than what a modestly higher learning rate gives (Q1 scratch: higher rates grok sooner in sym).
  - It says nothing about a horizon beyond the current heading. H5's "visible destination" reading of F16 is withdrawn: what is visible is the heading, for a few hundred steps.

**F17 — The chain's stability ceiling survives output rescaling.** *Established (preregistered E4 P0, and the H8 falsifier met; recomputed with four alternative summaries).*
- The longest-chain graph (C4) has essentially no stable region at learning rates from 5e-4 to 6.4e-2, under std, a uniform 1/√L multiplier, or a per-block 1/√depth multiplier.
- C1 is stable up to 0.016–0.032.
- Strengthens F2.
- "No stable region" does not mean "cannot learn." At lr 2e-3, C4 groks in 3/3 seeds under std and P1 (2/3 under P2), reaching test accuracy ≈ 0.95–0.97, but with 14–22 collapses in 12,000 steps. Its window is squeezed from both sides: below about 2e-3, C2–C4 do not fit within 12,000 steps; above it, C4 collapses repeatedly. For C4 the two limits meet. (Read from `experiments/walk/results_data/e4_runs.json.gz`, wd = 1.)

**F18 — Under shared error, step size moves credit steeply, but it is not the only thing that sets credit.** *Mixed (E3: preregistered P1 and P3 supported in sym; P2 falsified as written in asym; matched-time reanalysis exploratory).*
- In the stable wiring, at equal memory time, the faster branch takes both kinds of function: about 2.8 logit units per unit of log speed ratio, steeper than the lazy-regime prediction.
- Compared at matched times, three things show up beyond speed:
  1. **Speed alone splits roles transiently** (sym lr-only, at mid-plateau): the rule goes to the faster branch first, while memorization stays put or moves slightly the other way.
  2. **In the chain-plus-shortcut graph, memorization sits in the chain regardless of nominal speed,** and a speed-independent split remains even at equal speeds: generalization spreads across both branches while memorization stays in the chain.
  3. **The higher-decay branch carries the grokking jump, then hands it back** within about 1,500 steps.

### From the "priors" round (25 Oct 2026; [RESULTS](../experiments/priors/RESULTS.md), independently recomputed)

**F19 — The data gap is the prior, not the search.** *Strong (preregistered P2, recomputed, robust to every matched criterion tried; one task, AdamW).*
- In the all-parallel network, a learned rule survives on a subset only down to about where training from scratch finds it: both thresholds are 870 pairs, so the search factor S ≈ 1.
- At grid resolution S is 0.8–1.25. It is at most 1.5 under matched variants, and 2 only when semi-grokked runs (50–85% test accuracy) count as keeping the rule.
- On a log scale, search accounts for 0–12% of the gap to the program-prior learner (870 vs 3). The prior (the equilibrium preference of AdamW, weight decay and this architecture) accounts for the rest.
- **Consequence:** better search inside this model class cannot close the gap. That covers credit routing, competing members, growth criteria, or optimizers that settle at the same equilibrium. Only a different prior can.
- **Prior art:** Varma et al. 2023 found S ≈ 2 in a transformer under a lenient criterion; we match that only under the same leniency.

**F20 — Width is not a lever on data efficiency here.** *Strong for the scratch floor (P4b), suggestive for the rest.*
- Narrowing 14× or widening 4× leaves the scratch floor at 870 pairs.
- The ungrokking threshold does not decrease with width (the opposite of Huang et al.'s transformer result; X17). Narrow is lowest or tied (522, borderline).
- Base against wide is decided by single-eval dips, so no ordering is claimed.
- Generic capacity restriction does not buy efficiency at the level of a member (I18, generic version).

**F21 — In the chain-plus-shortcut network, the extra data cost shows up on both sides.** *Suggestive; the preregistered P3 was a tie and is uninformative.*
- Ungrokking and scratch thresholds are both 1,218 pairs (sym: 870). That places the wiring's cost in the prior at least as much as in the search.
- Fragile: the asym floor rests on 2 of 3 seeds at one learning rate, and 24 asym scratch runs cross 0.95 without sustaining it.

### From rung 0 of the dynamic-depth ladder (27–28 Oct 2026; [REV3](../experiments/ptr/REV3.md), [PILOT1](../experiments/ptr/PILOT1.md), [PILOT2](../experiments/ptr/PILOT2.md))

**F22 — A shared block trained with a fixed loop or oracle halting iterates 4× beyond its training depth; the same block trained jointly with a learned halting policy does not.** *Strong (10 paired seeds per arm, preregistered arms, one task); mechanism.*
- LR (the stability exit) on the fixed-loop block: 1.00 to k = 16, 0.93 at 32 (trained on k ≤ 8); the oracle-trained block 0.92 at 32. On every learned-halting arm's own block it fails at the arm's own failure point (0.26 / 0.74 / 0.36 / 0.23 at k = 16 / 16 / 24 / 8 for LH-s / LH-Δ / LH-s24 / LH-s-k4).
- The heads are not the limit: chasing λ = 0.000 from step 2k in every arm; LH-s24's head continues to 36 loops.
- The failure point scales with the chain lengths seen (≈ 1.5·k_max) and moves out with the loop horizon (T = 24 → ≈ 22).
- Evidence: Entry 59; REV3 tables "LR@arm", "fixed@arm", λ by loop.

**F23 — Extrapolation under a learned halting policy is reached mid-training and then lost.** *Strong (8/10 seeds in each of two arms); mechanism (late-training degradation of the block).*
- At 4k steps LH-s and LH-Δ reach 0.89 and 0.88 at k = 16 (8/10 seeds ≥ 0.9 each); at 8k, 0.22 and 0.65 (0/10, 1/10). LH-s24 keeps it (5/5); LO and LF never lose it.
- Pilot 2's "fails beyond ≈ 1.5× the horizon" was this state at the end of training.
- Evidence: Entry 59; REV3 "Extrapolation over training".

**F24 — A change-reading halting head with a fast-enough head learning rate halts at k + 1 and extrapolates 4× with no decay.** *Exploratory (2 seeds, the gate's ×10 cell; not the preregistered cell); mechanism if confirmed.*
- 1.00 at every k ≤ 32, steps k + 1.1 (cheaper than the one-parameter exit's k + 1.3), LR on its own block 1.00, 0.85 → 1.00 at k = 16 over training. Heads at w_KL ≈ −3 to −5 against −0.66 at ×1.
- Confirmatory run: D93.

**F25 — Oracle halting with a dominant one-hop share does not learn the hop.** *Strong (0/10 runs at tails ≤ 10⁻¹; 1/10 at factor 1); optimization.*
- Training accuracy at k = 2 stays at chance for 6k steps while the loss (71–76 % one-hop queries) falls to 0.05: the one-loop map wins and the reusable hop is never learned. A block pools signal across depths only when training iterates it on every example.

**F26 — Unique-layer depth learns hops sequentially; the shared block does not.** *Strong (preregistered P5, 5–10 seeds); optimization.*
- S8 (lr 1e-4, bracketed): k = 1 / 2 / 3 / 4 at 500 / 2,500 / 6,600 / > 25k steps; LO all eight by 3,600; LF by 3,200. The static stack is 2.6× slower to k = 3 when the k ≥ 3 share drops 7×; the shared block under LH-s 1.3×.

**F27 — On this task the one-parameter stability exit beats every learned policy that was preregistered.** *Strong (10 seeds); performance.*
- Geiping-style KL exit on the fixed-loop block: 1.00 to k = 16, 0.97 / 0.93 at 24 / 32, k + 1.1–1.6 steps, flat across thresholds 10⁻⁴–10⁻². Only the exploratory ×10 change head (F24) is cheaper.

### Derived constraints (arguments checked numerically during design)

**F12 — Under AdamW, decoupled weight decay is itself the time filter.** *Established (derivation plus check).*
- The memory time constant is τ = 1/(lr·wd).
- AdamW cancels most pre-optimizer gradient scaling.
- A linear time-invariant filter of the gradient cannot select between memorization and generalization: it rescales every persistent component by the same factor.
- Evidence: [fastslow/DESIGN.md](../experiments/fastslow/DESIGN.md) §2 and [DESIGN_checks.md](../experiments/fastslow/DESIGN_checks.md).

---

## Failed predictions and refuted ideas

These are the negative results. They rule things out, and are kept so that nobody rediscovers them.

**X1 — A side "sink" branch quarantines memorization (Study 1, H_U).** *Failed.*
- Memorization formed first in the chain. Once strong, the sink hosted both components. The localization index was never positive, and was significantly negative at s = 4 (−0.059).
- Lesson: static wiring under shared error does not quarantine memorization. This agrees with MemSinks, where localization needed an input-dependent routing signal.

**X2 — The sink hosts most of the rule at large multipliers (Study 1, H_mono criterion 2).** *Failed* at the re-tuned optima: mean share 0.18. Only "larger multiplier → faster" survived, and that one is confounded with step size.

**X3 — A chain groks without weight decay, a parallel net doesn't (Study 1, H_depth).** *Inconclusive:* 0 of 30 runs grokked at wd 0. It was a floor effect, not a test.

**X4 — Short paths construct, long paths memorize (Study 4, H1).** *Nominally supported, substantively failed.*
- The pooled effect came almost entirely from one unstable grid-edge graph. Elsewhere, D = 0.02–0.05.

**X5 — An extra short reader lowers the long blocks' utility ("friction", Study 4, H2).** *Failed.*
- Both contrasts were also structurally flawed: sibling additivity, and an 8× lr difference.
- The only friction-like signal is small and exploratory: `pk0120` vs `pk0121` at a shared lr.

**X6 — Composition delays generalization as a topology effect (Study 4, H3).** *Nominally supported, substantively refuted:* it is the learning-rate assignment (F7).

**X7 — Chain instability is the Adam "slingshot" from weight-norm growth.** *Refuted in that form* (F8).

**X8 — The low-frequency part of the gradient is the generalizing part.** *Refuted by derivation* (Phase 5 design, 22 Oct discussion).
- With full batches, a fixed training set makes the memorizing signal static. It points the same way every step until fitted, so it is "low frequency" too.
- The axis that separates the two is agreement across examples, not frequency in time (F12).
- Minibatch noise is a third, separate component.

**X9 — Per-branch time constants give branches different roles, as measured at the end of training (Phase 5B as designed).** *Measure failed before any test* (F9).
- The preregistered statistic could not move by the preregistered effect size.
- The underlying idea is on hold, not refuted (see H2).

**X10 — Static forward wiring separates memorization from generalization under shared-error training.** *Not supported anywhere it was tested:* Studies 1 and 4, and the pilot.
- This is the program's original bet in its simplest form. Across every wiring tried, topology changed *speed, stability and which path carries everything*, never *what each path carries*.
- Scope: one shallow task (modular addition), AdamW, parent-selection graphs.
- *2026-10-23:* the first possible exception is exploratory: E3's matched-time reanalysis in asym finds generalization spread across both branches while memorization stays in the chain, even at equal speeds (F18, I15). Not preregistered, branch-level only. X10 stands until I15 is tested.

### From rung 0 (28 Oct 2026, [REV3](../experiments/ptr/REV3.md))

**X18 — "What the halting policy reads" decides extrapolation (rev-3 P1, P2).** *Falsified as preregistered.* LH-Δ (change features) at the gate-chosen head rate: 0.65 at k = 16, 0.15 at 32; LH-s24 (state head, T = 24): 0.96 at 16, 0.44 at 24. In both the stability exit on the arm's own block fails at the same point (F22): the block lost the signal; the head did not misread it. The ×10 head (F24) is the open exception.

**X19 — A learned change head is cheaper than the stability exit (rev-3 P3).** *Falsified at ×1* (k + 2.3 vs k + 1.3; the head commits late, post-answer λ 0.3–0.8). *Exploratory counter-evidence* at ×10 (k + 1.1).

**X20 — The oracle-halted block's time to learn a depth is insensitive to how rare that depth is (rev-3 P6a).** *Refuted in the opposite direction* (F25): at tails ≤ 10⁻¹ the oracle arm never learns k = 2.

**X21 — The state head's failure tracks the loop horizon T (pilot-2 reading, rev-3 P2's LH-s-k4 line).** *Falsified:* with k_max = 4 and T = 10 the failure is at k ≈ 6–7 (≈ 1.5·k_max), not ≈ 15; with T = 24 it is at ≈ 22. The block's trained range, not a fixed horizon.

### From "diagnosing the walk" (E1–E4, 23 Oct 2026)

Preregistered predictions that failed, and ideas the runs killed. Details in [walk/RESULTS.md](../experiments/walk/RESULTS.md).

**X11 — The answer is readable from the current features before the network produces it (E2 P1, readout visibility).** *Failed, as the pilot predicted.*
- A logistic readout refit on frozen features leads the network by a median 0.006 of the time to 0.9 test accuracy, against ≥ 0.2 required. The lead is mostly exactly one 100-step eval.
- Kept here because it was a preregistered prediction. It was expected to fail; its failure is the positive finding F15 (the features, not the readout, are the bottleneck).

**X12 — The run's own drift points at the destination well ahead of time, in both wirings (E2 P1b, pooled).** *Failed pooled* (median relative lead 0.077, +11/−6, p = 0.33).
- Holds in sym alone (10/10 seeds, 15–22% earlier), and not at all in asym. That split is exploratory (F16).
- Extrapolating further (H = 4,000) overshoots: median relative lead −1.4.

**X13 — Co-location persists under any speed ratio in the chain-plus-shortcut graph (E3 P2, asym).** *Falsified as written* (τ = 0.80; shifts up to 2.4 logits).
- Generalization followed the speed ratio while memorization stayed in the chain.
- The exploratory matched-time reanalysis reads this differently: most of the *trend* came from comparing memorization early with generalization late (L14). What remains is a speed-independent gap: generalization spreads, memorization stays in the chain (I15).
- In sym, P2 was inconclusive: significant but small, and dependent on the logit clip.

**X14 — Credit is set by speed, so memorization and generalization always move together (H1, strong form).** *Refuted, on exploratory evidence.*
- At matched times in sym lr-only, speed sent the rule to the faster branch (+6.9 logits, 10/0 seeds) while memorization moved slightly the other way (−0.25, 0/10).
- In asym, the chain holds memorization at every speed ratio.
- The weak form survives (F18): speed is a steep determinant of credit (≈ 2.8 logits per unit log ρ in sym).

**X15 — A depth-aware output parameterization aligns stability regions across wirings (H8).** *Failed* (E4; F17).
- Neither a uniform 1/√L nor a per-block 1/√depth multiplier gives C4 a stable region. Under P1 the C1–C4 gap is ≥ 7 grid steps; under P2, 5.
- Consequence: structure comparisons and any search over structures need per-structure settings, or a stabilizer that changes the model class.

**X16 — A decay-aware extrapolation of the drift predicts where the weights are going.** *Refuted.*
- Proposed by the pre-run review as a fix to the linear operator. On real runs it predicted nothing (R² ≈ 0 in the operator pilot; median lead 0.0 in E2). It had been checked only on a weight with no gradient (L11).
- The plain linear extrapolation, capped at H ≤ 1,000, is what showed the sym lead.

**X17 — Wider networks generalize from less data (priors P4, prediction from Huang et al. 2024).** *Failed* in our MLP setting, under every variant tried.
- The ungrokking threshold did not decrease with width, and the scratch floor did not move with width at all (F20).
- Our 24 Oct conjecture, the opposite direction (narrower generalizes from less data), isn't supported either. Narrow was lowest or tied on the ungrokking side only, and borderline.

---

## Open hypotheses

Each has a prediction and a falsifier. Most are arguments from the 22 Oct discussion and have not been tested.

**H1 — Credit by speed.** *Argument (first-order kernel reasoning); consistent with F4 and F6.*
- **Claim:** under one shared error, component k's share of fitting a function-space mode m is η_k λ_km / Σ_j η_j λ_jm, i.e. learning rate × kernel eigenvalue along that mode.
- **Predicts:**
  - co-location (F4);
  - learning-rate migration (F6);
  - "fast memorizer plus slow generalizer" being self-defeating (H2).
- **Falsifier:** measured shares are not ordered by η·λ; a slower component wins a mode it is slower on.
- **Cheapest test:** during training, measure each branch's credit share along the rule direction and along per-example directions, across learning rates and wirings. Compare with the formula on Study 4 graphs.
- *2026-10-23:* **E3 results → partly supported, partly falsified** (F18; [RESULTS](../experiments/walk/RESULTS.md)).
  - Speed moves credit steeply, and in the stable wiring it moves both kinds of function together when measured as preregistered.
  - But at matched times speed alone splits roles transiently, and in asym the wiring sets a speed-independent split.
  - H1's strong form ("credit is set by speed, so memorization and generalization always move together") is **refuted**.
  - What survives: speed is a strong, steep determinant of credit, and kernel-level effective speed is untested.
  - The refuted strong form is recorded as X14.
- *2026-10-22:* argument → **preregistered as E3** ([walk/PREREGISTRATION.md](../experiments/walk/PREREGISTRATION.md)): direct manipulation of the branch speed ratio ρ at fixed lr·wd.
  - Revised after review: lr-only and wd-only controls separate speed from weight scale.
  - P2: co-location persists (trend test in logit space).
  - P3: step size alone moves credit.

**H2 — Under shared error, a fast memorizer takes every mode, the rule included.** *Argument (corollary of H1).*
- **Claim:** complementary learning systems work in brains because the cortex learns from replay, not from an error it shares with the hippocampus.
- **Falsifier:** a fast flat-kernel component and a slow component, under shared error, end up dividing rule and exceptions.
- **Cheapest test:** a planted two-branch case with known kernels.
- *2026-10-23:* **not directly tested; weakened.** E3's branches had the same kernel, so it is not H2's test. As preregistered, the faster branch took both kinds of function (consistent with H2). But at matched times, speed alone sent the rule to the faster branch first while memorization stayed or moved the other way (X14). So "the fast component takes every mode" does not hold at every moment, even with identical kernels. The flat-kernel test is still open.

**H3 — Masking.** *Argument; partial support from pilot plateau dynamics.*
- **Claim:** once the memorizing component fits the training set, it removes the error the generalizer needs. Weight decay then drives the plateau.
- **Open sub-question:** is masking harmful, or the mechanism by which grokking happens at all?
- **Falsifier:** removing masking (e.g. the unmasked arm) leaves the generalizer's trajectory unchanged.
- *2026-10-24 (interpretation, from discussion):* summed branch logits are a product of experts, and shared error is the correct credit for that kind of conjunctive composition. Masking is then "explaining away": once one explanation (memorization) accounts for the data, the others get no signal. Bayesian explaining-away ranks explanations by prior × likelihood; gradient descent ranks them by which arrives first, and weight decay slowly corrects towards the prior. On this reading, grokking is explaining-away arriving late, and masking is a symptom of a prior that undercharges lookup tables (I17) as much as a credit problem.
- *2026-10-23:* **still untested directly** (the unmasked arm was never run). Consistent observations, none decisive:
  - at the peak of memorization, total generalization is at or below zero in most E3 runs: memorization pushes unseen inputs toward wrong answers;
  - E1 measures that as a cost, 12–29 nats per new pair (F14);
  - E2 shows the plateau is spent building features, not fitting a readout (F15).
  - None of these separates "memorization removes the generalizer's error" from "the generalizer is slow anyway".

- *2026-10-25:* **F19 supports the "prior" reading of masking.** Below about 870 pairs, the equilibrium itself prefers memorizing (a learned rule ungroks), so masking there is not a failure of credit or search. Explaining-away, as read in the 24 Oct note, follows the prior. The open question shifts from "how to stop memorization masking the rule" to "what prior would make the rule cheaper than a lookup table at small data".

**H4 — The grokking delay is the price of overcompleteness.** *Prior art (partly answered).*
- **Claim:** a hypothesis that can express the rule but cannot memorize finds it without a plateau. Masking needs two ingredients: shared error, and a hypothesis that offers the shortcut.
- **Prior art:** [Omnigrok](https://arxiv.org/abs/2210.01117) (Liu, Michaud & Tegmark) eliminated grokking on algorithmic data by constraining weight norm. The lazy-to-rich account (Kumar et al.) is related.
- **Open part:** whether *structural* constraint (wiring, capacity) does the same; and whether constrained hypotheses get stuck instead, i.e. how much of overparameterization's value is easy optimization.

**H5 — The answer is visible long before it is reached.** *Open; the researcher's "minima on the horizon".*
- **Claim:** the final generalizing solution is predictable early in training from signals the learner already has: cross-example gradient agreement, internal representations, the direction of drift.
- **Falsifier:** prediction quality stays near chance until close to grokking. That would mean the walk is doing irreducible work.
- **Why it matters:** it decides whether "seeing instead of walking" is possible from information a learner already holds. It is the most decision-relevant open question.
- *2026-10-22:* **partly prior art.** [Two Speeds of Learning](https://arxiv.org/abs/2605.27078) finds that representations improve before the test-accuracy jump, using linear probes on frozen features. [Canalization Before Generalization](https://arxiv.org/abs/2608.25813) detects the coming generalization about 4,000 epochs early, from weight-decay perturbation responses, while static progress measures lag.
- *2026-10-22:* **preregistered as E2**, adding what those papers lack:
  - the timing of a lead, from cheap static operators (logistic readout refit, drift extrapolation);
  - an intervention: jumping ahead along the drift.
  - Instrument pilot (p = 23, 2–3 seeds; not evidence): the refit readout tracks the network with no lead, which points to the features rather than the readout as the bottleneck. Drift jumps saved 100–400 steps in sym and nothing in asym.
- *2026-10-23:* **E2 results** ([RESULTS](../experiments/walk/RESULTS.md)):
  - **P1 failed, as expected:** the readout version of H5 is refuted (F15).
  - **P1b failed** pooled as preregistered, but holds in sym alone (10/10 seeds) and not in asym (F16).
  - **P2 supported** in sym, modestly.
  - H5 status: argument → **suggestive, wiring-dependent.** The horizon is partly visible in the stable network, but only a fraction of the walk can be skipped by extrapolation. The large gap is in data (F13), not in steps.

**H6 — Agreement across halves separates the signals without labels.** *Argument (derivation).*
- **Claim:** for disjoint halves A and B of the training set, E⟨G_A, G_B⟩ measures the shared (population) part of the gradient, and ‖G_A‖² − ⟨G_A, G_B⟩ measures the memorizing part.
- **Falsifier:** the estimate disagrees with the test-label-based gen/mem gradient split.
- **Cheapest test:** compare it against the existing 5A measurements.
- *2026-10-23:* untested; E1–E4 did not touch it.

**H7 — Chain instability comes from compounding Jacobians.** *Argument (Phase 0 Prop. 3); open since Study 2.*
- **Claim:** sharpness, or the effective step on the end-to-end map, grows with chain length C.
- **Falsifier:** sharpness does not scale with C at matched training progress.
- **Cheapest test:** track the top Hessian eigenvalue on a subset of Study 2 graphs.

**H8 — A depth-aware parameterization aligns stability regions across structures.** *Argument (μP literature).*
- **Claim:** with μP-style depth scaling, the optimal lr is shared across structures, so any remaining instability differences are structural.
- **Falsifier:** optimal lrs still differ by structure under that parameterization.
- **Cheapest test:** a learning-rate sweep per structure, with and without the scaling.
- *2026-10-23:* **E4 → NOT supported** (F17). Neither uniform 1/√L nor per-block 1/√depth output scaling aligns stability across wirings: the deepest chain has no stable region either way. Structure comparisons and structure search need per-structure settings, or a stabilizer that changes the model class.
  Recorded as X15.
- *2026-10-22:* **preregistered as E4.** Stability regions for C = 1–4 under std, uniform 1/√L and per-block 1/√depth output multipliers, on an lr × wd grid. Depth-μP's known result is transfer across depth; whether it transfers across wirings at fixed depth is the open part.

**H9 — Competing hypotheses scored by prequential surprise avoid masking.** *Argument.*
- **Claim:** if hypotheses are not summed or trained jointly, each predicts alone and is scored by its surprise on each example *before* learning from it. Then memorization earns nothing at the level of which structures survive. Masking inside a branch disappears if the branch cannot express the shortcut, and true exceptions go to a separate memory store.
- **Falsifier:** prequential selection still favors overcomplete or memorizing structures; or constrained structures cannot be optimized.
- *2026-10-23:* **the criterion has teeth (E1).** Scored prequentially, a memorizing gradient learner costs 12–29 nats per new pair against 4.1 for a uniform guess, and 4.5–7× uniform over a stream (F14). A selection by prequential score would reject it long before it produces the rule. Whether such selection finds better structures, rather than merely rejecting bad ones, is untested. E4 adds a constraint: structures differ in their stable settings (X15), so a selection must not score hyperparameter luck.

**H10 — Schema assimilation separates a real learner from gradient descent.** *Argument, with an animal analogue (Tse et al. 2007).*
- **Claim:** once a rule is learned, new consistent examples should be absorbed almost immediately and generalize, while exceptions should be stored without disturbing the rule. Standard gradient training will walk for both and may overwrite old examples.
- **Falsifier:** standard training already assimilates consistent examples in one or two steps without interference.

---

**H11 — Joint training with a halting policy degrades the shared block through the policy's objective, and the degradation goes to whichever part can move.** *Argument from rev-3 data (F22–F24); untested.*
- **Claim:** the PonderNet KL-to-prior term (and more generally any pressure from the halting loss) reaches the block through the head's input when the head's own parameters move slowly; it pushes intermediate states to look "settled"; inside the trained range the cross-entropy holds the block to the answer, outside it the pressure wins and iteration fails. A head that moves fast absorbs the pressure itself.
- **Predicts:** LH-Δ ×10 at 10 seeds extrapolates (≥ 8/10 ≥ 0.9 at k = 32) with no decay; LH-Δ ×1 with its input detached from the block (stop-gradient) extrapolates; β = 0 at ×1 extrapolates; LH-s ×10 extrapolates too (the input does not matter when the head absorbs the pressure).
- **Falsifiers:** stop-gradient still decays (the damage is not through the head's input); β = 0 still decays (not the prior); ×10 at 10 seeds < 6/10 at k = 32.
- **Cheapest test:** D93, ≈ 4 T4-hours.

## Ideas (not yet hypotheses)

| ID | idea | origin | status |
|---|---|---|---|
| I1 | The backward graph as a design variable: one-way coupling, branch dropout, cross-fitting, stop-gradients | 22 Oct discussion | **Deprioritized.** The researcher: patching the backward graph ignores the observation that the structure of learning is flawed. Branch dropout is the least ad hoc variant. |
| I2 | A population of structures (e.g. wirings) in VRAM, refined by gradients, selected by prequential surprise (sequential Monte Carlo over models) | researcher, 22 Oct | Open. The fleet trainer can already hold the population. Needs H8 and stability regions first, or selection rewards whatever fits the hyperparameters. |
| I3 | A two-system learner: a fast memory store, plus a slow learner trained from replay, with no shared error | complementary learning systems discussion | Open. Matters only for streams, not fixed datasets. |
| I4 | Amortized / in-context learner across a task family (moduli, group operations): learning moved into inference | 22 Oct discussion | Open. "Gradients at the meta level, reasoning at the task level." |
| I5 | A rule-plus-exceptions task, so memorization persists and composition stays measurable | pilot analysis | Open. Implemented as `modadd_exc`; the exception fraction is uncalibrated. |
| I6 | Sparse parity as a testbed with a proven separation between averaging and reasoning learners (statistical-query lower bounds) | 22 Oct discussion | Open. |
| I7 | Group-product composition task (abelian control vs S4), gated on all-short networks failing | Phase 5 discussion | Open. |
| I8 | Topology as a hypothesis about how the data is generated: a learner discovers its own wiring from per-example surprise | 22 Oct discussion | Open. Crosses the repo's "no architecture search" rule; that boundary is the researcher's to redraw. |
| I9 | Phase 5 B–D as designed: per-branch time constants, update routing, consolidation, within one shared loss | Phase 5 design | **On hold** (X9, H1–H2). Stage-1 grids deleted unlaunched. |
| I10 | Measure how much information a training run has versus what it uses: steps or examples needed to determine the function, versus to produce it | 22 Oct discussion | **Measured in E1** (22 Oct design; results 23 Oct, F13–F14: 3 examples vs about 1,044; worse than uniform for most of a stream). Prequential code length of gradient descent on streams vs a program-prior learner (k* = 3 examples), plus the data floor on fixed datasets. It became a measurement, not a hypothesis test, after the review showed the original primary prediction could not fail. Nearest prior art: weight-compression complexity of grokking (no ideal-learner reference); Power et al. on training fraction. |
| I11 | Identity-path credit as a backward-graph experiment: each block learns from the error at the top of the residual stream only, dropping the terms that pass back through later blocks. If co-location (F4), migration (F6) and credit by speed (H1) are unchanged, the wiring acts on learning almost entirely through what blocks read (the forward graph), not through how error reaches them. | Q Labs "Noise" (Oct 2026): a community record uses "identity-path credit" and trains well | Open. Parameter-free; keeps one loss. |
| I12 | Test the claim that backprop "explores the loss landscape poorly", and that this is why networks need so much data, on our toy, where data efficiency is exactly measurable: perturbation-based credit (Dust-style node perturbation, per-example noise) vs backprop, compared on the E1 scoreboard (data floor, prequential code length) and the grokking delay. | [Noise](https://qlabs.sh/noise), [Dust](https://qlabs.sh/research/dust/) | Open. Prediction from the statistical-query argument: perturbation credit is still an averaging learner, so any gain is a constant factor or regularization, not reasoning. It should not beat gradient descent on sparse parity. |
| I13 | Matched-norm random-direction jump control for E2: does jumping along the drift help more than an equally large kick in a random direction? Needed before "the drift direction helps" can be claimed, especially for the chaotic chain network. | E2 verification, 23 Oct | Open. Cheap. *2026-10-24:* **preregistered as Q2**, with a radial-matched null (the drift is largely a per-tensor rescaling, found by the pre-run review) plus random and rescale-only nulls. *2026-10-25:* **done** (priors Q2, P5 supported, with qualifications; see the F16 note). |
| I14 | Why is the chain network's trajectory not extrapolatable when the parallel one is? Curvature of the path, or collapse cycles (F2)? | E2, 23 Oct | Open. |
| I15 | The wiring-set division of labor in asym (E3, exploratory): generalization spreads across both branches while memorization stays in the chain, even at equal speeds. Preregister it at matched times, block-level, with a kernel measure of each branch's effective speed. | E3 matched-time reanalysis, 23 Oct | Open. The first positive lead on the original question since Study 1, and still exploratory. |
| I16 | The higher-decay branch carries the grokking jump, then hands it back (E3, exploratory). Is the circuit formed in the strongly regularized branch and then copied? | E3 recomputation, 23 Oct | Open. |
| I17 | Decompose the data gap into prior vs search. At each training-set size, compare what the regularizer charges (weight norm) for the rule solution and for the memorizing solution. Rule already cheaper below the data floor but not found → search failure. Memorizing genuinely cheaper there → prior failure. | 24 Oct discussion | Open. Decides between "fix the prior inside gradient learning" and "change the learning mechanism". Check the grokking-at-low-training-fraction literature first (L2). *2026-10-24:* **preregistered as Q1** of the priors round ([priors/PREREGISTRATION.md](../experiments/priors/PREREGISTRATION.md)), via the ungrokking protocol of Varma et al. 2023 (prior art: S ≈ 2 in a transformer): the threshold at which a learned rule survives on a subset vs the from-scratch threshold on the same pairs. *2026-10-25:* **answered (F19):** S ≈ 1 (matched criteria, 1.0–1.5); the gap is the prior. |
| I18 | The member-level claim: a restricted member (e.g. one whose primitive is a rotation by a learned frequency), chosen from a population by predict-then-learn scoring, learns modular addition from a handful of examples where the general network needs ≈ 1,044. | 24 Oct discussion | Open. One of the two claims the library/population program rests on. *2026-10-24:* its generic version (does width restriction lower the thresholds?) is preregistered as Q1 P4/P4b; prior art (Huang et al. 2024) predicts the opposite direction to our conjecture. *2026-10-25:* **generic version answered: no** (F20). Efficient members would need structure matched to the task, not less capacity. |
| I19 | The transfer claim: blocks grown or selected on one family of tasks make learning on held-out tasks efficient. Without it, a library is a lookup table of past solutions (memorization one level up). | 24 Oct discussion | Open. The other load-bearing claim. The choice of task family *is* the world prior and must be explicit. |
| I20 | Mixture of competing members with responsibility assigned by out-of-sample prediction, not current fit, plus a one-shot memory member that predicts uniformly on anything unseen. Prediction: the rule member claims rule examples once it predicts them, exceptions go to memory (the rule member is confidently wrong, worse than uniform: F14), and no member wins by memorizing. | 24 Oct discussion; Jacobs et al. 1991 | Open (argument). In-sample responsibility produces fragmentation (one expert per case) and starvation (masking by responsibility). *2026-10-24 revision:* only needed when members are trained jointly **and** the data is a mixture of regimes (e.g. rule plus exceptions). With one rule, independent members weighted by predictive evidence need no coupling at all. *2026-10-25:* **deprioritized by F19.** Credit between members is a search mechanism, and search is not where the gap is. |
| I21 | Growing a network by held-out residual fitting: for each (block, link, site), insert it without changing the function, fit it on half A of the error signal (several inits in parallel), score the code-length reduction on half B minus the cost of describing the addition. Prediction: held-out scoring adds a rotation-like block before any lookup-like one; training-set scoring does the reverse, and growth stalls once the training set is memorized. | 24 Oct discussion; Cascade-Correlation, GradMax, boosting | Open (argument). Greedy growth is itself an averaging procedure and fails on parity; library blocks set how far one growth step can jump. *2026-10-25:* **deprioritized by F19** as a route to data efficiency; still of interest as a way to search over priors (structures), which is where F19 points. |
| I22 | Library size has an MDL optimum (library cost + solution cost), not diminishing-but-positive returns; an oversized library memorizes the task suite. Measure held-out-task code length vs library size. | 24 Oct discussion | Open (argument). |
| I23 | The amortized rival: a network meta-trained by gradient descent across a task family to learn in context, compared with an explicit population of restricted members on held-out tasks (examples needed, code length, extrapolation outside the family). It uses exactly our tooling and is the approach validated at scale, so the explicit-population program must beat or complement it. | 24 Oct discussion | Open. |
| I24 | Express structural diversity as masks or primitives inside one shared tensor shape, so a population of structures batches as dense matmuls (the fleet trainer already batches same-shape independent runs). Heterogeneous topologies break batching. | 24 Oct discussion | Open (design constraint from tooling). |
| I25 | A generator of worlds (PFN-style) for a library of uniform, composable blocks with a separate memory store: random symbol bijection per world, length-weighted programs over modular arithmetic and group operations, an exact Bayes reference from `program_prior.py`, a sweep of world diversity, splits on held-out compositions and lengths, and rule-plus-exception worlds. Claims C1–C5 and what each needs: [generator_design.md](generator_design.md). | 26 Oct discussion; literature review of generator design | Open. Not yet designed as an experiment; decisions in §5 of the note are the researcher's. *2026-10-26 decisions (researcher):* hand-chosen primitives with exact solutions; chunking allowed and measured; no memory store yet; monolithic PFN vs routed library of uniform blocks; instrumentation is a primary deliverable (generator_design.md §6). *2026-10-26:* consolidated into [worlds/DESIGN.md](../experiments/worlds/DESIGN.md): 3 arms (monolithic PFN, open-loop and execution-conditioned composers) × 9 training-mix cells × 3 seeds; mod-7 two-register instruction set (297,845 functions at ℓ ≤ 6); stacked-run batching implemented and tested; GPU throughput test pending. |
| I26 | Exact step classification on the toy, plus the commit-vs-average framing. From any partial program on Z_7, the remaining distance to the target (or to the version space consistent with the context) is computable by enumeration. So every step the meta-model takes has an exact regret, 1 + d(s') − d(s): 0 = optimal, ≥1 = suboptimal, budget-fatal when it exceeds the cap, and dead when the state merges two inputs the target separates (irreversible at any length). The Bayes-optimal step policy (posterior mass of minimal programs extending each prefix) is also exact. A static PFN averages worlds in output space; the dynamic meta-model averages committed, self-consistent worlds in path space, and the two have the same optimum under log loss. The differences are in errors (a wrong valid world vs an invalid blend), joint coherence across queries, verifiability against the context, and where novelty is extrapolated. Commitment is also irreversible: most of the branch space is dead. | 26 Oct discussion; `experiments/worlds/branch_classify.py` | Open. *Measured 26 Oct* (25 targets of minimal length 4 × 32 first moves): optimal 5.9%, lateral 24.9%, away or beyond 69.3%, of which provably dead 42.2%. To be folded into the redesign as diagnostic, exact step-level reference, and an optional privileged-signal arm. *26 Oct, revision 2 review:* the step-level references need a tie-break prior over equivalent programs (π* is not unique), and branch classes need a precedence order (history Entry 32). |
| I27 | Rung 0 of the dynamic-depth ladder (history Entries 45–54): pointer chasing with hidden, heavy-tailed chain length; a shared block with learned halting against static stacks and an oracle; headline = halting extrapolation beyond the training depth with the oracle separating block from policy; plus the signal-pooling test. Design, review and code: [ptr/DESIGN.md](../experiments/ptr/DESIGN.md). | 27 Oct; review 1 recorded | Open. Implemented with run-stacking. *Pilot 1 (27 Oct):* G2 passes (oracle-halted block extrapolates to k = 32; injection essential); the fixed loop overthinks beyond its horizon; LH-s collapsed from a halting-init asymmetry (fixed, L23); static arms slow. *Pilot 2:* G4 passes in a narrow region (lr 3e-4, weak prior); the state-reading halting policy extrapolates only ≈ 1.5× its horizon (its reading of the state drifts with depth) while a change-based exit reaches 0.97 at 4×; unique-layer depth learns hops one at a time (k = 3 at ~10k steps vs all eight by 4k for the shared block). *Revision 3 (28 Oct, Entry 58):* the question is now what the halting policy reads (LH-Δ on standardized change features; LH-s24 with T = 24; LR on every arm's own block); reviewed twice and revised; a gate (pilot 3) then a 93-run confirmatory grid, preregistered P1–P7. *Ran 28 Oct (Entry 59, F22–F27, X18–X21, H11):* P1–P4 falsified as preregistered, P5 holds, P6a refuted the other way; the limit is the block trained jointly with the policy, not the head, and it is lost late in training; a ×10 head learning rate (2 seeds, exploratory) removes it entirely. D93 deferred to rung 3 (D95, Entry 60); rung 0 closed. |
| I28 | Rung 1 of the ladder (history Entry 61): symbolic generic circuits (HOP, PREV) with a learned Markov router over an explicit product state (symbol × pending × set-here), trained by the exact route marginal from next-token loss; the P-trigger is **deferred** (flag one state earlier: a SET setup step, a bit carried across a hop; PP runs reach a pending count never seen in training); arms: RM, the symbol-only null (F4), the local-trigger control (δ = 0), teacher forcing, the state-blind index table, the looped block (read at the oracle depth) and static stacks on the same tokens. Preregistered P1–P9; design, reviews and code: [route/DESIGN.md](../experiments/route/DESIGN.md), [route/review/](../experiments/route/review/). | 28 Oct; review 1 recorded | Open. Revision 1 (local trigger) found by review to be in-distribution for a Markov router (the reviewer's 1,800-step CPU run passed P1–P3); revision 2 implemented and tested (222 tests); review 2 applied (counterfactual probes for precedence and count; P5 relabelled; the symbol-only router's marginal trick). *Pilot (28 Oct, Entry 62, [PILOT.md](../experiments/route/PILOT.md)):* rates unchanged; at 2 seeds RM is exact in range, at length and on alternation, SET flips 100 steps after teacher forcing, PP fails on the untrained precedence with the count read ordinally, the symbol-only router is 0 committed / 1.00 marginal (p(target) 0.75) on P-words, the looped block learns hops and nothing of the P-rule; λ selects between the carried-bit route and an always-SET locally-readable program. Main grid (≈ 8 T4-hours) next. |

---

## Lessons about our process

| ID | what went wrong | rule adopted |
|---|---|---|
| L1 | Phase 5's headline measure could not move by the target effect size; found only by the pilot, after a day of instrumentation | Before building or running a study, plant the effect in a synthetic case and confirm the measure detects it. |
| L2 | A "constrained structure without plateau" experiment was proposed as new; Omnigrok had largely answered it, and our own literature review cites it | Check the literature docs and this ledger before proposing an experiment. |
| L3 | Each confound became the next study (Study 3: five calibration rounds) | When a confound appears, ask whether it changes the question before building around it. |
| L4 | Answers built into contrasts (Study 4 H2), a leaky rule measure (Study 1), an lr rule picking unstable settings | An independent pre-run review for every study (it caught these); measures checked on planted cases. |
| L5 | Learning-rate rules defined by first crossing a threshold rewarded collapsing regimes (Study 4, pilot) | Selection rules must include stability, e.g. a post-fit minimum or a collapse count. |
| L6 | End-of-run readings discarded the dynamics where the effect lives | Observe training trajectories and the gradient's components as primary data; end-of-run values are summaries. |
| L7 | Proposed patching the backward graph to rescue a hypothesis | When an observation says the frame is wrong, change the frame; don't patch around the observation. |
| L8 | Structure comparisons at fixed settings mixed "bad structure" with "bad settings" | Compare structures by their stability regions, or at each one's best, or under a parameterization that aligns them. |
| L9 | Implementation ran ahead of the question several times; the researcher had to pull the discussion back to the central goal | Before any implementation, state the question, what each outcome would change, and the cheapest decisive test. |
| L10 | E2's first readout operator (least-squares refit) was a weaker classifier than the network's own readout (0.88–0.91 vs 0.99–1.0 after grokking), so it could never show a lead. Caught by a sanity run before Kaggle (22 Oct). | Check every "what could the network do now" operator against the network itself on a grokked checkpoint: it must at least match it. |
| L11 | The pre-run review of E1–E4 (22 Oct) found six blocking problems: a primary prediction that could not fail (censoring); a stability grid whose ceiling sat below the known optimum and whose definition differed from Study 2; a manipulation (E3) confounded with weight scale; and lesson L5 repeated in E2. One of its proposed fixes was also wrong: a decay-aware drift operator that the review checked only on a weight with no gradient, which predicted nothing on real runs. | An independent review before every run, and check each fix too, by measuring the operator against what actually happens. |
| L12 | Before the review, several predictions could only come out one way: E1 P2 ("gradient descent is ≥ 5× worse"); E3 P1, with speed and scale pushing the same way. | For each prediction, write down a concrete result that would make it fail, and check that the design can produce that result. |
| L13 | E2's noise floor (H = 0 jump) measured only floating-point nondeterminism, not how a run responds to a perturbation, so "saving above the noise floor" was close to automatic. Also, the sidedness of the sign tests was ambiguous in the preregistration. | A null for an intervention must be an intervention of the same size in an uninformative direction. State the sidedness of every test. |
| L14 | E3's co-location measure compared memorization at its peak with generalization at the end, so migration over time read as a role split. My first matched-time reanalysis then used early checkpoints, where generalization is below zero and its share is meaningless, and read agreement that was only saturation at 0 or 1. Both were caught by independent recomputation. | Compare role measures at the same time, at checkpoints where both quantities are well away from zero. Don't read agreement between saturated shares as evidence. Have exploratory reanalyses verified too, not just preregistered numbers. |
| L15 | E1–E4's raw results lived only on the Mac and in a gitignored folder, so RESULTS.md's numbers could not be rechecked from the repo. Fixed 23 Oct with `experiments/walk/results_data/`. | Commit a compact per-run copy, the analysis output and the verification scripts with every results report. |
| L16 | E4's "no stable region" for C4 read as "the chain can't learn" until the per-lr counts were checked: it groks at lr 2e-3, but collapses 14–22 times. | When a headline statistic is a threshold rule, also report what the runs did beneath it (fit, grokked, collapses). |
| L17 | The priors round's first draft had a headline condition (search share < 0.25) that could not fail on its own grid, because the log scale anchored at k\* = 3 makes "mostly prior" nearly automatic; and a random-direction null that differed from the drift on a known lever (weight norm). Both caught by the pre-run review (24 Oct). | Before preregistering a ratio or share, enumerate every value the grid can produce and check both verdicts are reachable. A null must match the intervention on every known lever except the one under test. |
| L18 | Near a threshold, a strict "≥ 0.95 at every eval" criterion mostly counted single-eval dips: base vs wide, and P3, turned on one dip in 21 evals (priors round, caught by recomputation). | Preregister a dip-robust criterion as primary, or at least as a planned sensitivity check (e.g. the window mean ≥ 0.95, or ≤ 1 dip), and check orderings against it before reporting them. |
| L19 | The worlds design (26 Oct) called a split of held-out functions "in support", though holding a function out gives it zero prior. It also tested composers on program lengths their decoder could never emit (HALT is forced at the training cap). Caught by the pre-design review. | Before naming a split or writing a prediction, check (a) the test target's probability under each arm's *training* prior, and (b) that every arm can *express* the target at all. |
| L20 | The worlds revision-3 derivation of superposition's gap (§5.1) averaged the KL over all 49 inputs, the context inputs included, while the training objective scores only the remaining inputs. This overstated the gap by about 2× and put the headline threshold at the bound. Caught by the independent review. | A derivation that sets a preregistered threshold must score exactly what the objective and the endpoint score: the same inputs, the same prior per cell, the same normalization. Check by reproducing one endpoint number through the derivation code. |
| L21 | Revision 3 derived the expected failures of the superposed arm (local optima), and none for the committed arm, then predicted that the committed arm's failure tail would be smaller. A per-context committed learner using the design's own estimator found 0 of 30 programs of length ≥ 3. | Derive or pilot trainability for *every* arm, especially the one the hypothesis favors, before writing predictions that compare their failures. |
| L22 | Four design revisions of the worlds round went to review before anyone checked that the toy could separate the thesis from known work. On pair worlds with the generator's own library, the posterior over composition classes equals the posterior over output tables, and exact inference is a 5 ms lookup. The thesis's distinguishing quantity was zero by construction. | Before designing arms, write down the quantity that distinguishes the thesis from the nearest known result, and compute it on the toy. If it is zero, or trivially cheap to get exactly, change the toy first. Ask the experimental-value question (what each outcome would change) at the first review, not the fourth. |
| L23 | Rung-0 pilot 1: the learned-halting arm collapsed in every configuration because its halting head started at λ ≈ 0.5 while the control's started at λ ≈ 0.12. With PonderNet's expected loss, early halting mass starves deep steps of gradient, the block never learns them, and early halting becomes self-confirming. | Initialize any learned halting or routing policy so that its probability mass is spread over the depths the task needs, and give every arm the same policy init. A halting arm that fails must be checked for this before it counts as a result (gate G4). |
| L24 | Rung-0 revision 3 (Entry 58): the new headline arm's head read raw log features spanning 18 units with a bias at −2; under AdamW the whole step budget moves a parameter by ≈ 1.1, so every working policy was unreachable, and the arm would have failed in a way that mimicked the mechanism under test. Caught by the design review's arithmetic, not by the tests (the planted head passed because its weights were set by hand). | Before an arm runs, check that a sufficient solution lies within what the optimizer can reach from the init in the step budget (Σ lr for Adam-type optimizers); standardize a policy head's inputs as the state head's are; and put the planted solution's parameters within that reach, so the planted test also tests reachability. |
| L25 | Rung-1 revision 1 (Entry 61): "new orders never seen in training" were defined on the data's surface (PP runs, alternation), but the router's receptive field was the matched pair, for which every stratum was in-distribution; the independent reviewer's 1,800-step CPU run passed the headline predictions at 7 % of the budget before any GPU step. | Define out-of-distribution relative to what the model's state and receptive field actually see, not the data's surface; before preregistering a generalization prediction, run the cheapest configuration of the thesis arm on CPU and check that the prediction is not already true there. |
