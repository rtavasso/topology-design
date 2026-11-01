# Rung 3 design review 1 (revision 1 of `commit/DESIGN.md`, 1 Nov 2026)

**Reviewer:** independent, adversarial; read LEDGER (1 Nov update, H12–H14, F35–F41, L27–L31), history Entries 69–70, `routefind/DESIGN.md` and `RESULTS.md`, `commit/DESIGN.md`, and the route code (`data.py`, `exec.py`, `router.py`, `estimators.py`, `train_route.py`, `routefind/derive/*`). Nothing in the repo was edited except this file. Started at `7310e91` with the then-uncommitted `pp_frac` / `pooled.py pp` changes, which were committed as `8bf5e3f` while the review was in progress; the review applies to DESIGN.md revision 1 at either commit.

Numbers below come from four CPU derivations (each < 10 s), kept in the scratch folder `/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/review_commit/`: `pooled_pp.txt` (the uncommitted `pooled.py 150 pp`), `pp_pooled.py/.txt` (the same fixed points evaluated on clean f = 0 words, class flips, the fixed-prior posterior's route-type split on pp8 words), `tie_break.py/.txt` (hand-built one-lookup policies E / L / L+ / Lx / Lxx on the mixture streams), `entropy_cov.py/.txt` (16-sample oracle coverage at softened all-H policies).

## Verdict: **revise first.** Do not implement stage 2 as written.

Core A's headline cell (shift B) rests on an expressivity claim that is false in the form that matters, its PF rows are contradicted by the design's own §5.1 derivation once it is run, its two doses realize 0.7 % and 4.6 % of queries (not 5 % and 25 %), and "moves" is never defined although its two natural readings (program identity on ordinary words; pp8 accuracy) dissociate by construction. Core B does not run the arm that would make risk 4 readable. Shift A is the good cell and should be the core, with a dose knob. Checks C are fine as checks but one of PC1's two clauses is refuted by PF's derived fixed point. The phase-1 → phase-2 mechanics lack a commitment gate and a reachability ceiling, without which "nothing moved" cannot be read.

---

## Blocking

### B1. Shift B's tie-breaker does not break the tie; "moves to E" and "passes pp8" are different events, and the design conflates them (§1, §2, §3, PA1–PA2)

**The expressivity claim (§2 "L cannot do them under δ = 1") is wrong in the sense the design needs.** What is true: the L router's *value-slot read* finds nothing at the second P-leaving state of a PP run (RESULTS §2: EI16's SET>HOP first deviation). What is false: that no L-type program completes PP words at the oracle's length. On a PP run (steps i, i+1 P) the shortest correct routes from s_{i−2} to s_{i+1} are five actions with two SETs, and at least five route types exist: E-E (SET, HOP, SET, PREV, PREV), E-L, E-double, L-L and L-double (HOP, SET, SET, PREV, PREV). `program_stats` scores shortest-correct as *label reached in the oracle's number of moves*, so every one of them counts at pp8. Hand-built one-lookup policies (`tie_break.txt`, 300 episodes):

| policy on the f = 0.25 stream | ordinary P-words | PP words | no-P |
|---|---|---|---|
| E (own flag; count-only PREV) | 1.000 | 1.000 | 1.000 |
| L (value-slot read) | 0.946 | 0.000 | 0.951 |
| Lxx = L + double SET at (own ∧ vf ∧ c=1 ∧ h=1) + SET at (own ∧ ¬vf ∧ c≥1 ∧ h=0) + (c≥1 ∧ h=0) → PREV | 0.946 | **0.904** | 0.951 |

So an L router can reach ≈ 0.9 at pp8 by adding three decisions at classes that f = 0 never labels, **while staying L on every ordinary word**. Conversely a router can move to E on ordinary words and still fail pp8 (rung 2's E routers do, SET>PREV). The prediction table's "PF moves to E and reaches pp8 ≥ 0.95", "IW16 stays L and fails pp8", "EX ≥ 3/5 move" treats one event; there are two, and the reading table (§7) attributes pp8 outcomes to the E/L axis.

**What PP words actually do to the tie** is not in the design: under δ = 1 a PP run flags pair(s_{i−1}) whose value is a *decoy*, so a flagged pair points at a key of another chain. L's read misfires there ("false pointers"): 1.1 % of queried oracle routes at f = 0.05, 4.9 % at f = 0.25 (`tie_break.txt`), which is the whole of L's 0.946 on ordinary words. That is a genuine, in-support push toward E (L's read is *inherently* ambiguous on the mixture; E's is not) and the design should name it; it is small, and it does not operate through pp8.

**The route posterior on PP words does not favour E either.** The exact fixed-prior posterior at the key decisions of pp8 words (`pp_pooled.txt`, 150 words): at s_{i−2}: SET 0.66 / HOP 0.34 (same as the ordinary 0.60/0.40 split); at s_{i−1} with c = 1: SET 0.45 / PREV 0.35 / HOP 0.20; at s_i with c = 0: SET 0.76; with c = 1: PREV 0.55. The likelihood and the prior split evenly over E-type and L-type completions.

**Fix.**
1. Define *mover* operationally and separately: (a) **program on ordinary words** = `set_late` on isolated-P words only, evaluated in a *clean* context (`pp_frac = 0` strata, see S4) and in the mixture context; (b) **PP competence** = pp8 shortest-correct, *by route type* (E-E / E-L / L-L / L-double / E-double / other, a small extension of `program_stats` over the PP run's SET positions); (c) the two decisions PP words label: the conjunction probe `cf_pending_set` at (own=1, vf=1, c=1, h=0) and a new probe at the second P-leaving state (own=0, vf=0, c∈{0,1}, h=0).
2. Rewrite §2's claim: "the L read fails at the second P-leaving state; L-type completions of PP words exist and are shortest; the data push toward E only through flagged decoys, at 1–5 % of ordinary P-words."
3. Decide what shift B is for. As it stands it tests *learning rare new decisions after commitment* (interesting; not the tie disagreement of Entry 70 §2). If the tie question is the aim, shift A is the cell that asks it (S1) and shift B should be reframed and shrunk, or replaced by a shift that relabels in-support decisions toward E.

### B2. PF's movement from the L router is prior-driven and f-independent; the PF rows of PA1 and PA2 are derivable, and the derived outcome is the opposite of the prediction (§5.1, PA1, PA2)

I ran the uncommitted §5.1 derivation (`pooled.py 150 pp`) and re-evaluated the pooled fixed points on clean f = 0 words (`pp_pooled.txt`):

| pooled PF fixed point trained at | in3P / in4P / in5P / in8P (clean words) | pp4 / pp6 / pp8 | classes whose argmax differs from f = 0 |
|---|---|---|---|
| f = 0 | 1.00 / 1.00 / 1.00 / 1.00 (program E) | 0.03 / 0.00 / 0.00 | – |
| f = 0.05 | 1.00 / 1.00 / 1.00 / 1.00 (E) | 0.01 / 0.01 / 0.01 | **none** |
| f = 0.25 | 1.00 / 1.00 / 1.00 / 1.00 (E) | 0.01 / 0.01 / 0.02 | **none** |

(The committed `experiments/commit/derive/pooled_pp_120.txt` shows in4P 0.62–0.75 and in8P 0.44–0.91 at f = 0.25 / 0.05: that is an evaluation artefact, not a change of the fixed point. Its "train" evaluation stream is generated with `pp_frac = f`, so its "P-words" include PP words, which the pooled policy fails. On clean isolated-P words the same fixed points are 1.00 at every k, and no key-state class changes its argmax; see `pp_pooled.txt`. The derive output should be re-run with a clean evaluation stream before it is cited.)

Two consequences. (i) PF's target already says E at f = 0 (rung 2 §5.2), so PF continued from an L router moves to E on ordinary words **with no PP words at all**: the "movement" PA1/PA2 credit to the data is the target's own tie-break (L28: the fixed point of a fixed-target arm is computable). (ii) Under one-lookup pooling the PF target **fails pp8 at both f**: every PP completion needs a decision at s_i that falls in the big ordinary classes (own=0, pred=0, c=1, h=0) → HOP 0.61 or (own=0, pred=0, c=0, h=0) → HOP 0.71, where the flat prior's redundant-SET routes through unflagged states outvote the PP words. So PA1's "PF … pp8 ≥ 0.95 in ≥ 4/5" and PA2's "PF still moves in ≥ 2/3" are predicted to come out *program E on ordinary words, pp8 ≈ 0*, at which point the design's own falsifier text would read "a 5 % minority is too little" or "exactness is not enough": a pooling artefact read as a plasticity result. If the neural router does pass pp8 under PF, that is a finding about its generalization at off-support classes (count-only PREV), not about plasticity.

**Fix.** PF from L is a **check** at both f (2 seeds), with the derived expectation stated (E on ordinary words, pp8 fails under one-lookup pooling; a pass means the router pools differently and is reported as such). Add the **f = 0 continuation from the L router for every phase-2 signal** (2 seeds each; derivable: PF → E, the self-consistent targets stay L, PF-IS16 stays L-leaning) as the control that any movement at f > 0 must beat. The stability cell in §3 (EI16 only) is not enough.

### B3. The doses are not what the text says, and the f = 0.05 cell cannot separate the readings (§3, PA2, H12)

`pp_frac` applies to chains with k ≥ 4 (`data.py`, `train_word`), which are ≈ 20 % of chains under the Zipf(1.5) lengths. Realized: PP words are **0.7 %** of queries at f = 0.05 and **4.6 %** at f = 0.25 (`tie_break.txt`); false pointers touch 1.1 % / 4.9 % of oracle routes. Per training step (384 queries) that is ≈ 2.7 and ≈ 18 PP queries. H12's falsifier ("every signal moves at a 25 % minority") and §3 ("a fraction f of chains carrying a PP word") misstate the manipulation by 5×. At 0.7 % with a few PP decisions per query, "every other signal ≤ 1/3" (PA2) is implied by sparsity alone under *both* readings of §1 (the researcher's reading predicts a rate set by the fraction of disambiguating data, which is ≈ 0 here), so PA2's non-PF rows are vacuous; its PF row is B2.

**Fix.** State the dose as the realized fraction of disambiguating *decisions per step* (PP decisions, false-pointer states), and calibrate it on CPU against a target that is known to move (the teacher-forced ceiling of B5): the lower dose should be one where the ceiling arm still converts within budget, or it measures nothing about targets. If one lower dose is kept, pick it by that criterion, not by a round number.

### B4. Core B does not run the arm that answers QB; its stage-2 gate hangs on an arm that is likely derivable to fail; §5.2's derivation is at the wrong policy (§1 QB, §3 core B, §5.2, PB1–PB2)

- **The missing arm.** QB and H14 ask whether a *reward* signal can be kept exploring so that bootstrapped vs Monte-Carlo value (P4) becomes readable. PFB16 is a posterior target (soft EM over a beam with a length prior), not a reward signal; nothing in core B feeds forced-breadth routes to RL16 / ACMC16 / AC16 (I33 as written). Stage 2 (the only place P4 is read) is conditional on RL16+H learning, which the design itself predicts not to happen (PB1). As designed, the most likely end state is "risk 4 stays blocked" — a foregone outcome (L27).
- **§5.2 is derived at the uniform init,** but an entropy arm does not sit there: it learns all-H in the first few hundred steps (F40) and sits at an all-H policy with residual mass set by β. The relevant quantity is coverage *at that policy*. With uniform residual mass ε_r over the other actions, 16 samples contain the oracle route on 1.6–4.4 % of P-words at ε_r ∈ [0.15, 0.6] (≈ 1–3 oracle samples per step; `entropy_cov.txt`), which is not "never". But the max-ent stationary policy is not uniform in the residual: at carried-count states PREV is punished on every all-H word by ≈ 0.9 reward units (PREV leaves the chain), so π(PREV | c ≥ 1) ≈ e^{−0.9/β} ≤ 1.2·10⁻⁴ at β = 0.1 and ≈ 0 at β ≤ 0.03, while at β large enough to lift it the policy is near-uniform everywhere and coverage is the init's 0.00–0.03. If that class-by-class derivation holds (do it with the exact Q under the all-H policy), RL16+H fails at every β *by derivation*, and PB1 is a check, not core.
- **PB2's dose comparison is on a different schedule:** PFB16 at 5,000 constant-lr steps against PF-IS16's 0/3 at the dose on rung 2's cosine schedule (which L31 says cuts slow arms off). "Degrades by less than PF-IS16 did" can be a schedule effect.

**Fix.** (1) Add the forced-breadth *reward* arms unconditionally: RL16-B, ACMC16-B, AC16-B, whose K routes are the ε-mixed beam's completions (all of them; `beam_search` already builds them) with off-policy correction π(r)/π̃_beam(r) or, more simply, the beam's completions as the sample set with a per-query baseline over them; 5 seeds on δ = 1. This reads P4 in one stage (bootstrapped vs MC value on the same routes). (2) Derive RL16+H's stationary coverage per β before the grid; if ≈ 0 everywhere, run it as a 2-seed check at the best β. (3) Keep PFB16, but label it as the F41 follow-up (risk 3 carry-over: "is the fixed prior's dose fragility in the target or in the proposal?"), and run PF-IS16 at the dose on the constant schedule (3 seeds) as its matched reference. (4) In the reading table, "the scalable form of the PFN signal" overstates PFB16: its support is one path per state (merge rule) and it has no importance correction, so it is biased toward the beam; see S7.

### B5. No commitment gate, no reachability ceiling (§3 phase 1, §5.3, §8; L24)

- Phase 2's cells are defined as "from the L-committed EI16 routers" and "from the E-committed EX routers", but nothing says what *committed* means at the 2,500-step checkpoint, nor what happens to a seed that is not (rung 2's programs were observed under cosine 5k; one IW16 seed was R; PF-IS16 is a mixture by construction). Preregister the gate: committed shortest-correct ≥ 0.95 at in4 and in8, program label (`set_late` ≤ 0.25 or ≥ 0.75 on isolated-P words), sampled shortest-correct ≥ 0.95 for the deterministic signals, R detection; an exclusion/relabel rule; and the realized number of usable routers per cell, which may fall below the seed counts in §6.
- **Reachability.** Phase 2 must overturn decisions an EM/imitation router has sharpened for 2,500 steps. Nothing records the logit margin at the decisions each shift must flip (shift A: HOP → PREV at (own=1, c=1, h=1); shift B: the three classes of B1) or checks that 2,500 effective steps at lr 1e-3 can move it (L24's Σ lr rule, with a fresh Adam state and a 500-step warm-up eating part of the budget). Without a ceiling, "no signal moves" and "the router cannot be moved in 3,000 steps by anything" are indistinguishable.

**Fix.** Record the margins at every checkpoint in the derive step. Add a **teacher-forced reference from the same checkpoints under each shift** (the oracle route under δ = 0 for shift A; the E route on the mixture for shift B; 2 seeds each). It is the full-information limit of PF and the one arm known to move if anything can; shift A currently has no speed ceiling at all (its references are rung 2's from-scratch cosine cells). Report every phase-2 arm's time-to-flip relative to it (L31).

---

## Should fix

### S1. Shift A is the clean tie-breaker and should carry core A; PA3's EX / EI16 rows are probably wrong and are not open-marked (§3 shift A, PA3)

Under δ = 0 the E rule is F29's stray rule (§2, correct), and the only decision that must change is HOP → PREV at (own=1, c=1, h=1): a class the E router visits on **every** P-word, relabeled by every P-word (18 % of queries). That is a 4–25× stronger dose than shift B and an in-support relabeling, so the two shifts are not two instances of one question and "moves under A but not B" says nothing about target type. Also: the E start is *easier* than EI16B's start, not "a cleaner replication of it". At the E router π(SET | flagged) ≈ 1, so the beam needs one improbable child (PREV at ≈ ε/4 → −3.6 nats under ε = 0.1), where EI16B's basin policy needed two (SET ≈ 0.04–0.22, then PREV). EX's own posterior at the E router puts the correct route at ≈ π(PREV | own=1,c=1,h=1)·0.98 against the decoy route's ε/15 ≈ 1.3·10⁻³ whenever the decoy chain misses the label (15/16 of P-words), i.e. tens of percent of the posterior if π(PREV) ≈ 10⁻³. I expect EX and EI16 to move on shift A, which is the informative outcome ("a committed start is easier than F29's stall"), but PA3 preregisters ≤ 1/3 for both.

**Fix.** Make shift A the core of risk 8. Mark EX, EI16 and PF-IS16 rows open pending §5.3 (coverage *and* EX's exact posterior mass on the correct route at each checkpoint, computable with `marginal`/`posterior_counts`). Give shift A a dose knob that keeps the task well-posed: `p_P` (0.3 → 0.06) scales the disambiguating P-words per step without the per-flag ambiguity a mixed-δ stream would create; run the ladder at two doses. Keep PA3's PF ≥ 2/3 (its target has no basin, as derived) as a check.

### S2. PA4's Kendall τ over ≈ 6 pairs is not a test

With n = 6, P(τ > 0 | null) ≈ 0.5, ties among "expected low" coverages are likely, and the mover counts are 0–5. Replace with a per-seed analysis: each saved router has its own coverage and its own phase-2 outcome, giving ≥ 20 paired points across signals × shifts × doses (AUROC of coverage for movement, or a preregistered coverage threshold and a 2 × 2 count), and a within-signal manipulation: EI16 at ε ∈ {0.1, 0.3} from the same checkpoints (coverage is a knob; if plasticity tracks it within one signal, PA4's claim is made cleanly).

### S3. PC1: one clause is refuted by the derived fixed point; the AUROC is stratum-confounded (§3 checks C, §5.4, PC1)

- **Entropy by decision type.** PF's pooled target has entropy 0.59 nats at ordinary HOP states (0.72/0.28), 0.67 at SET states (0.60/0.40) and ≈ 0.9–1.0 at PREV states (0.56–0.65 PREV, the rest on SET and HOP; `pooled_pp.txt`, rung 2 §5.2): ratios ≈ 1.1× and ≤ 1.7×, not "≥ 3×". A router fitting a soft target is not sharper than the target, so PC1's entropy clause fails by derivation for PF; for the deterministic routers entropy is ≈ 0 everywhere and the ratio is noise. Set the threshold from the derived fixed point (or drop the clause).
- **AUROC.** For learned stochastic routers committed error is ≈ 0 on in4/in8, ≈ 1 on pp strata, and mixed only on len{16–32}, alt and deepP8. Pooled across strata the AUROC is a stratum detector (longer or out-of-support words have more disagreement for length reasons); per stratum it is undefined where one class is missing. Preregister: per-stratum AUROC on strata with 10–90 % committed error, a length control (disagreement vs. k within the len strata among committed-correct queries), and the exact marginal's AUROC on the same queries as the ceiling. The design already notes the readout-entropy baseline is constant; say the same about route disagreement's length dependence.

### S4. Program identity and in-distribution accuracy are contaminated in phase 2 (§4)

- `set_late` on "P-words" of the phase-2 criterion set includes PP words, on which the E-E route classifies 0.5 late (its second SET is *at* a P-leaving symbol) → "mixed" by construction. Compute identity on isolated-P words only, and the PP route-type split separately (B1).
- Phase-2 strata and criterion sets are generated with the mixture config, so even the forced isolated-P query of in4/in8 sits in a PP-polluted context: an unchanged L router drops to 0.946 on ordinary P-words at f = 0.25 from false pointers alone (`tie_break.txt`; lower on in8, which has more states per route), at or below the "learned" threshold. Evaluate in4/in8 twice (clean `pp_frac = 0` strata and the mixture strata) and report accuracy at false-pointer states on its own.

### S5. Phase-2 clock and optimizer state (§3)

Phase 2 starts with fresh Adam moments and a 500-step warm-up, so ≈ 1/6 of the budget is spent re-warming; count "time to pp8" and "time to flip" from the end of warm-up, or shorten it to 100 steps. State in §6 that the budget was checked against the ceiling arm's time (B5).

### S6. Phase-1 length is an unexamined moderator

How far the margins have sharpened by the checkpoint sets every coverage number and therefore every "open" row. Record margins (B5) and run one duration arm (EI16 phase 1 at 1,500 vs 2,500 steps → shift A) so that "sticky" is reported as a function of commitment depth rather than of an arbitrary 2,500.

### S7. PFB16: well defined, but say what it is (§3 core B, PB2)

Given the beam's completion set C it is well defined (p(r) ∝ π₀(r)e^{−λ·moves}L_y(r), r ∈ C). Under the flat prior all shortest correct completions in C get equal weight, so the target is "length-weighted uniform over the beam's correct completions": distinct from EI16 (which keeps one MAP route) and from PF-IS16 (which has π₀/π importance weights on on-policy samples and is unbiased for PF's gradient; PFB16 has no correction and is biased toward the beam's support). Because the beam merges by state and keeps the MAP prefix, E and L prefixes to the same post-PREV state never coexist in C, so the fixed prior's 0.60/0.40 preference for E cannot act through PFB16; its program should be L like EI16's (near-derivable; X27's merge mechanism). Its fixed point is policy-dependent (not closed-form), which is a fair reason to keep it core for the dose question; PB2's "program open" is fine, but add the derived expectation and the merge caveat.

### S8. Budget allocation

IW16 phase-1 routers (5 × 2,500) are used nowhere except checks C, where a deterministic router carries no disagreement (AUROC undefined). EX phase 1 needs 3 usable E routers, not 5, unless the gate (B5) expects attrition. Spend the freed runs on the f = 0 controls (B2), the teacher-forced ceilings (B5), the PF-IS16 dose reference (B4) and the shift-A dose (S1).

### S9. Checkpoint loading across stack sizes (§10, G1)

Parameters are stacked (R, …); phase 1 saves stacks of 5 and phase 2 loads individual runs into stacks of a different size and order. The round-trip test should be: a loaded slice reproduces the saved router's logits on a fixed batch to float tolerance and its `evaluate()` record equals the phase-1 final record; the frozen bases (buffers) and `emb_seed` travel with the state. Say that Adam state is not restored (it is a design choice, S5).

### S10. References on mismatched schedules

Shift A's references are rung 2's cosine 5k from-scratch cells on other seeds; PB2's is PF-IS16's cosine dose cell. Add the few constant-lr from-scratch references the comparisons need (EX δ = 0, EI16 δ = 0, PF-IS16 dose; 2–3 seeds each) or restrict the comparisons to the first 2,500 steps where the cosine lr was ≥ 50 %.

---

## Minor

- **M1.** §2 "E can [do PP], once its precedence … is set": the E route needs two decisions PP words label, the conjunction (own=1, vf=1, c=1, h=0) → SET and the carried-count PREV at s_i (own=0, vf=0, c=1, h=0), which is off-support for a committed E router. The "speed ceiling" reference (EX from E) therefore learns two things, not "the precedence flip alone".
- **M2.** H12 and §1's table say "25 % minority"; see B3. Also "PP words favour E" should become the flagged-decoy statement of B1.
- **M3.** §10 says none of the code exists; `pp_frac`, its test and the `pooled.py pp` mode were committed during this review (`8bf5e3f`). Update §10; the test's window (0.35–0.65 among k ≥ 4) is fine, but add an assertion on the realized fraction over *all* queries so the dose of B3 is documented in code, and fix the `pp` mode's evaluation stream (B2's note).
- **M4.** `ei_eps` is a free knob of EI16's coverage and is fixed at rung 2's 0.1 without comment; name it in §3 and use it (S2).
- **M5.** The CPU sanity run (§8) should also emit the committed router's margins and coverage (`derive/search.py` on the CPU checkpoint) so the open rows can be previewed before the GPU job, and the phase-1 → phase-2 loader is exercised end to end.
- **M6.** PC1's "16-sample disagreement on the answer" for PF: PF's sampled answer accuracy is 0.37–0.38 (RESULTS §2 endpoints), so answer disagreement is near its maximum on *every* query; the informative statistic is more likely the sampled-answer *majority* margin or agreement with the committed answer. State which statistic before running.

---

## Summary of what the revision should do

1. Make shift A (δ = 0 from E, with a `p_P` dose) the core of risk 8; mark its EX / EI16 / PF-IS16 rows open pending §5.3 (coverage *and* exact posterior mass); add the teacher-forced ceiling and the margin record.
2. Reframe shift B honestly (rare-new-decision learning after commitment, plus a 1–5 % false-pointer push toward E), with separate endpoints for ordinary-word program and PP route types, f = 0 controls for every signal, PF as a check, and doses stated as realized fractions.
3. In core B, run the forced-breadth reward arms unconditionally and read P4 there; derive RL16+H's stationary coverage and demote it if ≈ 0; keep PFB16 as the F41 follow-up against a matched PF-IS16 reference.
4. Add the commitment gate, the clean-context evaluation, and a PC1 rewritten around per-stratum AUROC with a length control and the derived entropy ratios.
