# Rung 1: a learned per-token router over exact generic circuits (revision 2)

**Status:** design, 28 Oct 2026, revised after review 1 (`review/review1_design.md`, `review/review1_code.md`), for a short second review before the pilot. Rung 1 of the ladder (D72): symbolic generic circuits, routing only (Q1, Q2); the formalism of D75 (latent-route mixture, step-cost prior, soft routing as the first E-step). Carries rung 0's constraints (Entry 60). Revision 1 is in git (`de85390`); §15 lists what changed and why.

**Ledger:** thesis Entry 48; what the toy is for Entry 45; generic circuits Entry 44; residual signals Entry 46; setup credit Entry 47; pooling Entry 50; ragged depth Entry 52; module-network prior work Entry 33; rung 0's results Entries 55–59 (F22–F27); rung 1's design review Entry 61.

**Scope note (AGENTS hard rule 1).** This round compares a learned router over fixed exact circuits with static and looped learned models, at the researcher's direction. Every claim is labeled expressivity, optimization or mechanism.

---

## 1. The question

**Q1 (B2).** From next-token loss alone, does a per-token router learn *order-dependent serial chains of distinct generic circuits*, where the chain needed varies by token and context — including a **setup step** whose only effect is on a later step (D74)?

**Q2 (B3).** Do the learned chains generalize to longer chains and to orders never seen, better than a static stack of learned layers and a looped learned block at matched per-token compute?

**What review 1 showed about revision 1.** With the trigger written on the pair the P-step leaves from, the rule "value says ← ⇒ PREV" is a function of the matched pair alone; a Markov router reads it in one attention and every "new order" is in-distribution for it (the reviewer's 1,800-step CPU run passed P1–P3 at 7 % of the budget). Revision 2 makes the trigger **non-local**: the flag for a P-step sits one state *earlier* in the chain, so the router must take a setup action there (SET: remember that the exit from the *next* state is PREV), carry a pending bit across a HOP, and spend it at the next state. Now
- the route needs a **product state** (symbol × pending count × set-here), and whether the router uses it is the F4 question (an explicit richer state vs the symbol alone), tested by the symbol-only arm;
- the setup step has no effect on the prediction at the step it is taken: it is credited only through the chain rule of the marginal (D74 in its exact form);
- consecutive P-steps (PP runs) need a pending count of **2**, a state value never visited in training (training words have P-runs of length ≤ 1), so "new orders within length" is outside the training support of the router's *state*, not only of its context;
- the looped learned block must hold the pending bit **inside its stream** across a loop, so RM vs LF becomes "an explicit register vs memory inside the block" rather than only "learned vs given circuits" (review 1 A.iii).

**The bet that makes this different from module networks.** Learned layouts inferred *from the input* fail systematic generalization (SQOOP, CLOSURE; Entry 33). The router here infers *each step from the computational state*: "my pair is flagged and I have not set here ⇒ SET; I hold a pending flag that applies to this exit ⇒ PREV; the register holds an answer ⇒ HALT; else HOP". That rule is local in the product state, depth-free and content-free, so a router that learns it generalizes to any word over the circuits and any length, *if* it reads the count as an ordinal quantity. The static alternative must host every circuit at every depth and learn each instance from the tokens that need it there (pooling for selection, Entry 50).

**Also asked.** B4 (iterate until done, inherited from rung 0), the residual study of D73 in its halting part (does p(HALT) track the exact remaining distance), and the setup-credit question of D74 in its exact form (does the marginal credit SET, and how fast, relative to teacher forcing on traces).

**Not asked here.** B1 (the step-cost prior buys minimal routes: untestable at this rung, §10 P6), the remaining-distance probe (the router's input fixes remaining distance only as {0, 1, ≥ 2}; review 1 C.P8), learned primitives (rung 2–3), hard routing with search and a learned value (rung 4; B5, B6), transfer to new symbol alphabets (empty by construction with frozen bases, as in rung 0's H8).

## 2. The task

**Symbols.** 48 keys, 16 answers, one dead symbol ⊥ (n_sym = 65; ⊥ is also the "no previous pair" marker). Frozen orthonormal embeddings in disjoint subspaces of the stream, as in rung 0: key slot, value slot, **prev slot**, register, scratch (5 × 65 = 325 of d = 344; the 18 remaining directions, also frozen and orthogonal to the ones vector, carry the flag, the pending count and the set-here bit; the neural baselines use the same bases).

**Pair tokens.** Each pair token carries (key, value, prev, flag): the key, its value (a key or an answer), the key of the pair that *precedes it in context order*, and a flag bit. The prev slot is the previous-token head's output, provided in the token so that every circuit in the library is a one-layer match-and-read (§3); the baselines get the same tokens, the flag on a frozen free direction.

**Chains.** A chain is a sequence of states s₀ → s₁ → … → s_k with s_k an answer. Step i leaves s_{i−1} and is one of two circuits, written in the data:
- **H (hop):** pair(s_{i−1}).value = s_i (a key, or the answer on the last step);
- **P (prev):** pair(s_i) is the pair immediately before pair(s_{i−1}) in context order, and pair(s_{i−1}).value is a **decoy** — a random key of another chain, so that HOP at a P-step wanders into another chain instead of dying (a wrong route is not detectable step by step; it ends on a wrong answer with full mass, see §5).

**The trigger (the change from revision 1).** A P-step i is announced by the flag on pair(s_{i−1−δ}): with **deferral δ = 1** (the thesis setting) the flag sits on the pair *one state before* the one the P-step leaves from — "the exit from the next state is PREV" — so the route must carry that information across the hop out of s_{i−2}; with **δ = 0** (the control) the flag sits on pair(s_{i−1}) itself, a local trigger as in revision 1 but with the same four-action library and the same setup step. The first and the last step are always H (w₁ = w_k = H), so a flag never sits on a query token and an answer is reached only as a value. The **word** of a chain is its step sequence, e.g. HHPHPH; it is written in the data (flags mark exactly the P-steps) and is readable only by following the chain (F7).

**An episode.** A forest over the 48 keys: Qn = 6 queried chains plus distractor chains, P = 48 pair tokens, shuffled subject to the adjacency that P-steps force (a run of j P-steps is a block of j + 1 pairs in reverse order; blocks are shuffled). Queries: (start key, answer) token pairs, 61 tokens in all, as in rung 0. The loss is the cross-entropy over the 16 answers at the query positions.

**Lengths and words (training).** k ~ Zipf(1.5) truncated at k_max = 8. Each step 2 … k−1 is P with probability p_P = 0.3, **except that P-runs are limited to length 1 in training** (no two consecutive P's). So the training family W_train = words over {H, P} of length ≤ 8 with w₁ = w_k = H and no PP. Its statistics are in §5.

**Evaluation strata** (one forced query per episode plus Zipf fillers, 250 episodes per stratum, exact references for every query):
- **In range:** `in1`–`in8`: k ∈ {1, …, 8}, words from W_train; every in-range table also reports words with and without a P separately. `none3/4/8`: all-H words (the static reference).
- **Length:** `len10`–`len32`: k ∈ {10, 12, 16, 20, 24, 32}, words with the training statistics (p_P = 0.3, no PP).
- **New orders within length:** `pp4/6/8`: words with one PP or PPP run (never seen; under δ = 1 the pending count reaches 2 there, a state value never visited in training; under δ = 0 PP runs are local SET-PREV pairs, so that arm is the "same context, no new state" control).
- **Frequency strata (training words):** `alt4/6/8` (H P H P … H: HPHH is 21 % of k = 4 training queries, HPHPHH 4.4 % of k = 6, HPHPHPHH 0.9 % of k = 8) and `deepP8` (a P at step ≥ 6; ≈ 700 such steps per 100 training steps). These are **not** new orders (review 1 A.ii); they are reported as in-range frequency strata.
- **New orders, long:** `pp16`, `pp32` (PP/PPP runs at k = 16, 32); `alt16`, `alt32` (alternation at k = 16, 32: training words' statistics taken to length).

**Read-only context, ragged depth.** As in rung 0 (D79): context tokens are single-step; a query's route attends to the context's depth-1 states. The router has its own cross-attention over the context (Entry 52's deferred variant, needed here because the circuits are symbolic and there is no block to read through); **there is no attention among pair tokens** (a per-token MLP only), so the router's receptive field at a state is whatever its heads select from the context — it can match its register against the key slot (its own pair) or against the value slot (the pair that points to it), which is how a symbol-only router could still *infer* that its predecessor's pair was flagged (§6).

## 3. The library and the exact executor

**State.** (s, c, h): the register symbol s, the pending count c ∈ {0, 1, 2}, the set-here bit h ∈ {0, 1} (set by SET at the current symbol, cleared by any move). NS = 65 × 6 = 390 product states; the route starts at (s₀, 0, 0).

**Four actions.** With pair(s) = (s, v, q, f) when s is a key:
- **HOP:** s ← v, c kept, h ← 0. Answers absorb; ⊥ absorbs everything.
- **PREV:** needs c ≥ 1: s ← q (the previous pair's key), c ← c − 1, h ← 0; ⊥ if c = 0 or q = ⊥. Answers absorb.
- **SET:** c ← min(c + 1, 2), h ← 1; the symbol does not move. A no-op on answers and ⊥.
- **HALT:** stop; read out.

Both circuits are one-layer generic match-and-read circuits (match the register against the key slot, read the value slot or the prev slot); SET is the register operation that makes a deferred rule executable at all. A wrong action does not raise an error: HOP at a P-step follows the decoy into another chain; PREV without a pending count kills the route (⊥, uniform readout); PREV with a stale count at an H-state goes to a random neighbour and wanders. Wrong routes are detected only by the likelihood of the target — the untyped semantics on purpose: it is what makes the E-step a real inference rather than a lookup.

**The oracle rule** (`data.oracle_actions`, `exec.oracle_policy`): HALT if s is an answer (or ⊥); SET if pair(s) is flagged and h = 0; PREV if c − h·[δ = 1] ≥ 1 (under δ = 1 the flag set *here* applies to the *next* state's exit, so it is excluded; under δ = 0 it applies here); else HOP. The oracle route of a word with n_P P-steps has k + n_P moves (one SET per P-step) and then HALT. Every query's oracle route is checked by the generator (`check_episode`).

**Readout.** Register holding answer a: p(y = a) = 1 − ε, the rest ε/15, with ε = 0.02 fixed; register holding anything else: uniform over the 16 answers.

**The executor is symbolic** (`exec.py`): integer tensors, three transition tables (B, 390) per episode. No neural circuits run in the routed arms.

## 4. The model: a Markov router and the exact marginal

**Router.** π_θ(a | (s, c, h), ctx), a ∈ {HOP, PREV, SET, HALT}. Architecture: the state token = frozen register embedding of s + c · (a frozen free direction) + h · (another) + a learned marker; one cross-attention layer (2 heads) from the state token over the context's pair tokens (frozen slot embeddings + the flag direction + learned positions, each through a learned per-token MLP; no attention among pairs), then an MLP and a LayerNormed linear head to 4 logits (L24). The router sees **no step index** and **no route history**: it is Markov in the product state, which is the form that can pool across depth (Entry 50). **Symbol-only variant** (`state_input = "symbol"`): the state token omits the c and h directions, so π depends on (s, ctx) alone — the F4 null.

**Exact marginal likelihood.** Because the state is one of 390 values and π is Markov, the mixture over all routes is a forward recursion over product states: with alive mass b_t(x) (b₀ = δ_{(s₀,0,0)}),
- halt mass h_t(x) = b_t(x) · π(HALT | x, ctx),
- b_{t+1}(x') = Σ_x b_t(x) · Σ_{a ∈ {HOP, PREV, SET}} π(a | x, ctx) · [exec_a(x) = x'],
- forced halt at T: h_T(x) += b_T(x),
- q(y | ctx) = Σ_{t ≤ T} Σ_x h_t(x) · readout_x(y).

This is the latent-route mixture of Entry 48 summed exactly over all 4^T routes in O(T · NS · 4), because routes that share a state at a step are merged. The router is evaluated once per episode at all 390 states (cheaper than pruning). Gradients flow to π at every live state in proportion to the posterior mass of the routes through it, so the setup step SET — which changes nothing in the prediction at the step it is taken — gets the chain-rule credit of Entry 47 exactly; no route has to be *found* by sampling. Under a uniform router the oracle route's share of the target mass is small beyond k = 3 (§5), so the rule is learned on short chains first (Zipf), then applies at every depth.

**Training loss.** −log q(y | ctx) + λ · E[steps], with E[steps] = Σ_t t · Σ_x h_t(x) (the expected number of actions other than HALT, SET included), λ = 0.02 per step (so that one extra step costs about the log-likelihood gap of a 2 % readout error; swept in the pilot). T_train = 14 = k_max + 3 SETs + 3 slack; T_eval = 64 (alt32 needs 47 moves). AdamW, lr 1e-3, 500 warm-up, cosine, 6k steps, batch 64 (pilot: lr ∈ {3e-4, 1e-3, 3e-3}). Teacher forcing (RM-T): −Σ_t log π(a*_t | x*_t) along the oracle route.

**Test-time routing.** Two readings of the same model: **committed** (at each state take argmax_a π; the route the system would actually run, its steps are the compute) and **marginal** (q itself, the Bayes-mixture prediction, no compute saving). Route accuracy = the committed route equals the exact oracle route action for action and halts at the same step; answer accuracy = argmax of the committed route's readout; steps = actions before HALT. **The committed reading is the thesis reading** (what executes); the marginal reading is reported because a stochastic router can succeed in it without a deterministic program (§6, the symbol-only trick).

## 5. Derived before measuring (`derive/`, revision 2; the revision-1 numbers are in git)

- **Training words** (`derive/coverage.py`, 24,000 training queries). 54 distinct words; 16.4 % of queries contain a P, 8.8 % of steps are P. The most common word of each length covers 1.00 / 1.00 / 0.70 / 0.50 / 0.35 / 0.23 / 0.16 / 0.10 of the chains of length 1–8 (all-H for k ≤ 7; HHHPHHHH at k = 8). The best fixed program of any length is all-H: it covers **0.837** of training queries.
- **The distinguishing quantity (L22), both readings** (`derive/stateblind.py`: a step-indexed policy π(a | t) fitted by the exact marginal for 800 CPU steps, 2 seeds). A state-blind policy cannot halt on the answer, so its **route accuracy is 0 by construction**; its committed **answer** accuracy is the fixed-program coverage, Zipf-weighted in range **0.83 / 0.84** (by k: 1.00 / 1.00 / 0.67–0.73 / 0.49–0.53 / 0.36–0.43 / 0.26–0.30 / 0.04–0.06 / 0.03–0.05; the λ cost makes it halt at t ≈ 9), and its **marginal** accuracy is **0.85** (by k: 1.00 / 1.00 / 0.70–0.75 / 0.55–0.57 / 0.38–0.46 / 0.31–0.32 / 0.21–0.22 / 0.12). Revision 1's state-blind marginal reached 0.94 because wandering mass died on ⊥ and had no vote (review 1 B); with decoys it lands on wrong answers with full mass, so the Bayes mixture no longer rescues P-words (their marginal accuracy is ≈ 1/16, the chance that the decoy chain shares the answer). **So 0.84 (committed) / 0.85 (marginal) are the state-blind references for every in-range table, and 0 for route accuracy; the gap to a state-reading policy is 0.16 of the in-range stream, all of it on words with a P.** The fitted table's quirk — two SETs before the first HOP — is a marginal-likelihood artefact: SET converts mass that a stray PREV would kill into mass that wanders and keeps a 1/16 chance (review 1 should not be surprised by it in RM-idx's table).
- **Counts per (circuit, depth)** (`derive/counts.py`, per 100 training steps of 38,400 queries): P-steps at depth 1–8 are 0 / 3,366 / 1,572 / 1,252 / 795 / 490 / 208 / 0 (the first and last steps are H), H-steps 38,400 / 15,058 / 9,786 / 6,246 / 4,282 / 2,859 / 1,797 / 924. A P at depth ≥ 6 occurs ≈ 700 times per 100 steps: `deepP8` is a frequency stratum, not a new order.
- **The wandering mass** (`derive/wander.py`, fixed per review 1 B). Under a uniform router over four actions, the oracle route's share of the mass halting on the target is 0.66 / 0.37 / 0.13 / 0.03 / 0.00 / 0.00 at k = 1 / 2 / 3 / 4 / 6 / 8 (oracle lengths 1.0 / 2.0 / 3.3 / 4.5 / 7.0 / 9.4 with SETs), and the mass that halts on *any* answer is 0.35 / 0.14 / 0.07 / 0.05 / 0.04 / 0.04 (the rest sits on keys or ⊥ when the horizon forces the halt). The gradient for long chains is small at init; the local rule is learned on short chains first.
- **Static reach, and the static arm's budget.** With read-only context a static depth-L learned model reaches k ≤ L (rung 0's argument), and every layer must host HOP, PREV, the flag read of the *predecessor's* pair (under δ = 1 a two-hop attention: find the pair whose value is my key, read its flag) and the pending bookkeeping. Rung 0 found that S8 learns hops one layer at a time (k = 3 at ≈ 6.6k steps, k = 4 not by 25k; F26), so within 12k steps its reach is k ≤ 3 whatever the word; S8 is run as the reach reference only (3 seeds, 12k steps; review 1 D).
- **Compute.** The oracle route costs k + n_P + 1 router calls per query (k + n_P actions and a HALT decision); a router call is one cross-attention over 49 tokens from all 390 state tokens (evaluated once per episode), ≈ 6× revision 1's cost per step; measured in the pilot.

## 6. Exact references and planted models (L17)

- **Oracle route** for every query (`exec.oracle`, from the written rule, checked against the numpy reference per episode); remaining distance at every product state: exact (`remaining_distance`).
- **Planted router (tested, both deferrals, every stratum):** head 0 matches the register against the key slots (score β = 30 on the unique match, a BOS sink at κ = 15 otherwise) and reads the matched pair's flag back into the stream; the LayerNormed linear head then tests HALT on the register's answer directions, SET on (flag − set-here), PREV on (count − set-here·[δ = 1]) minus half the flag, HOP by a constant bias. Under the exact executor it reproduces the oracle route on every query at every k ≤ 32 and every word, committed (gain 3, bias 4: head column norms 4.5 / 8.5 / 6.0, a few units from the init, L24) and marginal (gain 6, bias 8: p(target) ≥ 0.958 at k = 32, E[steps] within 0.4 of the oracle's). The per-token MLPs and the state MLP are zero in the plant; the LayerNorm magnifies the frozen directions by √d / |x| ≈ 9–18, which is what lets a few-unit head be sharp (D91).
- **Planted failures (tested):** the all-H fixed program (never the oracle route on a word with a P; its answer is right only by the 1/16 decoy coincidence); a **symbol-only router** (any weights): the oracle route takes SET and then a move at the same symbol, so no deterministic symbol-only policy produces it — committed route accuracy on P-words is 0 by construction, and without SET the pending count stays 0, PREV is unavailable and HOP at the P-step follows the decoy. In the *marginal* reading a stochastic symbol-only router can still succeed: at the flagged symbol it SETs with probability a and HOPs with 1 − a; all mass eventually hops (after a geometric number of SETs) and a fraction a of it carries a pending count; if the router infers at the next symbol that its predecessor's pair was flagged (a value-slot match, available to its heads) it PREVs there with probability → 1, so marginal accuracy on single-P words can approach a per P-step at the price of E[steps] ≈ 1/(1 − a) extra per P-step and a horizon loss. **Predicted for RM-s: committed 0 on P-words, marginal well above 0, E[steps] inflated (P9).** A router that never halts: all mass alive at T, E[steps] = T, the answer still right (answers absorb) — the failure is cost, which the step term measures.
- **The looped baseline's expressivity (argued, not planted):** HOP and PREV fit in one block as two heads that match the register against the key slot and read the value slot and the prev slot; the pending bit needs a third read (the flag of the matched pair, written to a stream direction and kept across one loop) and a gate that selects the prev read when the bit is set and clears it. One block with 2–3 heads and the 4d MLP can express one oracle move per loop at d = 344 (SET needs no loop of its own: the bit is written in the same loop as the hop). The pilot's in-range LF accuracy on all-H and on P-words is the empirical check. The **planted-gate looped block** (circuits planted, only the gate learned) that review 1 asked for as the apples-to-apples selection arm is **deferred** to a follow-up of this rung (D99): it needs a planted PREV head and planted pending bookkeeping in the neural carrier, a day of verified engineering; it is run if P5 fails (then it says whether the failure is the gate or the circuits).

## 7. Arms

| arm | what is learned | role | seeds |
|---|---|---|---|
| **RM** | π_θ(a ∣ (s, c, h), ctx), δ = 1, exact marginal, λ = 0.02 | **the thesis arm** | 10 |
| **RM-s** | as RM with the symbol-only state token | the F4 null: is the explicit richer state needed? | 5 |
| **RM-d0** | as RM with δ = 0 (local trigger) | the control: same library, no carried bit; PP runs local | 5 |
| RM-T | as RM, trained by teacher forcing on the oracle route (δ = 1) | recognition upper bound / speed reference (NPI-style traces) | 5 |
| RM-idx | π_θ(a ∣ t) only (a learned fixed program) | the state-blind null; exact-circuit static | 5 |
| **LF / LR** | one looped learned block, T = 10 (rung 0's model, d = 344, prev slot and flag in the token), the stability exit at test; also read at the oracle depth | the m = 1 learned alternative: memory inside the block | 10 |
| LF-d0 | as LF with δ = 0 | the block with a local trigger | 5 |
| S8 | 8 unique learned layers, lr 1e-4, 12k steps | static learned stack; reach reference (F26) | 3 |
| S2 | 2 learned layers | compute-matched static sanity | 3 |
| oracle, fixed programs, state-blind table | — | exact references (§5, §6) | — |

Dropped from revision 1: RM-λ0 (P6 untestable, §10). Not run, with reasons: a soft superposed stack with context-conditioned mixing (dominated by S8 on the depth argument); per-layer head selection (a static stack with routing within a layer; its failure on new depths is S8's); transfer to new alphabets (empty by construction); the planted-gate looped block (deferred, §6).

## 8. Cells and grid

| cell | arms × seeds | steps | purpose |
|---|---|---|---|
| pilot | RM × lr {3e-4, 1e-3, 3e-3} × 2 seeds; RM λ ∈ {0.005, 0.1} × 2; RM-s × 2; RM-T × 2; LF lr {3e-4, 1e-3} × 2; S8 lr {3e-5, 1e-4} × 1 | 4k (routed), 6k (LF), 8k (S8) | rates, λ, the gate (§11), cost |
| main | RM 10, RM-s 5, RM-d0 5, RM-T 5, RM-idx 5, LF 10, LF-d0 5, S8 3, S2 3 | routed 6k, LF 6k, S8 12k, S2 8k | P1–P9 |

Cost: routed ≈ 0.08 s per run-step (estimated at 6× revision 1's measured 0.02 s; the pilot replaces it), LF ≈ 0.072, S8 ≈ 0.058, S2 ≈ 0.015: main ≈ 25 × 6k × 0.08 + 5 × 6k × 0.02 + 15 × 6k × 0.072 + 3 × 12k × 0.058 + 3 × 8k × 0.015 ≈ **6.0 T4-hours**, pilot ≈ 2.0; two T4 shards. Batching: the routed arms stack over runs like rung 0's models (a leading run dimension on the router; the executor is shared), the neural arms with `stacked.py`.

## 9. Endpoints

By stratum (in range, split by words with / without a P; all-H controls; length; new orders within length; frequency strata; new orders long), run-level:
1. Route accuracy (committed route = oracle route), answer accuracy (committed), marginal accuracy, steps and E[steps] against the oracle's; for the neural arms answer accuracy and steps (LR's steps), and for LF the accuracy at the oracle depth (loop k, k + 1, k + n_P) and at any loop, from the per-loop readouts (review 1 C.P5).
2. **Held-out criterion curves** (every 100 steps, a fixed held-out set of 256 episodes per run; review 1 D): committed route accuracy by k (answer accuracy by k for the neural arms), accuracy by the depth of the deepest P (pooling for selection), and for the routed arms the teacher-forced accuracy by action along the oracle route, from which the **SET and PREV flip times** are read.
3. **Residual study (D73, halting part):** along the committed route, p(HALT) and the router's entropy against the exact remaining distance (0, 1, ≥ 2, dead).
4. **The first deviation** from the oracle route (oracle action > taken action), summed over failures per stratum: which decision fails.
5. Router heads: column norms and biases (L24); the index arm's table.
6. The state-blind references (§5) and the oracle on every table.

## 10. Predictions (thresholds fixed here; run-level paired bootstrap, per-seed counts)

| # | prediction | falsifier | type |
|---|---|---|---|
| **P1** | **Learnability with a setup step (B2, D74).** RM reaches committed route accuracy ≥ 0.95 on words with a P at k ≤ 8 (`route_acc_P` on in4–in8) in ≥ 8/10 seeds within 6k steps; the order of acquisition on the held-out criterion set is HALT and HOP first, then SET and PREV *together* (neither has value without the other; a joint flip within the same 300 steps); RM-T flips both earlier (the speed difference is the setup-credit cost). | `route_acc_P` < 0.5 on in8 in ≥ 5/10 seeds at 6k steps while RM-T ≥ 0.95: the exact marginal does not credit the setup step enough to find SET→HOP→PREV from the likelihood (the E-step is exact, so the basin is the problem, as in rung 0's halting). Both < 0.5: the router architecture cannot read the flag / pending inputs. | optimization |
| **P2** | **Length (by construction given P1; review 1 C).** RM route accuracy at len16 and len32 within 0.05 of in8 in ≥ 8/10 seeds (no step input; the states at k = 32 take the same (symbol, c ≤ 1, h) values as in training). | A drop > 0.1 at len32 relative to in8: the router conditions on something that varies with length (then the first-deviation table says which decision). | mechanism (sanity check of the Markov form) |
| **P3** | **New orders (B3, the headline).** PP runs under δ = 1 visit c = 2, never seen in training. Prediction (moderate confidence): RM route accuracy on pp4/pp6/pp8 ≥ 0.8 in ≥ 6/10 seeds, because the count enters the state token linearly (c · direction) and the learned PREV decision extrapolates along that ray; on pp16/pp32 within 0.1 of pp8. RM-d0 is at its in-range level on pp (PP runs are local there: the same-context, no-new-state control), and the frequency strata (alt, deepP8) are at RM's in-range P-word level. | RM pp4 < 0.5 in ≥ 5/10 seeds while `route_acc_P` on in4 ≥ 0.95: the router's use of the count is not ordinal (it matched c = 1 as a category); the module-network outcome at the level of the state, and the first-deviation table shows PREV>HOP or PREV>SET at the c = 2 state. Either outcome is informative; the design does not favour one. | mechanism |
| **P4** | **Pooling for selection (Entry 50).** RM's held-out accuracy on words whose deepest P is at depth d is flat in d (≥ 0.9 for every d ≤ 7 once in range is learned), because one decision is shared across depths; LF's and S8's fall with d (the P-step count at depth ≥ 5 is ≈ 1.5k per 100 steps, F26's regime); for RM this is by construction (Markov + the same inputs) and the content is the neural arms' curves. | RM below 0.8 at some d ≤ 7 while ≥ 0.95 at others: the selection rule did not pool. | by construction (RM) / optimization (LF, S8) |
| **P5** | **The m = 1 learned alternative (the open one).** The looped block must hold the pending bit inside its stream across a loop. Prediction: LF reaches ≥ 0.9 answer accuracy on all-H words in range (`none4`, `none8`, `acc_noP`) but ≤ 0.6 on words with a P at k ≥ 4 (`acc_P` on in4–in8) within 6k steps in ≥ 7/10 seeds, **with its oracle-depth reading no better** (so the block, not the exit rule, is the limit); LF-d0 (local trigger) exceeds LF on `acc_P` at in8 by ≥ 0.2 (the carried bit is the hard part); LR's extrapolation on all-H words replicates rung 0 (≥ 0.9 at len16). | LF `acc_P` ≥ 0.9 on in8 in ≥ 5/10 seeds: the block learns the deferred memory as well as the router; the explicit register has no measured advantage at m = 2 and the contrast moves to learned circuits (rung 2). LF at the oracle depth ≥ 0.9 while LR ≤ 0.6: the exit rule, not the block (D93's territory). | mechanism (memory inside the block vs an explicit register); the library question's first data point |
| P6 | **Removed.** With absorbing answers and λ = 0 the HALT logit at answer states gets no likelihood gradient, so "halts late, accuracy unchanged" is guaranteed and there are no equivalent routes of different content to compare; λ identifies the halting time and nothing else is testable here (review 1 C). | — | — |
| **P7** | **The distinguishing quantity, per reading.** RM-idx reaches the state-blind references of §5 in range (committed answer accuracy 0.84 ± 0.05, marginal 0.85 ± 0.05, Zipf-weighted; route accuracy 0 by construction) and ≈ 1/16 on P-words; RM exceeds it by ≥ 0.1 on Zipf-weighted in-range answer accuracy and by ≥ 0.8 on `acc_P` at in8. | RM within 0.05 of 0.84 on in-range answer accuracy: the router did not learn the P-rule (coincides with ¬P1). RM-idx above 0.9 in either reading: the fitted reference of §5 was wrong (re-derive). | expressivity |
| **P8** | **Halting (D73, the testable part).** Along the committed route RM's p(HALT) is ≥ 0.95 at remaining distance 0 and ≤ 0.05 at distance ≥ 1, in range and at len32 and pp8 (the halting decision is "answer in register", a state feature; D93's failure mode — halting on "no change" — cannot occur because SET changes nothing and the router has no change feature). | p(HALT) > 0.2 at distance ≥ 1 at len32: the router halts on something that varies with length. | mechanism (diagnostic) |
| **P9** | **F4: the explicit richer state.** RM-s (symbol-only): committed route accuracy on words with a P is 0 at every k (by construction, §6) and its answer accuracy on them ≤ 0.15 (the 1/16 decoy coincidence); its all-H performance equals RM's (`route_acc_noP` within 0.05). Its **marginal** accuracy on single-P words is well above 0 (predicted 0.3–0.8 on in4) with E[steps] exceeding the oracle's by ≥ 1 per P-step — the stochastic SET trick of §6, which the committed reading cannot use. | RM-s `route_acc_P` > 0 anywhere: the executor or the state token leaks (c, h) (a bug). RM-s marginal `acc_P` ≈ 1/16: the router did not find the stochastic trick either (then its first-deviation table says whether it ever SETs). | by construction (committed) / mechanism (marginal) |

**Reading the outcomes.**
- P1 ∧ P3: the exact E-step learns a deferred rule with a setup step from next-token likelihood, and the explicit count extrapolates to an unseen value; Q1 and Q2 answered yes with exact circuits; rung 2 (neural primitives) is next.
- P1 ∧ ¬P3: the rule is learned but the state is read categorically; the next test is an ordinal state encoding (thermometer / magnitude) before rung 2, because rung 2's learned primitives will inherit the router.
- ¬P1 ∧ RM-T passing: the setup credit is the obstacle even with an exact E-step — the likelihood basin does not reach SET→HOP→PREV in 6k steps; the thesis's "learned from next-token loss" fails for setup steps at this scale, and curricula or an auxiliary signal become the question (F-level record, not a fix in this rung).
- ¬P1 ∧ ¬RM-T: architecture (the router cannot read the flag or the pending inputs); fix and rerun.
- P5 (LF fails on P-words with the oracle-depth reading no better, LF-d0 better): memory inside the block is the block's bottleneck; the explicit register is the library's first measured advantage, and the planted-gate block (D99) is the next arm.
- ¬P5: a single gated block holds the deferred bit as well as the router; the library's advantage, if any, is not generalization at m = 2 and must be sought in pooling at larger m or in compute (rung 2–3).
- P9's marginal line: whatever RM-s does in the marginal reading is a record about the exact marginal (a stochastic policy can substitute for state), not about the thesis reading.

## 11. Gates before the main grid

- **G1 tests (passed, 12):** generator (flags, decoys, oracle routes, strata classes, both deferrals), executor against the numpy reference (states, actions, distance), the exact marginal against brute-force enumeration over 3^L routes (state and step-indexed forms), the planted router on every stratum (both deferrals; committed exact, marginal sharp), the planted failures (all-H program, symbol-only routers, never-halt), run independence, teacher loss and action accuracy on the plant, the flag in the neural embedding, the trainers' record fields.
- **G2 derive (done):** §5 computed with revision-2 code (`derive/coverage.py`, `counts.py`, `wander.py`, `stateblind.py`).
- **G3 pilot:** learning rates and λ for RM chosen by in-range route accuracy at 4k steps (not by generalization strata); LF's and S8's rates; the cost per step. The pilot's RM-s and RM-T are checked against P9 and P1's ordering (a sanity gate, not a selection). If no RM pilot configuration passes `route_acc_P` ≥ 0.5 on in4 at 4k steps, the main grid still runs (P1's falsifier is a result), but the design is not changed after seeing pilot data except for the rates.
- **G4 reviews:** review 1 (done, `review/`); a short second review of this revision and of the code before the pilot.

## 12. Fairness (AGENTS rule 4)

- Learning rates per arm from the pilot; S8 bracketed (3e-5, 1e-4); the same frozen symbol bases and tokens for every arm (the flag on the same frozen direction); the same seeds (paired episode streams).
- Compute: steps per query (router calls or loops); S2 is compute-matched; the exact executor's cost is not counted against the routed arms because the neural arms' blocks are also one layer per step — stated as a limitation: the routed arms are given the circuits (review 1 A.iii: RM vs LF measures learned circuits + in-stream memory against given circuits + an explicit register; the planted-gate block that separates the two is D99).
- 10 seeds for the headline arms (RM, LF), 5 elsewhere, 3 for S2 and S8; per-seed counts in every prediction; paired bootstrap.
- Time-to-criterion on a held-out set per seed (review 1 D), every 100 steps, the same set at every evaluation; the training-batch accuracy is recorded beside it as the memorization reference; schedules differ by arm (stated).

## 13. What this rung does not answer

- Whether neural primitives can be made to act on one shared state without drift (rung 2, Q3), or learned jointly with the router without the damage of F22–F23 (rung 3, Q4).
- Hard routing, search and a learned value (rung 4): here the E-step is exact because the state is symbolic; nothing is said about finding routes when it is not.
- Selection vs circuits within the neural carrier (the planted-gate looped block, D99).
- Whether the remaining distance is represented anywhere (the probe of revision 1 had a ceiling of {0, 1, ≥ 2}; Entry 47's question needs a state that carries progress).
- m > 2 circuits and the basis-size frontier (Entry 51 level 1 applies; not run).

## 14. Prior work

Neural Module Networks and layout induction (Andreas 2016; Hu 2017 N2NMN; Johnson 2017 IEP; Bahdanau 2019 SQOOP; CLOSURE): layouts inferred from the input fail systematic generalization unless given. **Neural Programmer-Interpreter** (Reed & de Freitas 2016) and its **recursive** version (Cai 2017): a controller over exact programs trained on traces, generalizing in length once it conditions on the local frame — RM-T is that form and the Markov router is the recursion insight. **NRAM** (Kurach 2016; its fuzzy execution *is* this forward recursion), Neural Programmer (Neelakantan 2016), Neural GPU, differentiable Forth: soft selection over exact or differentiable modules trained from outputs, with length generalization as the test. **Neural Program Lattices** (Li 2017): marginalizing over latent traces with a forward lattice, the closest form of the exact E-step. **Compositional Recursive Learner** (Chang 2019): a controller composing a library serially with a halt action, tested on longer and novel compositions. RASP-L and length generalization (Zhou 2023). Transformer circuits (Elhage 2021, Olsson 2022): HOP and PREV are the copy/induction and previous-token heads. Review 1 E: P1–P3 of revision 1 read as replication of these; the contribution of revision 2 is the **setup step credited only through the exact marginal** (D74), the **state-only vs symbol-only contrast** (F4) on a trigger that must be carried, exact oracle routes on a token-sequence carrier, and the looped-block contrast as "memory inside the block vs an explicit register".

## 15. Review handling (revision 1 → 2)

| review 1 item | handling |
|---|---|
| A.ii "new orders" in-distribution for a Markov router with a local trigger | the **deferred trigger** (δ = 1): the flag one state earlier, a SET setup step, a pending count carried across a hop; PP runs reach c = 2, outside the training support of the state; δ = 0 kept as the control arm |
| A.i RM vs RM-T a foregone tie | kept as the speed reference; the information is the SET/PREV flip time (setup credit, D74) |
| A.iii RM vs LF apples to oranges; add the planted-gate block | the contrast is relabeled "memory inside the block vs an explicit register" and LF is read at the oracle depth; the planted-gate block is deferred (D99, §6) with the reason |
| B P7 wrong on the marginal reading (0.94) | decoys at P-steps (wandering mass votes for wrong answers); the state-blind reference re-derived in both readings (0.84 / 0.85, §5); P7 restated per reading |
| B λ makes RM-idx halt early; report by k | §5 reports it by k; P7 uses Zipf-weighted in-range accuracy |
| B wander.py defects | fixed (oracle share with post-answer absorption; mass on any answer via a zero-row readout) |
| B design/code mismatches | §2/§4 now say: no attention among pairs; all states evaluated; d = 344; the plant's gains |
| C P2, P3 foregone; alternation/deep-P not new orders | P2 labeled by construction; alt/deepP moved to frequency strata; P3 rests on pp (c = 2) |
| C P4 is F26 again | kept as a cheap curve (held-out, by deepest-P depth) for every arm; S8 cut to 3 × 12k |
| C P5 conflates block and exit rule | LF at the oracle depth (k, k + 1, k + n_P, any) recorded; P5 restated |
| C P6 degenerate | removed; RM-λ0 dropped |
| C P8 probe cannot pass | probe dropped; p(HALT) by distance kept |
| D held-out criterion | a fixed held-out set per run every 100 steps (`crit_n` = 256); training accuracy kept as the memorization reference |
| code review 1–3 (strata, P/no-P split, Pdepth for routed arms, answer-acc by k) | `none3/4/8` strata; `*_P` / `*_noP` per stratum in both trainers; `acc_by_Pdepth` and `ans_acc_by_k` on the routed criterion curve |
| code review minor (pp runs merging; split-retry; init seed by stack) | pp words now have H on both sides of the forced run; both halves retried; the stack-composition seed is kept and noted (a resumed stack re-inits differently; runs are still independent) |
| E prior work | §14 (NPI recursive, NRAM, Neural Program Lattices, CRL) |
