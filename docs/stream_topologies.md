# Streams in a loop: four lenses, predictions, and three rounds of checks

**15 Nov 2026. Exploration: derivations, random-initialisation analyses and small linear students. EXPLORATORY throughout; nothing here is a result about a trained network of any size.** Written for the researcher's request (history Entry 101) to look at the stream idea from several mathematical angles, state expectations about learnability and stability, verify them in code, and iterate. Companion to [routing_signal_stability.md](routing_signal_stability.md). Code and raw tables: [`experiments/theory/streams/`](../experiments/theory/streams/). Exact statements are tested in [`tests/test_stream_topologies.py`](../tests/test_stream_topologies.py). Every prediction was written into its script before the run.

## 0. Scoreboard, and what I would now say

| round | predictions | held | held in part | failed |
|---|---|---|---|---|
| 1 | 17 | 9 | 4 | 4 |
| 2 (revisions of round-1 failures) | 7 | 4 | 3 | 0 |
| 3 (the remaining failure) | 2 | 1 | 0 | 1 |

What survived three rounds:

1. **A gate that is told the current instruction makes streams learnable as primitives.** Held-out programs are computed exactly; with a softmax gate each teacher primitive ends with its own stream (cosine ≥ 0.99) even though nothing in the function requires it. The number of streams needed is the affine dimension of the primitives plus one, not their count.
2. **Order is a second-order quantity.** A student blind to order has an error floor proportional to ε² (step size squared) times the commutator of the primitives. With small residual steps a loop has almost nothing to gain from order; the gate's order-dependent learning signal is proportional to ε.
3. **A gate that reads only the state can run a stored program, stably, and training does not find it.** The hand-built solution is exact and survives noise; from scratch the student lands on the order-blind solution every time. An always-on bookkeeping stream did not help.
4. **Averaging different streams contracts.** Blending by a soft gate on a shared state, and coupling separate states through a graph, both pull the loop's gain toward that of the mean stream. I had left this out twice.
5. **A sharp state-read gate raises the gain through its own boundary term**, for most inputs, not through the composition of different streams (which lowers it). Its learning signal lives within a few temperatures of a region boundary.
6. **With separate stream states, diversity is set by the whole Laplacian spectrum of the coupling graph**, influence between streams has an exact light cone, and explicit coupling is unstable past c·λ_max = 2.

## 1. The family

One pass of a looped block with K streams f_k. The ways of organising it differ along three axes:

| axis | options |
|---|---|
| how streams recombine | plain sum · gated sum (soft) · one picked (hard) |
| what the gate sees | nothing (a constant) · the pass index · the current instruction · the state |
| what the streams share | one residual state · separate states coupled through a graph · nothing (an ensemble) |

The carrier for the trained checks: a state h ∈ ℝ^d, P instruction tokens, a program u = (u_1 … u_n); the teacher applies h ← h + ε A_{u_t} h; a student applies h ← h + ε Σ_k g_t[k] B_k h. Error is the squared error of the change h_n − h_0 relative to the teacher's change. Serial composition of streams inside a pass was not explored.

## 2. Lens 1 — symmetry and gauge: streams as a dictionary of operators

**Structure.** Permuting the streams leaves the block unchanged, so the configuration with identical streams is preserved by training. With linear streams and gates g_t = E[u_t], the student depends on (E, B) only through each token's *effective operator* C_i = Σ_k E_ik B_k: the K streams are a dictionary and a gate row is a set of coordinates. With unnormalised gates there is a continuous gauge freedom (E → EG, B → G⁻¹B); a softmax gate restricts each C_i to the convex hull of the streams.

| # | prediction | result |
|---|---|---|
| 1.1 | identical streams stay identical (exact) | held: largest difference 0.0 after 600 steps; the student is then one effective stream (held-out error 0.44) |
| 1.2 | gauge change leaves outputs unchanged (exact) | held: 1.2e-14 |
| 1.3 | K = 4 streams, 3 primitives, softmax gate: held-out error < 0.01, operators within 10 %, gates soft, **no stream matches a primitive** | generalisation held (error ~1e-32 on five seeds), gates soft (mean top gate 0.80–0.87); **the last clause failed**: every primitive had a stream at cosine 0.99–1.00, scaled by about 1/0.87 to make up for the soft gate |
| 1.4 | streams needed = affine dimension + 1: four primitives on a segment need K = 2; four generic ones fail at K = 2 (> 0.2) and K = 3 (> 0.05), succeed at K = 4 | held: 1e-31; 0.24; 0.12; 1e-31 |
| R1 | the alignment in 1.3 comes from the softmax's competition: unnormalised gates reach zero error with at least one primitive below cosine 0.9 | held on two of three seeds (0.77, 0.85), borderline on one (0.90); softmax 0.99 on all three |

**Reading.** In this setting the worry of §6 of the routing note — that a gated sum learns a blend and not primitives — did not materialise: competition between streams was enough to make them primitives, and the remaining softness was absorbed by rescaling. What made it easy is that the gate was *told* which instruction applies.

## 3. Lens 2 — order: streams as control vector fields

**Structure.** h ← h + ε Σ_k g_t[k] f_k(h) is an Euler step of dh/dt = Σ_k g_k(t) f_k(h): streams are vector fields, the gate is the control, a program is a control signal. Two orders of the same steps differ by the commutator, (I + εA)(I + εB) − (I + εB)(I + εA) = ε²[A, B] (exact). A gate that is constant over the passes realises (I + ε Σ_k g_k B_k)^n and cannot represent order.

| # | prediction | result |
|---|---|---|
| 2.2 | order-blind ("bag") student: floor ∝ ε², × 3–5 per doubling; ordered student < 1e-4 | held: 5.7e-4, 2.4e-3, 1.0e-2, 4.2e-2 at ε = 0.05 … 0.4 (ratios 4.1, 4.2, 4.2); ordered ~1e-31 |
| 2.3 | order-dependent share of the gate's gradient ∝ ε (× 1.6–2.4 per doubling) | failed as measured (× 2.7–3.3) |
| R2 | 2.3 pooled a set of programs closed under reversal, cancelling the first-order term; program by program it is linear | held at small steps (× 2.00, 2.40), super-linear by the last doubling (× 2.94) |
| 2.4b | gate given the pass index and the stored program generalises (< 0.01) | held: 8e-6 |
| 2.4c1 | gate reading only the state ends above 0.1 | failed as stated: 0.036 |
| 2.4c2 | an always-on stream lowers that threefold | failed: 0.034 |
| R3 | 0.036 is the order-blind floor; the ordered solution is representable (< 1e-6), locally stable (< 1e-3 from 5 % noise), and not found (from scratch, three times the steps, above half the floor) | all four held: floor 0.034; hand-built 7e-27; from noise 4e-5; from scratch 0.037 [0.021, 0.051] |

**Reading.** Even at ε = 0.4 with four passes, order accounts for 4 % of the variance of the change. So the information that distinguishes a program from a bag is small in exactly the regime where a loop is well conditioned, and grows only as steps get large. And where the gate must *infer* its place in the program from the state, the solution exists and is stable but lies outside what gradient descent reaches from a small start: all three seeds learned the bag.

## 4. Lens 3 — a gate that reads the state: a partition of state space

**Structure.** With g = softmax(Vh/τ) the loop iterates one glued vector field F(h) = Σ_k g_k(h) f_k(h): the gates are a soft partition of unity and each stream is the local rule on its region. The sequence of regions a trajectory visits is its itinerary — the program that input runs. The pass Jacobian is I + Σ_k g_k ∂f_k + (1/τ) F C V with C = diag(g) − ggᵀ. Random tanh streams, no training.

| # | prediction | result |
|---|---|---|
| 3.1 | typical gain of the loop peaks at an intermediate temperature | failed: it rises monotonically as the gate sharpens (median 3.0 at τ = 4 to 7.6 at τ = 0.05) |
| 3.2 | the tail grows as the gate sharpens (99th percentile / median at τ = 0.05 more than 3 × that at τ = 1) | held: 13.4 against 2.1 |
| 3.3 | the tenth of inputs nearest a boundary carries < 25 % of the gate's gradient at τ = 1 and > 50 % at τ = 0.1 | first half held (8 %), second failed (14 %; 20 % at τ = 0.05) |
| 3.4 | a 1 % input perturbation changes < 10 % of itineraries, ≥ 80 % of them near a boundary | held: 4.6 %, 88 % |
| R4 (i) | missing ingredient: averaging contracts. At τ = 4 the gain equals the uniform-blend loop's (within 10 %); at τ = 0.05 it equals single-stream loops' (within 25 %) | first held (3.04 against 2.97); second failed (7.75 against 5.17) |
| R5 | a tenth was the wrong yardstick; per input the gate's gradient falls with margin/τ: beyond 5τ it is under a tenth of that within τ | held: ratios 0.005 and 0.002 |
| round 3 | the excess over single streams is composition: with itineraries frozen the gain stays ≥ 6.2; far from boundaries soft = frozen | first failed: frozen 4.33. Second held: 5.58 = 5.58. The excess is the gate's boundary term (near a boundary: soft 8.87, frozen 4.00), and itineraries that switch streams have *lower* frozen gain (3.90) than single-stream ones (5.77) |

**Reading.** Three effects, now separated: blending averages streams and lowers gain; composing different streams along an itinerary lowers it further; the gate's own derivative raises it, and at τ = 0.05 with six passes 78 % of inputs pass within 5τ of a boundary somewhere, so this is the typical case and not a tail. The router is taught only by inputs near its boundaries (R5), which are also the inputs where the loop is least well conditioned.

## 5. Lens 4 — separate stream states coupled through a graph

**Structure.** X_k ← X_k + ε f_k(X_k) + c Σ_j W_kj (X_j − X_k), i.e. X ← X + εF(X) − cLX with L the graph Laplacian. c = 0 is an ensemble of K loops; a complete graph with c = 1/K averages the states every pass, which is the shared residual stream. Disagreement mode i is damped by (1 − cλ_i) per pass.

| # | prediction | result |
|---|---|---|
| 4.1 | diversity ordered none > ring > star > complete; complete graph matches (1 − (1 − cK)^n)/(cKn) within 30 % | formula held (within 4 %); ordering failed: the star stays more diverse than the ring |
| R6 | every mode counts at eight passes: D = D₀·sqrt(mean_i φ(cλ_i)²) predicts all three graphs within 15 % | held: within 4 % at every coupling |
| 4.2 | light cone: on a ring, stream k's state after n passes is independent of stream j's parameters when n < 1 + distance | held exactly (zeros where predicted) |
| 4.3 | complete graph unstable at c = 0.3, stable at 0.2; ring stable at both | held (diversity 0.31 → 4.17; ring 1.61 → 1.29) |
| 4.4 | coupling leaves the common-mode gain unchanged and damps the relative mode | relative mode held (2.57 → 0.03); common mode failed (2.16 → 1.27) |
| R4 (ii) | at c = 1/K the common-mode gain equals that of a shared state adding the mean of the streams | held: 1.28 against 1.24 |

**Reading.** The star beat the ring because it has fewer edges: at a handful of passes total diversity follows the whole spectrum (roughly the number of edges), and the second eigenvalue matters only in the long run. Coupling is not neutral on the shared direction, because streams that respond differently to the same input turn a shared perturbation into disagreement, which the coupling then removes.

## 6. Expectations about each way of organising, after the checks

| organisation | learnability | stability |
|---|---|---|
| gated sum, gate told the instruction | learned exactly here; streams become primitives under a softmax gate; held-out programs exact | no problem observed |
| gate constant over passes | learns the order-blind part; floor ∝ ε²‖[A, B]‖² | the best conditioned |
| gate reads the pass index and a stored program | learned (8e-6 held out) | no problem observed |
| gate reads only the state | representable and locally stable; not found from scratch (3 of 3 seeds learned the bag) | gain rises as the gate sharpens, by the boundary term; signal confined to within a few τ of boundaries |
| separate states, weak coupling | not trained here | diversity kept; gain not damped |
| separate states, strong coupling / shared state | — | gain damped toward the mean stream; explicit coupling unstable past cλ_max = 2 |

Three trade-offs run through all of it, each with one knob:
- **step size ε** — order information grows as ε² while conditioning worsens;
- **temperature τ** — a readable itinerary against gain and a signal confined to boundaries;
- **coupling c** — mixing between streams against their diversity.

## 7. Limits

Linear streams in every trained check; d = 4–8; two to four primitives; four passes; three to five seeds; one optimiser and step size per check. Lens 3 and lens 4 are random-initialisation analyses, not trained models. The state-read result (R3) is one task and one way of storing the program. "Held in part" entries are listed with their numbers above; thresholds were mine and some were badly chosen (2.4c1's was above the order-blind floor it should have been compared with). Not explored: serial composition inside a pass, hard picks, nonlinear streams in the trained checks, any interaction between the lenses.
