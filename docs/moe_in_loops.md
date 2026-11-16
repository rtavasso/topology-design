# Experts in a loop: a formalisation, what the literature does for stability, and a search for the architecture

**Status: EXPLORATORY.** Small synthetic testbed, CPU, five seeds per cell (ten for the final round). Nothing here is a result about a trained transformer; it is a search for a candidate and for the reasons behind each of its parts. Written 15 Nov 2026 (history Entry 104). **Independently reviewed the same day** (`experiments/theory/streams/review/review_moe_in_loops.md`; verdict: sound with corrections — no bug, every number matched, three of my readings did not survive). The corrections and the checks the review called for (round H) are folded in; where a reading changed, the text says so. Code: `experiments/theory/streams/loopmoe.py` and `arch_*.py`; tables `arch_*.md` beside them; exact statements in `tests/test_moe_in_loops.py`. Predictions were written into each script before it ran.

The object, as the researcher put it: several learned streams **in parallel inside one looped block**; they all read the same state and a gate decides what each writes back; the gate changes across loop passes, depends on **the original sequence and the state**, and acts **before** the streams so that an unneeded stream is not computed.

## 0. In brief

**The candidate** (§5). One looped block whose state is the data plus a small control register. Each pass: the register, advanced by an always-on linear map, forms a query; the query attends over the original sequence and reads back the instruction for this pass; that reading, together with the data state when choices depend on the data, gives one logit per stream; a softmax over all streams (never renormalised) is the gate. In training every stream runs until the gate has decided (mean top probability above 0.8), after that — and always at run time — only the top stream runs.

**What it did** (round E, ten seeds). One stream per pass matched the network that runs every stream: program task 3.0e-5 against 3.2e-5 (four streams), 1.8e-5 against 2.6e-5 (eight streams); branch task 1.1e-3 against 1.4e-3; all 10/10. Trained on 3–5 passes it ran 6 and 8 with no loss (10/10; 3.1e-5 at eight passes). Training cost 1.1–2.3 streams per pass instead of 4 or 8. Five other teachers gave the same (round H).

**What the search found** (84 predictions written before the runs: 57 held, 8 partly, 19 failed).

1. **The gate needs two separate things from the state.** A progress signal — which pass is this — and, for data-dependent choices, the data. A query read from the data state alone leaves the gate order-blind (0/5). A register that the block advances every pass solves it (5/5), as well as one router per loop does, and it learned to move from a standing start — I had predicted it often would not. The register is itself a clock, carried in the state; what it adds over one router per loop is a single tied gate and (point 5) extra passes.
2. **Gating before the streams from the first training step is unreliable, and how unreliable depends on what the gate reads.** Gate that reads only the instruction: solved 2/10 (0/5 when the register must also learn to move), and none of the published aids I tried rescued it at five seeds (sigmoid scores, expert dropout, selection noise, stand-ins for unrun streams, an extra explorer stream). Gate that also reads the data: 8/10 with four streams, 3/10 with eight, 10/10 on the branch task. *Corrected by the review:* I first wrote that it "cannot be trained sparse from the start"; that was a statement about instruction-only gates.
3. **A dense start makes it reliable; why is not established.** Every stream run until the gate's mean top probability passes 0.8, then only the top one: 10/10 in all seven ten-seed cells, equal to running everything throughout. A fixed 10 % start was 5/5 twice but 8/10 and 9/10 at eight streams. The idea is EvoMoE's dense-to-sparse gate, found after the fact. *Corrected by the review:* my explanation (A7: sparsity costs information only while the gate is undecided) is a sufficient condition, not the reason — runs switched after 150 of 3,000 steps, with mean top probability 0.5, still solved 8/10. The reviewer's alternative (the address and a one-to-one token-to-stream assignment are in place at the switch) failed the same test: agreement 22/40.
4. **Softmax or sigmoid scores: not settled.** At top-2 from the start softmax did better (5/5 against 2/5, five seeds — the opposite of the published recommendation and of my prediction), but under the candidate's own schedule sigmoid solved 9/10 and softmax 8/10 (softmax with 4.7 times lower median error). The one thing to avoid is renormalising over the streams that ran: at top-1 the gate then gets no gradient (A3d; 0/5). *Corrected by the review:* I first reported softmax as the finding.
5. **More passes than trained needs a built-in advance.** A learned advance, or one router per loop, fails at six passes (0–2/5); a register started as the exact one-position shift runs eight (5/5), and the streams' error does not grow with the extra passes.
6. **Reading the data is where the loop's sensitivity rises.** Register-driven switching adds no term to it (A2); with the data read, the largest gain over 96 sampled cases was 2.7 times higher. No run diverged.
7. **Against one monolithic block (round M, added after the first version of this note): streams do not recombine better, gating is what buys accuracy, and running one stream buys compute.** With the same controller, no block had a gap between held-out and training programs (ratio 0.98–1.22 in all sixteen rows), so on this testbed recombination comes from the controller, not the block. At equal parameters a block that takes the instruction as one more input was about 80–170 times worse than the candidate (36 times at twice the parameters, 41 times at twice the training). A block with a soft gate on every hidden unit, all of them computed, matched the candidate (0.8 and 1.1 times its error) at 3.3 and 5.6 times the multiply-adds per pass.

**Limits** (§7). One-vector state, tiny streams, at most eight passes; the register is a clock carried in the state, not yet control flow; the sequence names its operations outright; one teacher for all but two checks; why the dense start works is not known; the monolithic baselines of round M are untuned beyond their learning rate.

## 1. The family

State after pass t: the data state h_t ∈ ℝ^d and, optionally, a small control register c_t ∈ ℝ^m. The original sequence is u = (u_1 … u_n); position i has a key κ_i and its token a value v_i = E[u_i]. One pass of the block:

| Step | Formula | Options searched |
|---|---|---|
| 1. Query | q_t = Q(state) | a constant; one learned vector per pass ("clock" — the analogue of one router per loop); a read of h_t; the register c_t |
| 2. Address the sequence | a_t = softmax_i(q_t · κ_i),  r_t = Σ_i a_{t,i} v_i | learned keys; fixed sinusoidal keys |
| 3. Selection logits | s_t = G r_t  (+ MLP(r_t, h_t) if the gate also reads the data) | data read: none / normalised / raw |
| 4. Gate, before the streams | g_t = Γ(s_t): scores, then keep the top k | softmax or sigmoid scores; renormalised or not; k = K (dense), 2, 1; noise, expert dropout, usage-entropy bonus, stand-ins, explorers, schedules of k |
| 5. Streams that were kept | h_{t+1} = h_t + ε Σ_{k kept} g_{t,k} f_k(h_t),  f_k(h) = V_k tanh(B_k h) | K = 4 or 8 streams |
| 6. Register | c_{t+1} = c_t + M c_t  (always on, linear) | M starts near zero; starts as a rotation; frozen rotation; the exact one-position shift |

Everything is tied across passes except in the "clock" option. Two tasks, both with a ground truth for which stream should run when:

- **Program.** Teacher primitives tanh(A_p h), p = 1…P. The operation of pass t is token u_t. A gate that knows the whole program but not the pass is "order-blind" and has an error floor (0.030 here).
- **Branch.** One more token, IF: apply primitive 0 if ⟨w, h⟩ > 0, else primitive 1. Only a gate that reads the data state can do it.

Error throughout: squared error of the change h_n − h_0 relative to the teacher's change, **on programs never trained on** — except the length-4 column of the mixed-length rounds D and E, which is "in range": its programs are drawn independently and about 39 % also occur in training (lengths 6 and 8 are clean). "Solved" means below 10⁻². Every round uses one random teacher (seed 10) except H9.

## 2. Seven exact statements

Each is a test in `tests/test_moe_in_loops.py`. A1–A6 were written before any round; A7 after round B2. Their weight, as the review assessed it: A3, the first half of A2, A5 and the bound in A7 are substantive; A1 is true by definition; the second halves of A2 and A4 are tested only qualitatively; A6 is exact only on a symmetrised batch (on an ordinary batch the stand-in is small, 0.09 against a typical stream output of 0.45, not zero); A7 is a statement about one pass.

| | Statement | What it is for |
|---|---|---|
| A1 | A tied gate's output can differ between two passes only if its input differs. With a constant query, or a register that does not move, the gate is the same on every pass whatever the program. | Reconciles two papers that disagree (§3): a tied router varies across passes exactly as much as the state it reads does. |
| A2 | If the gate reads only the sequence and the register, and the register's update does not read the data, then ∂h_n/∂h_0 is the product of the data-path factors I + ε Σ_k g_k f_k′ with the gates held fixed. The gate contributes **no** term. If the gate reads h, it contributes ε F (∂g/∂s)(∂s/∂h) per pass. | Separates the two kinds of state dependence. Register-driven switching is free of the "boundary" gain found in lens 3 of `stream_topologies.md`; data-driven switching pays it. |
| A3 | Gating before the streams with top-k. (a) An unrun stream gets zero gradient. (b) Sigmoid scores: an unrun stream's logit gets zero gradient; a run stream's logit gets σ′(s_j)·ε⟨λ, f_j⟩ — an absolute signal that involves no other stream. (c) Softmax over all streams, not renormalised: an unrun stream's logit gets −p_j·ε Σ_{k run} p_k⟨λ, f_k⟩ — non-zero, but it contains nothing about stream j. (d) Softmax renormalised over the run streams with k = 1: the gate is identically 1 and no logit gets any gradient. | The cost of pre-gating, case by case. Explains before the fact why (d) cannot learn. |
| A4 | One-pass gain ‖∂h_{t+1}/∂h_t‖ ≤ 1 + ε Σ_k g_k L_k (gate not reading h; L_k the stream's Lipschitz constant). Softmax gates sum to 1; sigmoid gates sum to as much as K when dense and as much as k when sparse. | With sigmoid scores, sparsity is also what bounds the step. |
| A5 | With sinusoidal position keys and a register advanced by the matching rotation, the register on pass t *is* the key of position t, for any number of passes up to the period of the keys (16 here). | A built-in "read one position further each pass" that does not depend on the trained length. |
| A6 | If streams are odd functions of the state and the state distribution is symmetric, each stream's mean output is zero, so a running-mean stand-in for an unrun stream is zero. | Says in advance that the Default-MoE stand-in carries no signal on this testbed. |
| A7 | Softmax over all streams, not renormalised: the dense and the top-1 output of a pass differ by at most ε(1 − g_top)·max_k‖f_k‖, and in the dense pass stream k's gradient is proportional to g_k. | A sufficient condition for a harmless switch to one stream. It is a one-pass bound, and it turned out *not* to be the reason the dense start works (round H). |

## 3. What the literature does, placed in the family

Full notes with links: `docs/literature.md`, "How published work trains experts inside loops stably".

| Step of §1 | Published practice | Source |
|---|---|---|
| 1. Query / which pass | Nothing: the router reads the token's state and is tied across repeats (MoEUT, Sparse Universal Transformer). Or one router per loop, because a shared one "collapses" to the same selection on every loop (Looping Beyond Twice). | A1 says when each is right |
| 3. What the selector reads | Normalise the selector's input, not the main path (MoEUT's peri-layernorm); keep logits small (ST-MoE's z-loss) | |
| 4. Scores | Sigmoid, top-K, no renormalisation (σ-MoE; adopted by MoEUT and Looping Beyond Twice); K ≥ 2 (ST-MoE) | |
| 4. Collapse / exploration | Usage-entropy or mutual-information bonus on the batch average; expert dropout (σ-MoE: helps; ST-MoE: hurts); noisy top-K (Shazeer) | |
| 4. Signal for unrun experts | Running-mean stand-ins (Default MoE); a dense start annealed to top-1 over the first 7.5–10 % of training (EvoMoE) | |
| 5. Main path | No normalisation on it; residual scale shrinking with loop count; a repeating group of two or more distinct layers | not tested here |

All of these route per token. None has a task where the right expert for each pass is known.

## 4. The rounds

Scoreboard of the predictions written before each run (details in each script's docstring):

| Round | Question | Predictions | Held | Partly | Failed |
|---|---|---|---|---|---|
| A | What tells a tied gate which pass it is in | 9 | 5 | 2 | 2 |
| B | Training a gate that acts before the streams | 12 | 5 | 2 | 5 |
| B2 | Sparse training that is never dense; eight streams | 10 | 4 | 1 | 5 |
| B3 | How short the dense start can be | 5 | 4 | 1 | 0 |
| C | A gate that must read the data | 12 | 10 | 0 | 2 |
| D | More passes than trained | 5 | 5 | 0 | 0 |
| E | The candidate at ten seeds | 7 | 7 | 0 | 0 |
| H | Checks called for by the review (predictions written after reading it) | 8 | 6 | 0 | 2 |
| M | Against one monolithic block: first design (7 scored of 9), revised after the design review (9) | 16 | 11 | 2 | 3 |
| | **Total** | **84** | **57** | **8** | **19** |

### Round A — the progress signal (`arch_a.md`)

Dense softmax gates, four streams, program task. Held-out error, median of five seeds; solved out of five.

| Query built from | Held-out error | Solved |
|---|---|---|
| nothing (order-blind reference) | 3.0e-2 | 0 |
| the data state | 3.1e-2 | 0 |
| one learned vector per pass (clock) | 2.6e-5 | 5 |
| a register that starts still, advanced by an always-on linear map | 2.7e-5 | 5 |
| a register that starts as a random rotation | 3.7e-5 (one seed 3.7e-3) | 5 |
| the same, frozen | 4.9e-5 (one seed 6.0e-3) | 5 |

- **A query read from the data state leaves the gate order-blind** (as the MLP gate was in Entry 102): the data state does not say which pass it is.
- **A register in the state does, and it need not be given a start.** I predicted the still register would learn in three or four seeds of five and worse than the clock (failed prediction): it learned in all five and matched the clock. This reverses the reading of lens 2 in `stream_topologies.md`, where a counter living *inside the gated data state* had an exact solution that training did not find. The difference is that here the counter has its own always-on linear path and the streams do not have to write it.
- In every solved variant three of the four streams are used, and each token has its own stream — except the rotating register with the data read (purity 0.68, and the highest error of the solved variants).
- Letting the gate also read the data costs nothing on held-out programs when it is not needed (2.7e-5, round C's C8). My prediction that it would hurt held numerically for the rotating register (4.4e-3) and failed for the clock (3.1e-5); the cause in the first case was the rotating start (its training error was 2.6e-3: an optimisation failure), not the data read.

Honest limit: this register runs on its own — it never reads the data or the sequence — so it is a clock that the state carries. What it buys over one-router-per-loop is one tied gate, and (round D) the possibility of running more passes than were trained.

### Rounds B, B2, B3 — gating before the streams (`arch_b.md`, `arch_b2.md`, `arch_b3.md`)

Every variant is scored as it would be run: only the top k streams computed. Program task, still register.

| Training of the gate | k at run time | Four streams: solved, median error | Eight streams: solved, median error |
|---|---|---|---|
| Dense throughout (reference) | all | 5/5, 2.7e-5 | 5/5, 2.8e-5 |
| Top-1 from the start, softmax renormalised | 1 | 0/5, 0.36 | |
| Top-1 from the start, softmax | 1 | 0/5, 0.23 | |
| Top-1 from the start, sigmoid | 1 | 0/5, 0.18 | |
| Top-1, sigmoid + expert dropout 0.2 | 1 | 0/5, 0.074 | |
| Top-1, softmax + selection noise | 1 | 0/5, 0.31 | |
| Top-1, softmax + running-mean stand-ins (Default MoE) | 1 | 0/5, 0.26 | |
| Top-1, softmax + linear stand-ins | 1 | 1/5, 0.22 | |
| Top-1 + one explorer (sampled / uniform), then top-1 | 1 | 2/5, 3/5 | 1/5 (sampled) |
| Top-2 from the start, sigmoid (the σ-MoE form) | 2 | 2/5, 1.0e-2 | |
| … + expert dropout + usage-entropy bonus | 2 | 1/5, 1.6e-2 | |
| Top-2 from the start, softmax not renormalised | 2 | 5/5, 4.1e-5 | 2/5, 1.6e-2 |
| Top-2 for the first half, then top-1 | 1 | 5/5, 5.6e-5 | 1/5, 2.8e-2 |
| Dense for 3 % of training, then top-1 | 1 | 1/5, 0.19 | 0/5, 0.17 |
| **Dense for 10 % of training, then top-1** | 1 | **5/5, 2.9e-5** | **5/5, 2.8e-5** |
| Dense for 50 % of training, then top-1 | 1 | 5/5, 2.8e-5 | 5/5, 2.3e-5 |
| **Dense until the gate's mean top probability passes 0.8, then top-1** | 1 | **5/5, 3.0e-5** (dense for 443 of 3,000 steps) | **5/5, 2.2e-5** (813 of 3,000) |

- **Top-1 from the start fails for a gate that reads only the instruction**, far above the order-blind floor (0.2–0.4 against 0.03), since a single stream per pass cannot fall back on the order-blind blend. Two different failures are mixed in these rows, as the review's probes showed. With the still register, the register never learns to address: attention on the right position in 0 % of passes, all tokens on one stream (the low purity in these rows means the gate is not keyed on the current token). With the address given (the candidate's register), the failures put several tokens on one stream: purity 1.0 with one or two streams used, 2/10 solved (round H). A3 says which gradients are missing; I have not shown that their absence is the cause.
- **Top-2 with a softmax works with four streams and not with eight** (2/5). Two of four is half the streams; two of eight is not enough. The prediction that it would degrade held; my reason — streams never selected never train — did not (all five needed streams were in use, with purity 0.6).
- **Sigmoid against softmax is not settled.** At top-2 from the start sigmoid did worse (2/5 against 5/5; five seeds; I had predicted the opposite order). Dense, they are equal (5/5). Under dense-10 %-then-top-1 at eight streams and ten seeds: sigmoid 9/10, softmax 8/10 (round H).
- **None of the published aids rescued top-1 from the start at five seeds**: expert dropout (0/5, though the median error fell from 0.18 to 0.074), selection noise (0/5), an explorer stream (2/5 and 3/5), usage-entropy bonus with dropout at top-2 (1/5 against 2/5). Five seeds cannot tell "no help" from "some help" for dropout and explorers. The running-mean stand-in is untestable here rather than unhelpful (A6).
- **A dense start is what made one-stream execution reliable, and it can be short**: 10 % of training, not 3 %. I had predicted 10 % would be too short (failed). This is EvoMoE's dense-to-sparse gate (2021), which spends 7.5–10 % of training dense on a temperature schedule; I found the paper only after the result.
- **Why — not established.** After B2 I wrote A7 (with a softmax over all streams, the top-1 pass differs from the dense pass by at most ε(1 − g_top)·max‖f‖) and used it to predict B3: run every stream until the gate's mean top probability passes 0.8, then only the top one — 5/5 at both sizes, held. The review pointed out that the fixed 300-step start works although the 0.8 rule had fired in no seed by then, so A7 cannot be the reason; round H tested both candidate explanations and neither held. The confidence rule is therefore a conservative trigger that has worked every time it was tried, not an explanation.
- What a model trained with top-2 loses when run at top-1 without retraining: solved 1/5 (I predicted at least 4/5). The second stream carries real weight, so "train at k, run at fewer" is not free.

### Round C — a gate that must read the data (`arch_c.md`)

Branch task; dense softmax gates unless stated.

| Gate | Held-out error | Solved | Largest gain ‖∂h₄/∂h₀‖ over 96 cases |
|---|---|---|---|
| does not read the data | 0.11 | 0/5 | 4.1 |
| query from the data, no register | 0.031 | 0/5 | 5.3 |
| register + reads the normalised data | 2.1e-3 | 3/5 | 11.2 |
| register + reads the raw data | 1.5e-3 | 5/5 | 11.3 |
| + penalty on large logits | 1.6e-3 | 4/5 | 10.8 |
| clock + reads the normalised data | 1.5e-3 | 5/5 | 11.1 |
| register + data, dense for half of training then top-1 | 1.6e-3 | 3/5 | 7.7 |
| register + data, top-2 from the start | 1.3e-2 | 1/5 | 6.7 |
| (teacher, away from its threshold) | | | 4.9 |

- **Both kinds of state dependence are needed and they are separable**: the register says where the loop is, the data read says which branch. Either alone fails (0.11, 0.031); together the error is 50 times lower. Ten of twelve predictions held; the failures were top-2 from the start (1/5), which round B2 had already shown to be fragile, and the raw read at the higher learning rate, which was better than predicted (3.3 times lower error, outside the stated factor of 3).
- **The data read is where the gain comes from**, as A2 says: the median gain is unchanged (2.06 against 1.97) and the largest, over 96 sampled cases, is 2.7 times higher. The teacher's own map jumps at the threshold, so a faithful student must be steep there — this is the price of the task, not a defect — but it is the term to watch when loops get long. (The lower figures in the two sparse rows are not comparable: the Jacobian of a hard top-1 map does not see the jump where the choice changes.) A learning rate three times higher raised it to 20 without any run diverging.
- **Penalising large logits made no measurable difference; normalisation is unresolved for the still register.** The "raw" row removes normalisation from the register's query as well as from the data read (a confound the review caught), and its 5/5 against 3/5 recurred at three times the learning rate (5/5 against 3/5, same seeds). For the candidate, where only the data read differed, there was no difference at ten seeds (round E).

### Round D — more passes than trained (`arch_d.md`)

Trained on programs of length 3–5, tested at 6 and 8. Dense softmax gates.

| Progress signal, keys | Length 4 | Length 6 | Length 8 | Solved at 6, at 8 |
|---|---|---|---|---|
| one query per pass, learned keys | 3.9e-5 | 4.5e-2 | 7.7e-2 | 0/5, 0/5 |
| register that starts still, learned keys | 4.0e-5 | 9.4e-2 | 0.18 | 0/5, 0/5 |
| register that starts still, sinusoidal keys | 3.6e-5 | 8.3e-2 | 0.21 | 1/5, 0/5 |
| register that starts as a random rotation, sinusoidal keys | 3.4e-5 | 7.0e-2 | 0.12 | 2/5, 0/5 |
| **register that starts as the exact one-position shift, sinusoidal keys** | 3.3e-5 | **3.1e-5** | **3.7e-5** | **5/5, 5/5** |

All five predictions held. A learned advance fits the trained lengths and nothing forces it to be a shift; one router per loop has no router for the sixth loop. Started as the exact shift (A5) and then trained with everything else, the register keeps reading one position further at any length, and **the streams' own error does not compound over twice the trained number of passes** (3.7e-5 at length 8 against 3.3e-5 at length 4).

### Round E — the candidate at ten seeds (`arch_e.md`)

| Task | Candidate: held-out error, solved | Every stream always run | Streams per pass: in training, afterwards | Dense steps of 3,000 |
|---|---|---|---|---|
| Program, 4 streams | 3.0e-5, 10/10 | 3.2e-5, 10/10 | 1.20, 1 | 198 [154, 866] |
| Program, 8 streams, 5 primitives | 1.8e-5, 10/10 | 2.6e-5, 10/10 | 2.32, 1 | 566 [425, 784] |
| Branch, 4 streams | 1.1e-3, 10/10 | 1.4e-3, 10/10 | 1.12, 1 | 123 [90, 223] |
| Program, trained on lengths 3–5 | length 6: 2.7e-5, 10/10; length 8: 3.1e-5, 10/10 | | 1.21, 1 | 212 [134, 565] |
| Branch, trained on lengths 3–5 | length 6: 9.5e-4, 10/10; length 8: 9.6e-4, 10/10 | | 1.12, 1 | 125 [92, 138] |

All seven predictions held. Running one stream per pass, the candidate matches the network that runs all of them, on both tasks, at both sizes, and at twice the trained number of passes. Whether the gate reads the data normalised or raw made no difference at ten seeds (10/10 both; 9.5e-4 and 1.1e-3) (round C's comparison was of a different network and is confounded; see there). On the branch task the dense start turned out to be unnecessary (round H).

### Round H — after the independent review (`arch_h.md`)

The review's own small runs contradicted three of my readings. These reruns at ten seeds, with predictions written after reading the review, are the basis of points 2–4 in §0.

| Top-1 from the first training step (candidate network) | Solved | Median error |
|---|---|---|
| Program, 4 streams, gate reads only the instruction | 2/10 | 0.21 |
| Program, 4 streams, gate also reads the data | 8/10 | 8.8e-5 |
| Program, 8 streams, gate also reads the data | 3/10 | 5.6e-2 |
| Branch, 4 streams | 10/10 | 1.1e-3 |

| With a dense start | Solved | Median error |
|---|---|---|
| Program, 8 streams, gate also reads the data, dense until confident | 10/10 | 1.8e-5 |
| Program, 8 streams, dense 10 %, sigmoid scores | 9/10 | 8.5e-5 |
| Program, 8 streams, dense 10 %, softmax scores | 8/10 | 1.8e-5 |
| Candidate under five other teachers: program; branch | 5/5; 5/5 | 1.9e-5; 1.2e-3 |

What is in place when the dense start ends (still register, four streams, ten seeds per row):

| Dense steps of 3,000 | Address learned | Each token has its own top stream | Mean top probability | Ends solved |
|---|---|---|---|---|
| 90 | 0/10 | 0/10 | 0.42 | 1/10 |
| 150 | 2/10 | 0/10 | 0.52 | 8/10 |
| 200 | 9/10 | 3/10 | 0.64 | 9/10 |
| 300 | 10/10 | 7/10 | 0.77 | 10/10 |

- **Whether sparse-from-the-first-step works depends on what the gate reads.** With the data read it mostly works at four streams and mostly fails at eight; on the branch task it always worked. Why the data read helps is not established (one guess: the choice then varies from input to input, so every stream is run on something from the start).
- **Neither explanation of the dense start survives.** At 150 dense steps no run had a one-to-one assignment, two had the address, the gate's mean top probability was 0.5 — and eight of ten went on to solve the task under top-1. "Ends solved" agreed with "address and assignment in place at the switch" in 22 of 40 runs (I predicted at least 34). Something is established between steps 90 and 150 that neither probe sees.
- **A fixed 10 % is not fully reliable at eight streams** (8/10, 9/10); the confidence trigger was (10/10, and 10/10 in round E).


### Round M — the streamed block against one monolithic looped block (`arch_m.md`)

Added after the first version of this note, at the researcher's request. The design was committed before any run, reviewed independently before running (`review/review_round_m_design.md`: "run after changes"), revised, and committed again. Everything outside the block is shared — register, keys, attention read of the sequence, residual step, optimiser, 3,000 steps, ten paired seeds — and only the block differs:

- **Streams, the candidate**: one stream computed per pass after the dense start.
- **One block, additive**: V tanh(B h + C r + b) — the instruction read r is one more input. Also with two hidden layers.
- **Per-unit gate**: V (sigmoid(C r + b) · tanh(B h)), every unit computed. This is *not* a monolith in I37's sense; it is the streams family with as many width-one streams as units. It separates "gating" from "grouping into streams and running one".
- **One block, no attention**: the whole padded sequence and the register as inputs, three times the parameters.

Parameters are matched to within 3 % unless stated; compute is reported as multiply-adds per pass inside the block. Each family ran at its own best of four learning rates, chosen on three other seeds by training error (`arch_m_sweep.md`). Trained on lengths 3–5; errors below are on held-out programs of length 4.

| Block | Multiply-adds per pass | Held-out error (median) | Error against the candidate's (median of per-seed ratios) | Candidate better in |
|---|---|---|---|---|
| **Program, four streams, three primitives** | | | | |
| Streams, candidate | 176 | 6.5e-5 | — | — |
| Per-unit gate | 580 | 2.8e-5 | 0.8 | 3/10 |
| One block, additive | 580 | 5.8e-3 | 93 | 10/10 |
| One block, two layers | 576 | 4.7e-3 | 79 | 10/10 |
| One block, additive, twice the parameters | 1,160 | 1.8e-3 | 36 | 10/10 |
| Per-unit gate, 18 units (two thirds of the parameters) | 360 | 8.8e-4 | 7.4 | 8/10 |
| One block, no attention, three times the parameters | 2,112 | 3.1e-2 | 527 | 10/10 |
| **Program, eight streams, five primitives** | | | | |
| Streams, candidate | 208 | 6.8e-5 | — | — |
| Per-unit gate | 1,160 | 7.0e-5 | 1.1 | 5/10 |
| One block, additive | 1,160 | 1.1e-2 | 174 | 10/10 |
| One block, two layers | 1,196 | 6.1e-3 | 105 | 10/10 |
| **Branch task, four streams** | | | | |
| Streams, candidate | 608 | 1.1e-3 | — | — |
| One block, two layers | 989 | 8.0e-3 | 7.8 | 10/10 |
| One block, two layers, twice the parameters | 2,016 | 6.3e-3 | 5.8 | 10/10 |
| One block, one layer | 1,000 | 2.1e-2 | 21 | 10/10 |

- **No block has a recombination gap.** Held-out over training error at length 4 is between 0.98 and 1.22 in all sixteen rows. The controller hands the block one instruction per pass, and a held-out program is a new order of the same instructions, so any block that fits the training programs fits the held-out ones. On this testbed I37's null holds by construction: recombination comes from the controller, and the testbed cannot show streams recombining better than a monolith. The block with no attention does not get that far: at three times the parameters it does not fit the *training* programs (2.4e-2).
- **What separates the blocks is multiplicative gating, not streams.** A block whose every unit has its own soft gate matches the candidate (0.8 and 1.1 times its error; within a factor 3 for four of five other teachers). Blocks that take the instruction as an added input are 79–174 times worse at equal parameters, with one hidden layer or two; 36 times at twice the parameters; 41 times after twice the training (three seeds). For all five other teachers the additive block was 57–285 times worse.
- **What grouping into streams and running one of them buys is compute.** The candidate reaches the per-unit-gated block's error with 176 multiply-adds per pass against 580 (four streams) and 208 against 1,160 (eight). Shrinking the per-unit-gated block towards the candidate's cost loses accuracy: at 360 multiply-adds its median error is 7.4 times the candidate's (I predicted at least 10; one seed matched the candidate outright).
- **On the branch task the candidate is 6–8 times better than a two-layer block**, at equal and at twice the parameters, in every seed — and at 0.6 and 0.3 of the multiply-adds.
- **Outside the trained range there is no consistent story.** More passes: every block's error at length 8 is 0.9–1.6 times its error at length 4 in the median (2.2 for the one-layer branch block); my first prediction that the additive block would degrade failed. In one to three seeds of nine of the fifteen blocks that read the sequence by attention — one seed each for the candidate and the per-unit gate, three for the two-layer block — the error at length 8 jumps by 12 to 3,000 times; a probe of the worst seed of each shows the trained register losing the address at passes 6–8, so this is the shared controller, in every kind of block. Inputs twice as large: on the program task the candidate's error rises least (2 times, against 3.5–4.3 for the per-unit gate and 4–14 for the additive and two-layer blocks), but on the branch task it rises most (13 times, against 3.6–4.6), which shrinks its lead over the two-layer blocks to 2–3 times, and no block is below 10⁻² on the branch task with doubled inputs. My revised prediction that no block would stand out failed.
- **Caveats.** The three-seed selection put the candidate at a learning rate of 0.01; over the ten seeds that gave 6.5e-5, about twice its error at 0.003 in round E (2.7e-5), and its error wanders between 5e-6 and 2e-4 late in training. The ordering does not depend on it (the per-unit gate is at 2.8e-5). Nothing has converged at 3,000 steps. The additive and two-layer blocks are the plainest monoliths; a sceptic's stronger monolith with multiplicative conditioning is the per-unit-gated block, which is already the streams family.

Predictions. First design (nine, seven scored, two rows dropped as uninformative): four held, two partly, one failed. Revised after the design review (nine; four of them had been previewed by the reviewer's one-seed probes): seven held, two failed.


## 5. The candidate

```
state      data h (dimension d);  control register c (small)
sequence   keys k_i = sinusoidal code of position i (fixed);  values v_i = E[token_i]

each pass
  q   = beta * c / rms(c)                    query from the register
  a   = softmax_i(q . k_i);  r = sum_i a_i v_i     what the sequence says about this pass
  s   = G r  +  MLP([r, h])                  K selection logits (the MLP term when choices depend on the data)
  p   = softmax(s)                           over ALL streams; never renormalised
  run = every stream      while training and the gate is undecided (mean top probability <= 0.8)
        argmax p only     afterwards, and always at run time
  h  <- h + eps * sum_{k in run} p_k f_k(h)  no normalisation on this path
  c  <- c + M c                              always on;  M starts as the one-position shift and then trains
```

| Part | What it is for | Evidence | What it was chosen over |
|---|---|---|---|
| Register in the state, advanced by an always-on linear map | Tells a tied gate which pass it is in (A1) | Round A: 5/5, equal to one query per pass | A query from the data state (order-blind, 0/5); one router per loop (cannot run a sixth loop, round D) |
| Query attends over the original sequence | Addresses the instruction for this pass; shared across positions | Rounds A, D; Entry 102 (an MLP over the whole program did not generalise even when told the pass) | Feeding the gate the whole program |
| Register started as the exact shift on sinusoidal keys | More passes than trained | Round D: 5/5 at lengths 6 and 8; learned advances 0–2/5 | A register that learns its own advance |
| Gate also reads the data state | Choices that depend on the data | Round C: 0.11 → 1.5e-3; costs nothing when not needed | — |
| Scores over all streams, never renormalised | Renormalising over the streams that ran leaves a top-1 gate without gradient (A3d) | Round B: renormalised 0/5. Softmax against sigmoid is not settled (round H: 8/10 against 9/10, softmax with the lower error); softmax kept | Renormalising |
| Every stream until the gate's mean top probability passes 0.8, then top-1 | Makes one-stream execution reliable; mechanism not identified | 10/10 in seven ten-seed cells, equal to dense; 4–19 % of training steps dense (medians) | Top-1 from the first step (2/10 instruction-only; 8/10 and 3/10 with the data read at four and eight streams; 10/10 on the branch task); top-2 from the start; a fixed 10 % (8/10, 9/10 at eight streams) |
| No normalisation on the data path | Nothing depended on it here | Round D: error flat to eight passes | (not compared) |

Against the researcher's two requirements. *The gate depends on the original sequence and the state*: yes — the register (state) addresses the sequence, and the data state enters the logits. *The gate acts before the streams*: at run time always; in training after the first 4–19 % of steps (medians over seeds; 29 % in the slowest seed), so the training cost is 1.1–2.3 streams per pass instead of 4 or 8. On the branch task the gate could act before the streams from the first step (10/10, round H).

What is and is not new in it. Nothing is new piece by piece, and the gate is close to an existing design. Stack-NMN (Hu et al. 2018) has a controller that attends over the instruction's words at every step and emits softmax weights over modules; its weights become nearly one-hot without supervision and can be replaced by the arg-max at test time for under four points. One module per step for the whole state is also Routing Networks and Modular Networks. Dense-then-sparse is EvoMoE, which also lets the gate's own scores decide which experts run. An address advanced by a fixed shift is location addressing from the Neural Turing Machine. What differs here: the streams are learned rather than designed; the controller is tied and takes its progress signal from a register instead of a per-step transform, which is what lets it run extra passes; only the chosen stream is computed for most of training; and the right stream for each pass is known, so the choices can be checked. (The review named Stack-NMN and MAC; I read both afterwards — `literature.md`.)

## 6. Where the published advice held on this testbed, and where it did not

| Advice | Source | Here |
|---|---|---|
| Sigmoid scores, no renormalisation | σ-MoE, MoEUT, Looping Beyond Twice | Not renormalising held (a renormalised top-1 gate cannot learn). Sigmoid against softmax: not settled — worse at top-2 from the start (2/5 against 5/5, five seeds), equal when dense, 9/10 against 8/10 under dense-then-top-1. |
| Top-2 rather than top-1 | ST-MoE | Held with four streams (5/5 against 0/5); not enough with eight (2/5). |
| Expert dropout, usage-entropy bonus | σ-MoE | No rescue at five seeds (1/5 against 2/5 at top-2). ST-MoE lists both among things that hurt pre-training or did not help fine-tuning. |
| Noisy top-k | Shazeer et al. | No rescue at five seeds (0/5). |
| Running-mean stand-ins for unrun experts | Default MoE | Untestable here rather than unhelpful: these streams' mean output is near zero (A6). 0/5. |
| Dense first, then sparse | EvoMoE | **Held.** A fixed 10 %: 5/5 twice with the still register, 8/10 and 9/10 at eight streams with the candidate's; 3 % was too little; the confidence trigger 10/10 everywhere. |
| One router per loop, because a tied router repeats itself | Looping Beyond Twice | A tied gate fed a register matched it. The register is a clock, so this is a per-loop signal by another name; the point is only that the signal can live in the state while the gate stays tied (A1), which also allows extra passes. It says nothing about per-token routers. |
| A tied router given no pass signal | MoEUT, Sparse Universal Transformer (they give none; they do not claim none is needed) | Here a query from the data state alone left the gate order-blind (0/5). Their routers read token states that change with depth; whether those carry "which pass" is the question A1 poses. |
| Normalise what the selector reads; penalise large logits | MoEUT; ST-MoE | The penalty: no measurable effect. Normalisation: no effect for the candidate; unresolved (confounded) for the still register. Untested where the published reasons apply. |
| Residual scale shrinking with loop count; groups of distinct layers; halting | Looping Beyond Twice; MoEUT; Sparse Universal Transformer | Not tested. |

## 7. Limits

- **The testbed is far from a transformer.** The state is one vector, not a sequence of token states with attention between them; the gate makes one choice per pass; streams are one-hidden-layer maps in six dimensions; at most eight streams and eight passes; 3,000 training steps. A variant that failed here may only be slow.
- **The register is a clock.** It runs on its own and never reads the data or the sequence. Control flow that depends on what happened — a jump, a repeat, stopping early — would need the register's update to read them, and that is untested. Halting is untested.
- **The sequence names the operations.** Each token *is* the instruction for a pass. In a real task the instruction has to be inferred from the input.
- **The score function is not settled**, and neither is the reason a dense start works (round H).
- **Five seeds per cell in rounds A–D.** A negative claim ("did not help") at five seeds is weak; the comparisons the candidate rests on were rerun at ten in rounds E and H.
- **One teacher.** Every round uses the same random teacher; five others were tried only for the candidate (round H: 5/5 on each task).
- **The comparison with a monolithic block (round M) is on this testbed only, and its baselines are simple.** Each monolith got its own learning rate and nothing else; none converged in 3,000 steps; the eight-stream and width rows inherit their family's rate. The comparison I37 asks for on a carrier with known rules has not been made, and round M says what to expect of it: a gap between additively conditioned and gated blocks, not between gated blocks and streams.
