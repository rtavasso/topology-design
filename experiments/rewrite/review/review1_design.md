# Rewrite carrier design review 1 (revision 2 of `rewrite/DESIGN.md`, 6 Nov 2026)

**Reviewer:** independent, adversarial. Read AGENTS.md (hard rules; L27), LEDGER Direction (29 Oct, 1 Nov updates; risks 1–9; I36) and lessons L27–L33, history Entries 73–75, `rewrite/DESIGN.md` revision 2, and for form `route/DESIGN.md` §§2–6 and `commit/review/review1_design.md`. Nothing in the repo was edited except this file.

**What was computed.** A small implementation of the carrier as §1 specifies it, in the scratch folder `/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/review_rewrite/` (`carrier.py` + 8 derivation scripts; each run < 10 min on 2 CPU cores; outputs in `*.txt`, `*.log`, `*.json` beside them). Choices the design leaves open, fixed as follows and stated where they matter: terms sampled with size uniform on 1…5 (1…7 for evaluation), shape uniform over bracketings, leaves i.i.d. over {a, b, c, d, 0}; I← one-sided (x → x⊕0, as written; see S9); the random-walk generator picks uniformly among valid state-changing (rule, position) actions; "candidates" at a decision = distinct valid successors (no-ops excluded); σ = Zhang–Shasha tree-edit distance with unit costs on the ordered tree with node labels (`+`, atom) — **TED** — with string edit distance on the prefix tokens (SED) and a leaf-bag distance (bag) as sensitivity proxies; D* exactly as §1 defines it (smallest d at which the argmin of "min σ over continuations of ≤ d − 1 further steps" contains only correct actions); canonical proof = lexicographically first by (rule index in the order A→, A←, C, I→, I←, D→, D←; then position); lookahead-d reference policies break σ-ties by steps-to-the-minimum and then lexicographically. All distances are exact BFS; all (B*, D*) values are exact within the radius-k ball, which suffices (a correct successor's distance k − 1 and every lookahead ≤ remaining distance stay inside it).

---

## Verdict: **revise first.** Not ready for implementation as written; the carrier is worth keeping.

The carrier does what the design hopes in one respect: under a syntactic proxy, garden paths are common and deep (B5 below has the numbers), so the per-decision strata can be populated. But as specified, the check P1 is empty and the three core predictions are foregone (P2, P3) or half-derivable on a near-empty stratum (P4), for reasons that have nothing to do with the hypotheses they are meant to test; the arithmetic of the generator was not done before the predictions were written (L22, L25, L27):

1. **The generator cannot populate the length axis.** Random walk + rejection accepts 0–3 % of walks at k = 5 and ≤ 0.8 % at k = 6 in every training world, and 0 of 800 at k ≥ 5 in {A}, {C}, {D}; on the design's own Zipf(1.5) proposal the *realized* length distribution is 71–83 % k = 1 and ≤ 2 % k ≥ 4. Lengths 7–10 "within training worlds" (Q4, P1) do not exist at size ≤ 5 in six of the eight training worlds (diameters ≤ 6), k = 7 exists rarely in {A, I}, k = 7–8 in {A, C}, and 9–10 in none.
2. **The world almost never binds.** In 96–100 % of generated instances the shortest proof under W is also a shortest proof under the full set {A, C, I, D} and stays inside W; a cue-blind, world-free planner plus a mask is therefore the closure of the training worlds on the held-out worlds, and **a hand-written greedy-TED + mask policy already scores 0.925 on {A, D} and 0.985 on {C, I}** on the design's generator — above P2's ≥ 0.9 bar. P2 as stated is derivable in the direction predicted; P8's null ("at the single-world baseline") is derivable to come out near RT instead.
3. **P3's direction is fixed by the output head and the position encoding**, not by rules-vs-tables: a categorical (rule, position) head over ≤ 13 positions has rows 9–12 never labelled at size ≤ 5, and learned absolute positions past the training length are untrained; 3–40 % of size-6–7 instances need those rows. "Failures at unseen positions" is the signature of an untrained softmax row.

Plus a definitional problem in (B*, D*) that would be regretted at analysis time (B5), and no reachability/expressivity ceiling for the held-out worlds (B6). None of the fixes diverts from the rung-4 → rung-5 sequence; they change the generator, the strata, two arms and the predictions' conditioning.

---

## 0. The numbers

### 0.1 Reachable graphs (size-5 starts at cap 5; size-7 starts at cap 7; 4 starts per cell; `graph_sizes_n4.txt`)

| world | cap 5: nodes median [min, max] | diameter from t | cap 7: nodes | diameter | s per BFS (cap 7, Python) |
|---|---|---|---|---|---|
| A | 14 | 3–4 | 132 | 5–7 | 0.00 |
| C | 16 [8, 16] | 3–4 | 48 [32, 64] | 5–6 | 0.00 |
| I | 5 [1, 9] | 0–2 | 13 [1, 76] | 0–4 | 0.00 |
| D | 3 [1, 14] | 0–3 | 4 [1, 7] | 0–2 | 0.00 |
| AC | 630 [70, 840] | 5–8 | 41,580 [27,720, 166,320] | 12–14 | 6.8 |
| AI | 61 [14, 162] | 4–7 | 900 [132, 3,038] | 6–9 | 0.10 |
| AD | 61 [14, 115] | 3–8 | 834 [132, 4,837] | 6–15 | 0.14 |
| CI | 120 [60, 120] | 4–6 | 736 [32, 736] | 5–8 | 0.05 |
| CD | 16 | 4 | 256 [64, 480] | 6–8 | 0.02 |
| DI | 7 [1, 309] | 0–6 | 417 [18, 2,133] | 2–8 | 0.08 |
| ACI | 420 [280, 840] | 6–8 | 62,330 [28,980, 173,880] | 10–13 | 13 |
| ACD | 2,292 [504, 3,480] | 11–14 | > 300,000 (3 of 4 capped) | ≥ 12 | 35 |
| ADI | 124 [117, 382] | 6–9 | 6,001 [132, 26,301] | 7–13 | 1.4 |
| CDI | 350 [152, 932] | 6–10 | 496 [64, 5,696] | 6–10 | 0.2 |
| ACDI | 4,284 [2,796, 5,772] | 11–13 | > 300,000 (3 of 4 capped) | ≥ 9 | 41 |

The DP posterior is feasible at cap 5 (full world: ≤ 6k nodes; BFS 0.3–0.5 s in Python, the forward–backward over T = 14 steps 1–3 ms with a scipy sparse matrix; `misc_checks.txt` (3)). At cap 7 the full reachable graph of any world containing A and C is 30k–300k+ nodes (7–40 s per BFS); evaluation needs only the radius-k ball (≤ 1.6k nodes at k ≤ 4 for every held-out pair; `positions_cap7.txt`), so exact strata at evaluation are fine, but no exact-posterior arm can run at size 6–7 in those worlds.

### 0.2 The generator as specified: accepted / random walks tried (random walk from a size-1…5 term, cap 5, accept iff BFS distance = k; `run_cap5_*.log`, `bd_cap5_t1-5_k6.json`)

| world | k=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9–10 |
|---|---|---|---|---|---|---|---|---|---|
| A | 20/36 | 20/75 | 20/353 | 4/800 | 0/800 | 0/800 | – | – | – |
| C | 20/26 | 20/55 | 20/176 | 10/800 | 0/800 | 0/800 | – | – | – |
| I | 20/22 | 20/30 | 20/60 | 20/186 | 8/800 | 2/800 | – | – | – |
| D | 20/21 | 20/51 | 20/112 | 11/800 | 0/800 | 0/800 | – | – | – |
| AC | 40/53 | 40/88 | 40/163 | 40/434 | 40/1,348 | 12/1,600 | 1/1,600 | 1/1,600 | 0/3,200 |
| AI | 40/40 | 40/56 | 40/104 | 40/373 | 27/1,600 | 1/1,600 | 0/1,600 | 0/1,600 | 0/3,200 |
| AD | 40/40 | 40/54 | 40/107 | 40/296 | 40/1,449 | 9/1,600 | 2/1,600 | 1/1,600 | 0/3,200 |
| CI | 40/40 | 40/53 | 40/70 | 40/135 | 40/457 | 40/1,428 | 8/1,600 | 0/1,600 | 0/3,200 |
| CD | 40/40 | 40/47 | 40/105 | 40/470 | 9/1,600 | 0/1,600 | 0/1,600 | 0/1,600 | 0/3,200 |
| DI | 40/42 | 40/44 | 40/125 | 40/256 | 18/1,600 | 2/1,600 | 0/1,600 | 0/1,600 | 0/3,200 |
| ACI | 40/40 | 40/49 | 40/87 | 40/142 | 40/570 | 37/1,600 | 7/1,600 | 0/1,600 | 0/3,200 |
| ACD | 40/40 | 40/52 | 40/97 | 40/165 | 40/398 | 40/1,110 | 13/1,600 | 3/1,600 | (not run) |
| ADI | 40/40 | 40/57 | 40/93 | 40/223 | 40/909 | 8/1,600 | 0/1,600 | 0/1,600 | 0/3,200 |
| CDI | 40/40 | 40/49 | 40/75 | 40/137 | 40/408 | 28/1,600 | 1/1,600 | 0/1,600 | 0/1,600 |
| ACDI | 40/40 | 40/47 | 40/74 | 40/137 | 40/493 | 29/1,015 | (not run) | | |

Realized lengths when k is proposed from Zipf(1.5) on 1…6 and the instance is kept iff d = k (200 accepted per world; `greedy_zipf.txt`): {A, D} 149 / 40 / 9 / 2 / 0 / 0; {C, I} 142 / 39 / 10 / 7 / 1 / 1; {A, C} 166 / 24 / 7 / 2 / 1 / 0; {A, I} 153 / 36 / 10 / 1 / 0 / 0; {A, C, I, D} 144 / 41 / 9 / 4 / 2 / 0. Zipf(1.5) would be 109 / 39 / 21 / 14 / 10 / 7.

Where length 7–10 exists at all for size-≤ 5 endpoints (layer counts under cap 7, 4 starts; `misc_checks.txt` (2)): {A, C} has 131–425 states at k = 7 and 13–125 at k = 8 from 3 of 4 starts, 0–5 at k = 9, 0 at k = 10; {A, I} and {A, D} have k = 7–10 states only of size 6–7 (0–2 of size ≤ 5); {C, D} none. The singleton worlds have diameter ≤ 4 (I: ≤ 6 from small starts).

### 0.3 Per-decision difficulty under TED — the design's generator (first decision; `run_cap5_*.log`)

Share of accepted instances with D* = 1 at the first decision, under TED / SED / bag; B* = 1 under TED; and the share whose canonical proof has max D* ≥ 2 and ≥ 3 (TED):

| world | k=3: D*=1 (TED/SED/bag), B*=1, maxD*≥2, ≥3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|
| AC | .70/.73/.23, .30, .80, .25 | .43/.45/.00, .58, .80, .65 | .48/.53/.08, .53, 1.0, .88 | .42/.50/.08, .58, .92, .92 (n=12) |
| AI | .78/.78/.45, .65, .28, .20 | .85/.78/.68, .83, .33, .23 | .33/.37/.00, .52, .93, .81 (n=27) | – (n=1) |
| AD | .63/.50/.48, .83, .68, .35 | .70/.50/.53, .75, .78, .55 | .53/.35/.28, .65, .90, .80 | .56/.22/.22, .89, 1.0, .78 (n=9) |
| CI | .88/.65/.25, .70, .40, .08 | .73/.63/.28, .65, .55, .33 | .85/.73/.53, .88, .50, .28 | .60/.58/.35, .80, .70, .60 |
| CD | .83/.78/.45, .60, .28, .03 | .65/.60/.40, .53, .50, .28 | .67/.67/.56, .67, .56, .44 (n=9) | – |
| DI | .63/.43/.50, .40, .53, .25 | .63/.60/.55, .38, .75, .65 | .28/.22/.50, .67, 1.0, .94 (n=18) | – |
| ACDI | .48/.50/.23, .50, .70, .38 | .40/.30/.25, .45, .80, .65 | .40/.40/.25, .45, .90, .85 | .41/.31/.24, .45, 1.0, .93 (n=29) |

Singletons: {A} D* = 1 in 1.0 / 1.0 / 0.70 / 1.0 at k = 1…4 (TED); {C} 1.0 / 1.0 / 0.95 / 1.0; {I} 1.0 / 1.0 / 0.85 / 0.85; {D} 1.0 / 0.70 / 0.75 / 0.91. Tie multiplicity (number of shortest proofs as state sequences, median): 1–3 at k ≤ 3; 8–24 at k = 4; 11–35 at k = 5; 45–156 at k = 6; maxima 572 (ACDI k = 6), 852 (CDI k = 7). The share of instances whose correct first action is unique: 0.0–0.5 at k ≥ 4.

### 0.4 Per-decision difficulty — layer-sampled instances (s uniform over states at W-distance k; 24–30 per cell; `pol_pairs.log`, `pol_triples.log`, `pol_cap5_*.json`)

At the first decision under TED: **strict garden path** (min σ over correct actions > min σ over incorrect: the right move looks strictly worse), **σ-tie** (equal, with a wrong action in the argmin), and the share of instances whose canonical proof has D* = 1 at every decision; then the success of the symbolic reference ladder — greedy-TED (d = 1), lookahead-2, lookahead-3 (lex tie-break; random tie-break in brackets) with T = 2k + 2:

| world | k | strict GP | σ-tie | all D* = 1 | d = 1 | d = 2 | d = 3 |
|---|---|---|---|---|---|---|---|
| AC | 3 | .27 | .13 | .33 | .80 (.80) | .93 (.82) | 1.0 |
| AC | 4 | .29 | .36 | .07 | .07 (.18) | .21 (.32) | .57 |
| AC | 5 | .33 | .25 | .00 | .08 (.15) | .08 | .00 (.08) |
| AC | 6 | .40 | .40 | .00 | .00 | .00 (.08) | .10 (.08) |
| AI | 4 | .14 | .07 | .39 | .64 | .79 | .93 |
| AI | 5 | .28 | .44 | .00 | .39 (.29) | .22 (.28) | .61 (.75) |
| AD | 4 | .21 | .00 | .25 | .39 (.44) | .57 (.61) | .86 |
| AD | 5 | .39 | .11 | .00 | .00 (.03) | .00 (.03) | .33 (.38) |
| AD | 6 | .50 | .14 | .00 | .00 | .14 (.20) | .43 (.63) |
| CI | 4 | .00 | .20 | .47 | .73 (.83) | .87 (.85) | .93 (.95) |
| CI | 5 | .18 | .18 | .39 | .39 (.54) | .79 (.72) | .89 (.91) |
| CI | 6 | .30 | .09 | .26 | .22 (.21) | .43 | .65 (.55) |
| DI | 4 | .29 | .33 | .04 | .33 (.40) | .54 (.58) | .75 (.73) |
| ACDI | 3 | .21 | .38 | .17 | .29 (.30) | .71 (.65) | 1.0 |
| ACDI | 4 | .42 | .33 | .08 | .04 (.09) | .38 (.41) | .67 (.70) |
| ACDI | 5 | .46 | .25 | .04 | .08 (.09) | .21 | .50 (.46) |
| ACDI | 6 | .38 | .25 | .04 | .00 (.04) | .04 (.07) | .25 (.33) |

Under layer sampling the modal D* at the first decision for k ≥ 4 is **D* = k** (e.g. AC k = 6: 7 of 10; ACDI k = 5: 13 of 24): the argmin set keeps a wrong action until σ = 0 separates (see B5).

### 0.5 The same ladder on the design's generator (random walk, Zipf proposal, cap 5, T = 14, 200 instances per world; `greedy_zipf.txt`)

| world | greedy-TED + mask (d = 1) | lookahead-2 | lookahead-3 | d = 1 by k (1 / 2 / 3 / 4) |
|---|---|---|---|---|
| **{A, D}** (held out) | **0.925** | 0.975 | 1.000 | 1.0 / .80 / .33 / .50 |
| **{C, I}** (held out) | **0.985** | 0.995 | 0.995 | 1.0 / 1.0 / .90 / 1.0 |
| {A, C} | 0.955 | 0.990 | 1.000 | 1.0 / .83 / .57 / .50 |
| {A, I} | 0.995 | 1.000 | 1.000 | 1.0 / 1.0 / .90 / 1.0 |
| **{A, C, I, D}** (held out) | **0.940** | 0.975 | 0.990 | 1.0 / .88 / .78 / .25 |

### 0.6 Does the world bind? (`cue_needed.txt`, `binding.txt`)

On the design's generator (80 accepted instances per world, Zipf proposal): d_full(t, s) = d_W(t, s), with some shortest full-world proof inside W, in **1.00 / 1.00 / 0.96 / 0.99** of instances for {A} / {C} / {I} / {D}, **0.99 / 0.97 / 1.00 / 1.00 / 1.00 / 0.97** for AC / AI / AD / CI / CD / DI, **1.00 / 1.00 / 1.00 / 0.99** for ACI / ACD / ADI / CDI.

Where binding states exist at all (layer sampling, all states at W-distance k from 20 size-3…5 starts): share with d_full < d_W ("bind") and, among the rest, the share of full-world shortest proofs (action sequences) that stay inside W ("inW", what a cue-blind router choosing uniformly among full-world shortest proofs would score):

| world | k=2 bind / inW | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|
| AC | .00 / .95 | .03 / .88 | .09 / .78 | .18 / .72 | .22 / .58 |
| AI | .01 / .87 | .07 / .71 | .26 / .45 | .66 / .19 | .94 / .09 |
| AD | .01 / .87 | .09 / .82 | .24 / .74 | .49 / .71 | .68 / .67 |
| CI | .03 / .88 | .10 / .77 | .24 / .66 | .48 / .55 | .79 / .45 |
| CD | .02 / .87 | .08 / .80 | .11 / .74 | .00 / .65 | – |
| DI | .05 / .97 | .17 / .85 | .25 / .85 | .20 / .76 | .00 / .98 |
| I | .11 / .79 | .58 / .58 | .59 / .30 | .97 / .30 | 1.0 / – |
| ACI | .00 / .93 | .00 / .91 | .00 / .90 | .00 / .90 | .00 / .90 |
| ACD | .00 / .96 | .03 / .93 | .07 / .91 | .11 / .88 | .18 / .86 |
| ADI | .01 / .96 | .04 / .86 | .10 / .81 | .17 / .77 | .23 / .73 |
| CDI | .02 / .96 | .06 / .89 | .17 / .79 | .31 / .71 | .45 / .63 |

(The triples bind less than the pairs: with three axioms in force only one is missing, and D adds no shortcuts to A, C, I at all — ACI binds in 0 % of states at every k.)

### 0.7 Cross-axiom arbitration in the held-out worlds (layer-sampled; `cross_axiom.txt`)

At the first decision: share where the greedy-TED pick is wrong **and** its axiom differs from the axiom of every correct action ("cross-axiom garden path": the only first decisions whose outcome cannot come from per-axiom progress plus a mask), and the share where a wrong pick shares an axiom with a correct action:

| world | k=3 cross / same | k=4 | k=5 | k=6 | canonical proof interleaves axioms (k=4 / 6) |
|---|---|---|---|---|---|
| {A, D} | .07 / .17 | .08 / .19 | .12 / .38 | .07 / .29 | .50 / .86 |
| {C, I} | .00 / .10 | .04 / .19 | .04 / .04 | .00 / .37 | .56 / 1.0 |
| ACI | .13 / .00 | .00 / .17 | .11 / .18 | .00 / .18 | .40 / 1.0 |
| ACD | .10 / .23 | .17 / .28 | .14 / .32 | .23 / .45 | .59 / .82 |
| ADI | .20 / .20 | .37 / .17 | .52 / .24 | .35 / .27 | .67 / .92 |
| CDI | .17 / .13 | .20 / .23 | .27 / .10 | .10 / .31 | .50 / 1.0 |
| ACDI | .33 / .10 | .14 / .21 | .30 / .19 | .30 / .19 | .38 / .85 |

Training pairs for comparison: AC .09 / .45 / .20 / .00 at k = 3…6; AI .03 / .00 / .13 / .25; CD .03 / .00 / .20 / .00; DI .26 / .16 / .00 / .00.

### 0.8 Unseen positions at evaluation sizes (t of size 6–7, cap 7; 30 instances per cell; `positions_cap7.txt`)

Share of instances whose canonical proof uses a prefix position ≥ 9 (never a valid target at size ≤ 5), and the share of all canonical actions at such positions:

| world | k=1 | k=2 | k=3 | k=4 |
|---|---|---|---|---|
| A | .00 / .00 | .00 / .00 | .00 / .00 | .00 / .00 |
| C | .03 / .03 | .07 / .03 | .10 / .03 | .27 / .07 |
| AC | .00 / .00 | .03 / .02 | .10 / .03 | .03 / .01 |
| AD | .03 / .03 | .03 / .02 | .37 / .12 | .20 / .05 |
| CI | .20 / .20 | .17 / .08 | .30 / .11 | .40 / .10 |
| AI | .03 / .03 | .10 / .05 | .17 / .06 | .03 / .01 |

(The lex-first canonical proof prefers low positions, so these are lower bounds on what other shortest proofs need; "every correct first action at a position ≥ 9" is 0–20 %.)

### 0.9 Q7 (`misc_checks.txt` (1)): two example proofs (k from Zipf, the design's generator) use every axiom of W in 0.64 / 0.44 / 0.57 / 0.62 / 0.58 / 0.65 of episodes for AC / AI / AD / CI / CD / DI, 0.13 / 0.21 / 0.19 / 0.21 for the triples, 0.09 for {A, C, I, D}.

---

## Blocking

### B1. The generator cannot populate the strata it promises; the length axis is nearly empty at size ≤ 5 (§1 Instances, §3, Q4, P1)

- Random walk + rejection (§1) accepts ≤ 3 % of walks at k = 5 in 9 of 15 worlds and ≤ 0.8 % at k = 6 in 10 of 15, **0 of 800 at k ≥ 5 in {A}, {C}, {D}** and 0 of 1,600 at k = 6 in {C, D} (§0.2). k ~ Zipf(1.5) as written either loops forever on an impossible (W, k) or, if k is resampled, yields the realized distribution of §0.2: 71–83 % k = 1, ≤ 2 % k ≥ 4. Every "by k" or "by D*" stratum at k ≥ 4 is then a handful of instances per evaluation set, and the training stream teaches almost nothing beyond one-step rewrites.
- Lengths 7–10 "within training worlds" (Q4 / P1) do not exist: diameters at cap 5 are ≤ 4 for {A}, {C} and ≤ 5–6 (reached from a few percent of starts) for {D}, {I}, {C, D}, {D, I}, {C, I}; k = 7–8 exist in {A, C} (10–25 % of that world's states at k = 7 from most starts, but 1 in 1,600 random walks reaches them) and k = 7 in {A, I} / {A, D} at 0–2 per 1,600; **k = 9–10 exists in no training world** at size ≤ 5 (§0.2). P1 is therefore not a check but an empty cell, and Q4's "F30" framing is wrong for this carrier anyway: F30's closure was exact because the product state recurs; here a longer proof is a new (t, s) relation, so depth generalization is an extrapolation, not a closure.
- The design's own generator and a layer sampler produce different instance distributions: D* = 1 at the first decision is 0.43–0.85 (random walk) against 0.07–0.47 "all decisions D* = 1" (layer sampling) at k = 4 in the same worlds (§0.3–0.4). Random walks backtrack, so accepted instances are the easy ones. Whichever is chosen must be fixed in the derive step and not changed after data.

**Fix.** Generate s by **layer sampling**: BFS from t (cheap at cap 5), sample k from the intended distribution *restricted to the layers that exist*, and draw s uniformly from the layer; record the realized (world, k) table in the derive step and in every results table. Restate Q4 as lengths 7–8 in {A, C} (and, if the training cap is raised to 7, lengths 7–10 in AC / AI / AD at matched size, which then confounds with Q2 and must be reported as a separate stratum). Oversample k ≥ 3 and the garden-path strata explicitly (the design says "the generator oversamples them" — write the weights down). Batches "mix worlds uniformly" although the worlds' available k differ by 4×; say what is uniform.

### B2. The world almost never binds, so Q1 is a mask, P2 is derivable, and P8's null is derivable to be wrong (§1 Worlds, Q1, P2, P8, §2)

- On the generator as specified, **96–100 % of instances have d_full(t, s) = d_W(t, s) with a shortest full-world proof inside W** (§0.6). The axiom tokens then carry no planning information: a router that ignores them and plans under all four axioms produces a valid shortest proof in ≈ 0.9–1.0 of instances (the "inW" column: 0.87–1.0 at k ≤ 2, which is ≥ 90 % of the stream). Recombination "with rules it never saw together" reduces to the closure of a world-free (t, s) → action map under a per-action mask, which is exactly the L27 case of a derivable core: if the router learns the map on the training pairs and reads the mask, the held-out worlds follow; if it fails, the failure is in the map's generalization to new (t, s) distributions, not in composing axioms.
- The Lean analogy the design rests on (§0) holds only where the world binds: a missing lemma lengthens the proof. Here that is the stratum d_W > d_full, which is 0–4 % of generated instances and exists mainly at k ≥ 4 in the pair worlds (bind 0.24–0.94 at k = 4–6 in AI / AD / CI; §0.6), i.e. exactly where the generator does not go.
- P8 ("RT-shuffled-axioms at the single-world baseline"): the baseline is undefined, and the derived expectation for a cue-blind router is ≈ 0.9+ on every world (non-binding share × inW), not a single-world level. As written the null would likely sit within a few points of RT and be read as "the axioms are read" or as a bug, either way wrongly.

**Fix.** (1) Make the **binding stratum** (d_W(t, s) > d_full(t, s), k ≥ 3) the core of Q1 at a preregistered share (e.g. ≥ 50 % of the held-out evaluation sets and a stated share of training), generated by layer sampling with a rejection on the binding test (one extra BFS under the full set per instance; ≤ 0.5 s). In that stratum the router must plan a detour around a missing axiom it has never seen missing in that combination, which no world-free map supplies. (2) Derive and preregister the cue-blind ceiling per world (non-binding × inW) and report P8's null against it. (3) Stratify P2 by binding and by the cross-axiom garden paths of §0.7. (4) Reconsider the researcher's §12.2 default in this light: worlds stated in context are the core only if the stated world changes the proof.

### B3. P2's bar is met by a hand-written non-composing policy on the design's own generator (P2, §4 planted rows, AGENTS rule 4 "random-routing control")

Greedy-TED + mask (choose the valid in-world action whose result is closest to s in tree-edit distance; lexicographic tie-break) succeeds in **0.925 on {A, D}, 0.985 on {C, I}, 0.940 on {A, C, I, D}** with T = 14 on the generator as specified (§0.5); lookahead-2 is at 0.975–0.995. P2 ("≥ 0.9 success at in-range length and size in ≥ 6/10 seeds") is thus reachable with zero recombination, because the stream is 71–83 % one-step instances (B1) and the world rarely binds (B2). This is L25 again: the prediction is already true for a policy that does not have the property being tested.

**Fix.** Add the symbolic ladder (greedy-TED, lookahead-2, lookahead-3, oracle) as planted reference policies evaluated on every stratum; preregister P2 on the strata where the ladder's d = 1 rung fails (binding instances; strict garden paths; k ≥ 3) with the rung's own success as the floor; report the Zipf-weighted mean only as a sanity number. The ladder is also P4's reference (B5).

### B4. P3 is foregone in the predicted direction by the output head and the position encoding, and its "table signature" is an artefact (§1 Router, Q2, P3, §4 planted table router)

- The head "over (rule, position) + HALT" with ≤ 13 positions: at size ≤ 5 a term has ≤ 9 nodes, so **rows 9–12 are never a target in training** and are pushed down by every cross-entropy step; the learned absolute positions of tokens past the training sequence length (≈ 23 with axioms + goal + SEP + term; ≈ 31 at size 7) are untrained; and the HALT decision (goal = term) must compare token sequences at shifted offsets. Then every instance whose shortest proofs all need a position ≥ 9 fails by construction, and instances where the canonical proof touches one (3–40 % by world and k; CI 20–40 %, AD 3–37 %; §0.8) fail unless another shortest proof avoids it. "Size-6–7 extrapolation in ≤ 3/10 seeds, failures at unseen positions" is therefore the prediction *for a categorical head*, whatever the router learned about rules; it says nothing about rules against tables. (An absolute-position table and a structural rule are distinguishable only if the architecture can address a new position at all.)
- The planted table router "lookup over whole training terms, failing size 6–7 as predicted" fails because it has no entries; it tests the harness, not the hypothesis.

**Fix.** Make the position addressing structural: a **pointer head** (per term token, one logit per rule, from that token's final representation; HALT from the SEP or goal token) with **relative or tree-structural positions** (depth and left/right path, or relative attention), so that "position" is never a category. Run the designed categorical/absolute router as the **identical-architecture control** (same parameters, same training): Q2 becomes "which position representation extrapolates, and does either learn a table?", a real contrast with a mechanism, and both outcomes are informative. Replace the table router with two symbolic references that match the two hypotheses: a pattern-at-absolute-position table and a pattern-at-structural-position rule, both built from the training stream, both evaluated at sizes 6–7.

### B5. (B*, D*) is well-defined but tie-dominated and proxy-relative in a way that decides P4; the lookahead-d policy as described is ill-defined past the remaining distance (§1 per-decision difficulty, Q3, P4, §7)

- **Well-definedness.** With candidates = valid state-changing actions (state this: with no-ops as candidates a no-op is the argmin whenever every move raises σ), D* exists and D* ≤ remaining distance (at d = remaining distance the correct actions reach σ = 0 and no incorrect action can). Computable exactly within the radius-k ball.
- **Tie dominance.** D* as defined (argmin ⊆ correct) treats a σ-tie between a correct and a wrong action as a garden path. In small trees σ is a small integer and ties are everywhere: at the first decision, strict garden paths (the right move looks *strictly* worse) are 0–60 % and σ-ties with a wrong action 0–44 % (§0.4; AI k = 6: 0.60, n = 10, not in the table); under layer sampling the modal D* at k ≥ 4 is D* = k because a wrong neighbour keeps tying until σ = 0 separates (AC k = 6: 7/10; ACDI k = 5: 13/24). So "D* ≥ 3" mixes "needs three steps of lookahead" with "needs a tie-breaker", and P4's "≤ 0.5 at D* ≥ 3" will be read on a stratum that is mostly ties.
- **Proxy dependence.** D* = 1 at the first decision moves by 0.2–0.6 across TED / SED / bag at the same k (§0.3; bag is unusable). TED itself is a weak proxy for rewrite distance (a leaf swap at different depths is TED 2 and rewrite distance ≥ 3), which is *why* garden paths are common: the strata measure the gap between TED and the true distance as much as the task. A router can learn a better one-step progress measure than TED and then be flat in TED-D* without "absorbing lookahead"; P4's falsifier reading ("the amortized prior does the search's work from data") would be wrong in that case, and the derive step cannot settle it by choosing σ.
- **The lookahead policy.** "Choosing by min σ over the best d-step continuation" is ill-defined once d exceeds the remaining distance: every neighbour reaches s within d − 1 steps by backtracking, so all tie at σ = 0. Observed: a lookahead-3 policy with σ-only ordering fell to 0–0.4 success at k ≤ 2 in every world; ordering by (σ, steps-to-reach-it) restores 1.0. The same defect will sit inside any rung-5 rollout value that scores "best state within d steps" without a step count.

**Fix.** (1) Report three per-decision quantities, each exact: GP1 (strict garden path at d = 1), D*_sep (as designed: strict separation), and D*_any (first d at which some correct action is in the argmin: the depth at which a random tie-breaker has a chance); state the candidate set; order lookahead values by (σ, steps). (2) Make P4 operational against the ladder of B3: the prior's reach = the largest d at which RT-canon's success per k and per GP1 stratum is ≥ the lookahead-d rung's, with the ladder's numbers preregistered from the derive step (they are in §0.4–0.5 for TED). Keep TED-D* as a secondary cut labelled TED-relative; add SED as the sensitivity proxy. (3) Preregister the stratum sizes (B1): on the generator as specified the D* ≥ 3 stratum within training worlds is ≈ 1–3 % of instances.

### B6. No reachability or expressivity ceiling for the held-out worlds; the planted router is the oracle (§4, §5, L24, L17)

- "Hand-written: match rule patterns at positions; choose by the oracle's distance table" is the BFS oracle wearing a hat; it is not a point in the router's parameter space, so it tests neither expressivity nor reachability of the architecture (L24). The arms table has **no arm trained on the held-out worlds**, so "RT fails {A, D}" cannot be separated from "this router cannot learn {A, D} in 20k steps" (D-worlds are the hardest under greedy: §0.4), and P2's seed counts have no ceiling.
- RT-single-world × 8 at one seed each cannot supply that ceiling (one seed; wrong worlds).

**Fix.** Add **RT-all** (trained on all 15 worlds, 3 seeds) as the in-distribution ceiling for every held-out world and stratum. Keep the oracle as the evaluation reference, not as a "planted router". If a planted *neural* solution is wanted for L24, plant the HALT/equality circuit and the mask read (both are one attention each) and check their reach from the init; the planning part has no hand-writable weights, which is fine to say.

---

## Should fix

### S1. The exact-posterior arms and the executor cost (§1, §4 PF, §9)

PF's target depends only on (W, t, s): precompute it with the dataset (BFS 0.3–0.5 s per instance in Python for the full world at cap 5, forward–backward 1–3 ms with a sparse matrix; a 100k-instance dataset is ≈ 10 CPU-hours in Python, minutes with a compiled BFS). An EX-style own-policy posterior needs the router evaluated at every reachable state (3–6k forward passes per instance in the full world) and is not feasible in a training loop; it is not in the arms table, so §1 should not list it as a per-episode computation. The executor itself is cheap (≈ 50 µs per step per episode in Python); §9's "vectorized rewriting is the pilot's first measurement" worries about the wrong cost — the label generation is the cost.

### S2. RT-post's target is under-specified, and its fixed point is computable (L28)

State: the posterior is uniform over shortest proofs as *action* sequences or as *state* sequences (they differ: I← at two positions can produce the same term; C at x⊕x is a self-loop and must be excluded); which states are visited in teacher forcing (states along one sampled proof, weighted by the number of proofs through them, or every state on every shortest proof); and the pooled fixed point: a router that cannot distinguish two states pools their soft targets, so RT-post's end state on the training worlds is derivable in the same way PF's was (L28). P6's "RT-post keeps a mixture" is then by construction; make it a check and say what mixture.

### S3. Q6 counts orderings as strategies (Q6, P6)

Tens to hundreds of shortest proofs per instance (§0.3) are mostly interleavings of independent moves at disjoint positions, not strategies. Define strategy equivalence modulo commuting moves (Mazurkiewicz traces: two proofs are equivalent if they differ by swapping adjacent moves at non-overlapping positions) before asking "which strategy each signal commits to". Note also that the canonical label is lexicographic by rule order (A→ < A← < C < I→ < I← < D→ < D←) then position, so RT-canon's "strategy" is the same lex rule in every world; "committed strategies differ by world" would be the rule order, not a finding.

### S4. RT-examples has a ceiling by construction (Q7, P7)

Two example proofs reveal every axiom of W in 44–65 % of pair-world episodes and 9–21 % of triple / full episodes (§0.9). The arm can still succeed when the axioms the proof needs are among those shown, so derive the ceiling (share of episodes with needed ⊆ shown, which is ≥ the cover rate) per world before predicting "≥ 0.9 on training worlds". The falsifier "attention reads the generator's signature directly" does not say what signature; the (t, s) pair itself reveals the needed axioms in ≈ 96–100 % of instances (B2), so a high RT-examples score is the expected outcome, not an anomaly.

### S5. Evaluation protocol: curves, counts, masking, outcome categories (L31–L33)

- No held-out criterion curve is specified; every prediction is an endpoint. Add a fixed held-out set per stratum evaluated every 250–500 steps and classify runs from the curve (learned-then-collapsed, slow-rising, flat), as the route/commit rounds did.
- Report every stratum's instance count with its success; a per-seed ≥ 0.9 threshold on a 20-instance stratum is one failure away from the bar (L18).
- State whether the committed route is evaluated with or without a validity mask. EI16 searches "over valid actions with the verifier"; RT's committed route is unmasked and a single invalid action repeats to T (deterministic policy, unchanged state). Report both readings; the unmasked one is the thesis reading, the masked one is the fair comparison with EI16.
- Outcome categories per failed episode: invalid-action loop (and whether out-of-world or non-matching), wrong halt, no halt at goal, wandering within valid moves, horizon.
- Say which shortest proof "max D* along the oracle proof" uses (canonical) and also report the minimum over shortest proofs of the max (the easiest proof).

### S6. The held-out split (§3)

{A, D} and {C, I} are the *least* informative pairs for recombination: cross-axiom garden paths are 0–12 % (§0.7) and binding is rare on the generator. Under layer sampling the binding-rich pairs are {A, I} (0.26–0.94 at k = 4–6), {C, I} (0.24–0.79) and {A, D} (0.24–0.68); {C, D} and {D, I} bind little. Recommended: hold out {A, I} and {C, I} (or {A, I} and {A, D}), train on the other four pairs and the singletons, and oversample the binding stratum in both; the triples and the full set are where cross-axiom arbitration is common (10–52 %) and should carry P2's second row. Also consider one held-out *singleton* ({I} alone, trained on pairs only) as the decomposition direction, which a mask handles if the mask is read — a cheap check.

### S7. The rung-5 outline (§7; I36, D122)

- The three rollout forms are comparable on the same strata only if the strata are instance properties that do not depend on any form's own value: k, binding, GP1, D*_any; not a TED-D* read through a learned value.
- "The latent-against-exact lookahead curve — the share of (B)'s gain over the no-rollout prior that (A) and (C) capture, per D*" is ill-defined when (B)'s gain is ≈ 0 (every stratum where the prior is already ≥ 0.9, and the D* = 1 stratum by construction), when "lookahead" is not the same unit across forms (a latent rollout of d steps, an executor rollout of d steps, and a beam of N expansions), when the no-rollout prior differs between forms (A trains it jointly with the dynamics), and when the terminal value has no step count (B5's defect). Define the unit as expansions (candidates × depth) and the terminal evaluation per form (goal test + σ for B; the learned value for A; both for C), and report the curve at matched expansions per k and per GP1, with the undefined cells left blank rather than ratioed.
- The CoT opponent needs its training signal stated: written rollouts generated by the executor are richer supervision than next-step labels, so "matched in exploration budget" must also say matched in what the training data contain. If the opponent picks among its B chains with the executor's goal test it is form (B) in disguise; if with its own judgement it is form (A). "Matched tokens" favours the library by construction (a written term is ≈ 13 tokens); matched expansions is the fair primary unit, matched tokens the secondary.
- "Dense-L cliffs at a D* set by L": D* is TED-relative; a dense model's internal computation is not; preregister the cliff against the ladder of B3, not against TED-D*.

### S8. Things likely to be regretted at analysis time (item 7 of the brief)

- The generator's realized length distribution (§0.2) will make every "by k" table at k ≥ 4 a table of single digits unless B1 is fixed; the temptation will be to read endpoints at k ≤ 2.
- P2's ≥ 0.9 as a Zipf-weighted mean (B3); P8's undefined baseline (B2); P3's position artefact (B4); D* ≥ 3 as a tie stratum (B5). Each is a quantity that will be measured correctly and read wrongly (L32).
- The first-decision (B*, D*) and the max-along-the-proof are different strata; the design reports both but predicts on neither explicitly. Pick the primary one now.
- "RT-single-world × 8 at 1 seed" will be cited as a baseline it cannot be (no distribution; the wrong worlds).

### S9. I← is not the inverse of I→ (§1 Rules)

I→ removes a 0 on either side; I← adds x⊕0 on the right only. In worlds without C the graph is directed (0⊕x → x has no inverse), d(t, s) ≠ d(s, t), and the planted-reference intuition "reverse the proof" fails. Either add x → 0⊕x to I← (two more actions per position) or state the asymmetry and its consequence for the reverse-distance computations and ties.

---

## Minor

- **M1.** §1 "≤ 13 positions, ≤ 92 actions" is right (7 × 13 + HALT). Say how W is tokenized (four presence tokens, or a list), since the shuffled-axioms null permutes them.
- **M2.** §1 "k ~ Zipf(1.5)": state the realized distribution per world (B1) and the resampling rule when a (W, k) layer is empty.
- **M3.** §1 "untyped semantics, as the route carrier's": in Lean a failing tactic is an observed error, so the no-op semantics are a choice that makes the toy *harder* than the target in one respect (no feedback) and easier in another (a wasted step costs one of T = 2k + 2). Say so in §0 or §11.
- **M4.** §9 budget: the dataset build dominates (S1); 10–20k steps × batch 64–256 = 0.6–5M episodes, so a fixed precomputed dataset of ~100k instances with replacement, or a compiled BFS, must be the plan before the pilot.
- **M5.** P5 "EI16 generalizes in no fewer seeds than RT-canon's rate" compares 3 seeds against 10; state it as a rate with its uncertainty or run EI16 at 5.
- **M6.** Batches mix worlds uniformly, but singleton worlds have no k ≥ 5 and far fewer distinct instances (≤ 16 states per start in {A}, {C}); the per-world instance diversity should be recorded (how many distinct (t, s) pairs the stream contains per world).
- **M7.** §13 prior work: equational rewriting with learned heuristics has a rewrite-distance literature (e.g. learned term-rewriting heuristics and E-graph / egg guidance); the TED-vs-rewrite-distance gap found here is the known reason syntactic heuristics fail there. Worth one line so (B*, D*) under TED is not presented as new in that part.

---

## Checked and found sound

- **The carrier is implementable as specified** and small: seven rules at positions, worlds as masks, exact BFS, exact reverse distances, exact shortest-proof counts and exact (B*, D*) all run in well under a second per instance at cap 5 in pure Python; the full reachable graph at cap 5 is ≤ 6k nodes in every world (§0.1), so the DP posterior and the planted references are feasible (S1 on where it is not).
- **Garden paths exist and are deep under TED** (§0.3–0.4): strict garden paths at the first decision are 0.14–0.60 at k ≥ 5 across the multi-axiom worlds, greedy-TED succeeds in only 0–0.4 of layer-sampled instances at k ≥ 5 (0.04–0.73 at k = 4), and lookahead-3 recovers 0.57–0.93 at k = 4 and 0–0.89 at k = 5–6. Q3 is a real question on this carrier once the strata are generated deliberately (B1) and defined without tie contamination (B5). The carrier is not greedy-solvable.
- **D* is well-defined and bounded** by the remaining distance, and exact within the radius-k ball, so the generator can compute it per instance as §1 claims.
- **Ties are real** (median 8–156 shortest proofs at k = 4–6, §0.3), so RT-canon against RT-post is a genuine contrast once strategy equivalence is defined (S3).
- **Cross-axiom arbitration exists** in the triples and the full set (10–52 % of first decisions at k ≥ 3, §0.7): the recombination question has a non-derivable residue on this carrier; it is just not where P2's headline row (the pairs) looks.
- **The axiom tokens leak nothing beyond W**; the leak runs the other way (the goal reveals the needed axioms), which is B2's content, not a confound of the cue.
- **HALT and T = 2·k_max + 2** are consistent with the no-op semantics (an invalid action repeats to T under a deterministic policy; one wasted step per T is affordable at k ≤ 6).
- **The arms are parameter-matched** and learning rates are per arm; the seed counts (10 / 10 / 3) are adequate for count-based predictions if the strata are large enough (S5).
- **Q5 (labels against search)** is a fair reference as posed, with the masking caveat of S5.
- **The two-reading report (I36)** is honest: rung 4 tests the prior under either reading and says so.
- **Evaluation at sizes 6–7 is feasible** without the full graph: the radius-k ball from a size-7 term is ≤ 1.6k nodes at k ≤ 4 in every pair world (§0.8 run).

---

## Summary of what the revision should do

1. Replace the random-walk generator with layer sampling, record the realized (world, k) table, oversample k ≥ 3, the binding stratum and the garden-path strata with stated weights; restate Q4 as k = 7–8 in {A, C} or raise the cap with the size confound named (B1).
2. Make the binding stratum (d_W > d_full) the core of Q1; derive the cue-blind ceiling per world and read P8 against it; stratify P2 by binding and cross-axiom garden paths; reconsider the held-out pairs (B2, S6).
3. Add the symbolic ladder (greedy-TED, lookahead-2, lookahead-3, oracle) as planted references; preregister P2 above the d = 1 rung on the strata where it fails, and P4 as the prior's reach along the ladder (B3, B5).
4. Give the router a pointer head with structural positions and run the categorical/absolute design as the identical-architecture control; replace the table router with absolute-position-table and structural-rule references (B4).
5. Report GP1, D*_sep and D*_any with the candidate set stated and lookahead ordered by (σ, steps); label TED-D* as proxy-relative with SED as sensitivity (B5).
6. Add RT-all (3 seeds) as the in-distribution ceiling for the held-out worlds; drop the oracle-as-planted-router claim (B6).
7. Specify RT-post's posterior and visitation, strategy equivalence for Q6, RT-examples' ceiling, the criterion curves, stratum counts, the masking reading and the outcome categories (S2–S5); fix the I← asymmetry or state it (S9).
