# Rewrite carrier code review 2 (gate G4: the implementation of DESIGN revision 3 + §12a, 6 Nov 2026)

**Reviewer:** independent, adversarial. Read AGENTS.md, LEDGER L24–L33, DESIGN.md (rev. 3, §12a), review1_design.md (§0, B1–B6), and every file under review. Nothing in the repo was edited except this file. Scratch scripts and their outputs are in `/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/review_rewrite_code/` (`check_math.py`, `time_arms.py`, `profile_step.py`, `mem_probe.py`, `feature_rows.py` + `.log`, `time_eval.py`, `bundle/`, `derive_copy/`).

**What was run.** `pytest -q tests/test_rewrite.py`: 29 passed in 126 s. Brute-force re-derivations of the lookahead values, D\*_sep / D\*_any / GP1 and TED properties on random instances. The training step of every arm timed on CPU with the real `data/rewrite/train_*` files; the step profiled into data path and model path; the encoding cache's memory growth measured for RT-canon and RT-examples; a structural router trained 2,000 steps on 7 real training worlds and evaluated on the smoke eval sets with raw and with range-clamped structural features; the Kaggle code bundle reconstructed from `launch.py`'s include list and the test gate run inside it; `derive.py` (a scratch copy) re-run on the smoke data and `analyze.py` run on the fixture results with that `derived.json`.

---

## Verdict: **fix first.** Five items would each either stop the session, kill a shard, or make a preregistered reading wrong; all are cheap (hours, not days). The carrier, posteriors, strata and generator are correct where I could check them exactly.

---

## Blocking

### B1. The Kaggle test gate fails on the uploaded bundle, so the grid never starts (`experiments/fleet/kaggle/launch.py:49`, `run_on_kaggle.py` job "rewrite")

`REWRITE_INCLUDE = ["tests/test_rewrite.py", "experiments/rewrite/run_rewrite.py"]`. `tests/test_rewrite.py` loads `experiments/rewrite/analyze.py` by path and `tests/fixtures/rewrite/*.json.gz`; neither is in the tar. Reconstructed the bundle from the include lists (`bundle/`) and ran the gate command inside it: `FileNotFoundError: .../bundle/tests/../experiments/rewrite/analyze.py`, `1 failed` → `run_on_kaggle.py` prints "TESTS FAILED: grid not run" and exits 1. The launch burns the upload and a GPU slot and returns nothing.

**Fix.** `REWRITE_INCLUDE += ["experiments/rewrite/analyze.py", "tests/fixtures/rewrite"]` (tarfile adds directories recursively). Also add `experiments/rewrite/derive/derived.json` if the analysis is to run on Kaggle.

### B2. The encoding cache will OOM the RT-examples shard, and costs 1.5–3 GB per process for the other arms (`src/topology_design/rewrite/train.py:110–120`)

`encode_cached` keeps up to 400,000 encodings, each holding an int64 `rel` matrix (L × L) and `pathcode` (L × 8). Measured (`mem_probe.py`, RSS after loading 7 training worlds = 1.49 GB):

- RT-examples: the key includes the two random example proofs, so every row is a new key (hit rate ≈ 0); 257 new entries per step at **31–40 KB each** (L ≈ 77). The cap is reached at step ≈ 1,550 with **+12–16 GB** resident. The shard process dies; every later run assigned to that shard is lost for the session (results already saved survive the relaunch).
- RT-canon: 7.5 KB per entry; the training stream has 160k distinct (world, s, state) keys in 7 worlds (≈ 190k in 8), RT-post 315k (≈ 380k), RT-all ≈ 400k, plus ≈ 100k evaluation states → the cache alone reaches 1.5–3 GB per process on top of 1.2 GB (8 worlds) to 2.8 GB (15 worlds) of records. Two shard processes plus CUDA contexts is at or over what a Kaggle kernel has; `--per-gpu 2` would be impossible.

**Fix.** (i) Do not cache encodings that carry examples (or cache the base (world, s, t) encoding and append the example tokens per row). (ii) Store `rel` as int16 (N_REL = 475) and `pathcode` as int8, converting in `collate` — ≈ 4× smaller. (iii) Lower the cap to ≈ 150k, or evict. (iv) In `load_set` for training, drop `pf` unless the arm is PF (it is most of the AC file).

### B3. RT-all silently becomes a third RT-canon when the held-out training files are missing — and the default build does not make them (`train.py:394–399`, `experiments/rewrite/build_dataset.py:143`)

`run()` takes `ALL_WORLDS` for RT-all, `load_set` skips files that do not exist, and the only guard is `assert train`. `build_dataset.py` builds the `train` set for `TRAIN_WORLDS` only (line 143); the running build (`data/rewrite/build_train.log`) has produced exactly the eight training worlds' `train_*` files (train_ID finished during this review; it is now on eval5) and no held-out `train_*` file is in its plan. Launched as is, the three RT-all runs train on the eight training worlds, their held-out success is RT-canon's, and P2's "RT-all ceiling ≥ 0.9 … the gap is recombination, not reachability" is read against the wrong arm (B6 of review 1 undone).

**Fix.** Build `--only train --worlds AI,CI,ACI,ACD,AID,CID,ACID` before launch; in `run()` assert `set(train) == set(worlds)`; in `launch.py`, when the grid contains RT-all, require the 15 `train_*` files.

### B4. P3 is foreclosed in both directions by untrained embedding rows, not by position addressing (`model.py:224–231, 161–162`; DESIGN §1 "position is never a category", §12a.5)

The pointer head does address every node of a size-7 term (verified; `test_matched_configs…`). But the *inputs* of both routers are categorical and out of range at sizes 6–7:

- **Structural:** `depth` rows 5–6, `size` (subtree leaves) rows 6–7, `path` level-4/5 L/R rows, and the ancestor/descendant-by-5/6 relation rows are never used at size ≤ 5 (training max depth 4, verified on the real stream) and sit at their N(0, 1) init (`nn.Embedding` default; `tok` is N(0, 0.02)). On the smoke eval7 set **100 %** of instances expose a size-6/7 root (t and s), **68 %** expose depth ≥ 5 in t or s (70 % along the canonical proof). Probe (`feature_rows.py`, 2,000 CPU steps on 7 real worlds, then the *same* model evaluated with raw and with range-clamped features — training never leaves the range, so clamping at evaluation is exactly the clamped model): eval5 identical (0.706 / 0.264 masked, train / held-out, as it must be); **eval7 training worlds 0.312 raw → 0.412 clamped** (n = 160; 21 of 300 eval7 outcomes flip; first-decision 0.55 → 0.58). The untrained rows cost the structural arm ≈ 0.1 of its size-6–7 success at this stage of training; the design's prediction P3 is a ratio with a 0.9 threshold. A failure would be read as "the structural router also learned a table".
- **Absolute control:** `self.pos` rows ≥ 23 are untrained (training sequence length ≤ 4 + 9 + 1 + 9 = 23, verified); every eval7 sequence is 27–31 tokens, so **100 %** of eval7 instances put the last 4–8 term tokens on N(0, 1) position vectors. The derived unaddressable share (categorical head rows 9–12) is 0–0.3 by world, ≈ 0.1 pooled. The absolute arm will therefore fail on *addressable* instances too, which P3's falsifier reads as "a table beyond the artefact". That outcome is fixed by the code, not by what the router learned.

**Fix.** Structural: clamp the categorical features to the training range inside the model (`size` ≤ 5, `depth` ≤ 4, path levels 0–3 only, ancestor/descendant distance ≤ 4) so that no row used at size 6–7 is untrained, and say so in §12a.5 (the exact equal-path relation for HALT is unaffected; it uses full paths). Absolute: give the control fixed sinusoidal absolute positions (the categorical head's unlabelled rows remain the designed artefact and P3's second clause becomes readable), or preregister that the clause cannot be read and drop it. Either way, init all feature embeddings at std 0.02 like `tok` (S8).

### B5. The automatic P4 reading passes at zero success (`experiments/rewrite/analyze.py:227`)

`between = … d2 <= s_ < d3`. On the GP1 ∧ k ≥ 4 stratum the d = 1 and d = 2 rungs are ≈ 0 (review 1 §0.4; the smoke fixture's own P4 row: n = 23, d1 = 0.00, d2 = 0.00, d3 = 0.35, RT-canon success 0.00 → **"between d=2 and d=3: 1/1"** and simultaneously "≤ d=1: 1/1"). The same seed is counted as matching lookahead-2 and as absorbing no lookahead. P2's two conditions (line 184, 186) are counted in separate seeds rather than jointly, so "≥ greedy+0.2 in 6/10 and ≥ 0.7 in 6/10" can hold with no seed meeting both.

**Fix.** Strict inequalities and a separation guard: `between` requires d2 < s_ < d3 and d3 − d2 ≥ 0.1; print a cell as "rungs not separated" otherwise. Count P2's conjunction per seed. See S7 for the stratum itself.

---

## Should fix

### S1. P3's comparison is not matched (`analyze.py:201–203`)
`s7/s5` pools all 15 worlds; eval5 at k ≤ 4 and eval7 differ in binding share and k mix after rejection (eval7's binding target may be infeasible at k ≤ 4 and sizes 6–7, pushing q = p to the floor), and the held-out worlds mix Q1's difficulty into a position question. Compute the ratio on (world, k, binding) cells reweighted to eval7's composition, and make training worlds the primary row.

### S2. The P2–P4 tables are masked-only (`analyze.py:173–232`)
DESIGN §1 calls the unmasked route the thesis reading. Add `ok_u` columns to P2 and P4 and state in §6 which reading each pass rule uses (currently masked, implicitly).

### S3. P6 has no derived target; P7/P8 branches are untested (`analyze.py:253–268`; `tests/test_rewrite.py:583`)
"Concentration on the best shortest-path action" cannot tell "ended at the RT-post target mixture" from any other spread. The target at t is in every eval record (`post[state == t]`); compute TV/KL between the masked first-action distribution and that target for RT-canon, RT-post and PF, and the lexicographic-class share for RT-canon. The fixtures hold RT-canon, RT-absolute and EI16 only, so the P7/P8/P6-RT-post branches never run under test (L33); the `derived.json` in `rw_smoke2/derive_test` predates `eval7_unaddressable`, so the P3 split branch was also unexercised until I re-ran derive (it works). Add 20-step fixture runs for all eight arms and a derived fixture. Also save `visited_goal` per instance (computed in `rollout`, dropped at `train.py:368–372`).

### S4. PF targets are computed for all 10,000 records and 7,500 are thrown away (`build_dataset.py:87–89, 95–98`)
`pf_first = per` on every chunk; the parent deletes `pf` beyond the quota. For a held-out ACID training file that is ≈ 10k full-world BFS + DP for nothing. Either keep them all (removes the confound that the PF arm trains on a quarter of RT's distinct instances — any PF-vs-RT difference in generalization is then also a data-size difference) or pass `pf_first = max(0, quota − n_pf)`.

### S5. The pilot sweeps lr for four arms only (`make_grids.py:24`)
RT-post (soft targets), RT-examples (3× context), RT-all and RT-shuffled get 1e-3 untuned. AGENTS rule 4: tune per arm. Eight more 4k-step runs.

### S6. Budget and session time (`run_rewrite.py:27`, `make_grids.py`)
Measured on CPU (1 thread, 256 rows): RT-canon 359 ms/step (data path 21 ms, fwd+bwd 319 ms), RT-absolute 288, RT-post 345, PF 328, RT-shuffled 346, **RT-examples 1,426** (L = 77), **EI16 993** with an untrained model (Python-bound beam; it does not shrink on a GPU). One curve evaluation with a half-trained model: 6.6–7.7 ms per instance → **30–36 s per 250 steps on CPU** (4,650 instances × 2 readings × 6.3 steps mean), 122–143 s for the final full sets. Extrapolated to a T4 (fwd/bwd ≈ 10–15 ms, Python unchanged): RT arms ≈ 7 min training + ≈ 8–10 min of 49 evaluations ≈ 15–18 min per run; RT-examples ≈ 25 min; EI16 ≈ 90–120 min. 41 runs ≈ 18–20 GPU-hours ≈ 9–10 h per shard on 2×T4 — inside a 12 h session with little margin and a 2× uncertainty on my extrapolation. The evaluation cadence costs as much as training. Read the pilot's `seconds` (its EI16 at 4k steps) before launching main; run `--per-gpu 2` once B2 is fixed (the work is Python-bound); consider the curve subset at 100 per world or eval every 500 for the 10-seed arms if the pilot says so — a design change, so write it into §12a before the main grid. `COST["RT-examples"] = 1.5` should be ≈ 4.

### S7. P4's stratum does not separate its rungs (DESIGN §6 P4; `derive.py` §3)
On GP1 ∧ k ≥ 4 the d = 1 rung fails the first decision by construction and d = 2 is ≈ 0 in the A/D-containing worlds (review 1 §0.4: AC k = 5–6 d1 = d2 = 0.00–0.08; AD k = 5 0.00/0.00). "Between d = 2 and d = 3" is then "> 0 and < d3". Preregister P4 on cells where d2 − d1 ≥ 0.1 and d3 − d2 ≥ 0.1 (the derive step has the table), and report Q3's quantity as the design states it: the largest d with success ≥ rung_d, per cell.

### S8. Embedding init scales (`model.py:235`)
`tok` std 0.02, every structural / segment / position embedding std 1 (default): token identity is ≈ 2 % of the input norm at init in both arms; the root's input is dominated by eight summed path-"none" vectors. Init all embeddings at 0.02. Not an asymmetry between the arms, but it slows both and magnifies any untrained row (B4).

### S9. Text (`DESIGN.md:134–135`)
Matched FFN width is 540, not 534 (629,321 vs 629,225 parameters, 0.015 %). "Position is never a category" needs B4's qualification. World I's stream is 79 % binding because the GP1 target is infeasible there and the fallback maximises the GP1 ratio by pushing p to the floor (`meta.json` rates: q = 1.00, p = 0.05; k = 1 is 6 % of I's stream against 57 % of A's) — within §12a.3's letter, but state it in the realized table's caption.

---

## Minor

- **M1** `search.py:153–157`: "independent actions commute" is false under the size cap: at cap 5, `Df@L` then `Db@R` on `((a+a)+(b+c))` is valid, the swap is not (verified). Foata classes may contain unexecutable orderings; the class count of valid shortest proofs is still a count under a coarser relation. Say so.
- **M2** `data.py:271`: `summarize`'s D\*_any shares stop at 7; `evalk78` has D\* up to 9.
- **M3** `analyze.py:277`: P7's ceiling is an unweighted mean over worlds while the rate pools instances; weight by `binding_n`.
- **M4** P8: the cue-blind ceiling is what a cue-blind *perfect* planner scores (ACID: 1.000), so "within 0.05 of the ceiling" also tests planning ability; read shuffled against ceiling × RT-canon's non-binding success, or on the binding stratum only.
- **M5** `train.py:276–299`: the model is run at t = T before the forced halt (one wasted forward per episode); `alive.index(i)` is O(n) per instance.
- **M6** `train.py:357`: the curve's last point is the full 600/world set while earlier points are the first 150; `classify` compares them. Harmless in expectation; note it.
- **M7** EI16 trains on ≈ 32 × (k + 1) ≈ 140 rows per step when every beam succeeds, RT on 256: "matched steps" gives EI16 ≈ 55 % of RT's rows. Record cumulative imitated rows next to `ei_found`.
- **M8** `build_dataset.py:59`: the main phase's `n_none` is discarded (only the warm-up's is recorded; M2 of review 1 asked for the resampling count).
- **M9** `train.py:388` documents `pf_only`; the PF filter is unconditional.
- **M10** `derive.py` writes `derived.json` next to itself (tracked); fine, but a `--out` would let a reviewer run it without touching the tree.

---

## Checked and found sound

- **Semantics (DESIGN §1).** Eight rules as written, `If` tries x⊕0 before 0⊕x, `IbL`/`IbR` both present; no-op semantics (invalid = out of world / no node / no match / over cap, all `None`); candidates = valid state-changing successors (C at x⊕x excluded; a self-loop under the unmasked reading ends the episode as the Markov router would repeat it); binding = `d_full < d_W` with the full-world BFS cut at radius k − 1 (exactly the test); four presence tokens; `{D, I}` is "ID" everywhere (build, train, analyze, derive agree).
- **Math.** `lookahead_value` / `lookahead_profile` agree with brute-force walk enumeration in 1,791 / 1,791 checks; `decision`'s GP1 / D\*_sep / D\*_any agree with an independent implementation in 116 / 116 instances (39 GP1); D\*_any ≤ D\*_sep ≤ k holds and the oracle rung d = k always succeeds under the (σ, steps) order. TED is symmetric, zero on identity and satisfies the triangle inequality on 300 random triples; hand values as expected. `shortest_posterior` weights = share of state sequences through the state, action targets split equal-successor actions uniformly, Σ weights = k + 1; `pf_posterior` matches route enumeration with the forced halt at T (the repo's tests, re-read line by line). Dgoal from the radius-6 ball is exact wherever it is used (every correct successor's path to s stays inside the ball).
- **Generation.** Layer sampling as §12a.2 (k restricted to non-empty layers, s uniform in the layer, `None` counted); rejection is unbiased within categories (q, p depend on the category only); eval7 at radius 4 with k ∈ {1..4}; evalk78 at radius 8; resume per (set, world); realized tables, distinct (t, s) and starts recorded. GP1 and binding are impossible at k = 1 and the tables skip empty cells. Real stream (7 worlds): binding 0.01–0.34, GP1 0.00–0.36, k ≥ 3 share 0.19–0.86, distinct pairs 4.8–7.7k of 10k; RT-post's 24-state cap applies to 10 % of AC instances and the importance weights are unbiased.
- **Model.** Pointer head addresses all 13 nodes (finite logits at size 7; −inf on unaddressable slots at size 2); HALT from SEP in both routers; relation ids exact (leading-1 bit code; equal / ancestor-by-d / descendant-by-d / disjoint × segment pair); no absolute index reaches the structural router; parameter match 0.015 %; the loss drops −inf log-probs safely (target is zero there).
- **Training.** Per-instance loss mass k + 1 for RT-canon, RT-post and PF; AdamW (0.9, 0.98), wd 0.01, warm-up 200 then constant (L31), clip 1; RT-shuffled draws a random training world's pattern per instance and shows the true one at evaluation; RT-examples excludes the instance by identity and duplicates are negligible (≤ 0.02 % per draw at 6.5k distinct pairs); EI16's beam: masked support, ε-mixing, merge by state keeping the best prefix, MAP key log π̃ − 0.02·moves with HALT scored at every goal visit and the forced halt at T, found / total counted.
- **Evaluation.** Forced halt at T = 2k + 2; categories ok / wrong halt / no halt at goal / horizon / invalid (out of world vs no match); first-decision correctness against all shortest-path actions; `p_first` saved for the masked reading; per-instance records ≈ 6 MB JSON per run (fixtures scaled), fine.
- **Analysis.** Cells and per-instance outcomes agree on the fixtures (the repo's test); ladder rungs are read from the same instances as the router's cell (matched denominators); n per seed printed everywhere; EI16 bootstrap CIs present; the P3 addressability split works against a current `derived.json` (268 / 32 on the smoke eval7).
- **Kaggle.** Data travels inside the code tar and `run_on_kaggle.py` points `--data` at it; prior results are copied in for resume; shards are assigned by estimated cost; a failed run is caught and the shard continues.

---

## Summary of what to do before the pilot

1. Add `analyze.py` and `tests/fixtures/rewrite` to the bundle (B1). 2. Stop caching example-bearing encodings, shrink the cache's dtypes and cap (B2). 3. Build the seven held-out training files and assert their presence for RT-all (B3). 4. Clamp the structural features to the training range and fix the absolute control's positions (or drop P3's second clause); init embeddings at 0.02 (B4, S8). 5. Make P4's and P2's pass rules strict and joint, preregister P4 on separated cells, add the unmasked columns and the P6 target comparison, and fixtures for all arms (B5, S2, S3, S7). 6. Sweep lr for every arm in the pilot and read its `seconds` before sizing the main session (S5, S6).
