# Audit 4: every pass rule of §6 walked against the derive step's numbers, 9 Nov 2026

**Why.** The derive step had shown P4's preregistered cells unreadable (§16) although a reviewer had flagged the risk; nobody had done the same walk for the other predictions. **Auditor:** independent (a fresh agent with no part in the design, the implementation, the derive step or the pilot). Read-only; it worked from the design text, `analyze.py`, `derived.json` and the eval files, and built symbolic policies to test each reference. **It was told not to open the pilot's results and reports that it did not** (nor `PILOT.md`, nor the fixtures' outcomes). Scripts in [audit4_scripts/](audit4_scripts/) (they write their intermediate JSON next to themselves; run `s2_cueblind.py` and `s5_p2_p7.py` before the reports). Its report follows verbatim; the handling is DESIGN §17.

**Checked by the author before acting on it:** the central P2 claim recounted with the library's own search (`search.distance`) — 608 / 209 and 1,065 / 509, per world as reported; the training-world reference re-implemented in the library (`proxy.trainworld_rollout`) — 0.839 / 0.531 and 0.602 / 0.212, equal to the auditor's implementation to the instance count; the ladder's matched ratios (1.126 / 1.106 / 1.057) and the must-fail set (583 flagged, 265 with an addressable route within T, 318 must fail: D 169, AD 61, I 55, ID 33) recomputed in `derive.py` §9; P7's weighted ceiling (0.954, n 1,408). The cue-blind policies' rates (P8) and the P6 / P7 reference rates are the auditor's own and are quoted as such.

---

**Verdict: do not launch the main grid on the current preregistration.** P8 is not readable; P2, P3 and P7 can pass (or cannot pass) for reasons unrelated to their claims; P5 and P6 lack decision rules. All fixes are text + derive + analyze changes. Run records already save `ok_m`/`ok_u` per (world, t, s), so no trainer change is needed.

Everything below comes from the eval files, `derived.json` and symbolic policies. I did not read `PILOT.md`, `results/` or fixture outcomes. Nothing from memory was used.

| P | status | evidence | minimal change |
|---|---|---|---|
| **P2** | readable with changes (**blocking**) | Pairs binding n=608 (AI 305, CI 303): greedy 0.413, so the binding bar is 0.70, which sits on the lookahead-3 rung (0.692). Triples n=1065, greedy 0.275; full 0. **(a)** 66 % of the pair stratum (AI 85 %, CI 47 %) and 52 % of the triple stratum has a proof within T inside a *training* sub-world. A router that never composes scores 0.656 / 0.522: above greedy, so the falsifier cannot fire on it and the triple clause passes. **(b)** Masked: a token-blind policy that plans only inside the eight training worlds and lets the mask choose scores **0.839** on pairs (meets both pair conditions) and 0.602 on triples. **(c)** RT-all ≥ 0.9 needs +0.21 / +0.28 over lookahead-3; the statistic over 3 seeds is unstated; if it fails, P2 can neither pass nor be falsified. Power (binomial only): ≥ 6/10 with probability 0.87 at true rate 0.71; ≤ 3/10 with 0.95 at 0.68. Anything between greedy and 0.70 is undecided. | Score on binding instances with no proof within T in a training sub-world: pairs n=209 (AI 47, CI 162; greedy 0.397, d3 0.727; the non-composing token-blind policy gets 0.531, below greedy + 0.2), triples n=509 (0.153 / 0.507). Score unmasked beside masked: a world-blind full-world planner + mask still gets 0.708 on the 209. Fix RT-all's statistic and the reading when it is < 0.9. |
| **P3** | readable with changes | Clause 1: 4799/4800 eval7 instances matched (39 on eval5 cells with n < 10). eval7 is easier on matched cells: the rungs' own matched ratio is **1.13 / 1.11 / 1.06** (d = 1/2/3; 1.09 / 1.07 / 1.03 with GP1 in the key). Single-axiom worlds carry 51–58 % of successes. A planted router at the greedy rung, thinned to 85 % at sizes 6–7, scores 0.96 and passes 10/10. Clause 2: 583/4800 flagged unaddressable, but 265 have a route within T on positions ≤ 8 (C 28/28, AC 31/31, CD 57/57, AD 84/145, ID 32/65, I 22/77, D 11/180). Under the mask, in 81 of the remaining 318 the valid set at t is only never-labelled rows + HALT, so failure is not forced. No tolerance or seed count. | Require ratio ≥ 0.9 × the matched rung's own ratio and report multi-axiom worlds separately. Define must-fail at horizon T (n=318: D 169, AD 61, I 55, ID 33); score clause 2 unmasked with a stated tolerance and x/5. |
| **P5** | readable with changes (minor) | n=1673, greedy 0.325, d3 0.644. Both outcomes reachable; masked reading valid. EI16's statistic over 5 seeds is unstated; "more than 0.1 from RT but above greedy" is undecided. Only 718/1673 need an unseen combination. | Fix the statistic (seed-median); print the greedy rung and the 718 sub-stratum. |
| **P6** | readable with changes | n=2213 per seed (D contributes 0, A 61). The pooled fixed point equals the per-state target (0 conflicting targets per (s, state)). Canonical one-hot is TV 0.63 from the target; uniform over correct first actions is 0.10; the target is exactly uniform on 45 %. The canonical action is a target mode on 68 %. No tolerance. | TV ≤ 0.05 for RT-post; canonical-first share ≥ 0.9 for RT-canon. |
| **P7** | clause 1 not readable; clause 2 with changes | Training binding n=1408, ceiling 0.954. "Within 0.1" needs ≥ 0.854 where lookahead-3 gets 0.686, so it is a planning bar. The ceiling holds for a planner confined to the examples' axioms (0.961 / 0.919 / 0.857 against 0.961 / 0.904 / 0.844, my draws). It does not hold for policies reading (t, s) or the mask: on draws it counts as failures, an example-blind policy solves 0.46–0.52 unmasked and 0.83–0.93 masked. Clause 2 has no margin and no rule for 3 seeds against 10. | Compare with ceiling × RT-canon on the same instances (± 0.1), unmasked; clause 2 with a margin in 3/3. |
| **P8** | **not readable** | The ceiling assumes all 105 slots and a full-world planner. That planner under the mask scores 0.58–1.00 per world against ceilings 0.39–0.89: above ceiling + 0.05 in 14/14 non-full worlds, and 0.61 on pairs' binding (ceiling 0). Unmasked, a token-blind fewest-axioms planner scores 0.55–0.99, also 14/14 above; 0.67 on pairs' binding. The ACID ceiling is 1.000 by construction (lookahead-3 gets 0.71). Binding n = 8 (A), 7 (D). No count over 2 seeds; "well below" has no number. | Score unmasked: RT-canon − RT-shuffled ≥ δ on the training worlds' binding stratum (n=1408) in 2/2; report the ceiling unscored. |
| P1 | reporting gap | Ladder at floor (0.000 / 0.002 / 0.020, n=600): no reference. | Say so; add the unmasked column. |

**Analysis-code discrepancies (`analyze.py`)**
- No verdict is computed for P2, P3, P5–P8, only counts and a sentence. P2's falsifier and RT-all rule are not evaluated.
- The unmasked column is missing from the P3, P5, P8 and P1 tables; §6 says every table.
- P3 clause 2 is unscored: the split is printed with no tolerance, failure share is never compared with the unaddressable share, and the flag is shortest-proof only.
- P5 computes neither the greedy rung nor "within 0.1".
- P6 reads the first decision on eval instances only; the route's trace class is never read.
- "Counted jointly per seed" is implemented as (+0.2 and ≥ 0.7); the pair, triple and RT-all clauses are separate. The text is ambiguous.
- §16 says ACI binding is 0/600 at sizes 6–7; the data has 1/600.

**Rules without a planted pass-and-fail test**
- Only P4 has both.
- P2 has a planted pass only (all solved → 1/1).
- P3 has a planted ratio (0.5) but no assertion on the count column or the absolute clause.
- P5, P6, P7, P8 and P1 are checked for section presence only.
- The 7 existing analysis tests pass.

**Blocking, in order**
1. P2's stratum and reading.
2. P8's reference.
3. P7 clause 1 and P3's ratio.

**Not checked**
- Anything neural: whether the masked absolute head picks never-labelled rows, and what RT-shuffled actually learns.
- The training stream's (t, s) → world posterior; I did not stream the train files.
- The symbolic policies are existence proofs under each reading, not forecasts.
