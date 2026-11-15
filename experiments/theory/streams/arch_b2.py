"""Round B2: sparse training that is never dense, and whether round B's results survive more streams. EXPLORATORY.

Round B: top-1 from scratch failed under every aid tried; top-2 with a softmax over all streams (not renormalised)
solved 5/5; dense-then-top-1 solved 5/5. Softmax scores throughout here; register reg0. "Explorer" = during training
run the top stream plus ONE more, drawn from the gate's own distribution over the rest (sampled) or uniformly.

Predictions, written before the run (15 Nov 2026):
 four streams, three primitives (as before)
  E1  top-2 throughout, RUN at top-1 with no further training   solved at least 4/5, error more than 3x that of running top-2 (4e-5).
  E2  top-2 for the first half, then top-1                       solved 5/5, within 3x of dense (2.7e-5).
  E3  top-1 + sampled explorer for the first half, then top-1    solved at least 4/5.
  E4  top-1 + uniform explorer for the first half, then top-1    solved at least 4/5.
  E5  dense for the first 10 %, then top-1                       solved at most 3/5 (300 dense steps are too few to separate the streams).
 eight streams, five primitives (80 training programs of 625)
  E6  dense (reference)                                          solved 5/5, five streams used, purity above 0.95.
  E7  top-2 throughout                                           degrades: solved at most 2/5, fewer than five streams used when it fails
                                                                 (two of eight is a smaller share than two of four; streams never selected never train).
  E8  top-2 for the first half, then top-1                       no better than E7.
  E9  dense for the first half, then top-1                       solved 5/5.
  E10 top-1 + sampled explorer for the first half, then top-1    better than E7: solved at least 3/5.
"""
import sys
from loopmoe import table

R = dict(progress="reg0", score="softmax")
BIG = dict(R, K=8, P=5, ntrain=80, B=8)
ROWS = [("E1 top-2, run at top-1", dict(R, topk=2, topk_eval=1)),
        ("E2 top-2 then top-1", dict(R, topk=1, sched=[(0, 2, False), (0.5, 1, False)])),
        ("E3 top-1 + sampled explorer, then top-1", dict(R, topk=1, explore="sample", sched=[(0, 1, True), (0.5, 1, False)])),
        ("E4 top-1 + uniform explorer, then top-1", dict(R, topk=1, explore="uniform", sched=[(0, 1, True), (0.5, 1, False)])),
        ("E5 dense 10 % then top-1", dict(R, topk=1, anneal=0.1)),
        ("E6 8 streams: dense", dict(BIG)),
        ("E7 8 streams: top-2", dict(BIG, topk=2)),
        ("E8 8 streams: top-2 then top-1", dict(BIG, topk=1, sched=[(0, 2, False), (0.5, 1, False)])),
        ("E9 8 streams: dense then top-1", dict(BIG, topk=1, anneal=0.5)),
        ("E10 8 streams: top-1 + sampled explorer, then top-1", dict(BIG, topk=1, explore="sample", sched=[(0, 1, True), (0.5, 1, False)]))]
if __name__ == "__main__":
    table([ROWS[int(i)] for i in sys.argv[1:]], range(5))
