"""Round B of the architecture search: gating BEFORE the streams - which ways of training a sparse gate work. EXPLORATORY.

Program task as in round A; progress signal = the register that starts still (reg0, the best of round A); four
streams. Every variant is evaluated the way it would be run: only the top-k streams computed per pass.

What the exact statements say going in (tests A3, A6): an unrun stream gets no gradient; with sigmoid scores its logit
gets none either; with a softmax over all streams its logit gets one that says nothing about that stream; with top-1
and renormalisation the gate gets no gradient at all; a running-mean stand-in is zero for these streams.

Predictions, written before the run (15 Nov 2026):
  B1  top-1, softmax renormalised          fails, 0/5, at or above the order-blind floor (0.03): the gate cannot learn.
  B2  top-1, softmax not renormalised      mostly fails: solved at most 2/5, purity below 0.9 (lock-in on the first split).
  B3  top-1, sigmoid                       as B2: at most 2/5.
  B4  top-2, sigmoid (the sigma-MoE form)  better than top-1: solved at least 3/5.
  B5  top-2, softmax not renormalised      between B2 and B4.
  B6  top-2, sigmoid + expert dropout 0.1 + usage-entropy bonus 1e-3 (the full sigma-MoE recipe)   at least as many solved as B4.
  B7  top-1, sigmoid + expert dropout 0.2  more solved than B3.
  B8  top-1, softmax + selection noise 1.0 more solved than B2.
  B9  dense for the first half of training, then top-1 (softmax)   solved 5/5; error within 3x of the dense reference
      (2.7e-5) - round A's dense gates already give each token its own stream.
  B10 top-1, softmax + running-mean stand-ins (Default MoE)   no better than B2 (stand-ins are zero here, test A6).
  B11 top-1, softmax + linear stand-ins fitted to each stream's outputs when it runs   better than B2 but short of B9:
      a stream that never runs has no stand-in either.
  B12 dense sigmoid (reference, all four streams run)   solved 5/5 but streams mixed (purity below 0.9), as unnormalised
      gates were in lens 1.
"""
import sys
from loopmoe import table

R = dict(progress="reg0")
ROWS = [("B1 top-1 softmax renormalised", dict(R, score="softmax", topk=1, renorm=True)),
        ("B2 top-1 softmax", dict(R, score="softmax", topk=1)),
        ("B3 top-1 sigmoid", dict(R, score="sigmoid", topk=1)),
        ("B4 top-2 sigmoid", dict(R, score="sigmoid", topk=2)),
        ("B5 top-2 softmax", dict(R, score="softmax", topk=2)),
        ("B6 top-2 sigmoid + dropout + entropy", dict(R, score="sigmoid", topk=2, xdrop=0.1, ent=1e-3)),
        ("B7 top-1 sigmoid + dropout 0.2", dict(R, score="sigmoid", topk=1, xdrop=0.2)),
        ("B8 top-1 softmax + noise", dict(R, score="softmax", topk=1, noise=1.0)),
        ("B9 dense then top-1", dict(R, score="softmax", topk=1, anneal=0.5)),
        ("B10 top-1 + running-mean stand-ins", dict(R, score="softmax", topk=1, fill="ema")),
        ("B11 top-1 + linear stand-ins", dict(R, score="softmax", topk=1, fill="lin")),
        ("B12 dense sigmoid", dict(R, score="sigmoid"))]
if __name__ == "__main__":
    table([ROWS[int(i)] for i in sys.argv[1:]], range(5))
