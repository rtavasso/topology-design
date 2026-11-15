"""Round A of the architecture search: what tells a tied gate which pass it is in. EXPLORATORY.

Program task (three primitives, four passes, step 0.4; 40 programs for training, 41 held out), four streams, dense
softmax gates (every stream runs), gate = attention over the stored program with the query named below.

Predictions, written before the run (15 Nov 2026):
  A0 none            order-blind floor, about 0.03; solved 0/5.
  A1 clock           solved 5/5, median held-out error below 1e-3.
  A2 data            at the floor (above half of A0's error); solved 0/5 - as the MLP gate was in Entry 102.
  A3 reg0            learns, but not always: solved in 3 or 4 of 5; median error above A1's. (Nothing forces M to stay
                     at zero - its gradient is sum_t t * dL/dq_t c0^T - but the register must start moving before the
                     keys can separate, and the keys must separate before moving pays.)
  A4 regrot          solved 5/5, median within a factor 10 of A1.
  A5 regfix          solved 5/5, same order as A4 (four generic fixed queries are a clock).
  A6 regrot + data   still solved in at least 4/5, but held-out error more than twice A4's (the gate can now fit
                     training programs through the state).
  A7 clock + data    as A6 relative to A1.
  Streams: in every solved variant three of the four streams are used and each token has its own stream (purity
  above 0.95), as softmax gates produced in lens 1.
"""
import sys
from loopmoe import table

ROWS = [("A0 none", dict(progress="none")), ("A1 clock", dict(progress="clock")), ("A2 data", dict(progress="data")),
        ("A3 reg0", dict(progress="reg0")), ("A4 regrot", dict(progress="regrot")), ("A5 regfix", dict(progress="regfix")),
        ("A6 regrot + data", dict(progress="regrot", gate_h=True)), ("A7 clock + data", dict(progress="clock", gate_h=True))]
if __name__ == "__main__":
    pick = [ROWS[int(i)] for i in sys.argv[1:]]
    table(pick, range(5))
