**R3** program stored in the state; two primitives, four passes, step 0.4; three seeds, median [min, max] held-out error:

<!-- seed 0 done -->
<!-- seed 1 done -->
<!-- seed 2 done -->

| student | held-out error |
|---|---|
| (i) gate constant over the passes (order-blind reference; one free gate per program, fitted on all programs) | 3.4e-02 [3.3e-02, 3.7e-02] |
| (ii) hand-built state-read solution | 6.7e-27 [6.1e-27, 1.4e-26] |
| (iii) trained from the hand-built solution with 5 % noise | 4.1e-05 [1.3e-05, 4.4e-05] |
| (iv) state-read gate trained from scratch, 9,000 steps | 3.7e-02 [2.1e-02, 5.1e-02] |
