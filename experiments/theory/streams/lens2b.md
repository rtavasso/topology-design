
**2.4** program stored in the state (two primitives, four passes, step 0.4, 12 programs for training, 4 held out; three seeds):

| what the gate sees | training error, median [min, max] | held-out error, median [min, max] |
|---|---|---|
| (b) pass index and the tape | 8.8e-07 [6.3e-07, 1.1e-06] | 7.9e-06 [3.9e-06, 9.3e-06] |
| (c1) the state only | 2.4e-02 [1.6e-02, 2.5e-02] | 3.6e-02 [3.5e-02, 8.0e-02] |
| (c2) the state only, plus one always-on stream | 1.5e-02 [1.4e-02, 2.3e-02] | 3.4e-02 [2.7e-02, 7.3e-02] |
