# What the gate is fed, on the stored-program task (EXPLORATORY)

Two primitives, four passes, step 0.4; 12 programs for training, 4 held out; streams act on the data only; the gate is an MLP (one hidden layer of 32). Three seeds, median [min, max]. Predictions in `gate_inputs.py`. References from lens 2: order-blind floor 0.034; a gate built as a per-pass readout of the stored program reached 8e-6 held out.

| gate input | training error | held-out error |
|---|---|---|
| G0 state and stored program | 8.8e-03 [8.8e-03, 9.6e-03] | 5.3e-02 [3.8e-02, 5.8e-02] |
| G1 + the input | 3.4e-03 [3.4e-03, 4.0e-03] | 3.4e-02 [2.1e-02, 6.9e-02] |
| G2 + the previous pass's state | 7.5e-03 [7.1e-03, 8.9e-03] | 4.8e-02 [3.4e-02, 6.4e-02] |
| G3 + both | 3.3e-03 [2.9e-03, 3.3e-03] | 1.9e-02 [1.7e-02, 7.5e-02] |
| G4 + the pass index | 3.2e-04 [3.1e-04, 6.9e-04] | 3.1e-02 [1.6e-02, 6.5e-02] |
