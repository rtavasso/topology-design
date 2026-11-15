# Round D — more passes than trained (EXPLORATORY)

Program task, three primitives, four streams, dense softmax gates. Training programs of lengths 3, 4, 5 (40 each; passes = length); tested on 100 random programs each of length 4, 6, 8. Five seeds, median [min, max]. Predictions in `arch_d.py`.

| variant | training error | length 4 (in range) | length 6 | length 8 | solved at 6, at 8 |
|---|---|---|---|---|---|
| D1 clock, learned keys | 3.6e-05 | 3.9e-05 [2.4e-05, 4.4e-05] | 4.5e-02 [3.7e-02, 5.2e-02] | 7.7e-02 [6.6e-02, 8.6e-02] | 0/5, 0/5 |
| D2 reg0, learned keys | 3.7e-05 | 4.0e-05 [2.4e-05, 4.7e-05] | 9.4e-02 [8.3e-02, 1.1e-01] | 1.8e-01 [1.5e-01, 3.6e-01] | 0/5, 0/5 |
| D3 reg0, sinusoidal keys | 3.5e-05 | 3.6e-05 [2.6e-05, 4.5e-05] | 8.3e-02 [9.9e-04, 9.4e-02] | 2.1e-01 [1.1e-01, 5.3e-01] | 1/5, 0/5 |
| D4 regrot, sinusoidal keys | 3.3e-05 | 3.4e-05 [2.1e-05, 4.4e-05] | 7.0e-02 [2.2e-05, 8.7e-02] | 1.2e-01 [1.5e-02, 1.9e-01] | 2/5, 0/5 |
| D5 exact-shift register, sinusoidal keys | 3.2e-05 | 3.3e-05 [2.7e-05, 4.3e-05] | 3.1e-05 [2.7e-05, 4.5e-05] | 3.7e-05 [3.0e-05, 5.2e-05] | 5/5, 5/5 |
