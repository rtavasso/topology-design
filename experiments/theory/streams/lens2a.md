# Lens 2 — order (EXPLORATORY)

Two teacher primitives, d = 6, 4 passes; relative commutator ‖[A, B]‖ / (‖A‖‖B‖) = 0.53. Predictions in `lens2_order.py`.

**2.2** error floor of a student blind to order, and of the ordered student (median of three seeds):

| step size eps | bag student | ratio to the previous row | ordered student |
|---|---|---|---|
| 0.05 | 5.71e-04 | – | 4.1e-31 |
| 0.1 | 2.37e-03 | 4.1 | 2.2e-31 |
| 0.2 | 9.97e-03 | 4.2 | 1.0e-31 |
| 0.4 | 4.15e-02 | 4.2 | 1.6e-28 |

**2.3** share of the gate's gradient that depends on the order of the program (mean of five draws):

| step size eps | order-dependent share | ratio to the previous row |
|---|---|---|
| 0.05 | 0.003 | – |
| 0.1 | 0.008 | 2.72 |
| 0.2 | 0.025 | 3.15 |
| 0.4 | 0.084 | 3.32 |
