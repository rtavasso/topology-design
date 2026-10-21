# Derivation checks for DESIGN.md §2

**Filter:** $\tilde g_t = g_t + \lambda \bar g_t$, $\bar g_t = \alpha \bar g_{t-1} + (1-\alpha) g_t$, with α = 0.98 and λ = 2 (the Grokfast-EMA defaults).

**Frequency response:** $H(\omega) = 1 + \lambda(1-\alpha)/(1-\alpha e^{-i\omega})$, so $H(0) = 1+\lambda = 3$.

**Numerical check** (20,000 steps; script inline in the session, reproduced here):

| component | definition | cumulative gain Σỹ/Σg | rms gain |
|---|---|---|---|
| transient burst (memorization-like) | $e^{-t/150}$ | 3.000 | 2.651 |
| slow drift (generalization-like) | constant $10^{-3}$ | 2.995 | 2.996 |
| oscillation, period 300 (instability-like) | $0.5\sin(2\pi t/300)$ | — (zero mean) | 2.209 |

**Gain by period:**

| period | 10 | 50 | 300 | 3000 |
|---|---|---|---|---|
| \|H\| | 1.02 | 1.11 | 2.21 | 2.99 |

**Reading:**
- The burst's total displacement is amplified exactly as much as the drift's.
- Only step-to-step jitter, at periods ≲ 50 steps, is relatively suppressed.

```python
import numpy as np
a, lam, T = 0.98, 2.0, 20000
t = np.arange(T)
def filt(g):
    e, out = 0.0, np.empty_like(g)
    for i, x in enumerate(g):
        e = a * e + (1 - a) * x
        out[i] = x + lam * e
    return out
for name, g in [("burst", np.exp(-t / 150.)), ("drift", 1e-3 * np.ones(T)), ("osc", 0.5 * np.sin(2 * np.pi * t / 300.))]:
    y = filt(g)
    print(name, y.sum() / g.sum() if abs(g.sum()) > 1e-9 else None, np.sqrt((y ** 2).mean() / (g ** 2).mean()))
for P in (10, 50, 300, 3000):
    w = 2 * np.pi / P
    print(P, abs(1 + lam * (1 - a) / (1 - a * np.exp(-1j * w))))
```

## Corrections from the reviews of revision 1 (21 Oct 2026)

**The exact statement is selection, not exclusion.** Any linear time-invariant filter multiplies every non-oscillating component's integrated displacement by the same $H(0)$, so it cannot change their ratio. A pure high-pass filter ($H(0)=0$) does remove net displacement, for all components alike.

**Behaviour through AdamW** (technical reviewer, `review5tech/adam_toy.py` in the session scratchpad):
- λ = 2 low-pass emphasis raised a burst's displacement by about 9% and a drift's by about 0.3%, not 3×. Adam removes stable per-coordinate gains.
- Decoupled weight decay (wd = 1) shrank a burst's displacement to about 0.001 while a drift's stayed at about 1.0. Decay is itself a time filter with τ = 1/(lr·wd).

**On the real network** (technical reviewer, `review5tech/realrun.py`; pk012000, seed 500, shadow optimizers along one trajectory):
- Pre-Adam time routing raised the long branch's update norm 2–3×, with cosine 0.91–0.96 to the unrouted update.
- Two-sided routing gave the short branch an update with cosine −0.43 to −0.86 to its unrouted update.
- Agreement-mask density was 4–15%, near the 7% expected by chance.

**Biased split statistic** (`review5tech`): the original Σ has a non-zero random-split null, (√f − √(1−f))·[cos(−g,∇gen) − cos(−g,∇mem)]; predicted 0.0864, measured 0.0873. It is replaced by the share-of-projection Σ′.

**Two-branch Shapley** (`review5tech/shap2.py`): with no branch reading another's writes, total-effect equals direct attribution to about 1e-7.
