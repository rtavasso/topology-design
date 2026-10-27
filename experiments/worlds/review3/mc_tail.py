# Review 3: exact bias of -log floor(q̂) vs -log floor(q), q̂ ~ Binomial(M,q)/M, as a function of the model's q for the true value.
import numpy as np
from scipy.stats import binom
for M in (1024,8192):
  for eps in (1e-3,1e-2):
    fl=lambda q:(1-eps)*q+eps/7; n=np.arange(M+1)
    row=[]
    for q in (1e-4,3e-4,1e-3,3e-3,1e-2,0.05,0.2,0.5):
        p=binom.pmf(n,M,q); row.append(f"q={q:g}:{(p*-np.log(fl(n/M))).sum()+np.log(fl(q)):+.3f}")
    print(f"M={M} floor eps={eps}: "+"  ".join(row))
