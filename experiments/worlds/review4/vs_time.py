# Review 4: cost of exact "search" (version space over all 297,845 functions of l<=6) per context, CPU numpy.
import numpy as np, time
d=np.load("f6.npz"); T=d["T"]; rng=np.random.default_rng(0)
t=time.time(); n=200
for _ in range(n):
    f=T[rng.integers(len(T))]; ctx=rng.choice(49,8,replace=False); m=(T[:,ctx]==f[ctx]).all(1); m.sum()
print(f"{(time.time()-t)/n*1e3:.1f} ms per k=8 context, exact version space over l<=6")
