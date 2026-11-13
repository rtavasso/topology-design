"""Properties of beta (DESIGN §4, §7) under structured model error, and the share of the exact update that is a
rule-identity lookup. Reviewer's computation; H8, rho = 0.5, k in {2,3,4}, full prior over 255 worlds (as power.md Q1)."""
import random, sys, time, pickle
import numpy as np
from core import *
from reftrace import *

n = int(sys.argv[1]) if len(sys.argv) > 1 else 600
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
rng = random.Random(seed)
D = []          # decisions


def temp(v, T):
    if v is None:
        return None
    w = np.where(v > 0, v ** (1.0 / T), 0.0)
    return w / w.sum()


def hook(ins, v, b, P, A, rh, R):
    e0, e1 = ins.estart[v], ins.estart[v + 1]
    rules = ins.er[e0:e1]
    ne = e1 - e0
    q, m, frozen = R["q"], R["m"], R["frozen"]
    if q is None:
        return
    inP = np.array([(P >> int(r)) & 1 for r in rules], dtype=bool)
    inA = np.array([(A >> int(r)) & 1 for r in rules], dtype=bool)
    R["inP"], R["inA"] = inP, inA
    # --- wrong prior over worlds (model learned rho' instead of 0.5): exact update / mask under that prior
    for rho in (0.35, 0.75):
        wts = prior_vec(rho)
        R[f"q_rho{rho}"] = ins.mix(v, ins.wmask(v, b, P, A), wts)
        R[f"m_rho{rho}"] = masked(ins.mix(v, ins.wmask(v, b), wts), rh)
    # --- world-belief temperature (prior belief over worlds too sharp / too flat towards large worlds): weight by size
    for g in (-0.5, 0.5):                                   # prior tilted by exp(g * |W|)
        wts = ins.prior * np.exp(g * WSIZE)
        R[f"q_tilt{g}"] = ins.mix(v, ins.wmask(v, b, P, A), wts)
        R[f"m_tilt{g}"] = masked(ins.mix(v, ins.wmask(v, b), wts), rh)
    # --- action-level temperature
    for T in (0.5, 0.7, 1.5, 2.0, 3.0):
        R[f"q_T{T}"] = temp(q, T)
        R[f"m_T{T}"] = temp(m, T)
    # --- one-hot argmax
    for nm, x in (("q", q), ("m", m)):
        if x is not None:
            oh = np.zeros(ne); oh[int(np.argmax(x))] = 1.0
            R[f"{nm}_argmax"] = oh
    # --- uniform leak over valid actions, leak to invalid actions, canonical-action bias
    uni = np.ones(ne) / ne
    for eps in (0.1, 0.2, 0.3):
        R[f"q_uni{eps}"] = (1 - eps) * q + eps * uni
        R[f"q_inv{eps}"] = (1 - eps) * q                      # eps of the mass on actions outside the valid set
        can = np.zeros(ne); can[int(np.nonzero(q > 0)[0][0])] = 1.0   # first supported action in rule order
        R[f"q_can{eps}"] = (1 - eps) * q + eps * can
        if m is not None:
            canm = np.zeros(ne); canm[int(np.nonzero(m > 0)[0][0])] = 1.0
            R[f"m_can{eps}"] = (1 - eps) * m + eps * canm
            R[f"m_uni{eps}"] = (1 - eps) * m + eps * uni
    # --- rule-identity lookups on the frozen belief: remove rejected rules, boost accepted rules
    for alpha in (0.5, 1.0, 2.0):
        x = frozen * np.exp(alpha * inP) * (~inA)
        R[f"lookup{alpha}"] = renorm(x)
    # uniform over valid actions with rejected rules removed (no belief at all)
    R["uni_rm"] = renorm(np.ones(ne) * (~inA))


t0 = time.time()
for it in range(n):
    ins = Inst(rng)
    recs, summ = run(ins, "q", order="power", hook=hook)
    for j, d in enumerate(recs):
        d["inst"] = it
        d["prev"] = recs[j - 1] if j > 0 else None
    D += recs
print(f"n={n} seed={seed} {(time.time()-t0)/n:.2f}s/inst; decisions {len(D)}")
post = [d for d in D if d["calls"] > 0 and d["m"] is not None]
sep = np.array([tv(d["q"], d["m"]) for d in post])
print(f"post-verdict decisions {len(post)} ({len(post)/n:.2f}/inst); TV>0.05 {np.mean(sep>0.05):.3f}; >0.10 {np.mean(sep>0.10):.3f}; "
      f">0.20 {np.mean(sep>0.20):.3f}; mean {sep.mean():.4f}; mean on informative {sep[sep>0.05].mean():.3f}")
den = sum(float(((d["q"] - d["m"]) ** 2).sum()) for d in post)
print(f"signal energy sum|q-m|^2 = {den:.2f} ({den/n:.4f} per instance)")


def r2(recs, key):
    a = b = 0.0
    for d in recs:
        p = d.get(key)
        if p is None:
            continue
        a += float(((d["q"] - p) ** 2).sum()); b += float(((d["q"] - d["m"]) ** 2).sum())
    return 1 - a / b if b > 0 else float("nan")


print("\n== beta (and share of squared distance closed, 1 - |q-p|^2/|q-m|^2) of candidate 'models', decisions after a verdict ==")
keys = ["q", "m", "rm", "qA", "qP", "qP_rm", "uni_rm", "lookup0.5", "lookup1.0", "lookup2.0",
        "m_T0.5", "m_T0.7", "m_T1.5", "m_T2.0", "m_argmax", "m_can0.1", "m_can0.3", "m_uni0.1", "m_uni0.3",
        "m_rho0.35", "m_rho0.75", "m_tilt-0.5", "m_tilt0.5",
        "q_T0.5", "q_T0.7", "q_T1.5", "q_T2.0", "q_T3.0", "q_argmax", "q_uni0.1", "q_uni0.2", "q_uni0.3",
        "q_inv0.1", "q_inv0.2", "q_inv0.3", "q_can0.1", "q_can0.3", "q_rho0.35", "q_rho0.75", "q_tilt-0.5", "q_tilt0.5"]
for k in keys:
    bt, _, nn = beta(post, k)
    print(f"{k:12s} beta {bt:6.3f}   closed {r2(post, k):6.3f}   n {nn}")

print("\n== where the signal is ==")
for name, sel in (("last verdict = rejection (same state)", lambda d: d["last"] == "rej"),
                  ("last verdict = accept (state changed)", lambda d: d["last"] == "acc"),
                  ("no rejection so far (A empty)", lambda d: d["A"] == 0),
                  ("some rejection so far (A non-empty)", lambda d: d["A"] != 0),
                  ("true world can no longer finish", lambda d: not d["true_live"])):
    sub = [d for d in post if sel(d)]
    e = sum(float(((d["q"] - d["m"]) ** 2).sum()) for d in sub)
    s2 = np.array([tv(d["q"], d["m"]) for d in sub]) if sub else np.array([0.0])
    print(f"{name:42s} n {len(sub):5d} ({len(sub)/len(post):.2f})  energy share {e/den:.3f}  mean TV {s2.mean():.3f}  "
          f"beta[rm] {beta(sub,'rm')[0]:.3f} beta[qP] {beta(sub,'qP')[0]:.3f} beta[qP_rm] {beta(sub,'qP_rm')[0]:.3f} beta[qA] {beta(sub,'qA')[0]:.3f}")

# --- the non-trivial remainder: exact update against the rule mask
den_rm = sum(float(((d["q"] - d["rm"]) ** 2).sum()) for d in post if d["rm"] is not None)
sep_rm = np.array([tv(d["q"], d["rm"]) for d in post if d["rm"] is not None])
den_prm = sum(float(((d["q"] - d["qP_rm"]) ** 2).sum()) for d in post if d["qP_rm"] is not None)
sep_prm = np.array([tv(d["q"], d["qP_rm"]) for d in post if d["qP_rm"] is not None])
print(f"\nexact update against rule mask (frozen belief, all actions of rejected rules removed): energy {den_rm:.2f} "
      f"({den_rm/den:.3f} of sum|q-m|^2); TV>0.05 {np.mean(sep_rm>0.05):.3f}; mean TV {sep_rm.mean():.4f}")
print(f"exact update against [condition on accepted rules + rule mask]: energy {den_prm:.2f} ({den_prm/den:.3f}); "
      f"TV>0.05 {np.mean(sep_prm>0.05):.3f}; mean TV {sep_prm.mean():.4f}")

# --- a model-relative statistic on rejection-following decisions: update beyond masking ONE action of the model's OWN
#     previous distribution at the same state.  dq = q_t - mask(q_{t-1}, a);  dp = p_t - mask(p_{t-1}, a)
def step_stat(pkey_fn):
    num = dn = 0.0; cnt = 0
    for d in D:
        if d["last"] != "rej" or d["prev"] is None:
            continue
        pv = d["prev"]
        a = pv["choice"]
        q0 = masked(pv["q"], [a]); 
        p_now, p_prev = pkey_fn(d), pkey_fn(pv)
        if q0 is None or p_now is None or p_prev is None:
            continue
        p0 = masked(p_prev, [a])
        if p0 is None:
            continue
        dq = d["q"] - q0
        num += float((p_now - p0) @ dq); dn += float(dq @ dq); cnt += 1
    return num / dn if dn > 0 else float("nan"), dn, cnt

print("\n== model-relative step statistic on rejection-following decisions (zero = masking one action of the model's own previous distribution) ==")
def own_mask(d):          # a model with a wrong prior (rho 0.75) that only masks: its distribution = m under its prior
    return d.get("m_rho0.75")
for name, fn in (("exact q", lambda d: d["q"]),
                 ("mask-only m (exact prior)", lambda d: d["m"]),
                 ("mask-only, wrong prior rho 0.75", lambda d: d.get("m_rho0.75")),
                 ("mask-only, sharpened T 0.5", lambda d: d.get("m_T0.5")),
                 ("rule mask on frozen belief", lambda d: d.get("rm")),
                 ("exact update under wrong prior rho 0.75", lambda d: d.get("q_rho0.75")),
                 ("exact q, tempered T 2", lambda d: d.get("q_T2.0")),
                 ("A-only posterior", lambda d: d.get("qA"))):
    s, dn, cnt = step_stat(fn)
    print(f"{name:42s} beta_step {s:6.3f}  (energy {dn:.2f}, n {cnt})")
pickle.dump([{k: v for k, v in d.items() if k != "prev"} for d in D], open(f"dec_{seed}_{n}.pkl", "wb"))
