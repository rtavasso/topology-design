import os
"""Independent E3 analysis from extracted raw trajectories."""
import pickle, sys, itertools, collections
import numpy as np
from scipy.stats import kendalltau, binomtest
D = os.environ.get('VERIFY_DIR', 'verify_e34')  # arrays written by e34_extract.py
R = pickle.load(open(f'{D}/e3.pkl', 'rb'))
B0, B1 = [1, 2, 3], [4, 5, 6]
RHOS = [0.25, 0.5, 1, 2, 4]

def parse(tag):
    a, g, r = tag.split('-')
    arm = {'wE3': 'main', 'wE3lr': 'lr', 'wE3wd': 'wd'}[a]
    return g, arm, float(r[3:])

def cancels(x0, x1):
    return abs(x0 + x1) < 0.5 * (abs(x0) + abs(x1))

def sustained(step, acc, th=0.95, dur=2000):
    # first step from which acc stays >= th for >= dur steps (until end if end comes first -> needs dur)
    T = step[-1]
    for i in range(len(step)):
        if acc[i] < th: continue
        j = i
        while j < len(step) and acc[j] >= th: j += 1
        end = step[j - 1] if j == len(step) else step[j]
        if (j == len(step) and step[-1] - step[i] >= dur) or (j < len(step) and step[j] - step[i] >= dur):
            return step[i]
    return None

def measures(r, mem_total='all', gen_mode='ratio_of_means'):
    st, T = r['step'], r['steps']
    gen, mem = r['gen'], r['mem']
    w = st >= T - 2000
    g0, g1 = gen[w][:, B0].sum(1), gen[w][:, B1].sum(1)
    if gen_mode == 'ratio_of_means':
        G0, G1 = g0.mean(), g1.mean()
        sg = G0 / (G0 + G1)
    else:  # median of per-eval shares
        G0, G1 = g0.mean(), g1.mean()
        sg = np.median(g0 / (g0 + g1))
    tot = mem.sum(1) if mem_total == 'all' else mem[:, 1:].sum(1)
    k = int(np.argmax(tot))
    M0, M1 = mem[k, B0].sum(), mem[k, B1].sum()
    sm = M0 / (M0 + M1)
    return dict(sg=sg, sm=sm, G0=G0, G1=G1, M0=M0, M1=M1, t_mem=int(st[k]))

def eligible(r, m, grok='first'):
    st, T, s = r['step'], r['steps'], r['summary']
    if grok == 'first':
        tg = s['t_grok']
    else:
        tg = sustained(st, r['test_acc'])
    ok_grok = tg is not None and tg <= T - 5000
    w = st >= T - 2000
    ok_train = bool((r['train_acc'][w] >= 0.9).all())
    ok_div = s['diverged_at'] is None and np.isfinite(r['train_loss']).all()
    ok_canc = not cancels(m['G0'], m['G1']) and not cancels(m['M0'], m['M1'])
    return ok_grok and ok_train and ok_div and ok_canc, dict(grok=ok_grok, train=ok_train, div=ok_div, canc=ok_canc, tg=tg)

def lg(s):
    s = np.clip(s, 0.02, 0.98); return np.log(s / (1 - s))

def build(mem_total='all', gen_mode='ratio_of_means', grok='first'):
    rows = []
    for r in R:
        g, arm, rho = parse(r['tag'])
        m = measures(r, mem_total, gen_mode)
        e, why = eligible(r, m, grok)
        rows.append(dict(g=g, arm=arm, rho=rho, seed=r['seed'], el=e, why=why, **m, r=r))
    return rows

def perm_tau(rows, key, n=10000, two_sided=False, rng=np.random.default_rng(0)):
    x = np.array([q['rho'] for q in rows]); y = np.array([key(q) for q in rows])
    seeds = np.array([q['seed'] for q in rows])
    tau = kendalltau(x, y).statistic
    groups = [np.where(seeds == s)[0] for s in np.unique(seeds)]
    cnt = 0
    for _ in range(n):
        xp = x.copy()
        for gi in groups: xp[gi] = x[rng.permutation(gi)]
        t = kendalltau(xp, y).statistic
        cnt += (abs(t) >= abs(tau)) if two_sided else (t >= tau)
    return tau, (cnt + 1) / (n + 1)

def report(rows, label, nperm=10000):
    print(f'=== {label} ===')
    for g in ('asym', 'sym'):
        for arm in ('main', 'lr', 'wd'):
            out = []
            for rho in RHOS:
                c = [q for q in rows if q['g'] == g and q['arm'] == arm and q['rho'] == rho]
                if not c: continue
                el = [q for q in c if q['el']]
                fails = collections.Counter(k for q in c for k, v in q['why'].items() if v is False)
                mg = np.median([q['sg'] for q in el]) if el else np.nan
                mm = np.median([q['sm'] for q in el]) if el else np.nan
                out.append(f'rho{rho}: n={len(el)} gen={mg:.2f} mem={mm:.2f} fail={dict(fails)}')
            print(g, arm, '\n   ' + '\n   '.join(out))
    for g in ('asym', 'sym'):
        cells = {rho: [q for q in rows if q['g'] == g and q['arm'] == 'main' and q['rho'] == rho and q['el']] for rho in RHOS}
        test = [rho for rho in RHOS if len(cells[rho]) >= 6]
        rr = [q for rho in test for q in cells[rho]]
        print(f'{g} testable main cells: {test} (n runs {len(rr)})')
        if len(test) >= 3:
            tg, pg = perm_tau(rr, lambda q: lg(q['sg']), nperm)
            tm, pm = perm_tau(rr, lambda q: lg(q['sm']), nperm)
            print(f'  P1 tau_gen={tg:.3f} p={pg:.4g}  tau_mem={tm:.3f} p={pm:.4g}')
            td, pd = perm_tau(rr, lambda q: lg(q['sg']) - lg(q['sm']), nperm, two_sided=True)
            med = {rho: np.median([lg(q['sg']) - lg(q['sm']) for q in cells[rho]]) for rho in test}
            ref = med.get(1, np.nan)
            print(f'  P2 tau_d={td:.3f} p(2s)={pd:.4g}; median d per cell ' + ', '.join(f'{k}:{v:+.2f}' for k, v in med.items()) +
                  '; shifts vs rho1 ' + ', '.join(f'{k}:{v - ref:+.2f}' for k, v in med.items()))
        # P3 + diagnostics
        for arm in ('lr', 'wd', 'main'):
            a = {q['seed']: q for q in rows if q['g'] == g and q['arm'] == arm and q['rho'] == 0.25 and q['el']}
            b = {q['seed']: q for q in rows if q['g'] == g and q['arm'] == arm and q['rho'] == 4 and q['el']}
            pairs = sorted(set(a) & set(b))
            s = ''
            for key in ('sg', 'sm'):
                dif = [lg(b[k][key]) - lg(a[k][key]) for k in pairs]
                npos, nneg = sum(x > 0 for x in dif), sum(x < 0 for x in dif)
                p = binomtest(npos, npos + nneg, 0.5).pvalue if npos + nneg else np.nan
                p1 = binomtest(npos, npos + nneg, 0.5, alternative='greater').pvalue if npos + nneg else np.nan
                es = (np.median([lg(q[key]) for q in b.values()]) - np.median([lg(q[key]) for q in a.values()])) if a and b else np.nan
                s += f' {key}: +{npos}/-{nneg} p2={p:.3g} p1={p1:.3g} effect(med logit4-med logit1/4)={es:+.2f};'
            print(f'  {arm}: pairs={len(pairs)}{s}')

if __name__ == '__main__':
    nperm = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    report(build(), 'primary: first-crossing grok, mem total over all 7 comps, gen ratio of means', nperm)
