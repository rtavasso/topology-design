"""Independent E4 stability analysis."""
import pickle, numpy as np, collections, sys
D = '/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/verify_e34'
R = pickle.load(open(f'{D}/e4.pkl', 'rb'))
LRS = [0.0005, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 0.064]
ARMS = ['pk0000', 'pk0100', 'pk0120', 'pk0123']
PAR = ['std', 'P1', 'P2']
def stable(r, mode='observed'):
    st, ta = r['step'], r['train_acc']
    ok = np.where(ta >= 0.99)[0]
    if len(ok) == 0: return False
    tf = st[ok[0]]
    assert tf == r['summary']['t_fit']
    if mode == 'full' and tf + 6000 > st[-1]: return False
    w = (st >= tf) & (st <= tf + 6000)
    if not np.all(np.isfinite(r['train_loss'][w])): return False
    if r['summary']['diverged_at'] is not None and r['summary']['diverged_at'] <= tf + 6000: return False
    return bool((ta[w] >= 0.9).all())
def maps(mode='observed'):
    M = {}
    for r in R:
        p = r['tag'].split('-')[1]
        M.setdefault((p, r['arm'], r['wd'], r['lr']), []).append(stable(r, mode))
    return M
def lrstar(M, p, arm, wd):
    s = [lr for lr in LRS if sum(M[(p, arm, wd, lr)]) >= 2]
    return max(s) if s else None
def idx(lr): return None if lr is None else LRS.index(lr)
if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else 'observed'
    M = maps(mode)
    for p in PAR:
        print(f'=== {p} ({mode}) — stable seeds per cell, lr 5e-4 .. 6.4e-2')
        for wd in (0.3, 1.0, 3.0):
            for arm in ARMS:
                row = ''.join(str(sum(M[(p, arm, wd, lr)])) for lr in LRS)
                ls = lrstar(M, p, arm, wd)
                nst = sum(sum(M[(p, arm, wd, lr)]) >= 2 for lr in LRS)
                print(f'  wd={wd:<3} {arm} C{ARMS.index(arm)+1}: {row}  lr*={ls}  #stable cells={nst}  frac stable runs={sum(sum(M[(p,arm,wd,lr)]) for lr in LRS)/24:.2f}')
    print('--- predictions at wd=1')
    for p in PAR:
        a, b = lrstar(M, p, 'pk0000', 1.0), lrstar(M, p, 'pk0123', 1.0)
        ia = idx(a); ib = idx(b) if b is not None else -1  # no stable cell = below grid
        print(f'{p}: lr*(C1)={a} lr*(C4)={b} gap(steps)={ia - ib if ia is not None else None}')
