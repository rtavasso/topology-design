"""Extract compact arrays from raw run files (independent of analyze.py)."""
import gzip, json, glob, pickle, sys
import numpy as np
from multiprocessing import Pool
ROOT = '/home/claude/topology-design/results'
OUT = '/tmp/claude-0/-home-claude/3b1f2496-e305-5fa9-942f-5eec210bff0b/scratchpad/verify_e34'

def f2(x):
    return np.nan if x is None else float(x)

def load(fn):
    d = json.load(gzip.open(fn))
    c, s, t = d['config'], d['summary'], d['trajectory']
    r = dict(file=fn.split('/')[-1], arm=c['arm'], lr=c['lr'], wd=c['weight_decay'], steps=c['steps'],
             tag=c['tag'], seed=c['model_seed'], alphas=c.get('alphas'), blr=c.get('branch_lr_mult'),
             bwd=c.get('branch_wd_mult'), summary=s)
    r['step'] = np.array([e['step'] for e in t])
    for k in ('train_acc', 'test_acc', 'train_loss', 'test_loss'):
        r[k] = np.array([f2(e[k]) for e in t])
    for k in ('gen', 'mem'):
        r[k] = np.array([[f2(v) for v in e[k]] for e in t])
    if 'abl_block_test_acc' in t[0]:
        r['abl_test'] = np.array([[f2(v) for v in e['abl_block_test_acc']] for e in t])
    return r

if __name__ == '__main__':
    for exp in ('e3', 'e4'):
        fs = sorted(glob.glob(f'{ROOT}/kaggle_{exp}/results/*.json.gz'))
        with Pool(8) as p:
            runs = p.map(load, fs)
        pickle.dump(runs, open(f'{OUT}/{exp}.pkl', 'wb'))
        print(exp, len(runs))
