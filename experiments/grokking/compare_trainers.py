"""Compare per-run endpoints between two result dirs with the same run_ids (e.g. CPU vs batched-MPS).

Usage: PYTHONPATH=src python experiments/grokking/compare_trainers.py results/grokking/lr_tune results/grokking/lr_tune_mps
"""
import glob, json, os, sys
import numpy as np

a_dir, b_dir = sys.argv[1], sys.argv[2]
rows = []
for fa in sorted(glob.glob(os.path.join(a_dir, "*.json"))):
    fb = os.path.join(b_dir, os.path.basename(fa))
    if not os.path.exists(fb):
        continue
    ra, rb = json.load(open(fa)), json.load(open(fb))
    sa, sb = ra["summary"], rb["summary"]
    rows.append((ra["run_id"], sa["t_grok"], sb["t_grok"], sa["final_test_acc"], sb["final_test_acc"],
                 ra["trajectory"][4]["train_loss"] if len(ra["trajectory"]) > 4 else None,
                 rb["trajectory"][4]["train_loss"] if len(rb["trajectory"]) > 4 else None))
if not rows:
    sys.exit("no overlapping runs")
same_grok = np.mean([(x[1] is None) == (x[2] is None) for x in rows])
both = [(x[1], x[2]) for x in rows if x[1] is not None and x[2] is not None]
print(f"{len(rows)} paired runs; grok/no-grok agreement {same_grok:.2f}")
if both:
    d = np.array([b - a for a, b in both], dtype=float)
    print(f"t_grok difference (b - a) over {len(both)} runs that grokked in both: median {np.median(d):.0f}, "
          f"IQR [{np.percentile(d, 25):.0f}, {np.percentile(d, 75):.0f}] steps")
early = [abs(x[5] - x[6]) / max(abs(x[5]), 1e-9) for x in rows if x[5] is not None and x[6] is not None]
print(f"relative train-loss difference at step 1000: median {np.median(early):.2e}, max {np.max(early):.2e}")
for x in rows:
    print(f"  {x[0]}: t_grok {x[1]} vs {x[2]}, final test {x[3]:.3f} vs {x[4]:.3f}")
