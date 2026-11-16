"""Stage 3 of experiments/theory/worlds/DESIGN.md: mechanism checks for the arms that recombine. EXPLORATORY.

select, coverage 9, six splits, each arm at its stage-2 configuration, retrained.
(a) One pass from fresh states: does the branch with the larger pooled weight belong to the primitive the teacher
    applies? On seen worlds; on held-out worlds where trained comparisons force the choice; on held-out worlds where
    they do not (there, only an assumption about the form of the bids gives the right answer).
(b) tok_streams: which stream each token uses; removing one stream and measuring the error on seen worlds that contain
    its token against worlds that do not.
Expectation written before running: accuracy on forced held-out decisions close to accuracy on seen worlds (within
0.03) for both arms; on unforced held-out decisions lower but well above chance (above 0.85); in tok_streams each token
has its own stream in at least 5 of 6 splits, and removing a stream raises the error at least ten times more on the
worlds of its token than on the others.
"""
import statistics
import sys
import torch
from stage0 import cfg
from stage2 import best_cfg
from worlds import err, run

med = statistics.median

if __name__ == "__main__":
    print("| arm | decision accuracy, one pass: seen worlds | held out, forced by trained comparisons | held out, not forced | share of held-out decisions that are forced |\n|---|---|---|---|---|", flush=True)
    notes = []
    for arm in sys.argv[1:]:
        a, c = best_cfg("select", arm)
        acc = {"seen": [], "forced": [], "free": [], "share": []}
        one2one, ratios = 0, []
        for s in range(6):
            out, S, T, tr, te = run(cfg("select", a, 9, s, **c), s, ret_model=True)
            T.trained = torch.zeros(6, 6)
            for u, v in tr.tolist():
                T.trained[u, v] = T.trained[v, u] = 1.0
            with torch.no_grad():
                for name, ws in (("seen", tr), ("held", te)):
                    x = torch.randn(len(ws), 512, 6, generator=torch.Generator().manual_seed(31 + s))
                    S.forward(ws, x, 1, T, record=True)
                    right = (S.pools[0].argmax(-1) == T.choice(ws, x))
                    if name == "seen":
                        acc["seen"].append(right.double().mean().item())
                    else:
                        det = T.determined(ws, x)
                        acc["forced"].append(right[det].double().mean().item()); acc["free"].append(right[~det].double().mean().item())
                        acc["share"].append(det.double().mean().item())
                if a == "tok_streams":
                    x = torch.randn(len(tr), 256, 6, generator=torch.Generator().manual_seed(77 + s))
                    tgt = T.run(tr, x, 4)
                    base = ((S.forward(tr, x, 4, T) - tgt) ** 2).sum((1, 2))
                    assign = torch.softmax(S.p["E"] @ S.p["G"].T, -1).argmax(-1)          # token -> its stream
                    one2one += int(len(set(assign.tolist())) == 6)
                    for p in range(6):
                        k = assign[p].item()
                        keep = S.p["Vs"][k].clone(); S.p["Vs"][k].zero_()
                        d = ((S.forward(tr, x, 4, T) - tgt) ** 2).sum((1, 2)) - base
                        S.p["Vs"][k].copy_(keep)
                        has = (tr == p).any(1)
                        ratios.append((d[has].mean() / d[~has].mean().clamp(min=1e-12)).item())
        print(f"| {arm} | {med(acc['seen']):.3f} [{min(acc['seen']):.3f}, {max(acc['seen']):.3f}] | {med(acc['forced']):.3f} [{min(acc['forced']):.3f}, {max(acc['forced']):.3f}] | "
              f"{med(acc['free']):.3f} [{min(acc['free']):.3f}, {max(acc['free']):.3f}] | {med(acc['share']):.2f} |", flush=True)
        if a == "tok_streams":
            notes.append(f"{arm}: each token has its own stream in {one2one}/6 splits; removing a token's stream raises the error on seen worlds containing that token "
                         f"{med(ratios):.0f} times more than on the others (median over 36 removals; minimum {min(ratios):.1f}).")
    print()
    for n in notes:
        print(n)
