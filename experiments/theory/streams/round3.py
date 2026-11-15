"""Round 3 - the one round-2 prediction that failed. EXPLORATORY.

R4 (i) predicted the sharp-gate loop's typical gain would match single-stream loops (within 25 %); it was 50 % higher
(7.75 against 5.17). Two candidate causes: the gate's own Jacobian term near region boundaries, or the composition of
different streams along an itinerary.

Prediction written before the run (15 Nov 2026): composition is most of it. With each input's itinerary frozen (the
stream picked at each pass held fixed, so the gate contributes no derivative), the median gain is at least 6.2, i.e. at
least 60 % of the excess over single streams remains. And among inputs far from every boundary (smallest margin above
5 tau) the soft-gate gain and the frozen-itinerary gain agree within 5 %.
"""
import torch
from torch.func import jacrev, vmap
import lens3_partition as L3

torch.set_num_threads(2)
B, V, T, x = L3.make(0)
x = x[:1000]
tau = 0.05


def soft(h):
    return L3.loop(h, B, V, tau)[0]


def frozen(h, picks):
    for t in range(L3.N):
        h = h + L3.EPS * torch.tanh(B[picks[t]] @ h)
    return h


picks = vmap(lambda h: L3.loop(h, B, V, tau)[2])(x)
margin = vmap(lambda h: L3.loop(h, B, V, tau)[1])(x)
gs = torch.linalg.matrix_norm(vmap(jacrev(soft))(x), 2)
gf = torch.stack([torch.linalg.matrix_norm(jacrev(frozen)(x[i], picks[i]), 2) for i in range(len(x))])
far = margin > 5 * tau
same = torch.tensor([len(set(p.tolist())) == 1 for p in picks])
print("# Round 3 — where the sharp gate's extra gain comes from (EXPLORATORY)\n\nPrediction in `round3.py`.\n")
print(f"- median gain, soft gate at tau = {tau}: {gs.median().item():.2f}; with the itinerary frozen: {gf.median().item():.2f}; single streams alone (round 2): 5.17")
print(f"- inputs far from every boundary ({far.double().mean().item():.0%}): soft {gs[far].median().item():.2f}, frozen {gf[far].median().item():.2f}")
print(f"- inputs near a boundary ({(~far).double().mean().item():.0%}): soft {gs[~far].median().item():.2f}, frozen {gf[~far].median().item():.2f}")
print(f"- itineraries that use one stream throughout: {same.double().mean().item():.0%} of inputs, frozen gain {gf[same].median().item() if same.any() else float('nan'):.2f}; itineraries that switch: frozen gain {gf[~same].median().item():.2f}")
