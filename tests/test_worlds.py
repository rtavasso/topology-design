"""Exact statements behind experiments/theory/worlds/DESIGN.md."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "worlds"))
from worlds import DEF, Model, Teacher, all_worlds, err, graph_ok, references, sample_worlds  # noqa: E402

torch.set_default_dtype(torch.float64)
P = DEF["P"]


def test_world_sets_are_disjoint_complete_and_meet_the_graph_conditions():
    for cover in (7, 9, 11, 13):
        for seed in range(6):
            tr, te = sample_worlds(P, cover, seed)
            a, b = {tuple(r) for r in tr.tolist()}, {tuple(r) for r in te.tolist()}
            assert len(a) == cover and not (a & b) and (a | b) == set(all_worlds(P))
            assert graph_ok(P, sorted(a))
            assert {p for w in b for p in w} <= {p for w in a for p in w}      # every primitive is trained on
    tr, te = sample_worlds(P, 15, 0)
    assert len(tr) == 15 and len(te) == 0


def test_training_worlds_determine_what_a_held_out_world_should_do():
    # select: the rule in world {a, b} is the sign of <w_a - w_b, h>. On a connected set of training worlds every
    # held-out difference w_a - w_b is a signed sum of trained differences (rank of the signed incidence matrix P - 1).
    # sum: the update in world {a, b} is f_a + f_b. The f's are determined by the trained sums exactly when the
    # unsigned incidence matrix has rank P (connected and not bipartite).
    for cover in (7, 9, 11, 13):
        for seed in range(6):
            tr, te = sample_worlds(P, cover, seed)
            signed, unsigned = torch.zeros(len(tr), P), torch.zeros(len(tr), P)
            for i, (a, b) in enumerate(tr.tolist()):
                signed[i, a], signed[i, b] = 1.0, -1.0
                unsigned[i, a] = unsigned[i, b] = 1.0
            assert torch.linalg.matrix_rank(signed).item() == P - 1
            assert torch.linalg.matrix_rank(unsigned).item() == P
            for a, b in te.tolist():
                row = torch.zeros(P); row[a], row[b] = 1.0, -1.0
                coef = torch.linalg.lstsq(signed.T, row[:, None]).solution[:, 0]
                assert (signed.T @ coef - row).abs().max().item() < 1e-9


def _hand_built_streams(task, arm):
    """Streams set to the teacher's primitives; for 'streams_self', gates set by hand: the world makes two streams
    available, each stream bids its own trigger, a softmax picks the larger bid."""
    cfg = dict(DEF, task=task, arm=arm, score="softmax", w=DEF["d"], q=DEF["d"], K=P)
    T, S = Teacher(cfg), Model(cfg, 0)
    with torch.no_grad():
        S.p["Bs"].copy_(T.A); S.p["Vs"].copy_(torch.eye(cfg["d"]).expand(P, -1, -1) * (2.0 if (task == "sum" and arm != "oracle") else 1.0))
        if arm == "streams_self":
            S.p["E"].zero_(); S.p["E"][:, :P] = torch.eye(P)
            S.p["G"].zero_(); S.p["G"][:, :P] = 2000.0 * torch.eye(P)
            S.p["S1"].zero_(); S.p["S2"].zero_(); S.p["sb"].zero_()
            if task == "select":
                S.p["S1"][:, 0, :] = 1e-4 * T.w
                S.p["S2"][:, 0] = 60.0 / 1e-4
    return cfg, T, S


def test_the_modular_solution_is_exact_on_worlds_never_seen():
    worlds = torch.tensor(all_worlds(P))
    x = torch.randn(len(worlds), 64, DEF["d"], generator=torch.Generator().manual_seed(0))
    for task in ("select", "sum"):
        cfg, T, S = _hand_built_streams(task, "oracle")
        assert err(S.forward(worlds, x, cfg["n"], T), T.run(worlds, x, cfg["n"]), x).item() < 1e-24
        cfg, T, S = _hand_built_streams(task, "streams_self")
        e = err(S.forward(worlds, x, cfg["n"], T), T.run(worlds, x, cfg["n"]), x).item()
        assert e < (1e-2 if task == "select" else 1e-20), (task, e)       # select: a soft arg-max, exact as the bids sharpen


def test_arms_have_the_same_number_of_parameters_and_references_are_defined():
    n = {arm: sum(v.numel() for v in Model(dict(DEF, arm=arm), 0).params()) for arm in ("mono", "streams_joint", "streams_self", "unit")}
    for arm, v in n.items():
        assert abs(v - n["mono"]) / n["mono"] < 0.03, n
    tr, te = sample_worlds(P, 9, 0)
    r = references(dict(task="select"), te)
    assert 0.05 < r["blend"] < r["one"]                                  # blending the two beats always applying one
    assert references(dict(task="sum"), te)["one"] > 0.05
