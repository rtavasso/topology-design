"""Exact statements behind experiments/theory/worlds/DESIGN.md."""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "experiments", "theory", "worlds"))
from worlds import DEF, Model, Teacher, all_worlds, err, graph_ok, references, sample_worlds  # noqa: E402

torch.set_default_dtype(torch.float64)
P = DEF["P"]


def test_world_sets_are_nested_disjoint_complete_and_two_connected():
    for split in range(12):
        prev = set()
        for cover in (7, 9, 11, 13):
            tr, te = sample_worlds(P, cover, split)
            a, b = {tuple(r) for r in tr.tolist()}, {tuple(r) for r in te.tolist()}
            assert len(a) == cover and not (a & b) and (a | b) == set(all_worlds(P))
            assert graph_ok(P, sorted(a))                  # connected without any one primitive, and has an odd cycle
            assert prev <= a                               # the dial is nested
            prev = a
    tr, te = sample_worlds(P, 15, 0)
    assert len(tr) == 15 and len(te) == 0
    assert not graph_ok(P, [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 5), (3, 5)])      # a cut vertex
    assert not graph_ok(P, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (0, 3)])      # bipartite


def test_what_training_worlds_determine_about_a_held_out_world():
    # (1) IF bids are linear triggers, a held-out rule sign<w_a - w_b, h> is fixed up to the scales of the trained
    #     differences (signed incidence rank P - 1) - an assumption about form, not something the data force.
    # (2) Without that assumption, the data force a held-out choice only where a chain of trained comparisons does:
    #     a > c > ... > b at this state. `determined` computes that; the share is strictly between 0 and 1.
    # (3) sum: each primitive is recoverable from trained sums (unsigned incidence rank P).
    for split in range(6):
        for cover in (7, 9):
            tr, te = sample_worlds(P, cover, split)
            signed, unsigned = torch.zeros(len(tr), P), torch.zeros(len(tr), P)
            for i, (a, b) in enumerate(tr.tolist()):
                signed[i, a], signed[i, b] = 1.0, -1.0
                unsigned[i, a] = unsigned[i, b] = 1.0
            assert torch.linalg.matrix_rank(signed).item() == P - 1
            assert torch.linalg.matrix_rank(unsigned).item() == P
            r = references(dict(task="select", split=split, cover=cover), tr, te)
            assert 0.15 < r["forced"] < 0.95
            assert r["trans"] < r["blend"] < r["allmean"]
    cfg = dict(DEF)
    T = Teacher(cfg)
    worlds = torch.tensor(all_worlds(P))
    x = torch.randn(len(worlds), 32, cfg["d"], generator=torch.Generator().manual_seed(0))
    T.trained = torch.ones(P, P) - torch.eye(P)
    assert T.determined(worlds, x).all()                                        # every comparison trained: all forced
    assert torch.equal(T.run(worlds, x, 4, "trans"), T.run(worlds, x, 4))
    T.trained = torch.zeros(P, P)
    assert not T.determined(worlds, x).any()
    assert torch.equal(T.run(worlds, x, 4, "trans"), T.run(worlds, x, 4, "blend"))
    # a two-step chain: with (0,2) and (2,1) trained, world {0,1} is forced exactly where 2's bid lies between
    T.trained = torch.zeros(P, P)
    for a, b in ((0, 2), (2, 1)):
        T.trained[a, b] = T.trained[b, a] = 1.0
    w01 = torch.tensor([[0, 1]]); xs = x[:1]
    s = xs @ T.w.T
    between = ((s[..., 0] > s[..., 2]) & (s[..., 2] > s[..., 1])) | ((s[..., 1] > s[..., 2]) & (s[..., 2] > s[..., 0]))
    assert torch.equal(T.determined(w01, xs), between)


def _hand_built(task, arm):
    """Streams set to the teacher's primitives; gates set by hand: the world makes two streams available, each
    stream bids its own trigger, a softmax picks the larger bid (select) or splits evenly (sum)."""
    cfg = dict(DEF, task=task, arm=arm, score="softmax", w=DEF["d"], q=DEF["d"], K=P)
    T, S = Teacher(cfg), Model(cfg, 0)
    with torch.no_grad():
        S.p["Bs"].copy_(T.A)
        S.p["Vs"].copy_(torch.eye(cfg["d"]).expand(P, -1, -1) * (2.0 if (task == "sum" and arm != "oracle") else 1.0))
        if arm != "oracle":
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
        cfg, T, S = _hand_built(task, "oracle")
        assert err(S.forward(worlds, x, cfg["n"], T), T.run(worlds, x, cfg["n"]), x).item() < 1e-24
        for arm in ("streams_self", "tok_streams"):
            cfg, T, S = _hand_built(task, arm)
            e = err(S.forward(worlds, x, cfg["n"], T), T.run(worlds, x, cfg["n"]), x).item()
            assert e < (1e-2 if task == "select" else 1e-20), (task, arm, e)   # select: a soft arg-max, exact as bids sharpen


def test_arms_have_the_same_number_of_parameters():
    n = {arm: sum(v.numel() for v in Model(dict(DEF, arm=arm), 0).params())
         for arm in ("mono", "tok_mono", "streams_joint", "streams_self", "tok_streams", "unit")}
    for arm, v in n.items():
        assert abs(v - n["mono"]) / n["mono"] < 0.03, n


def test_a_world_that_does_not_matter_is_not_a_problem_for_the_references():
    cfg = dict(DEF, same=True)
    T = Teacher(cfg)
    worlds = torch.tensor(all_worlds(P))
    x = torch.randn(len(worlds), 16, cfg["d"], generator=torch.Generator().manual_seed(0))
    assert torch.allclose(T.run(worlds, x, 4), T.run(worlds.flip(0), x, 4))       # identical primitives: world irrelevant
