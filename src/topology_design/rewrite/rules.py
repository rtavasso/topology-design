"""The eight directed rewrite rules (the primitives), worlds, and the exact one-step executor (DESIGN §1).

Af  (x+y)+z -> x+(y+z)      Ab  x+(y+z) -> (x+y)+z
C   x+y -> y+x
If  x+0 -> x  or  0+x -> x  (x+0 first)      IbR  x -> x+0      IbL  x -> 0+x
Df  x+x -> x                                  Db   x -> x+x
A world W is a subset of the axioms {A, C, I, D}; a rule may be applied only if its axiom is in W. An action is
(rule, path) or HALT. `step` returns the new term or None when the action is invalid (out of world, no match at the
node, over the size cap, missing node) — the untyped semantics: a no-op costs a step and is detected only by the goal.
"""

from __future__ import annotations

from itertools import combinations

from .term import ZERO, is_leaf, nodes, replace_at, size, subterm_at

RULES = ("Af", "Ab", "C", "If", "IbR", "IbL", "Df", "Db")
RULE_AX = {"Af": "A", "Ab": "A", "C": "C", "If": "I", "IbR": "I", "IbL": "I", "Df": "D", "Db": "D"}
AXIOMS = ("A", "C", "I", "D")
HALT = "HALT"
WORLDS = tuple(frozenset(w) for r in range(1, 5) for w in combinations(AXIOMS, r))     # 15, singletons first


def world_name(W) -> str:
    return "".join(a for a in AXIOMS if a in W)


def world_from_name(name: str) -> frozenset:
    assert all(c in AXIOMS for c in name) and name, name
    return frozenset(name)


def apply_local(rule: str, u):
    """Apply a directed rule at the root of subterm u; None when it does not match."""
    if rule == "Af":
        if not is_leaf(u) and not is_leaf(u[0]):
            (x, y), z = u
            return (x, (y, z))
        return None
    if rule == "Ab":
        if not is_leaf(u) and not is_leaf(u[1]):
            x, (y, z) = u
            return ((x, y), z)
        return None
    if rule == "C":
        return None if is_leaf(u) else (u[1], u[0])
    if rule == "If":
        if not is_leaf(u):
            if u[1] == ZERO:
                return u[0]
            if u[0] == ZERO:
                return u[1]
        return None
    if rule == "IbR":
        return (u, ZERO)
    if rule == "IbL":
        return (ZERO, u)
    if rule == "Df":
        if not is_leaf(u) and u[0] == u[1]:
            return u[0]
        return None
    if rule == "Db":
        return (u, u)
    raise ValueError(rule)


def step(t, rule: str, path: str, W, cap: int):
    """The exact executor for one action. None = invalid (out of world / no node / no match / over cap)."""
    if RULE_AX[rule] not in W:
        return None
    try:
        u = subterm_at(t, path)
    except KeyError:
        return None
    v = apply_local(rule, u)
    if v is None:
        return None
    new = replace_at(t, path, v)
    if size(new) > cap:
        return None
    return new


def valid_actions(t, W, cap: int, state_changing: bool = True) -> list:
    """All valid actions at t: [(rule, path, new_term)]. With state_changing, self-loops (C at x+x, ...) are excluded —
    the candidate set of every per-decision quantity (DESIGN §1)."""
    out = []
    nds = nodes(t)
    for rule in RULES:
        if RULE_AX[rule] not in W:
            continue
        for path, u, _ in nds:
            v = apply_local(rule, u)
            if v is None:
                continue
            new = replace_at(t, path, v)
            if size(new) > cap:
                continue
            if state_changing and new == t:
                continue
            out.append((rule, path, new))
    return out


def successors(t, W, cap: int) -> dict:
    """Distinct state-changing successors: {new_term: [(rule, path), ...]}."""
    d = {}
    for rule, path, new in valid_actions(t, W, cap):
        d.setdefault(new, []).append((rule, path))
    return d


def verify(t, s, actions, W, cap: int) -> dict:
    """Check a route: every action valid and in-world, the final term equal to s, HALT at the end (or implied).
    Returns {"ok", "steps", "first_invalid", "reason", "final"}."""
    u = t
    for i, a in enumerate(actions):
        if a == HALT:
            return {"ok": u == s, "steps": i, "first_invalid": None, "reason": None if u == s else "wrong halt", "final": u}
        rule, path = a
        if RULE_AX[rule] not in W:
            return {"ok": False, "steps": i, "first_invalid": i, "reason": "out of world", "final": u}
        v = step(u, rule, path, W, cap)
        if v is None:
            return {"ok": False, "steps": i, "first_invalid": i, "reason": "no match", "final": u}
        u = v
    return {"ok": u == s, "steps": len(actions), "first_invalid": None, "reason": None if u == s else "no halt at goal", "final": u}
