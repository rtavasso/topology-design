"""Terms: nested tuples; leaves are atoms. Prefix order indexes nodes; a node's *path* is its left/right string from
the root, which is the structural position the router addresses (DESIGN §1: positions are structural, never a flat
index). All functions are pure and small; terms are hashable and used as dict keys throughout."""

from __future__ import annotations

from functools import lru_cache
from typing import Iterator

ATOMS = ("a", "b", "c", "d", "0")
ZERO = "0"
OP = "+"          # the prefix token of an internal node


def is_leaf(t) -> bool:
    return isinstance(t, str)


@lru_cache(maxsize=1 << 16)
def size(t) -> int:
    """Number of leaves."""
    return 1 if is_leaf(t) else size(t[0]) + size(t[1])


@lru_cache(maxsize=1 << 16)
def nnodes(t) -> int:
    return 1 if is_leaf(t) else 1 + nnodes(t[0]) + nnodes(t[1])


@lru_cache(maxsize=1 << 16)
def depth(t) -> int:
    return 0 if is_leaf(t) else 1 + max(depth(t[0]), depth(t[1]))


def prefix(t) -> list:
    """Prefix-order tokens: '+' for an internal node, the atom for a leaf."""
    out = []
    _prefix(t, out)
    return out


def _prefix(t, out):
    if is_leaf(t):
        out.append(t)
    else:
        out.append(OP)
        _prefix(t[0], out)
        _prefix(t[1], out)


def to_str(t) -> str:
    return t if is_leaf(t) else "(" + to_str(t[0]) + "+" + to_str(t[1]) + ")"


def parse(s: str):
    """Inverse of to_str on fully parenthesized strings, e.g. '((a+b)+0)'."""
    s = s.replace(" ", "")
    pos = 0

    def rec():
        nonlocal pos
        if s[pos] == "(":
            pos += 1
            left = rec()
            assert s[pos] == "+", s
            pos += 1
            right = rec()
            assert s[pos] == ")", s
            pos += 1
            return (left, right)
        c = s[pos]
        pos += 1
        assert c in ATOMS, c
        return c

    t = rec()
    assert pos == len(s), s
    return t


def nodes(t) -> list:
    """All nodes in prefix order as (path, subterm, depth); path is a string over {'L', 'R'} ('' is the root)."""
    out = []
    _nodes(t, "", 0, out)
    return out


def _nodes(t, path, d, out):
    out.append((path, t, d))
    if not is_leaf(t):
        _nodes(t[0], path + "L", d + 1, out)
        _nodes(t[1], path + "R", d + 1, out)


def subterm_at(t, path: str):
    for c in path:
        if is_leaf(t):
            raise KeyError(path)
        t = t[0] if c == "L" else t[1]
    return t


def replace_at(t, path: str, new):
    """The term with the subterm at `path` replaced by `new`."""
    if path == "":
        return new
    if is_leaf(t):
        raise KeyError(path)
    if path[0] == "L":
        return (replace_at(t[0], path[1:], new), t[1])
    return (t[0], replace_at(t[1], path[1:], new))


def paths_overlap(p: str, q: str) -> bool:
    """Two nodes overlap when one is an ancestor of (or equal to) the other: rewrites there do not commute."""
    return p.startswith(q) or q.startswith(p)


def structural_features(t) -> list:
    """Per node in prefix order: (depth, path as left/right bits, subtree leaves, is_leaf). What the structural router
    embeds instead of an absolute index."""
    return [(d, path, size(u), is_leaf(u)) for path, u, d in nodes(t)]


def iter_terms(max_leaves: int, atoms=ATOMS) -> Iterator:
    """Every term with at most max_leaves leaves (for brute-force tests)."""
    by_n = {1: list(atoms)}
    for n in range(2, max_leaves + 1):
        acc = []
        for i in range(1, n):
            for l in by_n[i]:
                for r in by_n[n - i]:
                    acc.append((l, r))
        by_n[n] = acc
    for n in range(1, max_leaves + 1):
        yield from by_n[n]
