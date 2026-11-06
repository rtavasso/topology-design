"""Training and evaluation of the rung-4 router on the rewrite carrier (DESIGN §1 Loss, §4 Arms, §6 Protocol).

Arms (cfg["arm"]):
  RT-canon      teacher forcing on the canonical shortest proof (hard labels on its states)
  RT-post       teacher forcing on the posterior over shortest proofs (every shortest-DAG state, weight = share of
                shortest proofs through it, soft target over the actions on shortest paths)
  PF            the precomputed flat-prior posterior over routes ≤ T reaching s (states sampled by visitation, soft targets)
  EI16          expert iteration: beam W=16 under (1−ε) π + ε uniform over valid actions, completions verified by the
                executor, the MAP completion (log π̃ − λ moves) imitated with hard labels
  RT-all        RT-canon on all fifteen worlds (needs the held-out worlds' training files)
  RT-absolute   RT-canon with the absolute-position / categorical control router
  RT-examples   RT-canon with the axioms unstated and two example proofs of the same world in context
  RT-shuffled   RT-canon with the presence tokens drawn from a random training world (independent of the true world)

Every training step draws instances uniformly by world then by instance and trains on cfg["rows"] teacher-forced rows.
Evaluation: committed rollouts (argmax) from t for T = 2k + 2 steps, masked (valid actions ∪ HALT) and unmasked (all
105 slots; an invalid action is a no-op the Markov router repeats, so the episode ends there), with failure categories,
first-decision correctness, and strata (world, k, binding, GP1, D*_any, size)."""

from __future__ import annotations

import gzip
import json
import math
import os
import random
import time
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F

from .data import dec_action, enc_action
from .model import (HALT_INDEX, MAX_NODES, N_ACTIONS, N_RULES, RULE_INDEX, Router, RouterConfig, action_slot, collate,
                    encode, matched_configs, n_params, slot_action)
from .rules import HALT, RULE_AX, RULES, step, valid_actions, world_from_name, world_name
from .term import nodes, parse, to_str

TRAIN_WORLDS = ["A", "C", "I", "D", "AC", "AD", "CD", "ID"]
ALL_WORLDS = ["A", "C", "I", "D", "AC", "AI", "AD", "CI", "CD", "ID", "ACI", "ACD", "AID", "CID", "ACID"]
HELD_OUT = [w for w in ALL_WORLDS if w not in TRAIN_WORLDS]
CURVE_PER_WORLD = 150


# ---------------- data ----------------

def read_jsonl(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f]


def load_set(data_dir: str, tag: str, worlds: list, limit: int | None = None) -> dict:
    out = {}
    for w in worlds:
        p = os.path.join(data_dir, f"{tag}_{w}.jsonl.gz")
        if not os.path.exists(p):
            continue
        recs = read_jsonl(p)
        if limit:
            recs = recs[:limit]
        for r in recs:
            prepare(r)
        out[w] = recs
    return out


def prepare(rec: dict):
    """Derived fields: the states along the canonical proof and the prefix index of every node of every teacher state."""
    if "canon_states" in rec:
        return rec
    W, cap = world_from_name(rec["world"]), rec["cap"]
    t = parse(rec["t"])
    states, u = [rec["t"]], t
    for a in rec["canon"][:-1]:
        rule, path = dec_action(a)
        u = step(u, rule, path, W, cap)
        assert u is not None, (rec["t"], rec["canon"])
        states.append(to_str(u))
    assert states[-1] == rec["s"]
    rec["canon_states"] = states
    # the correct first actions: those on a shortest path from t (the RT-post target at t lists exactly them)
    rec["first_actions"] = [a for p in rec["post"] if p["state"] == rec["t"] for a in p["target"]]
    return rec


@lru_cache(maxsize=1 << 17)
def P(t_str: str):
    return parse(t_str)


@lru_cache(maxsize=1 << 17)
def valid_slots(u: str, world: str, cap: int) -> tuple:
    """Slots of the valid state-changing actions at u under the world, plus HALT, and the successor strings."""
    pi = path_index(u)
    acts = valid_actions(P(u), world_from_name(world), cap)
    return tuple(pi[p] * N_RULES + RULE_INDEX[r] for r, p, _ in acts) + (HALT_INDEX,), tuple(to_str(v) for _, _, v in acts), tuple((r, p) for r, p, _ in acts)


@lru_cache(maxsize=1 << 16)
def path_index(t_str: str) -> dict:
    return {p: i for i, (p, _, _) in enumerate(nodes(parse(t_str)))}


@lru_cache(maxsize=1 << 16)
def node_paths(t_str: str) -> list:
    return [p for p, _, _ in nodes(parse(t_str))]


_ENC = {}


def encode_cached(world, s_str, t_str, examples=None):
    key = (world, s_str, t_str, tuple(examples) if examples else None)
    e = _ENC.get(key)
    if e is None:
        e = encode(world, s_str, t_str, [(a, b, list(c)) for a, b, c in examples] if examples else None)
        if len(_ENC) < 400000:
            _ENC[key] = e
    return e


def slots_of_target(t_str: str, target: dict) -> dict:
    pi = path_index(t_str)
    return {action_slot(dec_action(a), pi): p for a, p in target.items()}


# ---------------- rows per arm ----------------

def shown_world(rec, cfg, rng):
    if cfg["arm"] == "RT-shuffled":
        return rng.choice(TRAIN_WORLDS)
    if cfg["arm"] == "RT-examples":
        return None
    return rec["world"]


def example_proofs(rec, pool: list, rng, n=2):
    """Two other instances of the same world as (t, s, rules of the canonical proof)."""
    out = []
    tries = 0
    while len(out) < n and tries < 20:
        o = rng.choice(pool)
        tries += 1
        if o is rec:
            continue
        out.append((o["t"], o["s"], tuple(dec_action(a)[0] for a in o["canon"][:-1])))
    return out


def rows_rt(rec, world, examples, rng):
    rows = []
    for u, a in zip(rec["canon_states"], rec["canon"]):
        rows.append((world, rec["s"], u, examples, {action_slot(dec_action(a), path_index(u)): 1.0}, 1.0))
    return rows


def rows_post(rec, world, examples, rng, max_rows=24):
    post = rec["post"]
    total = sum(p["weight"] for p in post)                       # = k + 1
    if len(post) <= max_rows:
        return [(world, rec["s"], p["state"], examples, slots_of_target(p["state"], p["target"]), p["weight"]) for p in post]
    ws = [p["weight"] for p in post]
    picks = rng.choices(post, weights=ws, k=max_rows)
    return [(world, rec["s"], p["state"], examples, slots_of_target(p["state"], p["target"]), total / max_rows) for p in picks]


def rows_pf(rec, world, examples, rng, max_rows=None):
    pf = rec.get("pf")
    if pf is None:
        return []
    states = pf["states"]
    tot = sum(p["visits"] for p in states)
    m = max_rows or rec["k"] + 1
    picks = rng.choices(states, weights=[p["visits"] for p in states], k=m)
    # per-instance mass k + 1, as RT's
    return [(world, rec["s"], p["state"], examples, slots_of_target(p["state"], p["target"]), (rec["k"] + 1) / m) for p in picks]


# ---------------- the model's distribution over actions at a batch of states ----------------

def build_batch(rows, device):
    encs = [encode_cached(w, s, u, ex) for (w, s, u, ex, _, _) in rows]
    return collate(encs, device)


def valid_mask(rows_states, worlds, caps, device):
    m = torch.zeros(len(rows_states), N_ACTIONS, dtype=torch.bool)
    for i, (u, w, cap) in enumerate(zip(rows_states, worlds, caps)):
        m[i, list(valid_slots(u, w, cap)[0])] = True
    return m.to(device)


def masked_logits(logits: torch.Tensor, rows_states, worlds, caps):
    """Keep only the valid state-changing actions and HALT for each row (the masked reading / the beam's support)."""
    m = valid_mask(rows_states, worlds, caps, logits.device)
    return logits.masked_fill(~m, float("-inf"))


# ---------------- EI16 ----------------

@torch.no_grad()
def beam_search(model, recs, device, W=16, eps=0.1, lam=0.02):
    """Width-W beam per instance under π̃ = (1 − ε) π_masked + ε uniform over valid actions ∪ HALT; duplicates merged (most
    probable prefix kept); every kept state completed by HALT; the forced halt at T. Among completions whose final term is s
    the one maximising log π̃(route) − λ·moves is returned as a list of (state, action) pairs, or None."""
    n = len(recs)
    beams = [[(rec["t"], 0.0, [])] for rec in recs]                     # (state, logp, actions)
    best = [None] * n                                                  # (key, pairs)
    done = [False] * n
    Ts = [2 * r["k"] + 2 for r in recs]
    for t in range(max(Ts) + 1):
        rows = []
        owners = []
        for i, rec in enumerate(recs):
            if t > Ts[i]:
                continue
            for (u, lp, acts) in beams[i]:
                rows.append((rec["world"], rec["s"], u, None, None, None))
                owners.append(i)
        if not rows:
            break
        b = build_batch(rows, device)
        logits = model(b).float()
        states_ = [r[2] for r in rows]
        logits = masked_logits(logits, states_, [r[0] for r in rows], [recs[i]["cap"] for i in owners])
        probs = torch.softmax(logits, 1).cpu().numpy()
        new_beams = [[] for _ in recs]
        for ri, ((w, s, u, _, _, _), i) in enumerate(zip(rows, owners)):
            rec = recs[i]
            slots, succs, acts = valid_slots(u, w, rec["cap"])
            p = (1 - eps) * probs[ri, list(slots)] + eps / len(slots)
            entry = next(x for x in beams[i] if x[0] == u)
            lp0, prefix = entry[1], entry[2]
            # completion by HALT (the forced halt at T adds no HALT term)
            key = lp0 - lam * t + (math.log(p[-1]) if t < Ts[i] else 0.0)
            if u == s and (best[i] is None or key > best[i][0]):
                best[i] = (key, prefix + [(u, HALT)])
            if t == Ts[i]:
                continue
            for j, (a, v) in enumerate(zip(acts, succs)):
                new_beams[i].append((v, lp0 + math.log(p[j]), prefix + [(u, a)]))
        for i in range(n):
            if t >= Ts[i]:
                continue
            cand = sorted(new_beams[i], key=lambda x: -x[1])
            seen, kept = set(), []
            for c in cand:
                if c[0] in seen:
                    continue
                seen.add(c[0]); kept.append(c)
                if len(kept) >= W:
                    break
            beams[i] = kept
    return [b[1] if b else None for b in best]


# ---------------- rollouts ----------------

@torch.no_grad()
def rollout(model, recs, device, masked: bool, shown=None, examples=None, batch=2048):
    """Committed (argmax) rollouts. Returns per-instance dicts: ok, steps, category, first_ok, visited_goal.
    shown[i] = world string shown in the context (None = unstated), examples[i] = example proofs or None."""
    out = []
    for start in range(0, len(recs), batch):
        chunk = recs[start:start + batch]
        sh = shown[start:start + batch] if shown is not None else [r["world"] for r in chunk]
        ex = examples[start:start + batch] if examples is not None else [None] * len(chunk)
        res = [None] * len(chunk)
        cur = [r["t"] for r in chunk]
        visited = [r["t"] == r["s"] for r in chunk]
        first_ok = [None] * len(chunk)
        Ts = [2 * r["k"] + 2 for r in chunk]
        alive = list(range(len(chunk)))
        for t in range(max(Ts) + 1):
            if not alive:
                break
            rows = [(sh[i], chunk[i]["s"], cur[i], ex[i], None, None) for i in alive]
            logits = model(build_batch(rows, device)).float()
            if masked:
                logits = masked_logits(logits, [cur[i] for i in alive], [chunk[i]["world"] for i in alive], [chunk[i]["cap"] for i in alive])
            choice = logits.argmax(1).tolist()
            still = []
            for i, c in zip(alive, choice):
                rec = chunk[i]
                u = cur[i]
                a = slot_action(c, node_paths(u))
                if t == 0:
                    first_ok[i] = enc_action(a) in rec["first_actions"] if a != HALT and a[1] is not None else False
                if t == Ts[i]:
                    # forced halt at T
                    res[i] = {"ok": u == rec["s"], "steps": t, "category": "ok" if u == rec["s"] else ("no halt at goal" if visited[i] else "horizon")}
                    continue
                if a == HALT:
                    res[i] = {"ok": u == rec["s"], "steps": t, "category": "ok" if u == rec["s"] else "wrong halt"}
                    continue
                rule, path = a
                v = step(P(u), rule, path, world_from_name(rec["world"]), rec["cap"]) if path is not None else None
                if v is None or to_str(v) == u:
                    # invalid or self-loop: the Markov router repeats it to T
                    why = "invalid: out of world" if path is not None and RULE_AX[rule] not in world_from_name(rec["world"]) else "invalid: no match"
                    res[i] = {"ok": False, "steps": t, "category": why}
                    continue
                cur[i] = to_str(v)
                if cur[i] == rec["s"]:
                    visited[i] = True
                still.append(i)
            alive = still
        for i, r in enumerate(res):
            r["first_ok"] = bool(first_ok[i])
            r["visited_goal"] = bool(visited[i])
            out.append(r)
    return out


# ---------------- evaluation and strata ----------------

def stratum_keys(rec):
    f = rec["first"]
    return {"world": rec["world"], "k": rec["k"], "binding": int(rec["binding"]), "GP1": int(f["GP1"]), "Dany": f["Dany"],
            "size": rec["size_t"]}


def aggregate(recs, outs_m, outs_u):
    """Cells by (world, binding, GP1, k): n, success masked/unmasked, first-decision accuracy; plus per world totals and
    failure categories."""
    cells = {}
    for rec, om, ou in zip(recs, outs_m, outs_u):
        key = f"{rec['world']}|{int(rec['binding'])}|{int(rec['first']['GP1'])}|{rec['k']}"
        c = cells.setdefault(key, {"n": 0, "ok_m": 0, "ok_u": 0, "first": 0, "cat_m": {}, "cat_u": {}})
        c["n"] += 1; c["ok_m"] += om["ok"]; c["ok_u"] += ou["ok"]; c["first"] += om["first_ok"]
        c["cat_m"][om["category"]] = c["cat_m"].get(om["category"], 0) + 1
        c["cat_u"][ou["category"]] = c["cat_u"].get(ou["category"], 0) + 1
    return cells


def eval_sets_for(cfg, data_dir):
    sets = {"eval5": load_set(data_dir, "eval5", ALL_WORLDS), "eval7": load_set(data_dir, "eval7", ALL_WORLDS),
            "evalk78": load_set(data_dir, "evalk78", ["AC"])}
    return sets


def evaluate(model, sets, cfg, device, rng, full: bool):
    model.eval()
    report = {}
    for tag, by_world in sets.items():
        recs, shown, exs = [], [], []
        for w, rs in by_world.items():
            use = rs if full else rs[:CURVE_PER_WORLD]
            for r in use:
                recs.append(r)
                shown.append(None if cfg["arm"] == "RT-examples" else r["world"])      # RT-shuffled sees the true tokens at eval
                exs.append(example_proofs(r, rs, rng) if cfg["arm"] == "RT-examples" else None)
        if not recs:
            continue
        om = rollout(model, recs, device, masked=True, shown=shown, examples=exs)
        ou = rollout(model, recs, device, masked=False, shown=shown, examples=exs)
        report[tag] = {"cells": aggregate(recs, om, ou)}
        if full:
            report[tag]["per_instance"] = [{"world": r["world"], "t": r["t"], "s": r["s"], "k": r["k"], "binding": r["binding"],
                                            "GP1": r["first"]["GP1"], "Dany": r["first"]["Dany"], "size": r["size_t"],
                                            "ok_m": a["ok"], "ok_u": b["ok"], "cat_m": a["category"], "cat_u": b["category"],
                                            "steps_m": a["steps"], "first_ok": a["first_ok"], "ladder": r["ladder"]}
                                           for r, a, b in zip(recs, om, ou)]
    model.train()
    return report


# ---------------- training ----------------

def make_model(cfg):
    cfg_s, cfg_a, n_s, n_a = matched_configs(d=cfg.get("d", 128), layers=cfg.get("layers", 3), heads=cfg.get("heads", 4), d_ff=cfg.get("d_ff", 512))
    rc = cfg_a if cfg["arm"] == "RT-absolute" else cfg_s
    return Router(rc), rc


def run(cfg: dict, data_dir: str, device: str = "cpu", log=print, sets=None) -> dict:
    """One run. cfg: arm, seed, lr, steps, rows (teacher-forced rows per step), inst (instances per step for EI16),
    eval_every, warmup, worlds (training worlds; RT-all uses all fifteen), pf_only (restrict the stream to PF-labelled records)."""
    t0 = time.time()
    seed = int(cfg.get("seed", 0))
    torch.manual_seed(seed); np.random.seed(seed)
    rng = random.Random(seed)
    arm = cfg["arm"]
    worlds = cfg.get("worlds") or (ALL_WORLDS if arm == "RT-all" else TRAIN_WORLDS)
    train = load_set(data_dir, "train", worlds)
    if arm == "PF":
        train = {w: [r for r in rs if "pf" in r] for w, rs in train.items()}
    train = {w: rs for w, rs in train.items() if rs}
    assert train, "no training data"
    sets = sets or eval_sets_for(cfg, data_dir)
    model, rc = make_model(cfg)
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.01), betas=(0.9, 0.98))
    steps, rows_per_step, warmup = int(cfg["steps"]), int(cfg.get("rows", 256)), int(cfg.get("warmup", 200))
    eval_every = int(cfg.get("eval_every", 250))
    inst_per_step = int(cfg.get("inst", 32))
    wlist = sorted(train)
    curve, losses = [], []
    ei_found = [0, 0]
    log(f"run {cfg.get('name', arm)} arm={arm} params={n_params(model)} structural={rc.structural} worlds={wlist} "
        f"stream={ {w: len(rs) for w, rs in train.items()} }")
    for it in range(steps + 1):
        if it % eval_every == 0 or it == steps:
            rep = evaluate(model, sets, cfg, device, random.Random(seed * 7919 + it), full=(it == steps))
            point = {"step": it, "loss": float(np.mean(losses[-eval_every:])) if losses else None, "time": time.time() - t0}
            for tag, r in rep.items():
                point[tag] = r["cells"]
            if arm == "EI16":
                point["ei_found"] = list(ei_found)
            curve.append(point)
            summ = {tag: round(sum(c["ok_m"] for c in r["cells"].values()) / max(1, sum(c["n"] for c in r["cells"].values())), 3) for tag, r in rep.items()}
            log(f"  step {it} loss {point['loss']} masked success {summ} {time.time() - t0:.0f}s")
            if it == steps:
                final = rep
                break
        # ---- one training step
        rows = []
        if arm == "EI16":
            recs = [rng.choice(train[rng.choice(wlist)]) for _ in range(inst_per_step)]
            model.eval()
            found = beam_search(model, recs, device, W=int(cfg.get("beam", 16)), eps=float(cfg.get("eps", 0.1)), lam=float(cfg.get("lam", 0.02)))
            model.train()
            for rec, route in zip(recs, found):
                ei_found[1] += 1
                if route is None:
                    continue
                ei_found[0] += 1
                for (u, a) in route:
                    rows.append((rec["world"], rec["s"], u, None, {action_slot(a, path_index(u)): 1.0}, 1.0))
            if not rows:
                losses.append(float("nan"))
                continue
        else:
            while len(rows) < rows_per_step:
                w = rng.choice(wlist)
                rec = rng.choice(train[w])
                shown = shown_world(rec, cfg, rng)
                ex = example_proofs(rec, train[w], rng) if arm == "RT-examples" else None
                if arm == "RT-post":
                    rows += rows_post(rec, shown, ex, rng)
                elif arm == "PF":
                    rows += rows_pf(rec, shown, ex, rng)
                else:
                    rows += rows_rt(rec, shown, ex, rng)
        b = build_batch(rows, device)
        logits = model(b)
        target = torch.zeros(len(rows), N_ACTIONS, device=device)
        wts = torch.tensor([r[5] for r in rows], device=device, dtype=torch.float)
        for i, r in enumerate(rows):
            for slot, p in r[4].items():
                target[i, slot] = p
        logp = F.log_softmax(logits, 1)
        ce = -(target * logp.masked_fill(~torch.isfinite(logp), 0.0)).sum(1)
        loss = (wts * ce).sum() / wts.sum()
        lr = cfg["lr"] * min(1.0, (it + 1) / warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.detach()))
    return {"cfg": cfg, "params": n_params(model), "router": rc.dict(), "curve": curve, "final": final,
            "ei_found": ei_found, "seconds": time.time() - t0, "losses": losses[::10]}
