#!/usr/bin/env python3
"""
SNAKE // KEV  -  Kev-0.8B plays Snake, and you watch its decision graph fire.

One file, one native window (tkinter), no web server. Every move, the game state is written out as text
(the "state"), Kev reads it once (prefill), then answers a small graph of questions the way Jev does in the
Doom demo:

    known facts ──► offers goals ──► [goal] ──┐
                                    [which apple] ─► goal + apple → objective ──► [move] ─► answer → heading ─► HEADING
                                    [risk] ────────────────────────────────────┘    └─────► risk → alert ─────► ALERT
                offers moves (the non-lethal directions) ───────────────────────────► [move]  (or a reflex if only one is left)

Amber boxes are Kev questions (choice / score), ellipses are plain code combining answers, dashed boxes are
facts the game computes. Nodes light up in the order they are actually computed; the right-hand panel shows
each question with Kev's probability for every option.

Install (Python 3.10+; 3.12 or 3.13 recommended):
    pip install torch "transformers>=5.17" huggingface_hub
    (NVIDIA GPU: install the CUDA build of torch from pytorch.org first; it is picked up automatically)

Run:
    python snake_kev.py                # first run downloads Kev-0.8B + Qwen3.5-0.8B-Base (~1.7 GB) from Hugging Face
    python snake_kev.py --mock         # preview the UI with a fake random "model" (no download)
    python snake_kev.py --headless 20  # no window: play 20 moves in the terminal

Keys:  space pause/resume · n single step · r restart · + / - game speed · f fire-animation speed
       s situation report (the exact text Kev reads) · esc quit

Kev: https://github.com/jaredpalmer/kev (Apache-2.0). The model code below is a minimal re-implementation of
kev/model.py + kev/checkpoint.py (same token format, LoRA merged in fp32, pointer head, calibrated temperature);
it reproduces the reference implementation's probabilities exactly on the README example.
"""
import argparse
import collections
import copy
import math
import os
import queue
import random
import re
import sys
import threading
import time
import traceback

os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

KEV_REPO = "jaredpalmer/kev-0.8b"


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Kev: Qwen3.5 backbone + merged LoRA + pointer head
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

# Kev reuses rarely-used Qwen special tokens as delimiters: <state> <q> <opt> </opt> <decide>
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class Kev:
    """Prefill the state once, then answer each question as its own row continuing the cached state
    (the "row form" kev uses for the hybrid Qwen3.5 backbone). Returns calibrated probabilities."""

    def __init__(self, repo=KEV_REPO, device=None, threads=None, log=print):
        import torch
        import torch.nn as nn
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

        self.torch, self.F, self.DynamicCache = torch, torch.nn.functional, DynamicCache
        torch.set_grad_enabled(False)   # note: grad mode is per thread; prefill/ask also run under no_grad
        if threads: torch.set_num_threads(threads)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        log(f"downloading {repo} (adapter + head) ...")
        ck = snapshot_download(repo, allow_patterns=["*.json", "*.safetensors", "*.pt"])
        meta = torch.load(os.path.join(ck, "head.pt"), map_location="cpu")
        base, rev = meta["base"], meta.get("base_revision")
        log(f"loading tokenizer + {base} (first run downloads ~1.7 GB) ...")
        self.tok = AutoTokenizer.from_pretrained(base, revision=rev)
        lm = AutoModelForCausalLM.from_pretrained(base, revision=rev, dtype=torch.float32).model  # backbone only

        log("merging LoRA adapter ...")
        cfg = __import__("json").load(open(os.path.join(ck, "adapter_config.json"), encoding="utf-8"))
        scale = cfg["lora_alpha"] / (math.sqrt(cfg["r"]) if cfg.get("use_rslora") else cfg["r"])
        weights = load_file(os.path.join(ck, "adapter_model.safetensors"))
        modules = dict(lm.named_modules())
        for name, a in weights.items():
            if not name.endswith(".lora_A.weight"): continue
            stem = name[: -len(".lora_A.weight")]
            mod = modules[stem.replace("base_model.model.", "", 1)]
            mod.weight += (weights[stem + ".lora_B.weight"].float() @ a.float()) * scale   # exact in fp32
        del weights

        self.dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
        self.lm = lm.to(device=device, dtype=self.dtype).eval()
        d, dp = lm.config.hidden_size, meta.get("head_dim", 256)
        self.hq, self.hk = nn.Linear(d, dp), nn.Linear(d, dp)
        self.hq.weight.data, self.hq.bias.data = meta["head"]["q.weight"], meta["head"]["q.bias"]
        self.hk.weight.data, self.hk.bias.data = meta["head"]["k.weight"], meta["head"]["k.bias"]
        self.hq.to(device); self.hk.to(device)
        self.scale, self.temperature = 1 / math.sqrt(dp), float(meta.get("temperature", 1.0))
        self.ids = [self.tok.convert_tokens_to_ids(t) for t in SPECIAL]
        self.name = f"kev-0.8b · {device} {str(self.dtype).removeprefix('torch.')}"
        log("warming up ...")
        self.ask(self.prefill("warm up"), "Is this a test?", ["no", "yes"])

    def _tokens(self, text):
        # caller text can never forge a delimiter token
        return self.tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids

    def prefill(self, state):
        with self.torch.no_grad():
            return self._prefill(state)

    def ask(self, prefix, instructions, options):
        with self.torch.no_grad():
            return self._ask(prefix, instructions, options)

    def _prefill(self, state):
        torch = self.torch
        S = [self.ids[0]] + self._tokens(state)
        out = self.lm(input_ids=torch.tensor([S], device=self.device),
                      position_ids=torch.arange(len(S), device=self.device)[None],
                      past_key_values=self.DynamicCache(config=self.lm.config), use_cache=True)
        return len(S), out.past_key_values

    def _ask(self, prefix, instructions, options):
        torch = self.torch
        n_state, cache = prefix
        q_id, o_id, c_id, d_id = self.ids[1:]
        row, ends = [q_id] + self._tokens(instructions), []
        for opt in options:
            row += [o_id] + self._tokens(opt) + [c_id]
            ends.append(len(row) - 1)           # each option is read at its </opt> token
        row.append(d_id)                        # ... against the <decide> token
        h = self.lm(input_ids=torch.tensor([row], device=self.device),
                    position_ids=torch.arange(n_state, n_state + len(row), device=self.device)[None],
                    past_key_values=copy.deepcopy(cache), use_cache=True).last_hidden_state[0].float()
        z = (self.hk(h[ends]) @ self.hq(h[-1])) * self.scale / self.temperature
        return self.F.softmax(z, -1).tolist()


class MockKev:
    """Stand-in for --mock: noisy probabilities nudged by the hints, with a fake latency."""
    name = "mock model (random)"

    def prefill(self, state):
        time.sleep(0.15); return (len(state) // 4, None)

    def ask(self, prefix, instructions, options, hint=None):
        time.sleep(0.12)
        hint = hint or [0.0] * len(options)
        z = [h * 2.0 + random.gauss(0, 0.8) for h in hint]
        m = max(z); e = [math.exp(v - m) for v in z]; s = sum(e)
        return [v / s for v in e]


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Snake
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

DIRS = {"up": (0, -1), "right": (1, 0), "down": (0, 1), "left": (-1, 0)}
OPPOSITE = {"up": "down", "down": "up", "left": "right", "right": "left"}
ARROW = {"up": "↑", "right": "→", "down": "↓", "left": "←"}
STARVE = 150
GOLD_TTL = 35


class Snake:
    def __init__(self, n=12, seed=None):
        self.n, self.rng, self.best = n, random.Random(seed), 0
        self.reset()

    def reset(self):
        c = self.n // 2
        self.body = [(c, c), (c - 1, c), (c - 2, c)]   # head first
        self.heading, self.grow, self.alive, self.cause = "right", 0, True, ""
        self.score, self.moves, self.hunger, self.eaten = 0, 0, 0, 0
        self.apples = {}                                 # cell -> {"kind", "value", "ttl"}
        self.spawn("red")

    def free_cells(self):
        taken = set(self.body) | set(self.apples)
        return [(x, y) for x in range(self.n) for y in range(self.n) if (x, y) not in taken]

    def spawn(self, kind):
        free = self.free_cells()
        if free:
            self.apples[self.rng.choice(free)] = {"kind": kind, "value": 3 if kind == "gold" else 1,
                                                  "ttl": GOLD_TTL if kind == "gold" else None}

    def inside(self, c):
        return 0 <= c[0] < self.n and 0 <= c[1] < self.n

    def preview(self, d):
        """What moving `d` would do: (new head, lethal cause or '', body after the move)."""
        dx, dy = DIRS[d]
        hx, hy = self.body[0]
        nh = (hx + dx, hy + dy)
        if not self.inside(nh): return nh, "wall", None
        apple = self.apples.get(nh)
        tail_moves = self.grow + (apple["value"] if apple else 0) == 0
        blockers = self.body[:-1] if tail_moves else self.body
        if nh in blockers: return nh, "body", None
        return nh, "", [nh] + (self.body[:-1] if tail_moves else self.body)

    def step(self, d):
        if not self.alive: return
        if d == OPPOSITE[self.heading]: d = self.heading
        nh, cause, _ = self.preview(d)
        self.moves += 1
        if cause:
            self.alive, self.cause = False, f"hit the {cause}"; return
        apple = self.apples.pop(nh, None)
        if apple:
            self.grow += apple["value"]; self.score += apple["value"]; self.eaten += 1; self.hunger = 0
        else:
            self.hunger += 1
        self.body.insert(0, nh)
        if self.grow: self.grow -= 1
        else: self.body.pop()
        self.heading = d
        self.best = max(self.best, self.score)
        for c, a in list(self.apples.items()):
            if a["ttl"] is not None:
                a["ttl"] -= 1
                if a["ttl"] <= 0: del self.apples[c]
        if not any(a["kind"] == "red" for a in self.apples.values()): self.spawn("red")
        if not any(a["kind"] == "gold" for a in self.apples.values()) and self.rng.random() < 0.05: self.spawn("gold")
        if self.hunger >= STARVE:
            self.alive, self.cause = False, "starved"
        if not self.free_cells():
            self.alive, self.cause = False, "filled the board. perfect game!"


def bfs(n, start, goal, blocked):
    """Shortest path length from start to goal avoiding `blocked` (goal itself may be in blocked), or None."""
    if start == goal: return 0
    seen, frontier, dist = {start}, collections.deque([start]), {start: 0}
    while frontier:
        x, y = frontier.popleft()
        for dx, dy in DIRS.values():
            c = (x + dx, y + dy)
            if c in seen or not (0 <= c[0] < n and 0 <= c[1] < n): continue
            if c == goal: return dist[(x, y)] + 1
            if c in blocked: continue
            seen.add(c); dist[c] = dist[(x, y)] + 1; frontier.append(c)
    return None


def flood(n, start, blocked):
    """Number of free cells reachable from start (not counting start)."""
    seen, frontier = {start}, [start]
    while frontier:
        x, y = frontier.pop()
        for dx, dy in DIRS.values():
            c = (x + dx, y + dy)
            if c not in seen and c not in blocked and 0 <= c[0] < n and 0 <= c[1] < n:
                seen.add(c); frontier.append(c)
    return len(seen) - 1


def apple_name(a):
    return f"{a['kind']} apple"


def situation(g):
    """Everything the game knows (the 'known ...' inputs of the graph)."""
    n, head, tail = g.n, g.body[0], g.body[-1]
    now_blocked = set(g.body[1:-1]) if g.grow == 0 else set(g.body[1:])
    apples = []
    for c, a in g.apples.items():
        apples.append({**a, "cell": c, "dist": bfs(n, head, c, now_blocked), "manhattan": abs(c[0] - head[0]) + abs(c[1] - head[1])})
    apples.sort(key=lambda a: (a["dist"] is None, a["dist"] or a["manhattan"]))
    tail_dist = bfs(n, head, tail, set(g.body[1:-1])) if len(g.body) > 3 else None
    moves = {}
    for d in DIRS:
        if d == OPPOSITE[g.heading]: continue
        nh, cause, after = g.preview(d)
        m = {"dir": d, "cell": nh, "lethal": cause}
        if not cause:
            blocked = set(after[1:])
            m["open"] = flood(n, nh, blocked)
            m["apple_dist"] = {a["cell"]: (0 if nh == a["cell"] else bfs(n, nh, a["cell"], blocked)) for a in apples}
            tail_blocked = set(after[1:-1])
            m["tail_dist"] = bfs(n, nh, after[-1], tail_blocked) if len(after) > 3 else None
        moves[d] = m
    safe = [m for m in moves.values() if not m["lethal"]]
    return {"apples": apples, "tail_dist": tail_dist, "moves": moves, "safe": safe,
            "max_open": max((m["open"] for m in safe), default=0)}


def state_text(g, f):
    n = g.n
    grid = [["."] * n for _ in range(n)]
    for c, a in g.apples.items(): grid[c[1]][c[0]] = "G" if a["kind"] == "gold" else "R"
    for x, y in g.body[1:-1]: grid[y][x] = "o"
    tx, ty = g.body[-1]; grid[ty][tx] = "T"
    hx, hy = g.body[0]; grid[hy][hx] = "H"
    lines = [f"game: snake on a {n}x{n} board. x runs 0-{n-1} from left to right, y runs 0-{n-1} from top to bottom. "
             f"The border is a wall. The snake dies if its head hits the wall or its own body.",
             "board (H head, o body, T tail, R red apple, G gold apple, . empty):"]
    lines += ["".join(r) for r in grid]
    lines.append(f"snake: head at ({hx},{hy}), moving {g.heading}, length {len(g.body)}, tail at ({tx},{ty})")
    for a in f["apples"]:
        extra = f", disappears in {a['ttl']} moves" if a["ttl"] else ""
        path = f"{a['dist']} steps away" if a["dist"] is not None else "no open path right now"
        lines.append(f"{apple_name(a)} at ({a['cell'][0]},{a['cell'][1]}): worth {a['value']}, {path}{extra}")
    lines.append(f"moves since the last apple: {g.hunger} (the snake starves at {STARVE})")
    around = []
    for d, m in f["moves"].items():
        if m["lethal"] == "wall": what = "the wall"
        elif m["lethal"]: what = "its own body"
        else: what = f"free, {m['open']} open cells beyond"
        around.append(f"{d}: {what}")
    lines.append("next to the head: " + "; ".join(around))
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  The decision graph (one "think" per move)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

RISK_LEVELS = ["safe, plenty of room", "getting cramped", "nearly trapped"]
ALERTS = ["calm", "caution", "danger"]


def think(g, kev, emit, mock=False):
    """Compute one move. Emits ("fire", node, value, sources) in the order things are computed, ("answer", ...)
    for the decisions panel, and finally ("done", heading, info)."""
    t0 = time.time()
    f = situation(g)
    head = g.body[0]

    def ask(prefix, instr, opts, hint):
        t = time.time()
        p = kev.ask(prefix, instr, opts, hint) if mock else kev.ask(prefix, instr, opts)
        return p, (time.time() - t) * 1000

    # ── known facts
    ap = f["apples"]
    emit("fire", "board", f"len {len(g.body)} · move {g.moves}", None)
    emit("fire", "apples", " · ".join(f"{a['kind']} {a['dist'] if a['dist'] is not None else '✕'}" for a in ap) or "none", None)
    emit("fire", "hunger", f"{g.hunger} / {STARVE}", None)
    emit("fire", "tail", f"{f['tail_dist']} steps" if f["tail_dist"] is not None else ("cut off" if len(g.body) > 3 else "too short"), None)
    walls = [f"{d} {m['lethal']}" for d, m in f["moves"].items() if m["lethal"]]
    emit("fire", "danger", " · ".join(walls) if walls else "clear", None)
    emit("fire", "space", " ".join(f"{d[0].upper()}{m['open']}" for d, m in f["moves"].items() if not m["lethal"]) or "none", None)

    # ── offers (plain code)
    goals = []
    nearest = ap[0] if ap else None
    if ap:
        near = f"nearest is {nearest['dist']} steps away" if nearest["dist"] is not None else "no open path to any apple right now"
        goals.append(("eat_apple", f"go eat an apple ({near}; starves in {STARVE - g.hunger} moves)"))
    goals.append(("open_space", f"move toward the biggest open area ({f['max_open']} free cells)"))
    if f["tail_dist"] is not None:
        goals.append(("follow_tail", f"chase its own tail to stay safe (tail is {f['tail_dist']} steps away)"))
    emit("fire", "offer_goals", f"{len(goals)} goals", ["apples", "hunger", "tail", "space"])
    safe = f["safe"]
    emit("fire", "offer_moves", f"{len(safe)} safe of {len(f['moves'])}", ["danger", "space"])

    # ── Kev reads the state once
    state = state_text(g, f)
    t = time.time()
    prefix = kev.prefill(state)
    emit("fire", "prefill", f"{prefix[0]} tok · {(time.time() - t) * 1000:.0f} ms", ["board"])
    emit("state", state)

    # ── goal
    instr = ("What should the snake's top priority be right now? It grows by eating apples, "
             "and it dies if it hits a wall or its own body.")
    hint = [1.0 if k == "eat_apple" and nearest and nearest["dist"] is not None else 0.5 if k == "follow_tail" and f["max_open"] < len(g.body) else 0 for k, _ in goals]
    p, ms = ask(prefix, instr, [f"{k}: {d}" for k, d in goals], hint)
    gi = max(range(len(p)), key=p.__getitem__)
    goal = goals[gi][0]
    emit("answer", "goal", {"instr": instr, "labels": [k for k, _ in goals], "descs": [d for _, d in goals], "probs": p, "pick": gi, "ms": ms})
    emit("fire", "q_goal", goal, ["prefill", "offer_goals"])

    # ── which apple
    target = None
    if ap:
        instr = "Which apple should the snake go for?"
        descs = []
        for a in ap:
            path = f"{a['dist']} steps away" if a["dist"] is not None else "no open path right now"
            descs.append(f"worth {a['value']} point{'s' if a['value'] > 1 else ''}, {path}" + (f", disappears in {a['ttl']} moves" if a["ttl"] else ""))
        labels = [f"{a['kind']} ({a['cell'][0]},{a['cell'][1]})" for a in ap]
        if len(ap) == 1:
            p, ms, note = [1.0], 0.0, "only one apple: nothing to ask"
        else:
            hint = [(a["value"] / (1 + (a["dist"] if a["dist"] is not None else 99)) * 5) for a in ap]
            p, ms = ask(prefix, instr, [f"{a['kind']} apple: {d}" for a, d in zip(ap, descs)], hint)
            note = ""
        ai = max(range(len(p)), key=p.__getitem__)
        target = ap[ai]
        emit("answer", "apple", {"instr": instr, "labels": labels, "descs": descs, "probs": p, "pick": ai, "ms": ms, "note": note})
        emit("fire", "q_apple", f"{target['kind']}" + (" (only one)" if len(ap) == 1 else ""), ["prefill", "apples"])

    # ── risk (score)
    instr = "How dangerous is the snake's position right now?"
    hint = [1 if f["max_open"] > 2 * len(g.body) else 0, 1 if len(g.body) < f["max_open"] <= 2 * len(g.body) else 0, 1 if f["max_open"] <= len(g.body) else 0]
    p, ms = ask(prefix, instr, RISK_LEVELS, hint)
    risk = sum(i * pi for i, pi in enumerate(p))
    ri = max(range(len(p)), key=p.__getitem__)
    emit("answer", "risk", {"instr": instr, "labels": ["0 safe", "1 cramped", "2 trapped"], "descs": RISK_LEVELS, "probs": p, "pick": ri, "ms": ms, "score": risk})
    emit("fire", "q_risk", f"{RISK_LEVELS[ri].split(',')[0]} · score {risk:.2f}", ["prefill"])
    alert = ri   # the most likely level (the expected score is shown too)
    emit("fire", "alert", f"{RISK_LEVELS[ri].split(',')[0]} → {ALERTS[alert]}", ["q_risk"])
    emit("fire", "ALERT", ALERTS[alert], ["alert"], alert)

    # ── objective (code): goal + apple
    if goal == "eat_apple" and target:
        objective = f"eat the {apple_name(target)} at ({target['cell'][0]},{target['cell'][1]})"
        emit("fire", "objective", f"eat {target['kind']} ({target['cell'][0]},{target['cell'][1]})", ["q_goal", "q_apple"])
    else:
        objective = "move into the biggest open area" if goal == "open_space" else "follow its own tail"
        emit("fire", "objective", goal.replace("_", " "), ["q_goal"])

    # ── move
    if len(safe) <= 1:
        heading = safe[0]["dir"] if safe else g.heading
        why = f"only {heading} is safe" if safe else "no way out"
        emit("answer", "move", {"instr": "(skipped: reflex)", "labels": [heading], "descs": [why], "probs": [1.0], "pick": 0, "ms": 0.0,
                                "note": "reflex: fewer than two safe moves, Kev is not asked"})
        emit("fire", "reflex", why, ["offer_moves"])
        emit("fire", "heading", heading, ["reflex"])
    else:
        instr = f"The snake wants to {objective}. Its position is {RISK_LEVELS[ri].split(',')[0]}. Which way should it move next?"
        descs, hint = [], []
        for m in safe:
            parts = []
            if goal == "eat_apple" and target:
                d_now, d_new = target["dist"], m["apple_dist"].get(target["cell"])
                if d_new == 0: parts.append(f"eats the {apple_name(target)}")
                elif d_new is None: parts.append(f"no path to the {apple_name(target)}")
                else:
                    trend = "" if d_now is None else " (closer)" if d_new < d_now else " (farther)" if d_new > d_now else ""
                    parts.append(f"{d_new} steps from the {apple_name(target)}{trend}")
                h = 2 if d_new == 0 else 0 if d_new is None else (1 if (d_now is None or d_new < d_now) else 0)
            elif goal == "follow_tail":
                parts.append(f"tail {m['tail_dist']} steps away" if m["tail_dist"] is not None else "loses sight of the tail")
                h = 1 if m["tail_dist"] is not None else 0
            else:
                h = m["open"] / max(1, f["max_open"])
            dead_end = m["open"] < len(g.body)
            parts.append(f"{m['open']} open cells reachable" + (" (dead end!)" if dead_end else " (the most)" if m["open"] == f["max_open"] else ""))
            descs.append(", ".join(parts))
            hint.append(h - (2 if dead_end else 0))
        p, ms = ask(prefix, instr, [f"{m['dir']}: {d}" for m, d in zip(safe, descs)], hint)
        mi = max(range(len(p)), key=p.__getitem__)
        heading = safe[mi]["dir"]
        emit("answer", "move", {"instr": instr, "labels": [m["dir"] for m in safe], "descs": descs, "probs": p, "pick": mi, "ms": ms})
        emit("fire", "q_move", f"{heading} {p[mi] * 100:.0f}%", ["prefill", "objective", "q_risk", "offer_moves"])
        emit("fire", "heading", heading, ["q_move"])
    emit("fire", "HEADING", f"{ARROW[heading]} {heading.upper()}", ["heading"])
    emit("done", heading, {"ms": (time.time() - t0) * 1000, "alert": alert})


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Headless mode
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

def run_headless(kev, moves, grid, mock, seed):
    g = Snake(grid, seed)
    for _ in range(moves):
        if not g.alive:
            print(f"  x {g.cause} - score {g.score}\n"); g.reset()
        answers, result = {}, {}

        def emit(kind, *a):
            if kind == "answer": answers[a[0]] = a[1]
            if kind == "done": result["heading"], result["info"] = a[0], a[1]
        think(g, kev, emit, mock)
        line = []
        for q in ("goal", "apple", "risk", "move"):
            if q in answers:
                r = answers[q]
                line.append(f"{q}={r['labels'][r['pick']]}({r['probs'][r['pick']]:.2f})")
        print(f"move {g.moves:3d} len {len(g.body):2d} score {g.score:2d} | " + "  ".join(line) + f"  -> {result['heading']}  [{result['info']['ms']:.0f} ms]")
        g.step(result["heading"])
    print(f"final: score {g.score}, best {g.best}, alive {g.alive} {g.cause}")


# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════
#  Window (tkinter)
# ═══════════════════════════════════════════════════════════════════════════════════════════════════════════

BG, PANEL, GRIDC = "#040705", "#070c08", "#0e1a11"
GREEN, GREEN_DIM, GREEN_FAINT, EDGE_DIM = "#3dff72", "#1d4a28", "#16301c", "#1f4629"
TXT, TXT_DIM = "#9dffb8", "#2f6b3d"
AMBER, AMBER_FILL, AMBER_DIM, AMBER_FILL_DIM, AMBER_TXT = "#ffb13b", "#3a2607", "#4d3610", "#110b02", "#ffd488"
RED, RED_DIM = "#ff4a5e", "#4a1a20"
GOLD = "#ffd24a"
ALERT_COL = [GREEN, AMBER, RED]

W, H = 1200, 814          # base layout (scaled by --zoom / DPI)
GX, GY, GW, GH = 12, 486, 1176, 320   # graph panel

# id: (kind, x, y, width, label)  (x, y = centre, relative to the graph panel)
NODES = {
    "board":       ("input",  92,  42, 164, "board state"),
    "apples":      ("input",  92,  92, 164, "known apples"),
    "hunger":      ("input",  92, 142, 164, "hunger"),
    "tail":        ("input",  92, 192, 164, "tail"),
    "danger":      ("input",  92, 242, 164, "danger"),
    "space":       ("input",  92, 292, 164, "open space"),
    "prefill":     ("comb",  277,  42, 156, "state → kev prefill"),
    "offer_goals": ("offer", 277, 117, 144, "offers goals"),
    "offer_moves": ("offer", 277, 267, 144, "offers moves"),
    "q_goal":      ("q",     458,  97, 164, "goal"),
    "q_apple":     ("q",     458, 162, 164, "which apple"),
    "q_risk":      ("q",     458, 232, 164, "risk"),
    "objective":   ("comb",  632, 130, 164, "goal + apple → objective"),
    "q_move":      ("q",     810, 175, 160, "move"),
    "reflex":      ("reflex", 810, 278, 160, "reflex"),
    "heading":     ("comb",  976, 175, 140, "answer → heading"),
    "alert":       ("comb",  976, 255, 140, "risk → alert"),
    "HEADING":     ("out",  1110, 175, 112, "HEADING"),
    "ALERT":       ("out",  1110, 255, 112, "ALERT"),
}
NODE_H = 34
EDGES = [
    ("board", "prefill", ""),
    ("prefill", "q_goal", "thin"), ("prefill", "q_apple", "thin"), ("prefill", "q_risk", "thin"), ("prefill", "q_move", "thin"),
    ("apples", "offer_goals", ""), ("hunger", "offer_goals", ""), ("tail", "offer_goals", ""), ("space", "offer_goals", ""),
    ("apples", "q_apple", ""),
    ("danger", "offer_moves", ""), ("space", "offer_moves", ""),
    ("offer_goals", "q_goal", ""),
    ("q_goal", "objective", ""), ("q_apple", "objective", ""),
    ("objective", "q_move", ""), ("q_risk", "q_move", ""), ("offer_moves", "q_move", ""),
    ("offer_moves", "reflex", "red"),
    ("q_move", "heading", ""), ("reflex", "heading", "red"),
    ("heading", "HEADING", ""),
    ("q_risk", "alert", ""), ("alert", "ALERT", ""),
]
PANEL_ORDER = [("goal", "GOAL"), ("apple", "APPLE"), ("risk", "RISK"), ("move", "MOVE")]


def mix(c1, c2, t):
    a = [int(c1[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(c2[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def rrect(x1, y1, x2, y2, r):
    return [x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r, x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2, x1, y2, x1, y2 - r, x1, y1 + r, x1, y1]


class App:
    def __init__(self, args):
        import tkinter as tk
        import tkinter.font as tkfont
        self.tk, self.args = tk, args
        if sys.platform == "win32":
            try:
                import ctypes
                ctypes.windll.shcore.SetProcessDpiAwareness(1)   # crisp text on high-DPI screens
            except Exception:
                pass
        self.root = root = tk.Tk()
        root.title("SNAKE // KEV")
        root.configure(bg=BG)
        s = args.zoom or root.winfo_fpixels("1i") / 96.0
        s = min(s, root.winfo_screenwidth() * 0.96 / W, root.winfo_screenheight() * 0.9 / H)
        self.S = max(0.5, s)
        fams = set(tkfont.families())
        self.family = next((f for f in ("Consolas", "Cascadia Mono", "JetBrains Mono", "DejaVu Sans Mono", "Menlo", "Courier New") if f in fams), "Courier")
        self.cv = tk.Canvas(root, width=W * self.S, height=H * self.S, bg=BG, highlightthickness=0)
        self.cv.pack()
        root.resizable(False, False)

        self.game = Snake(args.grid, args.seed)
        self.kev, self.events = None, queue.Queue()
        self.paused, self.busy, self.step_once = False, False, False
        self.move_delay, self.fire_gap = args.move_ms / 1000, args.fire_ms / 1000
        self.next_event, self.next_think = 0.0, 0.0
        self.answers, self.state, self.show_state = {}, "", False
        self.planned, self.alert, self.last_ms, self.status = None, 0, None, "loading"
        self.anims, self.error = [], None
        self.nodes, self.edges = {}, []

        self.draw_static()
        self.draw_graph()
        self.draw_game()
        self.draw_panel()
        for key, fn in {"<space>": self.toggle_pause, "n": self.step, "N": self.step, "r": self.restart, "R": self.restart,
                        "plus": self.faster, "equal": self.faster, "KP_Add": self.faster, "minus": self.slower, "KP_Subtract": self.slower,
                        "f": self.toggle_fire, "F": self.toggle_fire, "s": self.toggle_state, "S": self.toggle_state,
                        "<Escape>": lambda e=None: root.destroy()}.items():
            root.bind(key if key.startswith("<") else f"<{key}>" if len(key) > 1 else key, fn)
        threading.Thread(target=self.load, daemon=True).start()
        self.root.after(16, self.pump)

    # ── helpers for scaled drawing
    def P(self, *v):
        return [x * self.S for x in v]

    def font(self, size, bold=False):
        return (self.family, -max(6, round(size * self.S)), "bold" if bold else "normal")

    def text(self, x, y, s, size=11, color=TXT, anchor="nw", bold=False, tags=(), width=None):
        kw = {"width": width * self.S} if width else {}
        return self.cv.create_text(*self.P(x, y), text=s, fill=color, font=self.font(size, bold), anchor=anchor, tags=tags, **kw)

    # ── model loading
    def load(self):
        try:
            if self.args.mock:
                self.kev = MockKev()
            else:
                self.kev = Kev(device=self.args.device, threads=self.args.threads,
                               log=lambda m: self.events.put(("status", m)))
            self.events.put(("loaded",))
        except Exception as e:
            traceback.print_exc()
            self.events.put(("error", f"could not load Kev: {e}"))

    # ── static chrome
    def draw_static(self):
        c, S = self.cv, self.S
        self.text(14, 9, "SNAKE // KEV", 13, GREEN, bold=True)
        self.text(W - 14, 10, "[space] pause   [n] step   [r] restart   [+/-] speed   [f] fire pace   [s] situation   [esc] quit",
                  10, TXT_DIM, anchor="ne")
        c.create_line(*self.P(12, 30, W - 12, 30), fill=GREEN_FAINT)
        # board frame
        c.create_rectangle(*self.P(12, 38, 432, 458), outline=GREEN_DIM, width=S)
        # decisions frame
        c.create_rectangle(*self.P(442, 38, W - 12, 478), outline=GREEN_DIM, width=S, fill=PANEL)
        self.text(452, 43, "DECISIONS", 9, TXT_DIM)
        # graph frame
        c.create_rectangle(*self.P(GX, GY, GX + GW, GY + GH), outline=GREEN_DIM, width=S, fill=PANEL)
        self.text(GX + 8, GY + 4, "GRAPH", 9, TXT_DIM)

    # ── graph
    def node_box(self, nid):
        kind, x, y, w, _ = NODES[nid]
        h = 40 if kind == "comb" else NODE_H
        return GX + x - w / 2, GY + y - h / 2, GX + x + w / 2, GY + y + h / 2

    def draw_graph(self):
        c, S = self.cv, self.S
        incoming = collections.defaultdict(list)
        for src, dst, kind in EDGES: incoming[dst].append(src)
        for dst in incoming: incoming[dst].sort(key=lambda s: NODES[s][2])
        for src, dst, kind in EDGES:
            x1, y1a, x1b, y1b = self.node_box(src)
            x2a, y2a, x2b, y2b = self.node_box(dst)
            sx, sy = x1b, (y1a + y1b) / 2
            dx, dy = x2a, (y2a + y2b) / 2
            k = incoming[dst].index(src)
            xm = dx - 9 - 6 * k
            pts = [sx, sy, dx, dy] if abs(sy - dy) < 1 else [sx, sy, xm, sy, xm, dy, dx, dy]
            halo = c.create_line(*self.P(*pts), fill=BG, width=6 * S, joinstyle="round", capstyle="round")
            core = c.create_line(*self.P(*pts), fill=GREEN_FAINT, width=S, joinstyle="round",
                                 dash=(4, 3) if kind == "red" else ())
            self.edges.append({"src": src, "dst": dst, "kind": kind, "pts": pts, "halo": halo, "core": core, "lit": False})
        for nid, (kind, x, y, w, label) in NODES.items():
            x1, y1, x2, y2 = self.node_box(nid)
            halo = body = None
            if kind == "comb":
                halo = c.create_oval(*self.P(x1 - 3, y1 - 3, x2 + 3, y2 + 3), outline=BG, width=4 * S)
                body = c.create_oval(*self.P(x1, y1, x2, y2), outline=GREEN_DIM, fill=PANEL, width=S)
            else:
                r = NODE_H / 2 if kind == "offer" else 5
                halo = c.create_polygon(*self.P(*rrect(x1 - 3, y1 - 3, x2 + 3, y2 + 3, r + 2)), smooth=True, outline=BG, fill="", width=4 * S)
                body = c.create_polygon(*self.P(*rrect(x1, y1, x2, y2, r)), smooth=True, outline=GREEN_DIM, fill=PANEL, width=S,
                                        dash=(3, 3) if kind in ("input", "offer", "reflex") else ())
            t1 = self.text(GX + x, GY + y - 7, label, 9.5, TXT_DIM, anchor="center", bold=kind == "out")
            t2 = self.text(GX + x, GY + y + 7, "", 9.5, TXT_DIM, anchor="center", bold=kind == "out")
            self.nodes[nid] = {"kind": kind, "halo": halo, "body": body, "t1": t1, "t2": t2, "lit": False, "flash": 0.0,
                               "label": label, "w": w, "color": None}
        self.paint_all()

    def node_colors(self, n, now):
        kind, lit = n["kind"], n["lit"]
        f = max(0.0, 1 - (now - n["flash"]) / 0.55) if lit else 0.0
        if kind == "q":
            base = (AMBER, AMBER_FILL, AMBER_TXT) if lit else (AMBER_DIM, AMBER_FILL_DIM, AMBER_DIM)
        elif kind == "reflex":
            base = (RED, RED_DIM, "#ffb0b8") if lit else (RED_DIM, PANEL, RED_DIM)
        else:
            col = n.get("color") or GREEN
            base = (col, mix(PANEL, col, 0.12 if kind != "out" else 0.18), mix(col, "#ffffff", 0.45)) if lit else (GREEN_DIM, PANEL, TXT_DIM)
        outline, fill, txt = base
        if f:
            outline, fill, txt = mix(outline, "#ffffff", 0.7 * f), mix(fill, outline, 0.5 * f), mix(txt, "#ffffff", 0.8 * f)
        halo = mix(BG, outline, (0.28 + 0.5 * f)) if lit else BG
        return outline, fill, txt, halo, f

    def paint_node(self, nid, now=None):
        n, c = self.nodes[nid], self.cv
        outline, fill, txt, halo, f = self.node_colors(n, now or time.time())
        c.itemconfigure(n["body"], outline=outline, fill=fill, width=(2.2 if n["lit"] else 1) * self.S)
        c.itemconfigure(n["halo"], outline=halo)
        c.itemconfigure(n["t1"], fill=txt)
        c.itemconfigure(n["t2"], fill=txt)
        return f > 0

    def paint_edge(self, e):
        red = e["kind"] == "red"
        if e["lit"]:
            col = RED if red else GREEN
            self.cv.itemconfigure(e["core"], fill=col, width=(1.6 if e["kind"] == "thin" else 2.4) * self.S)
            self.cv.itemconfigure(e["halo"], fill=mix(BG, col, 0.22))
        else:
            self.cv.itemconfigure(e["core"], fill=RED_DIM if red else EDGE_DIM, width=self.S)
            self.cv.itemconfigure(e["halo"], fill=BG)

    def paint_all(self):
        for nid in self.nodes: self.paint_node(nid)
        for e in self.edges: self.paint_edge(e)

    def reset_graph(self):
        for n in self.nodes.values(): n["lit"] = False
        for e in self.edges: e["lit"] = False
        self.paint_all()

    def fire(self, nid, value, sources, color_idx=None):
        n, now = self.nodes[nid], time.time()
        n["lit"], n["flash"] = True, now
        if color_idx is not None: n["color"] = ALERT_COL[color_idx]
        width = int(n["w"] / (6.2 if n["kind"] != "out" else 6.8))
        v = str(value)
        v = v if len(v) <= width - 2 else v[: width - 3] + "…"
        self.cv.itemconfigure(n["t2"], text=f"▸ {v}")
        self.paint_node(nid, now)
        for e in self.edges:
            if e["dst"] != nid: continue
            if (sources is None and self.nodes[e["src"]]["lit"]) or (sources and e["src"] in sources):
                e["lit"] = True
                self.paint_edge(e)
                self.cv.tag_raise(e["halo"]); self.cv.tag_raise(e["core"])
                self.add_pulse(e)
        for k in ("halo", "body", "t1", "t2"):
            self.cv.tag_raise(n[k])

    def add_pulse(self, e):
        col = RED if e["kind"] == "red" else "#d9ffe4"
        dot = self.cv.create_oval(0, 0, 0, 0, fill=col, outline="")
        glow = self.cv.create_oval(0, 0, 0, 0, fill=mix(BG, GREEN if e["kind"] != "red" else RED, 0.45), outline="")
        pts = list(zip(e["pts"][::2], e["pts"][1::2]))
        segs = [(a, b, math.dist(a, b)) for a, b in zip(pts, pts[1:])]
        dur = max(0.18, min(0.45, sum(s[2] for s in segs) / 900)) * (self.fire_gap / 0.11 if self.fire_gap < 0.11 else 1)
        self.anims.append({"dot": dot, "glow": glow, "segs": segs, "t0": time.time(), "dur": dur})

    def animate(self):
        now, S, keep = time.time(), self.S, []
        for a in self.anims:
            u = (now - a["t0"]) / a["dur"]
            if u >= 1:
                self.cv.delete(a["dot"]); self.cv.delete(a["glow"]); continue
            total = sum(s[2] for s in a["segs"]) or 1
            d = u * total
            for (x1, y1), (x2, y2), L in a["segs"]:
                if d <= L or L == 0:
                    t = d / L if L else 1
                    x, y = x1 + (x2 - x1) * t, y1 + (y2 - y1) * t
                    break
                d -= L
            self.cv.coords(a["glow"], (x - 6) * S, (y - 6) * S, (x + 6) * S, (y + 6) * S)
            self.cv.coords(a["dot"], (x - 3) * S, (y - 3) * S, (x + 3) * S, (y + 3) * S)
            self.cv.tag_raise(a["glow"]); self.cv.tag_raise(a["dot"])
            keep.append(a)
        self.anims = keep
        for nid, n in self.nodes.items():
            if n["lit"] and now - n["flash"] < 0.6: self.paint_node(nid, now)

    # ── game board
    def draw_game(self):
        c, g, S = self.cv, self.game, self.S
        c.delete("game")
        bx, by, bs = 22, 48, 400
        cell = bs / g.n
        for i in range(g.n + 1):
            c.create_line(*self.P(bx + i * cell, by, bx + i * cell, by + bs), fill=GRIDC, tags="game")
            c.create_line(*self.P(bx, by + i * cell, bx + bs, by + i * cell), fill=GRIDC, tags="game")
        c.create_rectangle(*self.P(bx, by, bx + bs, by + bs), outline=GREEN, width=1.5 * S, tags="game")

        def box(cx, cy, inset):
            return self.P(bx + cx * cell + inset, by + cy * cell + inset, bx + (cx + 1) * cell - inset, by + (cy + 1) * cell - inset)
        for (x, y), a in g.apples.items():
            col = GOLD if a["kind"] == "gold" else RED
            c.create_oval(*box(x, y, cell * 0.18), fill=col, outline=mix(col, "#ffffff", 0.4), tags="game")
            c.create_line(*self.P(bx + (x + 0.5) * cell, by + (y + 0.18) * cell, bx + (x + 0.62) * cell, by + (y + 0.05) * cell),
                          fill=GREEN, width=2 * S, tags="game")
            if a["ttl"]:
                self.text(bx + (x + 0.5) * cell, by + (y + 0.5) * cell, str(a["ttl"]), 8.5, "#3a2a00", anchor="center", bold=True, tags="game")
        n = len(g.body)
        head_col = ALERT_COL[self.alert] if g.alive else "#777777"
        for i in range(n - 1, -1, -1):
            x, y = g.body[i]
            t = i / max(1, n - 1)
            col = head_col if i == 0 else mix("#2fd460", "#0f4a1f", t) if g.alive else "#333333"
            if i + 1 < n:   # connector to the next segment
                x2, y2 = g.body[i + 1]
                c.create_rectangle(*self.P(bx + (min(x, x2) + 0.22) * cell, by + (min(y, y2) + 0.22) * cell,
                                           bx + (max(x, x2) + 0.78) * cell, by + (max(y, y2) + 0.78) * cell),
                                   fill=col, outline="", tags="game")
            c.create_rectangle(*box(x, y, cell * (0.08 if i == 0 else 0.14)), fill=col, outline="", tags="game")
        hx, hy = g.body[0]
        dx, dy = DIRS[g.heading]
        for side in (-1, 1):
            ex = bx + (hx + 0.5 + dx * 0.2 + side * dy * 0.2) * cell
            ey = by + (hy + 0.5 + dy * 0.2 + side * dx * 0.2) * cell
            c.create_oval(*self.P(ex - 2.5, ey - 2.5, ex + 2.5, ey + 2.5), fill=BG, outline="", tags="game")
        if self.planned and g.alive:
            px, py = DIRS[self.planned]
            c.create_line(*self.P(bx + (hx + 0.5) * cell, by + (hy + 0.5) * cell, bx + (hx + 0.5 + px * 0.95) * cell, by + (hy + 0.5 + py * 0.95) * cell),
                          fill=AMBER, width=3 * S, arrow="last", arrowshape=(8 * S, 10 * S, 4 * S), tags="game")
        if not g.alive:
            c.create_rectangle(*self.P(bx + 40, by + bs / 2 - 34, bx + bs - 40, by + bs / 2 + 34), fill=BG, outline=RED, width=2 * S, tags="game")
            self.text(bx + bs / 2, by + bs / 2 - 12, "GAME OVER", 17, RED, anchor="center", bold=True, tags="game")
            self.text(bx + bs / 2, by + bs / 2 + 14, f"{g.cause} · score {g.score}", 11, TXT, anchor="center", tags="game")
        # status under the board
        speed = f"{self.last_ms / 1000:.2f} s/decision" if self.last_ms else "…"
        mode = "PAUSED" if self.paused else "AUTO"
        self.text(14, 462, f"LEN {len(g.body):<3} SCORE {g.score:<3} BEST {g.best:<3} MOVE {g.moves:<4} HUNGER {g.hunger}/{STARVE}",
                  10.5, GREEN, tags="game")
        name = self.kev.name if self.kev else "kev-0.8b"
        self.text(14, 474, f"{mode} · {name} · {speed}", 9.5, TXT_DIM if not self.paused else AMBER, tags="game")

    # ── decisions panel
    def draw_panel(self):
        c = self.cv
        c.delete("panel")
        x0, y, x1 = 452, 60, W - 22
        if self.error:
            self.text(x0, y, self.error, 11, RED, tags="panel", width=x1 - x0)
            return
        if self.kev is None:
            self.text(x0, y, "LOADING KEV-0.8B", 14, GREEN, bold=True, tags="panel")
            self.text(x0, y + 26, self.status, 11, TXT, tags="panel", width=x1 - x0)
            self.text(x0, y + 60, "The first run downloads the adapter and the Qwen3.5-0.8B base (~1.7 GB).\n"
                                  "Progress is printed in the terminal.", 10, TXT_DIM, tags="panel", width=x1 - x0)
            return
        if self.show_state:
            self.text(x0, y, "SITUATION REPORT  (the exact state text Kev reads)", 9.5, AMBER, tags="panel")
            self.text(x0, y + 16, self.state or "…", 9.5, TXT, tags="panel", width=x1 - x0)
            return
        bar_x, bar_w = x0 + 132, 210
        for qid, tag in PANEL_ORDER:
            a = self.answers.get(qid)
            c.create_rectangle(*self.P(x0, y + 1, x0 + 50, y + 14), outline=AMBER if a else AMBER_DIM, tags="panel")
            self.text(x0 + 25, y + 7.5, tag, 8.5, AMBER if a else AMBER_DIM, anchor="center", bold=True, tags="panel")
            if not a:
                self.text(x0 + 58, y + 1, "thinking…" if self.busy else "", 9.5, TXT_DIM, tags="panel")
                y += 36
                continue
            meta = f"{a['ms']:.0f} ms" if a["ms"] else a.get("note", "")
            if "score" in a: meta = f"score {a['score']:.2f} · " + meta
            it = self.text(x0 + 58, y + 1, a["instr"], 9.5, TXT, tags="panel", width=x1 - x0 - 150)
            self.text(x1, y + 1, meta, 9, TXT_DIM, anchor="ne", tags="panel")
            bb = c.bbox(it)
            y = max(y + 18, bb[3] / self.S + 4)
            for i, (lab, desc, p) in enumerate(zip(a["labels"], a["descs"], a["probs"])):
                pick = i == a["pick"]
                col = GREEN if pick else TXT_DIM
                self.text(x0 + 8, y, ("▸ " if pick else "  ") + lab[:16], 9.5, col, bold=pick, tags="panel")
                c.create_rectangle(*self.P(bar_x, y + 3, bar_x + bar_w, y + 11), fill=GREEN_FAINT, outline="", tags="panel")
                c.create_rectangle(*self.P(bar_x, y + 3, bar_x + max(1, bar_w * p), y + 11), fill=GREEN if pick else "#2c7a44", outline="", tags="panel")
                self.text(bar_x + bar_w + 8, y, f"{p * 100:5.1f}%", 9.5, col, tags="panel")
                d = desc if len(desc) <= 62 else desc[:61] + "…"
                self.text(bar_x + bar_w + 62, y, d, 8.5, TXT_DIM if not pick else TXT, tags="panel")
                y += 15
            y += 9

    # ── controls
    def toggle_pause(self, _=None):
        self.paused = not self.paused; self.draw_game()

    def step(self, _=None):
        self.paused = True; self.step_once = True; self.draw_game()

    def restart(self, _=None):
        if self.busy: self.restart_pending = True; return
        self.game.reset(); self.planned = None; self.alert = 0; self.answers = {}
        self.reset_graph(); self.draw_game(); self.draw_panel()

    def faster(self, _=None):
        self.move_delay = max(0.0, self.move_delay - 0.1)

    def slower(self, _=None):
        self.move_delay = min(3.0, self.move_delay + 0.1)

    def toggle_fire(self, _=None):
        self.fire_gap = {0.11: 0.35, 0.35: 0.02}.get(round(self.fire_gap, 2), 0.11)

    def toggle_state(self, _=None):
        self.show_state = not self.show_state; self.draw_panel()

    # ── main loop
    def start_think(self):
        self.busy, self.answers, self.planned = True, {}, None
        self.reset_graph(); self.draw_panel()
        snapshot = copy.deepcopy(self.game)
        emit = lambda *e: self.events.put(e)

        def work():
            try:
                think(snapshot, self.kev, emit, self.args.mock)
            except Exception as e:
                traceback.print_exc()
                emit("error", f"{type(e).__name__}: {e}")
        threading.Thread(target=work, daemon=True).start()

    def handle(self, ev):
        """Apply one worker event. Returns the pause before the next event (so fast GPUs still show the firing)."""
        kind = ev[0]
        if kind == "status":
            self.status = ev[1]; self.draw_panel(); return 0
        if kind == "loaded":
            self.draw_panel(); self.draw_game(); return 0
        if kind == "error":
            self.error, self.busy = ev[1], False; self.draw_panel(); return 0
        if kind == "state":
            self.state = ev[1]
            if self.show_state: self.draw_panel()
            return 0
        if kind == "answer":
            self.answers[ev[1]] = ev[2]; self.draw_panel(); return 0
        if kind == "fire":
            nid, value, sources = ev[1], ev[2], ev[3]
            self.fire(nid, value, sources, ev[4] if len(ev) > 4 else None)
            if nid == "ALERT":
                self.alert = ev[4]; self.draw_game()
            if nid == "HEADING":
                self.draw_game()
            if nid == "heading":
                self.planned = value; self.draw_game()
            return self.fire_gap * (0.3 if NODES[nid][0] == "input" else 1)
        if kind == "done":
            self.last_ms = ev[2]["ms"]
            self.game.step(ev[1])
            self.planned, self.busy = None, False
            if getattr(self, "restart_pending", False):
                self.restart_pending = False; self.restart()
            self.draw_game()
            self.next_think = time.time() + self.move_delay + (2.5 if not self.game.alive else 0)
            return 0
        return 0

    def pump(self):
        now = time.time()
        while now >= self.next_event:
            try: ev = self.events.get_nowait()
            except queue.Empty: break
            gap = self.handle(ev)
            if gap: self.next_event = now + gap; break
        if self.kev and not self.busy and not self.error and now >= self.next_think and self.events.empty():
            if not self.game.alive:
                if not self.paused or self.step_once:
                    self.restart()
                    self.next_think = now + 0.4
            elif not self.paused or self.step_once:
                self.step_once = False
                self.start_think()
        self.animate()
        self.root.after(16, self.pump)

    def run(self):
        self.root.mainloop()


def main():
    ap = argparse.ArgumentParser(description="Kev-0.8B plays Snake, with its decision graph firing live.")
    ap.add_argument("--mock", action="store_true", help="fake random model (preview the UI without downloading Kev)")
    ap.add_argument("--headless", type=int, metavar="N", help="no window: play N moves and print the decisions")
    ap.add_argument("--device", help="cuda / cpu (default: cuda if available)")
    ap.add_argument("--threads", type=int, help="torch CPU threads")
    ap.add_argument("--grid", type=int, default=12, help="board size (default 12)")
    ap.add_argument("--seed", type=int, help="random seed for apple placement")
    ap.add_argument("--move-ms", type=int, default=250, help="pause after each move (default 250)")
    ap.add_argument("--fire-ms", type=int, default=110, help="minimum time between graph nodes firing (default 110)")
    ap.add_argument("--zoom", type=float, help="UI scale (default: follow screen DPI)")
    args = ap.parse_args()
    if args.headless:
        kev = MockKev() if args.mock else Kev(device=args.device, threads=args.threads)
        run_headless(kev, args.headless, args.grid, args.mock, args.seed)
        return
    App(args).run()


if __name__ == "__main__":
    main()
