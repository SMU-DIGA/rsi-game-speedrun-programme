# ==================================================================================================
# Super Mario Bros. (NES) -- level 4-3  |  超级马里奥兄弟（NES） 关 4-3
# Pass rate on 100 random variants never used before: 100/100; fastest finish 880 frames (14.67 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 880 帧（60 帧/秒，约 14.67 秒）。
#
# What this file is / 这是什么:
#   A game-playing policy written by an AI agent (Claude Sonnet). Below this header the code is the
#   agent's own file, unchanged (only local paths are masked). The agent's comments are kept.
#   这是 AI 代理（Claude Sonnet）写的通关策略。本说明以下是代理写的原文件，未改动（只隐去了本地路径），代理自己的注释都保留。
#
# How it is run / 怎么运行:
#   The referee calls play(sim) once on a fresh random variant. Every frame the policy may read
#   sim.screen() -- the RGB pixels of the game window, nothing else (no game memory, no positions) --
#   and must call sim.step(keys) to hold a set of keys for one frame (keys: Left / Right / Up / Down, z = jump (A), x = run (B)).
#   Death or finishing is judged by the referee, not reported to the policy.
#   裁判在一个新的随机变体上调用一次 play(sim)。策略每一帧只能用 sim.screen() 读取游戏画面像素（读不到游戏内存和坐标），
#   用 sim.step(keys) 按住一组按键走一帧。死亡还是通关由裁判判定，不告诉策略。
#
# The agent's summary of its approach / 代理自己对思路的概括:
#   PB trace -> position-triggered macro plan (run / jump groups); save/load local search over tri
#   gger offset + hold lengths per slow group, scored by frames to a common reference state; verif
#   ied by repla
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def enc                            line 16    
#   def dec                            line 20    
#   def rle                            line 24    
#   def unrle                          line 39    
#   def _m                             line 74    ---------------------------------------------------------------- vision
#   def find_mario                     line 78    
#   def nonsky                         line 87    
#   def scroll_shift                   line 91    
#   class Env                          line 101   sim wrapper with screen-based tracking: world x (scroll + mario right edge), feet row, freeze detection
#   def tokens                         line 170   ---------------------------------------------------------------- plan
#   def mkgrp                          line 180   
#   def record                         line 188   
#   def build_items                    line 207   
#   class Player                       line 263   
#   def run_eval                       line 297   ---------------------------------------------------------------- local search
#   def run_roll                       line 339   full rollout from the checkpoint to the finish: (frames from checkpoint to finish or None, player item idx)
#   def explore                        line 366   caller has saved SLOT and ctx['snap'] at the checkpoint; returns candidates (score, group), best first
#   def chain_try                      line 439   cand (locally much faster) fails its plain rollout: tentatively apply it, then repair the group where
#   def sweep_old                      line 522   
#   def sweep                          line 614   forward pass; at each eligible group: local screening (save/load), then full rollout to the finish of the
#   def validate                       line 704   
#   def solve                          line 715   
#   def play                           line 765   
#
# This policy also reads sim.memory; that data is in 4-3.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 4-3.memory.json。
# ==================================================================================================

# idea: PB trace -> position-triggered macro plan (run / jump groups); save/load local search over trigger offset + hold lengths per slow group, scored by frames to a common reference state; verified by replays
import time
import traceback
import numpy as np

SKY = (104, 136, 252)
BROWN = (172, 124, 0)
FIN_STILL = 5
FIN_WX = 2200
SLOT = 1
BIT = {"Right": 1, "x": 2, "z": 4, "Left": 8}

EMBED = {}


def enc(keys):
    return sum(BIT[k] for k in keys if k in BIT)


def dec(n):
    return {k for k, b in BIT.items() if n & b}


def rle(trace):
    out = []
    prev, cnt = None, 0
    for e in trace:
        if e == prev:
            cnt += 1
        else:
            if prev is not None:
                out.append("%d*%d" % (prev, cnt))
            prev, cnt = e, 1
    if prev is not None:
        out.append("%d*%d" % (prev, cnt))
    return ",".join(out)


def unrle(txt):
    res = []
    for tok in txt.split(","):
        if tok:
            a, b = tok.split("*")
            res.extend([int(a)] * int(b))
    return res


PREFIX_RLE = ("3*77,7*4,3*29,7*19,3*14,7*5,14*1,7*1,14*9,7*2,3*28,7*36,3*8,7*2,14*1,7*1,3*1,10*1,3*45,7*4,3*35,7*4,"
              "3*33,7*10,3*34,10*1,3*35,7*15,3*14,7*2,14*1,7*15,3*13,7*10,3*15,10*2,3*13,7*4,3*17,10*1,3*9,10*5,"
              "3*8,7*2,14*2,7*40,10*2,3*15,7*4,14*1,7*2,14*1,7*10,3*18,7*10,3*15,7*3,14*2,7*3,3*17,7*10,3*19,7*2,"
              "14*12,3*13,14*4,10*2,3*33,7*10,3*17,7*3,14*1,7*5,14*1,7*8,3*58,7*10,3*2")
EMBED[941] = PREFIX_RLE
EMBED[881] = ("3*77,7*4,3*29,7*19,3*14,7*5,14*1,7*1,14*9,7*2,3*28,7*36,3*8,7*2,14*1,7*1,3*1,10*1,3*45,7*4,3*35,7*4,"
              "3*33,7*10,3*34,10*1,3*35,7*11,3*11,7*2,14*1,7*15,3*6,7*2,3*21,10*2,3*1,7*4,3*17,10*1,3*9,10*5,3*8,"
              "7*2,14*2,7*40,10*2,3*16,7*4,14*1,7*2,14*1,7*10,3*17,7*10,3*11,7*3,14*2,7*3,3*24,7*10,3*19,7*10,14*11,"
              "7*4,3*19,7*10,3*15,7*3,14*1,7*5,14*1,7*8,3*54,7*8")
SKIP_P = [238]
MAX_CHAIN = 0
EMBED[887] = ("3*77,7*4,3*29,7*19,3*14,7*5,14*1,7*1,14*9,7*2,3*28,7*36,3*8,7*2,14*1,7*1,3*1,10*1,3*45,7*4,3*35,7*4,"
              "3*33,7*10,3*34,10*1,3*35,7*11,3*11,7*2,14*1,7*15,3*6,7*2,3*21,10*2,3*1,7*4,3*17,10*1,3*9,10*5,3*8,"
              "7*2,14*2,7*40,10*2,3*16,7*4,14*1,7*2,14*1,7*10,3*17,7*10,3*11,7*3,14*2,7*3,3*24,7*10,3*17,7*2,14*12,"
              "3*7,7*4,3*27,7*10,3*15,7*3,14*1,7*5,14*1,7*8,3*54,7*8")
EMBED[926] = ("3*77,7*4,3*29,7*19,3*14,7*5,14*1,7*1,14*9,7*2,3*28,7*36,3*8,7*2,14*1,7*1,3*1,10*1,3*45,7*4,3*35,7*4,"
              "3*33,7*10,3*34,10*1,3*35,7*15,3*14,7*2,14*1,7*15,3*12,7*2,3*26,10*2,3*8,7*4,3*16,10*1,3*9,10*5,3*8,"
              "7*2,14*2,7*40,10*2,3*16,7*4,14*1,7*2,14*1,7*10,3*17,7*10,3*15,7*3,14*2,7*3,3*18,7*10,3*19,7*2,14*12,"
              "3*12,14*4,10*2,3*34,7*10,3*17,7*3,14*1,7*5,14*1,7*8,3*53,7*8")
EMBED[930] = ("3*77,7*4,3*29,7*19,3*14,7*5,14*1,7*1,14*9,7*2,3*28,7*36,3*8,7*2,14*1,7*1,3*1,10*1,3*45,7*4,3*35,7*4,"
              "3*33,7*10,3*34,10*1,3*35,7*15,3*14,7*2,14*1,7*15,3*13,7*10,3*15,10*2,3*13,7*4,3*17,10*1,3*9,10*5,"
              "3*8,7*2,14*2,7*40,10*2,3*15,7*4,14*1,7*2,14*1,7*10,3*18,7*10,3*15,7*3,14*2,7*3,3*17,7*10,3*19,7*2,"
              "14*12,3*13,14*4,10*2,3*33,7*10,3*17,7*3,14*1,7*5,14*1,7*8,3*54,7*8")


# ---------------------------------------------------------------- vision
def _m(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def find_mario(s):
    m = _m(s, BROWN)
    m[:40] = False
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def nonsky(s):
    return (s != np.array(SKY, np.uint8)).any(axis=2)[40:232]


def scroll_shift(pn, cn):
    W = cn.shape[1]
    best, bk = None, 0
    for k in range(0, 9):
        mm = int((cn[:, :W - k] != pn[:, k:]).sum())
        if best is None or mm < best - 8:
            best, bk = mm, k
    return bk


class Env:
    """sim wrapper with screen-based tracking: world x (scroll + mario right edge), feet row, freeze detection"""

    def __init__(self, sim):
        self.sim = sim
        self.reset_state()

    def reset_state(self):
        self.scroll = 0
        self.prev_ns = None
        self.last_mr = 0
        self.hist = []
        self.wxh = []
        self.wx = 0
        self.y = -1
        self.vis = False
        self.prev = None
        self.still = 0
        self.n = 0

    def observe(self):
        s = np.array(self.sim.screen())
        if self.prev is not None and np.array_equal(s, self.prev):
            self.still += 1
        else:
            self.still = 0
        self.prev = s
        cn = nonsky(s)
        if self.prev_ns is not None:
            self.scroll += scroll_shift(self.prev_ns, cn)
        self.prev_ns = cn
        m = find_mario(s)
        if m is not None:
            self.last_mr = m[1]
            self.vis = True
            self.y = m[3]
        else:
            self.vis = False
            self.y = -1
        self.wx = self.scroll + self.last_mr
        self.hist = (self.hist + [self.y])[-4:]
        self.wxh = (self.wxh + [self.wx])[-8:]

    def step(self, k, log=None):
        self.sim.step(dec(k))
        self.n += 1
        if log is not None:
            log.append(k)

    def grounded(self):
        h = self.hist
        return len(h) >= 2 and h[-1] > 0 and h[-1] == h[-2]

    def v6(self):
        w = self.wxh
        return (w[-1] - w[-7]) if len(w) >= 7 else 0

    def snap(self):
        return (self.scroll, self.prev_ns, self.last_mr, list(self.hist), list(self.wxh), self.wx, self.y,
                self.vis, self.prev, self.still, self.n)

    def restore(self, st):
        (self.scroll, self.prev_ns, self.last_mr, h, w, self.wx, self.y, self.vis, self.prev, self.still,
         self.n) = st
        self.hist = list(h)
        self.wxh = list(w)


# ---------------------------------------------------------------- plan
def tokens(T):
    out = []
    for i, k in enumerate(T):
        if out and out[-1][0] == k:
            out[-1][1] += 1
        else:
            out.append([k, 1, i])
    return out


def mkgrp(base, toks, dP):
    g = dict(base)
    g["toks"] = [list(t) for t in toks if t[1] > 0]
    g["dP"] = dP
    g["flat"] = [k for k, c in g["toks"] for _ in range(c)]
    return g


def record(sim, env, T):
    sim.reset()
    env.reset_state()
    rec = {"wx": [], "y": [], "gr": [], "v6": []}
    score = None
    for k in list(T) + [3] * 14:
        env.observe()
        if env.still >= FIN_STILL:
            if env.wx > FIN_WX:
                score = env.n - env.still + 1
            break
        rec["wx"].append(env.wx)
        rec["y"].append(env.y)
        rec["gr"].append(env.grounded())
        rec["v6"].append(env.v6())
        env.step(k)
    return rec, score


def build_items(T, rec):
    items = []
    cur = None
    for k, c, s in tokens(T):
        if k == 3 and c >= 6:
            if cur is not None:
                items.append(cur)
                cur = None
            items.append({"t": "r", "start": s, "count": c})
        else:
            if cur is None:
                cur = {"t": "g", "start": s, "toks": [], "dP": 0}
            cur["toks"].append([k, c])
    if cur is not None:
        items.append(cur)
    N = min(len(T), len(rec["wx"]))
    for gi, it in enumerate(items):
        if it["t"] != "g":
            continue
        n = sum(c for k, c in it["toks"])
        it["end"] = it["start"] + n
        it["P"] = rec["wx"][it["start"]]
        it["ng"] = rec["gr"][it["start"]]
    for gi, it in enumerate(items):
        if it["t"] != "g":
            continue
        egi = gi + 2 if (gi + 2 < len(items) and items[gi + 2]["t"] == "g") else gi
        e = items[egi]["end"]
        it["egi"] = egi
        it["ref"] = None
        s = it["start"]
        if egi + 1 < len(items) and items[egi + 1]["t"] == "r" and e < N - 1:
            rc = items[egi + 1]["count"]
            lim = min(e + rc, N) - 1
            found = None
            for i in range(max(e, 6), lim + 1):
                if rec["gr"][i] and rec["wx"][i] - rec["wx"][i - 6] >= 17:
                    found = i
                    break
            if found is None:
                found = min(e + min(rc, 12), N - 1)
                v = max(0, min(rec["v6"][found], 17) - 1)
            else:
                v = 16
            if rec["y"][found] < 0:
                v = -99
            it["ref"] = {"wx": rec["wx"][found], "y": rec["y"][found], "g": rec["gr"][found], "v": v, "idx": found}
            it["loss"] = (found - s) - (rec["wx"][found] - rec["wx"][s]) / 3.0
        else:
            it["loss"] = (N - s) - (rec["wx"][N - 1] - rec["wx"][s]) / 3.0
    for i in range(len(items)):
        if items[i]["t"] == "g":
            items[i] = mkgrp(items[i], items[i]["toks"], 0)
    return items


class Player:
    def __init__(self, items, i):
        self.items = items
        self.i = i
        self.sub = 0

    def done(self, egi):
        return self.i > egi or (self.i == egi and self.sub >= len(self.items[egi]["flat"]))

    def keys(self, wx, grounded):
        items = self.items
        while True:
            if self.i >= len(items):
                return 3
            it = items[self.i]
            if it["t"] == "r":
                if self.i + 1 >= len(items):
                    return 3
                nx = items[self.i + 1]
                if wx >= nx["P"] + nx["dP"] and (grounded or not nx["ng"]):
                    self.i += 1
                    self.sub = 0
                    continue
                return 3
            fl = it["flat"]
            if self.sub < len(fl):
                k = fl[self.sub]
                self.sub += 1
                return k
            self.i += 1
            self.sub = 0


# ---------------------------------------------------------------- local search
def run_eval(ctx, gi, cand, cap):
    sim, env, items = ctx["sim"], ctx["env"], ctx["items"]
    sim.load(ctx["slot"])
    env.restore(ctx["snap"])
    items2 = list(items)
    items2[gi] = cand
    egi = items[gi]["egi"]
    ref = items[gi]["ref"]
    pl = Player(items2, gi - 1)
    el = 0
    inv = 0
    ctx["nev"] += 1
    why = ""
    while True:
        if env.still >= FIN_STILL:
            if env.wx > FIN_WX:
                return float(el - env.still + 1)
            ctx["why"] = "froze wx=%d y=%d el=%d last=%s refwx=%s" % (env.wx, env.y, el, why,
                                                                      None if ref is None else ref["wx"])
            return None
        if el > cap:
            ctx["why"] = "cap " + why
            return None
        inv = 0 if env.vis else inv + 1
        if inv > 60:
            ctx["why"] = "invisible"
            return None
        if pl.done(egi):
            k = 3
            if ref is not None and env.wx >= ref["wx"]:
                if (abs(env.y - ref["y"]) <= 1 and (env.grounded() or not ref["g"])
                        and env.v6() >= ref["v"]):
                    return el - (env.wx - ref["wx"]) / 3.0
                why = "ref y=%d/%d gr=%s/%s v6=%d/%d" % (env.y, ref["y"], env.grounded(), ref["g"], env.v6(),
                                                          ref["v"])
        else:
            k = pl.keys(env.wx, env.grounded())
        env.step(k)
        el += 1
        env.observe()


def run_roll(ctx, gi, cand, cap):
    """full rollout from the checkpoint to the finish: (frames from checkpoint to finish or None, player item idx)"""
    sim, env, items = ctx["sim"], ctx["env"], ctx["items"]
    sim.load(ctx["slot"])
    env.restore(ctx["snap"])
    items2 = list(items)
    items2[gi] = cand
    pl = Player(items2, gi - 1)
    el = 0
    inv = 0
    ctx["nroll"] += 1
    while True:
        if env.still >= FIN_STILL:
            if env.wx > FIN_WX:
                return el - env.still + 1, pl.i
            return None, pl.i
        if el > cap:
            return None, -1
        inv = 0 if env.vis else inv + 1
        if inv > 60:
            return None, pl.i
        k = pl.keys(env.wx, env.grounded())
        env.step(k)
        el += 1
        env.observe()


def explore(ctx, gi, base, budget_g):
    """caller has saved SLOT and ctx['snap'] at the checkpoint; returns candidates (score, group), best first"""
    sim, env, items = ctx["sim"], ctx["env"], ctx["items"]
    it = items[gi]
    b0 = sim.budget_left
    ctx["nev"] = 0
    cur = it
    best = base
    found = []

    def left():
        return budget_g - (b0 - sim.budget_left)

    def attempt(cand):
        nonlocal best, cur
        if left() <= 0:
            return False
        sc = run_eval(ctx, gi, cand, min(int(best) + 14, 230))
        if sc is not None and sc < base - 0.5:
            found.append((sc, cand))
        if sc is not None and sc < best - 0.5:
            best = sc
            cur = cand
            return True
        return False

    dP0 = it["dP"]
    dlist = [-3, -6, -9, -12, -15, -18, -24, -30, -36, 3, 6, 9]
    if it["ref"] is None:
        dlist += [-21, -27, -42, -48, -54]
    for d in dlist:
        attempt(mkgrp(it, it["toks"], dP0 + d))
    ti = 0
    while ti < len(cur["toks"]):
        k, c = cur["toks"][ti]
        variants = []
        if k & 4:
            for dc in (-1, 1, -2, 2, -4, 4, -8, 8):
                if c + dc >= 1:
                    variants.append((k, c + dc))
        if k & 8:
            variants.append((k, 0))
            variants.append(((k & ~8) | 1, c))
            if c > 1:
                variants.append((k, c - 1))
                variants.append((k, c + 1))
        if k == 3 and c >= 2:
            variants.append((k, c - 1))
        for (k2, c2) in variants:
            if ti >= len(cur["toks"]):
                break
            toks = [list(t) for t in cur["toks"]]
            toks[ti] = [k2, c2]
            attempt(mkgrp(cur, toks, cur["dP"]))
        ti += 1
    if len(cur["toks"]) <= 4:
        for where in ("start", "end"):
            for tk in ((14, 1), (10, 1), (7, 1), (3, 1)):
                toks = [list(t) for t in cur["toks"]]
                toks.insert(0 if where == "start" else len(toks), list(tk))
                attempt(mkgrp(cur, toks, cur["dP"]))
    d0 = cur["dP"]
    for d in (-3, 3, -6, 6):
        attempt(mkgrp(cur, cur["toks"], d0 + d))
    print("  g%d start=%d loss=%.1f base=%.1f best=%.1f evals=%d used=%d/%d dP=%d toks=%s" % (
        gi, it["start"], it["loss"], base, best, ctx["nev"], b0 - sim.budget_left, budget_g, cur["dP"],
        cur["toks"]))
    sim.load(ctx["slot"])
    env.restore(ctx["snap"])
    found.sort(key=lambda t: t[0])
    return found


def chain_try(ctx, gi, cand, n0, total_cur, di, reserve):
    """cand (locally much faster) fails its plain rollout: tentatively apply it, then repair the group where
    Mario dies (local search from a checkpoint under the new timeline). returns new total or None"""
    sim, env, items = ctx["sim"], ctx["env"], ctx["items"]
    snap1 = ctx["snap"]
    orig = {}

    def setg(g, c):
        if g not in orig:
            orig[g] = items[g]
        items[g] = c

    def revert():
        for g, o in orig.items():
            items[g] = o
        ctx["slot"] = SLOT
        ctx["snap"] = snap1

    setg(gi, cand)
    cap = total_cur - n0 - 1
    repaired = set()
    for depth in range(3):
        if di is None:
            rel, di = run_roll(ctx, gi, items[gi], cap)
            if rel is not None:
                if n0 + rel < total_cur:
                    print("   chain OK depth=%d changes=%s total=%d" % (depth, sorted(orig), n0 + rel))
                    ctx["slot"] = SLOT
                    ctx["snap"] = snap1
                    return n0 + rel
                print("   chain depth=%d finishes but total %d >= %d" % (depth, n0 + rel, total_cur))
                revert()
                return None
            print("   chain depth=%d rollout died, item %s" % (depth, di))
        if di is None or di < 0:
            revert()
            return None
        gd = di if (di < len(items) and items[di]["t"] == "g") else di - 1
        print("   chain depth=%d died at item %d -> repair g%d" % (depth, di, gd))
        if gd <= gi or gd in repaired or sim.budget_left < reserve + 2500:
            revert()
            return None
        repaired.add(gd)
        # advance to the checkpoint of gd under the current (tentative) plan
        sim.load(SLOT)
        env.restore(snap1)
        pl = Player(items, gi - 1)
        nx = items[gd]
        ok = False
        for _ in range(500):
            if env.still >= FIN_STILL:
                break
            if pl.i == gd - 1:
                ok = env.wx < nx["P"] + nx["dP"] + 6
                break
            if pl.i >= gd:
                break
            k = pl.keys(env.wx, env.grounded())
            env.step(k)
            env.observe()
        if not ok:
            print("   chain: cannot reach checkpoint of g%d (pl.i=%d wx=%d P=%d still=%d)" % (
                gd, pl.i, env.wx, nx["P"], env.still))
            revert()
            return None
        sim.save(2)
        ctx["slot"] = 2
        ctx["snap"] = env.snap()
        found = explore(ctx, gd, 1e9, 1300)
        ctx["slot"] = SLOT
        ctx["snap"] = snap1
        if not found:
            print("   chain: no valid repair for g%d" % gd)
            revert()
            return None
        setg(gd, found[0][1])
        print("   chain: repair g%d dP=%d toks=%s local=%.1f" % (gd, found[0][1]["dP"], found[0][1]["toks"],
                                                              found[0][0]))
        di = None
    revert()
    return None


def sweep_old(sim, env, items, ctx, reserve):
    sim.reset()
    env.reset_state()
    log = []
    pl = Player(items, 0)
    explored = set()
    elig = [gi for gi, it in enumerate(items) if it["t"] == "g" and it["loss"] >= 2.5]
    ctx["elig"] = set(elig)
    print("eligible groups:", [(gi, items[gi]["start"], round(items[gi]["loss"], 1)) for gi in elig])
    last_i = -1
    stack = []
    free_slots = [2, 3, 4, 5, 6, 7, 8, 9]
    ngroups = sum(1 for it in items if it["t"] == "g")
    env.observe()
    while sim.budget_left > 50 and env.n < 1700:
        if env.still >= FIN_STILL:
            break
        i = pl.i
        if i != last_i:
            if i < len(items) and items[i]["t"] == "g":
                print("  fire g%d at n=%d (base start %d) wx=%d P=%d" % (i, env.n, items[i]["start"], env.wx,
                                                                         items[i]["P"]))
            last_i = i
        if i < len(items) and items[i]["t"] == "r" and i + 1 < len(items):
            gi = i + 1
            nx = items[gi]
            if (gi not in explored and env.wx >= nx["P"] + nx["dP"] - 24
                    and env.wx < nx["P"] + nx["dP"] + 6 and sim.budget_left > reserve):
                explored.add(gi)
                sim.save(SLOT)
                snap = env.snap()
                ctx["snap"] = snap
                ctx["nev"] = 0
                ctx["why"] = ""
                loglen = len(log)
                base = run_eval(ctx, gi, items[gi], 230)
                if base is None:
                    print("  chk g%d n=%d INVALID: %s (stack=%s)" % (gi, env.n, ctx["why"], [e["gi"] for e in stack]))
                    if stack:
                        ent = stack.pop()
                        items[ent["gi"]] = ent["old"]
                        print("  ROLLBACK g%d" % ent["gi"])
                        sim.load(ent["slot"])
                        env.restore(ent["snap"])
                        free_slots.append(ent["slot"])
                        del log[ent["loglen"]:]
                        pl.i = ent["gi"] - 1
                        pl.sub = 0
                        explored = set(g for g in explored if g <= ent["gi"])
                        last_i = -1
                    else:
                        sim.load(SLOT)
                        env.restore(snap)
                    continue
                # valid: confirm older changes
                for e in list(stack):
                    if gi >= e["gi"] + 2:
                        e["confirm"] += 1
                        if e["confirm"] >= 3:
                            stack.remove(e)
                            free_slots.append(e["slot"])
                if gi in ctx["elig"]:
                    rem = sum(items[g]["loss"] for g in ctx["elig"] if (g not in explored) or g == gi)
                    left_groups = sum(1 for g in range(gi + 1, len(items)) if items[g]["t"] == "g")
                    pool = sim.budget_left - reserve - (1000 - env.n) - 90 * left_groups
                    if pool > 150 and time.time() - ctx["t0"] < 1300:
                        share = pool * items[gi]["loss"] / max(rem, 1.0)
                        old = items[gi]
                        cur = explore(ctx, gi, base, int(share))
                        if cur is not old:
                            items[gi] = cur
                            if not free_slots:
                                e0 = stack.pop(0)
                                free_slots.append(e0["slot"])
                            slot = free_slots.pop(0)
                            sim.save(slot)
                            stack.append({"gi": gi, "slot": slot, "snap": snap, "loglen": loglen, "old": old,
                                          "confirm": 0})
                        last_i = -1
                    else:
                        sim.load(SLOT)
                        env.restore(snap)
                else:
                    sim.load(SLOT)
                    env.restore(snap)
        k = pl.keys(env.wx, env.grounded())
        env.step(k, log)
        env.observe()
    score = env.n - env.still + 1
    return log, score, (env.still >= FIN_STILL and env.wx > FIN_WX)


def sweep(sim, env, items, ctx, reserve):
    """forward pass; at each eligible group: local screening (save/load), then full rollout to the finish of the
    best candidates; a change is accepted only if the whole rollout finishes strictly earlier."""
    sim.reset()
    env.reset_state()
    log = []
    pl = Player(items, 0)
    explored = set()
    elig = [gi for gi, it in enumerate(items) if it["t"] == "g" and it["loss"] >= 2.5
            and not any(abs(it["P"] - p) <= 12 for p in SKIP_P)]
    ctx["elig"] = set(elig)
    ctx["nroll"] = 0
    ctx["nchain"] = 0
    ctx["slot"] = SLOT
    print("eligible groups:", [(gi, items[gi]["start"], round(items[gi]["loss"], 1)) for gi in elig])
    total_cur = None
    env.observe()
    nacc = 0
    while sim.budget_left > 50 and env.n < 1700:
        if env.still >= FIN_STILL:
            break
        i = pl.i
        if i < len(items) and items[i]["t"] == "r" and i + 1 < len(items):
            gi = i + 1
            nx = items[gi]
            lead = 72 if nx["ref"] is None else 42
            res_eff = reserve + (0 if nx["ref"] is None else ctx["last_res"])
            if (gi in ctx["elig"] and gi not in explored and env.wx >= nx["P"] + nx["dP"] - lead
                    and env.wx < nx["P"] + nx["dP"] + 6 and sim.budget_left > res_eff
                    and time.time() - ctx["t0"] < 1300):
                explored.add(gi)
                sim.save(SLOT)
                snap = env.snap()
                ctx["snap"] = snap
                n0 = env.n
                ok = True
                if total_cur is None:
                    rel, _di = run_roll(ctx, gi, items[gi], 1500)
                    if rel is None:
                        print("  initial rollout failed -> abort optimisation")
                        return log, None, False
                    total_cur = n0 + rel
                    print("  initial rollout from n=%d: total=%d" % (n0, total_cur))
                if ok:
                    ctx["nev"] = 0
                    ctx["why"] = ""
                    base = run_eval(ctx, gi, items[gi], 230)
                    if base is None:
                        print("  g%d n=%d local base invalid: %s" % (gi, n0, ctx["why"]))
                    else:
                        rem = sum(items[g]["loss"] for g in ctx["elig"] if (g not in explored) or g == gi)
                        pool = sim.budget_left - res_eff - (1000 - n0)
                        share = min(1800.0, 0.4 * pool * items[gi]["loss"] / max(rem, 1.0))
                        if nx["ref"] is None:
                            share = min(2500.0, 0.9 * pool)
                        found = []
                        if pool > 150:
                            found = explore(ctx, gi, base, int(share))
                        tried = 0
                        for sc, cand in found:
                            if tried >= 4 or sim.budget_left < reserve + 900:
                                break
                            tried += 1
                            rel, di = run_roll(ctx, gi, cand, total_cur - n0 - 1)
                            print("   roll g%d cand local=%.1f dP=%d toks=%s -> %s (cur total %d) di=%s" % (
                                gi, sc, cand["dP"], cand["toks"], None if rel is None else n0 + rel, total_cur,
                                di))
                            if rel is not None and n0 + rel < total_cur:
                                items[gi] = cand
                                total_cur = n0 + rel
                                nacc += 1
                                break
                            if (rel is None and di is not None and di >= 0 and base - sc >= 3 and ctx["nchain"] < MAX_CHAIN
                                    and sim.budget_left > reserve + 4000):
                                ctx["nchain"] += 1
                                res = chain_try(ctx, gi, cand, n0, total_cur, di, reserve)
                                if res is not None:
                                    total_cur = res
                                    nacc += 1
                                    break
                sim.load(SLOT)
                env.restore(snap)
        k = pl.keys(env.wx, env.grounded())
        env.step(k, log)
        env.observe()
    score = env.n - env.still + 1
    print("sweep: accepted %d changes, expected total %s, actual %d" % (nacc, total_cur, score))
    return log, score, (env.still >= FIN_STILL and env.wx > FIN_WX)


def validate(sim, env, trace):
    sim.reset()
    env.reset_state()
    for k in list(trace) + [3] * 30:
        env.observe()
        if env.still >= FIN_STILL:
            return (env.n - env.still + 1) if env.wx > FIN_WX else None
        env.step(k)
    return None


def solve(sim):
    t0 = time.time()
    try:
        env = Env(sim)
        pool = dict(EMBED)
        try:
            d = sim.memory.get("traces")
            if d:
                for k, v in d.items():
                    pool[int(k)] = v
        except Exception:
            pass
        bs = min(pool)
        T = unrle(pool[bs])
        print("start trace score", bs, "len", len(T))
        rec, score = record(sim, env, T)
        print("record: replay score", score)
        if score is None:
            return []
        items = build_items(T, rec)
        print("items:", [(i, it["t"], it["start"]) for i, it in enumerate(items)][:80])
        reserve = 2500
        ctx = {"sim": sim, "env": env, "items": items, "t0": t0, "nev": 0, "last_res": 1500}
        log, sc2, fin = sweep(sim, env, items, ctx, reserve)
        print("sweep done: finished=%s score=%s len(log)=%d budget_left=%d t=%.0f" % (
            fin, sc2, len(log), sim.budget_left, time.time() - t0))
        if fin and sc2 < score:
            trace = log[:sc2 + 3]
            res = []
            for _ in range(2):
                if sim.budget_left < 1200:
                    break
                res.append(validate(sim, env, trace))
            print("validation:", res)
            if len(res) == 2 and all(r is not None for r in res):
                worst = max(res)
                if worst < score:
                    d = sim.memory.get("traces") or {}
                    d[str(worst)] = rle(trace)
                    sim.memory["traces"] = d
                    print("NEW BEST %d  RLE: %s" % (worst, rle(trace)))
                    try:
                        sim.memory.flush()
                    except Exception:
                        pass
    except Exception:
        traceback.print_exc()
    return []


def play(sim):
    pool = dict(EMBED)
    try:
        d = sim.memory.get("traces")
        if d:
            for k, v in d.items():
                pool[int(k)] = v
    except Exception:
        pass
    tr = unrle(pool[min(pool)])
    prev = None
    still = 0
    i = 0
    while sim.budget_left > 0:
        k = dec(tr[i]) if i < len(tr) else dec(3)
        try:
            sim.step(k)
        except Exception:
            break
        i += 1
        try:
            s = sim.screen()
            if prev is not None and np.array_equal(s, prev):
                still += 1
            else:
                still = 0
            prev = np.array(s)
            if still >= 40:
                break
        except Exception:
            pass

