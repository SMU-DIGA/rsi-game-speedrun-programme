# ==================================================================================================
# Super Mario Bros. (NES) -- level 1-2  |  超级马里奥兄弟（NES） 关 1-2
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1347 frames (22.45 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1347 帧（60 帧/秒，约 22.45 秒）。
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
#   c0 reactive vision policy + greedy event-branching search (save/load rollback, backtracking) t
#   hat picks (delay,hold) per jump event; hints keyed by scroll
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _mask                          line 24    
#   class Agent                        line 28    
#   def _mem_E                         line 250   
#   def play                           line 264   
#   def _save_png                      line 280   
#   def solve                          line 288   
#
# This policy also reads sim.memory; that data is in 1-2.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 1-2.memory.json。
# ==================================================================================================

# idea: c0 reactive vision policy + greedy event-branching search (save/load rollback, backtracking) that picks (delay,hold) per jump event; hints keyed by scroll
import copy
import json
import time
import numpy as np

RED = (248, 56, 0)
BRN = (172, 124, 0)
IGNORE = [(228, 92, 16), (252, 160, 68), (248, 56, 0), (172, 124, 0)]  # coins / ?-blocks / mario colours
RUN = ("Right", "x")
PIPE_SCROLL = 2495
TOTAL_SCROLL = 2750
OPTS = [(24, 0), (14, 0), (8, 0), (4, 0), (24, 5), (8, 6)]   # (hold, delay)
HORIZON = 60
MARGIN = 9.0
RESERVE = 2200
OLD_J = [[1388, 18], [1493, 4], [1521, 4]]   # c0 hints (known to finish in 1545)
DEFAULT_E = [[62, 4, 0], [128, 14, 0], [224, 24, 0], [545, 8, 0], [771, 24, 0], [804, 24, 0], [1132, 8, 0],
             [1510, 8, 0], [1510, 24, 0], [1602, 24, 0], [1693, 4, 0], [1769, 24, 0], [1955, 24, 0],
             [2165, 24, 0], [2282, 24, 5], [2514, 4, 0], [2514, 24, 0], [2527, 24, 0], [2600, 24, 0],
             [2741, 24, 0]]  # [scroll, hold, delay]; verified 1347 frames in test #10


def _mask(s, col):
    return (s[:, :, 0] == col[0]) & (s[:, :, 1] == col[1]) & (s[:, :, 2] == col[2])


class Agent:
    def __init__(self, J=None, E=None, cb=None):
        self.P = dict(W_WALL=22, W_PIT=8, HOLD=24, W_PIPE=44)
        self.J = [list(j) for j in (J or [])]
        self.E = [list(e) for e in (E or [])]
        self.eptr = 0
        self.cb = cb
        self.jidx = 0
        self.prev = None
        self.scroll = 0
        self.last_shift = 0
        self.jt = None
        self.hold = 0
        self.pend = None
        self.prev_fy = None
        self.stable = 0
        self.cool = 0
        self.mx = None
        self.fy = None
        self.stuck = 0
        self.noprog = 0
        self.frame = 0
        self.nomario = 0
        self.lastscr = None
        self.frozen = 0
        self.phase = 0
        self.out0 = None
        self.hist = []
        self.fin = None
        self.bestw = -1
        self.gain_frame = 0

    # ---- perception ----
    def mario(self, s):
        m = (_mask(s, RED) | _mask(s, BRN))
        m[:40] = False
        ys, xs = np.nonzero(m)
        if len(ys) < 8:
            return None
        return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    def track_scroll(self, s):
        g = s.sum(axis=2, dtype=np.int32)
        rows = np.r_[34:46, 216:240]
        cur = g[rows]
        if self.prev is not None:
            pv = self.prev
            best, bs = None, 0
            for sh in range(0, 5):
                if sh == 0:
                    d = np.abs(pv - cur).sum()
                else:
                    d = np.abs(pv[:, sh:] - cur[:, :-sh]).sum() * (256.0 / (256 - sh))
                key = (d, abs(sh - self.last_shift))
                if best is None or key < best:
                    best, bs = key, sh
            self.scroll += bs
            self.last_shift = bs
        self.prev = cur

    def update_phase(self, s):
        if self.phase == 0 and self.scroll >= PIPE_SCROLL and self.frozen >= 5:
            self.phase = 1
        if self.phase < 2 and self.scroll >= PIPE_SCROLL:
            blue = (s[:, :, 2] > 235) & (s[:, :, 0] < 120) & (s[:, :, 1] > 120) & (s[:, :, 1] < 180)
            if blue.mean() > 0.3:
                self.phase = 2
                self.out0 = self.scroll
                self.hist = []
        if self.phase == 2 and self.fin is None:
            self.hist.append((self.scroll, self.mx))
            if len(self.hist) > 12 and self.scroll - self.out0 >= 150:
                h = self.hist[-11:]
                if all(x == h[0] for x in h):
                    self.fin = self.frame - 10

    def dead(self):
        if self.phase != 0:
            return False
        return self.frozen >= 5 or self.nomario >= 12 or self.stalled()

    def stalled(self):
        return self.phase == 0 and self.frame - self.gain_frame > 90

    # ---- events ----
    def find_event(self):
        while self.eptr < len(self.E) and self.E[self.eptr][0] < self.scroll - 15:
            self.eptr += 1
        if self.eptr < len(self.E) and abs(self.E[self.eptr][0] - self.scroll) <= 15:
            e = self.E[self.eptr]
            self.eptr += 1
            return e[1], e[2]
        return None

    def apply(self, h, d, keys):
        if d <= 0:
            self.start_jump(h, keys)
        else:
            self.pend = [d, h]

    def commit(self, h, d, keys):
        del self.E[self.eptr:]
        self.E.append([self.scroll, h, d])
        self.eptr = len(self.E)
        self.apply(h, d, keys)

    def trigger(self, keys):
        e = self.find_event()
        if e is not None:
            self.apply(e[0], e[1], keys)
        elif self.cb is not None:
            h, d = self.cb(self)
            self.commit(h, d, keys)
        else:
            self.start_jump(self.P["HOLD"], keys)
        return keys

    def start_jump(self, hold, keys):
        self.jt = 1
        self.hold = hold
        keys.add("z")

    def act(self, s):
        self.frame += 1
        self.track_scroll(s)
        mar = self.mario(s)
        self.update_phase(s)
        if self.last_shift == 0:
            self.noprog += 1
        else:
            self.noprog = 0
        if mar is None:
            self.nomario += 1
            return set(RUN)
        self.nomario = 0
        x0, x1, y0, y1 = mar
        self.mx, self.fy = x1, y1
        if self.scroll + x1 > self.bestw + 2:
            self.bestw = self.scroll + x1
            self.gain_frame = self.frame
        keys = set(RUN)
        if self.prev_fy is not None and y1 == self.prev_fy:
            self.stable += 1
        else:
            self.stable = 0
        self.prev_fy = y1
        if self.jt is not None:
            self.jt += 1
            if self.jt <= self.hold:
                keys.add("z")
                return keys
            self.jt = None
            self.cool = 3
            return keys
        if self.pend is not None:
            self.pend[0] -= 1
            if self.pend[0] <= 0:
                h = self.pend[1]
                self.pend = None
                self.start_jump(h, keys)
            return keys
        if self.cool > 0:
            self.cool -= 1
            return keys
        grounded = self.stable >= 2
        if not grounded:
            return keys
        # ---- legacy hinted jump ----
        while self.jidx < len(self.J) and self.J[self.jidx][0] < self.scroll - 60:
            self.jidx += 1
        if self.jidx < len(self.J) and self.scroll >= self.J[self.jidx][0]:
            h = self.J[self.jidx][1]
            self.jidx += 1
            self.start_jump(h, keys)
            return keys
        # ---- look ahead ----
        W = max(self.P["W_WALL"], self.P["W_PIT"] + 4, self.P["W_PIPE"])
        xa, xb = x1 + 1, min(255, x1 + W)
        if xa >= 255:
            return keys
        top = max(40, y1 - 13)
        box = s[top:y1 - 1, xa:xb + 1].astype(np.int16)
        nb = box.any(axis=2)
        for c in IGNORE:
            nb &= ~((box[:, :, 0] == c[0]) & (box[:, :, 1] == c[1]) & (box[:, :, 2] == c[2]))
        cols = nb.any(axis=0)
        jump = False
        if cols[: self.P["W_WALL"]].any():
            jump = True
        green = (box[:, :, 1] > box[:, :, 0] + 60) & (box[:, :, 1] > box[:, :, 2] + 60)
        if green.any(axis=0)[: self.P["W_PIPE"]].any():
            jump = True
        if not jump and y1 + 4 < 239:
            fl = s[y1 + 4:240, x1 + 1: min(256, x1 + 1 + self.P["W_PIT"] + 6)]
            holes = ~(fl.any(axis=2).any(axis=0))
            idx = np.nonzero(holes)[0]
            if len(idx) and idx[0] <= self.P["W_PIT"]:
                jump = True
        if self.last_shift == 0:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck > 45:
            self.stuck = 0
            self.start_jump(self.P["HOLD"], keys)
            return keys
        if jump:
            return self.trigger(keys)
        return keys

    def step(self, sim):
        s = sim.screen()
        if self.lastscr is not None and np.array_equal(s, self.lastscr):
            self.frozen += 1
        else:
            self.frozen = 0
        self.lastscr = s
        keys = self.act(s)
        sim.step(keys)
        return s


def _mem_E(sim):
    try:
        raw = sim.memory.get("E2")
        if raw:
            E = json.loads(raw) if isinstance(raw, str) else raw
            if E:
                return [list(e) for e in E]
    except Exception:
        pass
    if DEFAULT_E:
        return [list(e) for e in DEFAULT_E]
    return None


def play(sim):
    E = _mem_E(sim)
    if E:
        ag = Agent(J=OLD_J, E=E)
    else:
        ag = Agent(J=OLD_J)
    n = 0
    while sim.budget_left > 0 and n < 7950:
        ag.step(sim)
        n += 1
        if ag.frozen > 300:
            return
        if n % 200 == 0:
            print("play f", n, "scroll", ag.scroll, "mx", ag.mx, "fy", ag.fy, "ph", ag.phase)


def _save_png(name, arr):
    try:
        from PIL import Image
        Image.fromarray(arr).save(name)
    except Exception:
        pass


def solve(sim):
    t0 = time.time()
    mem = sim.memory
    sim.reset()
    E0 = None
    try:
        raw = mem.get("E3p")
        if raw:
            E0 = [list(e) for e in json.loads(raw)]
    except Exception:
        E0 = None
    stack = []
    stats = {"events": 0, "back": 0, "evalframes": 0}
    flags = {"hopeless": False}
    failinfo = {"sc": -1000, "n": 0}

    def free_slot():
        used = set(e["slot"] for e in stack)
        for k in range(10):
            if k not in used:
                return k
        return None

    def run_option(c, keys):
        ph0 = c.phase
        sim.step(keys)
        for i in range(HORIZON - 1):
            if c.dead():
                return -1e6 + i
            if ph0 == 0 and c.phase >= 1:
                return 1e4 + (HORIZON - i)
            if c.fin is not None:
                return 1e5 + (HORIZON - i)
            c.step(sim)
        if c.dead():
            return -1e6 + HORIZON
        if ph0 == 0 and c.phase >= 1:
            return 1e4
        return c.scroll + (c.mx or 0)

    def decide(ag):
        stats["events"] += 1
        left = sim.budget_left
        per_px = (left - RESERVE) / max(300.0, TOTAL_SCROLL - ag.scroll)
        n = int(per_px * 80 / HORIZON)
        n = max(1, min(len(OPTS), n))
        if left < 400:
            n = 1
        if len(stack) >= 10:
            stack.pop(0)
        slot = free_slot()
        sim.save(slot)
        ag.cb = None
        base = copy.deepcopy(ag)
        ag.cb = decide
        base.cb = decide
        res = []
        for oi in range(n):
            sim.load(slot)
            c = copy.deepcopy(base)
            c.cb = None
            keys = set(RUN)
            h, d = OPTS[oi]
            c.apply(h, d, keys)
            f0 = c.frame
            res.append(run_option(c, keys))
            stats["evalframes"] += c.frame - f0 + 1
        sim.load(slot)
        order = sorted(range(n), key=lambda i: (-(res[i] - (0.0 if i == 0 else MARGIN + 0.5 * i)), i))
        stack.append({"slot": slot, "base": base, "order": order, "res": res, "k": 0, "sc": ag.scroll})
        h, d = OPTS[order[0]]
        if res[order[0]] < -9e5:
            flags["hopeless"] = True
        print("EV sc %d fr %d left %d n %d :: %s -> %s" % (
            ag.scroll, ag.frame, left, n,
            " ".join("%s:%d" % (OPTS[i], res[i]) for i in range(n)), OPTS[order[0]]))
        return h, d

    def backtrack(T=None, lb=0):
        # rewind to the earliest event within `lb` frames before the failure at frame T
        if T is not None and lb > 0:
            while len(stack) >= 2 and stack[-2]["base"].frame >= T - lb:
                stack.pop()
        while stack:
            ent = stack[-1]
            ent["k"] += 1
            k = ent["k"]
            if k < len(ent["order"]) and ent["res"][ent["order"][k]] > -9e5:
                oi = ent["order"][k]
                sim.load(ent["slot"])
                ag2 = copy.deepcopy(ent["base"])
                ag2.cb = decide
                keys = set(RUN)
                h, d = OPTS[oi]
                ag2.commit(h, d, keys)
                sim.step(keys)
                stats["back"] += 1
                print("BACKTRACK to sc %d option %s (k=%d) frame %d" % (ent["sc"], OPTS[oi], k, ag2.frame))
                return ag2
            stack.pop()
        return None

    ag = Agent(J=OLD_J, E=E0, cb=decide)
    finished = False
    try:
        while sim.budget_left > 60:
            ag.step(sim)
            if ag.fin is not None:
                finished = True
                print("SEARCH FINISHED proxy frame", ag.fin, "E", len(ag.E), "budget_left", sim.budget_left)
                break
            if ag.dead() or flags["hopeless"]:
                hopeless = flags["hopeless"]
                flags["hopeless"] = False
                print("DEAD frame %d scroll %d mx %s fy %s frozen %d stalled %s nomario %d hopeless %s" % (
                    ag.frame, ag.scroll, ag.mx, ag.fy, ag.frozen, ag.stalled(), ag.nomario, hopeless))
                if stats["back"] > 150:
                    print("too many backtracks")
                    break
                if abs(ag.scroll - failinfo["sc"]) < 90:
                    failinfo["n"] += 1
                else:
                    failinfo["n"] = 1
                failinfo["sc"] = ag.scroll
                lb = min(700, 90 + 90 * (failinfo["n"] - 1))
                ag2 = backtrack(ag.frame, lb)
                if ag2 is None:
                    print("no more backtrack options")
                    break
                ag = ag2
            if ag.frame % 300 == 0:
                print("main f", ag.frame, "scroll", ag.scroll, "ph", ag.phase, "left", sim.budget_left,
                      "t %.0f" % (time.time() - t0))
    except Exception as e:
        print("search exception", repr(e))
    E = [list(e) for e in ag.E]
    print("search end finished=%s frames=%d scroll=%d phase=%d stats=%s wall=%.0f" % (
        finished, ag.frame, ag.scroll, ag.phase, stats, time.time() - t0))
    print("E =", json.dumps(E))
    mem["E3p"] = json.dumps(E)
    if finished:
        # verification replay with the pure policy (no callbacks)
        if sim.budget_left > 1700:
            try:
                sim.reset()
                ag2 = Agent(J=OLD_J, E=E)
                n = 0
                while sim.budget_left > 0 and n < 2600:
                    ag2.step(sim)
                    n += 1
                    if ag2.fin is not None or ag2.dead():
                        break
                print("VERIFY finished=%s fin=%s frames=%d scroll=%d phase=%d dead=%s" % (
                    ag2.fin is not None, ag2.fin, n, ag2.scroll, ag2.phase, ag2.dead()))
                _save_png("verify_end.png", sim.screen())
                if ag2.fin is not None and ag2.fin <= 1535:
                    mem["E2"] = json.dumps(E)
                    mem["E2frames"] = int(ag2.fin)
            except Exception as e:
                print("verify exception", repr(e))
        else:
            print("no budget for verification; proxy", ag.fin)
            if ag.fin is not None and ag.fin <= 1535:
                mem["E2"] = json.dumps(E)
                mem["E2frames"] = int(ag.fin)
    return []

