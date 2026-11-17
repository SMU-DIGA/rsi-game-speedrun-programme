# ==================================================================================================
# Super Mario Bros. (NES) -- level 1-3  |  超级马里奥兄弟（NES） 关 1-3
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1017 frames (16.95 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1017 帧（60 帧/秒，约 16.95 秒）。
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
#   Mario physics is deterministic -> macro-level DFS with save/load rollback finds a frame-exact 
#   input trace; it is verified on fresh variants and repaired from checkpoints; play() replays th
#   e trace
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def to_mask                        line 16    
#   def from_mask                      line 23    
#   def bgmask                         line 27    
#   def find_mario                     line 37    
#   def gray                           line 56    
#   def detect_shift                   line 60    
#   class Agent                        line 82    
#   class Node                         line 237   ---------------------------------------------------------------- search
#   class Searcher                     line 241   
#   def save_png                       line 408   
#   def encode                         line 416   
#   def decode                         line 426   
#   def replay_verify                  line 433   replay trace from fresh reset; returns (ok, failframe, checkpoints)
#   def solve                          line 460   
#   def play                           line 531   ---------------------------------------------------------------- play
#   def _play                          line 538   
#
# This policy also reads sim.memory; that data is in 1-3.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 1-3.memory.json。
# ==================================================================================================

# idea: Mario physics is deterministic -> macro-level DFS with save/load rollback finds a frame-exact input trace; it is verified on fresh variants and repaired from checkpoints; play() replays the trace
import copy
import time
import numpy as np

T0 = time.time()
TIME_LIMIT = 1500.0
RUN = ("Right", "x")
KM = {"Right": 1, "x": 2, "z": 4, "Left": 8, "Down": 16, "Up": 32}
KN = {v: k for k, v in KM.items()}
FINISH_WX = 3300
POLE_MIN_WX = 1500
JHOLDS =[24, 12, 6, 18, 3, 9, 15, 21]


def to_mask(keys):
    m = 0
    for k in keys:
        m |= KM.get(k, 0)
    return m


def from_mask(m):
    return {KN[b] for b in KN if m & b}


def bgmask(a):
    r = a[..., 0]
    g = a[..., 1]
    b = a[..., 2]
    sky = (r == 104) & (g == 136) & (b == 252)
    cb = (r == 60) & (g == 188) & (b == 252)
    wh = (r == 252) & (g == 252) & (b == 252)
    return sky | cb | wh


def find_mario(s, prev):
    m = (s[:, :, 0] == 172) & (s[:, :, 1] == 124) & (s[:, :, 2] == 0)
    if prev is not None:
        m[:8] = False
        m[12:36, 80:104] = False   # HUD coin icon
        x0, x1, y0, y1 = prev
        w = np.zeros_like(m)
        w[max(8, y0 - 40):min(240, y1 + 41), max(0, x0 - 40):min(256, x1 + 41)] = True
        m = m & w
        need = 4
    else:
        m[:40] = False
        need = 8
    ys, xs = np.nonzero(m)
    if len(ys) < need:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def gray(s):
    return (s[40:, :, 0].astype(np.int32) * 3 + s[40:, :, 1].astype(np.int32) * 5 + s[40:, :, 2])


def detect_shift(pg, cg, lo, hi, last):
    best = None
    W = 256
    lo = max(0, lo)
    hi = min(W, hi)
    for sh in range(0, 6):
        a = pg[::2, sh:]
        b = cg[::2, :W - sh]
        ne = a != b
        l2 = min(lo, W - sh)
        h2 = min(hi, W - sh)
        ne[:, l2:h2] = False
        ncol = (W - sh) - (h2 - l2)
        if ncol <= 20:
            continue
        sc = ne.sum() / float(ne.shape[0] * ncol)
        key = (round(sc, 3), abs(sh - last), sh)
        if best is None or key < best[0]:
            best = (key, sh)
    return best[1] if best else 0


class Agent:
    def __init__(self):
        self.frame = 0
        self.box = None
        self.pg = None
        self.lastscr = None
        self.frozen = 0
        self.scroll = 0
        self.last_shift = 0
        self.xc = 0
        self.x1 = 0
        self.wx = 0
        self.y1 = 0
        self.prev_y1 = None
        self.nomario = 0
        self.grounded = False
        self.best_wx = 0
        self.best_wx_frame = 0
        self.bg = None
        self.mario_ok = False
        self.scr = None
        self.nstable = 0

    def observe(self, s):
        self.frame += 1
        if self.lastscr is not None and np.array_equal(s, self.lastscr):
            self.frozen += 1
        else:
            self.frozen = 0
        self.lastscr = s
        self.scr = s
        box = find_mario(s, self.box)
        if box is None and self.box is not None:
            box = find_mario(s, None)
        cg = gray(s)
        if box is None:
            self.nomario += 1
            self.pg = cg
            self.mario_ok = False
            self.grounded = False
            return
        self.nomario = 0
        self.mario_ok = True
        x0, x1, y0, y1 = box
        if self.pg is not None and self.box is not None:
            lo = min(x0, self.box[0] - 6) - 4
            hi = max(x1, self.box[1]) + 5
            sh = detect_shift(self.pg, cg, lo, hi, self.last_shift)
        else:
            sh = 0
        self.last_shift = sh
        self.scroll += sh
        self.pg = cg
        self.prev_y1 = self.y1 if self.box is not None else None
        self.box = box
        self.x1 = x1
        self.xc = (x0 + x1) // 2
        self.y1 = y1
        self.wx = self.scroll + self.xc
        if self.wx > self.best_wx:
            self.best_wx = self.wx
            self.best_wx_frame = self.frame
        bg = bgmask(s)
        self.bg = bg
        ya = min(239, y1 + 1)
        yb = min(240, y1 + 3)
        under = ~bg[ya:yb, max(0, self.xc - 3):min(256, self.xc + 4)]
        below = bool(under.any())
        if self.prev_y1 is not None and self.prev_y1 == y1:
            self.nstable += 1
        else:
            self.nstable = 0
        self.grounded = below and self.nstable >= 2

    def failed(self):
        if self.frozen >= 3:
            return "frozen"
        if self.nomario >= 3:
            return "nomario"
        if self.mario_ok and self.y1 >= 228:
            return "fell"
        if self.mario_ok and self.y1 >= 214 and not self.grounded:
            return "fell"
        if self.frame - self.best_wx_frame > 40:
            return "noprog"
        return None

    def pole_touch(self):
        """thin tall vertical bar (flagpole) overlapping Mario's box"""
        s = self.scr
        g = s[40:].astype(np.int16)
        same = (np.abs(g[1:] - g[:-1]).sum(axis=2) == 0) & (~self.bg[41:])
        blk = (g.sum(axis=2) == 0)
        org = (g[:, :, 0] == 228) & (g[:, :, 1] == 92) & (g[:, :, 2] == 16)
        same &= ~(blk | org)[1:]
        run = np.zeros(256, dtype=np.int32)
        best = np.zeros(256, dtype=np.int32)
        for r in range(same.shape[0]):
            run = np.where(same[r], run + 1, 0)
            best = np.maximum(best, run)
        cols = np.nonzero(best >= 70)[0]
        if len(cols) == 0:
            return False
        groups = []
        st = cols[0]
        pv = cols[0]
        for c in cols[1:]:
            if c != pv + 1:
                groups.append((st, pv))
                st = c
            pv = c
        groups.append((st, pv))
        x0, x1 = self.box[0], self.box[1]
        for a, b in groups:
            if b - a + 1 <= 6 and a - 2 <= x1 and b + 2 >= x0:
                return True
        return False

    def finished(self):
        if not self.mario_ok:
            return False
        if self.frozen >= 3 and self.y1 < 226 and self.best_wx >= FINISH_WX:
            return True
        if self.best_wx >= POLE_MIN_WX and self.frozen == 0:
            try:
                return self.pole_touch()
            except Exception:
                return False
        return False

    def probe(self):
        """(dE, dW): distance to floor edge ahead / obstacle at body height ahead (None if none)"""
        bg = self.bg
        y1 = self.y1
        x1 = self.x1
        xc = self.xc
        ya = min(239, y1 + 1)
        yb = min(240, y1 + 3)
        fl = (~bg[ya:yb, :]).any(axis=0)
        dE = None
        for c in range(xc + 1, 256):
            if not fl[c]:
                dE = c - x1
                break
        top = max(40, y1 - 12)
        body = ~bg[top:y1 - 1, x1 + 3:256]
        dW = None
        if body.size:
            cols = np.nonzero(body.any(axis=0))[0]
            if len(cols):
                dW = int(cols[0]) + 3
        return dE, dW


# ---------------------------------------------------------------- search
class Node:
    pass


class Searcher:
    def __init__(self, sim, trace, ag, origin_slot, origin_tlen, floor, tag="", maxfail=None):
        self.sim = sim
        self.trace = trace
        self.ag = ag
        self.origin_slot = origin_slot
        self.origin_tlen = origin_tlen
        self.floor = floor
        self.tag = tag
        self.stack = []
        self.free = list(range(8))
        self.nfail = 0
        self.nlog = 0
        self.maxfail = maxfail
        self.fails = {}
        self.dead = set()
        self.best_wx = -1
        self.best_trace = []

    def log(self, *a):
        if self.nlog < 700:
            self.nlog += 1
            print("[%s]" % self.tag, *a)

    def cands(self):
        ag = self.ag
        dE, dW = ag.probe()
        jumps = [("J", h) for h in JHOLDS] + [("B", 6), ("B", 14)]
        hz = None
        if dE is not None:
            hz = dE
        if dW is not None and (hz is None or dW < hz):
            hz = dW
        near = (dE is not None and dE <= 12) or (dW is not None and dW <= 28)
        if ag.frame <= 3:
            near = False
            hz = None
        if hz is None:
            n = 14
        else:
            n = int(max(2, min(14, (hz - 14) / 3.0)))
        run = ("R", n)
        if near:
            return jumps + [("W", 8), ("R", 2)], (dE, dW)
        return [run] + jumps, (dE, dW)

    def restore(self, node):
        sim = self.sim
        if node.slot is not None:
            sim.load(node.slot)
        else:
            sim.load(self.origin_slot)
            for i in range(self.origin_tlen, node.tlen):
                sim.step(from_mask(self.trace[i]))
            if self.free:
                node.slot = self.free.pop()
                sim.save(node.slot)
        del self.trace[node.tlen:]
        self.ag = copy.deepcopy(node.ag)

    def macro(self, cand):
        sim = self.sim
        kind, p = cand
        t = 0
        while True:
            keys = set(RUN)
            if kind == "J" and t < p:
                keys.add("z")
            if kind in ("B", "W") and t < p:
                keys = set()
            sim.step(keys)
            self.trace.append(to_mask(keys))
            s = sim.screen()
            ag = self.ag
            ag.observe(s)
            t += 1
            if ag.finished():
                return "done"
            r = ag.failed()
            if r:
                return r
            if kind in ("R", "B", "W"):
                if t >= p and ag.grounded:
                    return "ok"
            else:
                if t >= max(p, 4) and ag.grounded:
                    return "ok"
            if t > 150:
                return "long"
            if sim.budget_left <= self.floor:
                raise RuntimeError("budget")
            if time.time() - T0 > TIME_LIMIT:
                raise RuntimeError("time")

    def new_node(self):
        sim = self.sim
        key = (self.ag.wx, self.ag.y1)
        if key in self.dead:
            return None
        if not self.free:
            old = self.stack[0]
            # drop oldest slotted node
            for nd in self.stack:
                if nd.slot is not None:
                    old = nd
                    break
            self.free.append(old.slot)
            old.slot = None
        n = Node()
        n.slot = self.free.pop()
        sim.save(n.slot)
        n.tlen = len(self.trace)
        n.ag = copy.deepcopy(self.ag)
        n.cands, info = self.cands()
        n.i = 0
        n.info = info
        n.fresh = True
        n.key = key
        if self.ag.wx > self.best_wx:
            self.best_wx = self.ag.wx
            self.best_trace = list(self.trace)
        self.stack.append(n)
        self.log("node wx=%d f=%d y1=%d info=%s first=%s depth=%d bud=%d" % (
            self.ag.wx, self.ag.frame, self.ag.y1, info, n.cands[0], len(self.stack), sim.budget_left))
        return n

    def run(self):
        """returns True when finished (trace holds the full trace)"""
        sim = self.sim
        self.new_node()
        while True:
            top = self.stack[-1]
            if top.i >= len(top.cands):
                self.stack.pop()
                self.dead.add(top.key)
                if top.slot is not None:
                    self.free.append(top.slot)
                if not self.stack:
                    self.log("STACK EXHAUSTED")
                    return False
                self.restore(self.stack[-1])
                self.stack[-1].fresh = True
                continue
            cand = top.cands[top.i]
            top.i += 1
            if not top.fresh:
                self.restore(top)
            top.fresh = False
            r = self.macro(cand)
            if r == "done":
                self.log("FINISHED wx=%d frame=%d" % (self.ag.wx, self.ag.frame))
                return True
            if r == "ok":
                self.new_node()
                continue
            self.nfail += 1
            kk = top.ag.wx // 40
            if kk not in self.fails and len(self.fails) < 40:
                self.fails[kk] = 1
                save_png("node_%s_%d.png" % (self.tag, top.ag.wx), top.ag.scr)
                save_png("fail_%s_%d.png" % (self.tag, top.ag.wx), self.ag.scr)
            if top.i <= 1 or top.i == len(top.cands):
                self.log("fail %s cand=%s node_wx=%d at wx=%d y1=%d frame=%d" % (r, cand, top.ag.wx, self.ag.wx, self.ag.y1, self.ag.frame))
            if self.maxfail and self.nfail > self.maxfail:
                return False


def save_png(name, arr):
    try:
        from PIL import Image
        Image.fromarray(arr).save(name)
    except Exception:
        pass


def encode(trace):
    out = []
    for m in trace:
        if out and out[-1][0] == m:
            out[-1][1] += 1
        else:
            out.append([m, 1])
    return out


def decode(rle):
    t = []
    for m, c in rle:
        t.extend([m] * c)
    return t


def replay_verify(sim, trace, tag, ring=True):
    """replay trace from fresh reset; returns (ok, failframe, checkpoints)"""
    sim.reset()
    ag = Agent()
    ag.observe(sim.screen())
    cps = []
    slot = 0
    for i, m in enumerate(trace):
        if ring and i % 25 == 0 and ag.grounded:
            sim.save(slot)
            cps = [c for c in cps if c[0] != slot]
            cps.append((slot, i, copy.deepcopy(ag)))
            slot = (slot + 1) % 8
        sim.step(from_mask(m))
        ag.observe(sim.screen())
        if ag.finished():
            print("[%s] verify FINISHED at trace idx %d wx=%d" % (tag, i, ag.wx))
            return True, i, cps
        r = ag.failed()
        if r:
            print("[%s] verify FAIL %s at idx %d wx=%d y1=%d" % (tag, r, i, ag.wx, ag.y1))
            save_png("vfail_%s.png" % tag, sim.screen())
            return False, i, cps
    print("[%s] verify: trace ended without finish wx=%d" % (tag, ag.wx))
    return False, len(trace), cps


def solve(sim):
    global T0
    T0 = time.time()
    mem = sim.memory
    sim.reset()
    sim.save(9)
    ag = Agent()
    ag.observe(sim.screen())
    trace = []
    S = Searcher(sim, trace, ag, 9, 0, 400, tag="s0")
    try:
        ok = S.run()
    except Exception as e:
        print("search ended:", repr(e))
        ok = False
    print("search done ok=%s frames=%d budget_left=%d t=%.0f" % (ok, len(trace), sim.budget_left, time.time() - T0))
    if not ok:
        bt = S.best_trace if len(S.best_trace) > 0 else list(trace)
        mem["partial"] = encode(bt)
        print("partial best wx", S.best_wx, "len", len(bt))
        return []
    best = list(trace)
    mem["trace"] = encode(best)
    try:
        sim.memory.flush()
    except Exception:
        pass
    # verification / repair rounds on fresh variants
    passes = 0
    attempts = 0
    nrep = 0
    while passes < 3 and sim.budget_left > len(best) + 600 and time.time() - T0 < TIME_LIMIT:
        ok2, fi, cps = replay_verify(sim, best, "v%d" % attempts)
        attempts += 1
        if ok2:
            passes += 1
            continue
        passes = 0
        if not cps:
            break
        # repair from a checkpoint before the failure; go further back after each failed repair
        back = [40, 40, 90, 90, 160][min(nrep, 4)]
        nrep += 1
        cand = [c for c in cps if c[1] <= fi - back]
        cp = max(cand, key=lambda c: c[1]) if cand else min(cps, key=lambda c: c[1])
        sim.load(cp[0])
        sim.save(8)
        ag2 = copy.deepcopy(cp[2])
        ag2.frozen = 0
        pre = list(best[:cp[1]])
        S2 = Searcher(sim, pre, ag2, 8, cp[1], 300, tag="rep%d" % attempts, maxfail=120)
        try:
            r2 = S2.run()
        except Exception as e:
            print("repair ended:", repr(e))
            r2 = False
        if r2:
            best = list(S2.trace)
            mem["trace"] = encode(best)
            print("repaired; new length", len(best))
        try:
            sim.memory.flush()
        except Exception:
            pass
    print("verification: passes=%d attempts=%d budget_left=%d t=%.0f" % (passes, attempts, sim.budget_left, time.time() - T0))
    mem["verified"] = passes
    mem["trace"] = encode(best)
    return []


# ---------------------------------------------------------------- play
def play(sim):
    try:
        _play(sim)
    except Exception as e:
        print("play ended:", repr(e))


def _play(sim):
    rle = sim.memory.get("trace") or sim.memory.get("partial") or []
    trace = decode(rle)
    ag = Agent()
    n = 0
    for m in trace:
        if sim.budget_left <= 0 or n >= 7990:
            return
        ag.observe(sim.screen())
        if ag.finished():
            return
        sim.step(from_mask(m))
        n += 1
    # trace over: keep running right with simple jumps until frozen
    while n < 7990 and sim.budget_left > 0:
        s = sim.screen()
        ag.observe(s)
        if ag.frozen >= 5:
            return
        k = set(RUN)
        if ag.mario_ok and ag.grounded:
            dE, dW = ag.probe()
            if (dE is not None and dE <= 10) or (dW is not None and dW <= 24):
                k.add("z")
        elif ag.mario_ok and ag.frame % 2 == 0:
            k.add("z")
        sim.step(k)
        n += 1

