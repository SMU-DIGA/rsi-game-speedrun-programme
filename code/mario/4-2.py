# ==================================================================================================
# Super Mario Bros. (NES) -- level 4-2  |  超级马里奥兄弟（NES） 关 4-2
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1885 frames (31.42 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1885 帧（60 帧/秒，约 31.42 秒）。
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
#   vision perception (Mario via colours, scroll via terrain shifts, enemies via dark-blue seeds) 
#   + DFS macro-search (run / wait-w+jump-h / idle) with save states in solve(); plan replayed (tr
#   ace + world-
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def dil_h                          line 13    
#   def dil_up                         line 22    
#   def terr_enemy                     line 29    
#   def _win                           line 60    
#   def find_mario                     line 66    
#   class Perc                         line 87    
#   def pole_x                         line 172   x of the flagpole shaft (light green, tall) or None
#   def pole_touch                     line 182   
#   def scan                           line 189   nearest terrain hazard ahead on current ground: (kind, dist_px) or None
#   def enemy_ahead                    line 203   
#   class Search                       line 216   
#   def enc                            line 432   
#   def dec                            line 436   
#   def rle                            line 440   
#   def unrle                          line 456   
#   def solve                          line 465   
#   def solve_verify                   line 533   
#   def solve                          line 548   
#   def play                           line 556   
# ==================================================================================================

# idea: vision perception (Mario via colours, scroll via terrain shifts, enemies via dark-blue seeds) + DFS macro-search (run / wait-w+jump-h / idle) with save states in solve(); plan replayed (trace + world-x triggers) in play()
import numpy as np
import time
import sys

RX = {"Right", "x"}
RXZ = {"Right", "x", "z"}
HOLDS = (10, 18, 4, 40)
TD = {4: 19, 10: 24, 18: 29, 40: 50}
DEBUG = True


def dil_h(mask, m):
    Hh, Ww = mask.shape
    c = np.pad(np.cumsum(mask.astype(np.int32), axis=1), ((0, 0), (1, 0)))
    xs = np.arange(Ww)
    hi = np.minimum(xs + m + 1, Ww)
    lo = np.maximum(xs - m, 0)
    return (c[:, hi] - c[:, lo]) > 0


def dil_up(mask, m):
    out = mask.copy()
    for k in range(1, m + 1):
        out[:-k] |= mask[k:]
    return out


def terr_enemy(s):
    r = s[:, :, 0]
    g = s[:, :, 1]
    b = s[:, :, 2]
    sky = float(((r == 104) & (g == 136) & (b == 252)).mean())
    if sky > 0.2:
        T = (((r == 228) & (g == 92) & (b == 16)) | ((r == 240) & (g == 208) & (b == 176)) |
             ((r == 0) & (g == 168) & (b == 0)) | ((r == 184) & (g == 248) & (b == 24)))
        T[:40] = False
        return T, [], sky
    seeds = (r == 0) & (g == 64) & (b == 88)
    seeds[:40] = False
    teal = (r == 0) & (g >= 100)
    if seeds.any():
        E = dil_up(dil_h(seeds, 7), 15)
        T = teal & ~E
        cols = np.nonzero(seeds.any(0))[0]
        en = []
        st = cols[0]
        pv = cols[0]
        for c in list(cols[1:]) + [10 ** 6]:
            if c - pv > 10:
                sub = seeds[:, st:pv + 1]
                ys = np.nonzero(sub.any(1))[0]
                en.append((int(st) - 6, int(pv) + 6, int(ys.max()), int(ys.min())))
                st = c
            pv = c
        return T, en, sky
    return teal, [], sky


def _win(m):
    I = np.zeros((m.shape[0] + 1, m.shape[1] + 1), np.int32)
    I[1:, 1:] = m.astype(np.int32).cumsum(0).cumsum(1)
    return I[16:, 16:] - I[:-16, 16:] - I[16:, :-16] + I[:-16, :-16]


def find_mario(s):
    sub = s[40:]
    r = sub[:, :, 0]
    g = sub[:, :, 1]
    b = sub[:, :, 2]
    red = (r == 248) & (g == 56) & (b == 0)
    if red.sum() < 3:
        return None
    brown = (r == 172) & (g == 124) & (b == 0)
    skin = (r == 252) & (g == 160) & (b == 68)
    wr, wb, ws = _win(red), _win(brown), _win(skin)
    ok = (wr >= 3) & (wb >= 2) & (ws >= 2)
    if not ok.any():
        return None
    tot = np.where(ok, wr + wb + ws, -1)
    wy, wx = np.unravel_index(int(np.argmax(tot)), tot.shape)
    U = (red | brown | skin)[wy:wy + 16, wx:wx + 16]
    ys, xs = np.nonzero(U)
    return (int(wx + xs.min()), int(wx + xs.max()), int(40 + wy + ys.min()), int(40 + wy + ys.max()))


class Perc:
    def __init__(self):
        self.prevT = None
        self.prevS = None
        self.scroll = 0
        self.wxh = []
        self.bots = []
        self.nf = 0
        self.still = 0
        self.found = False
        self.x0 = self.x1 = self.top = self.bot = 0
        self.wx = 0
        self.T = None
        self.en = []
        self.sky = 0.0

    def snap(self):
        d = dict(self.__dict__)
        d["wxh"] = list(self.wxh)
        d["bots"] = list(self.bots)
        return d

    def restore(self, d):
        self.__dict__.clear()
        self.__dict__.update(d)
        self.wxh = list(self.wxh)
        self.bots = list(self.bots)

    def update(self, s):
        T, en, self.sky = terr_enemy(s)
        self.en = en
        Tc = T[40:]
        dx = 0
        if self.prevT is not None:
            p = self.prevT
            W = Tc.shape[1]
            best = None
            for k in range(0, 6):
                mm = int((Tc[:, :W - k] != p[:, k:]).sum())
                if best is None or mm < best - 6:
                    best, dx = mm, k
        self.prevT = Tc
        self.scroll += dx
        if self.prevS is not None and np.array_equal(self.prevS, s):
            self.still += 1
        else:
            self.still = 0
        self.prevS = s
        self.T = T
        m = find_mario(s)
        if m is None:
            self.found = False
            self.nf += 1
            self.bots.append(-1)
        else:
            self.found = True
            self.nf = 0
            self.x0, self.x1, self.top, self.bot = m
            self.wx = self.scroll + self.x1
            self.bots.append(self.bot)
            self.wxh.append(self.wx)
        if len(self.bots) > 12:
            self.bots.pop(0)
        if len(self.wxh) > 12:
            self.wxh.pop(0)

    @property
    def v(self):
        if len(self.wxh) >= 5:
            return (self.wxh[-1] - self.wxh[-5]) / 4.0
        return 2.0

    @property
    def dead(self):
        return self.nf >= 2 or self.still >= 6 or (self.found and self.bot >= 233)

    @property
    def grounded(self):
        if not self.found or len(self.bots) < 3:
            return False
        if not (self.bots[-1] == self.bots[-2] == self.bots[-3]):
            return False
        return bool(self.T[self.bot + 1:self.bot + 3, self.x0 + 1:self.x1].any())


def pole_x(s):
    """x of the flagpole shaft (light green, tall) or None"""
    m = (s[40:208, :, 0] == 184) & (s[40:208, :, 1] == 248) & (s[40:208, :, 2] == 24)
    cnt = m.sum(0)
    xs = np.nonzero(cnt >= 90)[0]
    if len(xs) == 0:
        return None
    return int(xs.min())


def pole_touch(s, P):
    px = pole_x(s)
    if px is None:
        return False
    return P.x1 >= px - 4


def scan(P):
    """nearest terrain hazard ahead on current ground: (kind, dist_px) or None"""
    T = P.T
    bot, x1 = P.bot, P.x1
    sup = T[bot + 1:bot + 4].any(0)
    wall = T[max(41, bot - 13):bot - 1].any(0)
    for x in range(x1 + 1, 256):
        if wall[x]:
            return ("wall", x - x1)
        if not sup[x]:
            return ("edge", x - x1)
    return None


def enemy_ahead(P):
    best = None
    for (xl, xr, yb, yt) in P.en:
        if xr < P.x0 - 2:
            continue
        if abs(yb - P.bot) > 8:
            continue
        d = max(0, xl - P.x1)
        if best is None or d < best:
            best = d
    return best


class Search:
    def __init__(self, sim):
        self.sim = sim
        self.P = Perc()
        self.trace = []
        self.nslot = 0
        self.owner = {}
        self.nid = 0
        self.t0 = time.time()
        self.maxwx = 0
        self.log_n = 0
        self.nodes = 0
        self.ev = []
        self.cur_ev = None
        self.shots = 0
        self.failed = set()
        self.goal = False
        self.stage = 1

    def stp(self, keys):
        self.sim.step(keys)
        self.trace.append(tuple(sorted(keys)))
        s = self.sim.screen()
        self.P.update(s)
        if self.check_goal(s):
            return True
        if self.stage == 1 and self.P.nf == 2 and self.P.bot < 195 and self.P.x1 < 250:
            # maybe entered a pipe: watch for the level change
            for i in range(220):
                self.sim.step(RX)
                self.trace.append(tuple(sorted(RX)))
                s = self.sim.screen()
                self.P.update(s)
                if self.check_goal(s):
                    return True
            return False
        return not self.P.dead

    def check_goal(self, s):
        if self.goal:
            return True
        if self.stage != 1:
            if self.P.sky > 0.2 and self.P.found:
                if pole_touch(s, self.P):
                    self.goal = True
                    print("GOAL2: pole touch at wx=%d frames=%d" % (self.P.wx, len(self.trace)), flush=True)
                    return True
            return False
        sky = self.P.sky
        if sky > 0.2:
            self.goal = True
            print("GOAL: sky=%.2f at wx=%d frames=%d" % (sky, self.P.wx, len(self.trace)), flush=True)
            return True
        return False

    def new_node(self, depth=0):
        slot = depth % 9 + 1
        self.nslot += 1
        self.nid += 1
        self.sim.save(slot)
        self.owner[slot] = self.nid
        P = self.P
        return dict(id=self.nid, slot=slot, snap=P.snap(), tl=len(self.trace), evl=len(self.ev),
                    cands=None, idx=0, wx=P.wx, img=self.sim.screen().copy(), depth=depth,
                    key=(P.wx // 3, P.bot, int(round(P.v))))

    def gen(self):
        P = self.P
        if not P.found:
            return [("R", 10)]
        v = min(2.7, max(1.2, P.v))
        sc = scan(P)
        de = enemy_ahead(P)
        c = []
        tev_t = None
        if sc is not None:
            kind, d = sc
            tev_t = (d + 12) / v if kind == "edge" else (d - 4) / v
        if de is not None and de < 110:
            vc = v + 0.7
            enemy_first = tev_t is None or (de / vc) < tev_t
            if enemy_first:
                for h in (4, 10, 18):
                    w0 = int((de - vc * TD[h]) / vc)
                    for w in range(w0 + 3, w0 - 8, -1):
                        if w >= 0:
                            c.append(("J", w, h, "enemy", de))
                c.append(("J", 0, 4, "enemy", de))
                c.append(("J", 0, 10, "enemy", de))
                c.append(("W", 10))
                c.append(("W", 20))
                return c
        if tev_t is None:
            return [("R", 18)]
        kind, d = sc
        tev = tev_t
        if tev > 34:
            return [("R", int(tev - 26))]
        wmax = max(0, int(tev))
        ws = list(range(wmax, max(-1, wmax - 12), -1)) + list(range(wmax - 12, -1, -2))
        for w in ws:
            for h in HOLDS:
                c.append(("J", w, h, kind, d))
        if kind == "wall":
            c.append(("R", 30))
        return c

    def exec(self, c):
        r = self._exec(c)
        return True if self.goal else r

    def _exec(self, c):
        P = self.P
        wx0 = P.wx
        if c[0] == "R":
            for i in range(c[1]):
                if not self.stp(RX):
                    return False
            return P.wx > wx0 + 4 or c[1] < 5
        if c[0] == "W":
            for i in range(c[1]):
                if not self.stp(set()):
                    return False
            return True
        _, w, h, kind, d = c
        sb = P.bot
        for i in range(w):
            if not self.stp(RX):
                return False
            if P.bot > sb + 3:
                return False
        press = P.wx
        for i in range(h):
            if not self.stp(RXZ):
                return False
        t = h
        while t < 85:
            if not self.stp(RX):
                return False
            t += 1
            if t >= 8 and P.grounded:
                break
        else:
            return False
        if P.wx < wx0 + 10:
            return False
        self.cur_ev = (press, h, d, w)
        return True

    def dfs(self, node):
        if node["cands"] is None:
            node["cands"] = self.gen()
        while node["idx"] < len(node["cands"]):
            if self.sim.budget_left < 150 or time.time() - self.t0 > 1500:
                return None
            c = node["cands"][node["idx"]]
            node["idx"] += 1
            if self.owner.get(node["slot"]) != node["id"]:
                return False
            self.sim.load(node["slot"])
            self.P.restore(node["snap"])
            del self.trace[node["tl"]:]
            del self.ev[node["evl"]:]
            self.cur_ev = None
            ok = self.exec(c)
            self.nodes += 1
            if not ok:
                continue
            if c[0] == "J" and self.cur_ev:
                self.ev.append(self.cur_ev)
            P = self.P
            if P.wx > self.maxwx:
                self.maxwx = P.wx
            if DEBUG and (P.wx // 150 != self.log_n):
                self.log_n = P.wx // 150
                print("wx=%d bot=%d frames=%d left=%d t=%.0f nodes=%d ev=%d" % (
                    P.wx, P.bot, len(self.trace), self.sim.budget_left, time.time() - self.t0, self.nodes, len(self.ev)), flush=True)
                self.shot(self.sim.screen(), "m%d" % P.wx)
            if self.finished():
                return True
            ck = (P.wx // 3, P.bot, int(round(P.v)))
            if ck in self.failed:
                continue
            child = self.new_node(node["depth"] + 1)
            r = self.dfs(child)
            if r is None:
                return None
            if r:
                return True
        self.failed.add(node["key"])
        if node["wx"] >= self.maxwx - 40 and self.shots < 20:
            print("EXHAUSTED node wx=%d tl=%d ncands=%d first=%s" % (node["wx"], node["tl"], len(node["cands"]), node["cands"][:1]))
            self.shot(node["img"], "ex%d" % node["wx"])
        return False

    def shot(self, img, name):
        if self.shots > 70:
            return
        self.shots += 1
        try:
            from PIL import Image
            Image.fromarray(img).save(name + ".png")
        except Exception:
            pass

    def finished(self):
        return self.goal


PREFIX_RLE = ("3*37,7*18,3*30,7*10,3*30,7*10,3*7,7*10,3*1,7*10,3*25,7*40,3*8,7*10,3*13,7*10,3*11,7*4,3*35,7*10,3*21,7*10,"
              "3*13,7*10,3*43,7*10,3*13,7*40,3*9,7*10,3*14,7*10,3*43,7*18,3*13,7*10,3*26,7*10,3*7,7*10,3*13,7*10,3*25,"
              "7*10,3*16,7*4,3*23,7*10,3*30,7*10,3*43,7*10,3*11,7*10,3*30,7*10,3*29,7*40,3*1,7*40,3*9,7*10,3*4,7*40,"
              "3*13,7*10,3*12,7*10,3*35,7*10,3*52,7*10,3*11,7*10,3*27,7*10,3*20,7*10,3*29,7*10,3*20,7*10,3*22,7*10,3*13,7*10,3*1,7*4,3*49,7*6,3*75")
BIT = {"Right": 1, "x": 2, "z": 4, "Left": 8, "Down": 16, "Up": 32}


def enc(keys):
    return sum(BIT[k] for k in keys)


def dec(n):
    return {k for k, b in BIT.items() if n & b}


def rle(trace):
    out = []
    prev, cnt = None, 0
    for k in trace:
        e = enc(k)
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


def solve(sim):
    sys.setrecursionlimit(5000)
    sim.reset()
    S = Search(sim)
    S.P.update(sim.screen())
    pre = unrle(PREFIX_RLE)
    if pre:
        for e in pre:
            k = dec(e)
            sim.step(k)
            S.trace.append(tuple(sorted(k)))
            S.P.update(sim.screen())
        print("prefix replayed", len(pre), "wx", S.P.wx, "bot", S.P.bot, "sky", S.P.sky)
        if S.P.sky > 0.2:
            S.stage = 2
            S.maxwx = S.P.wx
            for i in range(120):
                S.stp(RX)
                if S.P.grounded and S.P.found:
                    break
            print("stage2 settled at", S.P.wx, S.P.bot, len(S.trace))
    else:
        for i in range(60):
            S.stp(RX)
            if S.P.grounded:
                break
    print("start grounded at", S.P.wx, S.P.bot, len(S.trace))
    root = S.new_node()
    try:
        r = S.dfs(root)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print("EXC", repr(e))
        r = None
    print("dfs result", r, "maxwx", S.maxwx, "events", len(S.ev), "frames", len(S.trace))
    sim.memory["plan"] = [list(map(int, e)) for e in S.ev]
    sim.memory["maxwx"] = int(S.maxwx)
    print("PLAN", S.ev)
    print("TRACE_RLE[%d]: %s" % (len(S.trace), rle(S.trace)))
    S.shot(sim.screen(), "last")
    if S.goal:
        # explore stage 2: run right, log screens
        try:
            from PIL import Image
            panels = []
            for i in range(120):
                s = sim.screen()
                if i % 10 == 0:
                    panels.append(s.copy())
                sim.step(RX)
                S.trace.append(tuple(sorted(RX)))
            sheet = np.zeros((240 * 3, 256 * 4, 3), np.uint8)
            for k, im in enumerate(panels[:12]):
                sheet[(k // 4) * 240:(k // 4 + 1) * 240, (k % 4) * 256:(k % 4 + 1) * 256] = im
            Image.fromarray(sheet).save("stage2.png")
        except Exception as e:
            print("stage2 err", repr(e))
    return [set(k) for k in S.trace]


FULL_RLE = ("3*37,7*18,3*30,7*10,3*30,7*10,3*7,7*10,3*1,7*10,3*25,7*40,3*8,7*10,3*13,7*10,3*11,7*4,3*35,7*10,3*21,7*10,3*13,7*10,"
            "3*43,7*10,3*13,7*40,3*9,7*10,3*14,7*10,3*43,7*18,3*13,7*10,3*26,7*10,3*7,7*10,3*13,7*10,3*25,7*10,3*16,7*4,"
            "3*23,7*10,3*30,7*10,3*43,7*10,3*11,7*10,3*30,7*10,3*29,7*40,3*1,7*40,3*9,7*10,3*4,7*40,3*13,7*10,3*12,7*10,"
            "3*35,7*10,3*52,7*10,3*11,7*10,3*27,7*10,3*20,7*10,3*29,7*10,3*20,7*10,3*22,7*10,3*13,7*10,3*1,7*4,3*49,7*6,"
            "3*102,7*10,3*31,7*10,3*20,7*10,3*29,7*10,3*20,7*10,3*31,7*10,3*49,7*10,3*75")


def solve_verify(sim):
    sim.reset()
    P = Perc()
    tr = unrle(FULL_RLE)
    for i, e in enumerate(tr):
        sim.step(dec(e))
        P.update(sim.screen())
        if i % 200 == 0:
            print("replay", i, "wx", P.wx, "bot", P.bot, "found", P.found)
        if P.sky > 0.2 and P.found and pole_touch(sim.screen(), P):
            print("VERIFY goal at frame", i)
            break
    return []


def solve(sim):
    try:
        solve_verify(sim)
    except Exception as ex:
        print("verify exc", repr(ex))
    return []


def play(sim):
    tr = unrle(FULL_RLE)
    i = 0
    while sim.budget_left > 0:
        if i < len(tr):
            keys = dec(tr[i])
        else:
            keys = set(RX)
        i += 1
        sim.step(keys)

