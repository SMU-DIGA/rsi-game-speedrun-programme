# ==================================================================================================
# Super Mario Bros. (NES) -- level 3-1  |  超级马里奥兄弟（NES） 关 3-1
# Pass rate on 100 random variants never used before: 91/100; fastest finish 2085 frames (34.75 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 91/100；最快一局 2085 帧（60 帧/秒，约 34.75 秒）。
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
#   reactive SMB 3-1 pilot: run right; frame-diff movers tracked (world velocity, parabolic hop mo
#   del) and a short-horizon planner picks RUN / WAIT / JUMP(hold); black ground columns = pits (r
#   unning jump)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def log                            line 20    
#   def klog                           line 25    
#   def eq                             line 32    
#   def dil_x                          line 36    
#   class Track                        line 64    
#   class Pilot                        line 85    
#   def safe_act                       line 511   
#   def enc                            line 520   
#   def dec                            line 524   
#   def _shot                          line 543   
#   def _wx                            line 551   
#   def _search                        line 555   
#   def _verify                        line 632   replay the trace on a fresh variant; return first divergence frame or -1
#   def solve                          line 649   
#   def play                           line 671   
#   def _sig                           line 722   
#   def _crc                           line 726   
#   def _sigdist                       line 730   distance between two enemy signatures (lists of (x0,x1,top,bot))
#   def _mismatch                      line 742   
#   def _search_from                   line 747   
#   def _close                         line 826   
#   class Follower                     line 830   follows the library trace whose recorded Mario positions match the observed ones; falls back to the pilot
#   def _try_variant                   line 884   play the library on a fresh variant; if it dies, branch a new trace off the active one
#   def _dump                          line 968   
#   def _replay_seed                   line 981   replay a known finishing trace on the first variant, recording positions / enemy signatures
#   def solve                          line 1016  
#   def play                           line 1067  
#
# This policy also reads sim.memory; that data is in 3-1.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 3-1.memory.json。
# ==================================================================================================

# idea: reactive SMB 3-1 pilot: run right; frame-diff movers tracked (world velocity, parabolic hop model) and a short-horizon planner picks RUN / WAIT / JUMP(hold); black ground columns = pits (running jump); gray pipes = stop adjacent, wait for plant to hide, jump on top; blocked -> jump
import copy
import random
import time
import numpy as np

H, W = 240, 256
RED = (248, 56, 0)
GRAY = (188, 188, 188)
WHITE = (252, 252, 252)
GREEN = (0, 168, 0)

DEBUG = []
DBG = [0, -1]


KEY = []


def log(*a):
    if len(DEBUG) < 600:
        DEBUG.append(" ".join(str(x) for x in a))


def klog(*a):
    line = " ".join(str(x) for x in a)
    if len(KEY) < 200:
        KEY.append(line)
    log(line)


def eq(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def dil_x(mask, r):
    out = mask.copy()
    for d in range(1, r + 1):
        out[:, d:] |= mask[:, :-d]
        out[:, :-d] |= mask[:, d:]
    m2 = out.copy()
    for d in range(1, r + 1):
        m2[d:, :] |= out[:-d, :]
        m2[:-d, :] |= out[d:, :]
    return m2


# measured jump arcs (running at 3px/frame): mario top-row after k frames
_A4 = [194, 194, 188, 182, 176, 171, 166, 161, 158, 155, 153, 152, 152, 153, 154, 156, 159, 163, 168, 173, 179, 186, 191, 194]
_A8 = [194, 194, 188, 182, 176, 171, 166, 161, 156, 152, 147, 144, 141, 140, 139, 138, 139, 140, 143, 146, 150, 154, 159, 164, 169, 174, 179, 184, 189, 194]
_A12 = [194, 194, 188, 182, 176, 171, 166, 161, 156, 152, 147, 143, 140, 136, 133, 130, 128, 127, 127, 128, 129, 132, 135, 139, 144, 149, 154, 159, 164, 169, 174, 179, 184, 189, 194]
_A16 = [194, 194, 188, 182, 176, 171, 166, 161, 156, 152, 147, 143, 140, 136, 133, 129, 126, 124, 121, 120, 119, 119, 119, 121, 123, 127, 131, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185, 190, 194]
_A26 = [194, 194, 188, 182, 176, 171, 166, 161, 156, 152, 147, 143, 140, 136, 133, 129, 126, 124, 121, 119, 117, 115, 113, 112, 111, 110, 109, 109, 108, 108, 108, 109, 111, 114, 117, 121]
_v = _A26[-1]
while _v < 193:
    _v += 5
    _A26.append(min(_v, 194))
ARCS = {h: [v - 193 for v in a] for h, a in ((4, _A4), (8, _A8), (12, _A12), (16, _A16), (26, _A26))}

GRAV = 0.0988
BOUNCE = -2.67


class Track:
    __slots__ = ("x0", "x1", "top", "bot", "wc", "vx", "vy", "age", "still", "h", "bots", "hopper", "hb", "hx", "moving")

    def future_bots(self, T):
        b = float(self.bot)
        ground = not self.hopper
        out = []
        v = self.vy
        for t in range(1, T + 1):
            if ground:
                out.append(207.0)
                continue
            v += GRAV
            b += v
            if b >= 207:
                b = 207.0
                v = BOUNCE
            out.append(b)
        return out


class Pilot:
    def __init__(self):
        self.n = 0
        self.prev = None
        self.hist = []
        self.cum = 0
        self.ry_hist = []
        self.jump_t = None
        self.hold = 0
        self.last_sc = 0
        self.vx = 0.0
        self.blocked = 0
        self.hidden = 0
        self.pwait = 0
        self.info = ""
        self.pm = None
        self.tracks = []
        self.cl = []
        self.waitn = 0
        self.noise = 0.0
        self.rng = random.Random(1)
        self.pause = 0
        self.esig = ()
        self.ewait = 0

    # ------------------------------------------------------------ perception
    def find_mario(self, s):
        m = eq(s, RED)
        m[:40] = False
        ys, xs = np.nonzero(m)
        if len(xs) == 0:
            return None
        if self.pm is not None:
            cx = (self.pm[0] + self.pm[1]) / 2
            k = (np.abs(xs - cx) < 40) & (np.abs(ys - self.pm[2]) < 50)
            if k.sum() >= 4:
                ys, xs = ys[k], xs[k]
        ry0, ry1 = int(ys.min()), int(ys.max())
        big = (ry1 - ry0) >= 20
        ml, mr = int(xs.min()) - 3, int(xs.max()) + 3
        feet = ry0 + (30 if big else 14)
        return ml, mr, ry0, feet, big

    def scroll(self, s):
        if self.prev is None:
            return 0
        a = s[214:238, 0:W - 8].astype(np.int16)
        errs = []
        for k in range(0, 6):
            b = self.prev[214:238, k:k + W - 8].astype(np.int16)
            errs.append(int((a != b).any(axis=2).sum()))
        errs = np.array(errs)
        if errs.max() - errs.min() < 6:
            return self.last_sc
        return int(np.argmin(errs))

    def movers(self, s, mario):
        mv = np.zeros((H, W), bool)
        if not self.hist:
            return mv
        old, ocum, obox = self.hist[0]
        S = self.cum - ocum
        if S < 0 or S > 14:
            return mv
        d = (s[:, :W - S] != old[:, S:]).any(axis=2)
        mv[:, :W - S] = d
        mv[:60] = False
        mv[:, W - 14:] = False
        ml, mr, ry0, feet, big = mario
        mv[max(0, ry0 - 3):feet + 3, max(0, ml - 3):mr + 3] = False
        if obox is not None:
            oml, omr, ory0, ofeet = obox
            oml -= S
            omr -= S
            mv[max(0, min(ry0, ory0) - 3):max(feet, ofeet) + 3, max(0, oml - 3):max(0, omr + 3)] = False
        return mv

    def clusters(self, E):
        cols = np.nonzero(E.any(axis=0))[0]
        out = []
        if len(cols) == 0:
            return out
        groups = []
        st = last = int(cols[0])
        for c in cols[1:]:
            c = int(c)
            if c - last > 7:
                groups.append((st, last))
                st = c
            last = c
        groups.append((st, last))
        for (a, b) in groups:
            sub = E[:, a:b + 1]
            if sub.sum() < 5:
                continue
            ys = np.nonzero(sub.any(axis=1))[0]
            out.append({"x0": a, "x1": b, "top": int(ys.min()), "bot": int(ys.max())})
        return out

    def find_pipes(self, s):
        P = eq(s, GRAY) | eq(s, WHITE)
        band = P[192:208].sum(axis=0) >= 5
        pipes = []
        c = 0
        while c < W:
            if band[c]:
                c2 = c
                gap = 0
                last = c
                while c2 < W and (band[c2] or gap < 2):
                    if band[c2]:
                        last = c2
                        gap = 0
                    else:
                        gap += 1
                    c2 += 1
                if last - c + 1 >= 14:
                    cols = slice(c, last + 1)
                    wdt = last - c + 1
                    top = 207
                    miss = 0
                    for y in range(207, 100, -1):
                        if P[y, cols].sum() >= 0.2 * wdt:
                            top = y
                            miss = 0
                        else:
                            miss += 1
                            if miss >= 2:
                                break
                    pipes.append((c, last, top))
                c = last + 1
            else:
                c += 1
        return pipes

    def update_tracks(self, cl, pipes):
        new = []
        for c in cl:
            # drop plants (inside pipe columns, above the pipe top)
            skip = False
            for (p0, p1, pt) in pipes:
                if c["x0"] >= p0 - 4 and c["x1"] <= p1 + 4 and c["bot"] <= pt + 8:
                    skip = True
            if skip:
                continue
            wc = (c["x0"] + c["x1"]) / 2.0 + self.cum
            best = None
            for t in self.tracks:
                d = abs(wc - t.wc)
                if d < 12 and abs(c["bot"] - t.bot) < 18 and (best is None or d < best[0]):
                    best = (d, t)
            tr = Track()
            tr.x0, tr.x1, tr.top, tr.bot, tr.wc = c["x0"], c["x1"], c["top"], c["bot"], wc
            if best is not None:
                o = best[1]
                tr.age = o.age + 1
                tr.vx = 0.9 * o.vx + 0.1 * (wc - o.wc)
                tr.hb = (o.hb + [c["bot"]])[-40:]
                tr.hx = (o.hx + [c["x0"] + self.cum])[-12:]
            else:
                tr.age = 0
                tr.vx = -0.55
                tr.hb = [c["bot"]]
                tr.hx = [c["x0"] + self.cum]
            tr.moving = len(tr.hx) >= 6 and (max(tr.hx) - min(tr.hx)) >= 3
            tr.vx = max(-1.0, min(0.8, tr.vx))
            bs = tr.hb
            rise = any(b - a <= -1 for a, b in zip(bs[:-1], bs[1:]))
            fall = any(b - a >= 1 for a, b in zip(bs[:-1], bs[1:]))
            tr.hopper = (max(bs) - min(bs)) >= 8 and min(bs) < 198 and ((rise and fall) or (c["x1"] - c["x0"]) >= 19)
            tr.vy = 0.0
            if len(bs) >= 5:
                d = bs[-1] - bs[-5]
                sgn = 1 if d > 1 else (-1 if d < -1 else 0)
                if tr.hopper:
                    vm = (0.1976 * max(0.0, tr.bot - 171)) ** 0.5
                    if sgn == 0 and tr.bot >= 202:
                        sgn = -1
                    tr.vy = sgn * vm
                tr.still = (max(bs[-5:]) - min(bs[-5:])) < 2
            else:
                tr.still = False
            new.append(tr)
        self.tracks = new
        out = []
        for t in new:
            if t.age < 2:
                continue
            if t.bot <= 185 and (t.bot - t.top) <= 19 and not t.hopper and not t.moving:
                continue            # flashing ? block
            out.append(t)
        return out

    # ------------------------------------------------------------ planner
    def simulate(self, mode, arg, d, cx, feet, v0, hm, trs, T=46):
        arc = ARCS[arg] if mode == "J" else None
        x = float(cx)
        v = v0
        prevdy = 0
        for t in range(1, T + 1):
            k = 0
            dy = 0
            if mode == "J" and t > d:
                k = t - d
                if k - 1 < len(arc):
                    dy = arc[k - 1]
                else:
                    dy = 0
                    if k - len(arc) > 3:
                        return True, None
            if mode == "W" and t <= d:
                v = max(0.0, v - 0.15)
            elif mode == "B" and t <= d:
                v = -1.2
            else:
                v = min(3.0, v + 0.06)
            x += v
            fy = feet + dy
            desc = dy > prevdy
            prevdy = dy
            for tr in trs:
                ex0 = tr.x0 + tr.vx * t + 1
                ex1 = tr.x1 + tr.vx * t - 1
                if x + 6 > ex0 and x - 6 < ex1:
                    bot = tr.bots[t - 1]
                    top = bot - tr.h
                    if fy > top + 2 and fy - hm < bot - 1:
                        if mode == "J" and k > 2 and desc and fy <= top + 11:
                            return True, None
                        return False, t
        return True, None

    def plan(self, cx, feet, v0, hm, trs):
        T = 46
        for tr in trs:
            tr.bots = tr.future_bots(T)
            tr.h = max(8, tr.bot - tr.top)
            if tr.h > 30:
                tr.h = 30
        ok_run, t_run = self.simulate("R", 0, 0, cx, feet, v0, hm, trs, T)
        if ok_run:
            return ("run", None, "")
        order = (8, 12, 4, 16, 26)
        now = [h for h in order if self.simulate("J", h, 0, cx, feet, v0, hm, trs, T)[0]]
        later = [h for h in order if self.simulate("J", h, 4, cx, feet, v0, hm, trs, T)[0]]
        if now and (not later or t_run <= 8):
            pick = now[0]
            if self.noise > 0 and self.rng.random() < self.noise * 0.5:
                pick = self.rng.choice(now)
            return ("jump", pick, "now%s later%s tr%d" % (now, later, t_run))
        if later and not now:
            return ("run", None, "later%s tr%d" % (later, t_run))
        for w in (10, 20, 34):
            if self.simulate("W", 0, w, cx, feet, v0, hm, trs, T)[0]:
                return ("wait", w, "tr%d" % t_run)
        if cx > 30:
            for d in (12, 24):
                if self.simulate("B", 0, d, cx, feet, v0, hm, trs, T)[0]:
                    return ("back", d, "tr%d" % t_run)
        if now:
            return ("jump", now[0], "forced now%s" % now)
        # emergency: jump with the longest time-to-collision
        best = max(order, key=lambda h: (self.simulate("J", h, 0, cx, feet, v0, hm, trs, T)[1] or 99))
        return ("jump", best, "emergency tr%d" % t_run)

    # ------------------------------------------------------------ main
    def act(self, s):
        self.n += 1
        keys = {"Right", "x"}
        mar = self.find_mario(s)
        if mar is None:
            self.prev = s
            return keys
        ml, mr, ry0, feet, big = mar
        sc = self.scroll(s)
        self.last_sc = sc
        self.cum += sc
        self.ry_hist.append(ry0)
        self.ry_hist = self.ry_hist[-4:]
        sup = s[min(239, feet + 1):min(240, feet + 3), ml + 3:mr - 1]
        has_sup = int(sup.any(axis=2).sum()) >= 3
        grounded = len(self.ry_hist) >= 3 and max(self.ry_hist[-3:]) - min(self.ry_hist[-3:]) <= 1 and has_sup
        prog = sc + ((ml - self.pm[0]) if self.pm is not None else 0)
        self.vx = 0.6 * self.vx + 0.4 * prog

        mv = self.movers(s, mar)
        if mv.sum() > 4000:
            mv[:] = False
        E = dil_x(mv, 2) & s.any(axis=2)
        E[:100] = False
        E[208:] = False
        cl = self.clusters(E)
        pipes = self.find_pipes(s)
        trs = self.update_tracks(cl, pipes)
        self.cl = cl
        self.esig = tuple(sorted((int(t.x0), int(t.x1), int(t.top), int(t.bot)) for t in trs))
        cx = (ml + mr) / 2.0
        trs = [t for t in trs if t.x1 >= ml - 2 and t.x0 <= cx + 160]

        # ---- pits (ground row black)
        gnd = s[209:221].any(axis=2).any(axis=0)
        pit_x = None
        x = mr
        while x < W - 4:
            if not gnd[x]:
                x2 = x
                while x2 < W and not gnd[x2]:
                    x2 += 1
                if x2 - x >= 6 or x2 >= W:
                    pit_x = x
                    break
                x = x2
            else:
                x += 1
            if x - mr > 90:
                break
        pit_d = None if pit_x is None else pit_x - mr

        # ---- pipes
        ahead_p = [p for p in pipes if p[1] >= mr - 2 and p[0] > ml and p[2] <= feet - 10]
        pipe = min(ahead_p, key=lambda p: p[0]) if ahead_p else None
        pipe_d = None
        plant = False
        if pipe is not None:
            pipe_d = pipe[0] - mr
            gm = eq(s, GREEN)
            plant = bool(gm[max(40, pipe[2] - 34):pipe[2], pipe[0]:pipe[1] + 1].sum() >= 4)

        zhold = False
        if self.jump_t is not None:
            self.jump_t += 1
            zhold = self.jump_t <= self.hold
            if self.jump_t >= 4 and grounded:
                self.jump_t = None
        want = None
        self.info = ""
        if self.jump_t is None and grounded:
            v0 = max(0.5, min(3.0, self.vx))
            hm = 30 if big else 14
            if pit_d is not None and pit_d <= 9:
                want = (26, "pit d%d" % pit_d)
            if want is None and pipe is not None and pipe_d <= 6:
                keys.discard("x")
                if plant:
                    self.hidden = 0
                else:
                    self.hidden += 1
                self.pwait += 1
                if self.hidden >= 4 or self.pwait > 400:
                    want = (26, "pipe jump top%d hid%d wait%d" % (pipe[2], self.hidden, self.pwait))
                    self.pwait = 0
                else:
                    self.info = "pipe-wait hid%d plant%d" % (self.hidden, int(plant))
                    if self.pwait % 8 == 1:
                        log(self.n, "PIPEWAIT", pipe, "d", pipe_d, "plant", int(plant), "hid", self.hidden)
            elif pipe is None or pipe_d > 40:
                self.pwait = 0
                self.hidden = 0
            elev = [t for t in trs if (not t.hopper) and t.bot < feet - 10 and t.top > 90 and t.x0 - mr < 130 and t.x1 > ml - 8]
            if want is None and elev and self.ewait < 420 and not (pipe is not None and pipe_d <= 6):
                keys.discard("Right")
                keys.discard("x")
                self.ewait += 1
                self.info = "elev-wait"
                if self.ewait % 20 == 1:
                    log(self.n, "ELEVWAIT", [(t.x0, t.x1, t.top, t.bot) for t in elev], "feet", feet)
            elif not elev and self.ewait > 0:
                self.ewait = max(0, self.ewait - 1)
            if want is None and trs and not (elev and self.ewait < 420):
                act, arg, why = self.plan(cx, feet, v0, hm, trs)
                if DBG[0] <= self.n <= DBG[1]:
                    log(self.n, "m", ml, mr, ry0, "v%.1f" % self.vx, act, arg, why,
                        [(t.x0, t.x1, t.top, t.bot, round(t.vx, 1), round(t.vy, 1), int(t.hopper)) for t in trs])
                if act == "jump":
                    want = (arg, "enemy " + why)
                elif act == "back":
                    keys.discard("Right")
                    keys.discard("x")
                    keys.add("Left")
                    self.info = "back %s" % why
                    self.waitn += 1
                    if self.waitn % 10 == 1:
                        log(self.n, "BACK", why, [(t.x0, t.x1, t.top, t.bot, round(t.vx, 1), round(t.vy, 1)) for t in trs])
                elif act == "wait":
                    keys.discard("Right")
                    keys.discard("x")
                    self.info = "wait %s" % why
                    self.waitn += 1
                    if self.waitn % 10 == 1:
                        log(self.n, "WAIT", why, [(t.x0, t.x1, t.top, t.bot, round(t.vx, 1), round(t.vy, 1)) for t in trs])
                else:
                    self.waitn = 0
            if want is None and self.blocked >= 4 and self.n > 20 and not (pipe is not None and pipe_d <= 8):
                want = (26, "blocked")
            if self.noise > 0 and want is None:
                r = self.rng.random()
                if r < self.noise * 0.04:
                    want = (self.rng.choice((4, 8, 12, 16, 26)), "noise")
                elif r < self.noise * 0.07 and self.pause == 0:
                    self.pause = self.rng.randint(4, 18)
            if want is not None:
                if self.noise > 0:
                    want = (max(3, want[0] + self.rng.randint(-3, 4)), want[1])
                self.jump_t = 0
                self.hold = want[0]
                zhold = True
                self.blocked = 0
                log(self.n, "mx", ml, mr, "ry", ry0, "JUMP h%d" % want[0], want[1],
                    [(t.x0, t.x1, t.top, t.bot, round(t.vx, 1), round(t.vy, 1)) for t in trs][:3])
        if grounded and self.jump_t is None and "Right" in keys and prog <= 0 and self.vx < 0.3:
            self.blocked += 1
        else:
            self.blocked = 0
        if self.pause > 0 and grounded and self.jump_t is None:
            self.pause -= 1
            keys.discard("Right")
            keys.discard("x")
        if zhold:
            keys.add("z")
        self.hist.append((s, self.cum, (ml, mr, ry0, feet)))
        self.hist = self.hist[-3:]
        self.pm = (ml, mr, ry0, feet)
        self.prev = s
        return keys


def safe_act(P, s):
    try:
        return P.act(s)
    except Exception as e:
        import traceback
        log("act error", type(e).__name__, e, traceback.format_exc()[-300:])
        return {"Right", "x"}


def enc(keys):
    return "0123456789abcdef"[(1 if "Right" in keys else 0) | (2 if "Left" in keys else 0) | (4 if "x" in keys else 0) | (8 if "z" in keys else 0)]


def dec(c):
    v = int(c, 16)
    k = set()
    if v & 1:
        k.add("Right")
    if v & 2:
        k.add("Left")
    if v & 4:
        k.add("x")
    if v & 8:
        k.add("z")
    return k


FINISH_X = 3100
SEARCH_SECONDS = 1000
SEED_TRACE = "555555555555555555555555555555555555555555555555555555555555555555555555555555555555555555505555505550000005ddddddddddddddddd555555555555555555555555555555500000000000000111111100000000000000000000000000000001111201121111000111111111111111111111009dddddddddddddddddddddddddd555555555555555555555555555555555555555551155555555519ddddddddddddddddddddddddd5555555555555555555555555555555555555555555ddddddddddddddddddddddddddd5555555555555555555555555555555555555511111111111111111111111111111111111111119ddddddddddddddddd5555555555555555555555555555ddddddddd5555555555555555555000000001111111111119ddddddddddddddddddddddd55555555555555555555505555555555000000000000000ddddddddddddddddddddddddddd55555555555555000000000000000000000000000000000000000000000000000000000008dddddddddddddddddddddddddddd555555555555555555555555555555555555555555555555555555500000ddddddddddddddddddddddddddddddd555555555dddddddddd55555555555555580555555555555555555555555555555555001111111111111111111111111111111111111111111111111111111111111111111111111111119ddddddddddddddddddddddddd55555555555555555555555555555555555555552222202222ddddddddd555555555555555005555555555000000555555500000500050000dddddddddd55555555555555000000000005555555555555555555555555555555555555ddddd55555555555555555555555555555555555555ddddddddddddddd8000050005555555555555555555220dddddddddddddddd5555555555555555555555555500000005555ddddddddddddddddddddddddddd55555555555ddddddddddddddddddd5555555555d55555555555555555555555555555555555555555555555555555555555555555555555555550000000000000000000000002222555555555555555555555555555555550000550525ddddd555555555555555555555555555555555555555555555ddddddddddddddddd55555555555555555555555555dddddddddddddddddddddddddddddd55555555dddddddddddddddddddddddddd55555555550000000000000000000000000000000000000000000000000000000000000ddddddddddddd55555555555555555555555500005552222222222222222ddddddddddddd5555555555555555555555555555555555555555ddddddddddddddd5555555555555555555555555ddddddddddddddddddddddddddddd5555555555555555555555555555555555555555555555555555555555555dddddddd555555"


def _shot(s, name):
    try:
        from PIL import Image
        Image.fromarray(s).save(name)
    except Exception:
        pass


def _wx(P):
    return P.cum + (P.pm[0] if P.pm else 0)


def _search(sim, t0, reserve):
    sim.reset()
    P = Pilot()
    trace, poss = [], []
    cps, free = [], list(range(10))
    fails = 0
    deaths = 0
    best_cum = 0
    last_cp_n = -100
    best = (0, [], [])
    prev = None
    frozen = 0
    while sim.budget_left > reserve and time.time() - t0 < SEARCH_SECONDS:
        s = sim.screen()
        if prev is not None and np.array_equal(s, prev):
            frozen += 1
        else:
            frozen = 0
        prev = s
        if frozen >= 8:
            wx = _wx(P)
            if wx >= FINISH_X:
                log("FINISHED wx", wx, "frames", len(trace), "deaths", deaths, "t", int(time.time() - t0))
                _shot(s, "finish.png")
                return True, trace[:-6], poss[:-6]
            deaths += 1
            fails += 1
            if wx > best[0]:
                best = (wx, list(trace[:-8]), list(poss[:-8]))
            log("DEATH #%d wx %d n %d fails %d budget %d" % (deaths, wx, len(trace), fails, sim.budget_left))
            if deaths <= 12:
                _shot(s, "death_%d.png" % deaths)
            noise = min(1.0, 0.25 + 0.08 * fails)
            if not cps:
                sim.reset()
                P = Pilot()
                trace, poss = [], []
            else:
                back = min(fails // 2, len(cps) - 1)
                cp = cps[-1 - back]
                for c in cps[len(cps) - back:]:
                    free.append(c["slot"])
                cps = cps[:len(cps) - back]
                sim.load(cp["slot"])
                P = copy.deepcopy(cp["P"])
                trace = trace[:cp["n"]]
                poss = poss[:cp["n"]]
            P.noise = noise
            P.rng = random.Random(1000 + deaths)
            prev = None
            frozen = 0
            last_cp_n = len(trace)
            continue
        grounded = len(P.ry_hist) >= 3 and max(P.ry_hist[-3:]) == min(P.ry_hist[-3:])
        if len(trace) - last_cp_n >= 40 and frozen == 0 and P.jump_t is None and grounded and P.pm is not None:
            if P.cum > best_cum + 40:
                best_cum = P.cum
                fails = 0
                P.noise = 0.0
            if not free:
                old = cps.pop(0)
                free.append(old["slot"])
            slot = free.pop(0)
            sim.save(slot)
            cps.append({"slot": slot, "P": copy.deepcopy(P), "n": len(trace)})
            last_cp_n = len(trace)
        keys = safe_act(P, s)
        sim.step(keys)
        trace.append(enc(keys))
        poss.append((P.cum, P.pm[0] if P.pm else 0, P.pm[2] if P.pm else 0))
    wx = _wx(P)
    if wx >= best[0]:
        best = (wx, list(trace), list(poss))
    log("search ended unfinished wx", best[0], "frames", len(best[1]), "deaths", deaths)
    return False, best[1], best[2]


def _verify(sim, trace, poss):
    """replay the trace on a fresh variant; return first divergence frame or -1"""
    sim.reset()
    P = Pilot()
    div = -1
    for i, c in enumerate(trace):
        s = sim.screen()
        safe_act(P, s)
        if P.pm is not None and i < len(poss):
            w, m, r = poss[i]
            if abs(P.cum - w) > 3 or abs(P.pm[0] - m) > 4 or abs(P.pm[2] - r) > 5:
                div = i
                break
        sim.step(dec(c))
    return div


def solve(sim):
    t0 = time.time()
    try:
        sim.memory["trace"] = ""
        sim.memory["pos"] = []
        fin, trace, poss = _search(sim, t0, 100)
        sim.memory["trace"] = "".join(trace)
        sim.memory["pos"] = [list(p) for p in poss]
        sim.memory["finished"] = bool(fin)
        sim.memory.flush()
        if fin:
            for k in range(2):
                if sim.budget_left > len(trace) + 50 and time.time() - t0 < 1300:
                    d = _verify(sim, trace, poss)
                    log("verify", k, "divergence frame", d)
    except Exception as e:
        import traceback
        log("solve error", type(e).__name__, e, traceback.format_exc()[-500:])
    print("\n".join(DEBUG[-40:]))
    return []


def play(sim):
    tr, pos = "", []
    try:
        tr = sim.memory.get("trace", "") or ""
        pos = sim.memory.get("pos", []) or []
    except Exception:
        tr, pos = "", []
    tr = tr[:len(pos)]
    P = Pilot()
    diverged = not tr
    i = 0
    frozen = 0
    prev = None
    try:
        while sim.budget_left > 0:
            s = sim.screen()
            if prev is not None and np.array_equal(s, prev):
                frozen += 1
                if frozen > 60:
                    break
            else:
                frozen = 0
            prev = s
            kp = safe_act(P, s)
            keys = kp
            if not diverged:
                if i < len(tr):
                    w, m, r = pos[i]
                    if P.pm is not None and (abs(P.cum - w) > 3 or abs(P.pm[0] - m) > 4 or abs(P.pm[2] - r) > 5):
                        diverged = True
                        log("DIVERGED at", i, "cum", P.cum, w, "ml", P.pm[0], m, "ry", P.pm[2], r)
                    else:
                        keys = dec(tr[i])
                else:
                    diverged = True
                    log("trace ended at", i)
            sim.step(keys)
            i += 1
    except Exception as e:
        log("end", type(e).__name__, e)
    print("\n".join(DEBUG[-20:]))


# =====================================================================
# trace library with class branches (variants differ in enemy phase)
# =====================================================================
import zlib

SIGN = (27, 64)


def _sig(s):
    return (s[100:208:4, ::4].astype(np.int32).sum(axis=2) // 3).astype(np.uint8)


def _crc(s):
    return zlib.crc32(s.tobytes())


def _sigdist(a, b):
    """distance between two enemy signatures (lists of (x0,x1,top,bot))"""
    if a is None or b is None:
        return 10 ** 9
    a = [tuple(x) for x in a]
    b = [tuple(x) for x in b]
    d = 1000 * abs(len(a) - len(b))
    for p, q in zip(a, b):
        d += sum(abs(u - v) for u, v in zip(p, q))
    return d


def _mismatch(P, rec):
    w, m, r = rec
    return P.pm is not None and (abs(P.cum - w) > 3 or abs(P.pm[0] - m) > 4 or abs(P.pm[2] - r) > 5)


def _search_from(sim, st, cps, free, t0, reserve, may_reset, keep_base):
    P = st["P"]
    keys_l, poss, crcs, sigs = st["keys"], st["poss"], st["crcs"], st["sigs"]
    fails = 0
    deaths = 0
    best_cum = P.cum
    last_cp_n = len(keys_l) if cps else -100
    prev = None
    frozen = 0

    def trunc(n):
        del keys_l[n:], poss[n:], crcs[n:], sigs[n:]

    while sim.budget_left > reserve and time.time() - t0 < SEARCH_SECONDS:
        s = sim.screen()
        if prev is not None and np.array_equal(s, prev):
            frozen += 1
        else:
            frozen = 0
        prev = s
        if frozen >= 8:
            wx = _wx(P)
            if wx >= FINISH_X:
                trunc(len(keys_l) - 6)
                st["P"] = P
                log("FINISHED wx", wx, "frames", len(keys_l), "deaths", deaths, "t", int(time.time() - t0))
                _shot(s, "finish.png")
                return True
            deaths += 1
            fails += 1
            klog("DEATH #%d wx %d n %d fails %d budget %d" % (deaths, wx, len(keys_l), fails, sim.budget_left))
            if deaths <= 6:
                _shot(s, "death_%d_%d.png" % (len(DEBUG) % 7, deaths))
            if cps:
                back = min(fails // 2, len(cps) - 1)
                cp = cps[-1 - back]
                for c in cps[len(cps) - back:]:
                    free.append(c["slot"])
                del cps[len(cps) - back:]
                sim.load(cp["slot"])
                P = copy.deepcopy(cp["P"])
                trunc(cp["n"])
            elif may_reset:
                sim.reset()
                P = Pilot()
                trunc(0)
            else:
                st["P"] = P
                return False
            P.noise = min(1.0, 0.25 + 0.08 * fails)
            P.rng = random.Random(1000 + deaths)
            prev = None
            frozen = 0
            last_cp_n = len(keys_l)
            continue
        grounded = len(P.ry_hist) >= 3 and max(P.ry_hist[-3:]) == min(P.ry_hist[-3:])
        if len(keys_l) - last_cp_n >= 40 and frozen == 0 and P.jump_t is None and grounded and P.pm is not None:
            if P.cum > best_cum + 40:
                best_cum = P.cum
                fails = 0
                P.noise = 0.0
            if not free:
                old = cps.pop(1 if (keep_base and len(cps) > 1) else 0)
                free.append(old["slot"])
            slot = free.pop(0)
            sim.save(slot)
            cps.append({"slot": slot, "P": copy.deepcopy(P), "n": len(keys_l)})
            last_cp_n = len(keys_l)
        k = safe_act(P, s)
        sim.step(k)
        keys_l.append(enc(k))
        poss.append((P.cum, P.pm[0] if P.pm else 0, P.pm[2] if P.pm else 0))
        crcs.append(P.esig)
        sigs.append(0)
    st["P"] = P
    log("search ended unfinished wx", _wx(P), "frames", len(keys_l), "deaths", deaths)
    return False


def _close(h, q):
    return abs(h[0] - q[0]) <= 3 and abs(h[1] - q[1]) <= 4 and abs(h[2] - q[2]) <= 5


class Follower:
    """follows the library trace whose recorded Mario positions match the observed ones; falls back to the pilot"""

    def __init__(self, lib):
        self.lib = lib
        self.cur = 0
        self.diverged = False
        self.hist = []
        self.switched = False

    def _ok(self, j, i):
        e = self.lib[j]
        pos = e.get("pos") or e.get("poss")
        tr = e["trace"]
        if i >= len(tr) or i >= len(pos):
            return False
        for k, h in enumerate(reversed(self.hist)):
            q = i - k
            if q < 0:
                break
            if q >= len(pos) or not _close(h, pos[q]):
                return False
        return True

    def step(self, i, P):
        """returns recorded keys for frame i or None (use pilot)"""
        self.switched = False
        if P.pm is None:
            return None
        self.hist.append((P.cum, P.pm[0], P.pm[2]))
        self.hist = self.hist[-3:]
        lib = self.lib
        if not self.diverged and self._ok(self.cur, i):
            for j, ch in enumerate(lib):
                if ch.get("parent") == self.cur and ch.get("b") == i:
                    dc = _sigdist(P.esig, ch.get("sig_b"))
                    dp = _sigdist(P.esig, ch.get("psig_b"))
                    if dc < dp and self._ok(j, i):
                        self.cur = j
                        self.switched = True
                        break
        if self._ok(self.cur, i):
            self.diverged = False
            return dec(lib[self.cur]["trace"][i])
        for j in range(len(lib) - 1, -1, -1):
            if j != self.cur and self._ok(j, i):
                self.cur = j
                self.diverged = False
                self.switched = True
                return dec(lib[j]["trace"][i])
        self.diverged = True
        return None


def _try_variant(sim, lib, t0, reserve):
    """play the library on a fresh variant; if it dies, branch a new trace off the active one"""
    sim.reset()
    P = Pilot()
    fol = Follower(lib)
    played, opos, ocrc, osig = [], [], [], []
    ring, rfree = [], list(range(9))
    first_diff = None
    cur = 0
    diverged = False
    frozen = 0
    prev = None
    i = 0
    dead = False
    while sim.budget_left > reserve and time.time() - t0 < SEARCH_SECONDS:
        s = sim.screen()
        if prev is not None and np.array_equal(s, prev):
            frozen += 1
        else:
            frozen = 0
        prev = s
        if frozen >= 8:
            if _wx(P) >= FINISH_X:
                klog("variant ok, frames", i - 8, "diverged", diverged, "cur", cur)
                return "ok"
            dead = True
            break
        if i % 14 == 0 and frozen == 0:
            if not rfree:
                old = ring.pop(0)
                rfree.append(old["slot"])
            slot = rfree.pop(0)
            sim.save(slot)
            ring.append({"slot": slot, "i": i, "P": copy.deepcopy(P)})
        kp = safe_act(P, s)
        keys = kp
        c = P.esig
        sg = P.esig
        old_cur = fol.cur
        rk = fol.step(i, P)
        if fol.cur != old_cur or fol.switched:
            first_diff = None
            klog("follower switched to entry", fol.cur, "at frame", i)
        if rk is not None:
            keys = rk
        cur = fol.cur
        diverged = fol.diverged
        e = lib[cur]
        if first_diff is None and i < len(e["crcs"]) and c != e["crcs"][i]:
            first_diff = i
        sim.step(keys)
        played.append(enc(keys))
        opos.append((P.cum, P.pm[0] if P.pm else 0, P.pm[2] if P.pm else 0))
        ocrc.append(c)
        osig.append(sg)
        i += 1
    if not dead:
        return "budget"
    klog("variant died at", i - 8, "wx", _wx(P), "cur", cur, "first_diff", first_diff, "diverged", diverged)
    if first_diff is None or first_diff < 2:
        return "nodiff"
    cand = [r for r in ring if r["i"] >= first_diff and r["i"] < len(osig) and r["i"] < len(e["crcs"]) and osig[r["i"]] != e["crcs"][r["i"]]]
    if not cand:
        return "noring"
    r0 = cand[0]
    b = r0["i"]
    sim.load(r0["slot"])
    P2 = copy.deepcopy(r0["P"])
    sim.save(9)
    st = {"P": P2, "keys": list(played[:b]), "poss": list(opos[:b]), "crcs": list(ocrc[:b]), "sigs": list(osig[:b])}
    cps = [{"slot": 9, "P": copy.deepcopy(P2), "n": b}]
    free = list(range(9))
    fin = _search_from(sim, st, cps, free, t0, 150, False, True)
    if not fin:
        return "fail"
    # signature of this variant's screen at frame b (== observed screen before choosing keys at b)
    sig_b = osig[b] if b < len(osig) else None
    psig = e["crcs"][b] if b < len(e["crcs"]) else None
    lib.append({"trace": "".join(st["keys"]), "poss": list(st["poss"]), "crcs": list(st["crcs"]), "sigs": list(st["sigs"]),
                "parent": cur, "b": b, "sig_b": sig_b, "psig_b": psig})
    klog("branch added: parent", cur, "b", b, "len", len(st["keys"]))
    return "branched"


def _dump(sim, lib):
    out = []
    for e in lib:
        out.append({"trace": e["trace"], "pos": [list(p) for p in e["poss"]], "parent": e["parent"], "b": e["b"],
                    "sig_b": None if e["sig_b"] is None else [list(x) for x in e["sig_b"]],
                    "psig_b": None if e["psig_b"] is None else [list(x) for x in e["psig_b"]]})
    sim.memory["lib"] = out
    try:
        sim.memory.flush()
    except Exception:
        pass


def _replay_seed(sim, seed):
    """replay a known finishing trace on the first variant, recording positions / enemy signatures"""
    sim.reset()
    P = Pilot()
    poss, esigs = [], []
    prev = None
    frozen = 0
    for c in seed:
        s = sim.screen()
        if prev is not None and np.array_equal(s, prev):
            frozen += 1
        else:
            frozen = 0
        prev = s
        safe_act(P, s)
        poss.append((P.cum, P.pm[0] if P.pm else 0, P.pm[2] if P.pm else 0))
        esigs.append(P.esig)
        if frozen >= 8:
            break
        sim.step(dec(c))
    for _ in range(14):
        if frozen >= 8:
            break
        s = sim.screen()
        if np.array_equal(s, prev):
            frozen += 1
        else:
            frozen = 0
        prev = s
        sim.step({"Right", "x"})
    if frozen >= 8 and _wx(P) >= FINISH_X and len(poss) >= len(seed) - 8:
        return {"trace": seed, "poss": poss, "crcs": esigs, "sigs": [], "parent": None, "b": -1, "sig_b": None, "psig_b": None}
    return None


def solve(sim):
    t0 = time.time()
    lib = []
    try:
        sim.memory["lib"] = []
        lib0 = None
        if SEED_TRACE:
            try:
                lib0 = _replay_seed(sim, SEED_TRACE)
            except Exception as e:
                log("seed error", type(e).__name__, e)
                lib0 = None
            klog("seed replay", "ok" if lib0 is not None else "FAILED", "budget left", sim.budget_left)
        if lib0 is not None:
            lib.append(lib0)
            fin = True
        else:
            sim.reset()
            st = {"P": Pilot(), "keys": [], "poss": [], "crcs": [], "sigs": []}
            fin = _search_from(sim, st, [], list(range(9)), t0, 100, True, False)
            lib.append({"trace": "".join(st["keys"]), "poss": list(st["poss"]), "crcs": list(st["crcs"]), "sigs": list(st["sigs"]),
                        "parent": None, "b": -1, "sig_b": None, "psig_b": None})
            klog("T1 finished", fin, "len", len(st["keys"]), "budget left", sim.budget_left)
        _dump(sim, lib)
        if fin:
            streak = 0
            tries = 0
            while sim.budget_left > 2300 and time.time() - t0 < 1200 and tries < 12 and streak < 5:
                tries += 1
                res = _try_variant(sim, lib, t0, 150)
                klog("variant", tries, "result", res, "budget left", sim.budget_left)
                if res == "ok":
                    streak += 1
                else:
                    streak = 0
                if res == "branched":
                    _dump(sim, lib)
                if res in ("budget",):
                    break
    except Exception as e:
        import traceback
        log("solve error", type(e).__name__, e, traceback.format_exc()[-600:])
    try:
        _dump(sim, lib)
    except Exception:
        pass
    print("\n".join(KEY))
    print("\n".join(DEBUG[-12:]))
    return []


def play(sim):
    lib = []
    try:
        lib = list(sim.memory.get("lib", []) or [])
    except Exception:
        lib = []
    P = Pilot()
    fol = Follower(lib) if lib else None
    i = 0
    frozen = 0
    prev = None
    try:
        while sim.budget_left > 0:
            s = sim.screen()
            if prev is not None and np.array_equal(s, prev):
                frozen += 1
                if frozen > 60:
                    break
            else:
                frozen = 0
            prev = s
            kp = safe_act(P, s)
            keys = kp
            if fol is not None:
                try:
                    old = (fol.cur, fol.diverged)
                    rk = fol.step(i, P)
                    if (fol.cur, fol.diverged) != old:
                        log("follower", i, "cur", fol.cur, "diverged", fol.diverged)
                    if rk is not None:
                        keys = rk
                except Exception as e:
                    log("follower error", type(e).__name__, e)
            sim.step(keys)
            i += 1
    except Exception as e:
        log("end", type(e).__name__, e)
    print("\n".join(DEBUG[-20:]))

