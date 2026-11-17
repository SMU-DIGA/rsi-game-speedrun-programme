# ==================================================================================================
# Super Mario Bros. (NES) -- level 1-4  |  超级马里奥兄弟（NES） 关 1-4
# Pass rate on 100 random variants never used before: 100/100; fastest finish 769 frames (12.82 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 769 帧（60 帧/秒，约 12.82 秒）。
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
#   closed-loop vision policy (Mario tracker, wall/pit/flame reactive jumps); at the bridge run at
#    full speed and JUMP OVER Bowser (max-height jump ~60px before him), old run-under controller 
#   kept as fall
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _mask                          line 19    
#   class Agent                        line 23    
#   def play                           line 453   
#   def solve                          line 467   
# ==================================================================================================

# idea: closed-loop vision policy (Mario tracker, wall/pit/flame reactive jumps); at the bridge run at full speed and JUMP OVER Bowser (max-height jump ~60px before him), old run-under controller kept as fallback
import numpy as np

SKIN = (252, 160, 68)
BRN = (172, 124, 0)
RED = (248, 56, 0)
RUN = ("Right", "x")
BR_WX = 2120
FIN_SCROLL = 2000
FIN_MX = 225
JHINT = []
SHINT = [[2042, 2066]]
_HS = [155, 149, 144, 140, 136, 133, 131, 129, 128, 127, 128, 128, 130, 132, 134, 138, 142, 146, 151, 157]
HOP = [float(np.interp(t / 3.0, range(len(_HS)), _HS)) for t in range(58)]   # Bowser bottom y, frames since take-off
TH_IN = 9      # first frame (since take-off) his bottom is <= ~141
TH_OUT = 46    # last such frame


def _mask(s, col):
    return (s[:, :, 0] == col[0]) & (s[:, :, 1] == col[1]) & (s[:, :, 2] == col[2])


class Agent:
    def __init__(self, P=None):
        self.P = dict(W_WALL=20, W_PIT=7, HOLD=24, B_WALL=20, B_HOLD=24, FL_D=64, FL_HOLD=12,
                      AIR_D=60, GAP_LO=12, GAP_HI=24, VMAX=2.9, ACC=0.11, MARG_IN=2, MARG_OUT=3,
                      DEC=0.15, DEC_C=0.06, HOLD_W=8, HOLD_P=4, HOLD_AX=14, AIR_BRAKE=0, ML=6, MR=4, WALK=0, WALK_FROM=1995, HOLD_PW=7, W_AX=6, PAST=6, MARG_OUT_FORCED=-6,
                      OJ=1, OJ_D=60, OJ_LO=44, OJ_HOLD=40, OJ_FROM=2040)
        if P:
            self.P.update(P)
        self.prev = None
        self.scroll = 0
        self.wx = 0
        self.last_shift = 0
        self.jt = None
        self.hold = 0
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
        self.pm = None
        self.pflame = None
        self.pbot = None
        self.pbx = None
        self.vb = 0.0
        self.pwx = None
        self.vx = 0.0
        self.J = JHINT
        self.S = SHINT
        self.jidx = 0
        self.log = []
        self.go_until = -1
        self.bdone = False
        self.jkind = ""
        self.btrace = []
        self.th = None
        self.walk = False
        self.oj_done = False
        self.oj_frame = None

    # ---------------- perception ----------------
    def mario(self, s):
        m = _mask(s, BRN)
        m[:40] = False
        ys, xs = np.nonzero(m)
        if len(ys) < 8:
            return None
        if self.pm is not None:
            sel = (np.abs(xs - self.pm[0]) <= 32) & (np.abs(ys - self.pm[1]) <= 32)
            if sel.sum() >= 8:
                xs, ys = xs[sel], ys[sel]
        mx, my = int(np.median(xs)), int(np.median(ys))
        sel = (np.abs(xs - mx) <= 9) & (np.abs(ys - my) <= 10)
        if sel.sum() < 8:
            return None
        xs, ys = xs[sel], ys[sel]
        self.pm = (mx, my)
        return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    def track_scroll(self, s):
        g = s.sum(axis=2, dtype=np.int32)
        rows = np.r_[34:78:2, 162:190:2]
        cur = g[rows]
        if self.prev is not None:
            pv = self.prev
            best, bs = None, 0
            for sh in range(0, 6):
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

    def flame_dist(self, s, x1, y0, y1):
        xa, xb = x1 + 3, min(256, x1 + 100)
        ya, yb = max(40, y0 - 1), min(190, y1 + 1)
        if xa >= xb or ya >= yb:
            return None
        reg = s[ya:yb, xa:xb]
        m = _mask(reg, RED) | _mask(reg, SKIN)
        if m.sum() < 14:
            return None
        ys, xs = np.nonzero(m)
        w = int(xs.max() - xs.min()) + 1
        h = int(ys.max() - ys.min()) + 1
        if w >= 8 and h <= 14 and w <= 44:
            return int(xs.min()) + xa - x1
        return None

    def low_flame(self, s, x1, y1):
        xa, xb = x1 + 3, min(256, x1 + 110)
        ya, yb = max(40, y1 - 6), min(190, y1 + 36)
        if xa >= xb or ya >= yb:
            return None
        reg = s[ya:yb, xa:xb]
        m = _mask(reg, RED) | _mask(reg, SKIN)
        if m.sum() < 12:
            return None
        ys, xs = np.nonzero(m)
        w = int(xs.max() - xs.min()) + 1
        h = int(ys.max() - ys.min()) + 1
        if w >= 8 and h <= 14 and w <= 44:
            return int(xs.min()) + xa
        return None

    def bowser_box(self, s):
        reg = s[40:200, :].astype(np.int16)
        g = (reg[:, :, 1] > reg[:, :, 0] + 60) & (reg[:, :, 1] > reg[:, :, 2] + 60)
        if g.sum() < 40:
            return None
        ys, xs = np.nonzero(g)
        return int(xs.min()), int(xs.max()), int(ys.max()) + 40

    def fire_between(self, s, x1, y0, y1, bx0):
        """leftmost x of a fireball (red/orange) between Mario and Bowser, at Mario's height band"""
        xa, xb = x1 + 2, min(256, bx0 - 1)
        ya, yb = max(40, y0 - 14), min(190, y1 + 2)
        if xa >= xb or ya >= yb:
            return None
        reg = s[ya:yb, xa:xb]
        m = _mask(reg, RED)
        if m.sum() < 6:
            return None
        ys, xs = np.nonzero(m)
        return int(xs.min()) + xa

    def pit_width(self, s, x1, y1):
        """width in px of the hole ahead (first hole column to next solid column), 99 if open to screen edge"""
        ya, yb = y1 + 2, min(y1 + 34, 240)
        reg = s[ya:yb, x1 + 1:256].astype(np.int16)
        solid = reg.any(axis=2) & ~((reg[:, :, 0] == 248) & (reg[:, :, 1] == 56) & (reg[:, :, 2] == 0))
        col = solid.any(axis=0)
        idx = np.nonzero(~col)[0]
        if not len(idx):
            return 0
        a = idx[0]
        nz = np.nonzero(col[a:])[0]
        if not len(nz):
            return 99
        return int(nz[0])

    def in_skip(self):
        for a, b in self.S:
            if a <= self.wx < b:
                return True
        return False

    def start_jump(self, hold, keys):
        self.jt = 1
        self.hold = hold
        keys.add("z")

    # ---------------- bridge ----------------
    def update_clock(self, b):
        """self.th = frames since Bowser's current hop began (None when he stands on the ground)"""
        if b >= 153:
            self.th = None
            return
        if self.th is not None:
            self.th += 1
            return
        if self.pbot is not None and self.pbot >= 153 and b < self.pbot:
            self.th = 1          # just lifted off
            return
        # joined mid-hop: estimate the phase from height + direction
        if self.vb < 0 or (self.pbot is None and b > 135):
            cands = [t for t in range(0, 29) if HOP[t] <= b + 0.5]
            self.th = cands[0] if cands else 28
        elif self.vb > 0:
            cands = [t for t in range(29, 58) if HOP[t] >= b - 0.5]
            self.th = cands[0] if cands else 57
        else:
            self.th = 28

    def tcover(self, X, v0):
        """frames to cover X px from speed v0 holding Right+run"""
        v = max(v0, 0.0)
        x = 0.0
        t = 0
        P = self.P
        while x < X and t < 200:
            v = min(P["VMAX"], v + P["ACC"])
            x += v
            t += 1
        return t

    def bridge_ctl(self, s, x0, x1, y0, y1, bb):
        """returns key set (or None to fall through)"""
        P = self.P
        bx0, bx1, b = bb
        bx0 -= P["ML"]
        bx1 += P["MR"]
        w = bx1 - bx0 + 1
        g = bx0 - x1
        keys = set()
        if x0 > bx1 + P["PAST"]:
            return None      # past Bowser: normal logic (climb the axe pedestal)
        v = self.vx
        # Bowser hop phase (clock since take-off)
        airborne = self.th is not None
        if airborne:
            T_in = max(0.0, TH_IN - self.th)
            T_out = TH_OUT - self.th
        else:
            T_in = 1e9
            T_out = -1e9
        A = max(g, 0)
        C = max(g, 0) + w + 16
        ta = self.tcover(A, v) if A > 0 else 0
        tc = self.tcover(C, v)
        # fireball safety
        fb = self.fire_between(s, x1, y0, y1, bx0)
        if fb is not None and fb - x1 < 34 and y1 >= 156 and self.jt is None:
            self.log.append((self.frame, self.wx, "bfire", fb - x1))
            self.start_jump(10, keys)
            keys.add("Right")
            keys.add("x")
            return keys
        tag = "?"
        res = None
        vv = max(v, 0.0)
        stop_gap = g - (vv * vv) / (2 * P["DEC"])       # gap left if we brake now (Left)
        coast_gap = g - (vv * vv) / (2 * P["DEC_C"])    # gap left if we release everything
        cant_stop = stop_gap < 2
        mo = P["MARG_OUT_FORCED"] if cant_stop else P["MARG_OUT"]
        mi = 0 if g <= 0 else P["MARG_IN"]
        if airborne and ta >= T_in + mi and tc <= T_out - mo:
            self.go_until = self.frame + 70
            res, tag = set(RUN), "G"
        elif self.frame < self.go_until:
            res, tag = set(RUN), "C"   # committed, keep running under him
        else:
            # hold pattern: keep a gap
            if stop_gap < P["GAP_LO"]:
                if vv > 0.3:
                    res, tag = {"Left"}, "L"
                elif g < P["GAP_LO"] - 6:
                    res, tag = {"Left"}, "L"
                else:
                    res, tag = set(), "0"
            elif coast_gap > P["GAP_HI"]:
                res, tag = ({"Right", "x"}, "R") if g > 70 else ({"Right"}, "r")
            else:
                res, tag = set(), "0"
        self.btrace.append((self.frame, g, round(v, 1), b, round(self.vb, 1), "%s ta%d tc%d Ti%.0f To%.0f" % (
            tag, ta, tc, min(T_in, 99), max(T_out, -99))))
        return res

    # ---------------- policy ----------------
    def act(self, s):
        self.frame += 1
        self.track_scroll(s)
        mar = self.mario(s)
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
        self.wx = self.scroll + x1
        if self.pwx is not None:
            self.vx = 0.6 * self.vx + 0.4 * (self.wx - self.pwx)
        self.pwx = self.wx
        self.walk = bool(self.P["WALK"]) and self.P["WALK_FROM"] <= self.wx < 2190
        keys = {"Right"} if self.walk else set(RUN)
        dy = 0 if self.prev_fy is None else y1 - self.prev_fy
        if self.prev_fy is not None and y1 == self.prev_fy:
            self.stable += 1
        else:
            self.stable = 0
        self.prev_fy = y1
        bb = self.bowser_box(s) if self.wx >= 2000 else None
        if bb is not None:
            self.vb = (bb[2] - self.pbot) if self.pbot is not None else 0.0
            self.update_clock(bb[2])
            self.pbot = bb[2]
        else:
            self.pbot = None
            self.vb = 0.0
            self.th = None
        # --- low flame tracker ---
        lf = self.low_flame(s, x1, y1)
        fv = 0.0
        if lf is not None:
            if self.pflame is not None and self.frame - self.pflame[1] <= 2:
                fv = (self.pflame[0] - (lf + self.scroll))
            self.pflame = (lf + self.scroll, self.frame)
        else:
            self.pflame = None
        flame_close = (lf is not None and fv >= 1.0 and lf - x1 <= self.P["AIR_D"])
        if (flame_close and self.wx < BR_WX - 20 and self.stable < 2 and dy > 0
                and self.jt is None and y1 + 2 < 238):
            fl = s[y1 + 1:min(y1 + 40, 240), x0:x1 + 1].astype(np.int16)
            solid = fl.any(axis=2) & ~((fl[:, :, 0] == 248) & (fl[:, :, 1] == 56) & (fl[:, :, 2] == 0))
            if solid.any():
                self.log.append((self.frame, self.wx, "airdodge", lf - x1))
                return {"Left"}
        if self.jt is not None:
            self.jt += 1
            if self.jt <= self.hold:
                keys.add("z")
                return keys
            self.jt = None
            self.cool = 3
            return keys
        # in the air over the last pit with Bowser in view: once safely above the far platform, brake with Left
        if (bb is not None and self.wx >= 2090 and self.stable < 2 and self.P["AIR_BRAKE"]
                and y1 + 50 < 240 and x0 + 4 < 256):
            col = s[y1 + 1:y1 + 50, x0 + 4].astype(np.int16)
            solid = col.any(axis=1) & ~((col[:, 0] == 248) & (col[:, 1] == 56) & (col[:, 2] == 0))
            if solid.any() and (x0 > bb[1] or bb[0] - x1 > 6):
                self.log.append((self.frame, self.wx, "airbrake", bb[0] - x1))
                return {"Left"}
        # --- jump OVER Bowser (full-speed max jump) ---
        oj_wait = False
        if self.P["OJ"] and bb is not None and self.wx >= self.P["OJ_FROM"] and not self.oj_done:
            gap = bb[0] - x1
            if x0 <= bb[1] and gap >= -4:
                if gap > self.P["OJ_D"]:
                    oj_wait = True
                elif gap >= self.P["OJ_LO"] and self.stable >= 2 and y1 >= 154:
                    self.oj_done = True
                    self.oj_frame = self.frame
                    self.log.append((self.frame, self.wx, "overjump", gap, bb[2], round(self.vx, 2)))
                    self.start_jump(self.P["OJ_HOLD"], keys)
                    return keys
        in_bridge_zone = self.wx >= BR_WX + 20 and bb is not None and y1 >= 150 and self.stable >= 2
        if in_bridge_zone and oj_wait:
            in_bridge_zone = False
        if in_bridge_zone:
            r = self.bridge_ctl(s, x0, x1, y0, y1, bb)
            if r is not None:
                return r
        fd = self.flame_dist(s, x1, y0, y1)
        if (fd is not None and fd <= self.P["FL_D"] and y1 + 1 < 240 and bb is None):
            seg = s[y1 + 1, x0 + 1:x1].astype(np.int16)
            floor = seg.any(axis=1) & ~((seg[:, 0] == 248) & (seg[:, 1] == 56) & (seg[:, 2] == 0))
            if floor.any():
                self.log.append((self.frame, self.wx, "flame", fd))
                self.start_jump(self.P["FL_HOLD"], keys)
                return keys
        if self.cool > 0:
            self.cool -= 1
            return keys
        if self.stable < 2:
            return keys
        while self.jidx < len(self.J) and self.J[self.jidx][0] < self.wx - 40:
            self.jidx += 1
        if self.jidx < len(self.J) and self.wx >= self.J[self.jidx][0]:
            h = self.P["HOLD_PW"] if self.walk else self.J[self.jidx][1]
            self.jidx += 1
            self.log.append((self.frame, self.wx, "hint", h, None if bb is None else (bb[0] - x1, bb[2], self.th)))
            self.start_jump(h, keys)
            return keys
        ww = self.P["W_AX"] if (self.wx >= 2190 and bb is not None) else self.P["W_WALL"]
        W = max(ww, self.P["W_PIT"] + 4)
        xa, xb = x1 + 1, min(255, x1 + W)
        if xa >= 255:
            return keys
        top = max(40, y1 - 13)
        box = s[top:y1 - 1, xa:xb + 1].astype(np.int16)
        nb = box.any(axis=2)
        for c in (RED, SKIN, BRN):
            nb &= ~((box[:, :, 0] == c[0]) & (box[:, :, 1] == c[1]) & (box[:, :, 2] == c[2]))
        cols = nb.any(axis=0)
        jump = False
        self.jkind = ""
        if cols[:ww].any():
            jump = True
            self.jkind = "wall%d" % int(np.nonzero(cols)[0][0])
        if not jump and y1 + 3 < 236 and self.wx < 2040:
            fl = s[y1 + 2:min(y1 + 34, 240), x1 + 1: min(256, x1 + 1 + self.P["W_PIT"] + 1)].astype(np.int16)
            solid = fl.any(axis=2) & ~((fl[:, :, 0] == 248) & (fl[:, :, 1] == 56) & (fl[:, :, 2] == 0))
            holes = ~(solid.any(axis=0))
            idx = np.nonzero(holes)[0]
            if len(idx) and idx[0] <= self.P["W_PIT"]:
                jump = True
                self.jkind = "pit%d" % self.pit_width(s, x1, y1)
        if self.last_shift == 0:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.stuck > 45:
            self.stuck = 0
            self.start_jump(self.P["HOLD"], keys)
            return keys
        if jump:
            if self.in_skip():
                return keys
            self.log.append((self.frame, self.wx, "jump", self.jkind,
                             None if bb is None else (bb[0] - x1, bb[2], self.th)))
            if self.wx >= 2190:
                h = self.P["HOLD_AX"]
            elif self.walk:
                h = self.P["HOLD_PW"]
            else:
                h = self.P["HOLD_W"] if self.jkind.startswith("wall") else self.P["HOLD_P"]
            self.start_jump(h, keys)
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

    def finished(self):
        return (self.frozen >= 5 and self.nomario == 0 and self.scroll >= FIN_SCROLL
                and self.mx is not None and self.mx >= FIN_MX)


def play(sim):
    ag = Agent()
    n = 0
    try:
        while sim.budget_left > 0 and n < 7950:
            ag.step(sim)
            n += 1
            if ag.frozen > 40:
                break
    finally:
        print("play end frame", n, "wx", ag.wx, "oj", ag.oj_done, "log", ag.log[-6:])
        print("btrace tail", ag.btrace[-4:])


def solve(sim):
    return []

