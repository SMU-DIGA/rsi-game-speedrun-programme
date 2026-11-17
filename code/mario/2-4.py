# ==================================================================================================
# Super Mario Bros. (NES) -- level 2-4  |  超级马里奥兄弟（NES） 关 2-4
# Pass rate on 100 random variants never used before: 87/100; fastest finish 999 frames (16.65 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 87/100；最快一局 999 帧（60 帧/秒，约 16.65 秒）。
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
#   c13 reactive pixel policy + flame tracker + trajectory-model air dodge (short hops choose hold
#    / release early so Mario's feet-path avoids incoming Bowser-fire lanes)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _ext                           line 37    
#   def mario_box                      line 54    
#   def fire_comps                     line 62    connected comps (radius 2) of a boolean mask -> list of (xl, xr, yt, yb, n)
#   class Agent                        line 86    
#   def play                           line 567   
#   def solve                          line 584   
# ==================================================================================================

# idea: c13 reactive pixel policy + flame tracker + trajectory-model air dodge (short hops choose hold / release early so Mario's feet-path avoids incoming Bowser-fire lanes)
import numpy as np

RUN = frozenset({"Right", "x"})
RUNJ = frozenset({"Right", "x", "z"})
HOPJ = frozenset({"x", "z"})
BRAKE = frozenset({"Left"})
NONE = frozenset()
HINTS = [[622, 'P', 80], [1261, 'J', 12]]
FLAME_X0 = 1250   # world x from which the fire spawner is active
SHORT_GAP = 48    # pillar gaps narrower than this get a short (low airtime) jump
SHORT_HOLD = 16
LEAP = False
CLOSE_TRIG = 30
RIGHT_MAX = 206
HOP_FWD_GAP = 45
WAIT_LO = 8
WAIT_HI = 16
WALKJ = frozenset({"Right", "z"})
ARENA_MARGIN = 0   # was 7: hopping into a flame that passes just over Mario's head killed him

# --- measured jump trajectories: feet y after t steps, A held for h frames (run speed), ground = 160
_RAW = {
    1: [160, 160, 154, 148, 143, 139, 135, 133, 131, 130, 130, 131, 132, 135, 138, 142, 147, 152, 157, 160],
    2: [160, 160, 154, 148, 142, 137, 133, 130, 128, 127, 126, 126, 127, 129, 131, 135, 139, 144, 149, 154, 159, 160],
    3: [160, 160, 154, 148, 142, 137, 132, 129, 126, 124, 122, 122, 122, 123, 125, 128, 132, 136, 141, 147, 152, 157, 160],
    4: [160, 160, 154, 148, 142, 137, 132, 127, 124, 121, 119, 118, 118, 119, 120, 122, 125, 129, 134, 139, 145, 152, 157, 160],
    6: [160, 160, 154, 148, 142, 137, 132, 127, 122, 118, 115, 113, 111, 111, 111, 112, 114, 116, 120, 124, 129, 134, 139, 144, 149, 154, 159, 160],
    8: [160, 160, 154, 148, 142, 137, 132, 127, 122, 118, 113, 110, 107, 106, 105, 104, 105, 106, 109, 112, 116, 120, 125, 130, 135, 140, 145],
    10: [160, 160, 154, 148, 142, 137, 132, 127, 122, 118, 113, 109, 106, 103, 100, 99, 98, 99, 100, 102, 104, 108, 112, 117, 122, 127, 132],
    12: [160, 160, 154, 148, 142, 137, 132, 127, 122, 118, 113, 109, 106, 102, 99, 96, 94, 93, 95, 98, 101, 106, 111, 116, 121, 126, 131],
    14: [160, 160, 154, 148, 142, 137, 132, 127, 122, 118, 113, 109, 106, 102, 99, 95, 92, 94, 97, 101, 105, 110, 115, 120, 125, 130, 135],
    16: [160, 160, 154, 148, 142, 137, 132, 127, 122, 118, 113, 109, 106, 102, 99, 95, 92, 94, 96, 99, 102, 107, 112, 119, 124, 129, 134],
}


def _ext(row):
    r = list(row)
    while r[-1] < 160:
        r.append(min(160, r[-1] + 5))
    while len(r) < 64:
        r.append(160)
    return r


TAB = {}
for _h in range(1, 17):
    if _h in _RAW:
        TAB[_h] = _ext(_RAW[_h])
for _h in (5, 7, 9, 11, 13, 15):
    TAB[_h] = [(a + b) // 2 for a, b in zip(TAB[_h - 1], TAB[_h + 1])]


def mario_box(s):
    m = (s[32:, :, 0] == 172) & (s[32:, :, 1] == 124) & (s[32:, :, 2] == 0)
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()) + 32, int(ys.max()) + 32


def fire_comps(A):
    """connected comps (radius 2) of a boolean mask -> list of (xl, xr, yt, yb, n)"""
    ys, xs = np.nonzero(A)
    if len(xs) == 0 or len(xs) > 900:
        return []
    pts = set(zip(ys.tolist(), xs.tolist()))
    out = []
    while pts:
        st = [pts.pop()]
        c = list(st)
        while st:
            y, x = st.pop()
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.remove(q)
                        st.append(q)
                        c.append(q)
        a = np.array(c)
        out.append((int(a[:, 1].min()), int(a[:, 1].max()), int(a[:, 0].min()), int(a[:, 0].max()), len(c)))
    return out


class Agent:
    def __init__(self, H=None):
        self.H = [list(h) for h in (H or [])]   # [wx, kind, param]  kind 'J' hold / 'P' pause
        self.hidx = 0
        self.prev = None
        self.scroll = 0
        self.last_shift = 0
        self.wx = 0
        self.frame = 0
        self.hold = 0
        self.hop = False
        self.cool = 0
        self.pause = 0
        self.committed = False
        self.waited = 0
        self.last_hz = -1000
        self.prev_d = None
        self.nobox = 0
        self.lastscr = None
        self.frozen = 0
        self.dead = False
        self.ring = []
        self.mx = None
        self.fy = None
        self.noprog = 0
        self.pflames = []     # previous-frame flame candidates (world xl, yt)
        self.flames = []      # confirmed flames this frame: (xl, xr, yt, yb)
        self.gate_wait = 0
        self.gate_cool = 0
        self.bwait = 0
        self.pmx = None
        self.hop_hold = 13
        self.pbb = None
        self.dash = False
        self.vsm = 0.0
        self.hop_fwd = True
        self.backing = False
        self.dbg = None
        self.log = []
        # air-dodge state
        self.jk = 0          # steps with jump held so far in this jump
        self.jbase = 160     # feet y at launch
        self.rmin = 99       # shortest total hold allowed for the current jump
        self.dodge_ok = False
        self.ndodge = 0
        self.paf = []
        self.af = []

    # ------------------------------------------------------------------ air dodge
    def hits(self, r, jk, base, flames, x0, x1, vxm):
        """would holding A for r total frames (jump already held jk) put Mario into a flame?"""
        row = TAB[max(1, min(16, r))]
        for (fxl, fxr, fyt, fyb, fvs, fvy) in flames:
            if fxl < x0 - 6:
                continue
            closing = max(0.8, fvs + vxm)
            left = fxl - 8
            t_in = int((left - (x1 + 1)) / closing) - 1
            t_out = int((fxr - (x0 - 1)) / closing) + 2
            vy = max(0, fvy)
            for t in range(max(1, t_in), min(t_out, 70) + 1):
                idx = min(max(jk - 1 + t, 0), len(row) - 1)
                ya = min(base + row[idx] - 160, base)
                fb = min(fyb + vy * t, max(fyb, 152))
                ft = fb - (fyb - fyt)
                if ya >= ft - 1 and ya - 14 <= fb + 1:
                    return True
        return False

    def track_scroll(self, s):
        g = s[40:232].sum(axis=2, dtype=np.int32)
        if self.prev is not None:
            pv = self.prev
            best, bs = None, 0
            for sh in range(0, 5):
                if sh == 0:
                    d = np.count_nonzero(pv != g) / 256.0
                else:
                    d = np.count_nonzero(pv[:, sh:] != g[:, :-sh]) / float(256 - sh)
                key = (round(d, 1), abs(sh - self.last_shift), sh)
                if best is None or key < best:
                    best, bs = key, sh
            self.scroll += bs
            self.last_shift = bs
        self.prev = g

    def detect_flames(self, s, bx):
        """Bowser fire: ~14x4 light-orange (252,160,68) blob that moves horizontally."""
        x0, x1, y0, y1 = bx
        A = (s[:, :, 0] == 252) & (s[:, :, 1] == 160) & (s[:, :, 2] == 68)
        A[:34] = False
        A[205:] = False
        A[max(0, y0 - 2):y1 + 3, max(0, x0 - 2):x1 + 3] = False
        cands = []
        for (xl, xr, yt, yb, n) in fire_comps(A):
            w = xr - xl + 1
            h = yb - yt + 1
            if n >= 24 and 11 <= w <= 20 and 3 <= h <= 8:
                cands.append((xl, xr, yt, yb))
        conf = []
        cur = []
        for (xl, xr, yt, yb) in cands:
            wxl = self.scroll + xl
            cur.append((wxl, yt, xl))
            for (pw, py, pxs) in self.pflames:
                if abs(py - yt) <= 1 and 0.5 <= (pw - wxl) <= 9:
                    conf.append((xl, xr, yt, yb, max(0.5, float(pxs - xl)), yt - py))
                    break
        self.pflames = cur
        self.flames = conf
        return conf

    def arena_flames(self, s, bx, bow):
        """Bowser fire in the arena: light-orange/red blobs in the low lanes, outside Mario/Bowser/platform/axe."""
        x0, x1, y0, y1 = bx
        r = s[:, :, 0]
        g = s[:, :, 1]
        b = s[:, :, 2]
        A = ((r == 252) & (g == 160) & (b == 68)) | ((r == 248) & (g == 56) & (b == 0))
        A[:60] = False
        A[165:] = False
        A[:, 228:] = False
        A[max(0, y0 - 3):y1 + 4, max(0, x0 - 3):x1 + 4] = False
        if bow is not None:
            A[max(0, bow[2] - 12):bow[3] + 6, max(0, bow[0] - 6):min(256, bow[1] + 8)] = False
        out = []
        for (xl, xr, yt, yb, n) in fire_comps(A):
            if n >= 8 and yt >= 106 and (xr - xl) <= 30 and (yb - yt) <= 12:
                out.append((xl, xr, yt, yb))
        return out

    def act(self, s):
        self.frame += 1
        self.track_scroll(s)
        if self.last_shift == 0:
            self.noprog += 1
        else:
            self.noprog = 0
        r = s[:, :, 0]
        g = s[:, :, 1]
        b = s[:, :, 2]
        bx = mario_box(s)
        keys = RUN
        jump = False
        if bx is None:
            self.nobox += 1
            if (self.nobox >= 4 and (self.fy or 0) > 150) or self.nobox >= 70:
                self.dead = True
            return RUN
        self.nobox = 0
        x0, x1, y0, y1 = bx
        self.bx_dbg = bx
        self.mx, self.fy = x1, y1
        self.wx = self.scroll + x1
        if y1 >= 232:
            self.dead = True
        frame = self.frame
        flames = []
        if self.wx >= FLAME_X0:
            flames = self.detect_flames(s, bx)
        nz = (r > 0) | (g > 0) | (b > 0)
        red = (r == 248) & (g == 56) & (b == 0)
        gray = (r == g) & (g == b)
        G = gray & ((r == 124) | (r == 188))
        green = (r == 0) & (g == 168) & (b == 0)
        xr = min(x1 + 1, 248)
        ground = nz.copy()
        ground[205:] &= ~red[205:]
        ground[:34] = False
        below = ground[y1 + 1:y1 + 4, x0:x1 + 1]
        grounded = int(below.sum()) >= 3
        wall = int(G[y0 + 2:y1 - 1, xr + 1:xr + 9].sum()) >= 6
        ya = min(y1 + 2, 236)
        yb = min(y1 + 26, 238)
        colg = ground[ya:yb, :].any(axis=0)
        xs = None
        for x in range(xr, 256):
            if not colg[x]:
                xs = x
                break
        d = None
        lava_gap = False
        xe = 255
        if xs is not None:
            xe = xs
            while xe < 255 and not colg[xe + 1]:
                xe += 1
            d = xs - x1
            if xe - xs >= 10 or xe >= 255:
                lava_gap = int(red[212:238, xs:xe + 1].sum()) >= 8
            else:
                d = None
        gap_edge = grounded and d is not None and d <= 5 and (xe - xs >= 10 or xe >= 255)
        hz_now = False
        if d is not None and lava_gap:
            za = max(0, xs - 8)
            zb = min(255, min(xe, xs + 70) + 8)
            rz = red[34:204, za:zb + 1].copy()
            my0, my1 = max(0, y0 - 3 - 34), y1 + 4 - 34
            mx0, mx1 = x0 - 3 - za, x1 + 4 - za
            if mx1 > 0 and mx0 < rz.shape[1]:
                rz[max(0, my0):max(0, my1), max(0, mx0):max(0, mx1)] = False
            hz_now = int(rz.sum()) >= 4
            if hz_now:
                self.last_hz = frame
        since = frame - self.last_hz
        safe = (not hz_now) and (0 < since <= 12)
        bow = None
        gy, gx = np.nonzero(green[40:210])
        if len(gx) >= 40:
            bow = (int(gx.min()), int(gx.max()), int(gy.min()) + 40, int(gy.max()) + 40)
        bow_jump = False
        if bow is not None:
            gp = bow[0] - x1
            if bow[3] >= y0 + 6 and 0 < gp < 46:
                bow_jump = True
            fy0 = max(34, y0 - 6)
            fy1 = min(204, y1 + 3)
            fx0 = min(255, x1 + 3)
            fx1 = min(255, x1 + 40)
            if int(red[fy0:fy1, fx0:fx1].sum()) >= 6:
                bow_jump = True

        # flame analysis
        threat = False   # flame about to hit Mario's body: hop in place
        gate = False     # flame on screen ahead in the rows Mario will use: hold launches
        vxm = (x1 - self.pmx) if self.pmx is not None else 0
        self.pmx = x1
        self.hop_hold = 13
        fblock = False
        arena = bow is not None and self.wx >= 2060
        self.vsm = 0.6 * self.vsm + 0.4 * vxm
        if arena:
            self.pbb_g = bow[3]
            self.af = self.arena_flames(s, bx, bow)
            for (fxl, fxr, fyt, fyb) in self.af:
                if fxr < x0 - 4:
                    continue
                d = fxl - x1
                cl = 1.5 + max(self.vsm, 0.0)
                ta = d / cl
                # vertical drift of this flame (match with last frame's blobs)
                fvy_a = 0
                for (pxl, pxr, pyt, pyb_) in self.paf:
                    if 0 <= pxl - fxl <= 8 and abs(pyb_ - fyb) <= 3:
                        fvy_a = fyb - pyb_
                        break
                ahead = max(ta, 0.0) + (6.0 if fvy_a > 0 else 0.0)
                wb = fyb + fvy_a * ahead
                wt = fyt + fvy_a * max(ta, 0.0)
                if wb + ARENA_MARGIN >= y0 - 3 and wt <= y1 + 2 and ta <= 7.5 and d > -16:
                    threat = True
                    self.hop_hold = 30
                if d < 140 and fyb + 40 >= y0 - 3:
                    fblock = True
            self.paf = list(self.af)
            flames = []
        else:
            self.paf = []
        for (fxl, fxr, fyt, fyb, fvs, fvy) in flames:
            if fxl < x0 - 6:
                continue
            dx = (fxl - 8) - x1          # flame sprite tip is ~8px left of its light-orange core
            closing = max(0.8, fvs + vxm)
            ta = dx / closing
            vy = max(0, fvy)
            pyb = min(fyb + vy * max(ta, 0.0), 152)      # flames sink to the ground-level lane
            pyt = pyb - (fyb - fyt)
            if arena:
                if pyb >= y0 - 3 and pyt <= y1 + 2 and ta <= 11:
                    threat = True
                    self.hop_hold = 30
            elif fyb >= y0 - 3 and fyt <= y1 + 2 and ta <= 11:
                threat = True
                fl = max(ta, 0.0) + 35.0 / closing + 5.0
                self.hop_hold = int(min(24, max(10, fl / 2.0)))
            fyb_fin = min(fyb + vy * 40, 152)
            if dx < 130 and fyb_fin >= y0 - 3 and fyt <= y1 + 2 + 40:
                fblock = True
            if dx <= 200 and 90 <= fyt <= 215:
                gate = True
        self.dbg = (fblock, threat, arena, self.hop_hold, self.dash, round(self.vsm, 1))
        if self.gate_cool > 0:
            self.gate_cool -= 1

        # ---- in-air dodge: shorten the held jump if the flame path would hit Mario
        if self.hold > 0 and self.dodge_ok and flames and not arena:
            plan = self.jk + self.hold
            if self.hits(plan, self.jk, self.jbase, flames, x0, x1, vxm):
                lo = max(self.rmin, self.jk)
                for rr in range(plan - 1, lo - 1, -1):
                    if not self.hits(rr, self.jk, self.jbase, flames, x0, x1, vxm):
                        self.hold = rr - self.jk
                        self.ndodge += 1
                        self.log.append(("dodge", frame, self.wx, y1, plan, rr, self.jk))
                        if self.hold == 0:
                            self.cool = 3
                            self.dodge_ok = False
                        break

        braking = False
        if self.hold > 0:
            jump = True
            self.hold -= 1
            self.jk += 1
            if self.hold == 0:
                self.cool = 3
                self.dodge_ok = False
        elif self.cool > 0:
            self.cool -= 1
        elif self.pause > 0:
            self.pause -= 1
            keys = NONE
        elif grounded:
            self.dodge_ok = False
            if threat and self.gate_cool == 0 and not self.dash:
                # fire coming at body height: hop straight up and let it pass underneath
                jump = True
                self.hold = self.hop_hold
                self.hop = True
                self.hop_fwd = (bow is None) or (bow[0] - x1 >= HOP_FWD_GAP)
                self.gate_cool = 14
                self.committed = False
                self.log.append(("hop", frame, self.wx, y1, flames[:2]))
            elif (bow is not None and self.wx >= 2050 and bow[1] >= x1 - 2 and self.bwait < 700
                  and bow[1] - bow[0] <= 40):
                # Bowser: keep away while he stands on the bridge, dash underneath while he is in the air
                self.hop = False
                bl = bow[0] - 12
                br = bow[1] + 4
                bb = bow[3]
                gp = bl - x1
                rising = self.pbb is not None and bb < self.pbb - 0.5 and bb < 153
                high = bb <= y0 - 8
                lure = False
                bgrd = bb >= 154 and self.pbb is not None and abs(bb - self.pbb) < 0.6
                gpl = bow[0] - x1
                if LEAP and arena and bgrd and not self.dash and not self.hop:
                    # Bowser stands on the bridge: build speed, then leap over him
                    if self.backing and (gpl >= 100 or x1 < 45):
                        self.backing = False
                    if (not self.backing) and x1 >= 45 and (gpl < 30 or (gpl < 70 and self.vsm < 1.8)):
                        self.backing = True
                    if (not self.backing) and self.vsm >= 2.2 and 30 <= gpl <= 46:
                        jump = True
                        self.hold = 30
                        self.log.append(("leap", frame, x1, bb, gpl, round(self.vsm, 1)))
                    elif self.backing:
                        keys = BRAKE
                    else:
                        keys = RUN
                    braking = True
                    self.pbb = bb
                    self.log.append(("bow", frame, x1, bb, gpl, keys == RUN)) if frame % 10 == 0 else None
                    self.prev_d = d
                    if jump:
                        keys = RUNJ
                    return keys
                if not self.dash and ((rising and bb > 138 and gp <= CLOSE_TRIG and not fblock and bow[1] <= RIGHT_MAX) or (x1 >= bl and x1 < 45)):
                    self.dash = True
                    self.log.append(("dash", frame, x1, bb, gp))
                if self.dash and x1 < bl - 10 and self.pbb is not None and bb > self.pbb + 0.5:
                    self.dash = False   # Bowser is coming down before we are under him: back off
                    self.log.append(("abort", frame, x1, bb, gp))
                if self.dash:
                    if wall and x0 >= br - 6:
                        jump = True
                        self.hold = 30
                        self.dash = False
                    else:
                        keys = RUN
                else:
                    self.bwait += 1
                    if lure:
                        # Bowser hugs the axe wall: back off so he follows us into open space
                        keys = BRAKE if gp < 66 else (NONE if gp < 80 else RUN)
                    else:
                        vv = max(self.vsm, 0.0)
                        stop_gap = gp - vv * vv / 0.30
                        coast_gap = gp - vv * vv / 0.12
                        if stop_gap < WAIT_LO:
                            keys = BRAKE if (vv > 0.3 or gp < WAIT_LO) else NONE
                        elif coast_gap > WAIT_HI:
                            keys = RUN if gp > 70 else frozenset({"Right"})
                        else:
                            keys = NONE
                braking = True
                self.pbb = bb
                self.log.append(("bow", frame, x1, bb, gp, keys == RUN)) if frame % 10 == 0 else None
            else:
                self.hop = False
                self.dash = False
                hinted = False
                while self.hidx < len(self.H) and self.H[self.hidx][0] < self.wx - 30:
                    self.hidx += 1
                if self.hidx < len(self.H) and self.wx >= self.H[self.hidx][0] and not (gate and self.H[self.hidx][1] == 'J' and self.gate_wait < 150):
                    h = self.H[self.hidx]
                    self.hidx += 1
                    hinted = True
                    if h[1] == 'J':
                        jump = True
                        self.hold = int(h[2])
                        self.committed = False
                        self.waited = 0
                        self.jk = 1
                        self.jbase = y1
                        self.rmin = 99
                        self.dodge_ok = False
                    else:
                        self.pause = int(h[2])
                        keys = NONE
                        self.pause -= 1
                if not hinted:
                    gated = False
                    if gate and self.gate_wait < 500:
                        # slow down before gaps/walls while fire is ahead; wait on the edge
                        near_gap = d is not None and d <= 70 and (xe - xs >= 10 or xe >= 255)
                        if gap_edge or wall:
                            gated = True
                            keys = NONE
                        elif near_gap:
                            gated = True
                            speed = (self.prev_d - d) if self.prev_d is not None else 0
                            keys = BRAKE if (speed > 0.6 and d <= 40) else NONE
                        if gated:
                            self.gate_wait += 1
                            braking = True
                    if not gated:
                        if lava_gap and d is not None and not self.committed and 24 <= d <= 62:
                            speed = (self.prev_d - d) if self.prev_d is not None else 0
                            if safe or self.waited > 350:
                                self.committed = True
                            else:
                                braking = True
                                self.waited += 1
                                keys = BRAKE if speed > 0.6 else NONE
                        if not braking and (wall or gap_edge or bow_jump):
                            jump = True
                            self.hold = 30
                            self.jk = 1
                            self.jbase = y1
                            self.rmin = 99
                            self.dodge_ok = False
                            if (self.wx >= 1650 and gap_edge and not wall and not bow_jump and bow is None
                                    and xe < 255 and xe - xs < SHORT_GAP):
                                self.hold = SHORT_HOLD
                                self.rmin = 3
                                self.dodge_ok = True
                                if flames:
                                    # choose the hold length whose flight path avoids every visible flame
                                    for rr in (17, 12, 8, 6, 4, 3):
                                        if not self.hits(rr, 0, y1, flames, x0, x1, vxm):
                                            self.hold = rr - 1
                                            break
                                    self.log.append(("hopsel", frame, self.wx, y1, self.hold + 1))
                            self.committed = False
                            self.waited = 0
                            self.gate_wait = 0
        self.prev_d = d
        if jump:
            if self.hop:
                keys = WALKJ if (arena and self.hop_fwd) else HOPJ
            else:
                keys = RUNJ
        return keys

    def step(self, sim):
        s = sim.screen()
        if self.lastscr is not None and np.array_equal(s, self.lastscr):
            self.frozen += 1
        else:
            self.frozen = 0
        self.lastscr = s
        self.ring.append(s)
        if len(self.ring) > 24:
            self.ring.pop(0)
        keys = self.act(s)
        sim.step(keys)
        return s


def play(sim):
    ag = Agent(HINTS)
    n = 0
    told = False
    while sim.budget_left > 0 and n < 7950:
        ag.step(sim)
        n += 1
        if ag.dead and not told:
            told = True
            print("DEAD f", n, "wx", ag.wx, "fy", ag.fy, "log", ag.log[-8:], flush=True)
        if ag.frozen > 60:
            print("play end f", n, "wx", ag.wx, "log", ag.log[-6:], flush=True)
            return
        if n % 100 == 0:
            print("play f", n, "wx", ag.wx, "mx", ag.mx, "fy", ag.fy, "hops", len(ag.log), "dodges", ag.ndodge, flush=True)


def solve(sim):
    # nothing to learn offline: the policy is purely reactive
    return []

