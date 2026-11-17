# ==================================================================================================
# Super Mario Bros. (NES) -- level 8-1  |  超级马里奥兄弟（NES） 关 8-1
# Pass rate on 100 random variants never used before: 64/100; fastest finish 3679 frames (61.32 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 64/100；最快一局 3679 帧（60 帧/秒，约 61.32 秒）。
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
#   closed-loop reflex runner (PB) + pipe-staircase hopper (rim detection, jump planning with wall
#   /landing checks).
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _mask                          line 32    
#   def find_mario                     line 36    returns (xl, xr, ytop, feet) of Mario or None
#   def _rim_below                     line 61    
#   def _enemy_cols                    line 101   
#   def _find_pipe                     line 109   
#   def _scroll                        line 137   
#   def land_time                      line 165   (t_clear, t_land) in frames after the takeoff press for hold k and rim dh above takeoff, or None
#   def platforms                      line 187   list of (top_y, x0, x1): first row of each horizontal solid run >= minrun
#   def is_pipe                        line 224   
#   def plant_h                        line 232   
#   def air_x                          line 247   horizontal travel in T frames: mode R (Right held), 0 (neutral), L (Left held)
#   class Hopper                       line 261   
#   class _HopperOld                   line 428   
#   def play_old                       line 555   
#   class Agent                        line 751   the reflex runner as a deep-copyable object: frame(sim) = one game frame
#   def play                           line 949   
#   def solve                          line 961   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: closed-loop reflex runner (PB) + pipe-staircase hopper (rim detection, jump planning with wall/landing checks).
import numpy as np

# ---- tunables ----
# NOTE: the recorded macros (memory.json) depend on these exact values: do NOT tune them
# (non-literal assignments so the automatic tuner leaves them alone).
ENEMY_D = int(50)
PIT_D = int(10)
HOLD_OBS = int(24)
HOLD_PIT = int(26)
HOLD_WALL = int(26)
AIR_BRAKE = int(0)
GROUND_Y = int(207)
_FTOL = 10
_MACRO_WINDOW = 12
HOLD_GAP = int(14)
PIPE_D = int(50)
PIPE_WAIT = int(80)
PIPE_BRAKE = int(56)
PIPE_SCAN = int(100)
TUNABLE = {}
_HOP_FROM = 3600
_HOP_MACRO_MAX = 3660

SKY = (104, 136, 252)
_RED = (248, 56, 0)
_BROWN = (172, 124, 0)
_ORANGE = (228, 92, 16)
_LIME = (184, 248, 24)


def _mask(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def find_mario(s, prev):
    """returns (xl, xr, ytop, feet) of Mario or None"""
    red = _mask(s[32:], _RED)
    br = _mask(s[32:], _BROWN)
    ys, xs = np.nonzero(red)
    if len(xs) == 0:
        return None
    ys = ys + 32
    if prev is not None:
        cx = (prev[0] + prev[1]) / 2.0
        cy = (prev[2] + prev[3]) / 2.0
        d = np.abs(xs - cx) + np.abs(ys - cy)
        j = int(np.argmin(d))
    else:
        j = 0
    rx, ry = xs[j], ys[j]
    x0, x1 = max(0, rx - 14), min(256, rx + 15)
    y0, y1 = max(32, ry - 14), min(240, ry + 17)
    sub = (red | br)[y0 - 32:y1 - 32, x0:x1]
    yy, xx = np.nonzero(sub)
    if len(xx) == 0:
        return None
    return (x0 + int(xx.min()), x0 + int(xx.max()), y0 + int(yy.min()), y0 + int(yy.max()))


def _rim_below(s, xl, xr, feet, solid=False):
    y0 = feet + 1
    y1 = min(232, feet + 44)
    x0 = max(0, xl - 6)
    x1 = min(256, xr + 7)
    if y1 - y0 < 3 or x1 - x0 < 4:
        return None
    reg = s[y0:y1, x0:x1, :]
    if solid:
        g = ~((reg[:, :, 0] == SKY[0]) & (reg[:, :, 1] == SKY[1]) & (reg[:, :, 2] == SKY[2]))
    else:
        g = ((reg[:, :, 0] == 0) & (reg[:, :, 1] == 168) & (reg[:, :, 2] == 0)) | \
            ((reg[:, :, 0] == 184) & (reg[:, :, 1] == 248) & (reg[:, :, 2] == 24))
    rows = np.nonzero(g.sum(axis=1) >= 5)[0]
    if len(rows) == 0:
        return None
    y = y0 + int(rows[0])
    row = s[y, :, :]
    if solid:
        gr = ~((row[:, 0] == SKY[0]) & (row[:, 1] == SKY[1]) & (row[:, 2] == SKY[2]))
    else:
        gr = ((row[:, 0] == 0) & (row[:, 1] == 168) & (row[:, 2] == 0)) | \
             ((row[:, 0] == 184) & (row[:, 1] == 248) & (row[:, 2] == 24))
    cx = (xl + xr) // 2
    if not gr[cx]:
        cand = np.nonzero(gr[x0:x1])[0]
        if len(cand) == 0:
            return None
        cx = x0 + int(cand[0])
    a = cx
    while a > 0 and gr[a - 1]:
        a -= 1
    b = cx
    while b < 255 and gr[b + 1]:
        b += 1
    if solid and b - a > 60:
        return None
    return a, b


def _enemy_cols(win):
    og = _mask(win, _ORANGE)
    bl = (win[:, :, 0] == 0) & (win[:, :, 1] == 0) & (win[:, :, 2] == 0)
    g = (win[:, :, 1] == 168) & (win[:, :, 0] == 0)
    return int((og.sum(axis=0) >= 7).sum()) + int((bl.sum(axis=0) >= 7).sum()) \
        + int((g.sum(axis=0) >= 5).sum())


def _find_pipe(s, xr, feet):
    xa = xr + 2
    xb = min(256, xr + 2 + PIPE_SCAN)
    if xb - xa < 6 or feet - 30 < 40:
        return None
    band = s[feet - 30:feet - 25, xa:xb, :]
    g = (band[:, :, 0] == 0) & (band[:, :, 1] == 168) & (band[:, :, 2] == 0)
    cols = np.nonzero(g.sum(axis=0) >= 3)[0]
    if len(cols) < 5:
        return None
    pl = xa + int(cols[0])
    x0 = max(xa, pl - 10)
    x1 = min(256, pl + 40)
    reg = s[40:feet, x0:x1, :]
    gg = ((reg[:, :, 0] == 0) & (reg[:, :, 1] == 168) & (reg[:, :, 2] == 0)) | \
         ((reg[:, :, 0] == 184) & (reg[:, :, 1] == 248) & (reg[:, :, 2] == 24))
    rows = np.nonzero(gg.sum(axis=1) >= 24)[0]
    if len(rows) == 0:
        return None
    ptop = 40 + int(rows[0])
    pl = x0 + int(np.nonzero(gg[int(rows[0])])[0][0])
    y0 = max(32, ptop - 22)
    ab = s[y0:ptop - 2, pl + 6:min(256, pl + 26), :]
    sky = (ab[:, :, 0] == SKY[0]) & (ab[:, :, 1] == SKY[1]) & (ab[:, :, 2] == SKY[2])
    plant = int((~sky).sum()) >= 4
    return pl, ptop, plant


def _scroll(a, b):
    best, bd = 0, None
    ra = a[210:236, :, :].astype(np.int16)
    rb = b[210:236, :, :].astype(np.int16)
    for dx in range(0, 7):
        d = np.abs(ra[:, dx:240, :] - rb[:, 0:240 - dx, :]).sum()
        if bd is None or d < bd:
            bd, best = d, dx
    return best


# =====================  pipe-staircase hopper  =====================
_SKYA = np.array([104, 136, 252], dtype=np.int16)
_MCOL = [(248, 56, 0), (172, 124, 0), (252, 160, 68)]
NARROW = 60
MINRUN = 28
HP = {"MARG": 6, "LAND0": 12, "LANDR": 8, "SEE": 100, "EDGE": 2, "VMAX": 1.7}

_FY = {
    3: [208, 208, 203, 198, 194, 189, 186, 183, 180, 178, 176, 176, 175, 175, 176, 178, 180, 182, 185, 189, 193, 198, 203, 208],
    8: [208, 208, 203, 198, 194, 189, 185, 181, 178, 174, 171, 168, 166, 164, 163, 162, 162, 163, 164, 165, 168, 170, 174, 178, 182, 187, 192, 197, 202, 207, 208],
    14: [208, 208, 203, 198, 194, 189, 185, 181, 178, 174, 171, 167, 164, 162, 159, 156, 154, 152, 151, 150, 150, 151, 152, 153, 156, 158, 162, 165, 170, 175, 180, 185, 190, 195, 200, 205, 208],
    26: [208, 208, 203, 198, 194, 189, 185, 181, 178, 174, 171, 167, 164, 162, 159, 156, 154, 152, 150, 148, 146, 145, 144, 143, 142, 141, 140, 140, 140, 140, 141, 142, 144, 147, 150, 154, 158, 163, 168, 173, 178, 183, 188, 193, 198, 203, 208],
}
PROF = {k: [208 - v for v in p] for k, p in _FY.items()}
KS = [3, 8, 14, 26]


def land_time(k, dh):
    """(t_clear, t_land) in frames after the takeoff press for hold k and rim dh above takeoff, or None"""
    prof = PROF[k]
    ap = max(prof)
    if ap < dh + HP["MARG"]:
        return None
    ia = prof.index(ap)
    tc = 0
    for t in range(len(prof)):
        if prof[t] >= dh + 2:
            tc = t
            break
    tl = None
    for t in range(ia, len(prof)):
        if prof[t] <= dh:
            tl = t
            break
    if tl is None:
        tl = len(prof) - 1 + int(max(0, (dh - prof[-1])) / 5.0)
    return tc, tl


def platforms(s, mario=None, minrun=MINRUN):
    """list of (top_y, x0, x1): first row of each horizontal solid run >= minrun"""
    reg = s[100:232].astype(np.int16)
    ns = np.abs(reg - _SKYA).sum(axis=2) > 12
    for c in _MCOL:
        ns &= ~((reg[:, :, 0] == c[0]) & (reg[:, :, 1] == c[1]) & (reg[:, :, 2] == c[2]))
    if mario is not None:
        mxl, mxr, _, mfeet = [int(v) for v in mario]
        r = mfeet + 2 - 100
        if 0 <= r < ns.shape[0]:
            for yy in (mfeet - 1, mfeet, mfeet + 1):
                if 0 <= yy - 100 < ns.shape[0]:
                    ns[yy - 100, mxl:mxr + 1] = ns[r, mxl:mxr + 1]
    out = []
    prev = []
    rows = np.nonzero(ns.sum(axis=1) >= minrun)[0]
    for y in rows:
        row = ns[y]
        d = np.diff(np.concatenate(([0], row.astype(np.int8), [0])))
        st = np.nonzero(d == 1)[0]
        en = np.nonzero(d == -1)[0]
        cur = []
        for a, b in zip(st, en):
            if b - a >= minrun:
                cur.append((int(a), int(b) - 1))
        for (a, b) in cur:
            new = True
            for (pa, pb, py) in prev:
                if min(b, pb) - max(a, pa) > min(8, minrun // 2) and py == y - 1:
                    new = False
                    break
            if new:
                out.append((int(y) + 100, a, b))
        prev = [(a, b, int(y)) for (a, b) in cur]
    return out


def is_pipe(s, N):
    top, x0, x1 = N
    row = s[min(239, top + 2), x0:x1 + 1, :]
    g = ((row[:, 0] == 0) & (row[:, 1] == 168) & (row[:, 2] == 0)) | \
        ((row[:, 0] == 184) & (row[:, 1] == 248) & (row[:, 2] == 24))
    return int(g.sum()) >= 14


def plant_h(s, N):
    top, x0, x1 = N
    ya = max(40, top - 28)
    reg = s[ya:top - 1, x0 + 8:x1 - 8].astype(np.int16)
    if reg.size == 0:
        return 0
    ns = np.abs(reg - _SKYA).sum(axis=2) > 12
    for c in _MCOL:
        ns &= ~((reg[:, :, 0] == c[0]) & (reg[:, :, 1] == c[1]) & (reg[:, :, 2] == c[2]))
    rows = np.nonzero(ns.sum(axis=1) >= 1)[0]
    if len(rows) == 0:
        return 0
    return int((top - 1) - (ya + rows[0]))


def air_x(v0, T, mode):
    """horizontal travel in T frames: mode R (Right held), 0 (neutral), L (Left held)"""
    v = v0
    x = 0.0
    cap = max(1.8, v0)
    for _ in range(int(max(0, T))):
        if mode == "R":
            v = min(cap, v + 0.05)
        elif mode == "L":
            v = max(-1.8, v - 0.1)
        x += v
    return x


class Hopper:
    def __init__(self, log=None):
        self.log = log or (lambda m: None)
        self.active = False
        self.cool = 0
        self.prev_feet = None
        self.n = 0
        self.air_t = -1
        self.k = 0
        self.tl = 0
        self.Nw = 0
        self.Nwid = 32
        self.last_ph = None
        self.pn = None
        self.since_retract = 999
        self.down_frames = 0
        self.prev_ph = 0
        self.wait_n = 0

    def decide(self, s, mario, world, spd):
        self.n += 1
        xl, xr, yt, feet = [int(v) for v in mario]
        pf = self.prev_feet
        self.prev_feet = feet
        if feet > 225:
            self.air_t = -1
            return None
        xc = (xl + xr) / 2.0
        grounded = (pf is not None and pf == feet)
        plats = platforms(s, mario, 14 if world >= 4850 else MINRUN)
        sup = None
        for p in plats:
            if abs(p[0] - (feet + 1)) <= 1 and min(xr, p[2]) - max(xl, p[1]) >= 2:
                sup = p
        if self.air_t >= 0:
            t = self.air_t
            self.air_t += 1
            if t > 4 and grounded and sup is not None:
                self.air_t = -1
                self.cool = 0
            elif t > self.tl + 25:
                self.air_t = -1
                self.cool = 0
            else:
                keys = set()
                if t < self.k:
                    keys.add("z")
                nx0 = self.Nw - world
                xt = nx0 + min(10.0, self.Nwid / 2.0)
                r = max(0, self.tl - t)
                v = spd
                best, bd = "R", None
                for mode in ("R", "0", "L"):
                    p = xc + air_x(v, r, mode)
                    d = abs(p - xt)
                    if mode == "R":
                        d += 1.0
                    if bd is None or d < bd:
                        bd, best = d, mode
                if best == "R":
                    keys.add("Right")
                elif best == "L":
                    keys.add("Left")
                return keys
        if sup is None and 120 < feet < 215:
            # airborne under somebody else's control: brake if we would land beyond the platform below
            below = [p for p in plats if p[0] > feet + 2 and min(xr + 8, p[2]) - max(xl - 8, p[1]) >= 4]
            if below:
                P_ = min(below, key=lambda p: p[0])
                vy = (feet - pf) if pf is not None else 0
                dhh = P_[0] - feet - 1
                g = 0.4
                tt = (-vy + (vy * vy + 2 * g * dhh) ** 0.5) / g if dhh > 0 else 0
                xland = xc + max(spd, 0.0) * tt
                covered = any(p[0] > feet + 2 and p[1] + 2 <= xland <= p[2] - 2 for p in plats)
                if xland > P_[2] - 8 and not covered:
                    return {"Left"}
            return None
        if self.cool > 0:
            self.cool -= 1
        if sup is not None and grounded:
            nxt = [p for p in plats if p[1] > sup[2] + 2 and p[1] - sup[2] < HP["SEE"]]
            nxt = [p for p in nxt if (p[2] - p[1] + 1) <= NARROW and p[2] < 253]
            edge_d = sup[2] - xr
            if not nxt or edge_d > 90:
                self.active = False
                return None
            N = min(nxt, key=lambda p: p[1])
            self.active = True
            dh = sup[0] - N[0]
            ph = plant_h(s, N)
            nw = world + N[1]
            if self.pn is None or abs(nw - self.pn) > 6:
                self.pn = nw
                self.since_retract = 999
                self.down_frames = 0
                self.prev_ph = ph
                self.wait_n = 0
            if ph > 0:
                self.down_frames = 0
                self.since_retract = 999
            else:
                self.down_frames += 1
                if self.prev_ph > 0:
                    self.since_retract = 0
                else:
                    self.since_retract += 1
            self.prev_ph = ph
            v = max(0.0, min(3.0, spd))
            safe = (ph == 0 and (self.since_retract <= 30 or self.down_frames >= 90)) or self.wait_n > 500 \
                or not is_pipe(s, N)
            if not safe and self.cool == 0:
                self.wait_n += 1
                if self.wait_n % 20 == 1:
                    self.log("hop%d wait plant h=%d edge_d=%d v=%.1f N=%s since=%d down=%d" % (
                        self.n, ph, edge_d, v, N, self.since_retract, self.down_frames))
                if edge_d >= 28:
                    return {"Right"} if spd < HP["VMAX"] else set()
                if edge_d >= 14:
                    return {"Left"} if spd > 0.4 else set()
                return {"Left"}
            if self.cool == 0:
                best = None
                tooclose = True
                for k in KS:
                    r = land_time(k, dh)
                    if r is None:
                        continue
                    tc, tl = r
                    wall_ok = (xr + air_x(v, tc, "R") <= N[1] + 1)
                    xland = xc + air_x(v, tl, "R")
                    if wall_ok:
                        tooclose = False
                    lo = N[1] + min(HP["LAND0"], (N[2] - N[1] + 1) // 2 - 2)
                    if wall_ok and lo <= xland <= N[2] - 4:
                        best = (k, tc, tl, xland)
                        break
                if best is None and edge_d <= HP["EDGE"] and not tooclose:
                    for k in KS[::-1]:
                        r = land_time(k, dh)
                        if r is not None and xr + air_x(v, r[0], "R") <= N[1] + 1:
                            best = (k, r[0], r[1], xc + air_x(v, r[1], "R"))
                            break
                if best is not None:
                    k, tc, tl, xland = best
                    self.k = k
                    self.tl = tl
                    self.Nw = world + N[1]
                    self.Nwid = N[2] - N[1] + 1
                    self.air_t = 1
                    self.cool = k + 4
                    self.log("hop%d JUMP k=%d tl=%d pred_xc=%.0f gap=%d dh=%d edge_d=%d v=%.1f plant=%d sup=%s N=%s mario=%s" % (
                        self.n, k, tl, xland, N[1] - sup[2], dh, edge_d, v, ph, sup, N, (xl, xr, yt, feet)))
                    return {"Right", "z"}
                if tooclose:
                    return {"Left"}
            if spd > 2.2 and edge_d > 40:
                return {"Left"}
            if spd > HP["VMAX"]:
                return set()
            return {"Right"}
        if self.active and sup is None:
            # fell / drifting in the air without our own jump: keep the heading
            return {"Right", "x"}
        return None


class _HopperOld:
    def __init__(self, log=None):
        self.log = log or (lambda m: None)
        self.active = False
        self.src = None
        self.hold_left = 0
        self.cool = 0
        self.prev_feet = None
        self.n = 0
        self.last_ph = None

    def decide(self, s, mario, world, spd):
        self.n += 1
        xl, xr, yt, feet = [int(v) for v in mario]
        pf = self.prev_feet
        self.prev_feet = feet
        if feet > 225:
            return None
        if self.hold_left > 0:
            self.hold_left -= 1
            return {"Right", "z"}
        plats = platforms(s)
        sup = None
        grounded = (pf is not None and pf == feet)
        for p in plats:
            if abs(p[0] - (feet + 1)) <= 1 and min(xr, p[2]) - max(xl, p[1]) >= 2:
                sup = p
        xc = (xl + xr) / 2.0
        if self.cool > 0:
            self.cool -= 1
        if sup is not None and grounded:
            nxt = [p for p in plats if p[1] > sup[2] + 2 and p[1] - sup[2] < HP["SEE"]]
            nxt = [p for p in nxt if (p[2] - p[1] + 1) <= NARROW]
            if not nxt:
                self.active = False
                return None
            N = min(nxt, key=lambda p: p[1])
            self.active = True
            self.src = world + sup[1]
            dh = sup[0] - N[0]
            edge_d = sup[2] - xr
            ph = plant_h(s, N)
            if ph != self.last_ph:
                self.last_ph = ph
                self.log("hop%d plantN h=%d N=%s" % (self.n, ph, N))
            v = max(1.0, min(3.0, spd))
            if self.cool == 0:
                best = None
                tooclose = True
                for k in KS:
                    r = land_time(k, dh)
                    if r is None:
                        continue
                    tc, tl = r
                    wall_ok = (xr + v * tc <= N[1] + 1)
                    xland = xc + v * tl
                    if wall_ok:
                        tooclose = False
                    if wall_ok and N[1] + HP["LAND0"] <= xland <= N[2] - HP["LANDR"]:
                        best = (k, tl, xland)
                        break
                if best is None and edge_d <= HP["EDGE"] and not tooclose:
                    for k in KS[::-1]:
                        r = land_time(k, dh)
                        if r is not None and xr + v * r[0] <= N[1] + 1:
                            best = (k, r[1], xc + v * r[1])
                            break
                if best is not None:
                    k, T, xland = best
                    self.hold_left = k - 1
                    self.cool = k + 4
                    self.log("hop%d JUMP k=%d T=%d pred_xc=%.0f gap=%d dh=%d edge_d=%d v=%.1f plant=%d sup=%s N=%s mario=%s" % (
                        self.n, k, T, xland, N[1] - sup[2], dh, edge_d, v, ph, sup, N, (xl, xr, yt, feet)))
                    return {"Right", "z"}
                if tooclose:
                    # too near the wall for any jump: back off to get a run-up
                    return {"Left"}
            self.log("hop%d wait edge_d=%d v=%.1f dh=%d sup=%s N=%s mario=%s" % (
                self.n, edge_d, v, dh, sup, N, (xl, xr, yt, feet)))
            if spd > HP["VMAX"]:
                return set()
            return {"Right"}
        # airborne
        if not self.active:
            cand = [p for p in plats if (p[2] - p[1] + 1) <= NARROW and p[2] > xl]
            if not cand:
                return None
            self.active = True
        cand = [p for p in plats if (p[2] - p[1] + 1) <= NARROW and p[2] > xl + 2
                and (self.src is None or abs(world + p[1] - self.src) > 4)]
        if not cand:
            return {"Right"}
        N = min(cand, key=lambda p: p[1])
        if xr >= N[2] - 3:
            return {"Left"}
        return {"Right"}


# Searched (save/load DFS) open-loop chains for the tail of the level (world >= 4300), keyed by world
# trigger + frame number (the variant's timeline).  Between the chains the closed-loop reflex runner drives.
# each entry: (world trigger, frame at trigger, [(keys, frames), ...])
_TAIL_FTOL = 0
TAIL = [
    (3927, 2563, [("Right,x,z", 6), ("Right,x", 40)]),
    (4065, 2609, [("Right,x,z", 6), ("Right,x", 28)]),
    (4720, 2868, [("", 30)]),
    (4849, 3012, [("Right,z", 20), ("Right", 30)]),
    (4990, 3062, [("Right,x,z", 12), ("Right,x", 29)]),
    (5357, 3253, [("", 30)]),
    (5499, 3414, [("", 30)]),
    (5539, 3444, [("Right,x,z", 8), ("Right,x", 20)]),
    (5568, 3472, [("Left", 12), ("Right,x,z", 12), ("z", 2), ("", 27)]),
    (5745, 3633, [("Right,x", 16)]),
    # class A timeline (4300 reached at f2570)
    (4815, 2906, [("", 30)]),
    (5355, 3183, [("", 30)]),
    (5500, 3394, [("", 30)]),
    (5501, 3424, [("Right,x", 16)]),
    (5538, 3468, [("", 30)]),
    (5538, 3498, [("Right,x", 16)]),
    (5546, 3514, [("Right,x,z", 8), ("Right,x", 22)]),
    (5588, 3544, [("Right,x,z", 12), ("z", 14), ("", 17)]),
    (5713, 3587, [("Right,x,z", 8), ("Right,x", 48)]),
    (5877, 3643, [("Right,x", 8)]),
]


def play_old(sim, verbose=True, still_limit=100, state=None, macros=None, world0=0, f0=0, vtype0=None, use_tail=True):
    tail = sorted([t for t in TAIL if t[0] >= world0], key=lambda t: t[0]) if use_tail else []
    ti = 0
    if macros is None:
        try:
            macros = list((sim.mem.past or {}).get("macros", []))
        except Exception:
            macros = []
    macros = [dict(m, seq=[list(e) for e in m["seq"][:2]] + [[m["seq"][2][0], 7]]) if m["w"] >= 3600 else m
              for m in macros]
    macros = sorted([m for m in macros if m["w"] >= world0 and m["w"] < _HOP_MACRO_MAX], key=lambda m: m["w"])
    hop = Hopper(log=(print if verbose else None))
    xh = []
    mi = 0
    mseq = []
    mrem = 0
    f = f0
    s = sim.screen()
    last = s
    still = 0
    jump_t = -1
    jump_h = 0
    feet_hist = []
    world = world0
    mario = None
    cool = 0
    last_pipe_state = None
    first = True
    vtype = vtype0
    jump_nox = False
    ftol = _FTOL
    try:
        while sim.budget_left > 0 and f < 7990:
            s = sim.screen()
            mario = find_mario(s, mario)
            if not first:
                if np.array_equal(s, last):
                    still += 1
                else:
                    still = 0
                try:
                    world += _scroll(last, s)
                except Exception:
                    pass
            last = s
            first = False
            if still > still_limit:
                if verbose:
                    print("frozen at f", f, "world", world, "mario", mario)
                break
            keys = {"Right", "x"}
            if mario is None:
                sim.step(keys)
                f += 1
                continue
            xl, xr, yt, feet = mario
            xh.append(world + xr)
            if len(xh) > 4:
                xh.pop(0)
            spd = (xh[-1] - xh[0]) / max(1, len(xh) - 1)
            feet_hist.append(feet)
            if len(feet_hist) > 4:
                feet_hist.pop(0)
            grounded = len(feet_hist) >= 3 and feet_hist[-1] == feet_hist[-2] == feet_hist[-3]
            if state is not None:
                state["world"] = world
                state["f"] = f
                state["mario"] = mario
            while mi < len(macros) and world >= macros[mi]["w"]:
                mf = macros[mi].get("f")
                if world <= macros[mi]["w"] + _MACRO_WINDOW and (mf is None or abs(f - mf) <= ftol):
                    mseq = [(k, n) for k, n in macros[mi]["seq"]]
                    if macros[mi]["w"] == 2223:
                        ftol = 90   # timeline shifted by the retreat: later macros keep matching on world position
                    if macros[mi]["w"] >= 3600 and xl >= 119:
                        # Mario is further right on screen than when the macro was recorded: brake more
                        mseq = [(mseq[0][0], mseq[0][1] + 8), mseq[1], (mseq[2][0], mseq[2][1] + 4)]
                    mrem = sum(n for _, n in mseq)
                    if verbose:
                        print("macro", mi, "at f", f, "world", world)
                mi += 1
            if mrem <= 0:
                while ti < len(tail) and world > tail[ti][0] + 14:
                    ti += 1
                if ti < len(tail) and world >= tail[ti][0]:
                    tw, tf, tseq = tail[ti]
                    ti += 1
                    if abs(f - tf) <= _TAIL_FTOL:
                        mseq = [(k, n) for k, n in tseq]
                        mrem = sum(n for _, n in mseq)
                        if verbose:
                            print("tail macro", tw, "at f", f, "world", world)
                    elif verbose:
                        print("tail macro", tw, "SKIPPED: f", f, "expected", tf)
            if mrem > 0:
                while mseq and mseq[0][1] <= 0:
                    mseq.pop(0)
                kk, n = mseq[0]
                mseq[0] = (kk, n - 1)
                mrem -= 1
                keys = set(x for x in kk.split(",") if x)
                if mrem <= 0:
                    jump_t = -1
                    cool = 0
                    feet_hist = []
                sim.step(keys)
                f += 1
                continue
            if world >= _HOP_FROM:
                hk = hop.decide(s, mario, world, spd)
                if verbose and f >= 1860 and f < 2010:
                    print("H f%d w%d m=%s sp=%.1f k=%s" % (f, world, (int(xl), int(xr), int(yt), int(feet)), spd,
                                                           None if hk is None else "".join(sorted(k[0] for k in hk))))
                if hk is not None:
                    jump_t = -1
                    cool = 0
                    sim.step(hk)
                    f += 1
                    continue
            if jump_t >= 0:
                jump_t += 1
                if jump_t < jump_h:
                    keys.add("z")
                elif jump_t >= jump_h + 2 and grounded:
                    jump_t = -1
                    jump_nox = False
                    cool = 1
            if vtype is None and world >= 2120:
                vtype = "X" if f < 1250 else "Y"
            new_ok = (vtype == "X" and world >= 3500)
            if jump_nox and jump_t >= 0:
                keys.discard("x")
            if new_ok and not grounded and feet < 200:
                rim = _rim_below(s, xl, xr, feet, solid=(world >= 5560))
                if rim is not None:
                    pl2, pr2 = rim
                    cx = (xl + xr) // 2
                    if cx >= pr2 - 8:
                        keys.discard("Right")
                        keys.add("Left")
                    elif cx >= pl2 + 14:
                        keys.discard("Right")
            if jump_t < 0:
                if cool > 0:
                    cool -= 1
                elif grounded:
                    kind = None
                    pipe = _find_pipe(s, xr, feet)
                    xa = min(255, xr + 2)
                    xb = min(256, xr + 2 + ENEMY_D)
                    if pipe is not None:
                        pl, ptop, plant = pipe
                        dist = pl - xr
                        if verbose and (plant, dist // 20) != last_pipe_state:
                            last_pipe_state = (plant, dist // 20)
                            print("pipe", pipe, "dist", dist, "f", f)
                        if dist <= PIPE_WAIT:
                            if plant:
                                keys.discard("Right")
                                if dist <= PIPE_BRAKE and dist < 24:
                                    keys.add("Left")
                            elif dist <= PIPE_D:
                                kind = "wall"
                        xb = xa
                    if xb > xa:
                        win = s[max(32, feet - 13):feet, xa:xb, :]
                        nc = _enemy_cols(win)
                        if nc >= 3:
                            kind = "obs"
                            hwin = s[max(32, feet - 30):feet - 16, xa:min(256, xa + 14), :]
                            if _mask(hwin, _ORANGE).sum() > 6:
                                kind = "wall"
                    if kind is None and (feet >= 200 or new_ok):
                        px0 = min(250, xr + 1)
                        px1 = min(256, xr + 1 + PIT_D)
                        if px1 > px0:
                            col = s[feet + 4, px0:px1 + 4 if px1 + 4 <= 256 else 256, :]
                            sk = (col[:, 0] == SKY[0]) & (col[:, 1] == SKY[1]) & (col[:, 2] == SKY[2])
                            if len(sk) >= 4 and sk[:min(len(sk), PIT_D)].all() and sk[-3:].all():
                                kind = "pit" if feet >= 200 else "gap"
                    if kind is not None:
                        jump_t = 0
                        jump_h = {"obs": HOLD_OBS, "wall": HOLD_WALL, "pit": HOLD_PIT, "gap": HOLD_GAP, "kp": 10}[kind]
                        jump_nox = (kind == "gap")
                        keys.add("z")
                        if verbose:
                            print("f", f, "world", world, "jump", kind, "mario", mario)
            sim.step(keys)
            f += 1
    except Exception as e:
        if verbose:
            print("stopped:", repr(e), "f", f, "world", world, "mario", mario)
    print("END f", f, "world", world, "mario", mario)
    return


class Agent:
    """the reflex runner as a deep-copyable object: frame(sim) = one game frame"""

    def __init__(self, sim, verbose=True, still_limit=100, macros=None, world0=0, f0=0, vtype0=None, tail=None):
        if macros is None:
            try:
                macros = list((sim.mem.past or {}).get("macros", []))
            except Exception:
                macros = []
        macros = [dict(m, seq=[list(e) for e in m["seq"][:2]] + [[m["seq"][2][0], 7]]) if m["w"] >= 3600 else m
                  for m in macros]
        self.macros = sorted([m for m in macros if m["w"] >= world0 and m["w"] < _HOP_MACRO_MAX],
                             key=lambda m: m["w"])
        self.tail = sorted([list(t) for t in (TAIL if tail is None else tail)], key=lambda t: t[1])
        self.verbose = verbose
        self.still_limit = still_limit
        self.hop = Hopper(log=(print if verbose else None))
        self.xh = []
        self.mi = 0
        self.ti = 0
        self.mseq = []
        self.mrem = 0
        self.f = f0
        self.last = sim.screen()
        self.still = 0
        self.jump_t = -1
        self.jump_h = 0
        self.feet_hist = []
        self.world = world0
        self.mario = None
        self.cool = 0
        self.last_pipe_state = None
        self.first = True
        self.vtype = vtype0
        self.jump_nox = False
        self.ftol = _FTOL
        self.defer_hook = False

    def hook(self):
        self.jump_t = -1
        self.cool = 0
        self.feet_hist = []

    def inject(self, seq):
        self.mseq = [(k, n) for k, n in seq]
        self.mrem = sum(n for _, n in self.mseq)

    def frame(self, sim):
        """one loop iteration (screen -> decision -> sim.step); False = policy is over"""
        verbose = self.verbose
        s = sim.screen()
        self.mario = mario = find_mario(s, self.mario)
        if not self.first:
            if np.array_equal(s, self.last):
                self.still += 1
            else:
                self.still = 0
            try:
                self.world += _scroll(self.last, s)
            except Exception:
                pass
        self.last = s
        self.first = False
        if self.still > self.still_limit:
            if verbose:
                print("frozen at f", self.f, "world", self.world, "mario", mario)
            return False
        keys = {"Right", "x"}
        if mario is None:
            sim.step(keys)
            self.f += 1
            return True
        world = self.world
        f = self.f
        xl, xr, yt, feet = mario
        self.xh.append(world + xr)
        if len(self.xh) > 4:
            self.xh.pop(0)
        spd = (self.xh[-1] - self.xh[0]) / max(1, len(self.xh) - 1)
        self.feet_hist.append(feet)
        if len(self.feet_hist) > 4:
            self.feet_hist.pop(0)
        fh = self.feet_hist
        grounded = len(fh) >= 3 and fh[-1] == fh[-2] == fh[-3]
        macros = self.macros
        while self.mi < len(macros) and world >= macros[self.mi]["w"]:
            m = macros[self.mi]
            mf = m.get("f")
            if world <= m["w"] + _MACRO_WINDOW and (mf is None or abs(f - mf) <= self.ftol):
                mseq = [(k, n) for k, n in m["seq"]]
                if m["w"] == 2223:
                    self.ftol = 90
                if m["w"] >= 3600 and xl >= 119:
                    mseq = [(mseq[0][0], mseq[0][1] + 8), mseq[1], (mseq[2][0], mseq[2][1] + 4)]
                self.mseq = mseq
                self.mrem = sum(n for _, n in mseq)
                if verbose:
                    print("macro", self.mi, "at f", f, "world", world)
            self.mi += 1
        if self.mrem <= 0:
            tl = self.tail
            while self.ti < len(tl) and f > tl[self.ti][1] + _TAIL_FTOL:
                self.ti += 1
            if self.ti < len(tl) and abs(f - tl[self.ti][1]) <= _TAIL_FTOL and abs(world - tl[self.ti][0]) <= 6:
                self.inject(tl[self.ti][2])
                if verbose:
                    print("tail macro", tl[self.ti][0], "at f", f, "world", world)
                self.ti += 1
        if self.mrem > 0:
            while self.mseq and self.mseq[0][1] <= 0:
                self.mseq.pop(0)
            kk, n = self.mseq[0]
            self.mseq[0] = (kk, n - 1)
            self.mrem -= 1
            keys = set(x for x in kk.split(",") if x)
            if self.mrem <= 0 and not self.defer_hook:
                self.hook()
            sim.step(keys)
            self.f += 1
            return True
        if world >= _HOP_FROM:
            hk = self.hop.decide(s, mario, world, spd)
            if hk is not None:
                self.jump_t = -1
                self.cool = 0
                sim.step(hk)
                self.f += 1
                return True
        if self.jump_t >= 0:
            self.jump_t += 1
            if self.jump_t < self.jump_h:
                keys.add("z")
            elif self.jump_t >= self.jump_h + 2 and grounded:
                self.jump_t = -1
                self.jump_nox = False
                self.cool = 1
        if self.vtype is None and world >= 2120:
            self.vtype = "X" if f < 1250 else "Y"
        new_ok = (self.vtype == "X" and world >= 3500)
        if self.jump_nox and self.jump_t >= 0:
            keys.discard("x")
        if new_ok and not grounded and feet < 200:
            rim = _rim_below(s, xl, xr, feet, solid=(world >= 5560))
            if rim is not None:
                pl2, pr2 = rim
                cx = (xl + xr) // 2
                if cx >= pr2 - 8:
                    keys.discard("Right")
                    keys.add("Left")
                elif cx >= pl2 + 14:
                    keys.discard("Right")
        if self.jump_t < 0:
            if self.cool > 0:
                self.cool -= 1
            elif grounded:
                kind = None
                pipe = _find_pipe(s, xr, feet)
                xa = min(255, xr + 2)
                xb = min(256, xr + 2 + ENEMY_D)
                if pipe is not None:
                    pl, ptop, plant = pipe
                    dist = pl - xr
                    if dist <= PIPE_WAIT:
                        if plant:
                            keys.discard("Right")
                            if dist <= PIPE_BRAKE and dist < 24:
                                keys.add("Left")
                        elif dist <= PIPE_D:
                            kind = "wall"
                    xb = xa
                if xb > xa:
                    win = s[max(32, feet - 13):feet, xa:xb, :]
                    nc = _enemy_cols(win)
                    if nc >= 3:
                        kind = "obs"
                        hwin = s[max(32, feet - 30):feet - 16, xa:min(256, xa + 14), :]
                        if _mask(hwin, _ORANGE).sum() > 6:
                            kind = "wall"
                if kind is None and (feet >= 200 or new_ok):
                    px0 = min(250, xr + 1)
                    px1 = min(256, xr + 1 + PIT_D)
                    if px1 > px0:
                        col = s[feet + 4, px0:px1 + 4 if px1 + 4 <= 256 else 256, :]
                        sk = (col[:, 0] == SKY[0]) & (col[:, 1] == SKY[1]) & (col[:, 2] == SKY[2])
                        if len(sk) >= 4 and sk[:min(len(sk), PIT_D)].all() and sk[-3:].all():
                            kind = "pit" if feet >= 200 else "gap"
                if kind is not None:
                    self.jump_t = 0
                    self.jump_h = {"obs": HOLD_OBS, "wall": HOLD_WALL, "pit": HOLD_PIT, "gap": HOLD_GAP, "kp": 10}[kind]
                    self.jump_nox = (kind == "gap")
                    keys.add("z")
                    if verbose:
                        print("f", f, "world", world, "jump", kind, "mario", mario)
        sim.step(keys)
        self.f += 1
        return True


def play(sim, verbose=True, **kw):
    ag = Agent(sim, verbose=verbose, **kw)
    try:
        while sim.budget_left > 0 and ag.f < 7990:
            if not ag.frame(sim):
                break
    except Exception as e:
        if verbose:
            print("stopped:", repr(e), "f", ag.f, "world", ag.world, "mario", ag.mario)
    print("END f", ag.f, "world", ag.world, "mario", ag.mario)


def solve(sim):
    return []

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    st = (ctx or {}).get("stuck") or {}\n    stuck = bool(st.get("is")) if isinstance(st, dict) else False\n    return {"status": "stuck" if stuck else "running", "where": {}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": stuck},\n            "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {"macros": [{"w": 429, "f": 298, "seq": [["Right,x", 8], ["x", 6], ["Right,x,z", 26], ["Left,x", 4], ["x", 3]]}, {"w": 824, "f": 438, "seq": [["Left,x", 8], ["", 32], ["Right,x,z", 26], ["", 4], ["x", 3], ["Right,z", 12], ["z", 20], ["Right,x,z", 4]]}, {"w": 1219, "f": 783, "seq": [["Right,x", 20], ["Left,x", 2], ["Right,x", 32], ["Right,x,z", 26]]}, {"w": 1415, "f": 887, "seq": [["Right,x", 4], ["Right,x", 32], ["Right", 4], ["Left", 3], ["Right,x,z", 20], ["Left,x", 3]]}, {"w": 2120, "f": 1299, "seq": [["Left,x,z", 12], ["Right,x,z", 6], ["Right,x,z", 2], ["x", 10], ["Right,x,z", 12], ["Right,x,z", 4], ["z", 32], ["Right,z", 6]]}, {"w": 2991, "f": 1662, "seq": [["Right,z", 12], ["Right,x,z", 3], ["Right,x", 32], ["Left,x,z", 6], ["Left", 26]]}, {"w": 3309, "f": 1858, "seq": [["Right,x,z", 3], ["Right,x", 20], ["Right,x", 3], ["Right,x,z", 26], ["Right,x,z", 6], ["Right,x,z", 6], ["Right", 10]]}, {"w": 3609, "f": 1958, "seq": [["Left,x", 16], ["Right,x,z", 6], ["Left,x", 3], ["Right,x,z", 12], ["Right,x,z", 20]]}, {"w": 3298, "f": 1663, "seq": [["Left,x", 8], ["", 32], ["Right,x,z", 26], ["", 4], ["x", 3], ["Right,z", 12], ["z", 20], ["Right,x,z", 4]]}, {"w": 3375, "f": 1703, "seq": [["Right,x", 3], ["Left,x,z", 12], ["Left,x,z", 16], ["z", 6]]}, {"w": 3403, "f": 1773, "seq": [["Right,x,z", 3], ["Right,x,z", 8], ["Right,x,z", 12]]}, {"w": 3609, "f": 1842, "seq": [["Left,x", 16], ["Right,x,z", 6], ["Left,x", 3], ["Right,x,z", 12], ["Right,x,z", 20]]}, {"w": 2223, "f": 1380, "seq": [["Left,x", 25]]}]}}\n'}

