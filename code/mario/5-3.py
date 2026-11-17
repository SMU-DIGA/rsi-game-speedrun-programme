# ==================================================================================================
# Super Mario Bros. (NES) -- level 5-3  |  超级马里奥兄弟（NES） 关 5-3
# Pass rate on 100 random variants never used before: 38/100; fastest finish 1162 frames (19.37 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 38/100；最快一局 1162 帧（60 帧/秒，约 19.37 秒）。
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
#   hybrid closed-loop planner (c4 arc planner, then lift-aware planner c7) plus a Bullet-Bill spa
#   wn-phase gate: do not start the lift1->lift2 hop when an unseen Bill is predicted to spawn (pe
#   riod ~82 f)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _mod                           line 2914  
#   def solve                          line 2946  
#   def play                           line 2950  
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hybrid closed-loop planner (c4 arc planner, then lift-aware planner c7) plus a Bullet-Bill spawn-phase gate: do not start the lift1->lift2 hop when an unseen Bill is predicted to spawn (period ~82 f) around take-off/landing
import types
import numpy as np

SRC_A = r'''
import numpy as np


def kimg(img):
    return (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1]


K_SKY = (104 << 8) | 136
K_RED = (248 << 8) | 56
K_BROWN = (172 << 8) | 124
K_LIME = (184 << 8) | 248
K_TAN = (240 << 8) | 208
K_TRUNK = (228 << 8) | 92

ENEMIES = []
LIFTS = []
FAIL = {}


def _fill(sm):
    out = []
    for i, s in enumerate(sm):
        out.append(s)
        if i + 1 < len(sm):
            out.append((s + sm[i + 1]) // 2)
    return out


RISE = {
    2: [0, 0, 6, 12, 18, 23, 27, 30, 32, 33, 34, 34],
    4: [0, 0, 6, 12, 18, 23, 28, 33, 36, 39, 41, 42, 42],
    6: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 45, 47, 49, 49],
    8: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 50, 53, 54, 55, 56],
    11: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 63, 64, 64],
    14: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 71, 71],
    18: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 73, 75, 77, 78, 79],
    24: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 73, 75, 77, 79, 81, 82, 83, 84, 85, 85],
    32: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 73, 75, 77, 79, 81, 82, 83, 84, 85, 85, 86, 86, 86],
}
STAND = {
    8: _fill([0, 5, 14, 23, 30, 37, 42, 44]),
    14: _fill([0, 5, 14, 23, 30, 37, 43, 48, 53, 55]),
    24: _fill([0, 5, 14, 23, 30, 37, 43, 48, 53, 57, 60, 62, 63, 64]),
    32: _fill([0, 5, 14, 23, 30, 37, 43, 48, 53, 57, 60, 62, 63, 64, 64]),
}
KS = (2, 4, 6, 8, 11, 14, 18, 24, 32)
OPTS = [(k, None) for k in KS]
DY = [1, 2, 2, 4, 4, 5]
_VMAX = 3.0
_ACC = 0.07
_BRK = 0.16
_LOOK = 70
H_TRIG = 19
MIN_EXT = 10
WAIT_MAX = 260
EMARG = 3
AIR_ROOM = 50
_RMARG = 0
LAND_MARG = 30
LAND_BACK = 13
DEBUG = False
LATE_TF = 2
FLIP_FIX = False
BILL_VX = -2.6969
BILL_HOP = True
BILL_MARG = 0
BILL_LEAD = 46
BILL_HIGH = True
BH_F = 120
BH_X = 800
PREFER_LEVEL = True
BILL_WAIT = True
J1 = 0
HOLD_MAX = 0
WAIT_TOTAL = 700
STOMP_R = False
LAND_STRICT = [True]
AIRB_ON = False
K32_ON = False
K32_TB = None
K32_FORCE = True
AIRB_STOMP = 6


def rise_at(tab, t):
    if t < len(tab):
        return tab[t]
    r = tab[-1]
    for i in range(t - len(tab) + 1):
        r -= DY[min(i, len(DY) - 1)]
    return r


_XC = {}


def xs_table(v0, tbrake):
    key = (round(v0, 1), tbrake)
    X = _XC.get(key)
    if X is None:
        X = [0.0]
        v = v0
        for i in range(1, 150):
            if tbrake is not None and i > tbrake:
                v = max(v - _BRK, -1.5)
            else:
                v = min(_VMAX, v0 + _ACC * i)
            X.append(X[-1] + v)
        _XC[key] = X
    return X


class Perc:
    def __init__(self):
        self.mx = 40.0
        self.my = 192.0
        self.found = False
        self.scroll = 0
        self.prev = None

    def mario(self, K, lifts=()):
        m = (K == K_RED) | (K == K_BROWN)
        m[:32] = False
        for (top, x0, x1) in lifts:
            m[max(top - 1, 0):top + 9, x0:x1 + 1] = False
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            self.found = False
            return None
        sel = (np.abs(xs - (self.mx + 8)) < 28) & (ys > self.my - 30) & (ys < self.my + 40)
        if sel.sum() < 4:
            self.found = False
            return None
        ys = ys[sel]
        xs = xs[sel]
        cx = int(np.median(xs))
        cy = int(np.median(ys))
        sel = (np.abs(xs - cx) <= 9) & (np.abs(ys - cy) <= 9)
        ys = ys[sel]
        xs = xs[sel]
        self.found = True
        x0 = xs.min()
        y1 = ys.max()
        self.mx = float(x0 if xs.max() - x0 <= 13 else xs.max() - 12)
        self.my = float(y1 - 15)
        return self.mx, self.my

    def scroll_est(self, K):
        if self.prev is None:
            self.prev = K
            return 0
        a = self.prev[40:232:3]
        b = K[40:232:3]
        nz = a != K_SKY
        best, bs = -1, 0
        for s in range(0, 6):
            if s == 0:
                sc = ((a == b) & nz).sum()
            else:
                sc = ((a[:, s:] == b[:, :-s]) & nz[:, s:]).sum()
            if sc > best:
                best, bs = sc, s
        self.prev = K
        return bs


def find_lifts(K):
    red = K == K_RED
    red[:32] = False
    cnt = red.sum(1)
    rows = np.nonzero(cnt >= 24)[0]
    out = []
    last = -99
    for y in rows.tolist():
        if y - last <= 8 and out:
            last = y
            continue
        r = red[y].astype(np.int8)
        d = np.diff(np.concatenate(([0], r, [0])))
        st = np.nonzero(d == 1)[0]
        en = np.nonzero(d == -1)[0]
        ln = en - st
        i = int(np.argmax(ln))
        if ln[i] >= 24:
            out.append((y - 1, int(st[i]), int(en[i]) - 1))
            last = y
    return out


def solid_mask(K, scroll=0):
    sol = K == K_LIME
    h = np.zeros_like(sol)
    h[:, 2:] |= sol[:, :-2]
    h[:, :-2] |= sol[:, 2:]
    h[:, 1:] |= sol[:, :-1]
    h[:, :-1] |= sol[:, 1:]
    below = np.zeros_like(sol)
    below[:-1] |= sol[1:]
    below[:-2] |= sol[2:]
    sol = sol | (h & below)
    if scroll < 330:
        nz = K[208] != K_SKY
        if nz[0]:
            z = np.nonzero(~nz)[0]
            end = int(z[0]) if len(z) else 256
            sol[208:, :end] = True
    return sol


def find_clusters(m, minpts=5, cs=8):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return []
    cells = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        cells.setdefault((y // cs, x // cs), []).append((y, x))
    seen = set()
    out = []
    for key in list(cells):
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        pts = []
        while stack:
            c = stack.pop()
            pts.extend(cells[c])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (c[0] + dy, c[1] + dx)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(pts) < minpts:
            continue
        yy = [q[0] for q in pts]
        xx = [q[1] for q in pts]
        out.append((min(xx), max(xx), min(yy), max(yy), len(pts)))
    return out


WALL = [None]
USE_TRUNK = False
BG_ON = False
ZR_ON = True
ZR_V = 1.8
NOBSTOMP = True
AB_ON = False
AB_D = 70
BHZ_ON = False
ZK_ON = False
ZK_MIN = 11
BG_WIN = 30
BG_MAX = 330


def bgate(pol, wx, mx, F, v0):
    """tiny-cap (wx1100-1175, F~145): hold still until a Bullet Bill has just flown past Mario (none on the right)."""
    if not (BG_ON and 1100 < wx < 1175 and 138 <= F <= 152):
        return None
    bills = [e for e in ENEMIES if e[4] == 8 and 40 < e[1] < F + 6]
    right = any(e[0] > mx + 10 for e in bills)
    left = any(e[0] < mx - 6 for e in bills)
    fn = pol.fno
    if left and not right:
        pol.bg_pass = fn
    if fn < getattr(pol, "bg_open", -1):
        return None
    near = any(mx + 10 < e[0] < mx + 170 for e in bills)
    if not near:
        return None
    pol.bg_n = getattr(pol, "bg_n", 0) + 1
    if pol.bg_n > BG_MAX:
        pol.bg_open = fn + 150
        return None
    return {"Left"} if v0 > 0.6 else set()


def wallhit(sol, y0, y1, x):
    if sol[y0:y1, x].any():
        return True
    w = WALL[0]
    return w is not None and x < 256 and bool(w[y0:y1, x].any())


def en_pos(e, T):
    ex, ey, ev, vy, hw, h, y0, y1, st = e
    x = ex + ev * T
    if vy == 0 or y1 - y0 < 2:
        return x, ey
    L = float(y1 - y0)
    u = ((ey - y0) + vy * T) % (2 * L)
    if u > L:
        u = 2 * L - u
    return x, y0 + u


def run_only(sol, mx, F, v0, N=_LOOK, deep=True):
    X = xs_table(v0, None)
    for t in range(1, N + 1):
        x = mx + X[t]
        xi = int(x)
        if xi + 14 >= 256:
            return None, None
        for e in ENEMIES:
            exx, yb = en_pos(e, t)
            if abs(x + 6 - exx) < e[4] + 6 + EMARG and F - 15 < yb + 1 and F - 1 > yb - e[5]:
                return t, int(exx)
        if F - 14 >= 0 and wallhit(sol, F - 14, F - 2, xi + 12):
            return t, xi + 12
        if not sol[F, xi + 5]:
            vend = min(_VMAX, v0 + _ACC * t)
            r = sim_fall(sol, x - vend, F, vend, 0)
            if r is None:
                return t, xi + 6
            if r[1] - F > 100:
                return t, xi + 6
            if ext_right(sol, r[0] + 5, r[1]) < MIN_EXT:
                return t, xi + 6
            if deep and not cont_ok(sol, r[0], r[1], vend):
                return t, xi + 6
            return None, None
    return None, None


def sim_fall(sol, x, F, vx, di=0, maxt=60):
    for t in range(maxt):
        dy = DY[min(di + t, len(DY) - 1)]
        Fn = F + dy
        x += vx
        xi = int(x)
        if xi + 12 >= 256 or xi < 0:
            return None
        hit = None
        for r in range(F, Fn + 1):
            if r < 240 and sol[r, xi + 4:xi + 7].any():
                hit = r
                break
        if hit is not None:
            return x, hit
        F = Fn
        if F > 232:
            return None
    return None


def sim_fall_b(sol, x, F, vx, di, brake, maxt=60):
    for t in range(maxt):
        dy = DY[min(di + t, len(DY) - 1)]
        Fn = F + dy
        x += vx
        if brake:
            vx = max(vx - _BRK, -1.5)
        xi = int(x)
        if xi + 12 >= 256 or xi < 0:
            return None
        for e in ENEMIES:
            exx, yb = en_pos(e, t + 1)
            if abs(x + 6 - exx) < e[4] + 6 + EMARG and Fn - 15 < yb + 1 and Fn - 1 > yb - e[5]:
                if e[8] and Fn - 1 <= yb - 5:
                    return (x, Fn)
                return None
        hit = None
        for r in range(F, Fn + 1):
            if r < 240 and sol[r, xi + 4:xi + 7].any():
                hit = r
                break
        if hit is not None:
            return x, hit
        F = Fn
        if F > 232:
            return None
    return None


def ext_right(sol, xc, F, cap=40):
    n = 0
    xi = int(xc)
    while n < cap and xi + n < 255 and sol[F, xi + n]:
        n += 1
    if xi + n >= 255:
        return cap
    return n


def sim_jump(sol, mx, F0, v0, k, tb, delay, fail_x, min_ext=MIN_EXT):
    vt = min(_VMAX, v0 + _ACC * (delay + 1))
    tab = STAND.get(k) if vt < 2.2 else RISE[k]
    if tab is None:
        FAIL[(k, tb)] = "nostand"
        return None
    X = xs_table(v0, None if tb is None else delay + tb)
    xd = int(mx + X[delay + 1])
    if xd + 12 >= 256 or xd < 0 or not sol[F0, xd + 5]:
        FAIL[(k, tb)] = "nosupport"
        return None
    Fcur = F0
    prev_r = 0
    t = 0
    landed = None
    while t < 90:
        t += 1
        T = t + delay
        x = mx + X[T]
        xi = int(x)
        if xi < 0:
            FAIL[(k, tb)] = "left"
            return None
        if xi + 13 >= 256:
            FAIL[(k, tb)] = "edge t%d" % t
            return (x, Fcur, -1)
        r = rise_at(tab, t - 1)
        if r > 6:
            r -= _RMARG
        Fn = F0 - r
        for (lx0, lx1, ltop, lvy) in LIFTS:
            lt = ltop + lvy * T
            if xi + 12 >= lx0 - 1 and xi + 2 <= lx1 + 1:
                if r < prev_r and Fcur <= lt + 2 and Fn >= lt - 1 and xi + 7 >= lx0 and xi + 4 <= lx1:
                    return (x, int(lt), 30 if lx1 - (xi + 5) > 8 else lx1 - (xi + 5))
                if Fn - 17 < lt + 10 and Fn > lt - 2:
                    FAIL[(k, tb)] = "lift t%d" % t
                    return None
        for e in ENEMIES:
            exx, yb = en_pos(e, T)
            if abs(x + 6 - exx) < e[4] + 6 + EMARG + (BILL_MARG if e[4] == 8 else 0) and Fn - 17 < yb + 1 and Fn + 1 > yb - e[5]:
                if e[8] and r < prev_r and Fn - 1 <= yb - 5 and yb >= F0 - 20 and not (NOBSTOMP and e[4] == 8):
                    return (x, Fn, 30)
                FAIL[(k, tb)] = "enemy t%d" % t
                return None
        if r >= prev_r and t < len(tab) + 1:
            if Fn - 14 >= 0 and wallhit(sol, Fn - 14, Fn - 2, xi + 12):
                FAIL[(k, tb)] = "wallrise t%d" % t
                return None
            if Fn - 17 >= 0 and sol[Fn - 17, xi + 2:xi + 11].any():
                FAIL[(k, tb)] = "head t%d" % t
                return None
        else:
            if Fn - 14 >= 0 and wallhit(sol, Fn - 14, Fn - 2, xi + 12):
                FAIL[(k, tb)] = "walldesc t%d" % t
                return None
            hit = None
            for rr in range(Fcur, Fn + 1):
                if rr < 240 and sol[rr, xi + 4:xi + 7].any():
                    hit = rr
                    break
            if hit is not None:
                Fcur = hit
                landed = (x, hit)
                break
        prev_r = r
        Fcur = Fn
        if Fcur > 232:
            FAIL[(k, tb)] = "fell t%d" % t
            return None
    if landed is None:
        FAIL[(k, tb)] = "nolanding"
        return None
    xl, Fl = landed
    if xl + 5 < fail_x + 2:
        FAIL[(k, tb)] = "short"
        return None
    ext = ext_right(sol, xl + 5, Fl)
    if ext < min_ext:
        FAIL[(k, tb)] = "ext%d" % ext
        return None
    Tl = t + delay
    for e in (ENEMIES if LAND_STRICT[0] else ()):
        if e[3] == 0 and e[5] <= 14 and abs(e[1] - Fl) < 10:
            for dT in (0, 6):
                d = e[0] + e[2] * (Tl + dT) - (xl + 6)
                if -LAND_BACK < d < LAND_MARG:
                    FAIL[(k, tb)] = "landenemy"
                    return None
    return xl, Fl, ext


def cont_ok(sol, xl, Fl, vl):
    global ENEMIES, LIFTS
    se, sl = ENEMIES, LIFTS
    ENEMIES, LIFTS = [], []
    try:
        tf2, fx2 = run_only(sol, xl, Fl, vl, _LOOK, False)
        if tf2 is None:
            return True
        for d in range(0, min(tf2, 22) + 1, 2):
            for (k, tb) in OPTS[:len(KS)]:
                r = sim_jump(sol, xl, Fl, vl, k, tb, d, fx2)
                if r is not None and r[2] >= 0:
                    return True
        return False
    finally:
        ENEMIES, LIFTS = se, sl


def find_bills(K):
    blk = (K[32:] == 0).astype(np.int32)
    ii = np.zeros((blk.shape[0] + 1, blk.shape[1] + 1), np.int32)
    ii[1:, 1:] = blk.cumsum(0).cumsum(1)
    S = ii[8:, 8:] - ii[:-8, 8:] - ii[8:, :-8] + ii[:-8, :-8]
    m = S >= 34
    out = []
    if not m.any():
        return out
    for (x0, x1, y0, y1, n) in find_clusters(m, 1):
        w = x1 - x0 + 8
        h = y1 - y0 + 8
        if 8 <= w <= 26 and 6 <= h <= 22:
            out.append(((x0 + x1 + 8) / 2.0, y1 + 8 + 32))
    return out


def find_tan(K, excl=()):
    m = K == K_TAN
    m[:32] = False
    for (bx, by) in excl:
        m[max(int(by) - 17, 0):int(by) + 3, max(int(bx) - 11, 0):int(bx) + 12] = False
    out = []
    for (x0, x1, y0, y1, n) in find_clusters(m, 5, 4):
        out.append(((x0 + x1) / 2.0, y1 + 2))
    return out


TB_ON = True
TB_T = 0
TB_M = 8


def tower_brake(pol, sol, mx, my, dy):
    """airborne above the narrow tower top (wx 1240-1340) with speed: brake (Left) so a bounce/hop does not carry Mario over the edge."""
    wx = pol.P.scroll + mx
    feet = int(my) + 16
    if not (TB_ON and 1240 < wx < 1340 and feet <= 100):
        return None
    if pol.speed() < 1.2:
        return None
    xl = int(mx) + 5 - 14
    if xl < 0:
        return None
    surf = None
    for r in range(max(feet - 2, 0), 112):
        if sol[r, xl]:
            surf = r
            break
    if surf is None or surf < 90 or surf > 104:
        return None
    er = ext_right(sol, mx + 5, surf, 200)
    h = float(surf - feet)
    if dy >= 1:
        di = 0
        bst = 99
        for i, q in enumerate(DY):
            if abs(q - dy) < bst:
                bst = abs(q - dy)
                di = i
        rt = 0.0
    else:
        sp = max(0.0, -dy)
        rt = sp / 0.4
        h += sp * sp / 0.8
        di = 0
    hh = 0.0
    t = 0
    while hh < h and t < 60:
        hh += DY[min(di + t, len(DY) - 1)]
        t += 1
    T = rt + t + TB_T
    if 4 <= er and pol.speed() * T > er - TB_M:
        ks = {"Left", "x"}
        if dy < -0.5:
            ks.add("z")
        return ks
    return None


class Policy:
    def __init__(self):
        self.P = Perc()
        self.plan = None
        self.prev_en = []
        self.fno = 0
        self.lhist = {}
        self.wxh = []
        self.waiting = 0
        self.prev_my = None
        self.last_why = ""
        self.nowait_until = 0
        self.curF = 0
        self.hold = 0
        self.wait_total = 0

    def track(self, K, mx, my, lifts):
        global ENEMIES, LIFTS
        P = self.P
        cands = []
        bills = find_bills(K)
        F = int(my) + 16
        for (cx, ey) in find_tan(K, bills):
            if abs(cx - (mx + 6)) < 4 and abs(ey - F) < 6:
                continue
            cands.append(["g", cx, ey, 7, 14])
        for (cx, ey) in bills:
            cands.append(["b", cx, ey, 8, 14])
        red = K == K_RED
        red[:32] = False
        for (top, x0, x1) in lifts:
            red[max(top - 1, 0):top + 9, x0:x1 + 1] = False
        red[max(int(my) - 4, 0):int(my) + 20, max(int(mx) - 5, 0):int(mx) + 20] = False
        for (x0, x1, y0, y1, n) in find_clusters(red, 6):
            cands.append(["r", (x0 + x1) / 2.0, y1 + 8, 9, 26])
        cur = []
        used = set()
        for kind, cx, ey, hw, h in cands:
            wx = cx + P.scroll
            m = None
            for i, pe in enumerate(self.prev_en):
                if i in used or pe["kind"] != kind:
                    continue
                if abs(pe["wx"] - wx) < 6 and abs(pe["y"] - ey) < 10:
                    m = i
                    break
            if m is None:
                e = dict(kind=kind, wx=wx, y=ey, vx=(-0.6 if kind == "g" else (-1.5 if kind == "b" else 0.0)), vy=0.0, ymin=ey, ymax=ey, n=1, hw=hw, h=h)
            else:
                used.add(m)
                pe = self.prev_en[m]
                e = dict(pe)
                e["vx"] = 0.7 * pe["vx"] + 0.3 * (wx - pe["wx"])
                if kind == "r":
                    e["vy"] = 0.8 * pe["vy"] + 0.2 * (ey - pe["y"])
                e["wx"] = wx
                e["y"] = ey
                e["ymin"] = min(pe["ymin"], ey)
                e["ymax"] = max(pe["ymax"], ey)
                e["n"] = pe["n"] + 1
            cur.append(e)
        self.prev_en = cur
        out = []
        for e in cur:
            ex = e["wx"] - P.scroll
            if e["kind"] in ("g", "b"):
                evx = max(-2.5, min(2.5, e["vx"]))
                if e["kind"] == "b" and e["n"] >= 3:
                    evx = BILL_VX
                out.append((ex, e["y"], evx, 0.0, e["hw"], 14, e["y"], e["y"], True))
            else:
                y0, y1 = e["ymin"] - 3, e["ymax"] + 3
                vy = e["vy"]
                if abs(vy) < 0.12:
                    vy = 0.0 if e["n"] > 80 else 0.5
                if vy > 0:
                    y1 = max(y1, e["y"] + 45)
                elif vy < 0:
                    y0 = min(y0, e["y"] - 45)
                else:
                    y1 = y0 + 1
                out.append((ex, e["y"], max(-2.0, min(2.0, e["vx"])), vy, 9, 26, y0, y1, STOMP_R))
        ENEMIES = out
        self.fno += 1
        LIFTS = []
        for (top, x0, x1) in lifts:
            key = int((x0 + P.scroll) // 12)
            h = None
            for kk in (key, key - 1, key + 1):
                if kk in self.lhist:
                    h = self.lhist[kk]
                    break
            if h is None:
                h = []
            h.append((self.fno, top))
            del h[:-40]
            for kk in (key - 1, key + 1):
                self.lhist.pop(kk, None)
            self.lhist[key] = h
            vy = 0.0
            if len(h) >= 8:
                vy = (h[-1][1] - h[0][1]) / float(h[-1][0] - h[0][0])
                vy = max(-1.5, min(1.5, vy))
            LIFTS.append((x0, x1, top, vy))

    def speed(self):
        h = self.wxh
        if len(h) < 5:
            return 0.0
        v = (h[-1] - h[-5]) / 4.0
        return max(0.0, min(_VMAX, v))

    def decide(self, img):
        P = self.P
        LAND_STRICT[0] = True
        K = kimg(img)
        d = P.scroll_est(K)
        P.scroll += d
        lifts = find_lifts(K)
        mp = P.mario(K, lifts)
        keys = {"Right", "x"}
        if mp is None:
            pl = self.plan
            if pl is not None:
                pl["t"] += 1
                k2 = {"Right", "x"}
                if pl.get("stay"):
                    k2 = set()
                if pl["t"] < pl["k"]:
                    k2.add("z")
                if pl["tb"] is not None and pl["t"] >= pl["tb"]:
                    k2.discard("Right")
                    k2.add("Left")
                return k2, "air"
            return keys, None
        mx, my = mp
        self.wxh.append(P.scroll + mx)
        del self.wxh[:-8]
        self.track(K, mx, my, lifts)
        F = int(my) + 16
        sol = solid_mask(K, P.scroll)
        WALL[0] = None
        grounded = False
        for r_ in (F, F - 1, F + 1, F + 2):
            if 0 <= r_ < 240 and sol[r_, int(mx) + 4:int(mx) + 7].any():
                F = r_
                grounded = True
                break
        if not grounded:
            global LIFTS
            for lf in LIFTS:
                lx0, lx1, ltop, lvy = lf
                if abs(F - ltop) <= 3 and lx0 - 4 <= mx + 5 <= lx1 + 4:
                    grounded = True
                    F = int(ltop)
                    sol = sol.copy()
                    sol[F:F + 8, max(lx0, 0):lx1 + 1] = True
                    LIFTS = [l for l in LIFTS if l is not lf]
                    break

        if self.plan is not None:
            pl = self.plan
            pl["t"] += 1
            done_air = pl["t"] >= pl["k"] + 2 and (pl["tb"] is None or pl["t"] >= pl["tb"] + 14)
            if pl["t"] >= 6 and (grounded or done_air):
                self.plan = None
                if not grounded:
                    return self.air_control(sol, mx, my, keys)
            else:
                if pl.get("stay"):
                    keys = set()
                if pl["t"] < pl["k"]:
                    keys.add("z")
                if pl["tb"] is not None and pl["t"] >= pl["tb"]:
                    keys.discard("Right")
                    keys.add("Left")
                return keys, "air"
        if not grounded:
            return self.air_control(sol, mx, my, keys)
        self.prev_my = None

        v0 = self.speed()
        self.curF = F
        if self.waiting == 0 and v0 > 2.7:
            v0 = 3.0
        tf, fx = run_only(sol, mx, F, v0)

        if BILL_HOP and self.plan is None:
            bd = self.bill_ahead(F, mx)
            if bd is not None and bd < 62:
                global ENEMIES
                se = ENEMIES
                ENEMIES = []
                try:
                    room, _ = run_only(sol, mx, F, v0, _LOOK, False)
                finally:
                    ENEMIES = se
                tzone = BG_ON and 1100 < P.scroll + mx < 1175 and 138 <= F <= 152
                if (room is None or room > 20) and bd < 46 and not tzone:
                    self.plan = dict(k=8, tb=None, t=0, stay=False)
                    return {"z", "Right", "x"}, "billhop"
                if (12 < bd < 30 and v0 < 1.0 and bd < 46) or (tzone and 12 < bd < 46 and v0 < 1.5):
                    self.plan = dict(k=8, tb=None, t=0, stay=True)
                    return {"z"}, "billhopS"
                if BILL_HIGH and v0 >= 1.0 and bd < 62 and (room is not None and room <= 20) and F >= BH_F and P.scroll + mx > BH_X and not (BG_ON and 1100 < P.scroll + mx < 1175 and 138 <= F <= 152 and getattr(self, "bg_n", 0) < BG_MAX):
                    ENEMIES = [e for e in se if e[4] != 8]
                    try:
                        bestk = None
                        for o in OPTS:
                            if o[0] < 11:
                                continue
                            r = sim_jump(sol, mx, F, v0, o[0], o[1], 0, -999, 3)
                            if r is not None and r[2] >= 0 and (bestk is None or r[2] > bestk[1][2]):
                                bestk = (o, r)
                    finally:
                        ENEMIES = se
                    if bestk is not None:
                        k, tb = bestk[0]
                        self.plan = dict(k=k, tb=tb, t=0)
                        return {"Right", "x", "z"}, "billhigh%d" % k
                    if BHZ_ON and 1100 < P.scroll + mx < 1200 and 138 <= F <= 152:
                        self.plan = dict(k=24, tb=None, t=0)
                        return {"Right", "x", "z"}, "billhighZ"
                    self.plan = dict(k=14, tb=6, t=0)
                    return {"Right", "x", "z"}, "billhighB"

        if self.plan is None and self.waiting == 0:
            gk = bgate(self, P.scroll + mx, mx, F, v0)
            if gk is not None:
                return gk, "bgate"

        if self.waiting > 0:
            return self.wait_step(sol, mx, F, v0, tf, fx)

        if tf is None or tf > H_TRIG:
            return keys, "run"

        res0 = {}
        res1 = {}
        for o in OPTS:
            res0[o] = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fx)
            res1[o] = sim_jump(sol, mx + (J1 if tf <= 6 else 0), F, v0, o[0], o[1], 1, fx)
        ok0 = [o for o in OPTS if res0[o] is not None and res0[o][2] >= 0]
        ok1 = [o for o in OPTS if res1[o] is not None and res1[o][2] >= 0]
        if ZK_ON and 1100 < P.scroll + mx < 1175 and 138 <= F <= 152:
            ok0 = [o for o in ok0 if o[0] >= ZK_MIN]
            ok1 = [o for o in ok1 if o[0] >= ZK_MIN]
        if (ok0 or ok1) and ZR_ON and 1100 < P.scroll + mx < 1175 and 138 <= F <= 152 and v0 < ZR_V and tf > 3:
            return keys, "zrun"
        if ok0 and (not ok1 or tf <= LATE_TF):
            return self.do_jump(sol, ok0, res0, v0, keys)
        if ok0 or ok1:
            return keys, "wait%d" % tf
        fut = tf > 24
        for dd in range(2, min(tf, 26), 2):
            if fut:
                break
            for o in OPTS:
                r = sim_jump(sol, mx, F, v0, o[0], o[1], dd, fx)
                if r is not None and (r[2] >= 0 or tf > 14):
                    fut = True
                    break
        X = xs_table(v0, None)
        dist = X[min(tf, 149)] if tf is not None else 99
        stop = v0 * v0 / (2 * _BRK) + 3
        if fut:
            enemy_fail = any(str(FAIL.get(o, "")).startswith(("enemy", "landenemy")) for o in OPTS)
            if (enemy_fail and tf <= 14 and dist <= stop + 14 and self.fno >= self.nowait_until
                    and self.wait_total < WAIT_TOTAL):
                self.waiting = 1
                return {"Left"}, "brake"
            return keys, "wait%d" % tf
        if self.fno >= self.nowait_until and dist > stop:
            self.waiting = 1
            return {"Left"}, "brake"
        enemy_fail = any(str(FAIL.get(o, "")).startswith(("enemy", "landenemy")) for o in OPTS)
        if enemy_fail and v0 < 2.5 and tf <= 12 and self.hold < HOLD_MAX:
            self.hold += 1
            if self.hop_needed(F, mx):
                self.plan = dict(k=8, tb=None, t=0, stay=True)
                return {"z"}, "hop"
            return set(), "hold"
        if tf <= 3:
            return self.emergency(sol, mx, F, v0, fx, keys)
        return keys, "wait%d" % tf

    def air_control(self, sol, mx, my, keys):
        dy = 0.0
        if self.prev_my is not None:
            dy = my - self.prev_my
        self.prev_my = my
        tbk = tower_brake(self, sol, mx, my, dy)
        if tbk is not None:
            return tbk, "airTB"
        if dy < -0.5:
            keys = set(keys)
            keys.add("z")
            return keys, "air"
        if AB_ON and 1150 < self.P.scroll + mx < 1300:
            for e in globals()["ENEMIES"]:
                dxb = e[0] - (mx + 6)
                if e[4] == 8 and e[3] == 0 and e[2] < -0.3 and 0 < dxb < AB_D and 80 <= e[1] <= 106 and my + 16 < e[1] - 8:
                    return {"Left", "x"}, "airB"
        if dy >= 1 and my + 16 <= 225 and AIRB_ON:
            ab = self.bill_air(sol, mx, my, dy, keys)
            if ab is not None:
                return ab, "airB2"
        if dy < 1 or my + 16 > 225:
            return keys, "air"
        di = 0
        best = 99
        for i, v in enumerate(DY):
            if abs(v - dy) < best:
                best = abs(v - dy)
                di = i
        vx = max(1.0, self.speed())
        F = int(my) + 16
        global ENEMIES
        se = ENEMIES
        ENEMIES = []
        try:
            rr = sim_fall_b(sol, mx, F, vx, di, False)
            rl = sim_fall_b(sol, mx, F, vx, di, True)
        finally:
            ENEMIES = se
        runR = -1 if rr is None else ext_right(sol, rr[0] + 5, rr[1])
        runL = -1 if rl is None else ext_right(sol, rl[0] + 5, rl[1])
        if rr is None:
            if rl is not None and runL >= 12:
                return {"Left", "x"}, "airL"
            return keys, "air"
        if runR >= AIR_ROOM:
            return keys, "air"
        if rl is not None and runL > runR + 6 and runL >= 12:
            return {"Left", "x"}, "airL"
        return keys, "air"

    def bill_air(self, sol, mx, my, dy, keys):
        bills = [e for e in ENEMIES if e[4] == 8 and e[3] == 0 and e[2] < -0.3 and -24 < e[0] - (mx + 6) < 120]
        if not bills:
            return None
        di = 0
        best = 99
        for i, v in enumerate(DY):
            if abs(v - dy) < best:
                best = abs(v - dy)
                di = i
        v0 = max(0.0, self.speed())
        F0 = int(my) + 16
        res = []
        for name, ks, acc, lo, hi in (("R", {"Right", "x"}, 0.05, -1.5, 3.0), ("N", {"x"}, -0.02, -1.5, 3.0), ("L", {"Left", "x"}, -0.16, -1.5, 3.0)):
            x = mx
            v = v0
            F = F0
            sc = 60
            for t in range(1, 50):
                dyv = DY[min(di + t - 1, len(DY) - 1)]
                Fn = F + dyv
                v = max(lo, min(hi, v + acc))
                x += v
                xi = int(x)
                if xi + 12 >= 256 or xi < 0:
                    sc = 80
                    break
                stop = False
                for e in bills:
                    bx = e[0] + e[2] * t
                    if abs(x + 6 - bx) < 17 and Fn - 15 < e[1] + 1 and Fn - 1 > e[1] - 14:
                        if Fn - 1 <= e[1] - 14 + AIRB_STOMP:
                            sc = 5
                        else:
                            sc = 100
                        stop = True
                        break
                if stop:
                    break
                hit = None
                for r in range(F, Fn + 1):
                    if r < 240 and sol[r, xi + 4:xi + 7].any():
                        hit = r
                        break
                if hit is not None:
                    ext = ext_right(sol, x + 5, hit)
                    sc = 0 if ext >= 8 else 40
                    break
                F = Fn
                if F > 232:
                    sc = 90
                    break
            res.append((sc, name, ks))
        sr = res[0][0]
        if sr <= 5:
            return set(keys)
        bst = min(res, key=lambda r: r[0])
        if bst[0] < sr:
            return set(bst[2])
        return None

    def do_jump(self, sol, ok0, res0, v0, keys):
        cands = sorted(ok0, key=lambda o: (o[1] is not None, (PREFER_LEVEL and res0[o][1] > self.curF + 20),
                                           -min(res0[o][2], 30), o[0]))
        best = cands[0]
        n = 0
        for o in cands:
            xl, Fl, ext = res0[o]
            if ext >= 30 and xl + 40 > 250:
                best = o
                break
            n += 1
            if n > 5:
                break
            vl = 3.0 if o[1] is None else 1.5
            if cont_ok(sol, xl, Fl, vl):
                best = o
                break
        k, tb = best
        self.plan = dict(k=k, tb=tb, t=0)
        self.hold = 0
        keys = set(keys)
        keys.add("z")
        return keys, "jump%d%s" % (k, "" if tb is None else "b%d" % tb)

    def emergency(self, sol, mx, F, v0, fx, keys):
        global ENEMIES, LIFTS
        for e in ENEMIES:
            dxb = e[0] - (mx + 6)
            if e[3] == 0 and e[4] == 8 and e[2] < -0.3 and 0 < dxb < 90 and F - 50 < e[1] < F - 18 and BILL_WAIT:
                return {"Left"}, "billwait"
        LAND_STRICT[0] = False
        best = None
        for o in OPTS:
            r = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fx, 3)
            if r is not None and (best is None or r[2] > best[1][2]):
                best = (o, r)
        if best is None:
            se, sl = ENEMIES, LIFTS
            ENEMIES = []
            try:
                for o in OPTS:
                    r = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fx, 3)
                    if r is not None and (best is None or r[2] > best[1][2]):
                        best = (o, r)
            finally:
                ENEMIES = se
        if K32_ON and (best is None or best[0][0] == 24) and 1140 < self.P.scroll + mx < 1190 and 138 <= F <= 152:
            self.k32 = getattr(self, "k32", 0) + 1
            best = ((32, K32_TB), (0, 0, 0))
        if best is None:
            k, tb = 24, None
        else:
            k, tb = best[0]
        self.plan = dict(k=k, tb=tb, t=0)
        keys = set(keys)
        keys.add("z")
        return keys, "jumpE%d" % k

    def bill_ahead(self, F, mx):
        best = None
        for e in ENEMIES:
            if e[3] != 0 or e[4] != 8 or e[2] > -0.3:
                continue
            dx = e[0] - (mx + 6)
            if dx > 0 and F - 15 < e[1] + 1 and F - 1 > e[1] - e[5]:
                if best is None or dx < best:
                    best = dx
        return best

    def hop_needed(self, F, mx):
        for e in ENEMIES:
            if e[3] != 0 or e[4] not in (7, 8):
                continue
            dx = e[0] - (mx + 6)
            lead = 46 if e[4] == 8 else 38
            if 0 < dx < lead and e[2] < -0.3 and F - 15 < e[1] + 1 and F - 1 > e[1] - e[5]:
                return 2 if e[4] == 8 else 1
        return 0

    def wait_step(self, sol, mx, F, v0, tf, fx):
        self.waiting += 1
        self.wait_total += 1
        keys = {"Right", "x"}
        if self.waiting > WAIT_MAX:
            self.waiting = 0
            self.nowait_until = self.fno + 150
            return keys, "waitdone"
        if tf is None:
            self.waiting = 0
            return keys, "run"
        if v0 < 1.0:
            ok = {}
            for o in OPTS:
                r = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fx)
                if r is not None and r[2] >= 0:
                    ok[o] = r
            if ok:
                self.waiting = 0
                return self.do_jump(sol, list(ok), ok, v0, {"Right", "x"})
        found = False
        for dd in range(1, min(tf, 64), 3):
            if found:
                break
            for o in OPTS:
                r = sim_jump(sol, mx, F, v0, o[0], o[1], dd, fx)
                if r is not None and r[2] >= 0:
                    found = True
                    break
        if found and (tf > 12 or v0 > 1.0):
            if tf <= 12 and v0 > 1.0 and FLIP_FIX:
                return {"Left"}, "brake"
            self.waiting = 0
            return keys, "run"
        hn = self.hop_needed(F, mx)
        if hn:
            self.plan = dict(k=8, tb=None, t=0, stay=(hn == 1))
            return ({"z"} if hn == 1 else {"z", "Right", "x"}), "hop"
        if v0 > 0.4:
            return {"Left"}, "brake"
        xi = int(mx)
        if xi - 24 >= 0 and sol[F, xi + 5 - 24] and sol[F, xi + 5 - 12] and sol[F, xi + 5 - 5]:
            return {"Left"}, "retreat"
        return set(), "stand"
'''

SRC_B = r'''
import numpy as np


def kimg(img):
    return (img[:, :, 0].astype(np.uint16) << 8) | img[:, :, 1]


K_SKY = (104 << 8) | 136
K_RED = (248 << 8) | 56
K_BROWN = (172 << 8) | 124
K_LIME = (184 << 8) | 248
K_TAN = (240 << 8) | 208

ENEMIES = []
LIFTS = []
FAIL = {}
_DEBUG = False


def _fill(sm):
    out = []
    for i, s in enumerate(sm):
        out.append(s)
        if i + 1 < len(sm):
            out.append((s + sm[i + 1]) // 2)
    return out


RISE = {
    2: [0, 0, 6, 12, 18, 23, 27, 30, 32, 33, 34, 34],
    4: [0, 0, 6, 12, 18, 23, 28, 33, 36, 39, 41, 42, 42],
    6: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 45, 47, 49, 49],
    8: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 50, 53, 54, 55, 56],
    11: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 63, 64, 64],
    14: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 71, 71],
    18: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 73, 75, 77, 78, 79],
    24: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 73, 75, 77, 79, 81, 82, 83, 84, 85, 85],
    32: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 73, 75, 77, 79, 81, 82, 83, 84, 85, 85, 86, 86, 86],
}
STAND = {
    8: _fill([0, 5, 14, 23, 30, 37, 42, 44]),
    14: _fill([0, 5, 14, 23, 30, 37, 43, 48, 53, 55]),
    24: _fill([0, 5, 14, 23, 30, 37, 43, 48, 53, 57, 60, 62, 63, 64]),
    32: _fill([0, 5, 14, 23, 30, 37, 43, 48, 53, 57, 60, 62, 63, 64, 64]),
}
KS = (2, 4, 6, 8, 11, 14, 18, 24, 32)
OPTS = [(k, None) for k in KS]
OPTS_FUT = [(8, None), (14, None), (24, None), (32, None)]
DY = [1, 2, 2, 4, 4, 5]
_VMAX = 3.0
_ACC = 0.07
_BRK = 0.16
_LOOK = 70
H_TRIG = 19
MIN_EXT = 28
WAIT_MAX = 260
EMARG = 3
AIR_ROOM = 26
_RMARG = 0
ROOM_BRAKE = 44
STOMP_DROP = 400
BILL_GATE = False
GATE_X0 = 1080
GATE_X1 = 1235
GATE_BACK = 999
RETREAT = True
LIFT_LEAD = 8
LIFT_EDGE = 5
LIFT_MARG = 7
LIFT_EXT = 20
LRUN_ACC = 0.09
LIFT_BRAKE_X0 = 1250
REFLEX = False
REFLEX_DX = 30
REFLEX_K = 14
WAIT_EDGE = False
EDGE_X0 = 1262
EDGE_WX = 1304
HOP_EXT = 18
GBH_ON = True
EDGE_GUARD = 6
LS_GAP = 52
LS_K = 14
LS_KNT = 32
LS_RUNUP = True
LSTAY_ON = True
KHOP_ON = True
KHOP_D0 = 28
KHOP_D1 = 66
KHOP_K = 8
LS_GO2 = 1514
LS_GOWAIT2 = 100000
K_GROUND_KOOPA = True
KOOPA_STOMP = True
EM_CLIMB = True
GK_N = 10
LS_AFTER = 100000
LS_GOWAIT = 100000
LS_WX = 1399
LS_TRAP = 1305
LS_RELAX = False
LS_GO = 1390
DBG = None
LIFT_GAP = 66
LIFT_GAP2 = 100
LIFT_EDGEJ = 4
EDGE_PAD = 14
LAND_GAP = 28
STAND_VMAX = 1.8
LIFT_WAIT_MAX = 500
BILL_X = 100
BILL_MAXWAIT = 140
VPEN = 1.0
VREF = 2.0
GRELAX_MAX = 30
GRELAX_T = 250
SEEK_ON = False
SEEK_T = 300
SEEK_ROOM = 12
TVX_K = 0.0
TVX_G = 0.0
LK3 = 100000
LK4 = 124
TVX_MAX = 9.0
FEAS_ON = False
CAPK_ON = False
LK0 = 64
LK1 = 80
LK2 = 96
CAPK_D = 56
LBACK_ON = False
LBACK_X = 8
LBACK_N = 40
FEAS_T = 260
VCLAMP = 99.0
STEER_MARGIN = 0.5
CAPZ0 = 1575
CAPZ1 = 1650
LB_D = 230
LT_M = 12
NG_FIX = True
NG_N = 6
NG_DY = 9
SF_ON = False
SF_EXT = 14
LB_LOW = 32
G2_X0 = 1560
G2_X1 = 1665
G2_D = 150
G2_MAX = 120
SG_ON = True
LOWK_ON = False
NOLEFT = False
LOWK_V = 1.3
LOWK_K = 24
SG_MAX = 150
SG_PRE = 12
SG_POST = 10
ZP_ON = True
ZP_X0 = 1240
ZP_T = 2
ZP_D0 = 4
ZP_D1 = 60
ZP_MAX = 8
ZP_HOLD = 5
ZP_W = 44
EV_ON = False
WB_ON = False
LW_A = -11
LW_B = -2
LW_LG = 6
WB_X0 = 1650
WB_X1 = 1800
WB_MAX = 40
LF_ON = True
LF_D = 66
LF_K = 24
EV_A0 = 1215
EV_A1 = 1340
EV_B0 = 1760
EV_B1 = 1960
EV_LEAD = 26
EV_M = 0


def rise_at(tab, t):
    if t < len(tab):
        return tab[t]
    r = tab[-1]
    for i in range(t - len(tab) + 1):
        r -= DY[min(i, len(DY) - 1)]
    return r


_XC = {}


def xs_table(v0, tbrake, vmax=None):
    if vmax is None:
        vmax = _VMAX
    key = (round(v0, 1), tbrake, vmax)
    X = _XC.get(key)
    if X is None:
        X = [0.0]
        v = v0
        for i in range(1, 150):
            if tbrake is not None and i > tbrake:
                v = max(v - _BRK, -1.5)
            else:
                v = min(vmax, v0 + _ACC * i)
            X.append(X[-1] + v)
        _XC[key] = X
    return X


class Perc:
    def __init__(self):
        self.mx = 40.0
        self.my = 192.0
        self.found = False
        self.scroll = 0
        self.prev = None

    def mario(self, K, lifts=()):
        m = (K == K_RED) | (K == K_BROWN)
        m[:32] = False
        for (top, x0, x1) in lifts:
            m[max(top - 1, 0):top + 9, x0:x1 + 1] = False
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            self.found = False
            return None
        sel = (np.abs(xs - (self.mx + 8)) < 28) & (ys > self.my - 30) & (ys < self.my + 40)
        if sel.sum() < 4:
            self.found = False
            return None
        ys = ys[sel]
        xs = xs[sel]
        cx = int(np.median(xs))
        cy = int(np.median(ys))
        sel = (np.abs(xs - cx) <= 9) & (np.abs(ys - cy) <= 9)
        ys = ys[sel]
        xs = xs[sel]
        self.found = True
        x0 = xs.min()
        y1 = ys.max()
        self.mx = float(x0 if xs.max() - x0 <= 13 else xs.max() - 12)
        self.my = float(y1 - 15)
        return self.mx, self.my

    def scroll_est(self, K):
        if self.prev is None:
            self.prev = K
            return 0
        a = self.prev[40:232:3]
        b = K[40:232:3]
        nz = a != K_SKY
        best, bs = -1, 0
        for s in range(0, 6):
            if s == 0:
                sc = ((a == b) & nz).sum()
            else:
                sc = ((a[:, s:] == b[:, :-s]) & nz[:, s:]).sum()
            if sc > best:
                best, bs = sc, s
        self.prev = K
        return bs


def find_lifts(K):
    red = K == K_RED
    red[:32] = False
    cnt = red.sum(1)
    rows = np.nonzero(cnt >= 10)[0]
    out = []
    last = -99
    for y in rows.tolist():
        if y - last <= 8 and out:
            last = y
            continue
        r = red[y].astype(np.int8)
        d = np.diff(np.concatenate(([0], r, [0])))
        st = np.nonzero(d == 1)[0]
        en = np.nonzero(d == -1)[0]
        ln = en - st
        i = int(np.argmax(ln))
        if ln[i] >= 24 or (en[i] >= 256 and ln[i] >= 10 and 100 <= y <= 200):
            out.append((y - 1, int(st[i]), int(en[i]) - 1))
            last = y
    return out


def solid_mask(K, scroll=0):
    sol = K == K_LIME
    h = np.zeros_like(sol)
    h[:, 2:] |= sol[:, :-2]
    h[:, :-2] |= sol[:, 2:]
    h[:, 1:] |= sol[:, :-1]
    h[:, :-1] |= sol[:, 1:]
    below = np.zeros_like(sol)
    below[:-1] |= sol[1:]
    below[:-2] |= sol[2:]
    sol = sol | (h & below)
    if scroll < 330:
        nz = K[208] != K_SKY
        if nz[0]:
            z = np.nonzero(~nz)[0]
            end = int(z[0]) if len(z) else 256
            sol[208:, :end] = True
    return sol


def find_clusters(m, minpts=5, cs=8):
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return []
    cells = {}
    for y, x in zip(ys.tolist(), xs.tolist()):
        cells.setdefault((y // cs, x // cs), []).append((y, x))
    seen = set()
    out = []
    for key in list(cells):
        if key in seen:
            continue
        stack = [key]
        seen.add(key)
        pts = []
        while stack:
            c = stack.pop()
            pts.extend(cells[c])
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    n = (c[0] + dy, c[1] + dx)
                    if n in cells and n not in seen:
                        seen.add(n)
                        stack.append(n)
        if len(pts) < minpts:
            continue
        yy = [q[0] for q in pts]
        xx = [q[1] for q in pts]
        out.append((min(xx), max(xx), min(yy), max(yy), len(pts)))
    return out


def en_pos(e, T):
    ex, ey, ev, vy, hw, h, y0, y1, st = e
    x = ex + ev * T
    if vy == 0 or y1 - y0 < 2:
        return x, ey
    L = float(y1 - y0)
    u = ((ey - y0) + vy * T) % (2 * L)
    if u > L:
        u = 2 * L - u
    return x, y0 + u


def run_only(sol, mx, F, v0, N=_LOOK, deep=True):
    X = xs_table(v0, None)
    for t in range(1, N + 1):
        x = mx + X[t]
        xi = int(x)
        if xi + 14 >= 256:
            return None, None
        for e in ENEMIES:
            exx, yb = en_pos(e, t)
            if abs(x + 6 - exx) < e[4] + 6 + EMARG and F - 15 < yb + 1 and F - 1 > yb - e[5]:
                return t, int(exx)
        if F - 14 >= 0 and sol[F - 14:F - 2, xi + 12].any():
            return t, xi + 12
        if not sol[F, xi + 5]:
            vend = min(_VMAX, v0 + _ACC * t)
            r = sim_fall(sol, x - vend, F, vend, 0)
            if r is None:
                return t, xi + 6
            if r[1] - F > 100:
                return t, xi + 6
            if ext_right(sol, r[0] + 5, r[1]) < MIN_EXT:
                return t, xi + 6
            if deep and not cont_ok(sol, r[0], r[1], vend):
                return t, xi + 6
            return None, None
    return None, None


def sim_fall(sol, x, F, vx, di=0, maxt=60):
    for t in range(maxt):
        dy = DY[min(di + t, len(DY) - 1)]
        Fn = F + dy
        x += vx
        xi = int(x)
        if xi + 12 >= 256 or xi < 0:
            return None
        hit = None
        for r in range(F, Fn + 1):
            if r < 240 and sol[r, xi + 4:xi + 7].any():
                hit = r
                break
        if hit is not None:
            return x, hit
        F = Fn
        if F > 232:
            return None
    return None


def sim_fall_b(sol, x, F, vx, di, brake, maxt=60):
    F0s = F
    for t in range(maxt):
        dy = DY[min(di + t, len(DY) - 1)]
        Fn = F + dy
        x += vx
        if brake:
            vx = max(vx - _BRK, -1.5)
        xi = int(x)
        if xi + 12 >= 256 or xi < 0:
            return None
        for e in ENEMIES:
            exx, yb = en_pos(e, t + 1)
            if abs(x + 6 - exx) < e[4] + 6 + EMARG and Fn - 15 < yb + 1 and Fn - 1 > yb - e[5]:
                if e[8] and e[4] == 7 and Fn - 1 <= yb - 5 and yb <= F0s + STOMP_DROP:
                    return (x, Fn)
                if e[8] == 2 and Fn - 1 <= yb - 18 and yb <= F0s + 40:
                    return (x, Fn)
                return None
        hit = None
        for r in range(F, Fn + 1):
            if r < 240 and sol[r, xi + 4:xi + 7].any():
                hit = r
                break
        if hit is not None:
            return x, hit
        F = Fn
        if F > 232:
            return None
    return None


def ext_right(sol, xc, F, cap=40):
    n = 0
    xi = int(xc)
    while n < cap and xi + n < 255 and sol[F, xi + n]:
        n += 1
    if xi + n >= 255:
        return cap
    return n


def ext_left(sol, xc, F, cap=60):
    n = 0
    xi = int(xc)
    while n < cap and xi - n > 0 and sol[F, xi - n]:
        n += 1
    return n


def sim_jump(sol, mx, F0, v0, k, tb, delay, fail_x, min_ext=MIN_EXT):
    vt = min(_VMAX, v0 + _ACC * (delay + 1))
    if vt < 2.2:
        tab = STAND.get(k)
        if tab is None:
            ks = [q for q in STAND if q <= k]
            if not ks:
                FAIL[(k, tb)] = "nostand"
                return None
            tab = STAND[max(ks)]
    else:
        tab = RISE[k]
    X = xs_table(v0, None if tb is None else delay + tb, STAND_VMAX if vt < 2.2 else None)
    xd = int(mx + X[delay + 1])
    if xd + 12 >= 256 or xd < 0 or not sol[F0, xd + 5]:
        FAIL[(k, tb)] = "nosupport"
        return None
    Fcur = F0
    prev_r = 0
    t = 0
    landed = None
    while t < 90:
        t += 1
        T = t + delay
        x = mx + X[T]
        xi = int(x)
        if xi < 0:
            FAIL[(k, tb)] = "left"
            return None
        if xi + 13 >= 256:
            FAIL[(k, tb)] = "edge t%d" % t
            return (x, Fcur, -1)
        r = rise_at(tab, t - 1)
        if r > 6:
            r -= _RMARG
        Fn = F0 - r
        for (lx0, lx1, ltop, lvy, lvx) in LIFTS:
            lt = ltop + lvy * T
            lx0 = lx0 + lvx * T
            lx1 = lx1 + lvx * T
            if xi + 12 >= lx0 - 1 and xi + 2 <= lx1 + 1:
                if r < prev_r and Fcur <= lt + 2 and Fn >= lt - 1 and xi + 7 >= lx0 + LIFT_MARG and xi + 4 <= lx1 - 3:
                    le = lx1 - (xi + 5)
                    if le < LIFT_EXT:
                        FAIL[(k, tb)] = "liftext%d" % le
                        return None
                    return (x, int(lt), 30 if le > 8 else le)
                if Fn - 17 < lt + 10 and Fn > lt - 2:
                    FAIL[(k, tb)] = "lift t%d" % t
                    return None
        for e in ENEMIES:
            exx, yb = en_pos(e, T)
            if abs(x + 6 - exx) < e[4] + 6 + EMARG and Fn - 15 < yb + 1 and Fn - 1 > yb - e[5]:
                if e[8] and e[4] == 7 and r < prev_r and Fn - 1 <= yb - 5 and yb <= F0 + STOMP_DROP:
                    return (x, Fn, 29)
                if e[8] == 2 and r < prev_r and Fn - 1 <= yb - 18 and yb <= F0 + 40:
                    return (x, Fn, 29)
                FAIL[(k, tb)] = "enemy t%d" % t
                return None
        if r >= prev_r and t < len(tab) + 1:
            if Fn - 14 >= 0 and sol[Fn - 14:Fn - 2, xi + 12].any():
                FAIL[(k, tb)] = "wallrise t%d" % t
                return None
            if Fn - 17 >= 0 and sol[Fn - 17, xi + 2:xi + 11].any():
                FAIL[(k, tb)] = "head t%d" % t
                return None
        else:
            if Fn - 14 >= 0 and sol[Fn - 14:Fn - 2, xi + 12].any():
                FAIL[(k, tb)] = "walldesc t%d" % t
                return None
            hit = None
            for rr in range(Fcur, Fn + 1):
                if rr < 240 and sol[rr, xi + 4:xi + 7].any():
                    hit = rr
                    break
            if hit is not None:
                Fcur = hit
                landed = (x, hit)
                break
        prev_r = r
        Fcur = Fn
        if Fcur > 232:
            FAIL[(k, tb)] = "fell t%d" % t
            return None
    if landed is None:
        FAIL[(k, tb)] = "nolanding"
        return None
    for e in ENEMIES:
        if e[4] != 8 and e[5] == 14 and e[2] < 0.3:
            exx, yb = en_pos(e, T)
            if abs(yb - landed[1]) <= 6 and -10 < exx - (landed[0] + 6) < LAND_GAP:
                FAIL[(k, tb)] = "landenemy"
                return None
        elif e[4] == 9 and e[5] == 26 and e[2] < 0.3 and abs(e[1] - landed[1]) <= 8:
            for dT in (0, 8):
                exx = e[0] + e[2] * (T + dT)
                if -14 < exx - (landed[0] + 6) < LAND_GAP + 14:
                    FAIL[(k, tb)] = "landenemy"
                    return None
    xl, Fl = landed
    if xl + 5 < fail_x + 2:
        FAIL[(k, tb)] = "short"
        return None
    ext = ext_right(sol, xl + 5, Fl)
    if ext < min_ext:
        FAIL[(k, tb)] = "ext%d" % ext
        return None
    return xl, Fl, ext


def cont_ok(sol, xl, Fl, vl):
    global ENEMIES, LIFTS
    se, sl = ENEMIES, LIFTS
    ENEMIES, LIFTS = [], []
    try:
        tf2, fx2 = run_only(sol, xl, Fl, vl, _LOOK, False)
        if tf2 is None:
            return True
        for d in range(0, min(tf2, 22) + 1, 2):
            for (k, tb) in OPTS[:len(KS)]:
                r = sim_jump(sol, xl, Fl, vl, k, tb, d, fx2)
                if r is not None and r[2] >= 0:
                    return True
        return False
    finally:
        ENEMIES, LIFTS = se, sl


def find_bills(K):
    blk = (K[32:] == 0).astype(np.int32)
    ii = np.zeros((blk.shape[0] + 1, blk.shape[1] + 1), np.int32)
    ii[1:, 1:] = blk.cumsum(0).cumsum(1)
    S = ii[8:, 8:] - ii[:-8, 8:] - ii[8:, :-8] + ii[:-8, :-8]
    m = S >= 34
    out = []
    if not m.any():
        return out
    for (x0, x1, y0, y1, n) in find_clusters(m, 1):
        w = x1 - x0 + 8
        h = y1 - y0 + 8
        if 8 <= w <= 26 and 6 <= h <= 22:
            out.append(((x0 + x1 + 8) / 2.0, y1 + 8 + 32))
    return out


def find_tan(K, excl=()):
    m = K == K_TAN
    m[:32] = False
    for (bx, by) in excl:
        m[max(int(by) - 17, 0):int(by) + 3, max(int(bx) - 11, 0):int(bx) + 12] = False
    out = []
    for (x0, x1, y0, y1, n) in find_clusters(m, 5, 4):
        out.append(((x0 + x1) / 2.0, y1 + 2))
    return out


TB_ON = True
TB_T = 0
TB_M = 8


def tower_brake(pol, sol, mx, my, dy):
    """airborne above the narrow tower top (wx 1240-1340) with speed: brake (Left) so a bounce/hop does not carry Mario over the edge."""
    wx = pol.P.scroll + mx
    feet = int(my) + 16
    if not (TB_ON and 1240 < wx < 1340 and feet <= 100):
        return None
    if pol.speed() < 1.2:
        return None
    xl = int(mx) + 5 - 14
    if xl < 0:
        return None
    surf = None
    for r in range(max(feet - 2, 0), 112):
        if sol[r, xl]:
            surf = r
            break
    if surf is None or surf < 90 or surf > 104:
        return None
    er = ext_right(sol, mx + 5, surf, 200)
    h = float(surf - feet)
    if dy >= 1:
        di = 0
        bst = 99
        for i, q in enumerate(DY):
            if abs(q - dy) < bst:
                bst = abs(q - dy)
                di = i
        rt = 0.0
    else:
        sp = max(0.0, -dy)
        rt = sp / 0.4
        h += sp * sp / 0.8
        di = 0
    hh = 0.0
    t = 0
    while hh < h and t < 60:
        hh += DY[min(di + t, len(DY) - 1)]
        t += 1
    T = rt + t + TB_T
    if 4 <= er and pol.speed() * T > er - TB_M:
        ks = {"Left", "x"}
        if dy < -0.5:
            ks.add("z")
        return ks
    return None


class Policy:
    def __init__(self):
        self.P = Perc()
        self.plan = None
        self.prev_en = []
        self.prev_lifts = []
        self.fno = 0
        self.lhist = {}
        self.wxh = []
        self.waiting = 0
        self.prev_my = None
        self.last_why = ""
        self.nowait_until = 0
        self.curF = 0
        self.gate_n = 0
        self.retreat_n = 0
        self.on_lift = False
        self.lift_wait = 0
        self.lrun = 0
        self.fc = 0
        self.wxf = []
        self.wb_n = 0

    def track(self, K, mx, my, lifts):
        global ENEMIES, LIFTS
        P = self.P
        F = int(my) + 16
        cands = []
        bills = find_bills(K)
        for (cx, ey) in find_tan(K, bills):
            if abs(cx - (mx + 6)) < 4 and abs(ey - F) < 6:
                continue
            cands.append(["g", cx, ey, 7, 14])
        for (cx, ey) in bills:
            cands.append(["b", cx, ey, 8, 14])
        red = K == K_RED
        red[:32] = False
        for (top, x0, x1) in lifts:
            red[max(top - 1, 0):top + 9, x0:x1 + 1] = False
        red[max(int(my) - 4, 0):int(my) + 20, max(int(mx) - 5, 0):int(mx) + 20] = False
        for (x0, x1, y0, y1, n) in find_clusters(red, 6):
            if x1 >= 250 or (x1 - x0 >= 12 and y1 - y0 <= 9):
                continue
            cands.append(["r", (x0 + x1) / 2.0, y1 + 8, 9, 26])
        cur = []
        used = set()
        for kind, cx, ey, hw, h in cands:
            wx = cx + P.scroll
            m = None
            for i, pe in enumerate(self.prev_en):
                if i in used or pe["kind"] != kind:
                    continue
                if abs(pe["wx"] - wx) < 6 and abs(pe["y"] - ey) < 10:
                    m = i
                    break
            if m is None:
                e = dict(kind=kind, wx=wx, y=ey, vx=(-0.6 if kind == "g" else (-1.5 if kind == "b" else 0.0)), vy=0.0, ymin=ey, ymax=ey, n=1, hw=hw, h=h)
                if kind == "b" and cx > 226:
                    self.note_spawn()
            else:
                used.add(m)
                pe = self.prev_en[m]
                e = dict(pe)
                e["vx"] = 0.7 * pe["vx"] + 0.3 * (wx - pe["wx"])
                if kind == "r":
                    e["vy"] = 0.8 * pe["vy"] + 0.2 * (ey - pe["y"])
                e["wx"] = wx
                e["y"] = ey
                e["ymin"] = min(pe["ymin"], ey)
                e["ymax"] = max(pe["ymax"], ey)
                e["n"] = pe["n"] + 1
            cur.append(e)
        self.prev_en = cur
        out = []
        for e in cur:
            ex = e["wx"] - P.scroll
            if e["kind"] in ("g", "b"):
                out.append((ex, e["y"], max(-2.5, min(2.5, e["vx"])), 0.0, e["hw"], 14, e["y"], e["y"], True))
            else:
                y0, y1 = e["ymin"] - 3, e["ymax"] + 3
                vy = e["vy"]
                if abs(vy) < 0.12:
                    vy = 0.0 if e["n"] > 80 else 0.5
                stf = True
                if K_GROUND_KOOPA and e["n"] >= GK_N and e["ymax"] - e["ymin"] <= 3:
                    vy = 0.0
                    if KOOPA_STOMP and P.scroll + ex > 1560:
                        stf = 2
                if vy > 0:
                    y1 = max(y1, e["y"] + 45)
                elif vy < 0:
                    y0 = min(y0, e["y"] - 45)
                else:
                    y1 = y0 + 1
                out.append((ex, e["y"], max(-2.0, min(2.0, e["vx"])), vy, 9, 26, y0, y1, stf))
        ENEMIES = out
        self.fno += 1
        LIFTS = []
        for (top, x0, x1) in lifts:
            key = int((x0 + P.scroll) // 12)
            h = None
            for kk in (key, key - 1, key + 1):
                if kk in self.lhist:
                    h = self.lhist[kk]
                    break
            if h is None:
                h = []
            h.append((self.fno, top, x0 + P.scroll))
            del h[:-24]
            for kk in (key - 1, key + 1):
                self.lhist.pop(kk, None)
            self.lhist[key] = h
            vy = 0.0
            vx = 0.0
            if len(h) >= 6:
                h0 = h[-10] if len(h) >= 10 else h[0]
                dtf = float(h[-1][0] - h0[0])
                vy = (h[-1][1] - h0[1]) / dtf
                vy = max(-1.5, min(1.5, vy))
                vx = (h[-1][2] - h0[2]) / dtf
                vx = max(-1.6, min(1.6, vx))
                if abs(vx) < 0.12:
                    vx = 0.0
            LIFTS.append((x0, x1, top, vy, vx))

    def speed(self):
        h = self.wxh
        if len(h) < 5:
            return 0.0
        v = (h[-1] - h[-5]) / 4.0
        return max(0.0, min(_VMAX, v))

    def decide(self, img):
        P = self.P
        self.fc += 1
        K = kimg(img)
        d = P.scroll_est(K)
        P.scroll += d
        lifts = find_lifts(K)
        mp = P.mario(K, lifts)
        keys = {"Right", "x"}
        if mp is None:
            pl = self.plan
            if pl is not None:
                pl["t"] += 1
                k2 = {"Right", "x"}
                if pl.get("stay"):
                    k2 = set()
                if pl["t"] < pl["k"]:
                    k2.add("z")
                if pl["tb"] is not None and pl["t"] >= pl["tb"]:
                    k2.discard("Right")
                    k2.add("Left")
                return k2, "air"
            return keys, None
        mx, my = mp
        self.wxh.append(P.scroll + mx)
        del self.wxh[:-8]
        self.wxf.append((self.fc, P.scroll + mx))
        del self.wxf[:-12]
        self.track(K, mx, my, lifts)
        F = int(my) + 16
        sol = solid_mask(K, P.scroll)
        grounded = False
        self.on_lift = False
        for r_ in (F, F - 1, F + 1, F + 2):
            if 0 <= r_ < 240 and sol[r_, int(mx) + 4:int(mx) + 7].any():
                F = r_
                grounded = True
                break
        if not grounded:
            global LIFTS
            for lf in LIFTS:
                lx0, lx1, ltop, lvy, lvx = lf
                if abs(F - ltop) <= 3 and lx0 - LT_M <= mx + 5 <= lx1 + LT_M:
                    grounded = True
                    self.on_lift = True
                    F = int(ltop)
                    sol = sol.copy()
                    sol[F:F + 8, max(lx0, 0):lx1 + 1] = True
                    LIFTS = [l for l in LIFTS if l is not lf]
                    break
        if self.on_lift:
            self.ll = (F, self.fc)
            self.on_lift_n = getattr(self, "on_lift_n", 0) + 1
        elif (getattr(self, "on_lift_n", 0) > LS_AFTER or P.scroll >= LS_TRAP) and (not grounded) and self.plan is None and getattr(self, "ll", None) is not None:
            if self.fc - self.ll[1] <= 40 and -4 <= F - self.ll[0] <= 10 and not LIFTS:
                return set(), "lsocc"

        if self.plan is not None:
            pl = self.plan
            pl["t"] += 1
            done_air = pl["t"] >= pl["k"] + 2 and (pl["tb"] is None or pl["t"] >= pl["tb"] + 14)
            if pl.get("hold") or pl.get("force"):
                done_air = False
            if pl.get("force") and pl["t"] > 110:
                done_air = True
            if pl["t"] >= 6 and (grounded or done_air):
                self.plan = None
                if not grounded:
                    return self.air_control(self.with_lifts(sol), mx, my, keys)
            else:
                if pl.get("stay"):
                    keys = set()
                    if pl.get("bk"):
                        keys = {"Left"}
                    if pl.get("lstay") and LIFTS and LSTAY_ON:
                        cl = min(LIFTS, key=lambda l: abs((l[0] + l[1]) / 2.0 - (mx + 5)))
                        if abs((cl[0] + cl[1]) / 2.0 - (mx + 5)) < 40:
                            if cl[4] < -0.5:
                                keys = {"Left"}
                            elif cl[4] > 0.5:
                                keys = {"Right"}
                if pl["t"] < pl["k"]:
                    keys.add("z")
                if pl["tb"] is not None and pl["t"] >= pl["tb"]:
                    keys.discard("Right")
                    keys.add("Left")
                if pl.get("hold"):
                    sk = None
                    if "F0" in pl:
                        sk = self.steer_plan(pl, mx)
                    else:
                        dyh = (my - self.prev_my) if self.prev_my is not None else 0.0
                        self.prev_my = my
                        if dyh >= 1:
                            sk = self.steer(mx, my, dyh)
                    if sk is not None:
                        keys = sk
                        if pl["t"] < pl["k"]:
                            keys.add("z")
                return keys, "air"
        if not grounded:
            ll_ = getattr(self, "ll", None)
            if (NG_FIX and ll_ is not None and self.fc - ll_[1] <= NG_N and abs(F - ll_[0]) <= NG_DY
                    and P.scroll + mx > 1330 and self.plan is None and self.prev_my is None):
                self.prev_my = None
                return {"Left"}, "ngfix"
            return self.air_control(self.with_lifts(sol), mx, my, keys)
        self.prev_my = None
        zs_ = getattr(self, "zp_start", None)
        if zs_ is not None:
            age_ = self.fc - zs_
            if age_ <= ZP_HOLD:
                return {"z"}, "zhold"
            self.zp_start = None
            self.zp_rise = 0
            if age_ == ZP_HOLD + 1:
                return set(), "zrel"

        v0 = self.speed()
        self.curF = F
        if self.waiting == 0 and v0 > 2.7:
            v0 = 3.0
        tf, fx = run_only(sol, mx, F, v0)

        # ground Koopa walking towards Mario on a tower top: hop over it (hold Right) instead of the planner's tiny hops
        if KHOP_ON and (not self.on_lift) and P.scroll + mx > 1650 and self.waiting == 0 and v0 >= 0.5:
            for e in ENEMIES:
                if e[4] == 9 and e[5] == 26 and e[3] == 0 and e[8] == 2 and e[2] <= 0.3:
                    dxk = e[0] - (mx + 6)
                    if KHOP_D0 < dxk < KHOP_D1 and abs(e[1] - F) <= 14:
                        self.plan = dict(k=KHOP_K, tb=None, t=0, force=True)
                        return {"Right", "x", "z"}, "khop"
        # generic Bullet Bill dodge (B region): hop in place when a Bill at body height is 14-50 px ahead
        if GBH_ON and (not self.on_lift) and P.scroll + mx > 1285 and self.waiting == 0:
            for e in ENEMIES:
                if e[4] == 8 and e[3] == 0 and e[2] < -0.3:
                    dxb = e[0] - (mx + 6)
                    if 14 < dxb < 50 and F - 15 < e[1] + 1 and F - 1 > e[1] - e[5]:
                        if G2_X0 < P.scroll + mx < G2_X1 and v0 >= 1.2 and ext_right(sol, mx + 5, F, 200) <= 90:
                            continue
                        bk_ = bool(G2_X0 < P.scroll + mx < G2_X1 and v0 > 0.4)
                        self.plan = dict(k=8, tb=None, t=0, stay=True, bk=bk_)
                        return ({"z", "Left"} if bk_ else {"z"}), "billhop3"
        # a Bullet Bill flying at head height towards the tower edge: wait for it to pass before hopping
        if (not self.on_lift) and P.scroll + mx > 1230 and F <= 105 and self.waiting == 0 and getattr(self, "bw", 0) < 400:
            for e in ENEMIES:
                if e[4] == 8 and e[3] == 0 and e[2] < -0.3:
                    dxb = e[0] - (mx + 6)
                    if -20 < dxb < 170 and F - 95 < e[1] < F + 12:
                        if ext_right(sol, mx + 5, F, 200) <= 60:
                            if 12 < dxb < 50 and F - 15 < e[1] + 1 and F - 1 > e[1] - e[5]:
                                self.plan = dict(k=8, tb=None, t=0, stay=True)
                                return {"z"}, "billhop2"
                            self.bw = getattr(self, "bw", 0) + 1
                            return ({"Left"} if v0 > 0.5 else set()), "billwait2"
                        break
        # measured hop from the tower edge onto the first lift: jump k14, hold Left from t=4 until landing
        if (not self.on_lift) and P.scroll + mx > 1250 and F <= 105 and v0 >= 2.0:
            if any(l[1] > mx + 20 and l[0] > F + 10 for l in lifts):
                if ext_right(sol, mx + 5, F, 200) <= HOP_EXT:
                    self.plan = dict(k=14, tb=4, t=0, hold=True, F0=F, fast=(v0 >= 2.2))
                    self.waiting = 0
                    keys.add("z")
                    return keys, "hop1"
        if 1250 < P.scroll + mx < 1420:
            self.dbg = (round(v0, 2), ext_right(sol, mx + 5, F, 200), F, int(self.on_lift), len(lifts))
        # fallback: tower edge reached without hop1 firing (slow / no lift visible yet): jump anyway instead of walking off
        if (not self.on_lift) and 1250 < P.scroll + mx < 1420 and F <= 105 and v0 >= 0.5 and ext_right(sol, mx + 5, F, 200) <= 10:
            self.plan = dict(k=14, tb=None, t=0, hold=True, F0=F, fast=(v0 >= 2.2))
            self.waiting = 0
            keys.add("z")
            return keys, "hopF"

        if self.on_lift and self.waiting == 0 and (getattr(self, "on_lift_n", 0) > LS_AFTER or P.scroll >= LS_TRAP or getattr(self, "on_lift_n", 0) > 150) and self.lift_wait < 3000:
            ls = self.lift_simple(mx, F, lifts)
            if ls is not None:
                return ls
        if self.on_lift and self.waiting == 0 and self.lift_wait < 900:
            lr = self.lift_hop2(mx, F, lifts, sol)
            if lr is not None:
                return lr
        if self.on_lift and self.waiting == 0 and self.lift_wait < LIFT_WAIT_MAX:
            self.lift_wait += 1
            if self.last_why == "liftRun":
                self.lrun += 1
                v0 = min(3.0, LRUN_ACC * self.lrun)
            else:
                self.lrun = 0
                if self.last_why in ("lifthold", "liftL", "liftB", "liftC", "liftwait"):
                    v0 = 0.0
                elif self.last_why == "liftR":
                    v0 = min(v0, 1.2)
            fxx = fx if fx is not None else int(mx) + 30
            res0 = {}
            ok0 = []
            for o in OPTS:
                r = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fxx)
                res0[o] = r
                if r is not None and r[2] >= 0:
                    ok0.append(o)
            if ok0 and self.P.scroll + mx < 1000 and self.lift_wait < 200 and F < 168:
                ok0 = [o for o in ok0 if res0[o][1] < 190]
            if ok0:
                return self.do_jump(sol, ok0, res0, v0, keys)
            er = ext_right(sol, mx + 5, F, 200)
            if er < LIFT_EDGE and ext_left(sol, mx + 5, F, 200) > 9:
                return {"Left"}, "liftL"
            for dd in range(3, 28, 4):
                for o in OPTS_FUT:
                    r = sim_jump(sol, mx, F, v0, o[0], o[1], dd, fxx)
                    if r is not None and r[2] >= 0:
                        return {"Right", "x"}, "liftRun"
            for l in lifts:
                if abs(F - l[0]) <= 3 and l[1] - LT_M <= mx + 5 <= l[2] + LT_M:
                    if SEEK_ON and self.lift_wait > SEEK_T:
                        room = l[2] - (mx + 5)
                        if room > SEEK_ROOM:
                            return {"Right"}, "liftseek"
                        if room < 4:
                            return {"Left"}, "liftseek"
                        return set(), "liftseek"
                    rel = (mx + 5) - (l[1] + l[2]) / 2.0
                    lp = getattr(self, "lprev", None)
                    vrel = max(-3.0, min(3.0, rel - lp)) if lp is not None else 0.0
                    self.lprev = rel
                    rp = rel + vrel * abs(vrel) / 0.2
                    if rp < -11:
                        return {"Right"}, "liftC"
                    if rp > -2:
                        return {"Left"}, "liftC"
                    return set(), "liftC"
            return set(), "lifthold"

        if self.waiting > 0:
            return self.wait_step(sol, mx, F, v0, tf, fx)

        if tf is None or tf > H_TRIG:
            return keys, "run"

        res0 = {}
        res1 = {}
        for o in OPTS:
            res0[o] = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fx)
            res1[o] = sim_jump(sol, mx, F, v0, o[0], o[1], 1, fx)
        ok0 = [o for o in OPTS if res0[o] is not None and res0[o][2] >= 0]
        ok1 = [o for o in OPTS if res1[o] is not None and res1[o][2] >= 0]
        if ok0 and (not ok1 or tf <= 2):
            return self.do_jump(sol, ok0, res0, v0, keys)
        if ok0 or ok1:
            return keys, "wait%d" % tf
        fut = tf > 24
        for dd in range(2, min(tf, 26), 2):
            if fut:
                break
            for o in OPTS:
                r = sim_jump(sol, mx, F, v0, o[0], o[1], dd, fx)
                if r is not None and (r[2] >= 0 or tf > 14):
                    fut = True
                    break
        if fut:
            if WB_ON and WB_X0 < P.scroll + mx < WB_X1 and tf <= 14 and self.fno >= self.nowait_until and self.wb_n < WB_MAX:
                Xw = xs_table(v0, None)
                distw = Xw[min(tf, 149)]
                stopw = v0 * v0 / (2 * _BRK) + 3
                efail = any(str(FAIL.get(o, "")).startswith(("enemy", "landenemy")) for o in OPTS)
                if efail and distw <= stopw + 14 and ext_left(sol, mx + 5, F, 200) > 10:
                    self.waiting = 1
                    self.wb_n += 1
                    return {"Left"}, "brake"
            return keys, "wait%d" % tf
        X = xs_table(v0, None)
        dist = X[min(tf, 149)] if tf is not None else 99
        stop = v0 * v0 / (2 * _BRK) + 3
        room = ext_right(sol, mx + 5, F, 200)
        room_tot = room + ext_left(sol, mx + 5, F, 200)
        if self.fno >= self.nowait_until and dist > stop and room_tot >= ROOM_BRAKE and not (CAPZ0 < P.scroll + mx < CAPZ1):
            self.waiting = 1
            return {"Left"}, "brake"
        if RETREAT and self.on_lift and dist <= stop + 3 and self.retreat_n < 140 and ext_left(sol, mx + 5, F) > 7:
            self.retreat_n += 1
            return {"Left"}, "retreat"
        if tf <= 3:
            return self.emergency(sol, mx, F, v0, fx, keys)
        return keys, "wait%d" % tf

    def note_spawn(self):
        fc = self.fc
        ls = getattr(self, "last_spawn", None)
        if ls is not None and fc - ls < 25:
            return
        if ls is not None and 70 <= fc - ls <= 100:
            self.spawn_p = fc - ls
        self.last_spawn = fc

    def spawn_gate(self, rem):
        """do not start a lift->lift hop when a (still invisible) Bullet Bill is predicted to spawn around take-off/landing."""
        ls = getattr(self, "last_spawn", None)
        if (not SG_ON) or ls is None or getattr(self, "gate_n", 0) >= SG_MAX:
            return False
        P = getattr(self, "spawn_p", 82)
        since = self.fc - ls
        if since > 300:
            return False
        run = max(0.0, rem) / 2.4 + 1
        ph = (since + run) % P
        if ph >= P - SG_PRE or ph <= SG_POST:
            self.gate_n = getattr(self, "gate_n", 0) + 1
            return True
        return False

    def lift_simple(self, mx, F, lifts):
        """on lift1 with the lower lift2 ahead: stand at the right edge, jump (k14, hold Right) when the gap <= LS_GAP."""
        ol = None
        for l in lifts:
            if abs(F - l[0]) <= 3 and l[1] - LT_M <= mx + 5 <= l[2] + LT_M:
                ol = l
                break
        if ol is None:
            return None
        tg = [l for l in lifts if l is not ol and l[1] > ol[2] - 5 and l[0] > ol[0] + 8 and l[1] < 256]
        if not tg:
            if ol[0] < 135 and self.P.scroll + mx > 1250:
                w_ = self.P.scroll + ol[2]
                self.ls_maxw = max(getattr(self, "ls_maxw", 0), w_)
                n_ = getattr(self, "on_lift_n", 0)
                gap = LS_GAP if ((self.P.scroll >= LS_TRAP and w_ >= LS_WX) or (n_ > 250 and w_ >= self.ls_maxw - 10)) else 999
            else:
                return None
        else:
            if getattr(self, "on_lift_n", 0) <= LS_AFTER:
                return None
            tg.sort(key=lambda l: l[1])
            gap = tg[0][1] - ol[2]
        for e in ENEMIES:
            if e[3] == 0 and e[4] == 8 and e[2] < -0.3:
                dxb = e[0] - (mx + 6)
                if 0 < dxb < 52 and F - 15 < e[1] + 1 and F - 1 > e[1] - e[5]:
                    self.plan = dict(k=8, tb=None, t=0, stay=True, lstay=True)
                    return {"z"}, "liftbillhop"
        er = max(ol[2], ol[1] + 30) - (mx + 5)
        self.lift_wait += 1
        pe = getattr(self, "ls_er", None)
        self.ls_er = er
        vrel = (pe - er) if pe is not None else 0.0
        self.ls_n = getattr(self, "ls_n", 0) + 1
        pg = getattr(self, "ls_gap", 999)
        self.ls_gap = gap
        thr = LS_GAP
        if self.ls_n > 150 and LS_RELAX:
            thr = min(LS_GAP + 26, LS_GAP + (self.ls_n - 150) // 6)
        if gap <= thr and -3 <= er <= 6 and (thr == LS_GAP or gap <= pg):
            kk = LS_K if gap <= LS_GAP + 4 else (24 if gap <= LS_GAP + 14 else 32)
            if not tg:
                kk = LS_KNT
            self.plan = dict(k=kk, tb=None, t=0, force=True)
            return {"Right", "x", "z"}, "lsjump"
        if (not tg) and LS_RUNUP:
            if gap <= thr and er > 6:
                return {"Right", "x"}, "lsrun"
            if gap > thr:
                if er < 20:
                    return {"Left"}, "lsback"
                if er > 27 and vrel < 0.5:
                    return {"Right"}, "lsback"
                return set(), "lswait"
        if er > 24 and vrel < 1.2:
            return {"Right"}, "lsedge"
        if er > 7 and vrel < 0.5:
            return {"Right"}, "lsedge"
        if er < -1 and vrel > -0.5:
            return {"Left"}, "lsedge"
        return set(), "lswait"

    def lift_hop(self, mx, F, lifts):
        """standing on a lift with another lift ahead: wait (centred) until the gap is small, run to the edge, hop; steer in the air."""
        ol = None
        for l in lifts:
            if abs(F - l[0]) <= 3 and l[1] - LT_M <= mx + 5 <= l[2] + LT_M:
                ol = l
                break
        if ol is None:
            return None
        for e in ENEMIES:
            if e[3] == 0 and e[4] == 8 and e[2] < -0.3:
                dxb = e[0] - (mx + 6)
                if 0 < dxb < 52 and F - 15 < e[1] + 1 and F - 1 > e[1] - e[5]:
                    self.plan = dict(k=8, tb=None, t=0, stay=True, lstay=True)
                    return {"z"}, "liftbillhop"
        tg = [l for l in lifts if l is not ol and l[1] > ol[2] - 5 and l[0] >= ol[0] - 6 and l[1] < 256]
        if not tg:
            return None
        tg.sort(key=lambda l: l[1])
        t0 = tg[0]
        self.lift_wait += 1
        gap = t0[1] - ol[2]
        c = (ol[1] + ol[2]) / 2.0
        rel = (mx + 5) - c
        D = (t0[1] + t0[2]) / 2.0 - (mx + 5)
        bill = False
        for e in ENEMIES:
            if e[4] == 8 and e[3] == 0 and e[2] < -0.3:
                dxb = e[0] - (mx + 6)
                if -25 < dxb < LB_D and F - 100 < e[1] < F + LB_LOW:
                    bill = True
        lp = getattr(self, "lprev", None)
        vrel = max(-3.0, min(3.0, rel - lp)) if lp is not None else 0.0
        self.lprev = rel
        rp = rel + vrel * abs(vrel) / 0.2
        go = gap <= LIFT_GAP or (getattr(self, "lgo", False) and gap <= LIFT_GAP + 25)
        if (not go) or (bill and ol[2] - (mx + 5) > 12):
            self.lgo = False
            if rp < -11:
                return {"Right"}, "liftwait"
            if rp > -2:
                return {"Left"}, "liftwait"
            return set(), "liftwait"
        if bill and (self.wxh[-1] - self.wxh[-2]) < 1.2:
            return set(), "liftbill"
        if ol[2] - (mx + 5) > LIFT_EDGEJ:
            self.lgo = True
            return {"Right", "x"}, "liftgo"
        self.lgo = False
        if D <= 64:
            k = 4
        elif D <= 78:
            k = 8
        elif D <= 94:
            k = 11
        else:
            k = 14
        sp = abs(self.wxh[-1] - self.wxh[-2]) if len(self.wxh) >= 2 else 0.0
        self.plan = dict(k=k, tb=None, t=0, hold=True, F0=F, fast=(sp >= 2.2))
        return {"Right", "x", "z"}, "liftjump%d" % k

    def lift_hop2(self, mx, F, lifts, sol):
        """standing on a lift: hop to the next lift (moving) or to the next lime cap (static); wait centred until the gap is small."""
        P = self.P
        ol = None
        for l in lifts:
            if abs(F - l[0]) <= 3 and l[1] - LT_M <= mx + 5 <= l[2] + LT_M:
                ol = l
                break
        if ol is None:
            return None
        for e in ENEMIES:
            if e[3] == 0 and e[4] == 8 and e[2] < -0.3:
                dxb = e[0] - (mx + 6)
                if 0 < dxb < 52 and F - 15 < e[1] + 1 and F - 1 > e[1] - e[5]:
                    self.plan = dict(k=8, tb=None, t=0, stay=True, lstay=True)
                    return {"z"}, "liftbillhop"
        tg = [l for l in lifts if l is not ol and l[1] > ol[2] - 5 and l[0] >= ol[0] - 6 and l[1] < 256]
        tw = None
        if tg:
            tg.sort(key=lambda l: l[1])
            t0 = tg[0]
            tx0, tx1 = t0[1], t0[2]
            static = False
        else:
            xs = int(max(mx + 8, ol[2] + 3))
            if xs >= 236:
                return None
            sub = sol[max(int(F) - 6, 0):232, xs:252]
            idx = np.nonzero(sub.any(axis=0))[0]
            if len(idx) == 0:
                return None
            c0 = int(idx[0])
            col = np.nonzero(sub[:, min(c0 + 3, sub.shape[1] - 1)])[0]
            if len(col) == 0:
                return None
            yc = max(int(F) - 6, 0) + int(col[0])
            w = ext_right(sol, xs + c0 + 2, yc, 40)
            if w < 16 or yc < F - 8:
                return None
            tx0, tx1 = xs + c0, xs + c0 + w
            tw = (P.scroll + tx0, P.scroll + tx1, yc)
            static = True
        self.lift_wait += 1
        gap = tx0 - ol[2]
        c = (ol[1] + ol[2]) / 2.0
        rel = (mx + 5) - c
        D = (tx0 + min(tx1 - tx0, 30) / 2.0) - (mx + 5)
        tvx = 0.0
        if not static:
            for ll in LIFTS:
                if ll[0] == tx0 and ll[1] == tx1:
                    tvx = ll[4]
                    break
            D += TVX_K * tvx
        bill = False
        for e in ENEMIES:
            if e[4] == 8 and e[3] == 0 and e[2] < -0.3:
                dxb = e[0] - (mx + 6)
                if -25 < dxb < LB_D and F - 100 < e[1] < F + LB_LOW:
                    bill = True
        lp = getattr(self, "lprev", None)
        vrel = max(-3.0, min(3.0, rel - lp)) if lp is not None else 0.0
        self.lprev = rel
        rp = rel + vrel * abs(vrel) / 0.2
        gmax = LIFT_GAP2 if static else LIFT_GAP
        gmax += min(GRELAX_MAX, max(0, self.lift_wait - GRELAX_T) // 5)
        gap_eff = gap + (TVX_G * max(tvx, 0.0) if not static else 0.0)
        go = gap_eff <= gmax or (getattr(self, "lgo", False) and gap <= gmax + 25)
        if go and (not static) and ol[0] < 135 and (not getattr(self, "lgo", False)) and self.spawn_gate(ol[2] - (mx + 5)):
            go = False
        if (not static) and (not getattr(self, "lgo", False)) and tvx > TVX_MAX:
            go = False
        if FEAS_ON and go and (not static) and (not getattr(self, "lgo", False)) and self.lift_wait < FEAS_T:
            mxe = max(mx, ol[2] - 8.0)
            fbest = None
            for kf in (11, 14, 18, 24, 32):
                rf = sim_jump(sol, mxe, F, 2.0, kf, None, 0, -999, 0)
                if rf is not None and rf[2] >= 0 and (fbest is None or rf[2] > fbest[1]):
                    fbest = (kf, rf[2])
            if fbest is None:
                go = False
            else:
                self.fk = fbest[0]
        if (not static) and ol[0] < 135 and P.scroll + ol[2] < LS_GO and not getattr(self, "lgo", False) and self.lift_wait < LS_GOWAIT:
            go = False
        if static and ol[0] >= 135 and P.scroll + ol[2] < LS_GO2 and not getattr(self, "lgo", False) and self.lift_wait < LS_GOWAIT2:
            go = False
        if (not go) or (bill and ol[2] - (mx + 5) > 12):
            self.lgo = False
            if EDGE_GUARD and (ol[2] - (mx + 5)) < EDGE_GUARD:
                return {"Left"}, "liftwait"
            if EDGE_GUARD and ((mx + 5) - ol[1]) < LW_LG:
                return {"Right"}, "liftwait"
            if rp < LW_A:
                return {"Right"}, "liftwait"
            if rp > LW_B:
                return {"Left"}, "liftwait"
            return set(), "liftwait"
        sp = (self.wxh[-1] - self.wxh[-2]) if len(self.wxh) >= 2 else 0.0
        if bill and sp < 1.2:
            return set(), "liftbill"
        if LBACK_ON and (not static) and (not getattr(self, "lgo", False)) and (mx + 5) - ol[1] > LBACK_X and getattr(self, "lb_n", 0) < LBACK_N:
            self.lb_n = getattr(self, "lb_n", 0) + 1
            return {"Left"}, "liftback"
        if ol[2] - (mx + 5) > LIFT_EDGEJ:
            self.lgo = True
            return {"Right", "x"}, "liftgo"
        self.lgo = False
        self.lb_n = 0
        if D <= LK0:
            k = 4
        elif D <= LK1:
            k = 8
        elif D <= LK2:
            k = 11
        elif D <= LK3 or static:
            k = 14
        elif D <= LK4:
            k = 18
        else:
            k = 24
        if (not static) and ol[0] < 135 and k in (8, 11):
            k = 14 if D > 72 else 11
        if LF_ON and (not static) and ol[0] < 135 and D > LF_D:
            k = max(k, LF_K)
        if LOWK_ON and (not static) and ol[0] < 135 and self.speed() < LOWK_V and D > 50:
            k = max(k, LOWK_K)
        if CAPK_ON and static and abs(sp) < 2.2 and D > CAPK_D:
            k = 24 if D <= 100 else 32
        if SF_ON and static and abs(sp) < 2.2:
            kb = None
            for kf in (4, 8, 11, 14, 18, 24, 32):
                rf = sim_jump(sol, mx, F, max(sp, 0.0), kf, None, 0, -999, SF_EXT)
                if rf is not None and rf[2] >= SF_EXT:
                    kb = kf
                    break
            if kb is None:
                kb = 32 if D > 56 else k
            elif kb < k and D > 56:
                kb = k
            k = max(k, kb) if kb is not None else k
            self.sfk = (k, round(D), round(sp, 1))
        if FEAS_ON and (not static) and getattr(self, "fk", None) is not None:
            k = self.fk
        self.fk = None
        print("LJ", self.fc, "wx", int(P.scroll + mx), "sp", round(sp, 2), "D", round(D), "gap", round(gap), "k", k, "tvx", round(tvx, 2), "static", int(static), "er", round(ol[2] - (mx + 5)), "rel", round(rel))
        self.plan = dict(k=k, tb=None, t=0, hold=True, F0=F, fast=(abs(sp) >= 2.2))
        self.lift_wait = 0
        if tw is not None:
            self.plan["tw"] = tw
        return {"Right", "x", "z"}, "%sjump%d" % ("cap" if static else "lift", k)

    def vest(self):
        h = self.wxf
        if len(h) < 3:
            return 0.0
        fc, wx = h[-1]
        best = None
        for (f2, w2) in h[:-1]:
            d = fc - f2
            if 2 <= d <= 8 and (best is None or abs(d - 4) < abs(best[0] - 4)):
                best = (d, w2)
        if best is None:
            return 0.0
        return max(-1.6, min(3.0, (wx - best[1]) / float(best[0])))

    def steer_plan(self, pl, mx):
        """during a planned jump onto a lift: pick Right / Left / none from the predicted landing (full vertical model)."""
        P = self.P
        tgts = [(P.scroll + l[0], P.scroll + l[1], l[2], l[3], l[4]) for l in LIFTS]
        if pl.get("tw"):
            tgts.append((pl["tw"][0], pl["tw"][1], pl["tw"][2], 0.0, 0.0))
        if not tgts or len(self.wxh) < 3:
            return None
        k = pl["k"]
        t0 = pl["t"]
        F0 = pl["F0"]
        v0 = self.vest()
        smarg = 0.5
        if pl.get("tb") is None:
            v0 = min(v0, VCLAMP)
            smarg = STEER_MARGIN
        tab = RISE[k] if pl.get("fast", True) else STAND.get(k, RISE[k])
        Mx = P.scroll + mx
        bestc = None
        for ci, (ks, acc, lo, hi) in enumerate((({"Right", "x"}, 0.0, -9, 9), ({"Left", "x"}, -0.16, -1.5, 3.0), ({"x"}, -0.03, -1.5, 3.0))):
            if ci == 1 and NOLEFT and pl.get("tb") is None:
                continue
            x = Mx
            v = v0
            res = None
            for t in range(t0 + 1, t0 + 90):
                r_prev = rise_at(tab, t - 2) if t >= 2 else 0
                r_now = rise_at(tab, t - 1)
                Fy = F0 - r_now
                if ci:
                    v = max(lo, min(hi, v + acc))
                x += v
                if r_now < r_prev:
                    dt = t - t0
                    for (ta0, ta1, ltop, lvy, lvx) in tgts:
                        lt = ltop + lvy * dt
                        if Fy >= lt - 1 and F0 - rise_at(tab, t - 2) <= lt + 3:
                            a0 = ta0 + lvx * dt
                            a1 = ta1 + lvx * dt
                            if a0 - 4 <= x + 5 <= a1 + 4:
                                aim = min(a0 + 17.0, (a0 + a1) / 2.0)
                                res = (abs(x + 5 - aim) + VPEN * max(0.0, v - VREF))
                                break
                    if res is not None:
                        break
                if Fy > 235:
                    break
            if DBG is not None:
                DBG.append((self.fno, "ctrl", ci, None if res is None else round(res, 1)))
            if res is not None and (bestc is None or res < bestc[0] - smarg):
                bestc = (res, ks)
        if DBG is not None:
            DBG.append((self.fno, "t", t0, "Mx", round(Mx, 1), "v0", round(v0, 2), "F0", F0, [(round(a[0]), round(a[1]), a[2], round(a[3], 2), round(a[4], 2)) for a in tgts[:3]], sorted(bestc[1]) if bestc else None))
        if bestc is None:
            return None
        return set(bestc[1])

    def steer(self, mx, my, dy):
        """falling towards a lift: choose Right / Left / none so the predicted landing lies in the lift's middle."""
        if not LIFTS or len(self.wxh) < 3:
            return None
        P = self.P
        F0 = int(my) + 16
        v0 = self.wxh[-1] - self.wxh[-2]
        v0 = max(-1.6, min(3.0, v0))
        di = 0
        best = 99
        for i, v in enumerate(DY):
            if abs(v - dy) < best:
                best = abs(v - dy)
                di = i
        Mx = P.scroll + mx
        bestc = None
        for ci, (ks, acc, lo, hi) in enumerate((({"x"}, 0.0, -9, 9), ({"Left", "x"}, -0.16, -1.5, 3.0), ({"Right", "x"}, 0.09, -1.5, 3.0))):
            for (lx0, lx1, ltop, lvy, lvx) in LIFTS:
                if ltop < F0 - 2:
                    continue
                F = float(F0)
                x = Mx
                v = v0
                landed = None
                for t in range(1, 70):
                    F += DY[min(di + t - 1, len(DY) - 1)]
                    v = max(lo, min(hi, v + acc)) if ci else v
                    x += v
                    lt = ltop + lvy * t
                    if F >= lt:
                        landed = (x, t)
                        break
                if landed is None:
                    continue
                xl, t = landed
                a0 = P.scroll + lx0 + lvx * t
                a1 = P.scroll + lx1 + lvx * t
                c = a0 + 17.0
                cost = abs(xl + 5 - c) + VPEN * max(0.0, v - VREF)
                if cost > 40:
                    continue
                if bestc is None or cost < bestc[0] - 0.5:
                    bestc = (cost, ks)
        if bestc is None:
            return None
        return set(bestc[1])

    def with_lifts(self, sol):
        if not LIFTS:
            return sol
        s = sol.copy()
        for (lx0, lx1, ltop, lvy, lvx) in LIFTS:
            dxs = int(lvx * LIFT_LEAD)
            top = int(ltop + lvy * LIFT_LEAD)
            a = max(int(lx0) + dxs, 0)
            b = min(int(lx1) + dxs + 1, 256)
            if 0 <= top < 230 and b > a:
                s[top:top + 8, a:b] = True
        return s

    def zpulse(self, sol, mx, my, dy):
        """falling onto a tree with a head-height Bullet Bill about to arrive at the landing point: press jump 1-2 frames before touchdown so the hop starts at once (saves the detect+latency frames); hold it while rising."""
        if not ZP_ON or self.P.scroll + mx < ZP_X0:
            return None
        if getattr(self, "zp_rise", 0) > 0:
            if dy < -0.5:
                self.zp_rise -= 1
                return {"z"}, "zphold"
            self.zp_rise = 0
        if dy < 1 or my + 16 > 225 or getattr(self, "zp_cnt", 0) >= ZP_MAX:
            return None
        bills = [e for e in ENEMIES if e[4] == 8 and e[3] == 0 and e[2] < -0.3]
        if not bills:
            return None
        di = 0
        best = 99
        for i, v in enumerate(DY):
            if abs(v - dy) < best:
                best = abs(v - dy)
                di = i
        vx = max(0.5, self.speed())
        Fy = int(my) + 16
        x = mx
        tl = None
        Fl = Fy
        for t in range(1, 8):
            Fn = Fy + DY[min(di + t - 1, len(DY) - 1)]
            x += vx
            xi = int(x)
            if xi + 12 >= 256:
                return None
            hit = None
            for r in range(Fy, Fn + 1):
                if r < 240 and sol[r, xi + 4:xi + 7].any():
                    hit = r
                    break
            if hit is not None:
                tl = t
                Fl = hit
                break
            Fy = Fn
        if tl is None or tl > ZP_T:
            return None
        if ext_right(sol, x + 5, Fl, 60) + ext_left(sol, x + 5, Fl, 60) < ZP_W:
            return None
        for e in bills:
            dxl = (e[0] + e[2] * tl) - (mx + 6 + vx * tl)
            if ZP_D0 < dxl < ZP_D1 and Fl - 15 < e[1] + 1 and Fl - 1 > e[1] - e[5]:
                if getattr(self, "zp_start", None) is None:
                    self.zp_start = self.fc
                self.zp_cnt = getattr(self, "zp_cnt", 0) + 1
                self.zp_rise = 9
                return {"z"}, "zpulse"
        return None

    def evade(self, sol, mx, my, dy):
        """airborne with a head-height Bullet Bill closing on the landing point: pick Right / none / Left so that the Bill neither hits in the air nor is right on top at touchdown (lead >= EV_LEAD or already passed)."""
        wx = self.P.scroll + mx
        if not EV_ON or not (EV_A0 < wx < EV_A1 or EV_B0 < wx < EV_B1):
            return None
        bills = [e for e in ENEMIES if e[4] == 8 and e[3] == 0 and e[2] < -0.3 and -30 < e[0] - (mx + 6) < 170]
        if not bills:
            return None
        v0 = max(0.0, self.speed())
        F0 = int(my) + 16
        di0 = 0
        if dy >= 1:
            bst = 99
            for i, q in enumerate(DY):
                if abs(q - dy) < bst:
                    bst = abs(q - dy)
                    di0 = i
        res = []
        for name, ks, acc, lo, hi in (("R", {"Right", "x"}, 0.05, -1.5, 3.0), ("N", {"x"}, 0.0, -1.5, 3.0), ("L", {"Left", "x"}, -0.16, -1.5, 3.0)):
            x = mx
            v = v0
            F = float(F0)
            vy = dy
            rising = dy < -0.5
            di = di0
            sc = -50.0
            for t in range(1, 70):
                if rising:
                    Fn = F + vy
                    vy += 0.9
                    if vy >= 0:
                        rising = False
                        di = 0
                else:
                    Fn = F + DY[min(di, len(DY) - 1)]
                    di += 1
                v = max(lo, min(hi, v + acc))
                x += v
                xi = int(x)
                if xi + 12 >= 256 or xi < 0:
                    break
                hitb = False
                for e in bills:
                    bx = e[0] + e[2] * t
                    if abs(x + 6 - bx) < 14 + EV_M and Fn - 15 < e[1] + 1 and Fn - 1 > e[1] - e[5]:
                        hitb = True
                        break
                if hitb:
                    sc = -100.0 + t
                    break
                if not rising:
                    hit = None
                    for r in range(int(F), int(Fn) + 1):
                        if r < 240 and sol[r, xi + 4:xi + 7].any():
                            hit = r
                            break
                    if hit is not None:
                        if ext_right(sol, x + 5, hit) < 12:
                            sc = -60.0
                            break
                        sc = 60.0
                        for e in bills:
                            lead = e[0] + e[2] * t - (x + 6)
                            if lead > -20 and (hit - 15 < e[1] + 1 and hit - 1 > e[1] - e[5]):
                                sc = min(sc, lead)
                        break
                F = Fn
                if F > 232:
                    break
            res.append((sc, name, ks))
        sr = res[0][0]
        if sr >= EV_LEAD:
            return None
        bst = max(res, key=lambda r: r[0])
        if bst[0] > sr + 3 and bst[0] > -50:
            self.ev_n = getattr(self, "ev_n", 0) + 1
            return set(bst[2])
        return None

    def air_control(self, sol, mx, my, keys):
        dy = 0.0
        if self.prev_my is not None:
            dy = my - self.prev_my
        self.prev_my = my
        tbk = tower_brake(self, sol, mx, my, dy)
        if tbk is not None:
            return tbk, "airTB"
        zp =self.zpulse(sol, mx, my, dy)
        if zp is not None:
            return zp
        evk = self.evade(sol, mx, my, dy)
        if evk is not None:
            return evk, "airEv"
        if dy < 1 or my + 16 > 225:
            return keys, "air"
        if LIFTS and self.P.scroll + mx > 1250:
            sk = self.steer(mx, my, dy)
            if sk is not None:
                return sk, "air"
        di = 0
        best = 99
        for i, v in enumerate(DY):
            if abs(v - dy) < best:
                best = abs(v - dy)
                di = i
        vx = max(1.0, self.speed())
        F = int(my) + 16
        rr = sim_fall_b(sol, mx, F, vx, di, False)
        rl = sim_fall_b(sol, mx, F, vx, di, True)
        runR = -1 if rr is None else ext_right(sol, rr[0] + 5, rr[1])
        runL = -1 if rl is None else ext_right(sol, rl[0] + 5, rl[1])
        if rr is None:
            if rl is not None:
                return {"Left", "x"}, "airL"
            return keys, "air"
        if LIFTS and rl is not None and self.P.scroll + mx > LIFT_BRAKE_X0 and any(abs(rl[1] - l[2]) <= 4 or abs(rr[1] - l[2]) <= 4 for l in LIFTS):
            return {"Left", "x"}, "airL"
        if runR >= AIR_ROOM:
            return keys, "air"
        if rl is not None and runL > runR + 6 and runL >= 12:
            return {"Left", "x"}, "airL"
        return keys, "air"

    def bill_gate2(self, sol, mx, F, v0):
        """small cap before the tall tower (wx1560-1660): do not launch a jump while a Bullet Bill is crossing the jump envelope ahead."""
        wx = self.P.scroll + mx
        if self.on_lift or not (G2_X0 < wx < G2_X1) or getattr(self, "g2n", 0) >= G2_MAX:
            return None
        thr = False
        for e in ENEMIES:
            if e[4] == 8 and e[3] == 0 and e[2] < -0.3:
                dx = e[0] - (mx + 6)
                if -8 < dx < G2_D and F - 100 < e[1] < F + 6:
                    thr = True
        if not thr:
            return None
        er = ext_right(sol, mx + 5, F, 200)
        if er < 16 and v0 > 1.5:
            return None
        self.g2n = getattr(self, "g2n", 0) + 1
        if v0 > 0.8 and ext_left(sol, mx + 5, F, 200) > 12:
            return {"Left"}
        return set()

    def do_jump(self, sol, ok0, res0, v0, keys):
        g2 = self.bill_gate2(sol, self.P.mx, self.curF, v0)
        if g2 is not None:
            return g2, "billgate2"
        if self.on_lift:
            cands = sorted(ok0, key=lambda o: (o[1] is not None, res0[o][2] < 20, res0[o][1] // 12, -min(res0[o][2], 30), o[0]))
        else:
            cands = sorted(ok0, key=lambda o: (o[1] is not None, -min(res0[o][2], 30), o[0]))
        best = cands[0]
        n = 0
        for o in cands:
            xl, Fl, ext = res0[o]
            if ext >= 30 and xl + 40 > 250:
                best = o
                break
            n += 1
            if n > 5:
                break
            vl = 3.0 if o[1] is None else 1.5
            if cont_ok(sol, xl, Fl, vl):
                best = o
                break
        k, tb = best
        self.retreat_n = 0
        self.plan = dict(k=k, tb=tb, t=0)
        keys.add("z")
        return keys, "jump%d%s" % (k, "" if tb is None else "b%d" % tb)

    def emergency(self, sol, mx, F, v0, fx, keys):
        global ENEMIES, LIFTS
        g2 = self.bill_gate2(sol, mx, F, v0)
        if g2 is not None:
            return g2, "billgate2"
        best = None
        allc = []
        for o in OPTS:
            r = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fx, 3)
            if r is not None:
                allc.append((o, r))
            if r is not None and (best is None or r[2] > best[1][2] or (r[2] == best[1][2] and o[0] > best[0][0])):
                best = (o, r)
        if best is None:
            se, sl = ENEMIES, LIFTS
            ENEMIES = []
            try:
                for o in OPTS:
                    r = sim_jump(sol, mx, F, v0, o[0], o[1], 0, fx, 3)
                    if r is not None:
                        allc.append((o, r))
                    if r is not None and (best is None or r[2] > best[1][2] or (r[2] == best[1][2] and o[0] > best[0][0])):
                        best = (o, r)
            finally:
                ENEMIES = se
        if EM_CLIMB and best is not None and best[1][1] > F - 8:
            cl = [(o, r) for (o, r) in allc if r[1] <= F - 20 and r[2] >= 10]
            if cl:
                cl.sort(key=lambda c: (-min(c[1][2], 30), -c[0][0]))
                best = cl[0]
        if best is None:
            k, tb = 24, None
        else:
            k, tb = best[0]
        self.plan = dict(k=k, tb=tb, t=0)
        keys.add("z")
        return keys, "jumpE%d" % k

    def wait_step(self, sol, mx, F, v0, tf, fx):
        self.waiting += 1
        keys = {"Right", "x"}
        if self.waiting > WAIT_MAX:
            self.waiting = 0
            self.nowait_until = self.fno + 150
            return keys, "waitdone"
        if tf is None:
            self.waiting = 0
            return keys, "run"
        found = False
        for dd in range(0, min(tf, 64), 3):
            if found:
                break
            for o in OPTS:
                r = sim_jump(sol, mx, F, v0, o[0], o[1], dd, fx)
                if r is not None and r[2] >= 0:
                    found = True
                    break
        if found:
            self.waiting = 0
            return keys, "run"
        for e in ENEMIES:
            if abs(e[1] - F) < 18 and 0 < e[0] - (mx + 6) < 38 and e[2] < -0.3:
                self.plan = dict(k=8, tb=None, t=0, stay=True)
                return {"z"}, "hop"
        if v0 > 0.4:
            return {"Left"}, "brake"
        return set(), "stand"
'''


def _mod(name, src):
    m = types.ModuleType(name)
    exec(compile(src, name, "exec"), m.__dict__)
    return m


SWITCH_X = 1285
AIRB_X0 = 99999
BOUNCE_X = 1580
_DBG_ON = False
DBGZ = False
KW_ON = True
KW_D = 30
DBGC = False
DBGE = False
EJ_ON = True
EJ_CUT = 0
GATE_ON = True
GATE_X0 = 1830
GATE_X1 = 1905
GATE_MAX = 150
TCG_ON = False
TCG_MAX = 120
TCG_PRE = 8
TCG_Y0 = 45
TCG_Y1 = 135
FZB_ON = True
FZB_D = 80
FZB_W = 0
_MODS = {}


def solve(sim):
    return []


def play(sim):
    A = _mod("polA", SRC_A)
    B = _mod("polB", SRC_B)
    _MODS["B"] = B
    if _DBG_ON:
        B.DBG = []
    pa = A.Policy()
    pb = B.Policy()
    img = sim.screen()
    k0 = A.kimg(img)
    pa.P.prev = k0
    pa.P.mario(k0)
    pb.P.prev = k0
    pb.P.mario(k0)
    mode = "A"
    gstate = {"lmy": None, "ldy": 0.0, "bounce": False, "gn": 0}
    fstate ={"prev_my": None, "still": 0, "hold": 0, "cool": 0, "announced": False}
    dead = 0
    still = 0
    last = None
    f = 0
    while True:
        img = sim.screen()
        if last is not None and np.array_equal(img, last):
            still += 1
            if still > 45:
                print("RET still", f, "wx", int(pb.P.scroll + pb.P.mx))
                return
        else:
            still = 0
        last = img
        keys, why = {"Right", "x"}, "err"
        # final brick-ground / staircase zone: simple hopping runner
        try:
            Kz = B.kimg(img)
            if fstate["announced"]:
                fz = True
            elif pb.P.scroll + pb.P.mx > 1950 and int((Kz[208:240] == B.K_TAN).sum()) >= 60:
                fz = True
            else:
                fz = False
        except Exception:
            fz = False
        if fz:
            try:
                pb.P.scroll += pb.P.scroll_est(Kz)
                pb.P.mario(Kz)
                my = pb.P.my
                if fstate["prev_my"] is not None and abs(my - fstate["prev_my"]) < 0.4:
                    fstate["still"] += 1
                else:
                    fstate["still"] = 0
                fstate["prev_my"] = my
                keys = {"Right", "x"}
                Fz = int(my) + 16
                mxz = int(pb.P.mx)
                reg = Kz[max(Fz - 22, 0):max(Fz - 3, 1), mxz + 14:min(mxz + 14 + 56, 256)]
                obst = int((reg != B.K_SKY).sum()) > 3
                kdx = None
                if KW_ON:
                    kreg = Kz[max(Fz - 26, 0):max(Fz - 2, 1), mxz + 14:min(mxz + 14 + 100, 256)]
                    kmask = kreg == B.K_RED
                    if int(kmask.sum()) >= 4:
                        kdx = 14 + int(np.nonzero(kmask.any(axis=0))[0][0])
                if kdx is not None and fstate.get("pk") is not None:
                    fstate["vc"] = 0.5 * fstate.get("vc", 0.0) + 0.5 * max(0.0, fstate["pk"] - kdx)
                elif kdx is None:
                    fstate["vc"] = 0.0
                fstate["pk"] = kdx
                bthreat = False
                bwait = False
                if FZB_ON and fstate["still"] >= 1:
                    for (bcx, bby) in B.find_bills(Kz):
                        dxb = bcx - (mxz + 6)
                        if 0 < dxb < FZB_D and Fz - 16 < bby < Fz + 14:
                            bthreat = True
                        elif 0 < dxb < FZB_W and Fz - 95 < bby < Fz + (75 if Fz < 185 else 14):
                            bwait = True
                kthr = KW_D + 6.0 * fstate.get("vc", 0.0)
                if fstate["hold"] > 0 and fstate["hold"] <= 19 and fstate["still"] >= 2:
                    fstate["hold"] = 0
                    fstate["cool"] = 2
                if fstate["hold"] > 0:
                    keys.add("z")
                    fstate["hold"] -= 1
                    if fstate.get("ej") and fstate["hold"] <= EJ_CUT:
                        fstate["hold"] = 0
                    if fstate["hold"] == 0:
                        fstate["cool"] = 3
                elif fstate["cool"] > 0:
                    fstate["cool"] -= 1
                elif fstate["still"] >= 1 and FZB_ON and bthreat and fstate.get("bn", 0) < 12:
                    fstate["bn"] = fstate.get("bn", 0) + 1
                    fstate["hold"] = 24
                    fstate["ej"] = False
                    keys = {"x", "z"}
                    print("FZB bill hop", sim.frame, "wx", int(pb.P.scroll + pb.P.mx), "kdx", kdx)
                elif fstate["still"] >= 1 and FZB_ON and bwait and fstate.get("bw", 0) < 160:
                    fstate["bw"] = fstate.get("bw", 0) + 1
                    keys = set()
                elif fstate["still"] >= 1 and kdx is not None and kdx > kthr and fstate.get("kwait", 0) < 120:
                    fstate["kwait"] = fstate.get("kwait", 0) + 1
                    keys = set()
                elif fstate["still"] >= 1 and obst:
                    fstate["hold"] = 24
                    fstate["ej"] = False
                    keys.add("z")
                elif (EJ_ON and fstate["still"] >= 1 and mxz + 14 < 250 and Fz + 3 < 238
                        and bool((Kz[Fz:Fz + 2, mxz + 9:mxz + 12] == B.K_SKY).all())
                        and bool((Kz[Fz:Fz + 2, mxz + 2:mxz + 5] != B.K_SKY).any())):
                    fstate["hold"] = 24
                    fstate["ej"] = True
                    keys.add("z")
                fstate["n"] = fstate.get("n", 0) + 1
                if (fstate["n"] % 12 == 0 and fstate["n"] < 1500) or (fstate["n"] % 300 == 0):
                    print("Z", sim.frame, "wx", int(pb.P.scroll + pb.P.mx), "mx", mxz, "my", int(my), "still", fstate["still"], "hold", fstate["hold"], "cool", fstate["cool"], "kdx", kdx, "obst", obst, "kw", fstate.get("kwait", 0), "".join(sorted(k[0] for k in keys)))
                if not fstate["announced"]:
                    fstate["announced"] = True
                    print("FINAL ZONE f", f, "wx", int(pb.P.scroll + pb.P.mx))
            except Exception as ex_:
                keys = {"Right", "x"}
                if fstate.get("nerr", 0) < 5:
                    fstate["nerr"] = fstate.get("nerr", 0) + 1
                    print("FZERR", repr(ex_), f)
            pa.last_why = "fz"
            pb.last_why = "fz"
            sim.step(keys)
            f += 1
            continue
        if mode == "A":
            try:
                ka, wa = pa.decide(img)
            except Exception:
                ka, wa = {"Right", "x"}, "err"
            try:
                wxa = pa.P.scroll + pa.P.mx
                ls_ = getattr(pb, "last_spawn", None)
                if (TCG_ON and ls_ is not None and wa is not None and str(wa).startswith("jump") and pa.plan is not None
                        and pa.plan.get("t") == 0 and 1100 < wxa < 1180 and 138 <= pa.curF <= 152 and gstate.get("tcn", 0) < TCG_MAX):
                    P_ = getattr(pb, "spawn_p", 82)
                    ph_ = (pb.fc - ls_) % P_
                    vis_ = False
                    for (bcx_, bby_) in B.find_bills(B.kimg(img)):
                        if bcx_ > pa.P.mx + 4 and TCG_Y0 < bby_ < TCG_Y1:
                            vis_ = True
                    if vis_ or ph_ >= P_ - TCG_PRE:
                        gstate["tcn"] = gstate.get("tcn", 0) + 1
                        pa.plan = None
                        ka = {"Left"} if pa.speed() > 0.6 else set()
                        wa = "tcgate"
            except Exception:
                pass
            try:
                kb, wb = pb.decide(img)
            except Exception:
                kb, wb = {"Right", "x"}, "err"
            if ((pa.P.scroll + pa.P.mx > SWITCH_X and wa is not None) or wb in ("billwait2", "hop1", "billhop2", "zpulse", "zhold", "zphold")) and pa.plan is None:
                mode = "B"
                keys, why = kb, wb
            elif pa.plan is None and wb == "airEv" and wa in ("air", "airL", "airTB"):
                keys, why = kb, wb
            else:
                keys, why = ka, wa
        else:
            try:
                keys, why = pb.decide(img)
            except Exception:
                keys, why = {"Right", "x"}, "err"
        # stomp-bounce: hold jump (and Right) while rising after a bounce so the arc still reaches the target
        try:
            if pb.P.found:
                myc = pb.P.my
                if gstate["lmy"] is not None:
                    dyn = myc - gstate["lmy"]
                    if gstate["ldy"] > 0.5 and dyn < -1.5 and pb.P.scroll + pb.P.mx > BOUNCE_X:
                        gstate["bounce"] = True
                    elif gstate["bounce"] and dyn >= 0:
                        gstate["bounce"] = False
                    gstate["ldy"] = dyn
                gstate["lmy"] = myc
                if gstate["bounce"] and why not in ("zphold", "zhold", "zpulse"):
                    keys = {"Right", "x", "z"}
                    why = "bounce"
                # bill gate before the long jump onto the brick ground: wait on the tree while a Bullet Bill is incoming
                wxg = pb.P.scroll + pb.P.mx
                if (GATE_ON and GATE_X0 < wxg < GATE_X1 and pb.plan is None and pb.prev_my is None
                        and (why == "run" or str(why).startswith("wait")) and gstate["gn"] < GATE_MAX):
                    Kg = B.kimg(img)
                    bl = [b for b in B.find_bills(Kg) if b[0] > pb.P.mx + 24 and 100 < b[1] < 225]
                    if bl:
                        gstate["gn"] += 1
                        keys = {"Left"} if pb.speed() > 1.0 else set()
                        why = "billgate"
        except Exception:
            pass
        if why is None:
            dead += 1
            if dead > 120:
                return
        else:
            dead = 0
        if why and why != pa.last_why and why not in ("air", "run", "stand") and not str(why).startswith("wait") and why != "brake":
            ap0 = pb if mode == "B" else pa
            print("f", f, mode, "wx", int(ap0.P.scroll + ap0.P.mx), why)
        if DBGC and mode == "B" and 1380 < pb.P.scroll + pb.P.mx < 1600 and 985 <= sim.frame < 1060:
            try:
                pl = pb.plan
                print("C", sim.frame, "wx", int(pb.P.scroll + pb.P.mx), "mx", int(pb.P.mx), "my", int(pb.P.my), "v", round(pb.speed(), 1), why,
                      "".join(sorted(k[0] for k in keys)), "plan", None if pl is None else (pl.get("k"), pl.get("t")), "dbg", getattr(pb, "dbg", None),
                      "L", [(int(l[0]), int(l[1]), int(l[2]), round(l[3], 1), round(l[4], 1)) for l in B.LIFTS][:3])
            except Exception:
                pass
        pa.last_why = why
        pb.last_why = why
        if DBGE and mode == "B" and 1388 < pb.P.scroll + pb.P.mx < 1490 and 600 <= sim.frame < 1200 and pb.plan is not None and pb.plan.get("hold"):
            try:
                print("E", sim.frame, "wx", int(pb.P.scroll + pb.P.mx), "mx", int(pb.P.mx), "my", int(pb.P.my), why, "".join(sorted(k[0] + (k[1] if k[0] == "L" else "") for k in keys)),
                      "L", [(int(l[0]), int(l[1]), int(l[2]), round(l[3], 1), round(l[4], 1)) for l in B.LIFTS][:3], "lw", pb.lift_wait, "onl", int(pb.on_lift))
            except Exception:
                pass
        sim.step(keys)
        f += 1

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    try:\n        st = ctx.get("stuck") or {}\n        status = "running"\n        if st.get("is") and st.get("kind") == "frozen":\n            status = "dead"\n        elif st.get("is"):\n            status = "stuck"\n        return {"status": status, "where": {"hazard": None, "confidence": 0.3},\n                "progress": None, "resources": {}, "threats": [],\n                "stuck": {"is": bool(st.get("is"))},\n                "outlook": {"label": "ok", "p_through": 0.3}, "note": ""}\n    except Exception:\n        return {"status": "running", "stuck": {"is": False}, "outlook": {"label": "ok", "p_through": 0.3}}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

