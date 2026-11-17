# ==================================================================================================
# Super Mario Bros. (NES) -- level 7-1  |  超级马里奥兄弟（NES） 关 7-1
# Pass rate on 100 random variants never used before: 39/100; fastest finish 1395 frames (23.25 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 39/100；最快一局 1395 帧（60 帧/秒，约 23.25 秒）。
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
#   reactive pixel runner + tile-map physics sim (PB 9a3007f8) + stair/valley fixes: no back-off o
#   n 1st staircase, slow-beetle vs fast-bill veto split, valley retreat to floor
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def pack                           line 23    
#   def mario_box                      line 27    
#   def est_scroll                     line 58    
#   class Bot                          line 72    
#   def _eq                            line 327   
#   class OldAgent                     line 334   
#   class Hyb                          line 611   
#   def solve                          line 1363  
#   def play                           line 1367  
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: reactive pixel runner + tile-map physics sim (PB 9a3007f8) + stair/valley fixes: no back-off on 1st staircase, slow-beetle vs fast-bill veto split, valley retreat to floor
import numpy as np

_TPL = np.array([2, 4, 4, 4, 4, 4, 4, 6, 4, 22, 2, 6, 4, 4, 6, 26], np.float64)
_TPL = _TPL - _TPL.mean()
_VMAX = 3.0
_ACC = 0.15
_BODY_HW = 5.5
_SUP_HW = 3.5
_SUP_L = 10.0
_T = 64
DEBUG = True
PRINT_LIVE = True

_C_RED = (248 << 16) | (56 << 8)
_C_BRN = (172 << 16) | (124 << 8)
_C_SKY = (104 << 16) | (136 << 8) | 252
_C_WHITE = 252 * 65793
_C_GRAY = 124 * 65793
_SKY = np.array([104, 136, 252], np.uint8)


def pack(img):
    return (img[:, :, 0].astype(np.uint32) << 16) | (img[:, :, 1].astype(np.uint32) << 8) | img[:, :, 2]


def mario_box(p, prev):
    if prev is not None:
        px, py = prev
        x0 = max(0, int(px) - 40)
        x1 = min(256, int(px) + 40)
        y0 = max(24, int(py) - 60)
        y1 = min(240, int(py) + 60)
        sub = p[y0:y1, x0:x1]
        m = (sub == _C_RED) | (sub == _C_BRN)
        ys, xs = np.nonzero(m)
        if len(xs) >= 6:
            xs = xs + x0
            ys = ys + y0
        else:
            prev = None
    if prev is None:
        m = (p == _C_RED) | (p == _C_BRN)
        m[:24] = False
        ys, xs = np.nonzero(m)
    if len(xs) < 6:
        return None
    cx = int(np.median(xs))
    cy = int(np.median(ys))
    sel = (np.abs(xs - cx) <= 9) & (np.abs(ys - cy) <= 10)
    if sel.sum() < 4:
        return None
    xs = xs[sel]
    ys = ys[sel]
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def est_scroll(pp, p):
    a = pp[216:240]
    b = p[216:240]
    best = None
    for s in range(0, 6):
        if s == 0:
            mm = int((a != b).sum())
        else:
            mm = int((a[:, s:] != b[:, :-s]).sum()) * 256.0 / (256 - s)
        if best is None or mm < best[0]:
            best = (mm, s)
    return best[1]


class Bot:
    def __init__(self):
        self.W = 0
        self.prevp = None
        self.pprev2 = None
        self.p = None
        self.mprev = None
        self.mario = None
        self.tiles = {}
        self.kinds = {}
        self.vx = 0.0
        self.vy = 0.0
        self.last_mx = None
        self.prev_feet = None
        self.s = 0
        self.logs = []
        self.nlog = 0

    def update(self, img):
        s = 0
        chg = None
        p = pack(img)
        if self.prevp is not None:
            s = est_scroll(self.prevp, p)
        self.W += s
        e = self.phase_err(p)
        if e:
            self.W += e
            if self.last_mx is not None:
                self.last_mx += e
        if self.prevp is not None:
            pp = self.prevp
            if s == 0:
                chg = p != pp
            else:
                chg = np.zeros(p.shape, bool)
                chg[:, :-s] = p[:, :-s] != pp[:, s:]
        self.s = s
        mar = mario_box(p, self.mprev)
        self.mario = mar
        if mar is not None:
            xl, xr, yt, yb = mar
            cx = (xl + xr) / 2.0
            self.mprev = (cx, (yt + yb) / 2.0)
            wx = cx + self.W
            if self.last_mx is not None:
                self.vx = 0.6 * self.vx + 0.4 * (wx - self.last_mx)
            self.last_mx = wx
            if self.prev_feet is not None:
                self.vy = float(self.prev_feet - yb)
            self.prev_feet = yb
        self._tiles(p, chg, s)
        self.pprev2 = self.prevp
        self.prevp = p
        self.p = p

    def phase_err(self, p):
        blk = (p[208:240] == 0).sum(0).astype(np.float64)
        fold = blk.reshape(16, 16).sum(0)
        if fold.sum() < 80:
            return 0
        f = fold - fold.mean()
        sc = []
        for q in range(16):
            sc.append(float((np.roll(f, -q) * _TPL).sum()))
        qo = int(np.argmax(sc))
        srt = sorted(sc)
        if srt[-1] <= 0 or srt[-2] > 0.8 * srt[-1]:
            return 0
        qe = (-self.W) % 16
        e = (qe - qo) % 16
        if e >= 8:
            e -= 16
        return e

    def _tiles(self, p, chg, s):
        W = self.W
        ox = (-W) % 16
        n = (256 - ox) // 16
        crop = p[:240, ox:ox + 16 * n]
        ns = (crop != _C_SKY).reshape(15, 16, n, 16).sum((1, 3))
        frac = ns / 256.0
        white = (crop == _C_WHITE).reshape(15, 16, n, 16).sum((1, 3))
        dg = (crop == _C_GRAY).reshape(15, 16, n, 16).sum((1, 3))
        solid = ((frac >= 0.85) & (white <= 10)) | ((dg >= 30) & (frac >= 0.6))
        pipe = (dg >= 30) & (frac >= 0.6)
        if chg is not None:
            cc = chg[:, ox:ox + 16 * n].reshape(15, 16, n, 16).sum((1, 3)) / 256.0
        else:
            cc = np.zeros((15, n))
        base = (ox + W) // 16
        tiles = self.tiles
        kinds = self.kinds
        for j in range(n):
            fresh = (ox + 16 * j + 16 + s) > 256
            for r in range(2, 15):
                key = (base + j, r)
                if fresh or cc[r, j] < 0.02 or key not in tiles:
                    tiles[key] = bool(solid[r, j])
                    kinds[key] = 2 if pipe[r, j] else 0

    def solid(self, cw, r):
        if r < 2 or r >= 15:
            return False
        if cw <= 7 and r < 13:
            return False
        v = self.tiles.get((cw, r))
        if v is None:
            return r >= 13
        return v

    def plants(self, wx):
        out = []
        lo = int((wx - 100) // 16)
        hi = int((wx + 200) // 16)
        kinds = self.kinds
        for (cw, r), k in kinds.items():
            if k != 2 or cw < lo or cw > hi or r < 3 or cw <= 7:
                continue
            if kinds.get((cw, r - 1), 0) == 2 or self.tiles.get((cw, r - 1), False):
                continue
            if kinds.get((cw - 1, r), 0) == 2:
                continue
            if kinds.get((cw + 1, r), 0) != 2:
                continue
            out.append(((cw + 1) * 16.0, 16.0 * r - 13.0, 7.0, 13.0))
        return out

    def sim(self, plan, st, trajs, T=_T, vs=1.0, hwx=0.0):
        k, rel, dl = plan
        jumped = False
        self_land = None
        x = st["x"]
        y = st["y"]
        vx = st["vx"]
        vy = st["vy"]
        gr = st["gr"]
        ch = st["ch"]
        lag = st["lag"]
        solid = self.solid
        blocked = 0
        x0 = x
        soft = 0.0
        hw = _BODY_HW + hwx
        xmin = st["xmin"]
        for t in range(T):
            zp = dl <= t < dl + k
            if zp and gr and not jumped and not st["airborne"]:
                jumped = True
                gr = False
                vy = 6.0
                ch = True
                lag = 1
            if not zp and not gr:
                ch = False
            if t >= rel:
                vx = min(_VMAX, vx + _ACC)
            else:
                vx = vx - 0.12 if vx > 0 else min(0.0, vx + 0.12)
            if not gr:
                if lag > 0:
                    lag -= 1
                    ny = y
                else:
                    if vy > 0:
                        g = 0.2 if (zp and ch) else 0.6
                    else:
                        g = 0.7
                    vy -= g
                    if vy < -5.2:
                        vy = -5.2
                    ny = y - vy
            else:
                ny = y
            nx = x + vx * vs
            r0 = int((y - 14) // 16)
            r1 = int((y - 1) // 16)
            if vx != 0:
                ex = nx + (hw if vx > 0 else -hw)
                cw = int(ex // 16)
                wall = False
                for r in range(r0, r1 + 1):
                    if solid(cw, r):
                        wall = True
                        break
                if wall:
                    nx = cw * 16 - hw - 0.01 if vx > 0 else (cw + 1) * 16 + hw + 0.01
                    vx = 0.0
                    blocked += 1
            if nx < xmin:
                nx = xmin
                if vx < 0:
                    vx = 0.0
            x = nx
            c0 = int((x - _SUP_L) // 16)
            c1 = int((x + _SUP_HW) // 16)
            if gr:
                L = int(-(-y // 16)) * 16
                if not any(solid(c, L // 16) for c in range(c0, c1 + 1)):
                    gr = False
                    vy = 0.0
                    ch = False
            else:
                if vy < 0:
                    L = int(-(-y // 16)) * 16
                    if ny >= L and any(solid(c, L // 16) for c in range(c0, c1 + 1)):
                        ny = L
                        gr = True
                        if self_land is None:
                            self_land = (round(x - x0), L)
                        vy = 0.0
                        ch = False
                elif vy > 0:
                    rh = int((ny - 15) // 16)
                    if rh >= 0 and any(solid(c, rh) for c in range(int((x - hw) // 16), int((x + hw) // 16) + 1)):
                        ny = (rh + 1) * 16 + 15
                        vy = 0.0
                        ch = False
                y = ny
            if y > 236:
                return 10000.0 - t, dict(death=t, prog=x - x0, blocked=blocked)
            for tr in trajs:
                ex = tr[0][t]
                ey = tr[1][t]
                ehw = tr[2]
                ehh = tr[3]
                dx = abs(ex - x) - (ehw + 4.5 + hwx)
                dy = abs(ey - (y - 7)) - (ehh + 6.0)
                cl = dx if dx > dy else dy
                if cl < 0.5:
                    return 6000.0 - t, dict(coll=t, prog=x - x0, blocked=blocked)
                if cl < 5.0:
                    soft += (5.0 - cl) * 0.8
        prog = x - x0
        if blocked >= 3:
            return 3500.0 - prog, dict(wall=True, prog=prog, blocked=blocked)
        return -prog + soft, dict(prog=prog, blocked=blocked, soft=soft, land=self_land)

    def log(self, msg):
        self.logs.append(msg)
        if len(self.logs) > 40:
            del self.logs[:10]
        if DEBUG and PRINT_LIVE and self.nlog < 300:
            self.nlog += 1
            print(msg)


# ======================================================================= reactive base agent
_H = 240
_WD = 256
BILL_MIN = 3
_DEFAULTS = {"LOOK": 56, "PIT_LOOK": 10, "HOLD": 26, "BILL_R": 56, "BILL_L": 0,
             "BACK": 80, "RUN": 1, "THREAT": 1, "STILL": 0}


def _eq(img, c):
    return (img[..., 0] == c[0]) & (img[..., 1] == c[1]) & (img[..., 2] == c[2])


_SKYT = (104, 136, 252)


class OldAgent:
    def __init__(self):
        self.prev = None
        self.world = 0.0
        self.last_my1 = None
        self.hold_left = 0
        self.cool = 0
        self.prev_mario = None
        self.pimg = None
        self.back_left = 0
        self.back_cool = 0
        self.frozen = 0
        self.fprev = None
        self.n = 0
        self.dead_after = 90
        self.p = dict(_DEFAULTS)
        self.reason = None
        self.look_override = 0
        self.suppress = False
        self.safe_flyer = False
        self.noback = False

    def scroll(self, img):
        if self.prev is None:
            return 0
        a = self.prev[212:236].astype(np.int16)
        b = img[212:236].astype(np.int16)
        best, bs = None, 0
        for s in range(0, 5):
            if s == 0:
                d = np.abs(a - b).sum()
            else:
                d = np.abs(a[:, s:] - b[:, :_WD - s]).sum() * (_WD / (_WD - s))
            if best is None or d < best:
                best, bs = d, s
        return bs

    def find_mario(self, img):
        red = _eq(img, (248, 56, 0)) | _eq(img, (172, 124, 0))
        red[:32] = False
        ys, xs = np.nonzero(red)
        if len(xs) == 0:
            return None
        if self.prev_mario is not None:
            px, py = self.prev_mario
            m = (np.abs(xs - px) < 40) & (np.abs(ys - py) < 40)
            if m.sum() >= 3:
                xs, ys = xs[m], ys[m]
        return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    def act(self, img):
        self.n += 1
        if self.fprev is not None and (img == self.fprev).all():
            self.frozen += 1
        else:
            self.frozen = 0
        self.fprev = img
        sc = self.scroll(img)
        self.world += sc
        self.prev = img
        keys = self.decide(img, sc)
        self.pimg = img
        return keys, self.frozen >= self.dead_after

    def decide(self, img, sc):
        p = self.p
        self.reason = None
        keys = {"Right"}
        if p["RUN"]:
            keys.add("x")
        m = self.find_mario(img)
        if m is None:
            return keys
        x0, x1, y0, y1 = m
        self.prev_mario = ((x0 + x1) // 2, (y0 + y1) // 2)
        feet = y1
        support = ~_eq(img[min(feet + 1, _H - 1), x0:x1 + 1], _SKYT)
        on_ground = (self.last_my1 == y1) and bool(support.any())
        self.last_my1 = y1
        if self.hold_left > 0:
            self.hold_left -= 1
            keys.add("z")
            if self.hold_left == 0:
                self.cool = 3
            return keys
        if self.cool > 0:
            self.cool -= 1
            return keys
        if not on_ground:
            return keys
        LK = self.look_override or p["LOOK"]
        xa = min(x1 + 1, _WD - 1)
        xb = min(x1 + 1 + LK, _WD)
        gy = min(feet + 3, _H - 1)
        gx0 = min(x1 - 2, _WD - 1)
        gx1 = min(x1 + p["PIT_LOOK"], _WD)
        colsky = _eq(img[gy:_H, gx0:gx1], _SKYT).all(axis=0)
        pit = bool(colsky.sum() >= 4)
        threat = False
        rs = 0
        for rr in (feet - 9, feet - 7):
            if rr < 0 or rr >= _H:
                continue
            ns = ~_eq(img[rr, xa:xb + 12], _SKYT)
            run = 0
            for i in range(len(ns)):
                if ns[i]:
                    run += 1
                    if run >= 11 and (i - run + 1) <= LK:
                        rs = xa + i - run + 1
                        if self.safe_flyer:
                            reg = img[max(0, rr - 12):rr + 5, max(0, rs - 2):min(_WD, rs + run + 3)]
                            ng = int(((reg[..., 0] == 0) & (reg[..., 1] == 168) & (reg[..., 2] == 0)).sum())
                            if ng >= 4:
                                break
                        threat = True
                        break
                else:
                    run = 0
            if threat:
                break
        if self.world < 12 and x1 < 118:
            threat = False
        if self.suppress:
            threat = False
        if self.back_cool > 0:
            self.back_cool -= 1
        # never back off towards a pit behind us
        gb = _eq(img[gy:_H, max(0, x0 - 24):max(1, x0)], _SKYT).all(axis=0)
        pit_behind = bool(gb.sum() >= 2)
        if pit_behind:
            self.back_left = 0
        if threat and p["BACK"] > 0 and self.back_cool <= 0 and not pit_behind and not self.noback:
            col = min(rs + 3, _WD - 1)
            row = feet - 1
            while row > 0 and not _eq(img[row, col], _SKYT) and feet - row < 100:
                row -= 1
            h = feet - row
            dist = rs - x1 - 1
            if h >= 40 and dist < max(8, 0.85 * (h - 16)):
                self.back_left = int((p["BACK"] - dist) / 1.8)
                self.back_cool = self.back_left + 90
        if self.back_left > 0:
            self.back_left -= 1
            return {"Left", "x"}
        bill = False
        bill_low, bill_dist = False, 999
        if self.pimg is not None:
            r0, r1 = max(feet - 40, 0), max(feet - 2, 1)
            cx1 = min(x1 + 100, _WD - 8)
            cur = _eq(img[r0:r1, :cx1], (0, 0, 0))
            s = max(sc, 0)
            old = _eq(self.pimg[r0:r1, s:cx1 + s], (0, 0, 0)) if s else _eq(self.pimg[r0:r1, :cx1], (0, 0, 0))
            w = min(cur.shape[1], old.shape[1])
            d = (cur[:, :w] ^ old[:, :w])
            d[:, max(x0 - 6, 0):x1 + 7] = False
            dr = d[:, x1 + 7:]
            if dr.sum() >= BILL_MIN:
                bill = True
                ys, xs = np.nonzero(dr)
                bill_low = (ys.max() + r0) >= feet - 13
                bill_dist = int(xs.min()) + 7
        if self.look_override:
            bill = False
        if pit:
            self.hold_left = 30 if self.suppress else p["HOLD"]
            keys.add("z")
            self.reason = "pit"
        elif bill and bill_low and bill_dist <= p["BILL_R"]:
            self.hold_left = p["HOLD"]
            keys.add("z")
            self.reason = "bill"
        elif threat and not (bill and bill_dist <= 70):
            self.hold_left = p["HOLD"]
            keys.add("z")
            self.reason = "threat"
        return keys


# ======================================================================= hybrid
HOLD_ORDER = (26, 18, 13, 9, 6, 4, 40, 2)
SIM_LATE = 9
STAIR_VETO = True
USE_RELEASE = True
STAIR_FROM = 99999
CEIL_LOOK = 26
USE_FLYER = True
USE_PWAIT = True
USE_HIGHBILL = True
STAIRS_X = 2560
STAIR_CAP = 150
PLANT_GAP = 30
FLY_CLEAR = 20
FLY_STOP_D = 12
_C_GREEN = 168 << 8
MOD_FROM = 1450
# c0 stair fight
STAIRFIGHT = True
FIGHT_FROM = 2440
FIGHT_YMIN = 200
FIGHT_MAX = 400
FIGHT_R = 140
FIGHT_NEAR = 55
# low brick ceiling over ground enemy
LOWCEIL = True
LOW_TAP = 3
EARLY_FROM = 330
EARLY_TO = 470
EARLY_HOLD = 26
LOW_GAP = 66
LOW_FROM = 1250
LOW_TO = 1500
RUN_WAIT = True
RUN_WAITS = ((1285, 1335, 5),)
STAIR_BACK = True
STAIR_BACKS = ((2575, 2700, 8),)
STAIR_WAIT = True
STAIR_WAITS = ((2470, 2512, 45), (2515, 2560, 20), (1240, 1262, 30), (955, 1000, 30), (1025, 1065, 30))
LOW_TRIG = 66
PLANTSIM = True
BILLWAIT = True
BW_FROM = 2150
BW_TO = 2340
BW_MAX = 90
BW2_FROM = 1040
BW2_TO = 1110
HAMMER = True
HZ_FROM = 2040
HZ_TO = 2400
HZ_MAX = 160
HZ2_FROM = 99998
HZ2_TO = 99999
HZ_HOP = 24
HZ_HOLD = 16
HZ_FWD = True
HZ_LOOK = 80
# bills flying in from the LEFT at the stair foot / steps
BD_ON = False
BD_H = 55
BD_L = 12
BD_MAX = 30
BD_TO = 1500
FLYGATE = False
FG_D = 36
FG_FROM = 300
FG_TO = 470
FG_UP = 60
FG_MAX = 40
BL_ON = False
BL_FROM = 2380
BL_REACH = 72
BL_HOLD = 14
BL_MIN = 6
BL_COOL = 24
BW3_FROM = 99998
BW3_TO = 99999
SH_ON = False
SH_HOLD = 14
SH_MAX = 4
SH_TO = 1100
FB_D = 0
FB_MAX = 60
HZR_FROM = 99998
HZR_TO = 99999
HZR_NEAR = 34
HZR_MAX = 500
NOBACK = True
NB_FROM = 2400
NB_TO = 2545
VAL_MODE = 2
VAL_FAST = 24
VB_FROM = 2540
VB_TO = 2700
VB_MAX = 60
TUNABLE = {"MOD_FROM": (700, 1800)}


class Hyb:
    def __init__(self):
        self.bot = Bot()
        self.old = OldAgent()
        self.sveto = 0
        self.last_sveto = -99
        self.last_hold = 0
        self.pend = 0
        self.frame = 0
        self.dead_after = 60
        self.gprev = None
        self.fprev = None
        self.fvy = 0.0
        self.fflag = False
        self.stand = False
        self.waited = 0
        self.pveto = 0
        self.hb_until = -99
        self.last_pveto = -99
        self.pext = 0
        self.fw = 0
        self.ncancel = 0
        self.gz = 0
        self.wait_left = 0
        self.waited_set = set()
        self.wait_mode = "x"
        self.fl = None
        self.g_now = False
        self.back2 = 0
        self.nback2 = 0
        self.back_done = set()
        self.hj = 0
        self.blj = 0
        self.blc = 0
        self.bdn = 0
        self.fgn = 0
        self.hz = 0
        self.bw = 0
        self.bmh = []
        self.vb = 0
        self.hzr = 0
        self.fbn = 0
        self.sh = 0
        self.shn = 0

    def state(self, b):
        mar = b.mario
        if mar is None:
            return None
        xl, xr, yt, yb = mar
        wx = (xl + xr) / 2.0 + b.W
        cs = int((wx - _SUP_L) // 16)
        ce = int((wx + _SUP_HW) // 16)
        grounded = (yb % 16 == 0) and any(b.solid(c, yb // 16) for c in range(cs, ce + 1)) and abs(b.vy) < 0.6
        return wx, yb, grounded

    def bill_arc(self, b, reach=110, top=100):
        mar = b.mario
        if mar is None or b.p is None or b.pprev2 is None:
            return False
        x0, x1, y0, y1 = mar
        feet = y1
        r0, r1 = max(feet - top, 32), max(feet - 2, 33)
        cx1 = min(x1 + reach, 248)
        s = max(b.s, 0)
        cur = (b.p[r0:r1, :cx1] == 0)
        old = (b.pprev2[r0:r1, s:cx1 + s] == 0)
        w = min(cur.shape[1], old.shape[1])
        d = cur[:, :w] ^ old[:, :w]
        d[:, max(x0 - 6, 0):x1 + 7] = False
        return int(d[:, x1 + 7:].sum()) >= 6

    def green_gap(self, b):
        mar = b.mario
        if mar is None or b.p is None:
            return None
        x0, x1, y0, y1 = mar
        xa = min(x1 + 1, 255)
        xb = min(x1 + 100, 252)
        sub = (b.p[max(y1 - 24, 32):y1 + 1, xa:xb] == _C_GREEN)
        cols = np.nonzero(sub.any(axis=0))[0]
        if len(cols) == 0 or int(sub.sum()) < 4:
            return None
        return int(cols[0])

    def low_ceiling(self, b, wx, yb):
        R = yb // 16
        if yb % 16 != 0 or R < 12:
            return False
        c0 = int(wx // 16)
        for c in range(c0, c0 + 6):
            if b.kinds.get((c, R - 1), 0) == 2 or b.solid(c, R - 1) or b.solid(c, R - 2):
                return False
        for c in range(c0, c0 + 6):
            for r in (R - 4, R - 3):
                if b.solid(c, r) and not b.solid(c, r + 1):
                    return True
        return False

    def flyer_update(self, b, o):
        o.safe_flyer = False
        self.stand = False
        self.fl = None
        mar = b.mario
        if mar is None or b.p is None or not USE_FLYER or o.look_override:
            self.fprev = None
            return
        x0, x1, y0, y1 = mar
        xa = min(255, x1 + 1)
        xb = min(256, x1 + 110)
        if xb - xa < 8:
            self.fprev = None
            return
        sub = (b.p[60:222, xa:xb] == _C_GREEN)
        cols = sub.any(axis=0)
        if not cols.any():
            self.fprev = None
            self.waited = 0
            return
        idx = np.nonzero(cols)[0]
        left = int(idx[0])
        end = left
        for c in idx[1:]:
            if c - end > 6:
                break
            end = int(c)
        right = end
        if right - left + 1 > 24:
            return
        rows = np.nonzero(sub[:, left:right + 1].any(axis=1))[0]
        bot = 60 + int(rows[-1])
        dx = xa + left - x1
        wl = xa + left + b.W
        row = min(14, (bot + 2) // 16)
        for cw in range((xa + left + b.W) // 16, (xa + right + b.W) // 16 + 1):
            if b.kinds.get((cw, row), 0) == 2:
                self.fprev = None
                return
        fp = self.fprev
        if fp is not None and abs(wl - fp[0]) <= 8:
            vy = bot - fp[1]
            self.fvy = 0.5 * self.fvy + 0.5 * vy
            if bot <= 198 or abs(self.fvy) >= 0.6:
                self.fflag = True
        else:
            self.fvy = 0.0
            self.fflag = bot <= 198
        vxw = (wl - fp[0]) if (fp is not None and abs(wl - fp[0]) <= 8) else None
        self.fprev = (wl, bot)
        if self.fflag:
            self.fl = (dx, bot, vxw)
        if not self.fflag:
            return
        h = max(0.0, 205.0 - bot)
        vu = -self.fvy
        tc = max(0.0, (dx - 4) / 3.6)
        lo = tc - 2
        hi = tc + 9
        safe = True
        t = 0
        while t <= hi:
            t += 1
            h += vu
            vu -= 0.125
            if h <= 0:
                h = 0.0
                vu = 3.0
            if t >= lo and h < FLY_CLEAR:
                safe = False
                break
        if safe:
            o.safe_flyer = True
        elif dx > FLY_STOP_D and self.waited < 45:
            self.stand = True
            self.waited += 1

    def bill_high_only(self, b, reach=110, margin=17):
        mar = b.mario
        if mar is None or b.p is None or b.pprev2 is None:
            return False
        x0, x1, y0, y1 = mar
        feet = y1
        r0, r1 = max(feet - 100, 32), max(feet - 2, 33)
        cx1 = min(x1 + reach, 248)
        s = max(b.s, 0)
        cur = (b.p[r0:r1, :cx1] == 0)
        old = (b.pprev2[r0:r1, s:cx1 + s] == 0)
        w = min(cur.shape[1], old.shape[1])
        d = cur[:, :w] ^ old[:, :w]
        # mask only Mario's own rows (a bill passing right above his head must stay visible)
        d[max(y0 - 3 - r0, 0):max(y1 + 3 - r0, 0), max(x0 - 6, 0):x1 + 7] = False
        dr = d[:, max(x0 - 4, 0):]
        if int(dr.sum()) < 6:
            return False
        ys, xs = np.nonzero(dr)
        return (int(ys.max()) + r0) < feet - margin

    def no_pit_behind(self, b):
        mar = b.mario
        if mar is None or b.p is None:
            return False
        x0, x1, y0, y1 = mar
        gy = min(y1 + 3, 239)
        reg = b.p[gy:240, max(0, x0 - 28):max(1, x0)]
        if reg.size == 0:
            return False
        sky = (reg[..., 0] == 104) & (reg[..., 1] == 136) & (reg[..., 2] == 252)
        return int(sky.all(axis=0).sum()) < 2

    def blk_count(self, b):
        """number of black pixels that changed (moving dark sprite) right of / above Mario this frame"""
        mar = b.mario
        if mar is None or b.p is None or b.pprev2 is None:
            return 0
        x0, x1, y0, y1 = mar
        r0, r1 = max(y1 - 100, 32), max(y1 + 6, 33)
        cx1 = min(x1 + 110, 248)
        s = max(b.s, 0)
        cur = (b.p[r0:r1, :cx1] == 0)
        old = (b.pprev2[r0:r1, s:cx1 + s] == 0)
        w = min(cur.shape[1], old.shape[1])
        d = cur[:, :w] ^ old[:, :w]
        d[max(y0 - 3 - r0, 0):max(y1 + 3 - r0, 0), max(x0 - 6, 0):x1 + 7] = False
        return int(d[:, max(x0 - 4, 0):].sum())

    def bill_left(self, b):
        mar = b.mario
        if mar is None or b.p is None or b.pprev2 is None:
            return False
        x0, x1, y0, y1 = mar
        r0, r1 = max(y0 - 8, 32), min(y1 + 4, 236)
        lo = max(x0 - BL_REACH, 0)
        hi = max(x0 - 2, lo + 1)
        s = max(b.s, 0)
        cur = (b.p[r0:r1, lo:hi] == 0)
        old = (b.pprev2[r0:r1, lo + s:hi + s] == 0)
        w = min(cur.shape[1], old.shape[1])
        if w <= 0:
            return False
        d = cur[:, :w] ^ old[:, :w]
        return int(d.sum()) >= BL_MIN

    def bill_split(self, b):
        """(gap_high, gap_low) in px ahead of Mario of moving black sprites in head-height / foot rows"""
        mar = b.mario
        if mar is None or b.p is None or b.pprev2 is None:
            return None, None
        x0, x1, y0, y1 = mar
        r0, r1 = max(y1 - 54, 32), max(y1 - 1, 33)
        cx1 = min(x1 + 110, 248)
        s = max(b.s, 0)
        cur = (b.p[r0:r1, :cx1] == 0)
        old = (b.pprev2[r0:r1, s:cx1 + s] == 0)
        w = min(cur.shape[1], old.shape[1])
        d = cur[:, :w] ^ old[:, :w]
        d[:, max(x0 - 6, 0):x1 + 7] = False
        dr = d[:, x1 + 7:]
        hi = dr[:max(y1 - 22 - r0, 0)]
        lo = dr[max(y1 - 14 - r0, 0):]
        gh = gl = None
        if hi.size and int(hi.sum()) >= 4:
            gh = int(np.nonzero(hi.any(axis=0))[0][0]) + 7
        if lo.size and int(lo.sum()) >= 4:
            gl = int(np.nonzero(lo.any(axis=0))[0][0]) + 7
        return gh, gl

    def bill_any(self, b):
        mar = b.mario
        if mar is None or b.p is None or b.pprev2 is None:
            return False
        x0, x1, y0, y1 = mar
        feet = y1
        r0, r1 = max(feet - 100, 32), max(feet - 2, 33)
        c0 = 0
        cx1 = min(x1 + 100, 248)
        s = max(b.s, 0)
        cur = (b.p[r0:r1, c0:cx1] == 0)
        old = (b.pprev2[r0:r1, c0 + s:cx1 + s] == 0)
        w = min(cur.shape[1], old.shape[1])
        d = cur[:, :w] ^ old[:, :w]
        d[max(y0 - 3 - r0, 0):max(y1 + 3 - r0, 0), max(x0 - 6 - c0, 0):x1 + 7 - c0] = False
        return int(d.sum()) >= 6

    def plant_close_out(self, b, wx):
        if b.p is None:
            return False
        for (hx, hy, hw, hh) in b.plants(wx):
            cw = int(hx // 16) - 1
            gap = cw * 16 - (wx + _BODY_HW)
            if gap < -2 or gap > PLANT_GAP:
                continue
            r = int(round((hy + 13) / 16.0))
            x0 = int(cw * 16 + 6 - b.W)
            x1 = int(cw * 16 + 26 - b.W)
            y0 = max(32, r * 16 - 28)
            y1 = r * 16
            if x0 < 0 or x1 > 256 or y1 > 240:
                continue
            reg = b.p[y0:y1, x0:x1]
            ext = int((reg != _C_SKY).sum())
            self.pext = ext
            if ext >= 6:
                return True
            if self.bill_any(b):
                return True
        return False

    def shell_behind(self, b):
        mar = b.mario
        if mar is None or b.p is None:
            return False
        x0, x1, y0, y1 = mar
        lo = max(0, x0 - 48)
        if x0 - 1 <= lo:
            self.gprev = None
            return False
        sub = (b.p[max(32, y1 - 14):y1 + 1, lo:x0 - 1] == (168 << 8))
        if int(sub.sum()) < 6:
            self.gprev = None
            return False
        xs = np.nonzero(sub.any(axis=0))[0]
        right = lo + int(xs.max()) + b.W
        gap = (x0 + b.W) - right
        prev = self.gprev
        self.gprev = right
        if prev is not None and right - prev >= 1.5 and gap <= 34:
            return True
        return False

    def ground_green(self, b):
        mar = b.mario
        if mar is None or b.p is None:
            return None
        x0, x1, y0, y1 = mar
        xa = min(x1 + 1, 255)
        xb = min(x1 + HZ_LOOK, 252)
        if xb - xa < 4:
            return None
        sub = (b.p[170:216, xa:xb] == _C_GREEN)
        if int(sub.sum()) < 6:
            return None
        rows = np.nonzero(sub.any(axis=1))[0]
        if 170 + int(rows[-1]) < y1 - 8:
            return None
        cols = np.nonzero(sub.any(axis=0))[0]
        return int(cols[0])

    def run_wait_start(self, wx, yb, b):
        for i, (wa, wb, wn) in enumerate(RUN_WAITS):
            if wa <= wx <= wb and ("r", i) not in self.waited_set:
                self.waited_set.add(("r", i))
                self.wait_left = wn
                self.wait_mode = "r"
                b.log("f%d W%d wx%d y%d RUN-WAIT start %d" % (self.frame, b.W, wx, yb, wn))
                return True
        return False

    def act(self, img):
        keys, dead = self.act_inner(img)
        return keys, dead

    def act_inner(self, img):
        self.frame += 1
        b = self.bot
        o = self.old
        b.update(img)
        o.look_override = 0
        o.suppress = False
        if b.mario is not None:
            xl, xr, yt, yb0 = b.mario
            wx0 = (xl + xr) / 2.0 + b.W
            cw0 = int(wx0 // 16)
            o.noback = bool(NOBACK and NB_FROM <= wx0 <= NB_TO)
            rr = yb0 // 16 - 4
            if yb0 % 16 == 0 and any(b.solid(c, rr) for c in range(cw0 - 1, cw0 + 4)) \
                    and not any(b.solid(c, rr + 1) for c in range(cw0 - 1, cw0 + 4)):
                o.look_override = CEIL_LOOK
        self.flyer_update(b, o)
        self.bmh.append(self.blk_count(b))
        if len(self.bmh) > 12:
            del self.bmh[:4]
        keys, dead = o.act(img)
        info = self.state(b)
        if info is None:
            return keys, dead
        wx, yb, grounded = info
        self.g_now = grounded and self.pend == 0
        if self.sh > 0:
            self.sh -= 1
            o.hold_left = 0
            o.cool = 0
            self.last_hold = 0
            return {"x", "z"}, dead or o.frozen >= self.dead_after
        if SH_ON and self.stand and grounded and o.hold_left == 0 and self.pend == 0 and self.shn < SH_MAX \
                and wx < SH_TO and self.bill_left(b):
            self.shn += 1
            self.sh = SH_HOLD
            b.log("f%d W%d wx%d y%d STAND-HOP (bill from left)" % (self.frame, b.W, wx, yb))
            return {"x", "z"}, dead or o.frozen >= self.dead_after
        if self.stand and grounded and o.hold_left == 0 and "z" not in keys and self.pend == 0:
            if self.frame % 10 == 0:
                b.log("f%d W%d wx%d STAND for flyer" % (self.frame, b.W, wx))
            keys = {"x"}
            if FB_D and self.fl is not None and self.fl[0] <= FB_D and self.fbn < FB_MAX and self.no_pit_behind(b):
                keys = {"Left", "x"}
                self.fbn += 1
            self.last_hold = 0
            return keys, dead or o.frozen >= self.dead_after
        if self.shell_behind(b) and grounded and self.pend == 0 and o.hold_left == 0 and "z" not in keys:
            b.log("f%d W%d wx%d y%d JUMP[shell-behind]" % (self.frame, b.W, wx, yb))
            o.hold_left = 20
            keys = set(keys)
            keys.add("z")
            self.pend = 3
        if self.pend > 0:
            grounded = False
        st = dict(x=wx, y=float(yb), vx=max(-_VMAX, min(_VMAX, b.vx)), vy=b.vy, gr=grounded, ch=False, lag=0,
                  airborne=not grounded, xmin=b.W + 7.0)
        trajs = [([hx] * _T, [hy] * _T, hw, hh, None, 0.0) for (hx, hy, hw, hh) in b.plants(wx)]
        # valley: standing on a low step with a walker coming -> go back down to the floor first
        if VAL_MODE >= 2 and VB_FROM <= wx <= VB_TO and grounded and self.pend == 0 and o.hold_left == 0 \
                and 170 <= yb <= 196 and self.vb < VB_MAX and sum(self.bmh[-8:]) >= 6 \
                and self.bill_arc(b, 120, 120):
            self.vb += 1
            if self.vb % 10 == 1:
                b.log("f%d W%d wx%d y%d VALLEY-BACK %d bm%d" % (self.frame, b.W, wx, yb, self.vb, sum(self.bmh[-8:])))
            o.cool = 0
            self.last_hold = 0
            return {"Left", "x"}, dead
        # final staircase: a walker / bill comes down the steps -> stand on the floor, jump it when close
        if STAIRFIGHT and b.W >= FIGHT_FROM and grounded and yb >= FIGHT_YMIN and self.fw < FIGHT_MAX \
                and o.hold_left == 0 and self.pend == 0:
            if self.bill_arc(b, FIGHT_R, 150):
                self.fw += 1
                if self.bill_arc(b, FIGHT_NEAR, 60):
                    b.log("f%d W%d wx%d y%d STAIRFIGHT jump" % (self.frame, b.W, wx, yb))
                    keys = set(keys)
                    keys.add("z")
                    keys.add("Right")
                    o.hold_left = 25
                    o.reason = "fight"
                    self.pend = 3
                else:
                    if self.fw % 15 == 1:
                        b.log("f%d W%d wx%d y%d STAIRFIGHT wait" % (self.frame, b.W, wx, yb))
                    o.hold_left = 0
                    o.cool = 0
                    self.last_hold = 0
                    return {"x"}, dead
        # bill from the left at the staircase: hop over it
        if BL_ON:
            if self.blj > 0:
                self.blj -= 1
                o.hold_left = 0
                o.cool = 0
                self.last_hold = 0
                return {"x", "z", "Right"}, dead or o.frozen >= self.dead_after
            if self.blc > 0:
                self.blc -= 1
            if wx >= BL_FROM and grounded and self.pend == 0 and self.blc == 0 and o.hold_left == 0 \
                    and self.bill_left(b):
                b.log("f%d W%d wx%d y%d BILL-LEFT hop" % (self.frame, b.W, wx, yb))
                self.blj = BL_HOLD
                self.blc = BL_COOL + BL_HOLD
                o.back_left = 0
                self.back2 = 0
                self.wait_left = 0
                return {"x", "z", "Right"}, dead or o.frozen >= self.dead_after
        # low platform + ground hammer bro (wx~1330-1390): a hop cannot clear the bro (ceiling) -> keep distance
        if HZR_FROM <= wx <= HZR_TO and grounded and self.pend == 0 and self.hzr < HZR_MAX \
                and (o.hold_left == 0 or (self.last_hold == 0 and o.reason in ("threat", "bill"))):
            gg = self.ground_green(b)
            if gg is not None:
                self.hzr += 1
                o.hold_left = 0
                o.cool = 0
                o.back_left = 0
                self.last_hold = 0
                if gg <= HZR_NEAR:
                    if self.hzr % 10 == 1:
                        b.log("f%d W%d wx%d y%d HZ-RETREAT gg%d" % (self.frame, b.W, wx, yb, gg))
                    return {"Left", "x"}, dead or o.frozen >= self.dead_after
                if self.hzr % 20 == 1:
                    b.log("f%d W%d wx%d y%d HZ-STAND gg%d" % (self.frame, b.W, wx, yb, gg))
                return {"x"}, dead or o.frozen >= self.dead_after
        # hammer-bro zone: a bro drops to the ground in front of Mario -> stand, hop when it is adjacent
        if HAMMER:
            if self.hj > 0:
                self.hj -= 1
                o.hold_left = 0
                o.cool = 0
                self.last_hold = 0
                return ({"x", "z", "Right"} if HZ_FWD else {"x", "z"}), dead or o.frozen >= self.dead_after
            if (HZ_FROM <= wx <= HZ_TO or HZ2_FROM <= wx <= HZ2_TO) and grounded and self.pend == 0 and self.hz < HZ_MAX \
                    and (o.hold_left == 0 or (self.last_hold == 0 and o.reason in ("threat", "bill"))):
                gg = self.ground_green(b)
                if gg is not None:
                    self.hz += 1
                    o.hold_left = 0
                    o.cool = 0
                    o.back_left = 0
                    self.last_hold = 0
                    if gg <= HZ_HOP:
                        b.log("f%d W%d wx%d y%d HAMMER hop gg%d" % (self.frame, b.W, wx, yb, gg))
                        self.hj = HZ_HOLD
                        return ({"x", "z", "Right"} if HZ_FWD else {"x", "z"}), dead or o.frozen >= self.dead_after
                    if self.hz % 20 == 1:
                        b.log("f%d W%d wx%d y%d HAMMER stand gg%d" % (self.frame, b.W, wx, yb, gg))
                    return {"x"}, dead or o.frozen >= self.dead_after
        if self.pend > 0:
            grounded = False
        started = (o.hold_left > 0 or "z" in keys) and self.last_hold == 0 and "z" in keys and grounded and \
            self.pend == 0
        modok = b.W >= MOD_FROM
        # a flying Paratroopa is about to pass over / at Mario: a jump now flies into it -> stand (it passes over)
        if self.fl is None:
            self.fgn = 0
        elif FLYGATE and started and self.fgn < FG_MAX and FG_FROM <= wx <= FG_TO:
            fdx, fb, fvx = self.fl
            if -10 <= fdx <= FG_D and yb - FG_UP <= fb <= yb - 18:
                self.fgn += 1
                if self.fgn % 10 == 1:
                    b.log("f%d W%d wx%d y%d FLYGATE stand fdx%d fb%d" % (self.frame, b.W, wx, yb, fdx, fb))
                o.hold_left = 0
                o.cool = 0
                self.last_hold = 0
                return {"x"}, dead or o.frozen >= self.dead_after
        # stacked cannons: a high bill is about to pass over Mario while a low bill also comes -> a jump now
        # flies into the high one; keep running until the high bill has passed, then jump the low one
        if started and BD_ON and o.reason == "bill" and wx < BD_TO and self.bdn < BD_MAX:
            gh, gl = self.bill_split(b)
            if gh is not None and gh <= BD_H and (gl is None or gl > BD_L):
                self.bdn += 1
                if self.bdn % 5 == 1:
                    b.log("f%d W%d wx%d y%d BILL-DEFER gh%s gl%s" % (self.frame, b.W, wx, yb, gh, gl))
                o.hold_left = 0
                o.cool = 0
                self.last_hold = 0
                return {"Right", "x"}, dead or o.frozen >= self.dead_after
        if o.reason != "bill" and self.bdn and self.frame % 40 == 0:
            self.bdn = 0
        # world-x keyed waits before the jumps ahead of the final staircase
        if started and STAIR_WAIT and self.wait_left == 0:
            for i, (wa, wb, wn) in enumerate(STAIR_WAITS):
                if wa <= wx <= wb and i not in self.waited_set:
                    self.waited_set.add(i)
                    self.wait_left = wn
                    self.wait_mode = "x"
                    b.log("f%d W%d wx%d y%d STAIR-WAIT start %d" % (self.frame, b.W, wx, yb, wn))
                    break
        if started and RUN_WAIT and self.wait_left == 0 and self.run_wait_start(wx, yb, b):
            pass
        if started and STAIR_BACK and self.back2 == 0 and self.nback2 < 3 and self.wait_left == 0:
            for (wa, wb, wn) in STAIR_BACKS:
                if wa <= wx <= wb:
                    bucket = int(wx // 30)
                    if bucket not in self.back_done:
                        self.back_done.add(bucket)
                        self.nback2 += 1
                        self.back2 = wn
                        b.log("f%d W%d wx%d y%d STAIR-BACK start %d" % (self.frame, b.W, wx, yb, wn))
                    break
        if self.back2 > 0 and grounded and self.pend == 0:
            self.back2 -= 1
            o.hold_left = 0
            o.cool = 0
            o.back_left = 0
            self.last_hold = 0
            return {"Left", "x"}, dead or o.frozen >= self.dead_after
        if self.wait_left > 0 and grounded and self.pend == 0:
            self.wait_left -= 1
            o.hold_left = 0
            o.cool = 0
            o.back_left = 0
            self.last_hold = 0
            return ({"x"} if self.wait_mode == "x" else {"Right", "x"}), dead or o.frozen >= self.dead_after
        # block + cannon complex: don't enter the corridor while a bill is flying through it
        if BILLWAIT and started and o.reason == "threat" and (BW_FROM <= wx <= BW_TO or BW2_FROM <= wx <= BW2_TO
                                                              or BW3_FROM <= wx <= BW3_TO) \
                and self.bw < BW_MAX \
                and self.bill_arc(b, 150, 45):
            self.bw += 1
            if self.bw % 15 == 1:
                b.log("f%d W%d wx%d y%d BILL-WAIT %d" % (self.frame, b.W, wx, yb, self.bw))
            o.hold_left = 0
            o.cool = 0
            self.last_hold = 0
            return {"x"}, dead or o.frozen >= self.dead_after
        if started and o.reason == "threat" and USE_PWAIT and self.pveto < 220 and self.plant_close_out(b, wx):
            self.pveto += 1
            if self.pveto % 20 == 1:
                b.log("f%d W%d wx%d PLANT-WAIT %d ext%d" % (self.frame, b.W, wx, self.pveto, self.pext))
            o.hold_left = 0
            o.cool = 0
            self.last_hold = 0
            self.last_pveto = self.frame
            return {"x"}, dead or o.frozen >= self.dead_after
        if self.pveto and self.frame - self.last_pveto > 60:
            self.pveto = 0
        if self.sveto and self.frame - self.last_sveto > 20:
            self.sveto = 0
        stairs = wx >= STAIRS_X
        hb = False
        if started and USE_HIGHBILL and o.reason == "threat" and not o.look_override:
            hb = self.bill_high_only(b, 110, 3 if stairs else 17)
            if stairs:
                if hb:
                    self.hb_until = self.frame + 15
                elif self.frame <= self.hb_until:
                    hb = True
        if hb and stairs and VAL_MODE >= 1:
            sp = sum(self.bmh[-5:]) / 5.0
            if sp < VAL_FAST:
                hb = False
                self.hb_until = -99
        if hb and b.W < STAIR_FROM and self.sveto < (STAIR_CAP if stairs else 45):
            self.sveto += 1
            self.last_sveto = self.frame
            if self.sveto % 10 == 1:
                b.log("f%d W%d wx%d y%d HIGHBILL-VETO %d" % (self.frame, b.W, wx, yb, self.sveto))
            o.hold_left = 0
            o.cool = 0
            self.last_hold = 0
            return {"Right", "x"}, dead
        # low brick ceiling over a ground walker: tap-jump early so the arc stays under the bricks
        if LOWCEIL and grounded and not started and o.hold_left == 0 and self.pend == 0 and "z" not in keys \
                and LOW_FROM <= wx <= LOW_TO and b.W >= 700 and self.low_ceiling(b, wx, yb):
            g2 = self.green_gap(b)
            if g2 is not None and g2 <= LOW_TRIG:
                if RUN_WAIT and self.wait_left == 0 and self.run_wait_start(wx, yb, b):
                    self.wait_left -= 1
                    return {"Right", "x"}, dead
                b.log("f%d W%d wx%d y%d LOWTAP early gap%s" % (self.frame, b.W, wx, yb, g2))
                keys = set(keys)
                keys.add("z")
                keys.add("Right")
                o.hold_left = LOW_TAP - 1
                self.pend = 3
                self.last_hold = o.hold_left
                return keys, dead
        lowc = False
        gap = None
        if LOWCEIL and started and o.reason in ("threat", "bill"):
            lowc = (LOW_FROM <= wx <= LOW_TO) and self.low_ceiling(b, wx, yb)
        if lowc:
            gap = self.green_gap(b)
            if gap is not None and gap > LOW_GAP:
                o.hold_left = 0
                o.cool = 0
                self.last_hold = 0
                self.ncancel += 1
                if self.ncancel % 6 == 1:
                    b.log("f%d W%d wx%d y%d LOWCEIL wait gap%s" % (self.frame, b.W, wx, yb, gap))
                return {"Right", "x"}, dead
            b.log("f%d W%d wx%d y%d LOWCEIL tap gap%s" % (self.frame, b.W, wx, yb, gap))
            o.hold_left = LOW_TAP - 1
            self.pend = 3
        elif started and not modok:
            if EARLY_FROM <= wx <= EARLY_TO and o.reason in ("threat", "bill"):
                o.hold_left = min(o.hold_left, EARLY_HOLD - 1)
            self.pend = 3
            b.log("f%d W%d wx%d y%d vx%.1f J0[%s] hold%d" % (self.frame, b.W, wx, yb, b.vx, o.reason, o.hold_left))
        elif started:
            best = None
            first_safe = None
            res = []
            vis = 255 - (b.mario[1] if b.mario else 130) - 10
            for k in HOLD_ORDER:
                plan = (k, 0, 0)
                c1, i1 = b.sim(plan, st, trajs)
                if c1 >= 3000:
                    res.append((k, "X"))
                    continue
                c2, i2 = b.sim(plan, st, trajs, vs=0.88, hwx=0.5)
                c3, i3 = b.sim(plan, st, trajs, vs=1.1, hwx=1.0)
                if c2 >= 3000 or c3 >= 3000:
                    res.append((k, "r"))
                    continue
                ld = i1.get("land")
                res.append((k, ld))
                if first_safe is None:
                    first_safe = k
                if ld is not None and ld[0] <= vis:
                    best = k
                    break
            if best is None and first_safe is not None:
                best = first_safe
            if best is None:
                best = 26
                tag = "KEEP"
            else:
                tag = "SET"
            b.log("f%d W%d wx%d y%d J[%s] %s k%d %s" % (self.frame, b.W, wx, yb, o.reason, tag, best, res))
            o.hold_left = max(0, best - 1)
            self.pend = 3
        elif modok and grounded and "z" not in keys and o.hold_left == 0 and o.cool == 0 and o.back_left == 0 \
                and self.pend == 0:
            c, inf = b.sim((0, 0, 0), st, trajs)
            if c >= 3000 and ("death" in inf or "coll" in inf or (PLANTSIM and trajs and inf.get("wall"))):
                th = inf.get("coll", inf.get("death", 0))
                if "wall" in inf:
                    th = 0
                lim = SIM_LATE
                if PLANTSIM and trajs:
                    lim = 99
                if th <= lim:
                    best = None
                    for k in HOLD_ORDER:
                        c1, _ = b.sim((k, 0, 0), st, trajs)
                        if c1 >= 3000:
                            continue
                        if lim == 99:
                            c2, _ = b.sim((k, 0, 0), st, trajs, vs=0.88, hwx=0.5)
                            c3, _ = b.sim((k, 0, 0), st, trajs, vs=1.1, hwx=1.0)
                            if c2 >= 3000 or c3 >= 3000:
                                continue
                        best = k
                        break
                    if best is not None:
                        b.log("f%d W%d wx%d y%d JUMP[simlate] k%d th%.0f" % (self.frame, b.W, wx, yb, best, th))
                        o.hold_left = best - 1
                        keys = set(keys)
                        keys.add("z")
                        self.pend = 3
        elif USE_RELEASE and modok and not grounded and o.hold_left > 0 and "z" in keys and self.pend == 0:
            st["ch"] = b.vy > 0.3
            c1, _ = b.sim((o.hold_left + 1, 0, 0), st, trajs)
            if c1 >= 3000:
                c0, _ = b.sim((0, 0, 0), st, trajs)
                if c0 < 3000:
                    b.log("f%d W%d wx%d y%d RELEASE hold_left%d" % (self.frame, b.W, wx, yb, o.hold_left))
                    o.hold_left = 0
                    keys = set(keys)
                    keys.discard("z")
        # stale hold: z still held after landing -> release so a new jump can start
        if "z" in keys and grounded and self.pend == 0 and not started and o.hold_left > 0:
            self.gz += 1
            if self.gz >= 2:
                b.log("f%d W%d wx%d y%d STALE-HOLD release hold_left%d" % (self.frame, b.W, wx, yb, o.hold_left))
                o.hold_left = 0
                o.cool = 0
                keys = set(keys)
                keys.discard("z")
                self.gz = 0
        else:
            self.gz = 0
        if self.pend > 0 and not started:
            self.pend -= 1
        self.last_hold = o.hold_left if "z" in keys else 0
        return keys, dead or o.frozen >= self.dead_after


def solve(sim):
    return []


def play(sim):
    b = Hyb()
    n = 0
    while n < 8000:
        img = sim.screen()
        try:
            keys, dead = b.act(img)
        except Exception:
            import traceback
            traceback.print_exc()
            keys, dead = {"Right", "x"}, False
        if dead:
            return
        try:
            sim.step(keys)
        except Exception:
            return
        n += 1

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    stuck = False\n    try:\n        stuck = bool(ctx["stuck"]["is"])\n    except Exception:\n        pass\n    status = "stuck" if stuck else "running"\n    threats = []\n    try:\n        band = np.asarray(frame)[150:208, 100:230]\n        green = int(((band[..., 0] == 0) & (band[..., 1] == 168) & (band[..., 2] == 0)).sum())\n        black = int((band.sum(2) == 0).sum())\n        if green >= 6:\n            threats.append("green_enemy_near")\n        if black >= 120:\n            threats.append("bill_or_cannon_near")\n        if threats and not stuck:\n            status = "danger"\n    except Exception:\n        pass\n    return {"status": status, "where": {"hazard": threats[0] if threats else None, "confidence": 0.3 if threats else 0.0},\n            "progress": None, "resources": {}, "threats": threats, "stuck": {"is": stuck},\n            "outlook": {"label": "bad" if threats else "ok", "p_through": 0.25 if threats else 0.4}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

