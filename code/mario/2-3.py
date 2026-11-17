# ==================================================================================================
# Super Mario Bros. (NES) -- level 2-3  |  超级马里奥兄弟（NES） 关 2-3
# Pass rate on 100 random variants never used before: 82/100; fastest finish 1268 frames (21.13 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 82/100；最快一局 1268 帧（60 帧/秒，约 21.13 秒）。
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
#   closed-loop vision policy; measured jump tables + exact ballistic cheep model (apex y=62, g=.0
#   9) + per-frame rollout planner over {run, stop-k-then-run, hop(H), edge-gap-jump(H)} choosing 
#   the first ch
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def hkey                           line 69    
#   def jy                             line 73    
#   def land_k                         line 80    
#   def cmask                          line 93    
#   def bgmask                         line 97    
#   def terrmask                       line 104   
#   class Tracker                      line 109   
#   def find_cheeps                    line 150   
#   class Cheeps                       line 177   tracks in WORLD x; every flying cheep follows y(t)=62+0.045*(t+ph)^2 with constant world vx
#   def hit_time                       line 244   
#   def shift_est                      line 259   
#   def margin_at                      line 273   
#   def landing_info                   line 289   jump with hold H taken off at screen x=cx, feet y=gy, ground speed v.
#   def air_part                       line 327   
#   def room_ahead                     line 335   is there enough solid floor after the landing column c to slide to a stop?
#   def jump_hit                       line 345   first collision time of a jump followed by either running on or braking (None = a safe continuation exists)
#   def scan_gap                       line 363   
#   class Bot                          line 383   
#   def play                           line 721   
#   def solve                          line 743   
# ==================================================================================================

# idea: closed-loop vision policy; measured jump tables + exact ballistic cheep model (apex y=62, g=.09) + per-frame rollout planner over {run, stop-k-then-run, hop(H), edge-gap-jump(H)} choosing the first cheep-safe, terrain-valid plan
import math
import numpy as np

BG_COLORS = [(0x68, 0x88, 0xfc), (252, 252, 252), (60, 188, 252)]
RED = (248, 56, 0)
BROWN = (172, 124, 0)
LIGHT = (240, 208, 176)
COINO = (252, 160, 68)

V_RUN = 3.04
ACC = 0.1
TRIG = 8
TH = 74
DX, DY = 19.0, 17.0
CG2 = 0.045          # half gravity of the cheeps
CAPEX = 62.0         # apex y of every cheep arc
GAP_H = (1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18, 21, 24, 28)
HOPS = (14, 8, 22, 4, 28, 2, 11, 18)
S_KS = (3, 6, 10, 14, 19, 25, 32, 40, 50)
LEVELS_REL = (-64, -48, -32, -16, 0, 16, 32, 48, 64)

# ---------------- measured jump tables (feet y relative to take-off, per frame since the press) ----------------
_P = [-6, -12, -18, -23, -28, -33, -38, -42, -47, -51, -54, -58, -61, -65, -68, -70, -73, -75, -77, -79, -81, -82,
      -83, -84, -85, -85, -86, -86, -86]
_RAW = {
    1: (20, [-17, -21, -25, -27, -29, -30, -30, -29, -28, -25, -22, -18, -13, -8, -3, 0]),
    2: (22, [-27, -30, -32, -33, -34, -34, -33, -31, -29, -25, -21, -16, -11, -6, -1, 0]),
    3: (23, [-31, -34, -36, -38, -38, -38, -37, -35, -32, -28, -24, -19, -13, -8, -3, 0]),
    4: (24, [-36, -39, -41, -42, -42, -41, -40, -38, -35, -31, -26, -21, -15, -8, -3, 0]),
    5: (26, [-37, -41, -43, -45, -46, -46, -45, -43, -41, -38, -34, -29, -24, -19, -14, -9, -4, 0]),
    6: (28, [-45, -47, -49, -49, -49, -48, -46, -44, -40, -36, -31, -26, -21, -16, -11, -6, -1, 0]),
    7: (29, [-46, -49, -51, -52, -53, -52, -51, -49, -46, -42, -38, -33, -27, -22, -17, -12, -7, -2, 0]),
    8: (30, [-50, -53, -54, -55, -56, -55, -54, -51, -48, -44, -40, -35, -30, -25, -20, -15, -10, -5, 0]),
    9: (32, [-56, -58, -59, -59, -58, -56, -54, -50, -46, -41, -36, -31, -26, -21, -16, -11, -6, -1, 0]),
    10: (33, [-57, -60, -61, -62, -61, -60, -58, -56, -52, -48, -43, -38, -33, -28, -23, -18, -13, -8, -3, 0]),
    11: (34, [-63, -64, -64, -64, -63, -60, -58, -54, -49, -44, -38, -33, -28, -23, -18, -13, -8, -3, 0]),
    12: (35, [-64, -66, -67, -67, -66, -65, -62, -59, -55, -50, -45, -40, -35, -30, -25, -20, -15, -10, -5, 0]),
    13: (37, [-67, -68, -69, -69, -68, -66, -64, -61, -56, -51, -46, -41, -36, -31, -26, -21, -16, -11, -6, -1, 0]),
    14: (38, [-71, -71, -71, -70, -68, -65, -62, -57, -52, -46, -41, -36, -31, -26, -21, -16, -11, -6, -1, 0]),
    15: (38, [-72, -73, -74, -73, -72, -69, -67, -63, -58, -53, -47, -39, -34, -29, -24, -19, -14, -9, -4, 0]),
    16: (40, [-74, -75, -75, -75, -73, -71, -67, -63, -59, -54, -49, -44, -39, -34, -29, -24, -19, -14, -9, -4, 0]),
    17: (41, [-76, -77, -77, -76, -74, -72, -68, -64, -59, -54, -49, -44, -39, -34, -29, -24, -19, -14, -9, -4, 0]),
    18: (42, [-78, -79, -78, -77, -75, -72, -69, -64, -59, -53, -48, -43, -38, -33, -28, -23, -18, -13, -8, -3, 0]),
    19: (43, [-80, -80, -80, -78, -76, -73, -69, -64, -59, -53, -46, -41, -36, -31, -26, -21, -16, -11, -6, -1, 0]),
    20: (44, [-81, -81, -81, -79, -76, -73, -69, -64, -59, -54, -49, -44, -39, -34, -29, -24, -19, -14, -9, -4, 0]),
    21: (45, [-83, -82, -81, -79, -77, -73, -69, -64, -59, -54, -49, -44, -39, -34, -29, -24, -19, -14, -9, -4, 0]),
    22: (46, [-84, -83, -82, -80, -77, -73, -69, -63, -57, -52, -47, -42, -37, -32, -27, -22, -17, -12, -7, -2, 0]),
    23: (47, [-84, -84, -82, -80, -77, -73, -68, -63, -58, -53, -48, -43, -38, -33, -28, -23, -18, -13, -8, -3, 0]),
    24: (48, [-84, -82, -80, -76, -72, -67, -62, -57, -52, -47, -42, -37, -32, -27, -22, -17, -12, -7, -2, 0]),
    25: (49, [-85, -84, -82, -79, -76, -72, -66, -61, -56, -51, -46, -41, -36, -31, -26, -21, -16, -11, -6, -1, 0]),
    26: (49, [-85, -84, -82, -79, -75, -71, -65, -59, -52, -47, -42, -37, -32, -27, -22, -17, -12, -7, -2, 0]),
    27: (50, [-85, -84, -81, -78, -74, -69, -64, -59, -54, -49, -44, -39, -34, -29, -24, -19, -14, -9, -4, 0]),
    28: (51, [-85, -83, -80, -77, -73, -68, -63, -58, -53, -48, -43, -38, -33, -28, -23, -18, -13, -8, -3, 0]),
}
JT = {}
for _h, (_land, _tail) in _RAW.items():
    _n = _land - 2 - len(_tail)
    JT[_h] = [0, 0, 0] + _P[:_n] + _tail

# cumulative distance when releasing "Right" while running at full speed
STOPC = [0, 3, 6, 8, 11, 14, 16, 19, 22, 24, 27, 29, 31, 33, 35, 37, 39, 41, 42, 44, 45, 46, 47, 48, 49, 49, 50]
STV = []
for _k in range(27):
    _a, _b = max(_k - 3, 0), min(_k + 3, 26)
    STV.append((STOPC[_b] - STOPC[_a]) / float(_b - _a))


def hkey(H):
    return min(max(int(H), 1), 28)


def jy(H, k):
    tb = JT[hkey(H)]
    if k < len(tb):
        return tb[k]
    return 5 * (k - len(tb) + 1)


def land_k(H, dl):
    tb = JT[hkey(H)]
    ap = min(range(len(tb)), key=lambda i: tb[i])
    if dl < 0 and tb[ap] > dl:
        return None
    k = ap
    while k < 140:
        if jy(H, k) >= dl:
            return k
        k += 1
    return None


def cmask(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def bgmask(s):
    m = np.zeros(s.shape[:2], bool)
    for c in BG_COLORS:
        m |= cmask(s, c)
    return m


def terrmask(s):
    m = bgmask(s) | cmask(s, RED) | cmask(s, COINO) | cmask(s, BROWN)
    return ~m


class Tracker:
    def __init__(self):
        self.cx = 45
        self.cy = 200
        self.lost = 0

    def find(self, s):
        b = cmask(s, BROWN)
        b[:48] = False
        y0, y1 = max(48, self.cy - 45), min(240, self.cy + 45)
        x0, x1 = max(0, self.cx - 40), min(256, self.cx + 40)
        sub = b[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        if len(ys) < 5:
            self.lost += 1
            if self.lost > 4:
                ys, xs = np.nonzero(b)
                if len(ys) < 5:
                    return None
                x0 = y0 = 0
            else:
                return None
        self.lost = 0
        ys = ys + y0
        xs = xs + x0
        mx, my = int(np.median(xs)), int(np.median(ys))
        sel = (np.abs(xs - mx) < 12) & (np.abs(ys - my) < 14)
        ys, xs = ys[sel], xs[sel]
        xl, xr = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        base = max(0, ymin - 6)
        r = cmask(s[base:ymax + 4, max(0, xl - 3):xr + 4], RED)
        ry = np.nonzero(r.any(1))[0]
        if len(ry):
            ymin = min(ymin, int(ry.min()) + base)
            ymax = max(ymax, int(ry.max()) + base)
        self.cx = (xl + xr) // 2
        self.cy = (ymin + ymax) // 2
        return xl, xr, ymin, ymax


def find_cheeps(s, mbox):
    r = cmask(s, RED)
    r[:48] = False
    xl, xr, yt, yb = mbox
    r[max(0, yt - 4):yb + 5, max(0, xl - 5):xr + 6] = False
    ys, xs = np.nonzero(r)
    if len(ys) == 0:
        return []
    cl = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        for c in cl:
            if c[0] - 5 <= x <= c[1] + 5 and c[2] - 5 <= y <= c[3] + 5:
                c[0] = min(c[0], x)
                c[1] = max(c[1], x)
                c[2] = min(c[2], y)
                c[3] = max(c[3], y)
                c[4] += 1
                break
        else:
            cl.append([x, x, y, y, 1])
    out = []
    for c in cl:
        if c[4] >= 6:
            out.append(((c[0] + c[1]) / 2.0, (c[2] + c[3]) / 2.0, c[4]))
    return out


class Cheeps:
    """tracks in WORLD x; every flying cheep follows y(t)=62+0.045*(t+ph)^2 with constant world vx"""

    def __init__(self):
        self.tracks = []
        self.S = 0.0

    def update(self, det, S):
        self.S = S
        used = set()
        new = []
        for (x, y, n) in det:
            xw = x + S
            best, bd = None, 26.0
            for i, tk in enumerate(self.tracks):
                if i in used:
                    continue
                d = abs(tk["xw"] + tk["vx"] * (1 + tk["miss"]) - xw) + abs(tk["y"] + tk["dy"] * (1 + tk["miss"]) - y)
                if d < bd:
                    bd, best = d, i
            if best is None:
                rising = (y > 150) or (x > 215)
                new.append(dict(xw=xw, y=y, hx=[xw], hy=[y], vx=1.9, dy=(-4.0 if rising else 2.0), rising=rising,
                                n=1, miss=0))
            else:
                used.add(best)
                tk = self.tracks[best]
                tk["hx"] = (tk["hx"] + [xw])[-14:]
                tk["hy"] = (tk["hy"] + [y])[-14:]
                tk["xw"], tk["y"] = xw, y
                tk["n"] += 1
                tk["miss"] = 0
                k = len(tk["hx"])
                if k >= 4:
                    u = np.arange(k, dtype=float)
                    vx = float(np.polyfit(u, np.array(tk["hx"]), 1)[0])
                    for cv in (1.70, 1.94, 2.75):
                        if abs(vx - cv) < 0.2:
                            vx = cv
                            break
                    tk["vx"] = vx
                if k >= 3:
                    d = tk["hy"][-1] - tk["hy"][-3]
                    tk["dy"] = d / 2.0
                    if d < -0.8:
                        tk["rising"] = True
                    elif d > 0.8:
                        tk["rising"] = False
                new.append(tk)
        for i, tk in enumerate(self.tracks):
            if i not in used and tk["miss"] < 3:
                tk["miss"] += 1
                tk["xw"] += tk["vx"]
                tk["y"] += tk["dy"]
                new.append(tk)
        self.tracks = new

    def preds(self):
        """static parameters: (xs, vx, ph, extra_dx) with xs in current-screen coordinates"""
        out = []
        for tk in self.tracks:
            s = math.sqrt(max(tk["y"] - CAPEX, 0.0) / CG2)
            ph = -s if tk["rising"] else s
            out.append((tk["xw"] - self.S, tk["vx"], ph, 0.04 if tk["n"] < 8 else 0.012))
        return out


def hit_time(X, Y, preds, m):
    n = len(X)
    for t in range(1, n):
        my = Y[t] - 8.0
        mxx = X[t]
        for (xs, vx, ph, gr) in preds:
            cxp = xs + vx * t
            if abs(cxp - mxx) < DX * m + gr * t:
                q = t + ph
                cyp = CAPEX + CG2 * q * q
                if abs(cyp - my) < DY * m:
                    return t
    return None


def shift_est(pt, ct, last):
    sc = []
    for sh in range(0, 7):
        if sh:
            a = pt[:, sh:]
            b = ct[:, :256 - sh]
        else:
            a, b = pt, ct
        sc.append(int((a != b).sum()))
    best = min(sc)
    ties = [sh for sh, v in enumerate(sc) if v <= best + 3]
    return min(ties, key=lambda sh: abs(sh - last))


def margin_at(gcol, c):
    c = int(c)
    if c >= 244:
        return 30
    if c < 6:
        return -1
    if not gcol[c - 5:c + 6].all():
        return -1
    m = 30
    for d in range(6, 30):
        if (c - d >= 0 and not gcol[c - d]) or (c + d < 256 and not gcol[c + d]):
            m = d
            break
    return m


def landing_info(terr, wallm, cx, gy, H, v):
    """jump with hold H taken off at screen x=cx, feet y=gy, ground speed v.
    returns (best_margin, level_delta, land_col, land_frame); margin -1 = no valid landing."""
    best = (-1, None, None, None)
    for dl in LEVELS_REL:
        L = int(gy) + dl
        if L < 100 or L > 232:
            continue
        k = land_k(H, dl)
        if k is None:
            continue
        c = int(cx + v * k)
        if c >= 244:
            mg = 6
        else:
            lc = terr[L + 2:L + 5].any(0)
            mg = margin_at(lc, c)
            if mg < 0:
                continue
            w = wallm[max(0, L - 9):L - 1, max(0, c - 3):c + 4]
            if w.sum() >= 4:
                continue
            if dl < 0:
                a = c
                while a > 0 and lc[a - 1]:
                    a -= 1
                if a > cx:
                    te = int((a - cx - 6) / max(v, 1.0))
                    if te >= 0 and jy(H, te) > dl - 3:
                        continue
        if mg > best[0]:
            best = (mg, dl, c, k)
    return best


TAIL = 26


def air_part(x0, v, gy, H, dl, kl):
    xs, ys = [], []
    for j in range(1, kl + 1):
        xs.append(x0 + v * j)
        ys.append(gy + (jy(H, j) if j < kl else dl))
    return xs, ys


def room_ahead(terr, gy, dl, c, v):
    """is there enough solid floor after the landing column c to slide to a stop?"""
    if c >= 244:
        return True
    L = int(gy) + dl
    lc = terr[L + 2:L + 5].any(0)
    n = int(STOPC[26] * min(1.0, v / V_RUN)) + 6
    return bool(lc[c:min(c + n, 244)].all())


def jump_hit(pr, m, X0, Y0, xs, ys, v, yl, room):
    """first collision time of a jump followed by either running on or braking (None = a safe continuation exists)"""
    ratio = min(1.0, v / V_RUN)
    xl = xs[-1] if xs else X0[-1]
    Xr = X0 + xs + [xl + v * i for i in range(1, TAIL + 1)]
    Yr = Y0 + ys + [yl] * TAIL
    h1 = hit_time(Xr, Yr, pr, m)
    if h1 is None:
        return None
    if room:
        Xs = X0 + xs + [xl + STOPC[min(i, 26)] * ratio for i in range(1, TAIL + 1)]
        h2 = hit_time(Xs, Yr, pr, m)
        if h2 is None:
            return None
        return max(h1, h2)
    return h1


def scan_gap(colsolid, c0):
    n = 256
    c = max(0, c0)
    E = None
    F = None
    while c < n:
        if not colsolid[c]:
            d = c
            while d < n and not colsolid[d]:
                d += 1
            if d - c >= 7 or d >= n:
                E = c
                F = d if d < n else None
                break
            c = d
        else:
            c += 1
    return E, F


class Bot:
    def __init__(self):
        self.tr = Tracker()
        self.ch = Cheeps()
        self.hist_cx = []
        self.hist_b = []
        self.gy = 161.0
        self.jt = None
        self.H = 0
        self.fixed = False
        self.last_screen = None
        self.same = 0
        self.vmeas = V_RUN
        self.last_cmd = "run"
        self.last_z = False
        self.nfr = 0
        self.prev_terr = None
        self.last_shift = 3
        self.prev_cx = None
        self.events = []
        self.finished = False
        self.S = 0.0
        self.stuckn = 0
        self.wallm = None
        self.nstop = 0
        self.since_stop = 99
        self.cache = {}

    def log(self, *a):
        print("f%d" % self.nfr, *a)

    # ---------- planning helpers ----------
    def edge_options(self, ctx, x, v):
        key = (int(x), int(v * 10))
        if key in self.cache:
            return self.cache[key]
        plans = []
        for h in GAP_H:
            mg, dl, c, kl = landing_info(ctx["terr"], self.wallm, x, ctx["gy"], h, v)
            if mg >= 6:
                plans.append((-min(mg, 12), c, h, dl, kl, mg))
        plans.sort()
        self.cache[key] = plans
        return plans

    def rollout(self, ctx, kind, arg, m):
        cx, gy, v0, E, pr = ctx["cx"], ctx["gy"], ctx["v0"], ctx["E"], ctx["pr"]
        X = [float(cx)]
        Y = [float(gy)]
        terr = ctx["terr"]
        if kind == "hop":
            v = max(v0, 1.0)
            mg, dl, c, kl = landing_info(terr, self.wallm, cx, gy, arg, v)
            if mg < 6 or c >= 244:
                return None
            xs, ys = air_part(cx, v, gy, arg, dl, kl)
            hit = jump_hit(pr, m, X, Y, xs, ys, v, gy + dl, room_ahead(terr, gy, dl, c, v))
            return dict(hit=hit, H=arg, jump=True, t0=0)
        k1 = arg if kind == "stop" else 0
        ratio = min(1.0, v0 / V_RUN)
        x, v = float(cx), float(v0)
        for t in range(1, TH + 1):
            if (t - 1) >= k1 and E is not None and (E - x) <= TRIG:
                vv = max(v, 1.0)
                opts = self.edge_options(ctx, x, vv)
                if not opts:
                    return None
                best = None
                for (_, c, h, dl, kl, mg) in opts[:5]:
                    xs, ys = air_part(x, vv, gy, h, dl, kl)
                    hit = jump_hit(pr, m, X, Y, xs, ys, vv, gy + dl, room_ahead(terr, gy, dl, c, vv))
                    if hit is None:
                        return dict(hit=None, H=h, jump=True, t0=t - 1)
                    if best is None or hit > best[0]:
                        best = (hit, h)
                return dict(hit=best[0], H=best[1], jump=True, t0=t - 1)
            if (t - 1) < k1:
                x = cx + STOPC[min(t, 26)] * ratio
                v = STV[min(t, 26)] * ratio
            else:
                v = min(V_RUN, max(v, 0.0) + ACC)
                x += v
            if E is not None and x > E + 1:
                return None
            X.append(x)
            Y.append(float(gy))
        return dict(hit=hit_time(X, Y, pr, m), H=None, jump=False, t0=None)

    def decide_ground(self, ctx):
        """returns (action, H, why) with action in run/stop/jump"""
        self.cache = {}
        cands = [("run", 0)]
        if ctx["v0"] > 0.2 or True:
            for k in S_KS:
                cands.append(("stop", k))
        for h in HOPS:
            cands.append(("hop", h))
        results = {}
        for m in (1.0, 0.8, 0.6):
            results = {}
            for c in cands:
                r = self.rollout(ctx, c[0], c[1], m)
                results[c] = r
                if r is not None and r["hit"] is None:
                    return self.as_action(c, r), m
            if not ctx["pr"]:
                break
        # nothing is safe.  At a gap edge a missed landing is certain death while a predicted cheep hit is only a
        # probability, so take the jump with the best landing margin.
        E0 = ctx["E"]
        if E0 is not None and (E0 - ctx["cx"]) <= TRIG:
            opts = self.edge_options(ctx, ctx["cx"], max(ctx["v0"], 1.0))
            if opts:
                return ("jump", opts[0][2], "gap-forced"), -3.0
        # otherwise maximise time to hit among valid plans (prefer run)
        best = None
        for c in cands:
            r = results.get(c)
            if r is None:
                continue
            ht = r["hit"] if r["hit"] is not None else 999
            if best is None or ht > best[0]:
                best = (ht, c, r)
        if best is None:
            E = ctx["E"]
            if E is not None and (E - ctx["cx"]) <= TRIG:
                # no valid landing known: jump anyway with the least bad hold
                v = max(ctx["v0"], 1.0)
                bh, bm = 28, -2
                for h in GAP_H:
                    mg, dl, c, kl = landing_info(ctx["terr"], self.wallm, ctx["cx"], ctx["gy"], h, v)
                    if mg > bm:
                        bh, bm = h, mg
                return ("jump", bh, "gap-fallback"), -2.0
            return ("run", 0, "nobest"), 0.0
        return self.as_action(best[1], best[2]), -1.0

    def as_action(self, c, r):
        if c[0] == "run":
            if r.get("jump") and r.get("t0") == 0:
                return ("jump", r["H"], "gap")
            return ("run", 0, "")
        if c[0] == "stop":
            return ("stop", 0, "stop%d" % c[1])
        return ("jump", c[1], "hop")

    def air_plan(self, ctx, cxm, bottom, v, terr):
        jt = self.jt
        cands = [self.H]
        if jt < self.H and not self.fixed:
            for h in (max(jt, 1), 28):
                if h not in cands:
                    cands.append(h)
        gy = self.gy
        res = []
        pr = ctx["pr"]
        for h in cands:
            cx0 = cxm - v * jt
            mg, dl, c, kl = landing_info(terr, self.wallm, cx0, gy, h, v)
            valid = mg >= 5 and c is not None and (c < 244 or h == self.H)
            X = [float(cxm)]
            Y = [float(bottom)]
            if kl is None:
                kl2 = 40
                xs = [cxm + v * j for j in range(1, kl2)]
                ys = [gy + jy(h, jt + j) for j in range(1, kl2)]
                hit = hit_time(X + xs, Y + ys, pr, 1.0)
            else:
                n = max(1, kl - jt)
                xs = [cxm + v * j for j in range(1, n + 1)]
                ys = [gy + (jy(h, jt + j) if jt + j < kl else dl) for j in range(1, n + 1)]
                hit = jump_hit(pr, 1.0, X, Y, xs, ys, v, gy + dl, room_ahead(terr, gy, dl, c, v))
            res.append((h, valid, hit))
        return res

    # ---------- main per-frame ----------
    def act(self, s):
        self.nfr += 1
        keys = {"Right", "x"}
        why = ""
        if self.last_screen is not None and np.array_equal(s, self.last_screen):
            self.same += 1
            if self.same == 4:
                self.log("frozen; last events:")
                for e in self.events[-10:]:
                    print("   ", e)
            if self.same > 90:
                self.finished = True
        else:
            self.same = 0
        self.last_screen = s
        m = self.tr.find(s)
        terr = terrmask(s)
        self.wallm = cmask(s, (228, 92, 16)) | cmask(s, LIGHT)
        band = terr[150:240]
        if self.prev_terr is not None:
            sh = shift_est(self.prev_terr, band, self.last_shift)
        else:
            sh = 3
        self.prev_terr = band
        self.last_shift = sh
        self.S += sh
        if m is None:
            self.nomario = getattr(self, "nomario", 0) + 1
            if self.nomario in (1, 20):
                self.log("no mario; last events:", self.events[-3:])
            return keys
        xl, xr, ymin, ymax = m
        bottom = ymax
        cxm = (xl + xr) // 2
        top = bottom - 15
        self.hist_cx.append(cxm)
        self.hist_b.append(bottom)
        dcx = (cxm - self.prev_cx) if self.prev_cx is not None else 0
        self.prev_cx = cxm
        v_now = float(min(4.0, max(0.0, sh + dcx)))
        self.vmeas = 0.7 * self.vmeas + 0.3 * v_now
        r0 = min(238, bottom + 2)
        colsolid = terr[r0:min(240, r0 + 3)].any(0)
        c0 = max(0, cxm - 3)
        sup = bool(colsolid[c0:cxm + 4].sum() >= 4)
        sr = s[min(238, bottom):min(240, bottom + 3), c0:cxm + 4]
        surf = (cmask(sr, (0, 0, 0)) | cmask(sr, LIGHT) | cmask(sr, (184, 248, 24))).sum(1).max() if sr.shape[0] else 0
        if surf < 5:
            sup = False
        if len(self.hist_b) >= 2 and self.hist_b[-1] != self.hist_b[-2]:
            sup = False
        if len(self.hist_b) < 3:
            sup = True
        elif (not sup and self.jt is None and self.hist_b[-1] == self.hist_b[-2] == self.hist_b[-3]
              and abs(bottom - self.gy) <= 1 and colsolid[max(0, cxm - 8):cxm + 9].sum() >= 1):
            sup = True   # hanging over a ledge but still standing
        vm = self.vmeas if self.nfr > 3 else V_RUN
        if vm > 2.55:
            vm = V_RUN   # the measurement is noisy; full-speed running is always 3.04 px/frame
        vy = (self.hist_b[-1] - self.hist_b[-3]) / 2.0 if len(self.hist_b) >= 3 else 0.0
        self.ch.update(find_cheeps(s, (xl, xr, top, bottom)), self.S)
        if self.jt is not None:
            self.jt += 1
            if sup and self.jt >= 8 and vy >= 0:
                self.log("LAND jt", self.jt, "cx", cxm, "b", bottom)
                self.jt = None
                self.H = 0
        if sup and self.jt is None:
            self.gy = float(bottom)
        gy = self.gy
        g0 = min(236, int(gy) + 2)
        gcol = terr[g0:g0 + 3].any(0)
        pr = self.ch.preds()
        press = False
        stop = False
        dbg = ""
        if self.jt is None and sup:
            E, F = scan_gap(gcol, max(0, cxm - 2))
            lt = cmask(s[max(48, top + 2):max(49, bottom - 1), min(255, xr + 2):min(256, xr + 22)], LIGHT)
            blk = lt.sum() >= 4
            near_gap = E is not None and (E - cxm) <= TRIG
            if blk and not near_gap:
                self.jt, self.H, self.fixed = 0, 28, True
                press = True
                why = "block"
                self.log("JUMP block cx", cxm, "b", bottom)
            else:
                ctx = dict(cx=cxm, gy=gy, v0=vm, E=E, pr=pr, terr=terr)
                (act, h, w), mm = self.decide_ground(ctx)
                dbg = "m%.1f" % mm
                if act == "jump":
                    self.jt, self.H, self.fixed = 0, h, False
                    press = True
                    why = w
                    self.nstop = 0
                    self.log("JUMP", w, "H", h, "cx", cxm, "b", bottom, "E", E, "v", round(vm, 2), "m", mm,
                             "ch", [(round(p[0]), round(p[1], 1), round(p[2]), ) for p in pr][:4])
                elif act == "stop":
                    stop = True
                    why = w
                    self.nstop += 1
                    if self.nstop in (1, 30, 100):
                        self.log("STOP n", self.nstop, w, "cx", cxm, "E", E, "m", mm, "v", round(vm, 2),
                                 "ch", [(round(p[0]), round(p[1], 1), round(p[2])) for p in pr][:4])
                else:
                    self.nstop = 0
        elif self.jt is not None:
            v = max(vm, 1.0)
            if self.jt < self.H and not self.fixed:
                ctx = dict(pr=pr)
                res = self.air_plan(ctx, cxm, bottom, v, terr)
                cur = res[0]
                if not (cur[1] and cur[2] is None):
                    alt = [r for r in res[1:] if r[1] and r[2] is None]
                    if alt:
                        nh = alt[0][0]
                        self.log("AIR change H", self.H, "->", nh, "jt", self.jt)
                        self.H = nh
                    elif not cur[1]:
                        alt = [r for r in res[1:] if r[1]]
                        if alt:
                            self.H = alt[0][0]
            press = self.jt < self.H
        if press and self.jt == 0 and self.last_z:
            self.jt = None
            self.H = 0
            press = False
            why += "-wait"
        # stuck against something while running on the ground
        if stop:
            self.since_stop = 0
        else:
            self.since_stop += 1
        if self.jt is None and sup and not press and not stop:
            if sh == 0 and dcx == 0 and self.since_stop >= 24:
                self.stuckn += 1
            else:
                self.stuckn = 0
            if self.stuckn >= 14:
                self.jt, self.H, self.fixed = 0, 28, True
                press = True
                why = "stuck"
                self.stuckn = 0
                self.log("STUCK-JUMP cx", cxm, "b", bottom)
                if self.last_z:
                    self.jt = None
                    press = False
        else:
            self.stuckn = 0
        keys = {"x"} if stop else {"Right", "x"}
        if press:
            keys.add("z")
        self.last_z = "z" in keys
        self.events.append((self.nfr, cxm, bottom, int(sup), self.jt, self.H, why, round(self.vmeas, 1), sh, dbg,
                            [(round(p[0]), round(p[1], 1), round(p[2])) for p in pr][:4]))
        if len(self.events) > 60:
            self.events = self.events[-40:]
        if self.nfr % 100 == 0:
            self.log("state cx", cxm, "b", bottom, "sup", sup, "v", round(self.vmeas, 2), "ch", len(self.ch.tracks))
        return keys


def play(sim):
    bot = Bot()
    nerr = 0
    while True:
        keys = {"Right", "x"}
        try:
            s = sim.screen()
            keys = bot.act(s)
            if bot.finished:
                return
        except Exception as e:
            nerr += 1
            if nerr < 6:
                import traceback
                print("frame error:", type(e).__name__, e, traceback.format_exc()[-500:])
        try:
            sim.step(keys)
        except Exception as e:
            print("play end:", type(e).__name__, e)
            return


def solve(sim):
    return []

