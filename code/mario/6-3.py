# ==================================================================================================
# Super Mario Bros. (NES) -- level 6-3  |  超级马里奥兄弟（NES） 关 6-3
# Pass rate on 100 random variants never used before: 93/100; fastest finish 1599 frames (26.65 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 93/100；最快一局 1599 帧（60 帧/秒，约 26.65 秒）。
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
#   closed-loop vision bot (grey ground, red lifts) + cap/lift-top landing planner (jump table) wi
#   th margin checks, no blind leaps, in-air steering
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def masks                          line 71    
#   def _runs                          line 85    
#   def cap_segs                       line 98    mushroom caps: rows with a run>=10 of dark-grey pixels sitting on white. returns [(x0,x1,top)]
#   def find_spring                    line 120   
#   def land_dx                        line 145   
#   def plan_k                         line 154   choose hold length for the next jump; returns (k, tgt, margin)
#   def lift_pred                      line 203   
#   def model_at                       line 217   
#   def lift_model                     line 222   
#   def find_lifts                     line 259   
#   def pred_steer                     line 279   predictive in-air steering. returns None (no target) or a set of keys
#   class Bot                          line 321   
#   def solve                          line 1040  
#   def play                           line 1044  
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: closed-loop vision bot (grey ground, red lifts) + cap/lift-top landing planner (jump table) with margin checks, no blind leaps, in-air steering
import math
import numpy as np

LOOK = 8
PRED = 1
SPR_MIN = 14
SPR_BACK = 32
CLOUD_G_X = 2150
CLOUD_G_N = 6
FALL_X = 2195
FALL_SF = 0.6
FL_MARG = 6
FL_VM = 0.6
HB_D = 70
HB_N = 60
HB_X0 = 1930
HB_X1 = 2075
HB_MAX = 40
DROP_X = 1900
DROP_H = 40
DROP_NEAR = 50
GRAV = 0.45
PRED_X = 1900
PMARG = 6
ENEMY_DIST = 34
FARGAP = 90
SPR_D = 22
MARGIN_MIN = 6
SFR = 0.75
LEAP_K = 30
STEER = 1
COIN_FIX = 1
RIDE_H = 12
RUN_SF = 1.0
LIFT_SF2 = 0.7
LIFT_AIM = 8
LIFT_K = 2
WALL_K = 1
TALL_ON = 1
CLOUD_FIX = 1
CLOUD_X = 1900
BULLET_ON = 1
BULLET_X = 1650
BULLET_D = 30
WALL_PERSIST = 1
PERSIST_X = 1650
TALL_V = 1.3
TALL_X = 1450
WALL_MARGIN = 8
RIGHT_OFF = 8
VBOARD_X = 1040
VBOARD_X2 = 1140
LIFT_T_X = 1100
VE = 16
LIFT_SF = 0.85
AIM = None
STEER_X = 1000
TUNABLE = {"MARGIN_MIN": (0, 10), "SFR": (0.5, 1.1), "FARGAP": (60, 120), "SPR_D": (12, 30)}
_VERBOSE = [True]
DEFAULT_K = 30

# jump table measured at full run speed on flat ground: hold k -> (apex px, flight frames, distance px)
_KS = [2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24, 30]
_A = [34, 38, 42, 49, 56, 62, 67, 71, 75, 81, 85, 86]
_T = [22, 24, 26, 30, 32, 35, 37, 40, 42, 48, 51, 53]
_DX = [60, 68, 77, 89, 95, 104, 110, 119, 125, 128, 128, 123]
_EXT = [False]


def masks(img):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    g188 = (r == 188) & (g == 188) & (b == 188)
    g124 = (r == 124) & (g == 124) & (b == 124)
    grey = g188 | g124
    brown = (r == 172) & (g == 124) & (b == 0)
    skin = (r == 252) & (g == 160) & (b == 68)
    red = (r == 248) & (g == 56) & (b == 0)
    white = (r == 252) & (g == 252) & (b == 252)
    return grey, brown, skin, red, white, g124, g188


def _runs(xs):
    out = []
    start = xs[0]
    prev = xs[0]
    for x in xs[1:]:
        if x - prev > 2:
            out.append((int(start), int(prev)))
            start = x
        prev = x
    out.append((int(start), int(prev)))
    return out


def cap_segs(g124, white):
    """mushroom caps: rows with a run>=10 of dark-grey pixels sitting on white. returns [(x0,x1,top)]"""
    cm = np.zeros_like(g124)
    cm[:-2] = g124[:-2] & (white[1:-1] | white[2:])
    cm[:40] = False
    rows = np.nonzero(cm.sum(axis=1) >= 10)[0]
    segs = []
    for yy in rows:
        y = int(yy)
        xs = np.nonzero(cm[y])[0]
        for (a, b) in _runs(xs):
            if b - a + 1 < 10:
                continue
            for S in segs:
                if abs(S[0] - a) < 5 and abs(S[1] - b) < 5 and y - S[3] <= 3:
                    S[3] = y
                    break
            else:
                segs.append([a, b, y, y])
    return [(s[0], s[1], s[2]) for s in segs]


def find_spring(red, grey188):
    cand = []
    for y in range(40, 236):
        row = red[y]
        if row.sum() < 10:
            continue
        xs = np.nonzero(row)[0]
        start = xs[0]
        prev = xs[0]
        for x in list(xs[1:]) + [10000]:
            if x - prev > 2:
                w = prev - start + 1
                if 11 <= w <= 20:
                    cand.append((y, int(start), int(prev)))
                start = x
            prev = x
    for (y, a, b) in cand:
        for (y2, a2, b2) in cand:
            if 8 <= y2 - y <= 32 and abs(a - a2) <= 3 and abs(b - b2) <= 3:
                base = grey188[y2 + 1:y2 + 9, a:b + 1]
                if base.sum() >= 15:
                    return (a, b, y, y2 + 8)
    return None


def land_dx(i, h_up):
    A, T, DX = _A[i], _T[i], _DX[i]
    if h_up > A - 3:
        return None
    r = 1.0 - h_up / A if h_up > 0 else 1.0 + (-h_up) / A
    t = T / 2.0 * (1 + math.sqrt(max(r, 0.0)))
    return DX * t / T


def plan_k(cx, feet, segs, cur_x1=None, spr=None, sf=1.0, strict=False, aim=None):
    """choose hold length for the next jump; returns (k, tgt, margin)"""
    tgt = None
    base = feet + 1
    for (x0, x1, top) in sorted(segs):
        if x0 <= cx + 4:
            continue
        if cur_x1 is not None and x0 < cur_x1 - 2:
            continue
        if strict and x1 >= 252:
            continue
        h_up = base - top
        if h_up > 80 or h_up < -140:
            continue
        tgt = (x0, x1, top)
        break
    if tgt is None:
        return DEFAULT_K, None, -999
    x0, x1, top = tgt
    h_up = base - top
    best = None
    for i, k in enumerate(_KS):
        if k < 4 and not _EXT[0]:
            continue
        d = land_dx(i, h_up)
        if d is None:
            continue
        land = cx + d * sf
        if h_up > 0:
            dw = (x0 - 7) - cx
            if dw > 0:
                tw = dw * _T[i] / _DX[i] / sf
                half = _T[i] / 2.0
                y = _A[i] * (1 - ((tw - half) / half) ** 2) if tw < _T[i] else -1
                if y < h_up + 2:
                    continue
        margin = min(land - (x0 + 4), (x1 - 5) - land)
        score = margin
        if aim is not None and margin >= 0:
            score = 100 - abs(land - (x0 + aim))
        if spr is not None and x0 <= spr[0] <= x1:
            margin = score = -abs(land - (x0 + 6)) if land >= x0 + 2 else -200
        if best is None or score > best[0]:
            best = (score, k, margin)
    if best is None:
        return 30, tgt, -999
    return best[1], tgt, best[2]


def lift_pred(hist, t):
    if len(hist) < 30:
        return None
    xs = [h[1] for h in hist]
    xmax = max(xs)
    xmin = min(xs)
    if xmax - xmin < 10:
        return float(xs[-1])
    m = lift_model(hist)
    if m is None:
        return None
    return model_at(m, t)


def model_at(m, t):
    mid, amp, tref, H, sg = m
    return mid + sg * amp * math.cos(math.pi * (t - tref) / H)


def lift_model(hist):
    if len(hist) < 30:
        return None
    xs = [h[1] for h in hist]
    xmax = max(xs)
    xmin = min(xs)
    if xmax - xmin < 10:
        return (float(xs[-1]), 0.0, hist[-1][0], 120.0, 1)
    t0 = hist[0][0]
    t1 = hist[-1][0]

    def ext_time(cond):
        ts = [h[0] for h in hist if cond(h[1])]
        last = ts[-1]
        cl = [q for q in ts if last - q <= 20]
        return sum(cl) / float(len(cl))

    tmax = ext_time(lambda x: x >= xmax - 1)
    tmin = ext_time(lambda x: x <= xmin + 1)
    vmax = (t0 + 4 < tmax < t1 - 4)
    vmin = (t0 + 4 < tmin < t1 - 4)
    if vmax and vmin:
        H = abs(tmax - tmin)
        if H < 20:
            return None
        amp = (xmax - xmin) / 2.0
        mid = (xmax + xmin) / 2.0
        if tmax > tmin:
            return (mid, amp, tmax, H, 1)
        return (mid, amp, tmin, H, -1)
    if vmin:
        return (xmin + 26.5, 26.5, tmin, 120.0, -1)
    if vmax:
        return (xmax - 26.5, 26.5, tmax, 120.0, 1)
    return None


def find_lifts(red):
    cnt = red[40:].sum(axis=1)
    rows = np.nonzero(cnt >= 20)[0]
    out = []
    for yy in rows:
        y = int(yy) + 40
        xs = np.nonzero(red[y])[0]
        for (a, b) in _runs(xs):
            if b - a + 1 >= 20:
                merged = False
                for L in out:
                    if abs(L[1] - a) < 6 and abs(L[2] - b) < 6 and y - L[3] <= 8:
                        L[3] = y
                        merged = True
                        break
                if not merged:
                    out.append([y, int(a), int(b), y])
    return out


def pred_steer(cx, feet, vx, vy, segs, lifts, incl=False, vL=0.0):
    """predictive in-air steering. returns None (no target) or a set of keys"""
    cands = []
    for (x0, x1, top) in segs:
        cands.append((x0, x1, top, 0.0))
    if incl:
        for L in lifts:
            if L[2] - L[1] >= 20:
                cands.append((L[1], L[2], L[0], vL))
    best = None
    for (x0, x1, top, vt) in cands:
        dy = top - (feet + 1)
        if dy < -2:
            continue
        dy = max(dy, 0)
        disc = vy * vy + 2 * GRAV * dy
        t = (-vy + math.sqrt(max(disc, 0.0))) / GRAV
        t = max(t, 0.0)
        xp = cx + vx * t
        x0 = x0 + vt * t
        x1 = x1 + vt * t
        if xp < x0:
            dist = x0 - xp
        elif xp > x1:
            dist = xp - x1
        else:
            dist = 0.0
        key = (dist, top)
        if best is None or key < best[0]:
            best = (key, x0, x1, top, xp, t)
    if best is None:
        return None
    key, x0, x1, top, xp, t = best
    if key[0] > 60:
        return None
    if xp > x1 - PMARG:
        return {"Left"}
    if xp < x0 + PMARG:
        return {"Right", "x"}
    return set()


class Bot:
    def __init__(self):
        self.mx = None
        self.fy = None
        self.zheld = False
        self.ground_fy = None
        self.t = 0
        self.lost = 0
        self.log = []
        self.lifts = []
        self.top = None
        self.wait = 0
        self.on_lift = False
        self.zleft = 0
        self.pfy = None
        self.jt = 99
        self.lastz = False
        self.kplan = DEFAULT_K
        self.info = ""
        self.run_k = 0
        self.lh = {}
        self.back = 0
        self._pd_used = False
        self.align_on = False
        self.cum = 0
        self.ride_t = 0
        self.fyh = []
        self.pmask = None
        self.stage = 0
        self.ehist = []
        self._keys_override = None
        self.steer = ""
        self.wxh = []
        self.vflag = False
        self.runph = 0
        self.retry = 0
        self.lflag = False
        self.tflag = False
        self.wallprev = False
        self.dflag = False
        self.dpit_now = None
        self.sp_mode = 0
        self.sp_cnt = 0
        self.nflag = False
        self.hold_b = 0
        self.vL = 0.0
        self.prevL = []

    def par(self):
        if self.stage == 0:
            return (248, 14, 55)
        return (450, 30, 46)

    def lift_plan(self, e, cx, fy, cands):
        best = None
        n = int(e / 2.0) + 4
        sf = 0.55 + 0.45 * min(1.0, e / 45.0)
        cxt = cx + e + 4
        for L in cands:
            h = self.lh.get(L[0])
            if not h:
                continue
            h_up = fy + 1 - L[0]
            width = 31
            sf2 = sf
            if self.stage >= 1:
                if h_up > 12:
                    continue
                sf2 = sf * LIFT_SF
            for i, k in enumerate(_KS):
                if k < 4:
                    continue
                d = land_dx(i, h_up)
                if d is None:
                    continue
                r = 1.0 - h_up / _A[i] if h_up > 0 else 1.0 + (-h_up) / _A[i]
                tfl = _T[i] / 2.0 * (1 + math.sqrt(max(r, 0.0)))
                xp = lift_pred(h, self.t + n + tfl)
                if xp is None:
                    continue
                land = cxt + d * sf2
                margin = min(land - (xp + 3), (xp + width - 3) - land)
                if margin >= 2 and (best is None or margin > best[0]):
                    best = (margin, k)
        return None if best is None else best[1]

    def find_mario(self, img, brown, skin):
        lifts = self.lifts
        ms = brown | skin
        ms[:40] = False
        for L in lifts:
            ms[L[0]:L[3] + 3, max(0, L[1] - 1):L[2] + 2] = False
        if self.mx is not None and self.lost < 20:
            x0 = max(0, int(self.mx) - 30)
            x1 = min(256, int(self.mx) + 50)
            y0 = max(40, int(self.fy) - 60)
            y1 = min(240, int(self.fy) + 40)
        else:
            x0, x1, y0, y1 = 0, 256, 40, 240
        sub = ms[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        if len(ys) < 5:
            self.lost += 1
            return False
        mxm = np.median(xs)
        mym = np.median(ys)
        bm = brown[y0:y1, x0:x1] & sub
        bys, bxs = np.nonzero(bm)
        if len(bys) >= 3 and COIN_FIX:
            # anchor on Mario's brown pixels (coins are skin-coloured only) and drop far-away skin blobs
            bx = np.median(bxs)
            by = np.median(bys)
            k = (xs - bx >= -9) & (xs - bx <= 11) & (ys - by >= -11) & (ys - by <= 11)
        else:
            k = (abs(xs - mxm) <= 12) & (abs(ys - mym) <= 14)
        if k.sum() < 4:
            self.lost += 1
            return False
        xs = xs[k] + x0
        ys = ys[k] + y0
        self.mx = xs.min()
        self.fy = ys.max()
        self.top = ys.min()
        self.lost = 0
        return True

    def surface(self, grey, c, ytop):
        c = min(255, max(0, c))
        idx = np.nonzero(grey[ytop:, c])[0]
        if len(idx) == 0:
            return None
        return ytop + int(idx[0])

    def targets(self, g124, white, lifts, cur=None):
        """landing surfaces: caps and (non-current) lifts, as (x0,x1,top)"""
        segs = list(cap_segs(g124, white))
        for L in lifts:
            if cur is not None and L is cur:
                continue
            if L[2] - L[1] < 20:
                continue
            segs.append((L[1], L[2], L[0]))
        return segs

    def act(self, img):
        self.t += 1
        grey, brown, skin, red, white, g124, g188 = masks(img)
        m = (grey | white)[40::2]
        if self.pmask is not None:
            bs = None
            for sh in range(0, 7):
                c = int(np.count_nonzero(m[:, :256 - sh] != self.pmask[:, sh:]))
                if bs is None or c < bs[0] - 2:
                    bs = (c, sh)
            self.cum += bs[1]
        self.pmask = m
        self.lifts = find_lifts(red)
        self.dpit_now = None
        cur_l = [(self.cum + L[1], L[0], self.cum + L[2]) for L in self.lifts]
        dxs = []
        for a in cur_l:
            for b in self.prevL:
                if abs(a[1] - b[1]) <= 1 and abs(a[0] - b[0]) <= 8 and abs((a[2] - a[0]) - (b[2] - b[0])) <= 2:
                    dxs.append(a[0] - b[0])
                    break
        if dxs:
            dxs.sort()
            self.vL = 0.8 * self.vL + 0.2 * dxs[len(dxs) // 2]
        self.prevL = cur_l
        ok = self.find_mario(img, brown, skin)
        keys = {"Right", "x"}
        if not ok:
            self.zheld = not self.zheld
            if self.zheld:
                keys.add("z")
            return keys
        fy = int(self.fy)
        _EXT[0] = (self.cum + int(self.mx) >= TALL_X)
        self.wxh.append(self.cum + int(self.mx))
        if len(self.wxh) > 4:
            self.wxh.pop(0)
        self.fyh.append(fy)
        if len(self.fyh) > 6:
            self.fyh.pop(0)
        cx = int(self.mx) + 7
        front = int(self.mx) + 14
        lifts = self.lifts
        under = grey[fy:fy + 4, max(0, cx - 5):cx + 6]
        on_ground = bool(under.any())
        if on_ground and self.cum + int(self.mx) >= CLOUD_G_X:
            on_ground = bool((under.sum(axis=1) >= CLOUD_G_N).any())
        cur = None
        for L in lifts:
            if L[1] - 6 <= cx <= L[2] + 6 and fy - 3 <= L[0] <= fy + 5:
                cur = L
        on_lift = cur is not None
        self.on_lift = on_lift
        if not on_lift:
            self.ride_t = 0
            self.runph = 0
        if on_ground:
            self.ground_fy = fy
        grounded = bool(on_ground or on_lift)
        gy = self.ground_fy if self.ground_fy is not None else fy
        if on_lift:
            gy = fy
        ytop = fy - 14
        jump = False
        stop = False
        why = ""
        spring_ahead = False

        def surf(c):
            return self.surface(grey, c, ytop)

        lifts_ahead = [L for L in lifts if L[2] > front - 6 and L[1] < front + 130 and L[0] > fy - 90]
        self.kplan = DEFAULT_K

        def plan():
            segs = list(cap_segs(g124, white))
            if self.cum + int(self.mx) >= FALL_X:
                segs += [(L[1], L[2], L[0]) for L in lifts if L[2] - L[1] >= 20]
            spr_now = find_spring(red, g188)
            k, tgt, mg = plan_k(cx, fy, segs, None, spr_now, 1.0)
            self.kplan = k
            self.info = "k=%d tgt=%s mg=%d" % (k, tgt, mg)

        if grounded:
            if on_lift:
                fl_done = False
                if cur is not None and self.cum + int(self.mx) >= FALL_X:
                    vL = self.vL
                    own = ((self.wxh[-1] - self.wxh[-3]) / 2.0 - vL) if len(self.wxh) >= 3 else 0.0
                    tg = [(L[1], L[2], L[0], vL) for L in lifts if L is not cur and L[2] - L[1] >= 20 and L[2] > cx and L[1] < 254 and abs(L[0] - fy) < 90]
                    tg += [(q[0], q[1], q[2], 0.0) for q in cap_segs(g124, white) if q[1] > cx + 10 and fy - 95 <= q[2] <= fy + 70]
                    if tg:
                        fl_done = True
                        best = None
                        vm = max(0.6, min(2.7, max(own, 0.0) + FL_VM))
                        for (x0, x1, top, vt) in tg:
                            h_up = fy + 1 - top
                            for i, kk in enumerate(_KS):
                                if kk < 2:
                                    continue
                                d = land_dx(i, h_up)
                                if d is None:
                                    continue
                                r = 1.0 - h_up / _A[i] if h_up > 0 else 1.0 + (-h_up) / _A[i]
                                tfl = _T[i] / 2.0 * (1 + math.sqrt(max(r, 0.0)))
                                land = cx + vm * tfl
                                mgn = min(land - (x0 + vt * tfl + 4), (x1 + vt * tfl - 4) - land)
                                if best is None or mgn > best[0]:
                                    best = (mgn, kk, (x0, x1, top, vt))
                        if best is not None and best[0] >= FL_MARG:
                            jump = True
                            self.kplan = best[1]
                            self.info = "falllift k=%d tgt=%s mg=%d own=%.1f vL=%.1f" % (best[1], best[2], best[0], own, vL)
                            why = "falllift jump"
                        else:
                            stop = True
                            why = "falllift wait"
                target = None
                for d in range(4, 70, 3):
                    s = surf(front + d)
                    if s is not None and s >= fy - 52 and s <= fy + 40:
                        c = min(255, front + d)
                        row = grey[s, max(0, c - 16):c + 17]
                        if row.sum() >= 18:
                            target = (d, s)
                            break
                if fl_done:
                    pass
                elif target is not None and target[0] <= 30:
                    jump = True
                    why = "lift->ground d=%d s=%d fy=%d" % (target[0], target[1], fy)
                    plan()
                elif target is not None and target[0] <= 44 and cur is not None and cx < cur[2] - 8:
                    self._keys_override = {"Right"}
                    why = "walk on lift"
                else:
                    stop = True
                    why = "riding"
                    self.ride_t += 1
                    if cur is not None:
                        cap_t = sorted(s for s in cap_segs(g124, white) if s[0] > cx + 6 and fy - 95 <= s[2] <= fy + 60 and s[1] < 252)
                        ctgt = cap_t[0] if cap_t else None
                        hup = (fy + 1 - ctgt[2]) if ctgt is not None else 999
                        hlim = RIDE_H
                        sfj = RUN_SF
                        if (ctgt is None or hup > RIDE_H) and self.cum + int(self.mx) >= LIFT_T_X:
                            lift_t = sorted((L[1], L[2], L[0]) for L in lifts
                                            if L is not cur and L[2] - L[1] >= 20 and L[1] > cx + 4 and L[2] < 252 and abs(L[0] - fy) < 70)
                            if lift_t and fy + 1 - lift_t[0][2] <= 60:
                                cap_t = lift_t
                                ctgt = lift_t[0]
                                hup = fy + 1 - ctgt[2]
                                hlim = 60
                                sfj = LIFT_SF2
                        if self.runph == 0 and hup <= hlim and self.ride_t > 6:
                            self.runph = 1
                        if self.runph and hup > hlim + 25:
                            self.runph = 0
                        mid = (cur[1] + cur[2]) / 2.0
                        vw = (self.wxh[-1] - self.wxh[-3]) / 2.0 if len(self.wxh) >= 3 else 0.0
                        if self.runph == 1:
                            if cx <= cur[1] + 9 and vw < 0.8:
                                self.runph = 2
                            else:
                                self._keys_override = {"Left"}
                                why = "ride: to left end"
                        if self.runph == 2:
                            if cx >= cur[2] - (RIGHT_OFF if hlim == 60 else 8):
                                k, tgt, mg = plan_k(cx, fy, cap_t, None, None, sfj, False, LIFT_AIM if hlim == 60 else None)
                                if hlim == 60 and mg < -8 and self.retry < 4:
                                    self.retry += 1
                                    self.runph = 1
                                    self._keys_override = {"Left"}
                                    why = "ride: retry (mg=%d)" % mg
                                else:
                                    jump = True
                                    stop = False
                                    if hlim == 60 and LIFT_K > 0:
                                        k = LIFT_K
                                    self.kplan = k
                                    self.info = "ride k=%d tgt=%s mg=%d" % (k, tgt, mg)
                                    why = "ride: runup leap"
                                    if hlim == 60:
                                        self.lflag = True
                                    self.runph = 0
                                    self.retry = 0
                            else:
                                self._keys_override = {"Right", "x"}
                                why = "ride: runup"
                        elif self.runph == 0:
                            if cx > mid + 4 or vw > 1.8:
                                self._keys_override = {"Left"}
                                why = "riding: centre (left)"
                            elif cx < mid - 8:
                                self._keys_override = {"Right"}
                                why = "riding: centre (right)"
            else:
                spr = find_spring(red, g188)
                if spr is not None and spr[1] > cx - 4 and abs(spr[3] - fy) < 40 and spr[0] - front < 60:
                    spring_ahead = True
                dpit = None
                for d in range(0, 100):
                    s = surf(front + d)
                    if s is None or s > gy + 20:
                        dpit = d
                        break
                self.dpit_now = dpit
                wall = None
                for d in (3, LOOK):
                    s = surf(front + d)
                    if s is not None and s < gy - 2:
                        if CLOUD_FIX and self.cum + int(self.mx) >= CLOUD_X:
                            cc = min(255, front + d)
                            lo = cc
                            while lo > cc - 14 and lo > 0 and grey[s, lo - 1]:
                                lo -= 1
                            hi = cc
                            while hi < cc + 14 and hi < 255 and grey[s, hi + 1]:
                                hi += 1
                            if hi - lo + 1 < 10:
                                continue
                        wall = d
                        break
                if WALL_PERSIST and self.cum + int(self.mx) >= PERSIST_X:
                    raw = wall
                    if raw is not None and not self.wallprev:
                        wall = None
                    self.wallprev = raw is not None
                far = False
                if dpit is not None:
                    resume = None
                    for d in range(dpit + 2, 150, 4):
                        s = surf(front + d)
                        if s is not None and s <= gy + 60:
                            resume = d
                            break
                    far = resume is None or (resume - dpit) > FARGAP
                spb = False
                if spring_ahead:
                    dsp = spr[0] - front
                    if self.sp_mode == 0 and dsp < SPR_MIN:
                        self.sp_mode = 1
                    if self.sp_mode == 1:
                        room = bool(grey[fy:fy + 4, max(0, int(self.mx) - 8)].any())
                        if dsp >= SPR_BACK or self.sp_cnt >= 50 or not room:
                            self.sp_mode = 2
                        else:
                            self.sp_cnt += 1
                            spb = True
                else:
                    self.sp_mode = 0
                    self.sp_cnt = 0
                if spring_ahead and not spb and spr[0] - front <= SPR_D:
                    jump = True
                    segs = [s for s in cap_segs(g124, white) if s[0] > spr[1] + 4]
                    k, tgt, mg = plan_k(cx, fy, segs, None, None)
                    self.kplan = k
                    self.info = "spring k=%d tgt=%s" % (k, tgt)
                    why = "spring hop d=%d" % (spr[0] - front)
                    self.nflag = True
                elif spring_ahead and wall is not None and spr[0] - front < 24:
                    wall = None
                    why = "spring-run"
                tall = None
                if TALL_ON and self.cum + int(self.mx) >= TALL_X and dpit is not None and dpit <= 45:
                    for (tx0, tx1, ttop) in sorted(cap_segs(g124, white)):
                        if tx0 <= cx + 4:
                            continue
                        if fy + 1 - ttop >= 45 and tx0 - front <= 60:
                            tall = (tx0, tx1, ttop)
                        break
                drop = None
                if TALL_ON and tall is None and spr is not None and self.cum + int(self.mx) >= TALL_X and dpit is not None and dpit <= 50:
                    for (tx0, tx1, ttop) in sorted(cap_segs(g124, white)):
                        if tx0 <= cx + 4:
                            continue
                        if fy + 1 - ttop <= -30 and tx0 - 4 <= spr[0] <= tx1:
                            drop = (tx0, tx1, ttop)
                        break
                if drop is None and tall is None and not jump and self.cum + int(self.mx) >= DROP_X and dpit is not None and dpit <= 50:
                    for (tx0, tx1, ttop) in sorted(cap_segs(g124, white)):
                        if tx0 <= cx + 4:
                            continue
                        if fy + 1 - ttop <= -DROP_H and tx0 - front <= DROP_NEAR and tx1 - tx0 >= 24:
                            drop = (tx0, tx1, ttop)
                            self.info = "drop2 %s dpit=%d" % (str(drop), dpit)
                        break
                if jump:
                    pass
                elif spb:
                    self._keys_override = {"Left"}
                    why = "spring backoff d=%d" % (spr[0] - front)
                elif drop is not None:
                    vn0 = (self.wxh[-1] - self.wxh[-3]) / 2.0 if len(self.wxh) >= 3 else 0.0
                    self.lflag = True
                    self.dflag = True
                    if vn0 > 1.3 and dpit > 2:
                        self._keys_override = {"Left"}
                        why = "drop: brake d=%d v=%.1f" % (dpit, vn0)
                    else:
                        self._keys_override = {"Right"}
                        why = "drop: walk off d=%d v=%.1f" % (dpit, vn0)
                elif tall is not None:
                    vn0 = (self.wxh[-1] - self.wxh[-3]) / 2.0 if len(self.wxh) >= 3 else 0.0
                    if dpit <= VE and not (vn0 > TALL_V and dpit > 1):
                        jump = True
                        self.kplan = 30
                        self.vflag = True
                        self.tflag = True
                        why = "tall cap: slow vertical jump"
                        self.info = "tall=%s d=%d v=%.1f" % (tall, dpit, vn0)
                    elif vn0 > 1.3:
                        self._keys_override = {"Left"}
                        why = "tall: brake d=%d v=%.1f" % (dpit, vn0)
                    else:
                        self._keys_override = {"Right"}
                        why = "tall: walk d=%d v=%.1f" % (dpit, vn0)
                elif wall is not None:
                    jump = True
                    why = "wall d=%d" % wall
                    plan()
                    sw = surf(front + wall)
                    if sw is not None and WALL_K:
                        wh = fy + 1 - sw
                        kw = 30
                        for i in range(len(_KS)):
                            if _KS[i] >= 4 and _A[i] >= wh + WALL_MARGIN:
                                kw = _KS[i]
                                break
                        if kw > self.kplan:
                            self.kplan = kw
                            self.info += " wallk=%d wh=%d" % (kw, wh)
                elif dpit is not None and lifts_ahead and far:
                    e = dpit
                    self._pd_used = True
                    if self.run_k:
                        if e > 3:
                            self._keys_override = {"Right", "x"}
                            why = "lift run e=%d" % e
                        else:
                            jump = True
                            self.kplan = self.run_k
                            self.run_k = 0
                            self.lh = {}
                            self.back = 0
                            self.stage += 1
                            self.wait = 0
                            self.fyh = []
                            why = "lift jump"
                            self.info = "lift jump k=%d" % self.kplan
                    else:
                        eh = self.ehist
                        eh.append(e)
                        if len(eh) > 3:
                            eh.pop(0)
                        wh = self.wxh
                        vn = max(0.0, min(3.0, (wh[-1] - wh[-3]) / 2.0)) if len(wh) >= 3 else 0.0
                        v = 0.0 if len(eh) < 3 else max(0.0, min(3.0, (eh[0] - e) / 2.0))
                        if self.align_on and v < 2.0:
                            v = 0.0
                        stopd = v * v / 0.4 + v * 2.0
                        pw, pk, pr = self.par()
                        vc = None
                        if self.stage >= 1 and VBOARD_X <= self.cum + int(self.mx) <= VBOARD_X2:
                            for L in lifts:
                                hu = fy + 1 - L[0]
                                if 45 <= hu <= 92 and L[2] - L[1] >= 20 and L[2] < 254 and -30 < L[1] - front < 70:
                                    vc = L
                        if vc is not None:
                            # a high lift just beyond the edge: board it with a slow near-vertical full-height jump
                            if e <= VE and vn > 1.7 and e > 5:
                                self._keys_override = {"Left"}
                                why = "vbrake2 e=%d v=%.1f" % (e, vn)
                            elif e <= VE:
                                jump = True
                                self.kplan = 30
                                self.vflag = True
                                why = "vertical board"
                                self.info = "vboard e=%d v=%.1f lift=%s" % (e, vn, vc)
                            elif vn > 1.6:
                                self._keys_override = {"Left"}
                                why = "vbrake e=%d v=%.1f" % (e, vn)
                            else:
                                self._keys_override = {"Right"}
                                why = "vwalk e=%d v=%.1f" % (e, v)
                            self.wait = 0
                        elif e - pr > stopd + 4 and v > 0.7:
                            self._keys_override = {"Right", "x"} if v < 2.4 else {"Right"}
                            why = "approach d=%d v=%.1f" % (e, v)
                            self.wait = 0
                            self.lh = {}
                        elif v > 0.7 and e <= 8 and v >= 1.5 and self.stage >= 1:
                            # no room to brake: leap, but choose the hold from the visible targets
                            jump = True
                            segs = [s for s in self.targets(g124, white, lifts, cur) if s[0] > cx + 6]
                            k, tgt, mg = plan_k(cx, fy, segs, None, None, 1.0, True)
                            self.kplan = k if tgt is not None else LEAP_K
                            self.info = "leap k=%d tgt=%s mg=%d" % (self.kplan, tgt, mg)
                            why = "no room to brake: leap"
                        elif v > 0.7:
                            self._keys_override = {"Left"}
                            why = "brake d=%d v=%.1f" % (e, v)
                            self.lh = {}
                        elif e > pr + 10:
                            self._keys_override = {"Right"}
                            why = "creep d=%d" % e
                        else:
                            self.wait += 1
                            why = "wait edge"
                            cands = []
                            for L in lifts:
                                if L[2] - L[1] < 20 or not (fy - 85 <= L[0] <= fy + 40):
                                    continue
                                if L[2] >= 254:
                                    continue
                                x0e = L[2] - 31
                                cands.append((L[0], x0e))
                                hh = self.lh.setdefault(L[0], [])
                                if not hh or hh[-1][0] != self.t:
                                    hh.append((self.t, x0e))
                                if len(hh) > 400:
                                    hh.pop(0)
                            k = self.lift_plan(e, cx, fy, cands) if e >= pr - 10 else None
                            left_room = 0
                            for dxl in range(1, 40):
                                cc = int(self.mx) - dxl
                                if cc < 0 or not grey[fy:fy + 4, cc].any():
                                    break
                                left_room += 1
                            if self.wait % 40 == 1 and _VERBOSE[0]:
                                print("WAIT t=%d st=%d cx=%d e=%d fy=%d cands=%s k=%s" % (self.t, self.stage, cx, e, fy, cands, k))
                            if k is not None:
                                self.run_k = k
                                why = "start run k=%d e=%d" % (k, e)
                                self._keys_override = {"Right", "x"}
                            elif e < pr - 6 and self.back < 90 and left_room > 20:
                                self.back += 1
                                self._keys_override = {"Left"}
                                why = "back up d=%d" % e
                            elif self.wait >= pw:
                                self.run_k = pk
                                self._keys_override = {"Right", "x"}
                                why = "blind run"
                            else:
                                stop = True
                                self._keys_override = set()
                elif dpit is not None and dpit <= 6:
                    jump = True
                    why = "pit d=%d" % dpit
                    plan()
        if not jump and grounded and not stop and not spring_ahead:
            ens = (brown | skin)
            ens[:40] = False
            for L in lifts:
                ens[L[0]:L[3] + 3, max(0, L[1] - 1):L[2] + 2] = False
            sub = ens[fy - 16:fy + 1, front + 2:front + ENEMY_DIST]
            if sub.any():
                jump = True
                why = "enemy"
                self.kplan = 12
            elif BULLET_ON and self.cum + int(self.mx) >= BULLET_X:
                bsub = g188[max(0, fy - 15):fy, front + 2:front + BULLET_D]
                if int(bsub.sum()) >= 10 and find_spring(red, g188) is None:
                    jump = True
                    why = "bullet"
                    self.kplan = 8

        if grounded and BULLET_ON and HB_X0 <= self.cum + int(self.mx) <= HB_X1 and not on_lift:
            hb = g124[max(0, fy - 66):max(0, fy - 18), front + 2:front + HB_D]
            bl = int(hb.sum()) >= HB_N
            dp = self.dpit_now
            vbn = (self.wxh[-1] - self.wxh[-3]) / 2.0 if len(self.wxh) >= 3 else 0.0
            need = 8 + (vbn * vbn / 0.4 if vbn > 1.3 else 0.0)
            if bl and self.hold_b < HB_MAX and dp is not None and dp >= need:
                self.hold_b += 1
                jump = False
                stop = True
                self._keys_override = {"Left"} if (vbn > 1.3 and dp < 45) else None
                why = "bullet wait"
            elif not bl:
                self.hold_b = 0
        ov = self._keys_override
        self._keys_override = None
        if ov is not None and not jump:
            keys = set(ov)
        elif stop and not jump:
            keys = set()
        self.jt += 1
        prev_fy = self.pfy
        if self.zleft > 0 and grounded and self.jt > 6:
            self.zleft = 0
        if self.zleft > 0 and self.jt > 6 and prev_fy is not None and fy > prev_fy:
            self.zleft = 0
        self.pfy = fy
        if self.zleft > 0:
            keys = {"Right", "z", "x"}
            if self.vflag:
                keys.discard("x")
            if self.tflag and self.jt < 18:
                keys.discard("Right")
            self.zleft -= 1
        elif grounded:
            if jump and not self.lastz:
                keys.add("Right")
                keys.add("x")
                keys.add("z")
                self.zleft = self.kplan
                if _VERBOSE[0]:
                    print("JUMP t=%d wx=%d fy=%d k=%d %s | %s" % (self.t, self.cum + int(self.mx), fy, self.zleft, why, self.info))
                self.jt = 0
            elif jump:
                keys.discard("z")
            if self.vflag:
                keys.discard("x")
                if self.tflag and self.jt < 18:
                    keys.discard("Right")
                if self.jt > 8 and not jump:
                    self.vflag = False
                    self.tflag = False
            if self.lflag and self.jt > 8 and not jump:
                self.lflag = False
                self.nflag = False
                self.dflag = False
        else:
            keys = {"Right", "x"}
            if self.vflag:
                keys = {"Right"}
                if self.tflag and self.jt < 18:
                    keys = set()
            if self.lflag:
                keys = set()
            self.steer = ""
            if STEER and not self.nflag and not self.vflag and (not self.lflag or self.dflag) and prev_fy is not None and fy > prev_fy and self.jt > 4 and self.cum + int(self.mx) >= STEER_X:
                segs = [s for s in cap_segs(g124, white) if s[2] >= fy - 3 and s[2] <= fy + 130]
                best = None
                for (x0, x1, top) in segs:
                    if x1 < cx - 6:
                        continue
                    dist = 0 if x0 <= cx <= x1 else (x0 - cx)
                    if best is None or dist < best[0]:
                        best = (dist, x0, x1, top)
                if LIFT_T_X <= self.cum + int(self.mx) < FALL_X and self.jt > 12:
                    for L in lifts:
                        if L[1] - 2 <= cx <= L[2] + 2 and fy <= L[0] <= fy + 45 and cx > L[1] + 14:
                            keys = {"Left"}
                            self.steer = "Ll"
                            break
                prk = None
                if PRED and self.steer != "Ll" and self.cum + int(self.mx) >= PRED_X and len(self.wxh) >= 3 and len(self.fyh) >= 2:
                    vxm = (self.wxh[-1] - self.wxh[-3]) / 2.0
                    vym = float(self.fyh[-1] - self.fyh[-2])
                    prk = pred_steer(cx, fy, vxm, vym, segs, lifts, self.cum + int(self.mx) >= FALL_X, self.vL)
                if self.steer == "Ll":
                    pass
                elif prk is not None:
                    keys = set(prk)
                    self.steer = "P" + ("L" if "Left" in prk else ("R" if prk else "N"))
                elif best is not None and best[0] <= 40:
                    _, x0, x1, top = best
                    if cx > x1 - 12:
                        keys = {"Left"}
                        self.steer = "L"
                    elif cx >= x0 + 8 and cx > (x0 + x1) / 2.0 - 4:
                        keys = set()
                        self.steer = "N"
        self.lastz = "z" in keys
        if not self._pd_used:
            self.ehist = []
        self._pd_used = False
        self.dbg = (grounded, on_ground, on_lift, jump, stop, why, fy, int(self.mx), front)
        if why and grounded and why != "riding":
            self.log.append((self.t, why, self.info if jump else ""))
        return keys


def solve(sim):
    return []


def play(sim):
    bot = Bot()
    last = None
    same = 0
    n = 0
    while True:
        img = sim.screen()
        if last is not None and np.array_equal(img, last):
            same += 1
            if same > 90:
                break
        else:
            same = 0
        last = img
        keys = bot.act(img)
        n += 1
        if n % 40 == 0 and _VERBOSE[0]:
            print("f", n, bot.dbg, bot.info, bot.steer)
        try:
            sim.step(keys)
        except Exception:
            break
    print("end", n)

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    st = ctx.get("stuck") if isinstance(ctx, dict) else None\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": bool(st and st.get("is"))},\n            "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}'}

