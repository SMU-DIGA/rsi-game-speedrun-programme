# ==================================================================================================
# Super Mario Bros. (NES) -- level 5-4  |  超级马里奥兄弟（NES） 关 5-4
# Pass rate on 100 random variants never used before: 49/100; fastest finish 2172 frames (36.20 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 49/100；最快一局 2172 帧（60 帧/秒，约 36.20 秒）。
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
#   PB closed-loop pixel policy for castle 5-4 (c9 base, floor-firebar fix, pillar hops, bridge po
#   d avoid, Bowser dash) + guarded AXE_JUMP (hop the axe-pillar wall only when Bowser is not adja
#   cent)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def solid_orange                   line 208   
#   def gray_mask                      line 218   
#   def mario_box                      line 223   
#   def scroll_shift                   line 234   
#   def fire_mask                      line 247   
#   def blobs                          line 256   
#   def find_lifts                     line 285   
#   def podoboos                       line 322   
#   def predict_landing                line 358   
#   def flame_speed                    line 378   
#   def update_ftrack                  line 391   
#   def dx_est                         line 416   
#   def choose_hold                    line 434   
#   def bar_omega                      line 483   
#   def bar_crossing_safe              line 500   
#   def find_coins                     line 538   
#   def fireball_angle                 line 567   
#   def bar_info                       line 615   
#   def _jg_hits                       line 700   
#   def _jg_plan                       line 740   -> (wait, alt_hold): wait = a flame is predicted to hit the planned jump; alt_hold = another hold that avoids 
#   def play                           line 800   
#   def _play                          line 828   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: PB closed-loop pixel policy for castle 5-4 (c9 base, floor-firebar fix, pillar hops, bridge pod avoid, Bowser dash) + guarded AXE_JUMP (hop the axe-pillar wall only when Bowser is not adjacent)
import numpy as np

JUMP_HOLD = 26
LOOK = 6
FLAME_X = 200
HID_MIN = 60
HID_MAX = 92
BOW_GO = 95
LEDGE_ANG_MAX = 45
POD_MIN = 15
POD_MAX = 45
BAR_MIN = 16
BAR_WIN = 54
BAR_WAIT = 96
WAIT_DIST = 92
WAIT_CAP = 230
WAIT_UNKNOWN = 60
POD_PERIOD = 118
VJ_HOLD = 9
T_HOP = 24
HOP_UP = 4
HOP_DOWN = 30
LEDGE_DROP = False
HOP_HOLD = 5
AIR_BRAKE_NEUTRAL = 26
AIR_BRAKE_LEFT = 16
FLAME_LOOK = 60
FBAR_MAXWAIT = 110
PILLAR_HOP = True
CORR_BAR = False
PMODE = True
RIDE_JUMP = False
RIDE_WALK_FY = 150
BOW_CLEAR = 12
HOP_REACH = 27
FILL_GAP = False
HOP_LONG = 12
NOX = True
FLOOR_BRAKE = False
FB_MIN = 36
RIDE_FY = 140
RIDE_LIM = 8
BR_GATE = False
BR_X0 = 2056
BR_END = 2200
BR_CAP = 160
BR_HID_MIN = 14
BR_HID_MAX = 50
lifts_real_scene = False
HOP_CL_MAX = 10
LEDGE_FY = 142
BOW_EARLY = True
AIR_VX = 1.1
AIR_FY = 126
PM_MARGIN = 20
RIDE_HOLD = True
LEDGE_MARGIN = 5.0
BOW_FAR = 30
BOW_NEAR = 16
BOW_Y0 = 134
BOW_Y1 = 158
BOW_D0 = 8
BOW_D1 = 36
TGT_ADD = 12.0
VEFF0 = 1.6
MIN_HOLD = 12
BAR_ACC = 0.1
BAR_AIR_ACC = 0.08
DEBUG = False
DEBUG_X = 1700
BOW_OVER = False
BOW_OVER_D = 38
BR_LOST = 400
BR_NEAR = 22
BR_OFF = 0
FLOOR_HOP = True
FIX_OM = True
BAR_OM = -3.0
RIM_FIX = True
RIM_D = 7
CORR_YMAX = 232
A_VMIN = -5.0
LIFT_HIGH = True
LIFT_HIGH_FY = 125
LEDGE_DELAY = 3
UP_BLOCK = True
LAST_FLAME_WAIT = True
LAST_X0 = 1990
BR_AVOID = True
BR_AVOID_IN = 14
BR_AVOID_AHEAD = 44
NEW_BAR_RESET = True
LEDGE_AIR_RUN = False
STALL_F = 900
FH_VX = 1.0
VIRT = True
VIRT_AGE = 30
VIRT_V = 1.5
LIFT_TOL = 4
LIFT_PRED = 10
PIV_LIFT = 60
LAST_D = 160
LAST_CAP = 700
BR_PRED = False
BR_DG = True
DG_W = 18
BP_X0 = 2040
BP_BACK = 22
BP_AHEAD = 46
BP_COL = 18
BP_MIN = 6
BP_PRE = 42
BP_CAP = 260
NEAR_WAIT = False
NEAR_D = 115
NEAR_CAP = 200
NEG_FIX = False
NEG_V = -0.4
PG = False
PG_X = 2103
PG_A = 2030
PG_Z = 2120
PG_HOLD_X = 2046
PG_COMMIT = 2074
PG_FWD_X = 2046
PG_AIR = True
PG_AIR_L = 2062
PG_AIR_N = 2052
PG_Y = 150
PG_YMIN = 124
PG_FORCE_END = 2112
PG_BOW_MIN = 50
PG_GO_F = 45
PG_T_MAX = 24
PG_TRIG_X = 2028
PG_CAP = 420
BR_FAST = False
BR_DHI_V = 8.0
BR_RUN_X = 2090
FILT_PLAT = False
BJ_ON = True
BJ_D = 70
BJ_X = 2060
BR_SLOW = False
BR_SLOW_X = 2050
BR_VMAX = 1.3
POD_NOJUMP = True
POD_NJ_A = 30
POD_NJ_B = 20
AXE_JUMP = True
AXE_X = 2185
AXE_GAP = 24
RIM_FLAME_D = 7
NOWAIT_X = 0
FASTV_HOLD = 9
FASTV_ADD = 0
LIFT_GEO = False
AXE_SPRINT = False
AXE_OVR = 20
AXE_OVL = 24
AXE_SPRINT_X = 2205
AXE_SPRINT_FY = 152
AXE_BACK = True
AXE_BACK_CAP = 400

TUNABLE = {
    "FLAME_X": (60, 220),
    "TGT_ADD": (4.0, 18.0),
    "MIN_HOLD": (8, 20),
    "HID_MIN": (30, 75),
    "POD_MIN": (8, 30),
    "POD_MAX": (35, 70),
    "BAR_MIN": (8, 30),
    "BAR_WIN": (40, 90),
    "WAIT_CAP": (120, 400),
    "VJ_HOLD": (6, 14),
    "HOP_HOLD": (3, 9),
    "HOP_LONG": (8, 16),
    "FLAME_LOOK": (30, 100),
    "BR_CAP": (50, 250),
    "BR_HID_MIN": (6, 30),
    "BR_HID_MAX": (35, 90),
    "PM_MARGIN": (8, 34),
    "BOW_FAR": (28, 70),
    "BOW_NEAR": (10, 34),
    "BOW_OVER_D": (20, 60),
    "LAST_FLAME_WAIT": [True, False],
    "HOP_DOWN": (6, 30),
    "LEDGE_MARGIN": (3.0, 12.0),
    "DG_W": (12, 28),
    "LEDGE_ANG_MAX": (0, 90),
    "LEDGE_FY": (110, 175),
    "LIFT_HIGH_FY": (90, 150),
    "LAST_D": (100, 230),
    "LAST_CAP": (150, 900),
    "BP_COL": (10, 30),
    "BP_PRE": (20, 70),
    "BP_AHEAD": (24, 70),
    "NEAR_D": (60, 180),
    "NEAR_CAP": (80, 400),
    "LIFT_TOL": (0, 10),
    "LIFT_PRED": (0, 20),
    "AXE_GAP": (10, 60),
}


def solid_orange(o):
    e = o[:-1, :-1] & o[1:, :-1] & o[:-1, 1:] & o[1:, 1:]
    out = np.zeros_like(o)
    out[:-1, :-1] |= e
    out[1:, :-1] |= e
    out[:-1, 1:] |= e
    out[1:, 1:] |= e
    return out


def gray_mask(s):
    r = s[:, :, 0]; g = s[:, :, 1]; b = s[:, :, 2]
    return (r == g) & (g == b) & ((r == 188) | (r == 124))


def mario_box(s):
    ys, xs = np.nonzero(s[32:, :, 0] == 172)
    if len(ys):
        ys = ys + 32
        k = (s[ys, xs, 1] == 124) & (s[ys, xs, 2] == 0)
        ys = ys[k]; xs = xs[k]
    if len(ys) < 6:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def scroll_shift(pg, cg):
    best = 0; bs = -1e9
    a = pg[40:210]; b = cg[40:210]
    for d in range(0, 7):
        if d == 0:
            sc = np.count_nonzero(a & b) - np.count_nonzero(a ^ b) * 0.5
        else:
            sc = np.count_nonzero(a[:, d:] & b[:, :-d]) - np.count_nonzero(a[:, d:] ^ b[:, :-d]) * 0.5
        if sc > bs:
            bs = sc; best = d
    return best


def fire_mask(s, ymax=203):
    m = np.zeros(s.shape[:2], bool)
    if ymax > 40:
        t = s[40:ymax]
        r = t[:, :, 0]; g = t[:, :, 1]; b = t[:, :, 2]
        m[40:ymax] = ((r == 248) & (g == 56) & (b == 0)) | ((r == 252) & (g == 160) & (b == 68))
    return m


def blobs(m):
    h, w = m.shape
    c = m.reshape(h // 4, 4, w // 4, 4).sum((1, 3), dtype=np.uint8) >= 3
    out = []
    ys0, xs0 = np.nonzero(c)
    if len(ys0) == 0:
        return out
    yl = ys0.tolist(); xl = xs0.tolist()
    cells = set(zip(yl, xl))
    for y, x in zip(yl, xl):
        if (y, x) not in cells:
            continue
        cells.discard((y, x))
        st = [(y, x)]
        mnx = mxx = x; mny = mxy = y; n = 0
        while st:
            a, b = st.pop(); n += 1
            if b < mnx: mnx = b
            elif b > mxx: mxx = b
            if a < mny: mny = a
            elif a > mxy: mxy = a
            for p in (a - 1, a, a + 1):
                for q in (b - 1, b, b + 1):
                    if (p, q) in cells:
                        cells.discard((p, q)); st.append((p, q))
        out.append((mnx * 4, mny * 4, mxx * 4 + 3, mxy * 4 + 3, n))
    return out


def find_lifts(s):
    out = []
    cand = np.nonzero((s[32:230, :, 0] == 252).sum(1) >= 24)[0]
    if len(cand) == 0:
        return out
    fcache = {}

    def fcum(row):
        v = fcache.get(row)
        if v is None:
            r = s[row, :, 0]; gg = s[row, :, 1]; b = s[row, :, 2]
            fire = ((r == 248) & (gg == 56) & (b == 0)) | ((r == 252) & (gg == 160) & (b == 68))
            v = np.zeros(257, np.int16)
            np.cumsum(fire, out=v[1:])
            fcache[row] = v
        return v

    for y in (cand + 32).tolist():
        if y + 7 >= 240:
            continue
        r = s[y, :, 0]; gg = s[y, :, 1]; b = s[y, :, 2]
        white = (r == 252) & (gg == 252) & (b == 252)
        wc = np.zeros(257, np.int16)
        np.cumsum(white, out=wc[1:])
        xs = np.nonzero((wc[24:] - wc[:-24]) == 24)[0].tolist()
        if not xs:
            continue
        fa = fcum(y + 1); fb = fcum(y + 6)
        for x in xs:
            f1 = fa[x + 24] - fa[x]
            f2 = fb[x + 24] - fb[x]
            if f1 >= 20 and f2 >= 20:
                if not any(abs(x - o[0]) <= 8 and abs(y - o[1]) <= 3 for o in out):
                    out.append((int(x), int(y)))
    return out


def podoboos(s, mb, gm, orange, lifts=(), ymax=203):
    m = fire_mask(s, ymax)
    for (lx, ly) in lifts:
        m[ly:ly + 9, lx - 1:lx + 25] = False
    if mb is not None:
        x0, x1, y0, y1 = mb
        m[max(0, y0 - 6):y1 + 3, max(0, x0 - 3):x1 + 4] = False
        if lifts:
            m[y1 + 1:y1 + 11, max(0, x0 - 7):x1 + 8] = False      # the lift body under Mario's feet would merge with a flame
            orange = orange.copy()
            orange[y1 + 1:y1 + 11, max(0, x0 - 9):x1 + 10] = False  # ... and must not count as a firebar pivot next to a flame
            for (lx, ly) in lifts:
                orange[ly:ly + 9, max(0, lx - 1):lx + 25] = False
    res = []
    flames = []
    other = m.copy()
    for b in blobs(m):
        w = b[2] - b[0] + 1; h = b[3] - b[1] + 1
        ng = int(gm[b[1]:b[3] + 1, b[0]:b[2] + 1].sum())
        if w >= 20 and h <= 14 and b[4] >= 5 and b[1] < 200 and ng <= 6 and \
                int(orange[max(0, b[1] - 10):b[3] + 11, max(0, b[0] - 10):b[2] + 11].sum()) < (PIV_LIFT if lifts else 20):
            flames.append(b[:4])
            other[b[1]:b[3] + 1, b[0]:b[2] + 1] = False
        elif b[4] <= 24 and 12 <= w <= 24 and 12 <= h <= 32:
            if ng <= 24 and int(((s[b[1]:b[3] + 1, b[0]:b[2] + 1, 0] == 252) & (s[b[1]:b[3] + 1, b[0]:b[2] + 1, 1] == 252) & (s[b[1]:b[3] + 1, b[0]:b[2] + 1, 2] == 252)).sum()) >= 4:
                res.append(b[:4])
                other[b[1]:b[3] + 1, b[0]:b[2] + 1] = False
        elif 16 <= w <= 40 and h <= 14 and b[4] >= 5 and b[1] < 200 and ng <= 6:
            if w >= 2 * h - 2:
                near_piv = int(orange[max(0, b[1] - 10):b[3] + 11, max(0, b[0] - 10):b[2] + 11].sum())
                if near_piv < (PIV_LIFT if lifts else 20):
                    flames.append(b[:4])
                    other[b[1]:b[3] + 1, b[0]:b[2] + 1] = False
    return res, other, flames


def predict_landing(g, cx, fy, vx, vy, steps=70):
    x = float(cx); y = float(fy); v = float(vy)
    for k in range(steps):
        v = min(v + 0.4, 5.0)
        y0 = int(y); y += v; x += vx
        xi = int(x)
        if xi < 4 or xi > 251 or y > 236:
            return None
        for yy in range(y0 + 1, int(y) + 2):
            if yy < 240 and g[yy, xi - 3:xi + 4].any() and not g[yy - 1, xi - 3:xi + 4].all():
                r = xi
                while r < 255 and g[yy, r + 1]:
                    r += 1
                return xi, yy, r
    return None


FSPEED = {"v": 0.0, "n": 0}


def flame_speed(ftrack, b, frame, world=0):
    for e in ftrack:
        if e[4] == frame and abs(e[0] - b[1]) <= 4 and abs(e[3] - b[0]) <= 1:
            if frame - e[2] >= 4:
                v = (e[1] - (world + b[0])) / float(frame - e[2])
                v = max(1.0, min(4.0, v))
                FSPEED["v"] = v
                FSPEED["n"] += 1
                return v
            return FSPEED["v"] if FSPEED["n"] else 2.0
    return FSPEED["v"] if FSPEED["n"] else 2.0


def update_ftrack(ftrack, flames, frame, world=0):
    cur = []
    for b in flames:
        hit = None
        for e in ftrack:
            if e[4] < frame and frame - e[4] <= 8 and abs(e[0] - b[1]) <= 4 and -6 <= e[3] - b[0] <= 14 + 6 * (frame - e[4]):
                hit = e
                break
        if hit is None:
            hit = [b[1], world + b[0], frame, b[0], frame]
        else:
            hit[3] = b[0]; hit[4] = frame
        cur.append(hit)
    ftrack[:] = cur + [e for e in ftrack if e[4] < frame and frame - e[4] <= 8 and e not in cur]


_LASTP = {}
HOLD_T = {4: 23, 8: 28, 12: 33, 16: 37, 20: 45, 26: 62}


DX_RAMP = True
DX_VC = 1.75
DX_ACC = 0.07


def dx_est(h, vx):
    T = float(HOLD_T[h])
    v0 = max(vx, 0.0)
    if not DX_RAMP:
        return T * (VEFF0 + 0.1 * v0)
    vc = max(DX_VC, v0)
    if v0 >= vc:
        return vc * T
    ta = (vc - v0) / DX_ACC
    if T <= ta:
        return v0 * T + 0.5 * DX_ACC * T * T
    return v0 * ta + 0.5 * DX_ACC * ta * ta + vc * (T - ta)


_CXS = {"v": None}
GAPFIX = True


def choose_hold(sup, cx, vx):
    if _CXS.get("v") is not None:
        cx = _CXS["v"]
    n = len(sup)
    if FILL_GAP:
        sup = [bool(v) for v in sup]
        i = cx
        while i < n:
            if not sup[i]:
                j = i
                while j < n and not sup[j]:
                    j += 1
                if j < n and (j - i) <= 3:
                    for k in range(i, j):
                        sup[k] = True
                i = j
            else:
                i += 1
    x = cx
    while x < n and sup[x]:
        x += 1
    e1 = x
    while x < n and not sup[x]:
        x += 1
    if x >= n:
        return None
    e2 = x
    while x < n and sup[x]:
        x += 1
    e3 = x
    p = min(e3 - e2, 40)
    _LASTP["p"] = p
    _LASTP["w"] = e2 - e1
    _LASTP["need"] = e2 - cx
    target = (e2 - cx) + min(p / 2.0, TGT_ADD)
    v_eff = VEFF0 + 0.1 * max(vx, 0.0)
    best = None
    be = 1e9
    for h in (4, 8, 12, 16, 20, 26):
        dx = dx_est(h, vx)
        if dx * 0.9 < (e2 - cx) + 6:
            continue
        err = abs(dx - target)
        if err < be:
            be = err
            best = h
    return best


def bar_omega(hist, default):
    if len(hist) < 2:
        return default
    i = len(hist) - 1
    while i > 0 and hist[i][0] - hist[i - 1][0] <= 3:
        i -= 1
    f0, a0 = hist[i]
    f1, a1 = hist[-1]
    if f1 - f0 < 10:
        return default
    d = (a1 - a0 + 180.0) % 360.0 - 180.0
    w = d / float(f1 - f0)
    if -4.5 <= w <= -1.5:
        return w
    return default


def bar_crossing_safe(dxp, ang, omega, lane_y=17.0, margin=5.0, nfire=6, yc0=None, x_edge=None, v0=0.4, stop_land=False, delay=0):
    x = -float(dxp)
    v = v0
    th = ang
    yc = lane_y if yc0 is None else float(yc0)
    vyy = 0.0
    fall = False
    landed = 0
    for t in range(130):
        if -62.0 <= x <= 62.0:
            c = np.cos(np.radians(th)); s_ = np.sin(np.radians(th))
            for i in range(1, nfire + 1):
                r = 8.0 * i
                if abs(r * c - x) < 9.0 + margin and abs(-r * s_ - yc) < 12.0 + margin:
                    return False
        if x > 62.0:
            return True
        if t < delay:
            th += omega     # start-up latency: the bar keeps turning while Mario has not moved yet
            continue
        if fall and yc < lane_y:
            v = min(2.9, v + BAR_AIR_ACC)
        else:
            v = min(2.9, v + BAR_ACC)
        x += v
        th += omega
        if x_edge is not None and x >= x_edge:
            fall = True
        if fall and yc < lane_y:
            vyy = min(vyy + 0.4, 5.0)
            yc = min(lane_y, yc + vyy)
        if fall and yc >= lane_y:
            landed += 1
        if stop_land and landed >= 4:
            return True
    return True


def find_coins(o):
    ys, xs = np.nonzero(o[40:212])
    n = len(ys)
    if n == 0 or n > 1500:
        return []
    pts = set(zip((ys + 40).tolist(), xs.tolist()))
    out = []
    while pts:
        p = pts.pop()
        st = [p]
        comp = [p]
        while st:
            y, x = st.pop()
            for dy in (-1, 0, 1):
                for dx in (-2, -1, 0, 1, 2):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.discard(q)
                        st.append(q)
                        comp.append(q)
        cy = [c[0] for c in comp]
        cxx = [c[1] for c in comp]
        w = max(cxx) - min(cxx) + 1
        h = max(cy) - min(cy) + 1
        if 4 <= w <= 10 and 10 <= h <= 15 and len(comp) >= 20:
            out.append((min(cxx), max(cxx), min(cy), max(cy)))
    return out


def fireball_angle(s, px, py):
    y0 = max(0, py - 62); y1 = min(207, py + 62); x0 = max(0, px - 62); x1 = min(256, px + 62)
    sub = s[y0:y1, x0:x1]
    wm = (sub[:, :, 0] == 252) & (sub[:, :, 1] == 252) & (sub[:, :, 2] == 252)
    om = (sub[:, :, 0] == 252) & (sub[:, :, 1] == 160) & (sub[:, :, 2] == 68)
    ys, xs = np.nonzero(wm)
    if len(ys) == 0 or len(ys) > 400:
        return None, 0
    H, W = wm.shape
    cen = []
    for y, x in zip(ys.tolist(), xs.tolist()):
        a0 = max(0, y - 2); a1 = min(H, y + 3); b0 = max(0, x - 2); b1 = min(W, x + 3)
        if wm[a0:a1, b0:b1].sum() > 3 or om[a0:a1, b0:b1].sum() < 3:
            continue
        rx = x + x0 - px; ry = y + y0 - py
        r = (rx * rx + ry * ry) ** 0.5
        if r < 6 or r > 58:
            continue
        cen.append((rx, -ry))
    if not cen:
        return None, 0
    pred = _FB.get("pred")
    best = None
    for (cx_, cy_) in cen:
        th = np.degrees(np.arctan2(cy_, cx_))
        c = np.cos(np.radians(th)); s_ = np.sin(np.radians(th))
        sup = 0
        for (qx, qy) in cen:
            along = qx * c + qy * s_
            perp = abs(qx * s_ - qy * c)
            if along > 4 and perp <= 4.5:
                sup += 1
        score = float(sup)
        if pred is not None:
            d = abs((th - pred + 180.0) % 360.0 - 180.0)
            score -= d / 25.0
        if best is None or score > best[0]:
            best = (score, th, sup)
    if pred is not None:
        d = abs((best[1] - pred + 180.0) % 360.0 - 180.0)
        if d > 40.0 and best[2] < 3:
            return pred, 0
    return float(best[1]), best[2]


_FB = {}


def bar_info(orange, bar, red, cx, fy, s=None, frame=None):
    xa = max(0, cx - 20); ya = 140
    reg = orange[ya:215, xa:min(256, cx + 175)].astype(np.int32)
    h, w = reg.shape
    if h < 17 or w < 17:
        return None
    S = np.zeros((h + 1, w + 1), np.int32)
    S[1:, 1:] = reg.cumsum(0).cumsum(1)
    box = S[16:, 16:] - S[:-16, 16:] - S[16:, :-16] + S[:-16, :-16]
    mx = int(box.max())
    if mx < 40:
        return None
    cols = np.nonzero((box >= max(40, int(0.85 * mx))).any(0))[0]
    c0 = int(cols[0])
    rows = np.nonzero(box[:, c0] >= max(40, int(0.85 * mx)))[0]
    px = xa + c0 + 8
    py = ya + int(rows[0]) + 8
    yb0 = max(0, py - 60); xb0 = max(0, px - 56)
    sub = (bar & red)[yb0:py + 60, xb0:px + 56]
    bys, bxs = np.nonzero(sub)
    rx = bxs + xb0 - px
    ry = bys + yb0 - py
    d2 = rx * rx + ry * ry
    mm = (d2 > 100) & (d2 < 3136)
    n = int(mm.sum())
    ang = None
    if n >= 8:
        mdx = float(rx[mm].mean()); mdy = float(ry[mm].mean())
        ang = float(np.degrees(np.arctan2(-mdy, mdx)))
    if s is not None:
        if frame is not None:
            if abs(py - _FB.get("py", py)) > 10 or frame - _FB.get("f", -999) > 30:
                _FB["pred"] = None
            _FB["py"] = py
            if _FB.get("pred") is not None:
                om_ = -2.15 if py < 175 else -3.0
                _FB["pred"] = _FB["ang"] + om_ * (frame - _FB["f"])
        a2, n2 = fireball_angle(s, px, py)
        if a2 is not None:
            ang = a2
            n = max(n, n2)
            if frame is not None and n2 > 0:
                _FB["ang"] = a2
                _FB["f"] = frame
                _FB["pred"] = a2
        elif frame is not None and _FB.get("pred") is not None:
            ang = _FB["pred"]
    return {"px": px, "py": py, "dxp": px - cx, "n": n, "ang": ang}


RING = []
_DB = {}
STALE_FIX = True
LEDGE_RECHECK = True
LIFT_STAND_HOLD = 10
LEDGE_DROP2 = True
POD_GATE = False
DG_BM = 12
BR_AVOID_AHEAD2 = 44
POD_GATE_W = 28
VJ_LEFT_LIFT = True
DG_PH = 118
DG_CAP = 3000
FH_ASC = True
LIFT_VJ_HOLD = 9
LD_V = 1.2
FLOOR_DELAY = 2
JG = True
HOP_ABN = 6
HOP_ABL = 2
JG_PLAT_T = 30
JG_SPAWN = False
JG_PLAT = False
JG_CAP = 320
JG_XM = 5
JG_YM = 3
JG_DELAY = 3
JG_V = 1.7
JG_FV = 1.45


JG_RAMP = True
JG_REL = 3


def _jg_hits(mw, fy0, vm, hold, flames, world, fv, delay=JG_DELAY, plateau=0.0, vt=None, xm=None, ym=None, vx0=None):
    if xm is None:
        xm = JG_XM
    if ym is None:
        ym = JG_YM
    H = min(90.0, max(30.0, 34.0 + 52.0 * (hold - 4) / 22.0))
    tr = 8.0 + 0.6 * hold
    tf = tr + (2.0 * H / 0.4) ** 0.5 + 4.0
    if plateau > 0:
        tf += JG_PLAT_T
    for t in range(0, int(tf) + delay + 2, 2):
        s = t - delay
        if s <= 0:
            dy = 0.0
            mx = mw
        else:
            if JG_RAMP and vx0 is not None and vx0 < vm:
                ta = (vm - vx0) / 0.07
                mx = mw + (vx0 * s + 0.035 * s * s if s <= ta else vx0 * ta + 0.035 * ta * ta + vm * (s - ta))
            else:
                mx = mw + vm * s
            if s <= tr:
                dy = H * (1.0 - (1.0 - s / tr) ** 2)
            else:
                dy = max(plateau, H - 0.2 * (s - tr) ** 2)
        top = fy0 - dy - 16.0
        bot = fy0 - dy
        if vt is not None and t >= vt:
            vl = world + 237.0 + vm * max(0.0, vt - delay) - fv * (t - vt)
            vr = vl + 26.0
            if vr > mx - 7 - xm and vl < mx + 7 + xm and 158 + ym >= top and 112 - ym <= bot:
                return True
        for b in flames:
            fl = world + b[0] - fv * t
            fr = world + b[2] - fv * t
            if fr > mx - 7 - xm and fl < mx + 7 + xm and b[3] + ym >= top and b[1] - ym <= bot:
                return True
    return False


def _jg_plan(sup, cx, vx, fy, world, flames, gap_ahead, wall, fsp, st, pos_now, commit, vT=None):
    """-> (wait, alt_hold): wait = a flame is predicted to hit the planned jump; alt_hold = another hold that avoids it."""
    if abs(pos_now - st["pos"]) > 30:
        st["pos"] = pos_now
        st["n"] = 0
    fl = [b for b in flames if b[2] > cx - 20]
    relaxed = st["n"] >= JG_CAP
    if st["n"] >= JG_CAP * JG_REL or (not fl and vT is None):
        return False, None
    hold = 26
    need = None
    pw = 40
    if gap_ahead and not wall:
        _LASTP["p"] = 99
        ch = choose_hold(sup, cx, vx)
        need = _LASTP.get("need", None)
        pw = _LASTP.get("p", 40)
        if ch is not None:
            hold = max(ch, 12)
            if PILLAR_HOP and pos_now < 2090 and (_LASTP.get("p", 99) <= 24 or (PMODE and _LASTP.get("w", 99) <= 20 and _LASTP.get("need", 99) + 6 <= HOP_REACH)):
                hold = HOP_LONG if _LASTP.get("w", 99) <= 3 else HOP_HOLD
        else:
            need = None
    elif wall:
        hold = 20
    fv = JG_FV
    if fsp.get("n", 0) > 0:
        fv = min(2.2, max(1.1, fsp["v"]))
    vm = max(JG_V, vx)

    plat = 48.0 if (JG_PLAT and wall and not gap_ahead and fy >= 190) else 0.0

    def hits(h):
        for f_ in ((1.0,) if relaxed else (0.9, 1.12)):
            if _jg_hits(world + cx, fy, vm, h, fl, world, fv * f_, plateau=plat, vt=vT,
                        xm=(0 if relaxed else None), ym=(0 if relaxed else None), vx0=max(vx, 0.0)):
                return True
        return False

    if not hits(hold):
        return False, None
    if need is not None:
        v_eff = VEFF0 + 0.1 * max(vx, 0.0)
        tgt = need + min(pw / 2.0, TGT_ADD)
        opts = []
        for h in (4, 8, 12, 16, 20, 26):
            dx = dx_est(h, vx)
            if dx * 0.9 < need + 6 or dx > need + pw + 4:
                continue
            opts.append((abs(dx - tgt), h))
        opts.sort()
        for _e, h in opts:
            if h != hold and not hits(h):
                return False, h
    if commit:
        return False, None
    st["n"] += 1
    return True, None


def play(sim):
    try:
        _play(sim)
    finally:
        if DEBUG:
            print("RINGTAIL")
            for ln in RING[-70:]:
                print(ln)


VX_N = 4
PM_CV = 2.3
PM_VK = 6.0
LOWHOP = True
LOWHOP_D = 110
LOWHOP_H = 4
SUP_CLOSE = 2
BACK_FIX = True
GOV_V2 = 0.4
GOV_C = 4.0
GOV_D = 12.0
PM_LEFT = 5
GOV = True
GOV_V = 1.3
GOV_A = 6.0
GOV_B = 9.0


def _play(sim):
    posh = []
    vx = 0.0
    pod_cols = []
    last_go = -99
    vjump = False
    wall_tries = {}
    jump_lim = JUMP_HOLD
    prev_lifts = []
    hop_mode = False
    piv_commit = False
    bar_clear = 0
    wait_start = 0
    wait_pos = -999
    prev_pos = None
    pg = None
    world = 0
    prev_fy = None
    zprev = False
    hold = 0
    still = 0
    prev_scr = None
    lift_wait = 0
    lift_wait2 = 0
    ftrack = []
    vtrack = []
    fbar_piv = None
    br_blk = 0
    prog = [0, 0]
    fbar_commit = False
    fbar_wait = 0
    fbar_wait2 = 0
    fbar_hist = []
    bow_commit = False
    bow_prev = None
    vj_lim = VJ_HOLD
    vyh = []
    ride_on = False
    still_n = 0
    flw = 0
    hop_len = HOP_HOLD
    br_wait = 0
    hop_cl_n = 0
    why = ""
    gfy = 0
    lfw = 0
    lhw = 0
    jg_st = {"pos": -999, "n": 0}
    spawn_f = -999
    spawn_iv = []
    nw = 0
    bp_blk = 0
    dg_n = 0
    ldrop_on = False
    pg_go = 0
    pg_wait = 0
    pg_rising = False
    pg_prev_y = None
    pg_t0 = -999
    axe_back_n = 0
    while True:
        s = sim.screen()
        dist_gap = 99
        back = 0
        commit_flag = False
        bow_bb = None
        gm = gray_mask(s)
        orange_raw = (s[:, :, 0] == 228) & (s[:, :, 1] == 92) & (s[:, :, 2] == 16)
        orange = solid_orange(orange_raw)
        coins = find_coins(orange_raw)
        for (cx0, cx1, cy0, cy1) in coins:
            orange[cy0:cy1 + 1, max(0, cx0 - 1):cx1 + 2] = False
        g = gm | orange
        for (cx0, cx1, cy0, cy1) in coins:
            g[cy0:cy1 + 1, max(0, cx0 - 1):cx1 + 4] = False
        if pg is not None:
            world += scroll_shift(pg, gm)
        pg = gm
        lifts = find_lifts(s) if world < 1950 else []
        for (lx, ly) in lifts:
            g[ly:ly + 8, lx:lx + 24] = True
        new_lifts = []
        for (lx, ly) in lifts:
            wlx = world + lx
            v = 0.0
            for p in prev_lifts:
                if abs(p[0] - wlx) <= 4 and abs(p[1] - ly) <= 6:
                    v = 0.6 * p[2] + 0.4 * (ly - p[1])
                    break
            new_lifts.append((wlx, ly, v))
        prev_lifts = new_lifts
        lift_v = {(lx, ly): nl[2] for (lx, ly), nl in zip(lifts, new_lifts)}
        mb = mario_box(s)
        if prev_scr is not None and np.array_equal(prev_scr[::4, ::4], s[::4, ::4]):
            still += 1
        else:
            still = 0
        prev_scr = s
        if still > 90:
            if DEBUG:
                print("END frame", sim.frame, "world", world, "mario", mb)
            return
        if mb is None:
            sim.step({"Right"})
            continue
        rim_d = RIM_D
        axe_ov = False
        cx = (mb[0] + mb[1]) // 2
        fy = mb[3] + 1
        feet_solid = g[fy:fy + 2, max(0, cx - 5):cx + 6].any()
        keys = {"Right", "x"}
        vy = 0 if prev_fy is None else fy - prev_fy
        prev_fy = fy
        on_ground = feet_solid or (vy == 0 and bool(g[fy:fy + 6, max(0, cx - 5):cx + 6].any()))
        vyh.append(vy)
        if len(vyh) > 4:
            del vyh[0]

        def solid_at(dx0, dx1, y0, y1):
            return g[max(0, y0):max(0, y1), min(255, cx + dx0):min(256, cx + dx1)].any()

        pods, bar, flames = podoboos(s, mb, gm, orange, lifts, CORR_YMAX if (world + cx) < 1500 else 203)
        if FILT_PLAT and world + cx >= 1990:
            # the orange platform above the bridge (y96-107) looks like a flame but is static in the world
            flames = [b for b in flames if not (92 <= b[1] <= 99 and b[3] <= 110)]
        update_ftrack(ftrack, flames, sim.frame, world)
        if world + cx >= 1690:
            for b in flames:
                if b[0] >= 225 and 100 <= b[1] <= 160 and sim.frame - spawn_f > 15:
                    if spawn_f > 0 and sim.frame - spawn_f < 200:
                        spawn_iv.append(sim.frame - spawn_f)
                    spawn_f = sim.frame
                    break
        if VIRT and world + cx >= 1690:
            fr_ = sim.frame
            seen_e = set()
            for b in flames:
                wx0 = world + b[0]
                hit_e = None
                for ei, e in enumerate(vtrack):
                    if ei in seen_e or e[2] >= fr_:
                        continue
                    pred = e[1] - VIRT_V * (fr_ - e[2])
                    if abs(e[0] - b[1]) <= 6 and -8 <= pred - wx0 <= 10 + 2 * (fr_ - e[2]):
                        hit_e = ei
                        break
                if hit_e is None:
                    vtrack.append([b[1], wx0, fr_, max(24, b[2] - b[0]), max(8, b[3] - b[1]), 1])
                    seen_e.add(len(vtrack) - 1)
                else:
                    e = vtrack[hit_e]
                    e[0] = b[1]; e[1] = wx0; e[2] = fr_; e[3] = max(24, b[2] - b[0]); e[4] = max(8, b[3] - b[1])
                    e[5] += 1
                    seen_e.add(hit_e)
            keep_v = []
            for e in vtrack:
                if e[2] == fr_:
                    keep_v.append(e)
                    continue
                age = fr_ - e[2]
                px_ = e[1] - VIRT_V * age - world       # predicted screen x0
                if age <= VIRT_AGE and px_ > cx - 40:
                    keep_v.append(e)
                    # only extrapolate flames that vanished close to Mario (detection loss near his box)
                    if e[5] >= 3 and px_ - cx <= 50:
                        flames = list(flames) + [(int(px_), int(e[0]), int(px_ + e[3]), int(e[0] + e[4]))]
            vtrack[:] = keep_v
        redm = (s[:, :, 0] == 248) & (s[:, :, 1] == 56) & (s[:, :, 2] == 0)
        binfo = bar_info(orange, bar, redm, cx, fy, s, sim.frame) if lifts else None
        seen_ids = set()
        for b in pods:
            wx = world + (b[0] + b[2]) // 2
            for i, c in enumerate(pod_cols):
                if abs(c[0] - wx) <= 14:
                    c[0] = wx; c[1] = sim.frame; c[2] = True
                    seen_ids.add(i)
                    break
            else:
                pod_cols.append([wx, sim.frame, True, None, POD_PERIOD])
                seen_ids.add(len(pod_cols) - 1)
        for i, c in enumerate(pod_cols):
            if i in seen_ids or not c[2]:
                continue
            sxp = c[0] - world
            if sim.frame - c[1] >= 3 and 8 <= sxp <= 240:
                c[2] = False
                if c[3] is not None:
                    gp = c[1] - c[3]
                    if 90 < gp < 150:
                        c[4] = gp
                c[3] = c[1]
        pos_now = world + cx
        # fixed bridge podoboo column tracker (world x ~2103)
        pg_seen = None
        for b in pods:
            wxp = world + (b[0] + b[2]) // 2
            if abs(wxp - PG_X) <= 8:
                pg_seen = b[1] if (pg_seen is None or b[1] > pg_seen) else pg_seen
        if pg_seen is not None:
            if pg_seen >= 170:
                pg_rising = True
            elif pg_prev_y is not None and pg_seen > pg_prev_y + 1:
                pg_rising = False
            elif pg_prev_y is not None and pg_seen < pg_prev_y - 1:
                pg_rising = True
            if pg_rising and PG_YMIN <= pg_seen <= PG_Y and pg_go <= 0 and pos_now >= PG_TRIG_X:
                pg_go = PG_GO_F
            pg_prev_y = pg_seen
        else:
            pg_prev_y = None
        if pg_go > 0:
            pg_go -= 1
        if pos_now >= PG_Z:
            pg_wait = 0
        if pos_now > prog[0] + 6:
            prog[0] = pos_now
            prog[1] = sim.frame
        desp = (pos_now >= 1690) and (sim.frame - prog[1] > STALL_F)
        if (not on_ground) and (not zprev) and lifts and len(vyh) == 4 and pos_now >= 1330 and \
                all(abs(v) <= 1 for v in vyh) and sum(abs(v) for v in vyh) >= 3 and not (min(vyh) < 0 < max(vyh)):
            for b in flames:
                if b[2] > cx - 6 and b[3] >= fy - 15 and b[1] <= fy - 1 + LIFT_TOL + (LIFT_PRED if vy > 0 else 0) and 0 <= b[0] - (cx + 6) <= 44:
                    on_ground = True
                    break
        if zprev and hold < 4:
            keys = {"z"} if vjump else ({"Right", "z"} if hop_mode else {"Right", "x", "z"})
            if vjump and (pos_now >= 1690 or (lifts and VJ_LEFT_LIFT)) and vx > 0.6:
                keys = {"z", "Left"}
            hold += 1
        elif on_ground and vy != 0 and pos_now >= 1690 and not zprev:
            keys = {"Left"} if vx > 1.0 else set()
        elif on_ground:
            if zprev:
                keys = set() if vjump else ({"Right"} if hop_mode else {"Right", "x"})
                zprev = False
                hold = 0
                hop_mode = False
                vjump = False
            else:
                if STALE_FIX:
                    vjump = False       # landed: a stale vertical-jump flag made later ledge walk-offs brake with Left in the air
                    hop_mode = False
                sup = g[fy:fy + 17, :].any(0)
                if SUP_CLOSE and pos_now >= 1690:
                    d_ = sup.copy()
                    for s_ in range(1, SUP_CLOSE + 1):
                        d_[:-s_] |= sup[s_:]
                        d_[s_:] |= sup[:-s_]
                    e_ = d_.copy()
                    for s_ in range(1, SUP_CLOSE + 1):
                        e_[:-s_] &= d_[s_:]
                        e_[s_:] &= d_[:-s_]
                    sup = e_ | sup      # tiny (<=2*SUP_CLOSE px) holes inside a pillar are brick seams, not gaps
                dist_gap = 99
                cxs0 = cx
                cx_save = cx
                if GAPFIX and pos_now >= 1690 and not sup[cx]:
                    # centre hangs over the void (3-5 px of the foot still on the pillar): measure from the supported foot,
                    # otherwise the pillar we stand on looks like the landing platform of a tiny gap
                    for d_ in range(1, 7):
                        if cx + d_ < 254 and sup[cx + d_]:
                            cxs0 = cx + d_
                            break
                        if cx - d_ > 2 and sup[cx - d_]:
                            cxs0 = cx - d_
                            break
                _CXS["v"] = cxs0
                cx = cxs0
                for dx in range(0, 70, 2):
                    xx = cx + dx
                    if xx >= 254:
                        break
                    if not sup[xx]:
                        dist_gap = dx
                        break
                deep = g[fy:240, :].any(0)
                pit_dist = 999
                back = 0
                while back < 124 and cx - back > 2 and sup[cx - back]:
                    back += 2
                if BACK_FIX and back < 6 and cx - 8 > 2 and sup[cx - 6]:
                    b2_ = 0
                    while b2_ < 124 and cx - 6 - b2_ > 2 and sup[cx - 6 - b2_]:
                        b2_ += 2
                    if b2_ >= 4:
                        back = 6 + b2_      # centre hangs over the void but the left foot is supported: retreat left is possible
                for dx in range(0, 140, 2):
                    xx = cx + dx
                    if xx >= 254:
                        break
                    if not deep[xx]:
                        pit_dist = dx
                        break
                cx = cx_save
                gap_ahead =dist_gap <= LOOK + min(6, int(2 * max(vx, 0.0)))
                near_gap = pit_dist < 46
                bar_here = bar[max(0, fy - 40):fy + 10, max(0, cx - 60):cx + 60].any()
                can_wait = pit_dist < 138 and (not bar_here or pos_now >= 1690) and not lifts
                wait_d = int(min(WAIT_DIST, max(18, back - 8)))
                if pos_now >= 1690:
                    wait_d = min(wait_d, 26)
                vavg = 2.2 if wait_d >= 60 else (1.8 if wait_d >= 36 else 1.4)
                t_arr = pit_dist / vavg + 3.0
                wall = solid_at(7, 12, fy - 14, fy - 2)
                pmode = PMODE and 1700 <= pos_now < 2090 and fy < 190 and not lifts
                CV = PM_CV if pmode else 1.0
                lift_keys = None
                lift_hop = False
                cand = [L for L in lifts if L[0] > cx + 4 and L[0] <= cx + 110]
                if cand and dist_gap < 70:
                    x0n = min(L[0] for L in cand)
                    shaft = [L for L in cand if abs(L[0] - x0n) <= 6]
                    brake_d = 12 + 11 * max(vx, 0.0)
                    if vx > 1.5 and dist_gap < brake_d and back >= 8:
                        lift_keys = {"Left"}
                    elif dist_gap > 20:
                        lift_keys = {"Right"} if (dist_gap < 60 or vx > 1.7) else {"Right", "x"}
                    elif dist_gap > 9:
                        if vx < 1.0:
                            lift_keys = {"Right"}
                        elif vx > 1.5 and back >= 8:
                            lift_keys = {"Left"}
                        else:
                            lift_keys = set()
                    else:
                        okl = []
                        for L in shaft:
                            yl = L[1] + lift_v.get(L, 0.0) * T_HOP
                            if fy - HOP_UP <= yl <= fy + HOP_DOWN and (pos_now < 1360 or fy <= 140):
                                okl.append(L)
                        hop_closed = False
                        if okl and binfo is not None and binfo["ang"] is not None and binfo["px"] - x0n <= 80:
                            a = binfo["ang"]
                            if a < -8 or a > 142:
                                hop_closed = True
                        if okl and hop_closed:
                            hop_cl_n += 1
                        elif not okl:
                            hop_cl_n = 0
                        if okl and hop_closed and not (pos_now >= 1360 and fy < 75) and hop_cl_n <= HOP_CL_MAX:
                            okl = []
                        if okl and (pos_now >= 1360 or vx >= A_VMIN or dist_gap <= 2):
                            lift_keys = {"Right", "z"}
                            lift_hop = True
                            lift_wait = 0
                        elif okl:
                            lift_keys = {"Right"}      # shaft A: a standing/backwards hop falls short of the lift, get forward speed first
                        else:
                            lift_wait += 1
                            if vx > 0.3 or dist_gap < 4:
                                lift_keys = {"Left"}
                            else:
                                lift_keys = set()
                            if lift_wait > 400:
                                lift_keys = {"Right", "z"}
                                lift_hop = True
                danger = False
                commit_flag = False
                if can_wait and (not wall or pos_now >= 1690) and lift_keys is None:
                    chk = [c for c in pod_cols if pos_now - 10 <= c[0] <= pos_now + pit_dist + 24]
                    pod_ok = True
                    for c in chk:
                        if pos_now >= 1690:
                            hid = sim.frame - c[1]
                            if c[2] or hid < HID_MIN or (HID_MAX < hid <= 150):
                                pod_ok = False
                            continue
                        if c[3] is None:
                            pod_ok = False
                        else:
                            tau = (sim.frame + t_arr - c[3]) % c[4]
                            if not (POD_MIN <= tau <= POD_MAX):
                                pod_ok = False
                            if c[2] and t_arr < 10:
                                pod_ok = False
                    if pos_now >= 1690:
                        for pb in pods:
                            pcx = (pb[0] + pb[2]) // 2
                            if cx - 20 <= pcx <= cx + 70:
                                pod_ok = False
                    if not chk and sim.frame - wait_start < WAIT_UNKNOWN:
                        pod_ok = False
                    if sim.frame - wait_start > (450 if pos_now >= 1690 else WAIT_CAP):
                        if pos_now >= 1690:
                            pod_ok = not any(c[2] or (sim.frame - c[1]) < 25 for c in chk)
                        else:
                            pod_ok = True
                    if not pod_ok and vx > 1.8 and pit_dist < 75 and sim.frame - last_go <= 3:
                        pod_ok = True
                    if pod_ok:
                        last_go = sim.frame
                    if not pod_ok:
                        danger = True
                        if pos_now >= 1690:
                            cn = vx > CV and dist_gap < 4 + 4.0 * vx and pit_dist < 60
                        else:
                            cn = vx > 1.5 and dist_gap < 8 + 6.5 * vx and pit_dist < 60
                        if cn:
                            commit_flag = True
                            danger = False
                else:
                    wait_start = sim.frame
                if pos_now >= 1690 and vx > CV and dist_gap < 4 + 4.0 * vx and pit_dist < 60:
                    commit_flag = True
                if abs(pos_now - wait_pos) > 12:
                    wait_pos = pos_now
                    wait_start = sim.frame
                piv_keys = None
                ledge_closed = False
                ledge_keys = None
                xa = max(0, cx - 20)
                reg = orange[max(0, fy - 64):min(240, fy + 48), xa:min(256, cx + 140)]
                oys, oxs = np.nonzero(reg)
                piv_found = False
                lift_scene = len(lifts) > 0
                if len(oxs) >= 16 and (pit_dist >= 50 or lift_scene):
                    x0 = oxs.min()
                    sel = oxs <= x0 + 20
                    if sel.sum() >= 16:
                        px = xa + int(oxs[sel].mean())
                        py = max(0, fy - 64) + int(oys[sel].mean())
                        dxp = px - cx
                        piv_found = True
                        bar_active = bool(bar[max(0, py - 38):py - 6, max(0, px - 56):px + 56].any())
                        if bar_active:
                            bar_clear = 0
                        else:
                            bar_clear += 1
                        if dxp >= 60 and pit_dist >= 50 and not lift_scene and pos_now < 1500:
                            go = piv_commit or ((not bar_active) and BAR_MIN <= bar_clear <= BAR_WIN)
                            if go:
                                piv_commit = True
                            else:
                                if dxp > BAR_WAIT + 8 + 6 * max(vx, 0.0):
                                    piv_keys = {"Right", "x"}
                                elif dxp < BAR_WAIT - 2 and back >= 8:
                                    piv_keys = {"Left"}
                                elif vx > 1.2 and dxp < BAR_WAIT + 6 * vx:
                                    piv_keys = {"Left"}
                                else:
                                    piv_keys = set()
                        elif dxp < -60:
                            piv_commit = False
                        if pit_dist < 50:
                            piv_keys = None
                if not piv_found:
                    bar_clear = 0
                    piv_commit = False
                if lift_scene and not cand and binfo is not None and 4 <= binfo["dxp"] <= 110 and fy > 95 and fy <= 170:
                    ang = binfo["ang"]
                    ang_ok = ang is not None and -65 <= ang <= LEDGE_ANG_MAX
                    fl_ahead = False
                    for b in flames:
                        if b[2] > cx - 10 and b[0] < cx + 200 and b[3] >= fy - 100 and b[1] <= max(fy + 4, 166):
                            fl_ahead = True
                    lift_wait2 += 1
                    if lift_wait2 > 300:
                        ang_ok = True; fl_ahead = False
                    if (not ang_ok or fl_ahead) and fy < LEDGE_FY:
                        ledge_closed = True
                        under = [L for L in lifts if L[0] - 2 <= cx <= L[0] + 26 and abs(L[1] - fy) <= 8]
                        left_ok = (not under) or cx > under[0][0] + 9
                        if dist_gap > 14:
                            ledge_keys = {"Right"}
                        elif (vx > 0.8 or dist_gap < 4) and left_ok:
                            ledge_keys = {"Left"}
                        else:
                            ledge_keys = set()
                fbar_keys = None
                fbar_drop = False
                fbar_zone = False
                b2 = None
                ang2 = None
                dxp2 = 0
                ld2 = False
                ld_on = False
                if (pos_now >= 1500 or (CORR_BAR and 450 <= pos_now < 1300 and fy >= 190)) and not lifts:
                    b2 = bar_info(orange, bar, redm, cx, fy, s, sim.frame)
                    if b2 is not None and NEW_BAR_RESET:
                        pw_ = world + b2["px"]
                        if fbar_piv is None or abs(pw_ - fbar_piv) > 50:
                            fbar_piv = pw_          # a different firebar: forget the commitment made for the previous one
                            fbar_commit = False
                            fbar_wait = 0
                            fbar_wait2 = 0
                            del fbar_hist[:]
                        else:
                            fbar_piv = pw_
                    ld2 = LEDGE_DROP2 and b2 is not None and fy < 190 and 165 <= b2["py"] <= 195 and b2["dxp"] >= 45 \
                        and not fbar_commit and not zprev
                    if ld2:
                        # ledge above the floor bar: walk off slowly and brake in the air, land far (>=60 px) from the pivot,
                        # then cross the floor from a standstill with the bar simulation
                        fbar_drop = True
                        dxp2 = b2["dxp"]
                        ang2 = b2["ang"]
                        fbar_keys = {"Right"} if vx < LD_V else set()
                    ld_on = bool(ld2)
                    ldrop_on = ld_on
                    if (not ld2) and b2 is not None and (165 <= b2["py"] <= 195 or (pos_now < 1500 and 207 <= b2["py"] <= 226)) and -54 < b2["dxp"] <= 150:
                        fbar_drop = True
                        dxp2 = b2["dxp"]
                        ang2 = b2["ang"]
                        if dxp2 < 24:
                            fbar_commit = True
                        if ang2 is not None:
                            fbar_hist.append((sim.frame, ang2))
                            if len(fbar_hist) > 24:
                                del fbar_hist[0]
                        if fbar_commit and not (LEDGE_RECHECK and fy < 190 and dxp2 >= 40 and vx < 0.9 and ang2 is not None):
                            fbar_drop = fy < 190
                            fbar_zone = True
                            if FLOOR_BRAKE and fy >= 190 and dxp2 >= FB_MIN and ang2 is not None:
                                om = bar_omega(fbar_hist, -3.0)
                                lane = float(208 - 8 - b2["py"])
                                safe2 = bar_crossing_safe(max(dxp2, 10), ang2, om, lane_y=lane, v0=max(vx, 0.4), margin=LEDGE_MARGIN)
                                if not safe2:
                                    fbar_zone = False
                                    if vx > 0.5 or dxp2 < 70:
                                        fbar_keys = {"Left"} if back >= 8 else set()
                                    else:
                                        fbar_keys = set()
                        else:
                            om = BAR_OM if FIX_OM else bar_omega(fbar_hist, -3.0)
                            lane = float(208 - 8 - b2["py"])
                            if fy < 190:
                                yc0 = float(fy - 8 - b2["py"])
                                xe = -float(dxp2) + float(min(dist_gap, 70))
                            else:
                                yc0 = lane
                                xe = None
                            safe = ang2 is not None and bar_crossing_safe(max(dxp2, 10), ang2, om, lane_y=lane, yc0=yc0, x_edge=xe, margin=LEDGE_MARGIN, v0=max(vx, 0.4), delay=(LEDGE_DELAY if (fy < 190 and vx < 0.8) else (FLOOR_DELAY if (fy >= 190 and vx < 0.8) else 0)))
                            fbar_wait += 1
                            if safe and fy < 190:
                                for fb in flames:
                                    if fb[2] > cx - 6 and fb[3] >= fy - 15 and fb[1] <= fy - 1 and fb[0] - (cx + 6) < 110:
                                        safe = False
                                        break
                            if fbar_wait > FBAR_MAXWAIT * 3:
                                safe = True
                            if not safe:
                                bd = 8.0 * max(vx, 0.0) + 4.0
                                if dxp2 - 80 > bd + 6:
                                    fbar_keys = {"Right", "x"} if dxp2 > 150 else {"Right"}
                                elif vx > 0.5:
                                    fbar_keys = {"Left"}
                                elif dxp2 > 86:
                                    fbar_keys = {"Right"}
                                elif dxp2 < 70 and back >= 8:
                                    fbar_keys = {"Left"}
                                else:
                                    fbar_keys = set()
                            else:
                                fbar_commit = True
                                fbar_zone = True
                                fbar_drop = fy < 190
                    elif b2 is None or b2["dxp"] <= -54:
                        fbar_commit = False
                        fbar_wait = 0
                if DEBUG:
                    _DB["s"] = "B[%s dxp%s ang%s com%d zone%d fk%s w%d]" % (None if b2 is None else (b2["px"], b2["py"]), dxp2, None if ang2 is None else int(ang2), int(fbar_commit), int(fbar_zone), None if fbar_keys is None else "".join(sorted(k[0] for k in fbar_keys)), fbar_wait)
                if 1530 <= pos_now <= 1700 and fy < 190 and not lifts and not fbar_commit and fbar_keys is None:
                    fbar_wait2 += 1
                    if fbar_wait2 < 250:
                        fbar_keys = set()
                bow_keys = None
                bow_wait = None
                bow_over = False
                gm_g = (s[90:175, :, 0] == 0) & (s[90:175, :, 1] == 168) & (s[90:175, :, 2] == 0)
                bow_present = int(gm_g.sum()) >= 100
                bow_ov = False
                if bow_present:
                    gys, gxs = np.nonzero(gm_g)
                    bow_bb = (int(gxs.min()), int(gxs.max()))
                    bow_ov = int(gxs.max()) > cx - BOW_CLEAR
                    bx0 = int(gxs.min()); bx1 = int(gxs.max()); by1 = int(gys.max()) + 90
                    dB = bx0 - (cx + 6)
                    if bx1 > cx - 4 and bx0 > cx - 10:
                        bow_air = by1 <= 148 and (bow_prev is None or by1 <= bow_prev)
                        bow_rise = bow_prev is not None and by1 < bow_prev
                        bow_prev = by1
                        if bow_commit:
                            if dB > 14 and by1 >= 155:
                                bow_commit = False
                            else:
                                bow_keys = {"Right", "x"}
                        elif BOW_EARLY and bow_rise and BOW_Y0 <= by1 <= BOW_Y1 and BOW_D0 <= dB <= BOW_D1:
                            bow_commit = True
                            bow_keys = {"Right", "x"}
                        elif (not BOW_EARLY) and bow_air and dB <= BOW_GO and dB >= -4:
                            bow_commit = True
                            bow_keys = {"Right", "x"}
                        elif BOW_OVER and by1 >= 150 and -4 <= dB <= BOW_OVER_D and pos_now >= 2090 and not any(
                                (b[2] > cx - 6 and b[0] - cx < 130 and b[1] >= 105 and b[3] >= fy - 60 and b[1] <= fy + 4) for b in flames):
                            bow_over = True      # Bowser walked up to us: jump over him instead of waiting to be touched
                        else:
                            if dB > BOW_FAR:
                                bow_wait = {"Right"} if vx < 1.3 else set()
                            elif dB < BOW_NEAR and back >= 6:
                                bow_wait = {"Left"}
                            else:
                                bow_wait = set()
                    else:
                        bow_commit = False
                axe_w = False
                if AXE_JUMP and wall and bow_present and pos_now >= AXE_X and dist_gap > 12 and fy < 190 \
                        and bow_bb is not None and (bow_bb[0] - (cx + 6) >= AXE_GAP or bow_bb[1] < cx):
                    # axe pillar wall ahead: hop it instead of pressing into Bowser (only when Bowser is not adjacent)
                    axe_w = True
                    bow_keys = None
                    bow_wait = None
                    bow_over = False
                if AXE_BACK and wall and bow_present and pos_now >= AXE_X and fy < 190 and back >= 8 and axe_back_n < AXE_BACK_CAP \
                        and bow_bb is not None and bow_bb[0] <= cx + AXE_OVR and bow_bb[1] >= cx - AXE_OVL:
                    axe_ov = True       # pressed against the axe pillar with Bowser right above/on us: back off instead of being landed on
                br_keys = None
                if BR_GATE and pos_now >= BR_X0 and fy < 190 and not lifts_real_scene:
                    nearest = None
                    for c in pod_cols:
                        dxc = c[0] - pos_now
                        if -14 <= dxc <= 64:
                            hid = sim.frame - c[1]
                            unsafe = bool(c[2]) or hid < BR_HID_MIN or hid > BR_HID_MAX
                            if hid > BR_LOST and dxc > BR_NEAR:
                                unsafe = False
                            if unsafe and (nearest is None or dxc < nearest[0]):
                                nearest = (dxc, hid, bool(c[2]))
                    if nearest is not None:
                        br_wait += 1
                    else:
                        br_wait = max(0, br_wait - 2)
                    if nearest is not None and br_wait <= BR_CAP:
                        dxc = nearest[0]
                        if dxc < 18 and vx > 1.0 and not nearest[2]:
                            br_keys = None
                        elif dxc < BR_NEAR:
                            br_keys = {"Left"} if dxc >= 8 else {"Right", "x"}
                        elif dxc < 40:
                            br_keys = {"Left"} if vx > 0.6 else set()
                        else:
                            br_keys = {"Right"} if vx < 0.8 else set()
                fl_jump = False
                fl_stand = None
                fl_fwd = False
                fl_vjl = VJ_HOLD
                low = None
                for b in flames:
                    if b[2] > cx - 6 and b[3] >= fy - 15 and b[1] <= fy - 1 + ((LIFT_TOL + (LIFT_PRED if vy > 0 else 0)) if lifts else 0):
                        d = b[0] - (cx + 6)
                        if d < -10 and 1330 <= pos_now <= 1500:
                            # the lift body under Mario (white row hidden) looks like a flame behind him; a real flame
                            # touching it merges into one wide blob: its leading edge is then ~15 px left of the blob's right edge
                            if b[2] > cx + 18:
                                d = (b[2] - 15) - (cx + 6)
                            else:
                                continue
                        if low is None or d < low[0]:
                            low = (d, b)
                rim_d = RIM_FLAME_D if (low is not None and 1690 <= pos_now < 2090) else RIM_D
                rim = RIM_FIX and 1690 <= pos_now < 2090 and fy < 190 and not lifts and dist_gap < rim_d and back >= 6 and vx < 1.2 and not commit_flag
                if low is not None and not (fbar_zone and fy >= 190):
                    d, b = low
                    fvv = flame_speed(ftrack, b, sim.frame, world)
                    fl_fwd = dist_gap >= 60 and pit_dist >= 60 and not wall
                    if fl_fwd:
                        dhi = 28
                        if BR_FAST and pos_now >= 2020 and fy < 190:
                            dhi = 28 + BR_DHI_V * max(vx - 1.0, 0.0)
                    else:
                        if fvv < 1.5:
                            fl_vjl = 20; dhi = 40 * fvv - 38
                        elif fvv < 2.0:
                            fl_vjl = 14; dhi = 31 * fvv - 38
                        else:
                            fl_vjl = FASTV_HOLD; dhi = 26 * fvv - 38 + FASTV_ADD
                        if lifts and LIFT_VJ_HOLD > 0:
                            # on a moving lift a long vertical jump leaves the lift behind (fell into the shaft): short hop at close range
                            fl_vjl = LIFT_VJ_HOLD
                            dhi = 26 * fvv - 38
                        dhi = max(4 * fvv + 7, dhi) + (8 if vx > 1.0 else 0)
                    pod_col = False
                    if POD_GATE and pos_now >= 2040 and d > 4:
                        for pb_ in pods:
                            pc_ = (pb_[0] + pb_[2]) // 2
                            if abs(pc_ - cx) <= POD_GATE_W and pb_[3] >= 125 and not (
                                    bow_bb is not None and bow_bb[0] - 34 <= pc_ <= bow_bb[1] + 34 and pb_[1] <= DG_PH):
                                pod_col = True
                    if -30 < d <= dhi and not rim and not pod_col:
                        fl_jump = True
                    elif d <= FLAME_LOOK and not (vx > 0.8 and dist_gap <= 5 + 7 * vx and pit_dist < 60) \
                            and not (BR_FAST and fl_fwd and pos_now >= 2020 and fy < 190):
                        if vx > 0.6 and not lifts:
                            fl_stand = {"Left"}
                        else:
                            fl_stand = set()
                if UP_BLOCK and fl_jump and low is not None and low[0] > 6 and not fl_fwd and fy >= 190:
                    # a second flame at head height of the jump: jumping now flies into it -> let it pass first
                    for b in flames:
                        if b is low[1]:
                            continue
                        if fy - 90 <= b[3] < fy - 19 and b[2] > cx - 14 and b[0] - (cx + 6) <= 40:
                            fl_jump = False
                            fl_stand = set()
                            break
                if LOWHOP and fl_jump and (not fl_fwd) and low is not None and fy >= 190:
                    # floor flame to hop over while a lane-144 flame is inbound: a high hop flies into it -> minimal hop
                    for b in flames:
                        if b is low[1]:
                            continue
                        if fy - 90 <= b[3] < fy - 19 and b[2] > cx - 14 and b[0] - (cx + 6) <= LOWHOP_D:
                            fl_vjl = LOWHOP_H
                            break
                if POD_NOJUMP and fl_jump and pos_now >= 2040 and fy < 190 and not lifts:
                    # bridge: never launch a flame-dodge jump into a podoboo column (pod rises under/over the jump)
                    for pb in pods:
                        pcx_ = (pb[0] + pb[2]) // 2
                        if bow_bb is not None and bow_bb[0] - 34 <= pcx_ <= bow_bb[1] + 34:
                            continue
                        if pb[3] < fy - 100:
                            continue
                        if -POD_NJ_B <= pcx_ - cx <= POD_NJ_A:
                            fl_jump = False
                            fl_stand = set()
                            break
                ledge_drop = False
                if LEDGE_DROP and low is not None and fbar_drop and not fbar_zone and 140 < fy < 190 and 50 <= low[0] < 150 and ang2 is not None:
                    om_ = bar_omega(fbar_hist, -3.0)
                    lane_ = float(208 - 8 - b2["py"])
                    ok_drop = bar_crossing_safe(max(dxp2, 10), ang2, om_, lane_y=lane_, yc0=float(fy - 8 - b2["py"]),
                                                x_edge=-float(dxp2) + float(min(dist_gap, 70)), stop_land=True, v0=max(vx, 0.4))
                    if ok_drop:
                        fl_jump = False
                        fl_stand = None
                        ledge_drop = True
                fl_danger = False
                if gap_ahead or wall or near_gap or (pmode and pit_dist < 120):
                    for b in flames:
                        if cx - 20 <= (b[0] + b[2]) // 2 <= cx + FLAME_X and b[3] >= fy - 95 and b[1] <= fy + 4:
                            fl_danger = True
                if not fl_danger and flw > 0 and pmode and flw < 300:
                    flw = max(0, flw - 2)
                # last pillar -> bridge: never launch the big jump with a flame in the flight corridor
                last_wait = None
                if LAST_FLAME_WAIT and (not desp) and LAST_X0 <= pos_now < 2200 and fy < 190 and (gap_ahead or wall) and lfw < LAST_CAP and not commit_flag:
                    for b in flames:
                        if b[2] > cx - 6 and b[0] < cx + LAST_D and b[3] >= fy - 60 and b[1] <= fy + 6 and b[1] >= 105:
                            lfw += 1
                            if vx > 0.8 and back >= 6:
                                last_wait = {"Left"}
                            else:
                                last_wait = set()
                            break
                if NEAR_WAIT and last_wait is None and 1690 <= pos_now < 2090 and fy < 190 and (gap_ahead or wall) \
                        and not commit_flag and nw < NEAR_CAP:
                    hit_n = False
                    for b in flames:
                        if b[2] > cx - 6 and 0 <= b[0] - (cx + 6) <= NEAR_D and b[3] >= fy - 70 and b[1] <= fy + 4:
                            hit_n = True
                            break
                    if hit_n:
                        nw += 1
                        last_wait = {"Left"} if (vx > 0.8 and back >= 6) else set()
                    else:
                        nw = max(0, nw - 1)
                jg_wait = False
                jg_alt = None
                if JG and (gap_ahead or wall) and 1690 <= pos_now < 2200 and not lifts:
                    vT_ = None
                    if JG_SPAWN and spawn_f > 0 and pos_now < 2080:
                        P_ = 40.0 if not spawn_iv else max(30.0, 0.9 * min(spawn_iv[-6:]))
                        vT_ = max(0.0, spawn_f + P_ - sim.frame)
                    jg_wait, jg_alt = _jg_plan(sup, cx, vx, fy, world, flames, gap_ahead, wall, FSPEED, jg_st, pos_now, commit_flag, vT_)
                if NOWAIT_X > 0 and pos_now >= NOWAIT_X and fy < 190 and not lifts:
                    # last pillar / bridge: waiting at the lip is lethal (podoboo column), only go
                    last_wait = None
                    jg_wait = False
                    danger = False
                    fl_danger = False
                why = "-"
                if DEBUG:
                    why = "".join(c for c, f in (("J", fl_jump), ("S", fl_stand is not None), ("B", bow_keys is not None), ("W", bow_wait is not None),
                                                  ("R", br_keys is not None), ("F", fbar_keys is not None), ("E", ledge_closed), ("L", last_wait is not None),
                                                  ("D", fl_danger), ("T", lift_keys is not None), ("A", danger), ("P", piv_keys is not None), ("G", jg_wait),
                                                  ("C", commit_flag), ("o", bow_over)) if f) + "|bb%s" % (bow_bb,)
                if PG and PG_A <= pos_now < PG_Z and fy < 190 and not lifts and fl_jump:
                    if pg_go > 0 or pos_now >= PG_COMMIT:
                        fl_jump = False        # crossing the pod column: never be airborne at pod height
                        fl_stand = None
                    elif fl_fwd and pos_now >= PG_FWD_X:
                        fl_fwd = False         # waiting for the pod: dodge flames with a vertical jump only
                if fl_jump:
                    zprev = True
                    hold = 1
                    if fl_fwd and BJ_ON and bow_bb is not None and pos_now >= BJ_X and bow_bb[0] - (cx + 6) < BJ_D:
                        fl_fwd = False      # Bowser close ahead: dodge the flame in place, never hop forward into him
                    if fl_fwd:
                        keys = {"Right", "z"}
                        hop_mode = True
                        hop_len = HOP_HOLD
                        vjump = False
                    else:
                        keys = {"z"}
                        hop_mode = False
                        vjump = True
                        vj_lim = fl_vjl
                elif ledge_drop:
                    keys = {"Right", "x"}
                elif bow_keys is not None and br_keys is None:
                    keys = bow_keys
                elif bow_over and br_keys is None:
                    keys = {"Right", "x", "z"}
                    zprev = True
                    hold = 1
                    vjump = False
                    hop_mode = False
                    hop_len = HOP_HOLD
                    jump_lim = JUMP_HOLD
                elif fl_stand is not None:
                    keys = fl_stand
                elif br_keys is not None:
                    keys = br_keys if (bow_wait is None or "Right" in bow_wait) else bow_wait
                elif bow_wait is not None:
                    keys = bow_wait
                elif fbar_keys is not None:
                    keys = fbar_keys
                elif ledge_closed:
                    keys = ledge_keys
                elif last_wait is not None:
                    keys = last_wait
                elif fl_danger and not lift_scene and not fbar_zone and not commit_flag and flw < 300 and not desp:
                    flw += 1
                    if dist_gap >= 30:
                        keys = {"Right", "x"}
                    elif vx > 0.8 and back >= 6:
                        keys = {"Left"}
                    elif pmode and dist_gap < 12 and back >= 14:
                        keys = {"Left"}
                    else:
                        keys = set()
                elif lift_keys is not None:
                    keys = lift_keys
                    if lift_hop:
                        zprev = True
                        hold = 1
                        vjump = False
                        hop_mode = True
                        hop_len = HOP_HOLD if (vx >= 0.8 or pos_now >= 1420) else LIFT_STAND_HOLD
                        if LIFT_GEO and vx < 0.8 and pos_now < 1420 and cand:
                            # standing hop onto a lift: pick the hold whose travel lands on the lift centre (too long overshoots, too short misses)
                            tgt_ = (x0n + 12) - cx
                            hop_len = min((4, 8, 12, 16), key=lambda h_: abs(dx_est(h_, max(vx, 0.0)) - tgt_))
                elif danger and not desp:
                    stop_d = wait_d + 6 * max(vx, 0.0)
                    if dist_gap > stop_d + 8:
                        keys = {"Right"}
                    elif (dist_gap <= wait_d - 2 or (vx > 1.2 and dist_gap < stop_d)) and back >= 8:
                        keys = {"Left"}
                    elif vx > 0.8 and dist_gap < 8 + 7 * vx:
                        keys = {"Left"}
                    else:
                        keys = set()
                elif piv_keys is not None and not gap_ahead:
                    keys = piv_keys
                elif (gap_ahead or wall) and bow_present and ((BR_X0 - 10 <= pos_now < BR_END) or bow_ov) and still_n < 8 and not axe_w:
                    keys = {"Right", "x"}
                elif LIFT_HIGH and (gap_ahead or wall) and lifts and 1370 <= pos_now <= 1420 and fy > LIFT_HIGH_FY and lhw < 500:
                    lhw += 1
                    keys = set()        # riding lift A: a generic jump from a low lift falls short of the far ledge; wait until the lift is high
                elif jg_wait and not commit_flag:
                    keys = {"Left"} if (vx > 0.8 and back >= 6) else set()
                elif (gap_ahead or wall) and not (fbar_drop and bool(deep[min(255, cx + min(dist_gap, 60) + 6)]) and not wall):
                    keys = {"Right", "x", "z"}
                    zprev = True
                    hold = 1
                    vjump = False
                    hop_mode = False
                    hop_len = HOP_HOLD
                    jump_lim = JUMP_HOLD
                    if gap_ahead and not wall and pos_now >= 1690 and fy < 190:
                        _LASTP["p"] = 99
                        ch = choose_hold(sup, cx, vx)
                        if ch is not None:
                            jump_lim = max(ch, 12)
                            if PMODE and NOX and 1700 <= pos_now < 2090 and vx < 1.8:
                                jump_lim = max(ch, MIN_HOLD)
                        if PILLAR_HOP and pos_now < (2090 if PMODE else 1900) and (_LASTP.get("p", 99) <= 24 or (PMODE and _LASTP.get("w", 99) <= 20 and _LASTP.get("need", 99) + 6 <= HOP_REACH)):
                            keys = {"Right", "z"}
                            hop_mode = True
                            if _LASTP.get("w", 99) <= 3:
                                hop_len = HOP_LONG
                    if jg_alt is not None and gap_ahead and not wall:
                        jump_lim = jg_alt
                        hop_mode = False
                        keys = {"Right", "x", "z"}
                    if wall and not gap_ahead:
                        col = g[:, cx + 8:cx + 13].any(1)
                        wh = 0
                        yy = fy - 1
                        gapc = 0
                        while yy > 40 and wh < 110:
                            if col[yy]:
                                gapc = 0
                            else:
                                gapc += 1
                                if gapc >= 4:
                                    break
                            wh += 1
                            yy -= 1
                        wh = max(1, wh - gapc)
                        need = wh + 16
                        jump_lim = 26
                        for kk, ap in ((2, 34), (4, 42), (6, 49), (8, 56), (11, 64), (14, 71), (18, 79), (24, 85)):
                            if ap >= need:
                                jump_lim = kk + 2
                                break
                        wk = (world + cx) // 24
                        tries = wall_tries.get(wk, 0)
                        wall_tries[wk] = tries + 1
                        jump_lim = min(26, jump_lim + 6 * tries)
        else:
            lim = vj_lim if vjump else (hop_len if hop_mode else jump_lim)
            if zprev and hold < lim and vy <= 0:
                keys = {"z"} if vjump else ({"Right", "z"} if hop_mode else {"Right", "x", "z"})
                if vjump and (pos_now >= 1690 or (lifts and VJ_LEFT_LIFT)) and vx > 0.6:
                    keys = {"z", "Left"}
                hold += 1
            else:
                zprev = False
                if vjump and 1330 <= pos_now <= 1500 and len(vyh) == 4 and all(0 <= v <= 1 for v in vyh) and sum(vyh) >= 3:
                    vjump = False       # landed back on a descending lift: leave the stale vertical-jump state (it held Mario on the lift to the bottom)
                    keys = {"Right"}
                elif vjump:
                    keys = {"Left"} if vx > 0.4 else set()
                elif hop_mode:
                    keys = {"Right"}
            if vy > 0 and not vjump:
                lp = predict_landing(g, cx, fy, vx + (0.4 if pos_now >= 1690 and vx > 0.5 else 0.0), vy)
                if lp is None and vx > 0.8 and pos_now >= 1690:
                    lp0 = predict_landing(g, cx, fy, 0.0, vy)
                    if lp0 is not None:
                        keys = {"Left"}
                if lp is not None:
                    lx, ly, rext = lp
                    ly = min(ly, 238)
                    margin = rext - lx
                    abn = HOP_ABN if hop_mode else AIR_BRAKE_NEUTRAL
                    abl = HOP_ABL if hop_mode else AIR_BRAKE_LEFT
                    if pos_now >= 1690 and not hop_mode:
                        abn = 34
                        abl = 18
                    if rext < 245 and margin < abn and not g[ly:ly + 17, rext + 2:rext + 8].any():
                        keys = {"x"}
                        if margin < abl:
                            keys = {"x", "Left"}
                if PMODE and vy > 0 and not vjump and 1700 <= pos_now < 2090 and not lifts and not hop_mode and lp is not None \
                        and fy < 190:
                    lx_, ly_, rext_ = lp
                    ly_ = min(ly_, 238)
                    colm = g[ly_:ly_ + 10].any(0)
                    lft_ = lx_
                    while lft_ > 2 and colm[lft_ - 1]:
                        lft_ -= 1
                    rext_ = lx_
                    while rext_ < 253 and colm[rext_ + 1]:
                        rext_ += 1
                    pm_thr = PM_MARGIN + PM_VK * max(vx, 0.0)
                    if (rext_ - lx_) < pm_thr and (lx_ - lft_) >= PM_LEFT and vx > 0.3:
                        keys = {"Left"}
                    elif (rext_ - lx_) >= pm_thr:
                        keys = {"x"}
        if axe_ov and on_ground:
            keys = {"Left"}
            if "z" in keys or zprev:
                pass
            zprev = False       # cancel a jump launched this frame: Bowser is overhead, do not jump into him
            hold = 0
            hop_mode = False
            vjump = False
            axe_back_n += 1
        if LEDGE_DROP2 and ldrop_on and (not on_ground) and (not zprev) and fy < 205 and not lifts and pos_now < 1700:
            keys = {"Left"} if vx > 0.3 else set()      # falling from the ledge toward the floor bar: do not drift into its reach
        if LEDGE_AIR_RUN and (not on_ground) and (not zprev) and not lifts and 1540 <= pos_now <= 1680 and gfy < 190 \
                and fbar_commit and "z" not in keys:
            keys = {"Right", "x"}      # drop from the ledge to the floor bar: the safety simulation assumes full running speed
        if on_ground:
            gfy = fy
        elif FLOOR_HOP and gfy >= 190 and 1690 <= pos_now < 2090 and fy < 176 and (vy >= 0 or FH_ASC) and not lifts and not vjump \
                and vx > 0.3 and not zprev:
            keys = {"Left"} if vx > FH_VX else set()      # floor -> first pillar wall hop: do not carry run speed onto the 16 px pillar
        if PMODE and NOX and 1700 <= pos_now < 2090 and fy < 190 and not lifts and "x" in keys and \
                (not on_ground or zprev or "z" in keys):
            keys = set(keys)
            keys.discard("x")
        elif PMODE and on_ground and not zprev and 1700 <= pos_now < BR_RUN_X and fy < 190 and not lifts and "z" not in keys:
            keys = set(keys)
            keys.discard("x")
        if GOV and on_ground and (not zprev) and 1700 <= pos_now < 2090 and fy < 190 and not lifts and "z" not in keys \
                and dist_gap < 99 and back >= 6 and not bow_commit and \
                ((vx > GOV_V and dist_gap < GOV_A + GOV_B * vx) or (vx > GOV_V2 and dist_gap < GOV_C + GOV_D * vx * vx)):
            keys = {"Left"}      # arriving fast at a pillar edge: brake early instead of being forced into a jump later
        if RIM_FIX and on_ground and (not zprev) and 1690 <= pos_now < 2090 and fy < 190 and not lifts and \
                "z" not in keys and vx < 1.2 and dist_gap < rim_d and back >= 6 and not commit_flag:
            keys = {"Left"}      # never wait with the hitbox hanging over a lava pit (podoboos rise there)
        if BR_SLOW and on_ground and (not zprev) and pos_now >= BR_SLOW_X and fy < 190 and not lifts and not bow_commit \
                and "z" not in keys and "Right" in keys:
            # bridge: pods are only visible ~20 px ahead; keep walking speed low enough to be able to stop in time
            keys = {"Right"} if vx < BR_VMAX else set()
        if BR_AVOID and on_ground and (not zprev) and pos_now >= 2040 and fy < 190 and not lifts and not bow_commit \
                and "z" not in keys:
            # Bridge: podoboos leap through the bridge line; never walk into / stand in a visible pod's column
            for b in pods:
                pcx = (b[0] + b[2]) // 2
                if bow_bb is not None and bow_bb[0] - DG_BM <= pcx <= bow_bb[1] + 34 and b[1] <= DG_PH:
                    continue        # orange mane of Bowser: phantom
                if b[3] < fy - 100:
                    continue
                r = pcx - cx
                if br_blk > 2000:
                    break
                if 0 <= r <= BR_AVOID_IN:
                    keys = {"Left"} if back >= 6 else set()
                    br_blk += 1
                    break
                if BR_AVOID_IN < r <= BR_AVOID_AHEAD and "Right" in keys:
                    keys = {"Left"} if (vx > 1.0 and back >= 6) else set()
                    br_blk += 1
                    break
        if BR_PRED and on_ground and (not zprev) and pos_now >= BP_X0 and fy < 190 and not lifts and not bow_commit \
                and "z" not in keys and bp_blk < BP_CAP:
            for c in pod_cols:
                sxp = c[0] - world
                if bow_bb is not None and bow_bb[0] - 34 <= sxp <= bow_bb[1] + 34:
                    continue
                r = sxp - cx
                if r < -BP_BACK or r > BP_AHEAD:
                    continue
                hid = sim.frame - c[1]
                per = c[4]
                if c[3] is None and not c[2]:
                    continue
                unsafe = bool(c[2]) or hid < BP_MIN or (per - BP_PRE < hid <= 150)
                if hid > 150 and not c[2]:
                    unsafe = False
                if not unsafe:
                    continue
                if abs(r) <= BP_COL:
                    if r < 4 and back >= 6:
                        keys = {"Left"}
                    else:
                        keys = {"Right", "x"}
                    bp_blk += 1
                    break
                if r > BP_COL and "Right" in keys and r <= BP_AHEAD:
                    keys = {"Left"} if (vx > 1.0 and back >= 6) else set()
                    bp_blk += 1
                    break
        if BR_DG and on_ground and (not zprev) and pos_now >= BP_X0 and fy < 190 and not lifts \
                and "z" not in keys and dg_n < DG_CAP:
            # Bridge podoboos spawn at RANDOM columns and shoot up through the bridge line: step out of any visible pod's column
            for b in pods:
                pcx = (b[0] + b[2]) // 2
                if bow_bb is not None and bow_bb[0] - DG_BM <= pcx <= bow_bb[1] + 34 and b[1] <= DG_PH:
                    continue
                if b[3] < fy - 100:
                    continue
                dxp_ = pcx - cx
                if abs(dxp_) <= DG_W:
                    bow_close = bow_bb is not None and bow_bb[0] - cx < 60
                    if dxp_ < -2 or (abs(dxp_) <= 2 and not bow_close) or back < 6:
                        keys = {"Right", "x"}
                    else:
                        keys = {"Left"}
                    dg_n += 1
                    break
        if PG and on_ground and (not zprev) and fy < 190 and not lifts and PG_A <= pos_now < PG_Z and "z" not in keys \
                and not bow_commit:
            go_ = pg_go > 0 or pg_wait > PG_CAP or pos_now >= PG_COMMIT
            if go_:
                if "Right" not in keys and pos_now < PG_FORCE_END and (bow_bb is None or bow_bb[0] - cx > PG_BOW_MIN):
                    keys = {"Right", "x"}
            elif pos_now < PG_HOLD_X:
                pg_wait += 1
                dist_ = PG_HOLD_X - pos_now
                if dist_ > 9.0 * max(vx, 0.0) + 6.0:
                    keys = {"Right"}
                elif dist_ < 3.0 * max(vx, 0.0) + 1:
                    keys = {"Left"} if back >= 6 else set()
                else:
                    keys = set()
            else:
                pg_wait += 1
                keys = {"Left"} if (back >= 6 and (vx > 0.3 or pos_now >= PG_HOLD_X + 4)) else set()
        if PG and PG_AIR and (not on_ground) and fy < 190 and not lifts and PG_A <= pos_now < PG_COMMIT and pg_go <= 0 \
                and "z" not in keys and vy >= -1:
            if pos_now >= PG_AIR_L:
                keys = {"Left"}
            elif pos_now >= PG_AIR_N:
                keys = set()
        if NEG_FIX and (not on_ground) and (not lifts) and pos_now >= 1500 and vx < NEG_V and "Right" not in keys:
            keys = set(keys)
            keys.discard("Left")
            keys.add("Right")
        if AXE_SPRINT and on_ground and (not zprev) and pos_now >= AXE_SPRINT_X and fy <= AXE_SPRINT_FY and not lifts:
            keys = {"Right", "x"}      # standing on the axe pillar: the axe is a few px ahead, never wait/back off here
        ride_start = (len(vyh) == 4 and all(abs(v) <= 1 for v in vyh) and sum(abs(v) for v in vyh) >= 3 and
                      not (min(vyh) < 0 < max(vyh)))
        if on_ground or zprev or abs(vy) > 1:
            ride_on = False
        elif ride_start:
            ride_on = True
        if RIDE_HOLD and ride_on and (not on_ground) and (not zprev) and lifts and 1330 <= pos_now <= 1480 and \
                "Right" in keys and "z" not in keys:
            if RIDE_JUMP and fy >= RIDE_FY and sum(vyh) >= 3 and min(vyh) >= 0:
                keys = {"Right", "x", "z"}
                zprev = True
                hold = 1
                vjump = False
                hop_mode = False
                jump_lim = RIDE_LIM
                ride_on = False
            else:
                desc = sum(vyh) >= 3 and min(vyh) >= 0
                if desc and fy >= RIDE_WALK_FY:
                    pass
                else:
                    keys = {"Left"} if (desc and vx > 0.8) else set()
        pos = world + cx
        if prev_pos is not None and pos == prev_pos and on_ground:
            still_n += 1
        else:
            still_n = 0
        posh.append(pos)
        if len(posh) > VX_N + 1:
            del posh[0]
        if prev_pos is not None:
            vx = (posh[-1] - posh[0]) / float(len(posh) - 1)
        else:
            vx = 0.0
        prev_pos = pos
        if DEBUG:
            RING.append(_DB.get("s", "") + "lp%s/%s/%s jl%d/%d dg%d jg%d %d p%d cx%d fy%d g%d vx%.1f %s %s zp%d vj%d hm%d fl%s pods%s" % (
                _LASTP.get("need"), _LASTP.get("w"), _LASTP.get("p"), jump_lim, hop_len, dist_gap,
                jg_st["n"], sim.frame, pos, cx, fy, int(on_ground), vx, "".join(sorted(k[0] for k in keys)), why, int(zprev), int(vjump), int(hop_mode),
                [(int(b[0] - cx), int(b[1]), int(b[3])) for b in flames][:3], [(int((b[0] + b[2]) // 2 - cx), int(b[1])) for b in pods][:3]))
            if len(RING) > 400:
                del RING[:200]
        sim.step(keys)

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    try:\n        f = np.asarray(frame)\n        brown = (f[:, :, 0] == 172) & (f[:, :, 1] == 124) & (f[:, :, 2] == 0)\n        brown[:32] = False\n        ys = np.nonzero(brown)[0]\n        status = "running"\n        threats = []\n        if len(ys) < 6:\n            status = "dead"\n        elif int(ys.max()) > 225:\n            status = "danger"\n            threats.append("falling")\n        return {"status": status, "where": {"hazard": threats[0] if threats else None, "confidence": 0.5},\n                "progress": None, "resources": {}, "threats": threats, "stuck": {"is": False},\n                "outlook": {"label": "bad" if status != "running" else "ok", "p_through": 0.1}, "note": ""}\n    except Exception:\n        return None\n', 'memory.json': '{"human": {}, "past": {}}\n'}

