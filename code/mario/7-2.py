# ==================================================================================================
# Super Mario Bros. (NES) -- level 7-2  |  超级马里奥兄弟（NES） 关 7-2
# Pass rate on 100 random variants never used before: 23/100; fastest finish 2174 frames (36.23 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 23/100；最快一局 2174 帧（60 帧/秒，约 36.23 秒）。
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
#   PB underwater swimmer (terrain-chunk reachability planner + tracked blobs + depth hysteresis),
#    bit-identical decisions, ~40% cheaper: one fused pixel scan (terrain+enemies+scroll profile),
#    dead branch
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _nch_consts                    line 90    
#   def find_mario                     line 102   
#   def scan                           line 123   one pass over the water view: scroll profile, terrain chunk map, enemy pixels.
#   def blobs_from                     line 157   
#   def track                          line 204   
#   def reach                          line 229   
#   def reach_s                        line 236   
#   def scroll_est                     line 242   
#   def play                           line 252   
#   def solve                          line 465   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: PB underwater swimmer (terrain-chunk reachability planner + tracked blobs + depth hysteresis), bit-identical decisions, ~40% cheaper: one fused pixel scan (terrain+enemies+scroll profile), dead branches/logging removed, scalar reach, min/max instead of clip.
import numpy as np

ZONE_D0 = 860
TE_MAX = 18.0
TE_BIG = 50.0
BIG_EXTRA = 4.0
PREF_WALL = 160
PREF_D0 = 850
PREF_D1 = 1100
PREF_D2 = 1350
PREF_LANE = 103
LATE_D = 700
W_FAR_LATE = 600.0
DEC_POW = 1.0
HYST_EN = 12.0
HYST_FREE = 0.0

VMAR = 1.7
UPV = 1.4
SQ_MARGIN = 10
STROKE_ERR = 8
DEBOUNCE = 1
_CELL = 4
WALL_D0 = 850
WALL_D1 = 1150
PW_DIVE = 1.5
END_PLAN = True
END_MAX = 90
END_PY = 118
HZ0 = 900
HZ1 = 1070
HYST_WALL = 4.0
ABOVE_TE = 40.0
PW_D1 = 1100
WALL_ERR = 1
STILL_END = 150
PREF = 125
EN_PEN = 1500.0
LOOKAHEAD = 140
_CH = 6
END_X = 150
SQ_RISE = 0.3
SQ_RMAX = 40
CH_MARGIN = 14
T_DECAY = 25.0
G = 0.08
VMAXD = 2.2
V_STROKE = -1.5
NEAR_CH = 4
BEHIND = 8
NEAR_STILL = 40
CORE_M = 4
SQ_DOWN = 1.0
HARD_CH = 14
HARD_W = 1500.0
FAR_R = 70.0
OUTER_W = 0.5
HYST = 4.0
HYST_QUIET = 0.3
U_D0 = 2000
CEIL_ERR = 2
STROKE_U = 4
SWEEP = 4

# knobs of the wall / Blooper decisions (start values = PB); the system tunes them on training variants
TUNABLE = {"PREF_WALL": (120, 190), "PW_DIVE": (0.5, 3.0), "SQ_MARGIN": (5, 15), "SQ_DOWN": (0.2, 1.5),
           "SQ_RISE": (0.0, 1.0), "HYST_WALL": (1.0, 10.0), "CH_MARGIN": (9, 20), "T_DECAY": (12.0, 50.0),
           "STROKE_ERR": (4, 14), "WALL_ERR": (0, 5)}
_D_INIT = None

CANDS = np.arange(40, 193)
CANDS_U = np.arange(31, 193)
CANDS15 = CANDS + 15
CANDS_U15 = CANDS_U + 15

# per-channel colour-class bit tables: a pixel is class k iff bit k is set in all three tables
_LR = np.zeros(256, np.uint8)
_LG = np.zeros(256, np.uint8)
_LB = np.zeros(256, np.uint8)
for _bit, (_r, _g, _b) in ((1, (0, 168, 0)), (2, (184, 248, 184)), (4, (248, 88, 152)), (8, (228, 0, 88)),
                           (16, (252, 252, 252)), (32, (188, 188, 188)), (64, (248, 56, 0))):
    _LR[_r] |= _bit
    _LG[_g] |= _bit
    _LB[_b] |= _bit

_NCACHE = {}


def _nch_consts(nch):
    c = _NCACHE.get(nch)
    if c is None:
        tk = np.maximum(np.arange(nch) * _CH / VMAR, 4.0)
        NEAR = min(NEAR_CH, nch)
        wk = (T_DECAY / (T_DECAY + tk))[NEAR:]
        mid = min(HARD_CH, nch)
        c = (tk, np.arange(nch)[None, :], NEAR, wk[None, :], mid)
        _NCACHE[nch] = c
    return c


def find_mario(img, prev):
    if prev is not None:
        px, py = prev
        x0, x1 = max(0, int(px) - 30), int(px) + 30
        y0, y1 = max(36, max(0, int(py) - 30)), min(216, int(py) + 30)
    else:
        x0, x1, y0, y1 = 0, img.shape[1], 36, 216
    if y1 <= y0 or x1 <= x0:
        return None
    c = img[y0:y1, x0:x1]
    ys, xs = np.nonzero(c[:, :, 1] == 124)
    if len(ys) < 6:
        return None
    sel = (c[ys, xs, 0] == 172) & (c[ys, xs, 2] == 0)
    ys = ys[sel]
    xs = xs[sel]
    if len(ys) < 6:
        return None
    return float(xs.mean() + x0), float(ys.mean() + y0)


def scan(img, mx, my, xs0, nch):
    """one pass over the water view: scroll profile, terrain chunk map, enemy pixels."""
    slab = img[40:210]
    m = slab[:, :, 2] != 248
    prof = np.add.reduce(m[:168].view(np.uint8), axis=0, dtype=np.int16)
    x0c = max(0, int(mx) - 40)
    idx = np.flatnonzero(m)
    xs = idx & 255
    kx = xs >= x0c
    idx = idx[kx]
    xs = xs[kx]
    ys = idx >> 8
    px = np.take(slab.reshape(-1, 3), idx, axis=0)
    f = _LR[px[:, 0]] & _LG[px[:, 1]] & _LB[px[:, 2]]
    w = nch * _CH
    ti = np.flatnonzero(f & 15)
    txs = xs[ti]
    tys = ys[ti]
    t = (tys < 167) & (txs >= xs0) & (txs < xs0 + w)
    blk = np.zeros((nch, 240), bool)
    blk[(txs[t] - xs0) // _CH, tys[t] + 40] = True
    # enemy pixels (rows 44..209)
    ei = np.flatnonzero(f & 112)
    exs = xs[ei]
    eyy = ys[ei] + 40
    ef = f[ei]
    x0, x1 = max(0, int(mx) - 9), int(mx) + 9
    y0, y1 = max(0, int(my) - 11), int(my) + 11
    isr = ((ef & 64) != 0) & ~((eyy >= y0) & (eyy < y1) & (exs >= x0) & (exs < x1))
    oth = ((ef & 32) != 0) | isr
    keep = (eyy >= 44) & (((ef & 16) != 0) | oth)
    return prof, blk, eyy[keep], exs[keep], oth[keep]


def blobs_from(ys, xs, oth):
    out = []
    n = len(ys)
    if n < 12:
        return out
    ck = (ys // _CELL) * 512 + (xs // _CELL)
    uk, inv = np.unique(ck, return_inverse=True)
    ukl = uk.tolist()
    nc = len(ukl)
    idx = {k: i for i, k in enumerate(ukl)}
    parent = list(range(nc))
    get = idx.get
    for i in range(nc):
        c = ukl[i]
        for d in (1, 511, 512, 513):
            j = get(c + d)
            if j is not None:
                a = i
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                b = j
                while parent[b] != b:
                    parent[b] = parent[parent[b]]
                    b = parent[b]
                if a != b:
                    parent[b] = a
    roots = []
    for i in range(nc):
        a = i
        while parent[a] != a:
            a = parent[a]
        roots.append(a)
    pl = np.array(roots)[inv]
    cnts = np.bincount(pl, minlength=nc)
    first = np.full(nc, n, np.int64)
    first[pl[::-1]] = np.arange(n - 1, -1, -1)
    cand = np.flatnonzero(cnts >= 12)
    for c in cand[np.argsort(first[cand], kind="stable")].tolist():
        cnt = int(cnts[c])
        if cnt >= 12:
            sel = pl == c
            yy, xx = ys[sel], xs[sel]
            out.append([int(xx.min()), int(xx.max()), int(yy.min()), int(yy.max()), cnt, int(oth[sel].sum()) <= 1])
    return out


def track(blobs, prev_blobs):
    res = []
    used = set()
    for b in blobs:
        cx, cy = (b[0] + b[1]) / 2.0, (b[2] + b[3]) / 2.0
        best, bd = None, 14.0
        for i, p in enumerate(prev_blobs):
            if i in used:
                continue
            d = abs(p["cx"] - cx) + abs(p["cy"] - cy)
            if d < bd:
                best, bd = i, d
        sq = bool(b[5]) or (b[3] - b[2]) >= 18
        if best is not None:
            used.add(best)
            p = prev_blobs[best]
            vx = 0.5 * p["vx"] + 0.5 * (cx - p["cx"])
            vy = 0.5 * p["vy"] + 0.5 * (cy - p["cy"])
            sq = sq or p["sq"]
        else:
            vx, vy = -VMAR, 0.0
        res.append({"b": b, "cx": cx, "cy": cy, "vx": vx, "vy": vy, "sq": sq})
    return res


def reach(top, v0, t):
    up = np.maximum(UPV * np.maximum(t - 2.0, 0.0), max(0.0, -v0) * t)
    down = np.minimum(v0 * t + 0.5 * G * t * t, VMAXD * t)
    down = np.maximum(down, 0.0)
    return top - up, top + down


def reach_s(top, v0, t):
    up = max(UPV * max(t - 2.0, 0.0), max(0.0, -v0) * t)
    down = max(min(v0 * t + 0.5 * G * t * t, VMAXD * t), 0.0)
    return top - up, top + down


def scroll_est(p_old, p_new):
    best, bs = None, 0
    pn = p_new[20:200]
    for s in range(0, 4):
        d = np.abs(pn - p_old[20 + s:200 + s]).sum()
        if best is None or d < best:
            best, bs = d, s
    return bs


def play(sim):
    prof_old = None
    D = 0.0
    try:
        f0 = int(sim.frame)
    except Exception:
        f0 = 0
    if f0 > 60:  # mid-level start (practice checkpoint): estimate the scroll
        D = max(0.0, 1.6 * f0 - 80.0)
    if _D_INIT is not None:
        D = float(_D_INIT)
    prev = None
    last_img = None
    same = 0
    lastz = False
    f = 0
    v0 = 0.0
    thist = []
    prev_blobs = []
    end_cnt = 0
    hist = []
    since = 99
    prev_tgt = None
    cond_prev = False
    endm_n = 0
    npmin, npmax = np.minimum, np.maximum
    while sim.budget_left > 0 and f < 8000:
        img = sim.screen()
        if last_img is not None and (img[::16] == last_img[::16]).all() and np.array_equal(img, last_img):
            same += 1
            if same >= STILL_END:
                return
        else:
            same = 0
        last_img = img
        mar = find_mario(img, prev)
        if mar is None and prev is None:
            sim.step({"Right"})
            f += 1
            continue
        if mar is not None:
            prev = mar
        mx, my = prev
        top = my - 7.0
        front = mx + 8.0
        back = mx - 8.0
        xs0 = int(front)
        nch = max(1, min(LOOKAHEAD, 255 - xs0) // _CH)
        pr, blk, eys, exs, eoth = scan(img, mx, my, xs0, nch)
        if prof_old is not None:
            D += scroll_est(prof_old, pr)
        prof_old = pr
        blobs = blobs_from(eys, exs, eoth)

        # level end: camera stopped, Mario walks right up the stairs to the sideways pipe
        end_cnt = end_cnt + 1 if mx > END_X else 0
        endm = False
        if end_cnt >= 4:
            tr_e = track(blobs, prev_blobs)
            near_sq = any(e["sq"] and (mx - 30) < e["b"][0] < (mx + 70) for e in tr_e)
            if near_sq and END_PLAN and endm_n < END_MAX:
                endm = True
                endm_n += 1
            else:
                prev_blobs = tr_e
        if end_cnt >= 4 and not endm:
            hist.append((mx, my))
            blocked = (len(hist) >= 14 and abs(hist[-1][0] - hist[-14][0]) < 1.5
                       and abs(hist[-1][1] - hist[-14][1]) < 2.5)
            since += 1
            keys = {"Right", "x"}
            if blocked and since >= 14:
                xa, xb = int(mx) + 9, int(mx) + 17
                cr = img[max(36, int(my) - 36):max(37, int(my) - 12), xa:xb]
                gsum = ((cr[:, :, 0] < 60) & (cr[:, :, 1] > 140) & (cr[:, :, 2] < 60)).sum()
                if gsum < 4:
                    keys.add("z")
                    since = 0
            sim.step(keys)
            f += 1
            continue

        useU = D > U_D0
        cands = CANDS_U if useU else CANDS
        cands15 = CANDS_U15 if useU else CANDS15
        thist.append(top)
        if len(thist) > 5:
            thist.pop(0)
            vm = (thist[-1] - thist[0]) / 4.0
            v0 = 0.75 * v0 + 0.25 * vm

        cs = np.zeros((nch, 241), np.int16)
        np.cumsum(blk, axis=1, dtype=np.int16, out=cs[:, 1:])
        tk, kk, NEAR, wk, mid = _nch_consts(nch)
        lo, hi = reach(top, v0, tk)
        pos = npmin(npmax(cands[:, None], lo[None, :]), hi[None, :])
        pos = npmin(npmax(np.rint(pos).astype(int), 31), 192)
        btab = (cs[:, 46:208] - cs[:, 31:193]) > 0
        bl = btab[kk, pos - 31]
        bl_far = bl[:, NEAR:]
        x = bl_far * wk
        cost = x.sum(axis=1) * 25.0 + x.max(axis=1, initial=0.0) * 300.0
        if mid > NEAR:
            cost += bl[:, NEAR:mid].any(axis=1) * HARD_W
        near_rows = blk[:NEAR].any(axis=0)
        ncs = np.concatenate([[0], np.cumsum(near_rows)])
        cost += ((ncs[cands15] - ncs[cands]) > 0) * 1000.0

        n_en = 0
        late = D > LATE_D
        tr = track(blobs, prev_blobs)
        prev_blobs = tr
        for e in tr:
            bx0, bx1, by0, by1, n = e["b"][:5]
            xr0 = bx0 - front
            xr1 = bx1 - back
            tall = e["sq"]
            if xr1 < (-BEHIND if tall else -4):
                continue
            c = -e["vx"]
            if xr0 <= 0:
                t_out = xr1 / c if c > 0.3 else 30.0
                te = max(min(t_out, 30.0), 4.0)
            else:
                if c < 0.3:
                    if xr0 > NEAR_STILL:
                        continue
                    te = 20.0
                else:
                    te = max(xr0 / c, 4.0)
            hh = by1 - by0 + 1
            lim = TE_MAX if (tall and hh <= 26) else TE_BIG
            if late:
                lim = 80.0
            if te > lim:
                continue
            n_en += 1
            if tall and hh > 26:
                lo_b = by0 - CH_MARGIN - 15 - BIG_EXTRA
                hi_b = by1 + CH_MARGIN + BIG_EXTRA
                sh = max(0.0, CH_MARGIN - CORE_M)
            elif tall:
                lo_b = by0 - SQ_MARGIN - 15 - min(SQ_RMAX, SQ_RISE * te)
                hi_b = by1 + SQ_MARGIN + SQ_DOWN * te
                sh = max(0.0, SQ_MARGIN - CORE_M)
            else:
                lo_b = by0 - CH_MARGIN - 15
                hi_b = by1 + CH_MARGIN
                sh = max(0.0, CH_MARGIN - CORE_M)
            elo, ehi = reach_s(top, v0, te)
            p = npmin(npmax(cands, elo), ehi)
            outer = (p > lo_b) & (p < hi_b)
            core = (p > lo_b + sh) & (p < hi_b - sh)
            if c > 0.3 and not tall:
                t_a = max(xr0 / c, 2.0) if xr0 > 0 else 2.0
                t_l = min(max(xr1 / c, t_a), 40.0)
                if t_l - t_a > 3.0:
                    step_ = (t_l - t_a) / 3
                    ts = np.array([t_a, 1 * step_ + t_a, 2 * step_ + t_a, t_l])
                    slo, shi = reach(top, v0, ts)
                    ps = npmin(npmax(cands[:, None], slo[None, :]), shi[None, :])
                    outer = outer | ((ps > lo_b) & (ps < hi_b)).any(axis=1)
                    core = core | ((ps > lo_b + sh) & (ps < hi_b - sh)).any(axis=1)
            dec = (T_DECAY / (T_DECAY + te)) ** DEC_POW
            cost += (OUTER_W * outer + (1.0 - OUTER_W) * core) * (EN_PEN * dec)
            if late:
                ecy = (by0 + by1) / 2.0 - 8.0
                cost += W_FAR_LATE * dec * npmin(npmax(1.0 - np.abs(p - ecy) / FAR_R, 0.0), 1.0)

        cost += np.abs(cands - top) * 0.1
        pref = PREF
        if PREF_D0 < D < PREF_D1:
            pref = PREF_WALL
        elif PREF_D1 <= D < PREF_D2:
            pref = PREF_LANE
        pw = PW_DIVE if PREF_D0 < D < PW_D1 else 0.4
        if endm:
            pref, pw = END_PY, 1.0
        cost += np.abs(cands - pref) * pw
        if prev_tgt is not None:
            dd = np.minimum(np.abs(cands - prev_tgt), 60)
            hs = HYST_EN if n_en > 0 else HYST_FREE
            if late:
                hs = HYST
            if n_en > 0 and HZ0 < D < HZ1:
                hs = HYST_WALL
            cost += dd * hs
        ib = int(np.argmin(cost))
        tgt = float(cands[ib])
        prev_tgt = tgt
        keys = {"Right"}
        y_rest = top - (v0 * v0 / (2 * G) if v0 < 0 else 0.0)
        in_wall = (WALL_D0 < D < WALL_D1)
        cond = (y_rest - tgt > (WALL_ERR if in_wall else (STROKE_U if useU else STROKE_ERR)))
        if useU and tgt < 36 and (top - tgt) > CEIL_ERR:
            # ceiling hugging: the arc model thinks Mario is still rising; tap whenever he sags
            cond = True
            cond_prev = cond
        want_z = (cond and not lastz and (cond_prev or DEBOUNCE == 0 or in_wall))
        cond_prev = cond
        if want_z:
            keys.add("z")
            lastz = True
        else:
            lastz = False
        sim.step(keys)
        if "z" in keys:
            v0 = V_STROKE
        else:
            v0 = min(VMAXD, v0 + G)
        f += 1


def solve(sim):
    return []

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    w = (frame[44:210, :, 0] == 252) & (frame[44:210, :, 1] == 252) & (frame[44:210, :, 2] == 252)\n    danger = int(w.sum()) > 60\n    return {"status": "danger" if danger else "running",\n            "where": {"hazard": "blooper" if danger else None, "confidence": 0.4},\n            "progress": None, "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "bad" if danger else "ok", "p_through": 0.1 if danger else 0.2},\n            "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

