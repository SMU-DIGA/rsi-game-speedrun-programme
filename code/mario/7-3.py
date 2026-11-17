# ==================================================================================================
# Super Mario Bros. (NES) -- level 7-3  |  超级马里奥兄弟（NES） 关 7-3
# Pass rate on 100 random variants never used before: 36/100; fastest finish 1239 frames (20.65 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 36/100；最快一局 1239 帧（60 帧/秒，约 20.65 秒）。
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
#   closed-loop runner (Right+B held) with a vectorised look-ahead planner: Mario jump physics mod
#   el + tracked enemy motion + heightmap of floor/walls/pits; jump as late as safely possible.
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _eq                            line 89    
#   def find_mario                     line 93    
#   def threats                        line 107   
#   def _nbr                           line 148   
#   def heightmap                      line 161   
#   def _plans_ground                  line 176   
#   def simulate                       line 207   
#   class Track                        line 298   
#   def update_tracks                  line 309   
#   def predict_boxes                  line 338   
#   def commit_replan                  line 382   While a jump is held and Mario is still rising: how many more frames to hold A so the flight stays safe?
#   def measure_scroll                 line 411   2-D background shift (0..6 px/frame) from the right part of the screen; None if ambiguous.
#   def play                           line 428   
#   def _play                          line 445   
#   def solve                          line 862   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: closed-loop runner (Right+B held) with a vectorised look-ahead planner: Mario jump physics model + tracked enemy motion + heightmap of floor/walls/pits; jump as late as safely possible.
import numpy as np

BRICK = (228, 92, 16)
TAN = (240, 208, 176)
LIME = (184, 248, 24)
BROWN = (172, 124, 0)
RED = (248, 56, 0)
GREEN = (0, 168, 0)
INF = 9999.0

V0 = 5.25
GH = 0.15
GD = 0.54
HORIZ = 56
KS = (4, 6, 8, 12, 16, 20, 26)
WS = (0, 1, 2, 3, 4, 6, 8, 10, 12, 15, 18, 21, 25, 30, 35, 40)
STOP_FROZEN = 60
AIR_X = 0.94
DELAY = 2
POST_LAND = 12
TERR_POST = 11
W2S = (0, 4)
FW2S = (0, 2, 5, 9)
FK2S = (6, 10, 16)
K2S = (8, 16)
MS_LEVELS = (1.0, 0.5)
WAIT_MAX = 9
MARGIN_X = 3
MARGIN_Y = 2
AIRG = 0.927
AIRCAP = 12
USE_AIRCTL = True
PHANTOM = True
PH_DX = 95
AIRCTL_WX = 99999
SC_MARGIN = 1500
STOMP_FALLBACK = True
MAXW = 24
AIR_BS = (0, 2, 4, 6, 8, 11, 14, 18, 24, 32)
AIR_DEC = 0.1196
NOSTOMP_PIT = True
NS_D = 170
UP_WIN = 62
UP_FREF = 170
PITBAN = True
PITBAN_D = 70
ACC = 0.08
AIR_ACC = 0.0
SPMAX = 3.05
WALL_TOL = 2
FS_OFF = 1.0
ROBUST = 0
LATE_B = 0
UNK_ON = 0
UNK_X = 248
WAIT_BAD = 0
ROB_SLACK = 1
STAIRS_OFF = 1
GBRAKE = 1
GDEC = 0.3
LOWPREF = 0
NOSTOMP_AIR = 1
LOWTOL = 10
LAST_YL = None
LAST_XL = None
WORLD = [0.0]
AIR_BRAKE_MAX = 8
TRACE_LO = 10 ** 9
TRACE_HI = 10 ** 9
AIRBRAKE = 0
LAND_CAP = 0
PIT_OVERONLY = 1
GB_FIRST = False    # (True: 6/20 vs 8/20) # try ground-brake plans before the half-margin (ms<1) jump plans
COMMIT_REPLAN = False  # (True: 0/20 -- every variant died at the x1335 pit jump) # while holding a committed jump (rising), release A early if the planner predicts a hit
GFREEZE = True     # ground-brake plans: speed frozen at takeoff value while airborne (no in-air acceleration)
PH_DX2 = 0         # second phantom walker (distance ahead, px); 0 = off
TALL_STOMP = True   # airborne Koopa-sized (13x17) enemies (Paratroopa) are stompable; 11x12 Cheep-Cheeps never
TALL_H = 15
TALL_COL = False
GREEN_KEEP = True   # do not blank green (Koopa shell) pixels near Mario: they are never Mario's own
TALL_LOW = 45
TALL_UP = 10

TUNABLE = {"AIR_ACC": (0.0, 0.08), "AIR_DEC": (0.1, 0.5), "AIRG": (0.1, 1.2), "AIRCAP": (6, 40), "MARGIN_X": (1, 6), "MARGIN_Y": (0, 5), "WAIT_MAX": (0, 10), "POST_LAND": (6, 24),
           "TERR_POST": (3, 24), "AIR_X": (0.94, 1.02), "WALL_TOL": (2, 5)}


def _eq(img, c):
    return (img[:, :, 0] == c[0]) & (img[:, :, 1] == c[1]) & (img[:, :, 2] == c[2])


def find_mario(img, prev):
    b = _eq(img[32:], BROWN)
    ys, xs = np.nonzero(b)
    if len(xs) == 0:
        return None
    ys = ys + 32
    if prev is not None:
        d = np.abs(xs - prev[0]) + np.abs(ys - prev[1])
        m = d < 60
        if m.any():
            xs, ys = xs[m], ys[m]
    return int((xs.min() + xs.max()) // 2), int(ys.max()), int(ys.min())


def threats(img, mario):
    sub = img[40:232]
    mask = _eq(sub, RED) | _eq(sub, GREEN)
    ys, xs = np.nonzero(mask)
    out = []
    if len(xs) == 0:
        return out
    ys = ys + 40
    if mario is not None:
        cx, feet, top = mario
        isred = _eq(img[ys, xs][None], RED)[0] if GREEN_KEEP else np.ones(len(xs), bool)
        keep = ~((np.abs(xs - cx) <= 10) & (ys >= feet - 18) & (ys <= feet + 1) & isred)
        xs, ys = xs[keep], ys[keep]
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    used = np.zeros(len(xs), bool)
    for i in range(len(xs)):
        if used[i]:
            continue
        grp = [i]
        used[i] = True
        k = 0
        while k < len(grp):
            j0 = grp[k]
            k += 1
            d = (np.abs(xs - xs[j0]) <= 6) & (np.abs(ys - ys[j0]) <= 5) & (~used)
            idx = np.nonzero(d)[0]
            used[idx] = True
            grp.extend(idx.tolist())
        if len(grp) < 4:
            continue
        gx, gy = xs[grp], ys[grp]
        if (gy.max() - gy.min() + 1) <= 3 and (gx.max() - gx.min() + 1) >= 16:
            continue
        if (gx.max() - gx.min() + 1) > MAXW:
            continue
        out.append({"x0": int(gx.min()), "x1": int(gx.max()), "y0": int(gy.min()), "y1": int(gy.max()),
                    "n": len(grp)})
    return out


def _nbr(g, D, fn):
    n = len(g)
    if fn is np.maximum:
        pad = np.concatenate([np.full(D, g[0]), g, np.full(D, g[-1])])
    else:
        pad = np.concatenate([np.full(D, INF), g, np.full(D, INF)])
    out = g.copy()
    for d in range(1, D + 1):
        out = fn(out, pad[D - d:D - d + n])
        out = fn(out, pad[D + d:D + d + n])
    return out


def heightmap(img, fref):
    r0 = max(fref - (UP_WIN if fref >= UP_FREF else 24), 40)
    r1 = 234
    sub = img[r0:r1]
    S = _eq(sub, BRICK) | _eq(sub, TAN)
    low = _eq(sub, LIME) | _eq(sub, GREEN)
    cut = max(fref + 1 - r0, 0)
    S[cut:] |= low[cut:]
    S3 = S[:-2] & S[1:-1] & S[2:]
    has = S3.any(axis=0)
    first = np.argmax(S3, axis=0)
    g = np.where(has, first + r0, INF).astype(np.float64)
    return _nbr(g, 1, np.minimum)


def _plans_ground():
    pw = [10 ** 6]
    pk = [0]
    pw2 = [0]
    pk2 = [0]
    for w in WS:
        for k in KS:
            pw.append(w)
            pk.append(k)
            pw2.append(0)
            pk2.append(0)
    n1 = len(pw)
    for i in range(1, n1):
        for w2 in W2S:
            for k2 in K2S:
                pw.append(pw[i])
                pk.append(pk[i])
                pw2.append(w2)
                pk2.append(k2)
    for w2 in FW2S:
        for k2 in FK2S:
            pw.append(10 ** 6)
            pk.append(0)
            pw2.append(w2)
            pk2.append(k2)
    return np.array(pw), np.array(pk), np.array(pw2), np.array(pk2)


PW_G, PK_G, PW2_G, PK2_G = _plans_ground()


def simulate(y0, vy0, grounded0, cx, gsup, gall, gcen, fref, pw, pk, boxes, speed, pw2=None, pk2=None, jumped0=False, nostomp=False, spmat=None, unk_bad=False, freeze_spm=False):
    n = len(pw)
    spc = np.full(n, float(speed))
    y = np.full(n, float(y0))
    vy = np.full(n, float(vy0))
    gr = np.full(n, bool(grounded0))
    xf = np.full(n, float(cx))
    jumped = np.full(n, bool(jumped0))
    tland = np.full(n, 999)
    tland2 = np.full(n, 999)
    jumped2 = np.zeros(n, bool)
    yl = np.full(n, -1.0)
    xl = np.full(n, -1.0)
    bad = np.full(n, 999)
    dbl = (pk2 > 0) if pk2 is not None else np.zeros(n, bool)
    dead = [np.zeros(n, bool) for _ in boxes[0]]
    for t in range(HORIZ):
        press = (t >= pw + DELAY) & (t < pw + DELAY + pk)
        if pk2 is not None:
            t2 = tland + 1 + pw2 + DELAY
            press2 = (tland < 999) & (t >= t2) & (t < t2 + pk2)
            press = press | press2
        start = gr & press
        if pk2 is not None:
            jumped2 |= start & (tland < 999)
        vy = np.where(start, V0, vy)
        gr = gr & ~start
        jumped |= start
        air = ~gr
        if spmat is not None:
            if freeze_spm:
                spc = np.where(air, spc, spmat[:, t])
                sp_t = spc
            else:
                sp_t = spmat[:, t]
        else:
            sp_t = np.minimum(np.maximum(speed, SPMAX), speed + ACC * t)
        xf = xf + sp_t * np.where(air, AIR_X, 1.0)
        xi = np.clip(np.rint(xf).astype(np.int64), 0, 255)
        vb = vy.copy()
        grb = gr.copy()
        y = np.where(air, y - vy, y)
        gg = np.where(press & (vy > 0), GH, GD)
        vy = np.where(air, vy - gg, vy)
        fs = gsup[xi] - FS_OFF
        okg = gall[xi] < INF - 10
        okc = gcen[xi] < INF - 10
        deep = y - fs
        wall = (deep > WALL_TOL) & (grb | (vb > 0) | (deep > WALL_TOL + np.maximum(-vb, 0)))
        land = air & (deep >= 0) & ~wall & (vb <= 0) & okg
        snap = grb & (deep > 0) & (deep <= 4)
        y = np.where(land | snap, fs, y)
        vy = np.where(land, 0.0, vy)
        newland = land & (tland == 999) & jumped
        yl = np.where(newland, y, yl)
        tland = np.where(newland, t, tland)
        newland2 = land & (tland2 == 999) & jumped2
        tland2 = np.where(newland2, t, tland2)
        xl = np.where(newland | newland2, xf, xl)
        gr = gr | land
        off = gr & ~land & (((fs - y) > 1.0) | (~okc))
        gr = gr & ~off
        jumped |= off
        vy = np.where(off, 0.0, vy)
        fall = (~gr) & (y > 232)
        tb = wall | fall
        if unk_bad:
            tb = tb | ((~gr) & (xf >= UNK_X) & (tland == 999) & jumped & (~okg))
        active = (~jumped) | (t <= tland + TERR_POST) | (dbl & (t <= tland2 + TERR_POST))
        eb = np.zeros(n, bool)
        for e, (ex0, ey0, ex1, ey1, etop, egnd, etall) in enumerate(boxes[t]):
            hit = (xf + 5 >= ex0) & (xf - 5 <= ex1) & (y >= ey0) & (y - 13 <= ey1) & ~dead[e]
            if egnd:
                sallow = not nostomp
            elif etall and TALL_STOMP:
                sallow = not nostomp
            else:
                sallow = not NOSTOMP_AIR
            stomp = hit & air & (vb <= 0) & (y <= etop + 5 + np.maximum(-vb, 0)) & sallow
            vy = np.where(stomp, 4.5, vy)
            dead[e] = dead[e] | stomp
            eb |= hit & ~stomp
        eact = (~jumped) | (t <= np.where(dbl, tland2, tland) + POST_LAND)
        newbad = ((tb & active) | (eb & eact)) & (bad == 999)
        bad = np.where(newbad, t, bad)
    global LAST_YL, LAST_XL
    LAST_YL = yl
    LAST_XL = xl
    return bad, tland


class Track:
    __slots__ = ("hist", "x0", "x1", "y0", "y1")

    def __init__(self, c, frame):
        self.hist = [(frame, (c["x0"] + c["x1"]) / 2.0, (c["y0"] + c["y1"]) / 2.0)]
        self.update(c)

    def update(self, c):
        self.x0, self.x1, self.y0, self.y1 = c["x0"], c["x1"], c["y0"], c["y1"]


def update_tracks(tracks, clusters, frame):
    new = []
    used = set()
    for c in clusters:
        xc = (c["x0"] + c["x1"]) / 2.0
        yc = (c["y0"] + c["y1"]) / 2.0
        best = None
        for i, tr in enumerate(tracks):
            if i in used:
                continue
            f, px, py = tr.hist[-1]
            if frame - f > 3:
                continue
            d = abs(px - xc) + abs(py - yc)
            if abs(px - xc) <= 10 and abs(py - yc) <= 10 and (best is None or d < best[0]):
                best = (d, i)
        if best is not None:
            used.add(best[1])
            tr = tracks[best[1]]
            tr.hist.append((frame, xc, yc))
            if len(tr.hist) > 6:
                tr.hist.pop(0)
            tr.update(c)
            new.append(tr)
        else:
            new.append(Track(c, frame))
    return new


def predict_boxes(tracks, cx, fref, scroll, frame, ms=1.0):
    boxes = [[] for _ in range(HORIZ)]
    for tr in tracks:
        if tr.x1 < cx - 14:
            continue
        hh = tr.hist
        n = len(hh)
        ground = tr.y1 >= fref - 6
        if n >= 3:
            f0, x0_, y0_ = hh[0]
            f1, x1_, y1_ = hh[-1]
            dt = max(f1 - f0, 1)
            vxs = (x1_ - x0_) / dt
            vys = (y1_ - y0_) / dt
            unc = 0.0
        else:
            vxs = -(scroll + 0.6) if ground else -(scroll + 0.0)
            vys = 0.0
            unc = 5.0
        vxw = vxs + scroll
        ay = 0.1 if abs(vys) >= 1.0 else 0.0
        ey0 = tr.y0 - ((4 + 5 * ms) if ground else 2 * ms)
        ey1 = tr.y1 + 2 * ms
        etop = tr.y0 - (8 if ground else 0)
        if TALL_COL and (not ground) and (tr.y1 - tr.y0) >= TALL_H and tr.y1 >= fref - TALL_LOW:
            # hopping Paratroopa: unknown bounce phase -> block the whole column down to the deck
            ey1 = max(ey1, fref + 2)
            ey0 = ey0 - TALL_UP
        ex0 = tr.x0 - MARGIN_X * ms
        ex1 = tr.x1 + MARGIN_X * ms
        for t in range(HORIZ):
            tt = t + 1
            dx = vxw * tt
            dy = vys * tt + 0.5 * ay * tt * tt
            mx = 0.03 * tt * ms
            if ground:
                my = (unc + (0.25 if abs(vys) >= 1.0 else 0.06) * tt + MARGIN_Y) * ms
            else:
                my = (unc + min(AIRG * tt, AIRCAP) + MARGIN_Y) * ms
                mx = mx + 0.04 * tt * ms
            boxes[t].append((ex0 + dx - mx, ey0 + dy - my, ex1 + dx + mx, ey1 + dy + my, etop + dy, bool(ground), (tr.y1 - tr.y0) >= TALL_H))
    return boxes


def commit_replan(img, feet, lastfeet, cx, fref, tracks, scroll_now, speed, commit_left):
    """While a jump is held and Mario is still rising: how many more frames to hold A so the flight stays safe?
    Returns commit_left if the committed hold is safe (or nothing better exists), else the largest safe shorter hold."""
    g = heightmap(img, fref)
    gsup = _nbr(g, 5, np.minimum)
    gall = _nbr(g, 4, np.maximum)
    gcen = _nbr(g, 3, np.maximum)
    ns = bool(NOSTOMP_PIT and (gall[min(cx + 6, 255):min(cx + NS_D, 256)] > INF - 10).any())
    rs = sorted(set([commit_left] + [r for r in (0, 2, 4, 6, 8, 10, 12, 16, 20, 24) if r < commit_left]))
    n = len(rs)
    pw = np.full(n, -DELAY)
    pk = np.array([DELAY + r for r in rs])
    vy0 = float(lastfeet - feet) - GH
    ic = rs.index(commit_left)
    for ms in (1.0, 0.5):
        boxes = predict_boxes(tracks, cx, fref, scroll_now, 0, ms)
        bad, _tl = simulate(feet, vy0, False, cx, gsup, gall, gcen, fref, pw, pk, boxes, speed, None, None,
                            jumped0=True, nostomp=ns)
        if bad[ic] == 999:
            return commit_left
        ok = np.nonzero(bad == 999)[0]
        if len(ok):
            return int(rs[int(ok.max())])
    return commit_left


_ZR = [0]


def measure_scroll(prev_blk, blk):
    """2-D background shift (0..6 px/frame) from the right part of the screen; None if ambiguous."""
    errs = np.array([np.abs(prev_blk[:, s:s + 100] - blk[:, :100]).sum() for s in range(0, 7)])
    j = int(np.argmin(errs))
    srt = np.sort(errs)
    if srt[1] - srt[0] < SC_MARGIN:
        _ZR[0] = 0
        return None
    if j == 0:
        _ZR[0] += 1
        if _ZR[0] < 2:
            return None
    else:
        _ZR[0] = 0
    return j


def play(sim):
    try:
        _play(sim)
    except Exception as e:  # pragma: no cover
        print("play() exception: %r -> fallback runner" % (e,))
        try:
            k = 0
            while sim.budget_left > 0:
                keys = {"Right", "x"}
                if k % 40 < 14:
                    keys.add("z")
                sim.step(keys)
                k += 1
        except Exception:
            return


def _play(sim):
    try:
        pr = sim.mem.past.get("params", {})
        for k_, v_ in pr.items():
            if k_ in globals():
                globals()[k_] = v_
        if pr:
            print("params", pr)
    except Exception:
        pass
    prev = None
    lastfeet = None
    fref = None
    pressed_prev = False
    commit_left = 0
    commit_age = 0
    last_screen = None
    frozen = 0
    tracks = []
    S_est = 1.0
    prev_row = None
    prev_cx = None
    t = 0
    ring = []
    wx = 0.0
    wx0 = None
    fell_wx = False
    scroll = None
    dcx = 0
    abn = 0
    jlog = []
    nolost = 0
    ndump = 0
    while sim.budget_left > 0:
        img = sim.screen()
        frame = sim.frame
        m = find_mario(img, prev)
        press = False
        brake = False
        info = ""
        grounded = False
        if m is not None:
            cx, feet, top = m
            prev = (cx, feet)
            H, W = img.shape[:2]
            ysl = slice(min(feet + 1, H - 1), min(feet + 3, H))
            xsl = slice(max(cx - 4, 0), cx + 5)
            sub = img[ysl, xsl]
            under = bool((_eq(sub, BRICK) | _eq(sub, TAN) | _eq(sub, LIME)).any())
            grounded = under and (lastfeet is None or abs(feet - lastfeet) <= 1)
            if grounded or fref is None:
                fref = feet
            row = img[44:232, 140:256, :].astype(np.int32).sum(axis=2)
            scroll = None
            if prev_row is not None:
                scroll = measure_scroll(prev_row, row)
            prev_row = row
            dcx = 0 if prev_cx is None else (cx - prev_cx)
            prev_cx = cx
            if scroll is not None:
                S_est = 0.6 * S_est + 0.4 * min(scroll + max(dcx, 0), 3.5)
                scroll_now = float(scroll)
            else:
                scroll_now = S_est if cx >= 116 else 0.0
            speed = max(min(S_est, 3.2), 0.5)
            if wx0 is None:
                wx0 = cx
            wx += (scroll if scroll is not None else (scroll_now if cx >= 116 else 0.0)) + max(dcx, 0)
            WORLD[0] = wx + wx0 - cx
            if m is not None and feet >= 232 and not fell_wx:
                fell_wx = True
                print("FALL f%d cx=%d Mario world x ~ %d" % (frame, cx, wx + wx0))
            if feet < 200:
                fell_wx = False

            clusters = threats(img, m)
            tracks = update_tracks(tracks, clusters, frame)
            if commit_left > 0:
                commit_age += 1
                if grounded and commit_age >= DELAY + 3:
                    commit_left = 0
                    info = "commit-reset"
                else:
                    if COMMIT_REPLAN and (not grounded) and lastfeet is not None and feet < lastfeet and len(tracks) > 0:
                        try:
                            r_new = commit_replan(img, feet, lastfeet, cx, fref, tracks, scroll_now, speed, commit_left)
                            if r_new != commit_left:
                                info = "commit-cut %d->%d" % (commit_left, r_new)
                                commit_left = r_new
                        except Exception:
                            pass
                    if commit_left > 0:
                        press = True
                        commit_left -= 1
                        if not info:
                            info = "commit"
                    else:
                        press = False
            elif grounded:
                g = heightmap(img, fref)
                if frame < 50:
                    g[:84] = fref + 1
                g[:cx + 6] = fref + 1
                gsup = _nbr(g, 5, np.minimum)
                gall = _nbr(g, 4, np.maximum)
                gcen = _nbr(g, 3, np.maximum)
                ahead = gsup[min(cx + 6, 255):min(cx + 175, 256)]
                flat = bool((ahead == fref + 1).all()) if len(ahead) else True
                if flat and len(tracks) == 0:
                    info = "free"
                else:
                    chosen = False
                    pw, pk, pw2, pk2 = PW_G, PK_G, PW2_G, PK2_G
                    ns_base = NOSTOMP_PIT and bool((gall[min(cx + 6, 255):min(cx + NS_D, 256)] > INF - 10).any())
                    combos = [(ms_, ns_base) for ms_ in MS_LEVELS]
                    if ns_base and STOMP_FALLBACK:
                        combos.insert(2, (1.0, False))
                        combos.append((0.5, False))
                    gb_tried = False

                    def _gb():
                        try:
                            plans_g = []
                            for bb_ in (2, 4, 7, 10, 14, 20):
                                plans_g.append((bb_, 10 ** 6, 0))
                                for pw_ in (0, 3, 8, 15, 25):
                                    for k_ in (6, 12, 20, 26):
                                        plans_g.append((bb_, bb_ + pw_, k_))
                            nb_ = len(plans_g)
                            spm_g = np.zeros((nb_, HORIZ))
                            vtop_ = max(speed, SPMAX)
                            for ii_, (bb_, _pw, _pk) in enumerate(plans_g):
                                v_ = speed
                                for tt_ in range(HORIZ):
                                    if tt_ < bb_:
                                        v_ = max(v_ - GDEC, 0.0)
                                    else:
                                        v_ = min(v_ + ACC, vtop_)
                                    spm_g[ii_, tt_] = v_
                            boxes_g = predict_boxes(tracks, cx, fref, scroll_now, frame, 0.5)
                            pwg = np.array([p[1] for p in plans_g])
                            pkg = np.array([p[2] for p in plans_g])
                            bad_g, _tg = simulate(feet, 0.0, True, cx, gsup, gall, gcen, fref, pwg, pkg, boxes_g, speed,
                                                  None, None, spmat=spm_g, freeze_spm=GFREEZE)
                            okg_ = np.nonzero(bad_g == 999)[0]
                            if len(okg_):
                                return "gbrake b=%d pw=%d k=%d" % plans_g[int(okg_[0])]
                        except Exception:
                            pass
                        return None

                    for ms, ns_g in combos:
                        if GB_FIRST and GBRAKE and ms < 1.0 and not gb_tried:
                            gb_tried = True
                            r_gb = _gb()
                            if r_gb:
                                brake = True
                                info = r_gb
                                chosen = True
                                break
                        boxes = predict_boxes(tracks, cx, fref, scroll_now, frame, ms)
                        bad, tl = simulate(feet, 0.0, True, cx, gsup, gall, gcen, fref, pw, pk, boxes, speed, pw2, pk2,
                                           nostomp=ns_g, unk_bad=bool(UNK_ON and ns_g))
                        ok = bad == 999
                        if LAND_CAP > 0 and bool((gall[236:256] < INF - 10).all()):
                            okc_ = ok & ((LAST_XL < 0) | (LAST_XL <= LAND_CAP))
                            if (okc_ & (pk > 0)).any() or okc_[0]:
                                ok = okc_
                        if pressed_prev:
                            allow = (pw >= 1) | (pk == 0)
                            ok = ok & allow
                            bad = np.where(allow, bad, -1)
                        if ok[0]:
                            info = "none-ok ms=%.1f" % ms
                            chosen = True
                        elif (ok & (pk == 0) & (pk2 > 0)).any():
                            info = "fall-jump ms=%.1f" % ms
                            chosen = True
                        elif ok.any():
                            okj = ok & (pk > 0)
                            if LOWPREF > 0:
                                good_ = okj & (pk2 == 0) & (LAST_YL <= feet + LOWTOL)
                                if good_.any():
                                    okj = good_
                            single = okj & (pk2 == 0)
                            if single.any():
                                okj = single
                            if ROBUST and single.any():
                                nW, nK = len(WS), len(KS)
                                grid = ok[1:1 + nW * nK].reshape(nW, nK)
                                P = np.pad(grid.astype(np.int32), 1, constant_values=1)
                                sc9 = np.zeros((nW, nK), np.int32)
                                for dw in range(3):
                                    for dk in range(3):
                                        sc9 += P[dw:dw + nW, dk:dk + nK]
                                sc9 = np.where(grid, sc9, -1)
                                smax = int(sc9.max())
                                rows = np.nonzero((sc9 == smax).any(axis=1))[0]
                                iw = int(rows.max())
                                if 0 < iw and WS[iw] <= WAIT_MAX and grid[0].any() and int(sc9[0].max()) >= smax - ROB_SLACK:
                                    iw = 0
                                    smax = int(sc9[0].max())
                                ks_ = np.nonzero(sc9[iw] == smax)[0]
                                ik = int(ks_[len(ks_) // 2])
                                i = 1 + iw * nK + ik
                                wmax = int(WS[iw])
                            else:
                                wmax = int(pw[okj].max())
                                if LATE_B > 0 and wmax > 0:
                                    wsl = np.unique(pw[okj])
                                    lo_ = wsl[wsl <= wmax - LATE_B]
                                    wmax = int(lo_.max()) if len(lo_) else int(wsl.min())
                                sel = okj & (pw == wmax)
                                if 0 < wmax <= WAIT_MAX and (okj & (pw == 0)).any():
                                    sel = okj & (pw == 0)
                                    wmax = 0
                                cand = np.nonzero(sel)[0]
                                cand = cand[np.argsort(pk[cand], kind="stable")]
                                near_pit = bool((gall[min(cx + 6, 255):min(cx + 140, 256)] > INF - 10).any())
                                i = cand[len(cand) // 2] if near_pit else cand[0]
                            if wmax == 0:
                                press = True
                                commit_left = int(pk[i]) - 1
                                commit_age = 0
                            info = "jump k=%d w=%d k2=%d ms=%.1f tl=%d" % (pk[i], wmax, pk2[i], ms, tl[i])
                            chosen = True
                        if chosen:
                            break
                    if (not chosen) and GBRAKE and not gb_tried:
                        gb_tried = True
                        r_gb = _gb()
                        if r_gb:
                            brake = True
                            info = r_gb
                            chosen = True
                    if False:
                        try:
                            plans_g = []
                            for bb_ in (2, 4, 7, 10, 14, 20):
                                plans_g.append((bb_, 10 ** 6, 0))
                                for pw_ in (0, 3, 8, 15, 25):
                                    for k_ in (6, 12, 20, 26):
                                        plans_g.append((bb_, bb_ + pw_, k_))
                            nb_ = len(plans_g)
                            spm_g = np.zeros((nb_, HORIZ))
                            vtop_ = max(speed, SPMAX)
                            for ii_, (bb_, _pw, _pk) in enumerate(plans_g):
                                v_ = speed
                                for tt_ in range(HORIZ):
                                    if tt_ < bb_:
                                        v_ = max(v_ - GDEC, 0.0)
                                    else:
                                        v_ = min(v_ + ACC, vtop_)
                                    spm_g[ii_, tt_] = v_
                            boxes_g = predict_boxes(tracks, cx, fref, scroll_now, frame, 0.5)
                            pwg = np.array([p[1] for p in plans_g])
                            pkg = np.array([p[2] for p in plans_g])
                            bad_g, _tg = simulate(feet, 0.0, True, cx, gsup, gall, gcen, fref, pwg, pkg, boxes_g, speed,
                                                  None, None, spmat=spm_g, freeze_spm=GFREEZE)
                            okg_ = np.nonzero(bad_g == 999)[0]
                            if len(okg_):
                                brake = True
                                info = "gbrake b=%d pw=%d k=%d" % plans_g[int(okg_[0])]
                                chosen = True
                        except Exception:
                            pass
                    if not chosen and WAIT_BAD > 0 and int(bad[0]) >= WAIT_BAD:
                        info = "wait-unk bad0=%d" % int(bad[0])
                        chosen = True
                    if not chosen:
                        sc = bad * 1000 + (pk > 0) * 100 + (pw == 0) * 50 + pk
                        i = int(np.argmax(sc))
                        if pw[i] == 0 and pk[i] > 0 and not pressed_prev:
                            press = True
                            commit_left = int(pk[i]) - 1
                            commit_age = 0
                        info = "NOSAFE best k=%d w=%d bad=%d" % (pk[i], pw[i], bad[i])
            else:
                info = "air"
                g = heightmap(img, fref)
                gsup = _nbr(g, 5, np.minimum)
                gall = _nbr(g, 4, np.maximum)
                gcen = _nbr(g, 3, np.maximum)
                over_pit = bool((g[max(cx - 4, 0):cx + 5] > INF - 10).all())
                pit_ahead = bool((gall[min(cx + 6, 255):min(cx + PITBAN_D, 256)] > INF - 10).any())
                ah_ = g[min(cx + 6, 255):min(cx + 130, 256)]
                fin_ = ah_[ah_ < INF - 10]
                stairs = bool(STAIRS_OFF and len(fin_) >= 50 and len(np.unique((fin_ // 8).astype(np.int64))) >= 5)
                pitban = (PITBAN and (over_pit or (pit_ahead and not PIT_OVERONLY))) or stairs
                if AIRBRAKE and (not pitban) and lastfeet is not None and feet > lastfeet:
                    for tr in tracks:
                        if tr.y1 > fref + 25 and 0 < tr.x0 - cx < 100 and tr.y1 < 236:
                            brake = True
                            info = "air-brake"
                            break
                if USE_AIRCTL and wx < AIRCTL_WX and (not pitban) and (not brake) and commit_left == 0 and lastfeet is not None and (len(tracks) > 0 or PHANTOM):
                    tracks_real = tracks
                    try:
                        if PHANTOM and feet < fref - 8:
                            ph = Track({"x0": min(cx + PH_DX, 250), "x1": min(cx + PH_DX + 12, 255),
                                        "y0": fref - 16, "y1": fref}, frame)
                            tracks = tracks_real + [ph]
                            if PH_DX2 > 0:
                                ph2 = Track({"x0": min(cx + PH_DX2, 250), "x1": min(cx + PH_DX2 + 12, 255),
                                             "y0": fref - 16, "y1": fref}, frame)
                                tracks = tracks + [ph2]
                        dyv = float(lastfeet - feet)
                        vy0 = dyv - GD
                        npl = len(AIR_BS)
                        pwa = np.full(npl, 10 ** 6)
                        pka = np.zeros(npl, dtype=int)
                        spm = np.zeros((npl, HORIZ))
                        vtop = max(speed, SPMAX)
                        for ii, bb_ in enumerate(AIR_BS):
                            v_ = speed
                            for tt_ in range(HORIZ):
                                if 1 <= tt_ < 1 + bb_:
                                    v_ = max(v_ - AIR_DEC, -1.5)
                                elif tt_ >= 1:
                                    v_ = min(v_ + AIR_ACC, vtop)
                                spm[ii, tt_] = v_
                        pick = 0
                        ns_a = NOSTOMP_PIT and (over_pit or bool(
                            (gall[min(cx + 6, 255):min(cx + NS_D, 256)] > INF - 10).any()))
                        found = False
                        for nsf in ((True, False) if (ns_a and STOMP_FALLBACK) else (True,)):
                            for ms in (1.0, 0.5):
                                boxes = predict_boxes(tracks, cx, fref, scroll_now, frame, ms)
                                bada, _tl = simulate(feet, vy0, False, cx, gsup, gall, gcen, fref, pwa, pka, boxes, speed,
                                                     None, None, jumped0=True, spmat=spm,
                                                     nostomp=bool(ns_a and nsf))
                                if bada[0] == 999:
                                    pick = 0
                                    found = True
                                    break
                                if (bada == 999).any():
                                    pick = int(np.nonzero(bada == 999)[0][0])
                                    found = True
                                    break
                                pick = int(np.argmax(bada))
                                if bada[pick] <= bada[0]:
                                    pick = 0
                            if found:
                                break
                        if pick > 0:
                            brake = True
                            info = "air-ctl L%d" % pick
                    except Exception:
                        info = "air-ctl-exc"
                    finally:
                        tracks = tracks_real
            lastfeet = feet
        else:
            press = pressed_prev and commit_left > 0
        if m is not None:
            if grounded:
                abn = 0
            elif brake:
                abn += 1
                if abn > AIR_BRAKE_MAX:
                    brake = False
        keys = {"Right", "x"}
        if brake:
            keys = {"Left"}
        if press:
            keys.add("z")
        pressed_prev = press
        _near = ""
        if m is not None:
            _tt = [(tr.x0 - m[0], tr.y0 - m[1], tr.x1 - tr.x0, tr.y1 - tr.y0) for tr in tracks if abs(tr.x0 - m[0]) < 90]
            _near = str(_tt[:4])
        ring.append("f%d W=%d m=%s S=%.1f sc=%s dcx=%s %s%s %s T=%s" % (frame, int(WORLD[0] + (m[0] if m else 0)), m, S_est,
                                                                         scroll if m is not None else None, dcx,
                                                                         "L" if brake else "R",
                                                                         "Z" if press else "-", info[:44], _near[:70]))
        if len(ring) > 140:
            ring.pop(0)
        if TRACE_LO <= frame <= TRACE_HI:
            print(ring[-1])
        if info.startswith("jump") or info.startswith("NOSAFE") or info.startswith("fall"):
            jlog.append(ring[-1])
            if len(jlog) > 8:
                jlog.pop(0)
        if m is None:
            nolost += 1
            if nolost == 3 and ndump < 1:
                ndump += 1
                print("JUMP DECISIONS:")
                for ln in jlog:
                    print("  " + ln)
                print("RING (Mario lost at f%d):" % frame)
                for ln in ring[-120:]:
                    print("  " + ln)
        else:
            nolost = 0
        sim.step(keys)
        t += 1
        if t % 200 == 0:
            print("f%d mario=%s S=%.2f" % (sim.frame, m, S_est))
        if last_screen is not None and np.array_equal(img[40:], last_screen[40:]):
            frozen += 1
            if frozen == 1 and frame > 60 and ndump < 3:
                ndump += 1
                print("JUMP DECISIONS (frozen):")
                for ln in jlog:
                    print("  " + ln)
                print("RING (screen frozen at f%d):" % frame)
                for ln in ring[-60:]:
                    print("  " + ln)
            if frozen >= STOP_FROZEN:
                print("frozen -> stop at f%d, last mario=%s" % (sim.frame, m))
                return
        else:
            frozen = 0
        last_screen = img


def solve(sim):
    return []

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': '"""State assessment for SMB 7-3 (pixels only). Cheap."""\nimport numpy as np\n\nBROWN = (172, 124, 0)\nRED = (248, 56, 0)\nGREEN = (0, 168, 0)\n\n\ndef _eq(img, c):\n    return (img[:, :, 0] == c[0]) & (img[:, :, 1] == c[1]) & (img[:, :, 2] == c[2])\n\n\ndef assess(frame, ctx):\n    try:\n        sub = frame[40:232]\n        ys, xs = np.nonzero(_eq(sub, BROWN))\n        if len(xs) == 0:\n            return {"status": "dead", "where": {"hazard": "pit_or_hit", "confidence": 0.6}, "progress": None,\n                    "resources": {}, "threats": [], "stuck": {"is": False},\n                    "outlook": {"label": "bad", "p_through": 0.0}, "note": "no Mario pixels"}\n        cx = int((xs.min() + xs.max()) // 2)\n        feet = int(ys.max()) + 40\n        ty, tx = np.nonzero(_eq(sub, RED) | _eq(sub, GREEN))\n        ty = ty + 40\n        near = int(((tx >= cx - 10) & (tx <= cx + 60) & (ty >= feet - 40) & (ty <= feet + 4)\n                    & ~((np.abs(tx - cx) <= 10) & (ty >= feet - 18) & (ty <= feet + 1))).sum())\n        stuck = bool(ctx.get("stuck", {}).get("is")) if isinstance(ctx.get("stuck"), dict) else False\n        d = near >= 6\n        return {"status": "danger" if d else "running",\n                "where": {"hazard": "enemy_ahead" if d else None, "confidence": 0.5},\n                "progress": None, "resources": {}, "threats": [{"kind": "enemy", "pixels": near}] if near else [],\n                "stuck": {"is": stuck}, "outlook": {"label": "bad" if d else "ok", "p_through": 0.35 if d else 0.6},\n                "note": "mario cx=%d feet=%d" % (cx, feet)}\n    except Exception as e:\n        return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n                "resources": {}, "threats": [], "stuck": {"is": False}, "outlook": {"label": "ok", "p_through": 0.5},\n                "note": "assess error %r" % (e,)}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

