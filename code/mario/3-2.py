# ==================================================================================================
# Super Mario Bros. (NES) -- level 3-2  |  超级马里奥兄弟（NES） 关 3-2
# Pass rate on 100 random variants never used before: 93/100; fastest finish 1188 frames (19.80 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 93/100；最快一局 1188 帧（60 帧/秒，约 19.80 秒）。
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
#   PB closed-loop vision policy + cleaner Mario localisation (red/brown core only, skin only adja
#   cent) so pipes/walls/plants are detected correctly
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def mario_masks                    line 45    
#   def tall_walls                     line 57    gaps (from Mario's front) of tall (>=44px) wall starts ahead.
#   def plan_penalty                   line 82    Simulate a jump with hold h from now; 0 = safe, >0 = hazard (bigger = worse).
#   def ceiling_space                  line 115   free pixels between Mario's head and the nearest floating orange block above (999 if none).
#   def find_goombas                   line 148   nb: bool non-black mask (H,W). returns list of (x0,y0) of goomba-shaped blobs.
#   def find_green                     line 167   
#   def play                           line 189   
#   def solve                          line 488   
# ==================================================================================================

# idea: PB closed-loop vision policy + cleaner Mario localisation (red/brown core only, skin only adjacent) so pipes/walls/plants are detected correctly
import numpy as np

GOOMBA = [
    "......####......",
    ".....######.....",
    "....########....",
    "...##########...",
    "..#..######..#..",
    ".####.####.####.",
    ".####......####.",
    "#####.####.#####",
    "################",
    "################",
    ".##############.",
    "....########....",
    "....########....",
    ".....#####......",
    "......###.......",
]
GT = np.array([[c == '#' for c in r] for r in GOOMBA], dtype=bool)
GH, GW = GT.shape
G_ON = GT
G_OFF = np.zeros_like(GT)
G_OFF[0:3, 0:3] = True
G_OFF[0:3, 13:16] = True
G_OFF[11:15, 0:3] = True
G_OFF[11:15, 13:16] = True
G_OFF &= ~GT

try:
    from numpy.lib.stride_tricks import sliding_window_view
except Exception:  # pragma: no cover
    sliding_window_view = None


WALL_GAP = 18
VERBOSE = False
DBG = (0, -1)
P = {'order': 0, 'hz': 0, 'steer': 1, 'pit_steer': 0, 'retreat': 1, 'en_gap': 46, 'ledge': 0, 'mask': 0, 'acc_wx': 2450, 'porder': 0, 'pitland': 1}
ORDERS = [[14, 8, 24, 18, 4], [24, 18, 14, 8, 4], [8, 14, 4, 18, 24]]
PORDERS = [[4, 8, 14, 18, 24], [8, 4, 14, 18, 24], [8, 14, 4, 18, 24]]


def mario_masks(s):
    r, g, b = s[..., 0], s[..., 1], s[..., 2]
    core = ((r == 248) & (g == 56) & (b == 0)) | ((r == 172) & (g == 124) & (b == 0))
    skin = (r == 252) & (g == 160) & (b == 68)
    core[:40] = False
    skin[:40] = False
    return core, skin


JTAB = {4: (41, 17), 8: (55, 21), 14: (70, 34), 18: (77, 39), 24: (84, 46)}


def tall_walls(nb, y1, xf):
    """gaps (from Mario's front) of tall (>=44px) wall starts ahead."""
    res = []
    body = nb[y1 - 13:y1 - 1, :]
    wallc = body.sum(axis=0) >= 9
    x = xf + 1
    while x < 252:
        if wallc[x]:
            x2 = x
            while x2 < 256 and wallc[x2]:
                x2 += 1
            if x2 - x >= 7:
                xc_ = min(255, x + 4)
                hh_, yy2 = 0, y1
                while yy2 > 44 and nb[yy2, xc_]:
                    hh_ += 1
                    yy2 -= 1
                if hh_ >= 44:
                    res.append(x - xf)
            x = x2
        else:
            x += 1
    return res


def plan_penalty(h, vm, en_rel, pits, talls=(), horizon=999):
    """Simulate a jump with hold h from now; 0 = safe, >0 = hazard (bigger = worse)."""
    A, T = JTAB[h]
    v = min(3.0, max(1.5, vm))
    xm = 0.0
    pen = 0
    xl = 0.0
    hit = set()
    for t in range(1, T + 5):
        if t == T:
            xl = xm + min(3.0, v + 0.05)
        tt = min(t, T)
        hgt = A * 4.0 * (tt / T) * (1 - tt / T) if t < T else 0.0
        v = min(3.0, v + 0.05)
        xm += v
        for ei, (gap, ew) in enumerate(en_rel):
            xe = gap - 0.7 * t
            if xm + 1 > xe and xm - 13 < xe + ew and hgt < 20:
                pen += 100 - t
    for (gp, w) in pits:
        if xl > gp - 6 and xl - 12 < gp + w + 6:
            pen += 1000
    for gw in talls:
        if xl > gw - 36 and xl < gw + 44:
            pen += 500
    if P['hz'] == 0:
        if xl > horizon - 12:
            pen += 300
    elif xl > horizon + 32:
        pen += 40
    return pen


def ceiling_space(s, x0, x1, top):
    """free pixels between Mario's head and the nearest floating orange block above (999 if none)."""
    xa, xb = max(0, x0), min(256, x1 + 30)
    if top <= 44 or xb <= xa:
        return 999
    reg = s[40:top, xa:xb]
    orange = (reg[..., 0] == 228) & (reg[..., 1] == 92) & (reg[..., 2] == 16)
    rowhit = np.zeros(orange.shape[0], dtype=bool)
    for ri in range(orange.shape[0]):
        r_ = orange[ri]
        if r_.sum() < 12:
            continue
        run_, mx_ = 0, 0
        for v_ in r_:
            run_ = run_ + 1 if v_ else 0
            if run_ > mx_:
                mx_ = run_
        rowhit[ri] = mx_ >= 12
    best = 999
    n = len(rowhit)
    i = n - 1
    while i >= 0:
        if rowhit[i]:
            space = n - 1 - i
            if space >= 8:
                best = min(best, space)
            while i >= 0 and rowhit[i]:
                i -= 1
        else:
            i -= 1
    return best


def find_goombas(nb, ylo=100, yhi=200):
    """nb: bool non-black mask (H,W). returns list of (x0,y0) of goomba-shaped blobs."""
    sub = nb[ylo:yhi + GH, :]
    win = sliding_window_view(sub, (GH, GW))  # (Y, X, GH, GW)
    on = win[:, :, G_ON].mean(axis=2)
    cand = np.argwhere(on >= 0.93)
    res = []
    for (yy, xx) in cand:
        w = win[yy, xx]
        if w[G_OFF].mean() > 0.35:
            continue
        res.append((int(xx), int(yy + ylo)))
    out = []
    for (x, y) in sorted(res):
        if all(abs(x - a) > 6 or abs(y - b) > 6 for a, b in out):
            out.append((x, y))
    return out


def find_green(s):
    g = (s[..., 0] == 0) & (s[..., 1] == 168) & (s[..., 2] == 0)
    g[:40] = False
    cols = np.nonzero(g.any(axis=0))[0]
    blobs = []
    if len(cols) == 0:
        return blobs
    start = cols[0]
    prev = cols[0]
    runs = []
    for c in cols[1:]:
        if c - prev > 2:
            runs.append((start, prev))
            start = c
        prev = c
    runs.append((start, prev))
    for (a, b) in runs:
        rows = np.nonzero(g[:, a:b + 1].any(axis=1))[0]
        blobs.append((int(a), int(b), int(rows.min()), int(rows.max())))
    return blobs


def play(sim):
    KEYS = sim.KEYS
    prev_mario = None
    y_hist = []
    hold = 0
    cool = 0
    prev_scr = None
    static = 0
    stuck = 0
    f = 0
    last_log = -100
    prev_cx = None
    vm_hist = []
    jf = 0
    retreat = False
    retreat_f = 0
    n_retreat = 0
    og_prev = False
    free_until = 0
    prog_wx = 0.0
    last_plan_log = -100
    nomario = 0
    wx = 0.0
    prev_x1 = None
    airborne = False
    last_hold = 0
    j_f = j_wx = j_apex = 0
    j_h = 0
    try:
        while sim.budget_left > 2:
            s = sim.screen()
            f += 1
            core, skin = mario_masks(s)
            if P['mask'] == 0 and wx < P['acc_wx']:
                core = core | skin
                skin = np.zeros_like(skin)
            ys, xs = np.nonzero(core)
            keys = {"Right", "x"}
            if len(xs) == 0:
                nomario += 1
                if nomario > 88:
                    break
                sim.step(keys)
                continue
            nomario = 0
            # restrict to cluster near previous mario
            if prev_mario is not None:
                px, py = prev_mario
                sel = (np.abs(xs - px) < 30) & (np.abs(ys - py) < 60)
                if sel.sum() > 3:
                    xs, ys = xs[sel], ys[sel]
            x0, x1, y0, y1 = int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())
            # add adjacent skin pixels (hands / face), nothing farther than 2px from the core box
            ya, yb = max(40, y0 - 2), min(240, y1 + 3)
            xa, xb = max(0, x0 - 2), min(256, x1 + 3)
            sk = skin[ya:yb, xa:xb]
            if sk.any():
                sy, sx = np.nonzero(sk)
                x0 = min(x0, int(sx.min()) + xa)
                x1 = max(x1, int(sx.max()) + xa)
                y0 = min(y0, int(sy.min()) + ya)
                y1 = max(y1, int(sy.max()) + ya)
            prev_mario = ((x0 + x1) // 2, (y0 + y1) // 2)
            y_hist.append(y1)
            if len(y_hist) > 6:
                y_hist.pop(0)
            vy = 0 if len(y_hist) < 2 else y_hist[-1] - y_hist[-2]
            nb = s.sum(axis=2) > 0
            sup = nb[min(y1 + 1, 239):min(y1 + 2, 240), x0:x1 + 1].any()
            on_ground = bool(sup) and vy >= 0 and len(y_hist) >= 2
            xf = x1 + 2
            top = y1 - 15

            # scroll estimate from ground band
            scroll = 0
            if prev_scr is not None:
                a_ = s[212:236].astype(np.int16)
                b_ = prev_scr[212:236].astype(np.int16)
                bestv = None
                for sh in range(0, 6):
                    mm_ = int((a_[:, :256 - sh] != b_[:, sh:]).any(axis=2).sum())
                    if bestv is None or mm_ < bestv:
                        bestv, scroll = mm_, sh
            cx_ = float(x1)
            if prev_cx is not None:
                vm_hist.append(scroll + cx_ - prev_cx)
                if len(vm_hist) > 4:
                    vm_hist.pop(0)
            prev_cx = cx_
            vm_w = sum(vm_hist) / len(vm_hist) if vm_hist else 0.0

            wx += scroll + (x1 - prev_x1) if prev_x1 is not None else 0
            prev_x1 = x1
            if not on_ground:
                if not airborne:
                    airborne = True
                    j_f, j_wx, j_apex = f, wx, y1
                    j_h = last_hold
                j_apex = min(j_apex, y1)
            elif airborne:
                airborne = False

            # progress / stuck tracking
            if f % 100 == 0:
                if wx - prog_wx < 8:
                    free_until = f + 120
                    print('STUCK -> free mode at f', f)
                prog_wx = wx

            # static screen detection (death / end)
            if prev_scr is not None and np.array_equal(prev_scr, s):
                static += 1
            else:
                static = 0
            prev_scr = s
            if static > 90:
                break

            trig = None
            # ---- pits
            pits = []
            row = nb[212:236, :]
            pitcol = ~row.any(axis=0)
            x = xf + 1
            while x < 256:
                if pitcol[x]:
                    x2 = x
                    while x2 < 256 and pitcol[x2]:
                        x2 += 1
                    w_ = x2 - x
                    if w_ >= 5:
                        pits.append((x - xf, 40 if x2 >= 256 else w_))
                    x = x2
                else:
                    x += 1
            if on_ground and pits and pits[0][0] <= 22:
                trig = 'pit%d' % pits[0][0]
            # ledge: standing on an elevated platform whose edge is about to end with a pit beyond
            if P['ledge'] and on_ground and trig is None and pits and y1 < 200:
                xc_l = (x0 + x1) // 2
                row_l = nb[min(y1 + 1, 239)]
                ext_l = 0
                xx_ = xc_l
                while xx_ < 256 and row_l[xx_]:
                    ext_l += 1
                    xx_ += 1
                if ext_l <= 12 and pits[0][0] <= ext_l + 90:
                    trig = 'pit%d' % pits[0][0]
            # ---- walls
            if on_ground and trig is None:
                body = nb[y1 - 13:y1 - 1, :]
                cnt = body.sum(axis=0)
                wallc = cnt >= 9
                run = 0
                for x in range(xf + 1, min(xf + 60, 255)):
                    if wallc[x]:
                        run += 1
                        if run >= 7:
                            gap = x - run + 1 - xf
                            xc_ = min(255, x - run + 1 + 4)
                            hh_ = 0
                            yy2 = y1
                            while yy2 > 44 and nb[yy2, xc_]:
                                hh_ += 1
                                yy2 -= 1
                            lim = 42 if hh_ >= 44 else WALL_GAP
                            if gap <= lim:
                                if hh_ >= 44 and vm_w < 2.4 and not retreat and on_ground and n_retreat < 4 * P['retreat']:
                                    retreat = True
                                    retreat_f = f
                                    n_retreat += 1
                                    print('RETREAT at f', f, 'gap', gap, 'vm', round(vm_w, 2))
                                else:
                                    trig = 'wall%d' % gap
                            break
                    else:
                        run = 0
            # ---- enemies
            enemies = []
            gl = find_goombas(nb)
            for (gx, gy) in gl:
                enemies.append((gx, gx + GW - 1, gy, gy + GH - 1, 'g'))
            for (a, b, c, d) in find_green(s):
                if b - a <= 17 and d - c <= 26 and not (a >= x0 - 4 and b <= x1 + 4 and False):
                    enemies.append((a, b, c, d, 'k'))
            for (a, b, c, d, kind) in enemies:
                gap = a - xf
                if b < x0 - 2:
                    continue
                if d < y1 - 30 or c > y1 + 12:
                    continue
                if gap <= P['en_gap'] and gap >= -12:
                    if on_ground and trig is None:
                        trig = 'en%s%d' % (kind, gap)

            if f < free_until and on_ground and trig is None and f % 30 == 0:
                trig = 'free'

            if retreat:
                tg_ = tall_walls(nb, y1, xf)
                g0_ = min(tg_) if tg_ else 999
                if g0_ >= 95 or f - retreat_f > 50 or not on_ground and f - retreat_f > 3:
                    retreat = False
                else:
                    og_prev = on_ground
                    hold = 0
                    sim.step({'Left', 'x'})
                    continue

            # ---- air steering: avoid landing right in front of an enemy
            if not on_ground and vy > 0:
                colx = (x0 + x1) // 2
                floor_y = 240
                for yy in range(y1 + 2, 240):
                    if nb[yy, colx] or nb[yy, x0] or nb[yy, x1]:
                        floor_y = yy
                        break
                yy_, vv_, tl = float(y1), float(vy), 0
                while yy_ < floor_y - 1 and tl < 80:
                    vv_ = min(5.0, vv_ + 0.9)
                    yy_ += vv_
                    tl += 1
                danger = False
                for (a, b, c, d, kind) in enemies:
                    if d < floor_y - 26 or c < floor_y - 40:
                        continue
                    gap = a - xf
                    if gap < -14 or gap > 90:
                        continue
                    veff = min(3.0, max(vm_w, 0.0) + 0.06 * tl)
                    gl_ = gap - (veff + 0.7) * tl
                    if -2 <= gl_ <= 40:
                        danger = True
                if floor_y < 240 and pits and P['pit_steer']:
                    veff = min(3.0, max(vm_w, 0.0) + 0.06 * tl)
                    xl = veff * tl
                    for (gp, w_) in pits:
                        if xl > gp + 6 and xl - 12 < gp + w_ - 4:
                            danger = True
                if danger and P['steer']:
                    keys.discard('Right')
                    keys.add('Left')

            # ---- jump state machine
            if cool > 0:
                cool -= 1
            if hold > 0:
                jf += 1
                if (vy > 0 and jf >= 3) or (on_ground and jf >= 3):
                    hold = 0
                    cool = 1
                else:
                    hold -= 1
                    keys.add('z')
                    if hold == 0:
                        cool = 1
            elif trig is not None and on_ground and (og_prev or (P['pitland'] and str(trig).startswith('pit'))) and cool == 0:
                kind_ = str(trig)[:2]
                en_rel = [(e[0] - xf, e[1] - e[0] + 1) for e in enemies if e[3] >= y1 - 30 and e[1] >= x0 - 2]
                if kind_ == 'en':
                    cands = ORDERS[P['order']]
                elif kind_ == 'pi':
                    cands = PORDERS[P['porder']]
                else:
                    cands = []
                hold = 24
                talls_ = tall_walls(nb, y1, xf) if cands else []
                best_pen = None
                for h_ in cands:
                    p_ = plan_penalty(h_, vm_w, en_rel, pits, talls_, 255 - xf)
                    if best_pen is None or p_ < best_pen:
                        best_pen, hold = p_, h_
                    if p_ == 0:
                        break
                space = ceiling_space(s, x0, x1, top)
                if f < free_until:
                    space = 999
                    hold = 24
                if space < 999:
                    ok_h = [h_ for h_ in (24, 18, 14, 8, 4) if JTAB[h_][0] <= space - 3 and h_ <= hold]
                    new_h = ok_h[0] if ok_h else 4
                    hold = new_h
                jf = 0
                last_hold = hold
                keys.add('z')
                hold -= 1
            if VERBOSE and f - last_log >= 40:
                last_log = f
                print('f', f, 'mario', (x0, x1, y0, y1), 'trig', trig, 'g', on_ground, 'vm', round(vm_w, 2), 'sc', scroll)
            if DBG[0] <= f <= DBG[1]:
                print('D f', f, 'box', (x0, x1, y0, y1), 'g', int(on_ground), 'vy', vy, 'trig', trig, 'hold', hold, 'pits', pits[:2], 'en', [(e[0] - xf, e[2], e[4]) for e in enemies][:5], 'hh', hold, 'keys', ''.join(sorted(k[0] for k in keys)))
            og_prev = on_ground
            sim.step(keys)
    except Exception as e:  # BudgetExceeded etc.
        print('play ended:', type(e).__name__, e, 'f', f, 'wx', int(wx))
        return
    print('play loop ended f', f, 'static', static, 'nomario', nomario, 'wx', int(wx))


def solve(sim):
    return []

