# ==================================================================================================
# Super Mario Bros. (NES) -- level 4-1  |  超级马里奥兄弟（NES） 关 4-1
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1419 frames (23.65 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1419 帧（60 帧/秒，约 23.65 秒）。
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
#   c9 vision policy + (a) no castle/start false obstacles, (b) fly over 2-3 tile plant pipes with
#    a max-height jump timed from pipe distance instead of waiting for the plant.
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _mask_eq                       line 17    
#   def _nonsolid_px                   line 21    
#   def find_mario                     line 25    
#   def hold_for_rise                  line 38    
#   def hold_for_travel                line 45    
#   def ground_at                      line 52    
#   def enemy_groups                   line 64    narrow non-green solid groups on the ground row ahead: list of (xcenter, width)
#   def travel                         line 92    
#   def rear_groups                    line 96    narrow non-green solid groups just behind Mario
#   def pick_landing                   line 113   smallest hold >= n_min whose predicted landing spot has ground below and no enemy
#   def plan_jump                      line 136   pick smallest hold n>=n_min whose simulated arc clears every enemy and lands on ground; None if none
#   def find_egg                       line 163   centroid (x, y) of red projectile pixels above Mario's head, or None
#   def egg_dodge                      line 177   predicted landing x of a falling egg (None if unknown); hist = [(t,x,y)]
#   def support_edge                   line 209   first x right of Mario where the support under his feet ends (None if none on screen)
#   def col_top                        line 218   height (px above feet) of contiguous solid column at cx starting from feet-1 upward
#   def ahead_info                     line 230   returns dict with wall (dist), wall_h, green, plant, pit (dist), pit_w
#   def plant_e                        line 317   extension (px) of the piranha plant above a pipe whose body left edge is at xw
#   def tp_go_ok                       line 336   True if a plant with extension history hist (list of (t,e)) is predicted low while Mario crosses
#   def play                           line 397   robust wrapper: restart the policy on unexpected errors, never crash the run
#   def _play_inner                    line 416   
#   def solve                          line 738   
# ==================================================================================================

# idea: c9 vision policy + (a) no castle/start false obstacles, (b) fly over 2-3 tile plant pipes with a max-height jump timed from pipe distance instead of waiting for the plant.
import numpy as np

SKY = (104, 136, 252)
NONSOLID = [(104, 136, 252), (252, 252, 252), (60, 188, 252)]
GREENISH = [(0, 168, 0), (184, 248, 24), (0, 0, 0)]
DEBUG = []
P = {"W0": 45, "MAXH": 56, "TW": 64, "TP": False, "HILL": False, "W0T": 50, "TR": 16, "D0": 40, "WMIN": 70,
     "WLO": 100, "WHI": 125}

# (hold_frames_extra n, airtime frames, apex rise px) measured at full run speed
JT = [(0, 19, 29), (3, 23, 41), (6, 28, 52), (9, 32, 61), (12, 35, 68),
      (16, 40, 76), (20, 44, 82), (26, 49, 85)]
VX = 2.95


def _mask_eq(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def _nonsolid_px(p):
    return (int(p[0]), int(p[1]), int(p[2])) in NONSOLID


def find_mario(s):
    m = _mask_eq(s[:208], (172, 124, 0))
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    mxm = np.median(xs)
    sel = np.abs(xs - mxm) < 14
    xs, ys = xs[sel], ys[sel]
    if len(xs) < 4:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def hold_for_rise(rise):
    for n, a, r in JT:
        if r >= rise:
            return n
    return JT[-1][0]


def hold_for_travel(need, v0):
    for n, a, r in JT:
        if travel(a, v0) >= need:
            return n
    return JT[-1][0]


def ground_at(s, xc):
    W = s.shape[1]
    lo, hi = int(xc) - 5, int(xc) + 6
    if lo >= W:
        return True
    lo = max(lo, 0)
    hi = min(hi, W)
    row = s[230, lo:hi]
    sky = (row[:, 0] == SKY[0]) & (row[:, 1] == SKY[1]) & (row[:, 2] == SKY[2])
    return not sky.any()


def enemy_groups(s, right, feet):
    """narrow non-green solid groups on the ground row ahead: list of (xcenter, width)"""
    W = s.shape[1]
    x0 = min(right + 2, W - 1)
    y0, y1 = max(feet - 16, 30), max(feet - 1, 31)
    reg = s[y0:y1, x0:W]
    if reg.size == 0:
        return []
    ng = np.ones(reg.shape[:2], dtype=bool)
    for c in NONSOLID + GREENISH:
        ng &= ~_mask_eq(reg, c)
    cols = ng.sum(axis=0) >= 2
    groups = []
    i = 0
    n = len(cols)
    while i < n:
        if cols[i]:
            j = i
            while j < n and cols[j]:
                j += 1
            if j - i <= 20:
                groups.append((x0 + (i + j) / 2.0, j - i))
            i = j
        else:
            i += 1
    return groups


def travel(a, v0):
    return min(VX * a, v0 * a + 0.05 * a * a)


def rear_groups(s, x0, feet):
    """narrow non-green solid groups just behind Mario"""
    xa = max(0, x0 - 45)
    xb = max(0, x0 - 3)
    y0, y1 = max(feet - 14, 30), max(feet - 1, 31)
    reg = s[y0:y1, xa:xb]
    if reg.size == 0:
        return []
    ng = np.ones(reg.shape[:2], dtype=bool)
    for c in NONSOLID + GREENISH:
        ng &= ~_mask_eq(reg, c)
    cols = ng.sum(axis=0) >= 2
    if cols.sum() >= 3 and cols.sum() <= 22:
        return [1]
    return []


def pick_landing(s, xc0, n_min, vx, groups=()):
    """smallest hold >= n_min whose predicted landing spot has ground below and no enemy"""
    last = n_min
    best_ground = None
    for n, a, r in JT:
        if n < n_min:
            continue
        last = n
        lx = xc0 + travel(a, vx)
        if not ground_at(s, lx):
            continue
        if best_ground is None:
            best_ground = n
        bad = False
        for ex, ew in groups:
            if abs(lx - (ex - 0.8 * a)) < 9 + ew / 2.0:
                bad = True
                break
        if not bad:
            return n
    return best_ground if best_ground is not None else last


def plan_jump(s, xc0, v0, h0, n_min, groups):
    """pick smallest hold n>=n_min whose simulated arc clears every enemy and lands on ground; None if none"""
    for n, a, r in JT:
        if n < n_min:
            continue
        ok = True
        landed = False
        for t in range(1, 130):
            u = t / float(a)
            hh = h0 + 4.0 * r * u * (1.0 - u)
            xm = xc0 + travel(t, v0)
            for ex, ew in groups:
                if abs(xm - (ex - 0.8 * t)) < 10 + ew / 2.0 and hh < 19:
                    ok = False
                    break
            if not ok:
                break
            if hh <= 0 and t > 3:
                landed = True
                if not ground_at(s, xm):
                    ok = False
                break
        if ok and landed:
            return n
    return None


def find_egg(s, xc, ytop):
    """centroid (x, y) of red projectile pixels above Mario's head, or None"""
    xa, xb = max(0, xc - 70), min(256, xc + 70)
    ya, yb = 34, max(35, ytop)
    reg = s[ya:yb, xa:xb]
    if reg.size == 0:
        return None
    m = _mask_eq(reg, (248, 56, 0))
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    return (xa + float(xs.mean()), ya + float(ys.mean()))


def egg_dodge(hist, xc, ymid):
    """predicted landing x of a falling egg (None if unknown); hist = [(t,x,y)]"""
    if len(hist) < 4:
        return None
    ts = np.array([h[0] for h in hist], dtype=float)
    xs = np.array([h[1] for h in hist], dtype=float)
    ys = np.array([h[2] for h in hist], dtype=float)
    ts = ts - ts[-1]
    if ys[-1] - ys[0] < 2.0:
        return None
    try:
        a, b, c = np.polyfit(ts, ys, 2)
        px = np.polyfit(ts, xs, 1)
    except Exception:
        return None
    th = None
    if a > 1e-4:
        disc = b * b - 4 * a * (c - ymid)
        if disc >= 0:
            r = (-b + disc ** 0.5) / (2 * a)
            if r >= 0:
                th = r
    if th is None:
        vy = (ys[-1] - ys[0]) / max(1.0, ts[-1] - ts[0])
        if vy < 0.3:
            return None
        th = (ymid - ys[-1]) / vy
    if th < 0 or th > 90:
        return None
    return float(px[0] * th + px[1])


def support_edge(s, feet, x1):
    """first x right of Mario where the support under his feet ends (None if none on screen)"""
    row = s[min(feet, 239), x1:]
    for i in range(len(row)):
        if _nonsolid_px(row[i]):
            return x1 + i
    return None


def col_top(s, cx, feet, maxh=100):
    """height (px above feet) of contiguous solid column at cx starting from feet-1 upward"""
    H = s.shape[0]
    cx = min(cx, s.shape[1] - 1)
    y = min(feet - 1, 207)
    h = 0
    while y > 40 and h < maxh and not _nonsolid_px(s[y, cx]):
        h += 1
        y -= 1
    return h


def ahead_info(s, right, feet, T_wall, T_pit):
    """returns dict with wall (dist), wall_h, green, plant, pit (dist), pit_w"""
    H, W, _ = s.shape
    x0 = min(right + 2, W - 1)
    x1 = min(right + 2 + T_wall, W)
    y0 = max(feet - 22, 30)
    y1 = max(min(feet - 1, 207), y0 + 1)
    out = {"wall": None, "wall_h": 0, "green": False, "plant": False, "pit": None, "pit_w": 0}
    wall = None
    green_wall = False
    if x1 > x0:
        reg = s[y0:y1, x0:x1]
        non = np.ones(reg.shape[:2], dtype=bool)
        for c in NONSOLID:
            non &= ~_mask_eq(reg, c)
        h = non.sum(axis=0)
        for c in range(0, len(h)):
            if c >= 3:
                if h[c] >= 10 and h[c - 3] <= 3:
                    wall = c
                    break
            elif h[c] >= 19:
                wall = c
                break
        green_wall = wall is not None
        ng = non.copy()
        for c in GREENISH:
            ng &= ~_mask_eq(reg, c)
        nc = ng.sum(axis=0)
        idx = np.nonzero(nc >= 3)[0]
        if len(idx) and (wall is None or idx[0] < wall + 2):
            wall = int(idx[0])
            green_wall = False
    if wall is not None:
        xw = x0 + wall
        hh = 0
        for dx in (3, 5, 7, 9):
            hh = max(hh, col_top(s, xw + dx, feet))
        if not green_wall:
            hh = max(hh, 18)
        out["wall"] = wall
        out["wall_h"] = hh
        out["green"] = green_wall
        try:
            pr = s[max(feet - 14, 0):max(feet - 2, 1), xw + 2:min(xw + 10, W)]
            gcnt = int((_mask_eq(pr, (0, 168, 0)) | _mask_eq(pr, (184, 248, 24))).sum()) if pr.size else 0
            ct3 = col_top(s, min(xw + 3, W - 1), feet)
            ct5 = col_top(s, min(xw + 5, W - 1), feet)
            ct28 = col_top(s, min(xw + 20, W - 1), feet)
            out["pipe"] = bool(gcnt >= 20 and ct3 >= 26 and abs(ct3 - ct5) <= 2
                               and (wall < 4 or abs(ct3 - ct28) <= 3))
            ct9 = col_top(s, min(xw + 9, W - 1), feet)
            out["hill"] = bool(green_wall and not out["pipe"] and ct9 - ct3 >= 4 and ct3 <= 28)
            out["ph"] = max(ct3, ct5)
            out["ct"] = (ct3, ct5, ct28)
        except Exception:
            out["pipe"] = False
        if green_wall and hh >= 26:
            top = feet - 1 - col_top(s, xw + 3, feet)
            py0 = max(top - 28, 30)
            px0, px1 = min(xw + 7, W - 1), min(xw + 25, W)
            if top > py0 and px1 > px0:
                reg = s[py0:top, px0:px1]
                non = np.ones(reg.shape[:2], dtype=bool)
                for c in NONSOLID:
                    non &= ~_mask_eq(reg, c)
                out["plant_cnt"] = int(non.sum())
                out["plant"] = int(non.sum()) >= 25
    # pit
    xp1 = min(right + 2 + T_pit, W)
    if xp1 > x0:
        row = s[230, x0:W]
        sky = (row[:, 0] == SKY[0]) & (row[:, 1] == SKY[1]) & (row[:, 2] == SKY[2])
        idx = np.nonzero(sky[:xp1 - x0])[0]
        if len(idx):
            p0 = int(idx[0])
            p1 = p0
            while p1 < len(sky) and sky[p1]:
                p1 += 1
            width = p1 - p0
            if p1 >= len(sky):
                width = 200
            out["pit"] = p0
            out["pit_w"] = width
    return out


def plant_e(s, xw, feet, ph):
    """extension (px) of the piranha plant above a pipe whose body left edge is at xw"""
    ytop = feet - ph
    xa, xb = xw + 9, min(xw + 25, s.shape[1])
    if xa >= xb or xa < 0:
        return 0
    e = 0
    for y in range(ytop - 1, max(ytop - 32, 30), -1):
        row = s[y, xa:xb]
        ns = np.zeros(len(row), dtype=bool)
        for c in NONSOLID:
            ns |= (row[:, 0] == c[0]) & (row[:, 1] == c[1]) & (row[:, 2] == c[2])
        if int((~ns).sum()) >= 3:
            e += 1
        else:
            break
    return e


def tp_go_ok(hist, Ta):
    """True if a plant with extension history hist (list of (t,e)) is predicted low while Mario crosses
    (crossing window = Ta+14 .. Ta+36 frames from now)"""
    if len(hist) < 8:
        return False
    RS = 0.5
    e = hist[-1][1]
    e6 = hist[-7][1]
    de = e - e6
    es = [h[1] for h in hist]
    if de <= -2:
        tau0 = e / RS

        def f(tau):
            if tau < tau0:
                return e - RS * tau
            if tau < tau0 + 60:
                return 0.0
            return min(23.0, RS * (tau - tau0 - 60))
    elif de >= 2 or 2 < e < 20:
        tau1 = max(0.0, (23 - e) / RS)

        def f(tau):
            if tau < tau1:
                return e + RS * tau
            if tau < tau1 + 56:
                return 23.0
            return max(0.0, 23.0 - RS * (tau - tau1 - 56))
    elif e >= 20:
        obs = 0
        for v in reversed(es):
            if v >= 20:
                obs += 1
            else:
                break
        rem = max(0, 56 - obs)

        def f(tau):
            if tau < rem:
                return 23.0
            return max(0.0, 23.0 - RS * (tau - rem))
    else:
        obs = 0
        for v in reversed(es):
            if v <= 1:
                obs += 1
            else:
                break
        known = obs < len(es)
        rem = max(0.0, 64 - obs) if known else 0.0

        def f(tau):
            if tau < rem:
                return 0.0
            return min(23.0, RS * (tau - rem))
    worst = 0.0
    for tau in range(int(Ta) + 14, int(Ta) + 37):
        worst = max(worst, f(tau))
    return worst <= 11.0


def play(sim):
    """robust wrapper: restart the policy on unexpected errors, never crash the run"""
    for attempt in range(6):
        try:
            return _play_inner(sim)
        except Exception as e:  # noqa
            if type(e).__name__ == "BudgetExceeded":
                raise
            DEBUG.append("EXC attempt %d: %r" % (attempt, e))
    # last resort: dumb runner
    i = 0
    while sim.budget_left > 0:
        k = {"Right", "x"}
        if i % 30 < 14:
            k.add("z")
        sim.step(k)
        i += 1


def _play_inner(sim):
    prev = sim.screen()
    prev_row = prev[215].copy()
    ys_hist = []
    hold = 0       # frames A still to be held
    cool = 0       # frames A must be released
    still = 0
    frozen = 0
    last_x = None
    t = 0
    wait_cnt = 0
    elev_wait = 0
    egg_hist = []
    egg_miss = 0
    dodge_dir = 0
    dodge_cnt = 0
    last_wait_t = -999
    run_frames = 0
    pipe_t = -999
    tp_mode = 0
    tp_hist = []
    tp_off = False
    tp_lost = 0
    tp_wait = 0
    tp_fw_prev = None
    while sim.budget_left > 0:
        s = sim.screen()
        t += 1
        diff = int(np.abs(s[::4, ::4].astype(np.int16) - prev[::4, ::4].astype(np.int16)).sum())
        if diff == 0:
            frozen += 1
        else:
            frozen = 0
        if frozen > 30:
            DEBUG.append("frozen end at t=%d" % t)
            return
        m = find_mario(s)
        keys = {"Right", "x"}
        if m is None:
            sim.step(keys)
            prev = s
            continue
        x0, x1, y0, y1 = m
        feet = y1 + 1
        ys_hist.append(feet)
        if len(ys_hist) > 6:
            ys_hist.pop(0)
        sup = False
        if feet < 238:
            rowb = s[feet, max(x0, 0):min(x1 + 1, 256)]
            nons = np.zeros(len(rowb), dtype=bool)
            for c in NONSOLID:
                nons |= (rowb[:, 0] == c[0]) & (rowb[:, 1] == c[1]) & (rowb[:, 2] == c[2])
            sup = bool((~nons).any())
        grounded = sup
        row = s[215]
        scrolled = not np.array_equal(row, prev_row)
        moved = scrolled or (last_x is not None and x1 != last_x)
        last_x = x1
        prev_row = row.copy()
        still = 0 if moved else still + 1
        run_frames = run_frames + 1 if moved else 0
        vx = min(VX, 0.1 * run_frames + 0.2)
        right = x1 + 3
        info = ahead_info(s, right, feet, P["TW"], 24)
        if t <= 80 or (P["HILL"] and info.get("hill") and feet >= 200):
            # the start castle / green hills are background, not obstacles
            info["wall"] = None
            info["wall_h"] = 0
            info["green"] = False
            info["plant"] = False
        wall, pit = info["wall"], info["pit"]
        egg = find_egg(s, right - 8, y0 - 8)
        if egg is not None:
            egg_hist.append((t, egg[0], egg[1]))
            egg_hist = egg_hist[-7:]
            egg_miss = 0
        else:
            egg_miss += 1
            if egg_miss > 2:
                egg_hist = []
        xh = egg_dodge(egg_hist, right - 8, y0 + 8)
        xcm = right - 8
        if xh is None:
            dodge_dir = 0
            dodge = 0
        else:
            dd = abs(xh - xcm)
            if dd >= 27:
                dodge = 0
            else:
                if dodge_dir == 0:
                    away = 1 if xh <= xcm else -1
                    room_r = wall if wall is not None else 99
                    if away > 0 and room_r < (27 - dd):
                        away = -1
                    dodge_dir = away
                dodge = dodge_dir
        if dodge != 0 and len(DEBUG) < 600 and t % 4 == 0:
            DEBUG.append("EGG t=%d egg=%s xh=%s dodge=%d" % (t, egg, xh, dodge))
        # ---- tall (4-tile) pipes: time the run-up so that we fly over the plant while it is low ----
        tp_jump = False
        if P["TP"] and t > 80:
            if hold > 0 and tp_mode == 2:
                tp_mode = 0
                tp_hist = []
                tp_off = True
            if grounded and feet >= 200 and hold == 0 and cool == 0:
                far = ahead_info(s, right, feet, 150, 24)
                fw = far["wall"]
                ph_f = far.get("ph", 0)
                tall = bool(fw is not None and far.get("pipe") and 60 <= ph_f <= 74)
                if not tall:
                    tp_lost += 1
                    if tp_lost > 4:
                        tp_mode = 0
                        tp_hist = []
                        tp_off = False
                        tp_wait = 0
                elif not tp_off:
                    tp_lost = 0
                    xw_f = right + 2 + fw
                    e_now = plant_e(s, xw_f, feet, ph_f)
                    tp_hist.append((t, e_now))
                    tp_hist = tp_hist[-60:]
                    if tp_mode == 2:
                        if fw <= P["W0T"] and run_frames >= 16:
                            tp_jump = True
                            DEBUG.append("TPJUMP t=%d fw=%d rf=%d e=%d" % (t, fw, run_frames, e_now))
                    else:
                        if run_frames >= 20:
                            Ta = max(0.0, (fw - P["W0T"]) / 3.0)
                        else:
                            Ta = P["TR"] + max(0.0, (fw - P["W0T"] - P["D0"]) / 3.0)
                        ok = tp_go_ok(tp_hist, Ta)
                        dfw = (fw - tp_fw_prev) if tp_fw_prev is not None else 0
                        tp_fw_prev = fw
                        if len(tp_hist) < 8 and fw > P["W0T"] + 45:
                            pass    # still observing the plant, keep running
                        elif fw <= P["W0T"] + 10 and run_frames >= 20:
                            # too close to stop safely: hand over to the old plant-wait logic
                            tp_off = True
                            DEBUG.append("TPOFF-close t=%d fw=%d e=%d" % (t, fw, e_now))
                        elif ok and (run_frames >= 20 or fw >= P["WMIN"]):
                            tp_mode = 2
                            DEBUG.append("TPGO t=%d fw=%d rf=%d e=%d Ta=%.0f" % (t, fw, run_frames, e_now, Ta))
                            if fw <= P["W0T"] and run_frames >= 16:
                                tp_jump = True
                        else:
                            tp_wait += 1
                            last_wait_t = t
                            if tp_wait > 420:
                                tp_off = True
                            elif dodge == 0:
                                keys = set()
                                if fw < P["WLO"] or (dfw < -1 and fw < P["WLO"] + 35):
                                    keys.add("Left")
                                elif fw > P["WHI"] and dfw > -1:
                                    keys.add("Right")
                                tp_fw_prev = fw
                                still = 0
                                run_frames = 0
                                if tp_wait % 10 == 1:
                                    DEBUG.append("TPWAIT t=%d fw=%d e=%d hist=%s" % (
                                        t, fw, e_now, [h[1] for h in tp_hist[-8:]]))
                                sim.step(keys)
                                prev = s
                                continue
        want = False
        why = ""
        nhold = 26
        if info.get("pipe") and wall is not None:
            pipe_t = t
        if wall is not None and t > 40:
            hw = info["wall_h"]
            if info["plant"] and not (info.get("pipe") or (info["green"] and t - pipe_t <= 25)):
                info["plant"] = False   # hills / bushes are not plants
            trig = min(36, 14 + 0.4 * hw)
            ph = info.get("ph", hw)
            fly = (info["green"] and info.get("pipe") and 26 <= ph <= P["MAXH"] and grounded
                   and hold == 0 and cool == 0 and vx >= 2.6 and feet >= 200)
            if info["green"] and hw >= 26 and wall <= 50 and len(DEBUG) < 600 and t % 5 == 0:
                DEBUG.append("PIPE? t=%d wall=%d hw=%d pipe=%s ct=%s vx=%.1f feet=%d" % (
                    t, wall, hw, info.get("pipe"), info.get("ct"), vx, feet))
            fly_n = 12 if ph <= 40 else 20
            fly_w0 = 27 if ph <= 40 else 40
            if fly:
                # only fly if the jump lands on ground without hitting enemies
                gg = enemy_groups(s, right, feet)
                if plan_jump(s, right - 8, vx, 0, fly_n, gg) is None:
                    fly = False
            rear = False
            if (not fly) and wall <= 36 and info["plant"] and grounded and hold == 0 and wait_cnt < 240:
                rg = rear_groups(s, x0, feet)
                rear = len(rg) > 0
            if (not fly) and wall <= 36 and info["plant"] and grounded and hold == 0 and cool == 0 \
                    and wait_cnt < 240 and not rear:
                wait_cnt += 1
                last_wait_t = t
                keys.discard("Right")
                if dodge > 0:
                    keys.add("Right")
                elif dodge < 0:
                    keys.add("Left")
                still = 0
                run_frames = 0
                if wait_cnt % 10 == 1 and len(DEBUG) < 600:
                    DEBUG.append("WAIT plant t=%d wall=%d cnt=%s" % (t, wall, info.get("plant_cnt")))
                sim.step(keys)
                prev = s
                continue
            if fly:
                if wall <= fly_w0:
                    want = True
                    why = "fly%d/h%d" % (wall, hw)
                    nhold = fly_n
            elif wall <= trig:
                want = True
                why = "wall%d/h%d" % (wall, hw)
                nhold = hold_for_rise(hw + (24 if (wall <= 6 and hw >= 26) else 14))
        if wall is None:
            wait_cnt = 0
            if dodge == 0:
                dodge_cnt = 0
        if (dodge != 0 and grounded and hold == 0 and cool == 0
                and ((wall is not None and wall <= 8) or t - last_wait_t < 80)
                and dodge_cnt < 150):
            dodge_cnt += 1
            keys.discard("Right")
            keys.add("Right" if dodge > 0 else "Left")
            still = 0
            run_frames = 0
            sim.step(keys)
            prev = s
            continue
        if pit is not None and pit <= 6:
            want = True
            D = pit + info["pit_w"] + 14
            nhold = hold_for_travel(D, vx)
            why = "pit%d/w%d" % (pit, info["pit_w"])
        if still >= 6 and t > 40 and grounded:
            want = True
            why = "stuck"
            nhold = 26
        if tp_jump and grounded and hold == 0 and cool == 0:
            want = True
            why = "tall"
            nhold = 26
        if want and grounded and hold == 0 and cool == 0:
            groups = enemy_groups(s, right, feet)
            if feet < 195:
                groups = groups + enemy_groups(s, right, 208)
            h0 = max(0, 208 - feet)
            pn = plan_jump(s, right - 8, vx, h0, nhold, groups)
            if pn is not None:
                nhold = pn
            else:
                nhold = pick_landing(s, right - 8, nhold, vx, groups)
        if t % 25 == 0 and len(DEBUG) < 600:
            DEBUG.append("t=%d mario=%s feet=%d g=%d want=%s hold=%d" % (t, m, feet, grounded, why, hold))
        if grounded and feet < 195 and hold == 0 and cool == 0 and elev_wait < 90:
            # standing on a pipe / platform: do not step off onto an approaching enemy
            grp = enemy_groups(s, x0 - 8, 208)
            xc = right - 8
            if any(xc - 30 <= ex <= xc + 110 for ex, ew in grp):
                edge = support_edge(s, feet, x1 + 1)
                dist_edge = (edge - right) if edge is not None else 999
                if dist_edge <= 10:
                    pn = plan_jump(s, xc, vx, max(0, 208 - feet), 0, grp)
                    if pn is not None:
                        keys.add("z")
                        hold = pn
                        if len(DEBUG) < 600:
                            DEBUG.append("ELEVJUMP t=%d n=%d grp=%s" % (t, pn, grp))
                        sim.step(keys)
                        prev = s
                        continue
                    elev_wait += 1
                    still = 0
                    keys.discard("Right")
                    if elev_wait % 10 == 1 and len(DEBUG) < 600:
                        DEBUG.append("ELEVWAIT t=%d grp=%s" % (t, grp))
                    sim.step(keys)
                    prev = s
                    continue
        if grounded and feet >= 195:
            elev_wait = 0
        if (not grounded) and feet <= 192 and len(ys_hist) >= 2 and ys_hist[-1] > ys_hist[-2]:
            grp = enemy_groups(s, x0 - 8, 208)
            f = max(0.0, (203 - feet) / 4.5)
            xc = right - 8
            va = max(vx, 1.0)
            lx = xc + va * f
            lb = xc + max(0.0, va * f - 0.5 * 0.15 * f * f)
            marg_r = marg_b = 999.0
            for ex, ew in grp:
                exf = ex - 0.8 * f
                thr = 13 + ew / 2.0
                marg_r = min(marg_r, abs(lx - exf) - thr)
                marg_b = min(marg_b, abs(lb - exf) - thr)
            if marg_r < 0 and marg_b > marg_r:
                keys.discard("Right")
                keys.add("Left")
                if len(DEBUG) < 600:
                    DEBUG.append("AIRBRAKE t=%d xc=%d f=%.1f mr=%.0f mb=%.0f" % (t, xc, f, marg_r, marg_b))
        if hold > 0:
            keys.add("z")
            hold -= 1
            if hold == 0:
                cool = 2
        else:
            if cool > 0:
                cool -= 1
            elif want and grounded:
                keys.add("z")
                hold = nhold
                if len(DEBUG) < 600:
                    DEBUG.append("JUMP t=%d %s n=%d mario=%s" % (t, why, nhold, m))
        sim.step(keys)
        prev = s


def solve(sim):
    # nothing to learn offline; the policy is purely reactive
    return []

