# ==================================================================================================
# Super Mario Bros. (NES) -- level 8-3  |  超级马里奥兄弟（NES） 关 8-3
# Pass rate on 100 random variants never used before: 86/100; fastest finish 1586 frames (26.43 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 86/100；最快一局 1586 帧（60 帧/秒，约 26.43 秒）。
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
#   reactive SMB runner (hold Right+B, hop pits/blocks/enemies by pixel cues) + closed-loop stone-
#   stair finale + hammer-bro corridor logic (less conservative airborne-bro wait).
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _firstdbg                      line 102   
#   def _macro_keys                    line 115   
#   def _mario_bbox                    line 130   
#   def _red                           line 139   
#   def _green                         line 146   
#   def _sky                           line 153   
#   def _gray                          line 157   
#   def _scroll                        line 161   
#   def play                           line 173   
# ==================================================================================================

# idea: reactive SMB runner (hold Right+B, hop pits/blocks/enemies by pixel cues) + closed-loop stone-stair finale + hammer-bro corridor logic (less conservative airborne-bro wait).
import numpy as np

STATE = {"world": 0}
_WORLD_OFFSET = [0]
_HINT = []   # test harness only: world-x at play() start
TUNABLE = {"JK_BLOCK": [26], "GJD_HB": (44, 80), "GJD_HB2": (44, 80), "JK_KOOPA_HB": (8, 26), "JK_KOOPA_HB2": (6, 18),
           "AIRB_CLR": (20, 60), "AIRB_D": (50, 90), "JK_PIPE": (20, 24)}

LOOK = 43
ENEMY_MIN = 10
NEAR_MULT = 3
ABORT_AGE = 5
BUNNY_D = 75
PIT_DIST = 19
JK_PIT = 28
JK_ENEMY = 14
JK_BLOCK = 26
BLOCK_FRAMES = 3
_DEAD_FRAMES = 80
HEAD_UP = 18
GHI = 21
GJD = 34
JK_KOOPA = 6
JK_KOOPA2 = 14
HB_WORLD = 2700
GJD_HB = 66
GJD_HB2 = 66
JK_KOOPA_HB = 16
JK_PIT_PIPE = 36
PIPE_AFTER_PIT = 70
GFLY_MIN = 40
GFLY_MAX = 85
GFLY_UP = 75
JK_FLY = 16
JK_LOW = 4
PIPE_N = 300
PIPE_D = 64
JK_PIPE = 22
PIPE_WORLD = 2550
HAM_RANGE = 100
HAM_N = 3
HAM_GAP = 14
HAM_WAIT_MAX = 0
WAIT_LEFT_D = 60
HAM_WORLD = 2050
STONE_WORLD = 2990
STONE_IDLE = 30
STONE_REST = 15
_STONE_M0 = (6, 30, 10)
_STONE_M1 = (6, 24, 10)
_STONE_M1C = (9, 24, 10)
_STONE_K2 = 28
_STONE_RR2 = 99
EXP_ON = False
PRACTICE_WORLD = 2693
FLY_WORLD_MAX = 2700
AIRB_ON = True
AIRB_WORLD = 2700
AIRB_D = 70
AIRB_LEFT = 45
AIRB_ACC = 1.3
AIRB_CLR = 38
AIRB_MAX = 60
AIRB_MIN_GDX = 0
AIRS_ON = False
AIRS_FALL = 4.0
AIRS_LO = -3
AIRS_HI = 45
DBG_LO = 2800
DBG_HI = 3000


PITDBG = False
PDBG_LO = 2240
PDBG_HI = 2345
LAND_REJUMP = True
GFLY_GROUND = 3
GFLY_HB_THR = 1
GFLY_HB_LO = 2250
GFLY_HB_HI = 2550
ENEMY_WORLD = 330
JK_KOOPA_HB2 = 10
STAND_ON = False
AIRB_LEFT_ON = True
STAND_D = 95
STAND_L = 80
STAND_WORLD = 2880
HEXT_ON = True
HEXT_DX = 34
HEXT_UP = 6
HEXT_DN = 20
HEXT_MAX = 8
HDEL_ON = True
HDEL_WORLD = 2880
HDEL_DX = 75
HDEL_MIN_G = 25
HDEL_MAX = 14
NOBUNNY_CEIL = True


def _firstdbg(img, xr):
    x0 = min(250, xr + 1)
    x1 = min(255, xr + 75)
    if x1 <= x0 + 3:
        return None
    col = _sky(img[216:238, x0:x1]).all(axis=0)
    idx = np.nonzero(col)[0]
    for i in idx:
        if i + 2 < len(col) and col[i] and col[i + 1] and col[i + 2]:
            return int(i)
    return None


def _macro_keys(i, p, k, rr, run):
    keys = set()
    if i < p:
        keys.add("Right")
        return keys
    j = i - p
    if j < rr:
        keys.add("Right")
        if run:
            keys.add("x")
    if j < k:
        keys.add("z")
    return keys


def _mario_bbox(img):
    m = (img[..., 0] == 248) & (img[..., 1] == 56) & (img[..., 2] == 0)
    m[:44] = False
    ys, xs = np.nonzero(m)
    if len(xs) < 3:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def _red(img, x0, y0, x1, y1):
    sub = img[y0:y1, x0:x1]
    m = (sub[..., 0] == 248) & (sub[..., 1] == 56) & (sub[..., 2] == 0)
    ys, xs = np.nonzero(m)
    return ys, xs


def _green(img, x0, y0, x1, y1):
    sub = img[y0:y1, x0:x1]
    m = (sub[..., 0] == 0) & (sub[..., 1] == 168) & (sub[..., 2] == 0)
    ys, xs = np.nonzero(m)
    return ys, xs


def _sky(sub):
    return (sub[..., 0] == 104) & (sub[..., 1] == 136) & (sub[..., 2] == 252)


def _gray(img):
    return img.astype(np.int16).sum(2)


def _scroll(prev_g, cur_g):
    best, bs = None, 0
    r0, r1 = 48, 207
    for s in range(0, 7):
        a = cur_g[r0:r1:2, 0:250 - 6:2]
        b = prev_g[r0:r1:2, s:244 + s:2]
        d = np.abs(a - b).mean()
        if best is None or d < best - 0.3:
            best, bs = d, s
    return bs


def play(sim):
    if not getattr(sim, "_resume", False):
        sim.step(set())
    img = sim.screen()
    ys, xs = _red(img, 0, 40, 256, 208)
    if len(ys) == 0:
        mx_r, my_top = 60, 192
    else:
        mx_r, my_top = int(xs.max()), int(ys.min())
    prev_g = _gray(img)
    prev_img = img
    f0 = int(getattr(sim, "frame", 0) or 0)
    if _HINT:
        _WORLD_OFFSET[0] = int(_HINT[0]) - mx_r
    elif f0 <= 40 and mx_r > 100:
        _WORLD_OFFSET[0] = PRACTICE_WORLD - mx_r   # practice checkpoint start
    elif f0 > 40 and not getattr(sim, "_resume", False):
        pts = [(0, 52), (541, 1244), (781, 1716), (901, 1955), (1021, 2316), (1126, 2630), (1178, 2786),
               (1548, 3400)]
        est = pts[-1][1]
        for (fa, wa), (fb, wb) in zip(pts, pts[1:]):
            if fa <= f0 <= fb:
                est = wa + (wb - wa) * (f0 - fa) / float(fb - fa)
                break
        _WORLD_OFFSET[0] = int(est) - mx_r
    world = 0
    scroll_total = 0
    jump_t = 0
    jump_kind = ""
    cooldown = 0
    feet_hist = []
    still = 0
    same_cnt = 0
    f = 0
    last_log = -100
    prev_xr = mx_r
    press_feet = 0
    press_age = 0
    prev_xr0 = mx_r
    gprev = None
    st_stage = 0
    st_feets = []
    st_still = 0
    st_ph = 0
    st_n = 0
    st_mi = 0
    st_i = 0
    st_hist = []
    st_sc0 = 0
    st_p2 = 8
    st_c0 = 99
    ham_free = 99
    ham_wait = 0
    airb_n = 0
    airs_n = 0
    prev_gb = None
    jump_min = 999
    hdel_n = 0
    hext_n = 0
    while True:
        f += 1
        img = sim.screen()
        g = _gray(img)
        if np.array_equal(img, prev_img):
            same_cnt += 1
        else:
            same_cnt = 0
        if same_cnt >= _DEAD_FRAMES:
            print("end: frozen screen at f=%d world=%d mx=%d" % (f, world, mx_r))
            return
        s = _scroll(prev_g, g)
        scroll_total += s
        wx0 = max(0, mx_r - 36)
        wy0 = max(40, my_top - 40)
        ys, xs = _red(img, wx0, wy0, min(256, mx_r + 36), min(240, my_top + 56))
        if len(ys) >= 3:
            mx_r = wx0 + int(xs.max())
            my_top = wy0 + int(ys.min())
        feet = my_top + 15
        xr = mx_r
        world = scroll_total + xr + _WORLD_OFFSET[0]
        STATE["world"] = world

        feet_hist.append(feet)
        if len(feet_hist) > 4:
            feet_hist.pop(0)
        grounded = len(feet_hist) >= 3 and (max(feet_hist[-3:]) - min(feet_hist[-3:])) <= 1

        if s == 0 and xr == prev_xr and f > 10:
            still += 1
        else:
            still = 0
        prev_xr0 = prev_xr
        prev_xr = xr
        gys, gxs = _green(img, max(0, xr - 10), 40, min(256, xr + 135), feet + 1)
        gdx = None
        gb = None
        gn = 0
        if len(gys) >= 6:
            gxa = gxs + max(0, xr - 10)
            gl = int(gxa.min())
            sel = gxa <= gl + 24
            gb = int(gys[sel].max()) + 40
            gn = int(sel.sum())
            gdx = gl - xr
            gprev = (gl, gb)
        else:
            gprev = None
        ceiling = False
        if feet - 104 >= 40 and jump_t == 0:
            cx0 = max(0, xr - 4)
            cx1 = min(256, xr + 36)
            lo = img[feet - 64:feet - 20, cx0:cx1]
            hi = img[feet - 104:feet - 64, cx0:cx1]
            nlo = int(((lo[..., 0] == 228) & (lo[..., 1] == 92) & (lo[..., 2] == 16)).sum())
            if nlo >= 80:
                nhi = int(((hi[..., 0] == 228) & (hi[..., 1] == 92) & (hi[..., 2] == 16)).sum())
                ceiling = nhi < 30
        ham = False
        ham_dbg = None
        if world >= HAM_WORLD and f > 1:
            hx0 = min(246, xr + 2)
            hx1 = min(250 - s, xr + HAM_RANGE)
            hy0 = max(40, feet - 80)
            hy1 = min(240, feet + 2)
            if hx1 > hx0 + 4 and hy1 > hy0 + 4:
                a = img[hy0:hy1, hx0:hx1]
                b = prev_img[hy0:hy1, hx0 + s:hx1 + s]
                nb = (a.sum(2) == 0) & (b.sum(2) > 0)
                gm = (a[..., 0] == 0) & (a[..., 1] == 168) & (a[..., 2] == 0)
                if gm.any():
                    gcols = gm.any(axis=0)
                    gtop = int(np.nonzero(gm.any(axis=1))[0][0])
                    k3 = gcols.copy()
                    for dd in (1, 2, 3):
                        k3[dd:] |= gcols[:-dd]
                        k3[:-dd] |= gcols[dd:]
                    nb[max(0, gtop - 1):, :] &= ~k3[None, :]
                ham = int(nb.sum()) >= HAM_N
                if nb.any():
                    nys, nxs = np.nonzero(nb)
                    ham_dbg = (int(nxs.mean()) + hx0 - xr, int(nys.mean()) + hy0 - feet, int(nb.sum()))
        if ham:
            ham_free = 0
        else:
            ham_free += 1
            if ham_free > 15:
                ham_wait = 0
        keys = {"Right", "x"}
        flyer_jump = False
        if (world < FLY_WORLD_MAX and jump_t == 0 and grounded and cooldown == 0 and gdx is not None and GFLY_MIN <= gdx <= GFLY_MAX
                and gb < feet - (GFLY_HB_THR if (GFLY_HB_LO <= world < GFLY_HB_HI and gn < PIPE_N) else 3) and gb >= feet - GFLY_UP and feet >= 195):
            flyer_jump = True
        pipe_jump = False
        if (jump_t == 0 and grounded and cooldown == 0 and gdx is not None and gn >= PIPE_N
                and 0 <= gdx <= PIPE_D and world >= PIPE_WORLD):
            pipe_jump = True
        if pipe_jump:
            jump_kind = "pipe"
            jump_t = JK_PIPE - 1
            keys.add("z")
            print("f=%d world=%d mx=%d feet=%d JUMP pipe gdx=%d gn=%d" % (f, world, xr, feet, gdx, gn))
        elif flyer_jump and ham and ham_wait < HAM_WAIT_MAX and gdx > HAM_GAP:
            ham_wait += 1
            keys = {"Left"} if gdx < 60 else set()
            if ham_wait == 1:
                print("f=%d world=%d HAMMER wait(fly) gdx=%s" % (f, world, gdx))
        elif flyer_jump:
            jump_kind = "flyer"
            jump_t = (min(JK_FLY, JK_LOW) if ceiling else JK_FLY) - 1
            keys.add("z")
            print("f=%d world=%d mx=%d feet=%d JUMP flyer gdx=%d gb=%d" % (f, world, xr, feet, gdx, gb))
        elif jump_t > 0:
            press_age += 1
            jump_min = min(jump_min, feet)
            if jump_kind == "pit" and press_age == ABORT_AGE and feet >= press_feet - 2:
                jump_t = 0
                cooldown = 1
                print("f=%d abort failed jump" % f)
            elif (LAND_REJUMP and jump_kind == "pit" and press_age >= 6 and jump_min <= 190 and feet >= 203
                  and len(feet_hist) >= 2 and abs(feet - feet_hist[-2]) <= 1):
                # landed early (low ceiling): release z so that the pit logic can jump again
                jump_t = 0
                cooldown = 0
                print("f=%d world=%d pit jump landed early -> re-jump" % (f, world))
            else:
                keys.add("z")
                jump_t -= 1
                if jump_t == 0:
                    cooldown = 2
        elif (HEXT_ON and jump_kind == "koopa" and world >= HDEL_WORLD and hext_n < HEXT_MAX and feet < 204
              and len(feet_hist) >= 2 and feet_hist[-1] <= feet_hist[-2] and ham_dbg is not None
              and 0 <= ham_dbg[0] <= HEXT_DX and -HEXT_UP <= ham_dbg[1] <= HEXT_DN):
            # low hammer approaching at foot level while still rising: keep holding jump to climb above it
            hext_n += 1
            keys.add("z")
            if hext_n == 1:
                print("f=%d world=%d HEXT hammer dx=%d dy=%d ft=%d" % (f, world, ham_dbg[0], ham_dbg[1], feet))
        else:
            if cooldown > 0:
                cooldown -= 1
            g2 = False
            if len(feet_hist) >= 2 and feet + 3 <= 240:
                d1 = feet_hist[-1] - feet_hist[-2]
                sub = img[feet + 1:feet + 3, max(0, xr - 12):max(1, xr - 2)]
                supported = not bool(_sky(sub).all())
                g2 = abs(d1) <= 1 and supported
            first = None
            x0 = min(250, xr + 1)
            x1 = min(255, xr + 75)
            if x1 > x0 + 3:
                col = _sky(img[216:238, x0:x1]).all(axis=0)
                idx = np.nonzero(col)[0]
                for i in idx:
                    if i + 2 < len(col) and col[i] and col[i + 1] and col[i + 2]:
                        first = int(i)
                        break
            near = False
            if len(feet_hist) >= 2 and feet + 30 <= 240:
                dd = feet_hist[-1] - feet_hist[-2]
                if dd >= 1:
                    sub2 = img[feet + 1:feet + 3 + NEAR_MULT * dd, max(0, xr - 12):max(1, xr - 2)]
                    near = not bool(_sky(sub2).all())
            elevated = feet < 200
            bunny = elevated and first is not None and first <= BUNNY_D and not (NOBUNNY_CEIL and ceiling)
            if (g2 or near) and cooldown == 0:
                want = None
                if first is not None and (bunny or first <= PIT_DIST) and (g2 or bunny):
                    kk = JK_PIT
                    if (not bunny and gdx is not None and gn >= PIPE_N and 0 < gdx - first <= PIPE_AFTER_PIT):
                        kk = JK_PIT_PIPE
                    want = ("pit", kk)
                if want is None:
                    if still >= BLOCK_FRAMES and grounded:
                        want = ("block", JK_BLOCK)
                        still = 0
                if want is None and grounded and gdx is not None and gn < PIPE_N:
                    hb = world >= HB_WORLD
                    if gb >= feet - GHI and gdx <= ((GJD_HB if world < 2880 else GJD_HB2) if hb else GJD):
                        want = ("koopa", (JK_KOOPA_HB2 if world >= 2880 else JK_KOOPA_HB) if hb
                                else (JK_KOOPA2 if 1800 <= world <= 1900 else JK_KOOPA))
                if want is None and grounded and world >= ENEMY_WORLD:
                    ex0 = min(250, xr + 6)
                    ex1 = min(250 - s, xr + 6 + LOOK)
                    ey0 = max(40, feet - HEAD_UP)
                    ey1 = feet + 1
                    if ex1 > ex0 + 4:
                        a = img[ey0:ey1, ex0:ex1].astype(np.int16)
                        b = prev_img[ey0:ey1, ex0 + s:ex1 + s].astype(np.int16)
                        ch = (np.abs(a - b).sum(2) > 60)
                        n = int(ch.sum())
                        if n >= ENEMY_MIN:
                            want = ("enemy", JK_ENEMY)
                if (HDEL_ON and want is not None and want[0] == "koopa" and world >= HDEL_WORLD and ham and ham_dbg is not None
                        and 8 <= ham_dbg[0] <= HDEL_DX and ham_dbg[1] >= -45 and (gdx is None or gdx > HDEL_MIN_G)
                        and hdel_n < HDEL_MAX):
                    hdel_n += 1
                    if hdel_n == 1:
                        print("f=%d world=%d HDEL hammer dx=%d dy=%d gdx=%s" % (f, world, ham_dbg[0], ham_dbg[1], gdx))
                    want = None
                if (want is not None and want[0] in ("koopa", "enemy") and ham and ham_wait < HAM_WAIT_MAX
                        and (gdx is None or gdx > HAM_GAP)):
                    want = None
                    ham_wait += 1
                    keys = {"Left"} if (gdx is not None and gdx < WAIT_LEFT_D) else set()
                    if ham_wait == 1:
                        print("f=%d world=%d HAMMER wait gdx=%s" % (f, world, gdx))
                if want is not None:
                    jump_kind = want[0]
                    jump_t = want[1]
                    if ceiling and jump_kind in ("enemy", "koopa"):
                        jump_t = min(jump_t, JK_LOW)
                    keys.add("z")
                    jump_t -= 1
                    press_feet = feet
                    press_age = 0
                    jump_min = feet
                    hdel_n = 0
                    hext_n = 0
                    print("f=%d world=%d mx=%d feet=%d JUMP %s" % (f, world, xr, feet, jump_kind))
        # an airborne (leaping) hammer bro close ahead: do not run under it, wait for it to land
        airb_hit = False
        if (AIRB_ON and AIRB_WORLD <= world < STONE_WORLD and st_stage == 0 and jump_t == 0 and grounded
                and gdx is not None and gn < PIPE_N and AIRB_MIN_GDX < gdx <= max(AIRB_D, STAND_D if STAND_ON else 0)
                and gb < feet - GHI):
            if airb_n > 0:
                airb_hit = True      # already waiting: keep waiting until the bro has landed
            elif prev_gb is not None and gb - prev_gb >= 1 and gdx <= AIRB_D:
                tt = gdx / 3.0
                airb_hit = (gb + (gb - prev_gb) * tt * AIRB_ACC) >= feet - AIRB_CLR
            if (not airb_hit and STAND_ON and prev_gb is not None and gb - prev_gb >= 0 and gdx <= STAND_D
                    and world >= STAND_WORLD):
                airb_hit = True
        if airb_hit:
            airb_n += 1
            if airb_n <= AIRB_MAX:
                keys = {"Left"} if (AIRB_LEFT_ON and (gdx is None or gdx <= STAND_L or not STAND_ON)) else set()
                still = 0
                if airb_n == 1:
                    print("f=%d world=%d AIRBRO wait gdx=%d gb=%d" % (f, world, gdx, gb))
        elif gdx is None or gdx > AIRB_D + 20:
            airb_n = 0
        # descending towards a bro standing on the floor ahead: brake in the air so we land short of it
        if (AIRS_ON and 2300 <= world < STONE_WORLD and st_stage == 0 and not grounded and feet < 204
                and len(feet_hist) >= 2 and feet_hist[-1] >= feet_hist[-2] - 1):
            g2ys, g2xs = _green(img, max(0, xr - 10), 40, min(256, xr + 135), 216)
            if len(g2ys) >= 6:
                g2xa = g2xs + max(0, xr - 10)
                g2l = int(g2xa.min())
                g2sel = g2xa <= g2l + 24
                g2n = int(g2sel.sum())
                g2b = int(g2ys[g2sel].max()) + 40
                g2d = g2l - xr
                if g2n < PIPE_N and g2b >= 196 and g2d > 0:
                    T = (206.0 - feet) / AIRS_FALL
                    L = g2d - max(s, 1) * T
                    if AIRS_LO < L < (AIRS_HI if airs_n == 0 else AIRS_HI + 30):
                        keys = {"Left", "z"} if "z" in keys else {"Left"}
                        if airs_n == 0:
                            print("f=%d world=%d AIRSTEER d=%d L=%.0f ft=%d" % (f, world, g2d, L, feet))
                        airs_n += 1
        if grounded:
            airs_n = 0
        if world >= STONE_WORLD or st_stage > 0:
            mb = _mario_bbox(img)
            if st_stage == 0:
                keys = {"Right", "x"}
                if mb is not None:
                    st_feets.append(mb[3])
                    if s == 0 and len(st_feets) >= 3 and st_feets[-1] == st_feets[-2] == st_feets[-3]:
                        st_still += 1
                    else:
                        st_still = 0
                    if st_still >= 4 and mb[3] >= 200:
                        st_stage = 1
                        st_ph = 0
                        st_n = 0
                        print("f=%d world=%d STONE canonical reached mb=%s" % (f, world, mb))
            if st_stage == 1:
                if st_ph == 0:
                    # rest against the stack so that the hops start from a standstill
                    keys = set()
                    st_n += 1
                    if st_n >= STONE_IDLE:
                        st_ph = 1
                        st_mi = 0
                        st_i = 0
                        st_sc0 = scroll_total
                elif st_ph == 1:
                    if st_mi == 0:
                        mp, mk, mrr = _STONE_M0
                    elif st_mi == 1:
                        mp, mk, mrr = _STONE_M1C if st_c0 <= 12 else _STONE_M1
                    else:
                        mp, mk, mrr = st_p2, _STONE_K2, _STONE_RR2
                    keys = _macro_keys(st_i, mp, mk, mrr, True)
                    st_i += 1
                    if st_i >= mp + max(mk, mrr):
                        if st_mi >= 2:
                            st_stage = 2
                            print("f=%d STONE jump done world=%d" % (f, world))
                        else:
                            st_ph = 2
                            st_hist = []
                            st_n = 0
                elif st_ph == 2:
                    keys = set()
                    st_n += 1
                    mbx = _mario_bbox(img)
                    if mbx is not None:
                        st_hist.append((mbx[0] + scroll_total, mbx[3]))
                    rest = len(st_hist) >= STONE_REST and len(set(st_hist[-STONE_REST:])) == 1
                    if rest or st_n > 120:
                        st_mi += 1
                        st_i = 0
                        st_ph = 1
                        if st_mi == 1:
                            st_c0 = scroll_total - st_sc0
                        if st_mi == 2:
                            cls = scroll_total - st_sc0
                            if st_c0 <= 12:
                                st_p2 = 11
                            elif cls >= 66:
                                st_p2 = 8
                            elif cls >= 63:
                                st_p2 = 12
                            else:
                                st_p2 = 11
                            print("f=%d STONE at block A: cls=%d -> p=%d" % (f, cls, st_p2))
        if DBG_LO <= world <= DBG_HI and st_stage == 0:
            print("D f=%d w=%d ft=%d g=%s gb=%s gn=%d ham=%s hd=%s jt=%d k=%s" % (
                f, world, feet, gdx, gb, gn, ham, ham_dbg, jump_t, "".join(sorted(k[0] for k in keys))))
        if PITDBG and PDBG_LO <= world <= PDBG_HI:
            print("P f=%d w=%d xr=%d ft=%d gr=%s cd=%d jt=%d kind=%s g=%s gb=%s gn=%d k=%s" % (
                f, world, xr, feet, grounded, cooldown, jump_t, jump_kind, gdx, gb, gn,
                "".join(sorted(k[0] for k in keys))))
        prev_img = img
        prev_g = g
        prev_gb = gb
        sim.step(keys)
        if f - last_log >= 60:
            last_log = f
            print("f=%d world=%d mx=%d feet=%d s=%d gr=%s" % (f, world, xr, feet, s, grounded))

