# ==================================================================================================
# Super Mario Bros. (NES) -- level 6-2  |  超级马里奥兄弟（NES） 关 6-2
# Pass rate on 100 random variants never used before: 68/100; fastest finish 2360 frames (39.33 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 68/100；最快一局 2360 帧（60 帧/秒，约 39.33 秒）。
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
#   closed-loop SMB runner: Mario located by brown+red pixels; hold run+right, jump on colour-clas
#   sified obstacles/enemies/pits, wait for piranha plants.
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _eq                            line 16    
#   def find_mario                     line 20    
#   def jump_dist                      line 44    
#   def hold_for                       line 48    
#   def scroll_shift                   line 60    how many px the world moved left between two frames (0..6)
#   def play                           line 72    
# ==================================================================================================

# idea: closed-loop SMB runner: Mario located by brown+red pixels; hold run+right, jump on colour-classified obstacles/enemies/pits, wait for piranha plants.
import numpy as np

JUMP_BASE = 10
JUMP_PER_H = 0.45
PIPE_D = 42
ENEMY_D = 30
PIT_LOOK = 3
HOLD_TALL = 30
HOLD_LOW = 16
HOLD_ENEMY = 18
DEBUG = True
AIR_BRAKE = True


def _eq(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def find_mario(s, prev):
    br = _eq(s, (172, 124, 0))
    br[:40] = False
    if prev is not None:
        px, py = prev
        w = np.zeros_like(br)
        w[max(0, py - 40):py + 40, max(0, px - 40):px + 40] = True
        b2 = br & w
        if b2.sum() >= 3:
            br = b2
    ys, xs = np.nonzero(br)
    if len(xs) < 3:
        return None
    mx, my = int(np.median(xs)), int(np.median(ys))
    x0, x1 = max(0, mx - 12), min(256, mx + 13)
    y0, y1 = max(40, my - 14), min(240, my + 15)
    sub = s[y0:y1, x0:x1]
    m = _eq(sub, (172, 124, 0)) | _eq(sub, (248, 56, 0))
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    return int(xs.min()) + x0, int(xs.max()) + x0, int(ys.min()) + y0, int(ys.max()) + y0


def jump_dist(h, v):
    return int((0.22 * h + 2) * max(v, 1.2) + 4)


def hold_for(h):
    if h <= 20:
        return 8
    if h <= 34:
        return 12
    if h <= 50:
        return 18
    if h <= 66:
        return 24
    return 30


def scroll_shift(prev_rows, rows):
    """how many px the world moved left between two frames (0..6)"""
    best, bs = 0, None
    for sh in range(0, 7):
        a = rows[:, :250 - sh].astype(np.int16)
        b = prev_rows[:, sh:250].astype(np.int16)
        d = int(np.abs(a - b).sum())
        if bs is None or d < bs:
            best, bs = sh, d
    return best


def play(sim):
    prev = None
    held = 0
    cool = 0
    last_feet = None
    frozen = 0
    last_sig = None
    t = 0
    plant_wait = 0
    still = 0
    last_ground = None
    rel = 0
    jt = 0
    prev_rows = None
    edge_seen = False
    edge_wait = 0
    last_pa = -100
    fly_wait = 0
    fly_clear = 0
    stair_wait = 0
    og_wait = 0
    pw_prev = 0
    last_plant_t = 0
    last_og = -100
    tol_t = -1
    climbs = 0
    airleft = 0
    prev_x = -1
    lastx = -2
    backoff = 0
    vsc = 0.0
    while sim.budget_left > 0 and t < 7900:
        t += 1
        s = sim.screen()
        sig = s[100:240:5, ::7].tobytes()
        if sig == last_sig:
            frozen += 1
        else:
            frozen = 0
        last_sig = sig
        if frozen > 400:
            print("frozen end", t)
            return
        rows_now = s[226:236, :, 1]
        if prev_rows is not None:
            sh = scroll_shift(prev_rows, rows_now)
            vsc = 0.6 * vsc + 0.4 * sh
        prev_rows = rows_now
        gr = s[209:239:3, ::4]
        if last_ground is not None and np.array_equal(gr, last_ground) and prev_x == lastx:
            still += 1
        else:
            still = 0
        last_ground = gr
        if still > 40:
            tol_t = t + 500
        mar = find_mario(s, prev)
        keys = {"Right", "x"}
        if mar is None:
            if held > 0:
                keys.add("z")
                held -= 1
            sim.step(keys)
            continue
        x0, x1, y0, y1 = mar
        lastx = prev_x
        prev_x = x0
        prev =((x0 + x1) // 2, (y0 + y1) // 2)
        feet = min(y1, 232)
        vy = 0 if last_feet is None else feet - last_feet
        last_feet = feet
        sup = s[min(239, feet + 1):min(240, feet + 3), x0 + 1:x1 - 1]
        support = (sup.sum(axis=2) > 40).any()
        if held > 0:
            jt += 1
            if jt > 8 and vy == 0 and support:
                held = 0
                rel = 2
        else:
            jt = 0
        on_ground = (vy == 0) and held <= 0 and bool(support)
        xr = x1 + 3
        top = max(0, feet - 13)
        xe = min(256, xr + 90)
        info = ""
        want = False
        k = HOLD_LOW
        wait = False
        climb = False
        hop = False
        if on_ground and xr < 250:
            box = s[top:feet + 1, xr:xe]
            R, G, B = box[:, :, 0], box[:, :, 1], box[:, :, 2]
            orange = (R == 228) & (G == 92) & (B == 16)
            enemy = ((R == 252) & (G == 160) & (B == 68)) | ((R == 252) & (G == 252) & (B == 252)) | \
                    ((R == 240) & (G == 208) & (B == 176))
            oc = np.nonzero(orange.sum(axis=0) >= 2)[0]
            ec = np.nonzero(enemy.sum(axis=0) >= 2)[0]
            edge_left = not bool(s[min(239, feet + 1), max(0, x0 - 3):max(1, x0 - 1)].sum(axis=1).max() > 40)
            lb = s[max(0, feet - 7):feet + 1, max(0, x0 - 50):max(0, x0 - 1)]
            lr, lg, lbb = lb[:, :, 0], lb[:, :, 1], lb[:, :, 2]
            left_enemy = bool((((lr == 252) & (lg == 160) & (lbb == 68)) | ((lr == 252) & (lg == 252) & (lbb == 252)) |
                               ((lr == 240) & (lg == 208) & (lbb == 176))).sum() >= 3)
            # pipe: tall lime columns
            tall = s[max(0, feet - 62):feet + 1, xr:xe]
            lime = (tall[:, :, 0] == 184) & (tall[:, :, 1] == 248) & (tall[:, :, 2] == 24)
            pc = np.nonzero(lime.sum(axis=0) >= 26)[0]
            # pit
            pit_d = None
            dcs = (s[min(231, feet + 1):min(240, feet + 9), max(0, xr - 2):min(256, xr + 60)].sum(axis=2) < 40)
            if dcs.shape[0] >= 6:
                dcol = dcs.all(axis=0)
                run = 0
                for i, b in enumerate(dcol):
                    run = run + 1 if b else 0
                    if run >= 3:
                        pit_d = max(0, i - 2 - 2)
                        break
            if len(pc):
                pd = int(pc[0])
                info += " pipe%d" % pd
                cx = xr + pd
                # locate pipe top
                big = s[max(0, feet - 110):feet + 1, cx:min(256, cx + 26)]
                lm = (big[:, :, 0] == 184) & (big[:, :, 1] == 248) & (big[:, :, 2] == 24)
                rows = np.nonzero(lm.sum(axis=1) >= 3)[0]
                ptop = max(0, feet - 110) + (int(rows[0]) if len(rows) else 100)
                h = feet - ptop
                dj = jump_dist(h, vsc)
                dj_run = jump_dist(h, 3.0)
                info += " h%d dj%d v%.1f" % (h, dj, vsc)
                need = 0.2 * h
                plant = False
                if pd <= dj_run + 4:
                    reg = s[max(0, ptop - 26):max(0, ptop - 1), cx:min(256, cx + 26)]
                    rr, gg, bb = reg[:, :, 0], reg[:, :, 1], reg[:, :, 2]
                    plantpx = (reg.sum(axis=2) > 60) & ~((rr == 228) & (gg == 92) & (bb == 16))
                    plant = bool(plantpx.sum() >= 6)
                plant_w = plant and plant_wait < 110 and frozen < 100
                if h > 80 and pd < 20 and climbs < 8:
                    want = True
                    k = 14
                    climb = True
                    info += " CLIMB"
                elif plant_w:
                    info += " PLANT"
                    wait = True
                    plant_wait += 1
                    backoff = 0
                elif False and t > 1800 and pd < 80 and stair_wait < 300 and frozen < 100 and bool(
                        ((s[40:max(41, feet - 12), max(0, x0 - 30):min(256, xr + 90), 0] == 252) &
                         (s[40:max(41, feet - 12), max(0, x0 - 30):min(256, xr + 90), 1] == 160) &
                         (s[40:max(41, feet - 12), max(0, x0 - 30):min(256, xr + 90), 2] == 68)).sum() >= 3):
                    info += " PIPEFLY"
                    wait = True
                    stair_wait += 1
                elif backoff > 0:
                    if pd >= jump_dist(h, 1.6) + 6 or backoff == 1 or left_enemy or edge_left:
                        backoff = 0
                    else:
                        backoff -= 1
                        info += " BACK"
                elif pd < need and h > 20:
                    if left_enemy:
                        want = True
                        k = 30
                        info += " JUMPNOW"
                    else:
                        backoff = 40
                        info += " BACK0"
                if backoff > 0 or climb or wait or want:
                    pass
                elif pd <= dj and pd <= dj_run + 4:
                    want = True
                    k = hold_for(h)
            fa = s[max(88, feet - 75):max(89, feet - 12), xr:min(256, xr + 60)]
            far_, fag, fab = fa[:, :, 0], fa[:, :, 1], fa[:, :, 2]
            flyer = bool((((far_ == 252) & (fag == 160) & (fab == 68)) | ((far_ == 252) & (fag == 252) & (fab == 252))).sum() >= 4)
            if flyer and not wait and not want and not climb and fly_wait < 120 and frozen < 100 and not len(pc) and \
                    (pit_d is None or pit_d > 45):
                wait = True
                fly_wait += 1
                info += " FLY"
            elif not flyer:
                fly_clear += 1
                if fly_clear > 25:
                    fly_wait = 0
            if flyer:
                fly_clear = 0
            if not want and not wait and len(oc):
                od = int(oc[0])
                h = 0
                for dx in (1, 3, 5):
                    colx = min(255, xr + od + dx)
                    hh, gap = 0, 0
                    for yy in range(feet, max(0, feet - 100), -1):
                        if s[yy, colx].sum() > 40:
                            hh = feet - yy + 1
                            gap = 0
                        else:
                            gap += 1
                            if gap > (5 if still > 40 else 0):
                                break
                    h = max(h, hh)
                dj = jump_dist(max(h, 14), vsc)
                info += " orange%d h%d dj%d" % (od, h, dj)
                if DEBUG and t in (4, 33):
                    for dx in (1, 3, 5, 9, 15):
                        cxx = min(255, xr + od + dx)
                        print("COL", cxx, "feet", feet, "xr", xr, [int(s[yy, cxx].sum() > 40) for yy in range(feet - 80, feet + 2, 2)])
                    print("ROW", [int(s[feet - 8, xx].sum() > 40) for xx in range(x1 - 5, min(256, x1 + 40))])
                sw = s[40:max(41, feet - 12), max(0, x0 - 30):min(256, xr + 90)]
                stair_fly = bool(((sw[:, :, 0] == 252) & (sw[:, :, 1] == 160) & (sw[:, :, 2] == 68)).sum() >= 3)
                if stair_fly and h >= 30 and stair_wait < 150 and od < 60 and frozen < 100:
                    wait = True
                    stair_wait += 1
                    info += " STAIRFLY"
                elif backoff > 0:
                    if od >= jump_dist(h, 1.6) + 6 or backoff == 1 or left_enemy or edge_left:
                        backoff = 0
                    else:
                        backoff -= 1
                        info += " BACK"
                elif od < 0.2 * h and h > 34 and x0 > 100:
                    backoff = 40
                    info += " BACK0"
                elif od <= dj:
                    want = True
                    k = hold_for(h)
            if not want and not wait and len(ec):
                ed = int(ec[0])
                info += " enemy%d" % ed
                if ed <= ENEMY_D:
                    want = True
                    k = HOLD_ENEMY
            if not want and not wait and pit_d is not None and pit_d <= 40:
                info += " pit%d" % pit_d
                pa = s[max(88, feet - 110):min(240, feet + 40), xr + pit_d:min(256, xr + pit_d + 70)]
                pr, pg, pb = pa[:, :, 0], pa[:, :, 1], pa[:, :, 2]
                plant_ahead = (((pr == 252) & (pg == 160) & (pb == 68)) | ((pr == 252) & (pg == 252) & (pb == 252))).sum() >= 4
                pl = ((pa[:, :, 0] == 184) & (pa[:, :, 1] == 248) & (pa[:, :, 2] == 24)).sum() >= 40
                if plant_ahead:
                    edge_seen = True
                    last_pa = t
                if (plant_ahead or t - last_pa < 14) and plant_wait < 110 and frozen < 100 and \
                        pit_d <= 8 and vsc > 1.5:
                    want = True
                    k = 10
                    hop = True
                    plant_wait += 3
                    info += " HOPBACK"
                elif (plant_ahead or t - last_pa < 14) and plant_wait < 110 and frozen < 100:
                    wait = True
                    plant_wait += 1
                    info += " PLANT2"
                elif pl and not edge_seen and edge_wait < 150 and frozen < 100:
                    wait = True
                    edge_wait += 1
                    info += " EDGEW"
                elif pit_d <= PIT_LOOK:
                    want = True
                    k = HOLD_TALL
            if not want and not wait and still > 25:
                want = True
                k = HOLD_TALL
        if og_wait and "orange" not in info and t - last_og > 40:
            og_wait = 0
        if backoff > 0 and "BACK" not in info:
            backoff = 0
        if on_ground and "pit" not in info:
            edge_seen = False
            edge_wait = 0
        if want and on_ground and cool <= 0 and rel <= 0:
            held = k
            cool = 6
            if climb:
                airleft = 22
                climbs += 1
            elif hop:
                airleft = 24
        cool -= 1
        rel -= 1
        if backoff > 0 and on_ground:
            keys = {"Left", "x"}
        elif wait:
            keys = {"Left"} if vsc > 1.2 else set()
        elif plant_wait and not info:
            plant_wait = 0
        if DEBUG and (t % 200 == 0 or (t % 3 == 0 and t < 150 and t > 5) or (want and held == k and t < 1200) or (t % 6 == 0 and t < 1500 and info and ("BACK" in info or "PLANT" in info))):
            print("t", t, "m", x0, x1, y0, y1, info, "WANT" if want else "", "WAIT" if wait else "")
        if held > 0:
            keys.add("z")
            held -= 1
            if held == 0:
                rel = 3
        if (not on_ground) and vy > 0 and airleft <= 0 and AIR_BRAKE:
            ar = s[max(0, feet - 4):min(236, feet + 40), xr:min(256, xr + 34)]
            arr, arg, arb = ar[:, :, 0], ar[:, :, 1], ar[:, :, 2]
            if ((((arr == 252) & (arg == 160) & (arb == 68)) | ((arr == 252) & (arg == 252) & (arb == 252))).sum() >= 4):
                keys.discard("Right")
                keys.add("Left")
        if airleft > 0:
            airleft -= 1
            keys.discard("Right")
            keys.add("Left")
        sim.step(keys)

