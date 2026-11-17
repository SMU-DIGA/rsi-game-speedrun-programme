# ==================================================================================================
# Super Mario Bros. (NES) -- level 5-2  |  超级马里奥兄弟（NES） 关 5-2
# Pass rate on 100 random variants never used before: 80/100; fastest finish 1357 frames (22.62 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 80/100；最快一局 1357 帧（60 帧/秒，约 22.62 秒）。
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
#   PB closed-loop reactive runner (Right+B, colour-detected Mario, lookahead walls/enemies/pits, 
#   HIGH pipe route at x1790); NEW: in the x2350-2480 landing zone (after the floating-brick pit h
#   op) brake/st
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _eq                            line 210   
#   def solid_mask                     line 214   
#   def brick_mask                     line 246   
#   def mario_mask                     line 250   
#   def find_mario                     line 254   
#   def pick_k                         line 271   
#   def est_scroll                     line 278   
#   def find_bills                     line 295   black segments (bullet bills) in the lane band; returns list of (x0,x1)
#   def green_clusters                 line 321   list of (xa, xb, ya, yb, n) bounding boxes of green pixel clusters (grouped by column gaps)
#   def ground_goombas                 line 351   narrow solid-coloured objects standing on the ground (goombas / koopas): list of (x0, x1)
#   def pipe_mask                      line 370   
#   class Pilot                        line 374   
#   def play                           line 1354  
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: PB closed-loop reactive runner (Right+B, colour-detected Mario, lookahead walls/enemies/pits, HIGH pipe route at x1790); NEW: in the x2350-2480 landing zone (after the floating-brick pit hop) brake/steer in the air using static ground-goomba detection too, so Mario lands short of the 2 goombas instead of beside them.
import math
import numpy as np

NOFW_LO = 1740   # no flyer-wait (standing still) around the x1790 pipe: a high hop needs the run-up speed
NOFW_HI = 1830
SIN_HIGH = 150
SIN_GOD = 34
SIN_CREEP = 40
SIN_VY = 0.8
SIN_GO = 45
SIN_ON = True
SIN_LO = 1480
SIN_HI = 1700
SIN_C = 186.5
SIN_A = 19.5
SIN_W = 2 * 3.14159265 / 57.0
SIN_M = 12
SIN_COMMIT = 28
LOOK = 30       # px ahead of Mario's front edge to react to obstacles
PIT_LOOK = 8     # px ahead (of front) at which a pit triggers a jump
PIT_K = 14       # jump hold frames for pits
MIN_K = 4
STAIR_MARGIN = 7
GREEN_LOOK = 34
FLY_LOOK = 70
FLY_RANGE = 140
FLY_WAIT_MAX = 40
STAIR_WAIT_MAX = 90
NARROW_NEED = 65
NOX_LO = 1735
NOX_HI = 1850
KHIGH_ON = True
KHIGH_ALWAYS = True
BWE_ON = False
KH_D = 15
KHIGH_D = 130
KZ_ON = True
KZ_LO = 1830
KZ_HI = 1960
KZ_RANGE = 90
KZ_HOP = 11
KZ_MIN = -99
KZ_BACK_MAX = 40
KZ_BEHIND = -3
KZ_MAX = 260
KZ_K = 3
KHIGH_LO = 1740
KHIGH_HI = 1830
LEDGE_MAX_WX = 99999
FLY_JUMP_D = 44
LEDGE_NEED_EARLY = 64
WALK_LO = 985
WALK_HI = 1075
PITWALL_NEED = 80
PITWALL_E = 7
AIR_RANGE = 90
PREPRESS_D = 17
PREPRESS_F = 2
PREPRESS_K = 11
AIR_BRAKE = True
AIR_BRAKE_D = 80
AIR_LEFT_D = 45
HOP_D_FAST = 20
HOP_NEED_FAST = 64
LATE_WX = 1100
PIPE_NEED = 99
MV_MIN_WX = 880
MV_ND_MAX = 700
MV_RANGE = 90
HOP_D = 16
HOP_NEED = 45
MV_WAIT_MAX = 300
STUCK_N = 70
GO_Y = 135
PROFILE_LEN = 80
DEBUG = True
ELEV_NEED = 64
PLAT_D = 24
PITNF2_LO = 1300
PITNF2_HI = 1560
PITLOW_LO = 1400
PITLOW_HI = 1480
PITLOW_K = 6
PITLOW_LOOK = 1
AB2_BRAKE = True
AB2_STATIC = True
AB2_PD = 60
AB2_LO = 2350
AB2_HI = 2480
RET_LO = 1250
RET_HI = 1300
RET_D = 45
RET_MAX = 30
ABRAKE_M = 10
PITNF_LO = 2480
PITNF_HI = 2720
EDGE3_FLO = 1800
EDGE3_LO = 1935
EDGE3_HI = 2040
EDGE3_D = 8
EDGE3_SLOW = 71
PAUSE_WX = 640
PAUSE_N = 6
PLAT_LO = 1790
PLAT_HI = 1945
EDGE2_LO = 1930
EDGE2_HI = 2030
EDGE_COLS2 = 9
PLAT_RANGE = 80
PLAT_LEFT = 50
CANNON_LO = 1680
CANNON_HI = 1800
MVZ2_LO = 1560
MVZ2_HI = 1710
MVZ2_D = 40
PLANT_MAX = 0
CAN_LO = 1600
CAN_HI = 1760
LANE_LO = 1560
LANE_HI = 1710
KP_WAIT = False
PK3_LO = 2560
PK3_HI = 2720
PK3_K = 22
PLATAIR_ON = True
SZ_LO = 1200
SZ_HI = 1700
SZ_D = 30
PBILL_LO = 1700
PBILL_HI = 1850
PBILL_MAX = 0
PITB_LO = 2200
PITB_HI = 2340
PITB_D = 4
END_LO = 2760
END_HI = 2900
PW2_LO = 2200
PW2_HI = 2330
PW2_NEED = 71
FYZ_LO = 1550
FYZ_HI = 1750
FYZ_M = -2
MVZ_LO = 2400
MVZ_HI = 2480
MVZ_D = 30
MVZ_NEED = 52
CEIL_HI = 2300
PB_LO = 1200
PB_HI = 1700
PB_ZONE1 = 1340
PB_D1 = 40
PB_Y1 = 50
PB_D = 64
PB_MAX = 0
PB_Y = 24
BW_LO = 2300
BW_HI = 2520
BW_RANGE = 120
BW_MAX = 240
ROWS_LATE = 9
CEIL_ON = True
NOTAN = True
NOTAN_LO = 1200
NOJ_LO = 1085
NOJ_HI = 1140
PLAT_WAIT_MAX = 40
FLY2_LO = 2500
FLY2_HI = 2800
FLY2_NEED = 64
_DBG = ()
BE_LO = 2200
BE_HI = 2300
BE_K = 11
PLANT_LO = 99999  # DISABLED: earlier takeoff avoids the pipe-wall stall but lands on the x925-960 goomba stairs (deaths x968-977)
PLANT_HI = 880
PLANT_LOOK = 40
PLDBG = False
PW_ON = False   # refuted: Mario cannot skid to a stop in time after landing from the x736 jump (jumps at d=10 -> deaths x1078)
PW_LO = 730
PW_HI = 860
PW_DMIN = 22
PW_DMAX = 85
PW_MAX = 320
ROWJ_ON = True
ROWJ_LO = 2370
ROWJ_HI = 2400
ROWJ_Y0 = 120
ROWJ_Y1 = 140
FW_ON = True
FW_JUMP = True
FW_CEIL = 55
LC_MAX = 150
LC_D = 27
FW_RANGE = 110
FW_M = 16
FW_M_FAST = 40
FW_COMMIT = 0
CAN_STUCK = 14
FW_CREEP = False
LH_LO = 18
LH_HI = 46
FW_MAX = 160
WIDE_LO = 1950    # beetle groups (up to 3 merged blobs) as movers
WIDE_MAX = 52

KTAB = [(34, 2), (42, 4), (49, 6), (56, 8), (64, 11), (71, 14), (79, 18), (99, 26)]


def _eq(img, c):
    return (img[..., 0] == c[0]) & (img[..., 1] == c[1]) & (img[..., 2] == c[2])


def solid_mask(reg):
    r = reg[..., 0]
    g = reg[..., 1]
    b = reg[..., 2]
    brick = (r == 228) & (g == 92) & (b == 16)
    black = (r == 0) & (g == 0) & (b == 0)
    tan = (r == 240) & (g == 208) & (b == 176)
    green = (r == 0) & (g == 168) & (b == 0)
    return brick | black | tan | green


BRICKONLY_LO = 1930
HOLD_A_DESC = True
UR_LO = 99999
UR_HI = 2000
UR_Y = 150
PK_MAX = 0
PK_Y = 155
LEDGEK_LO = 2380
LEDGEK_HI = 2470
LEDGEK = 8
STOMPZ_LO = 2440
STOMPZ_HI = 2540
STOMPZ_D = 24
PIPEAIR_ON = False
PIPEAIR_LEFT = 70
PIPEZ_LO = 1740
PIPEZ_HI = 1830
PIPEZ_NEED = 52
PIPEZ_MIND = 14


def brick_mask(reg):
    return (reg[..., 0] == 228) & (reg[..., 1] == 92) & (reg[..., 2] == 16)


def mario_mask(reg):
    return _eq(reg, (248, 56, 0)) | _eq(reg, (172, 124, 0))


def find_mario(img, px, py):
    y0 = max(32, py - 40)
    y1 = min(240, py + 40)
    x0 = max(0, px - 30)
    x1 = min(256, px + 30)
    m = mario_mask(img[y0:y1, x0:x1])
    ys, xs = np.nonzero(m)
    if len(ys) < 8:
        m = mario_mask(img[32:240, :])
        ys, xs = np.nonzero(m)
        if len(ys) < 8:
            return None
        y0 = 32
        x0 = 0
    return (xs.min() + x0, xs.max() + x0, ys.min() + y0, ys.max() + y0)


def pick_k(need):
    for apex, k in KTAB:
        if apex >= need:
            return k
    return 26


def est_scroll(prev, cur):
    a = prev[216:240].astype(np.int16)
    b = cur[216:240].astype(np.int16)
    best = 0
    bs = -1
    for s in range(0, 6):
        if s == 0:
            eq = (a == b).all(axis=2).sum()
        else:
            eq = (a[:, s:] == b[:, :-s]).all(axis=2).sum()
        eq = eq - (0.3 * s)
        if eq > bs:
            bs = eq
            best = s
    return best


def find_bills(img):
    """black segments (bullet bills) in the lane band; returns list of (x0,x1)"""
    band = img[108:134]
    blk = (band[..., 0] == 0) & (band[..., 1] == 0) & (band[..., 2] == 0)
    colsum = blk.sum(axis=0)
    cols = colsum >= 5
    segs = []
    x = 0
    while x < 256:
        if cols[x]:
            x0 = x
            while x < 256 and cols[x]:
                x += 1
            segs.append((x0, x - 1))
        else:
            x += 1
    out = []
    for x0, x1 in segs:
        if 10 <= x1 - x0 + 1 <= 20:
            low = img[129:132, x0:x1 + 1]
            sky = (low[..., 0] == 104) & (low[..., 1] == 136) & (low[..., 2] == 252)
            if sky.mean() > 0.9:
                out.append((x0, x1))
    return out


def green_clusters(img, x0, x1, y0, y1, gap=6, minpix=8):
    """list of (xa, xb, ya, yb, n) bounding boxes of green pixel clusters (grouped by column gaps)"""
    x0 = max(0, x0)
    x1 = min(256, x1)
    if x1 <= x0 or y1 <= y0:
        return []
    reg = img[y0:y1, x0:x1]
    g = (reg[..., 0] == 0) & (reg[..., 1] == 168) & (reg[..., 2] == 0)
    colany = g.any(axis=0)
    out = []
    x = 0
    W = g.shape[1]
    while x < W:
        if colany[x]:
            xa = x
            last = x
            while x < W and (colany[x] or x - last <= gap):
                if colany[x]:
                    last = x
                x += 1
            sub = g[:, xa:last + 1]
            ys = np.nonzero(sub.any(axis=1))[0]
            n = int(sub.sum())
            if n >= minpix:
                out.append((xa + x0, last + x0, int(ys.min()) + y0, int(ys.max()) + y0, n))
        else:
            x += 1
    return out


def ground_goombas(img, gy=209, wmax=20):
    """narrow solid-coloured objects standing on the ground (goombas / koopas): list of (x0, x1)"""
    row = solid_mask(img[gy - 9:gy - 8, :])[0]
    out = []
    x = 0
    while x < 256:
        if row[x]:
            x0 = x
            while x < 256 and row[x]:
                x += 1
            x1 = x - 1
            w = x1 - x0 + 1
            if 11 <= w <= wmax and (x0 < 4 or not row[max(0, x0 - 4):x0].any()) and (x1 > 251 or not row[x1 + 1:x1 + 5].any()):
                out.append((x0, x1))
        else:
            x += 1
    return out


def pipe_mask(reg):
    return (reg[..., 0] == 124) & (reg[..., 1] == 124) & (reg[..., 2] == 124)


class Pilot:
    def __init__(self):
        self.px, self.py = 48, 200
        self.jump_hold = 0
        self.prev_img = None
        self.same = 0
        self.t = 0
        self.scroll = 0
        self.last_pass_t = -1000
        self.waiting = False
        self.fly_wait = 0
        self.dead = False
        self.info = {}
        self.best_wx = 0
        self.last_z = False
        self.pend = None
        self.prev_py = None
        self.prev2_py = None
        self.jt = 0
        self.gtracks = []
        self.mover_x = []
        self.long_jump = False
        self.vert = False
        self.nlog = 0
        self.prev3 = None
        self.hist = []
        self.ds_hist = []
        self.mv_wait = 0
        self.mid = False
        self.stuck_t = 0
        self.stuck_level = 0
        self.back_hold = 0
        self.ret_n = 0
        self.plat_wait = 0
        self.bw_wait = 0
        self.pb_n = 0
        self.pb_idle = 0
        self.bill_n = 0
        self.plant_n = 0
        self.plat_air = False
        self.plat_air_n = 0
        self.ftracks = []
        self.fw_n = 0
        self.fw_idle = 0
        self.lc_n = 0
        self.walk_n = 0
        self.go_t = 0
        self.bh = []
        self.khigh = False
        self.kz_n = 0
        self.kz_walk = 0
        self.kz_back = 0
        self.pw_done = False
        self.pw_seen_up = False
        self.pw_n = 0

    def wx(self):
        return self.scroll + self.px

    def ceil_clear(self, img, ymin, px, front):
        y0 = max(40, ymin - 60)
        y1 = ymin - 1
        if y1 <= y0:
            return None
        x0 = max(0, px - 4)
        x1 = min(256, front + 50)
        reg = img[y0:y1, x0:x1]
        r = reg[..., 0]
        g = reg[..., 1]
        b = reg[..., 2]
        s = ((r == 228) & (g == 92) & (b == 16)) | ((r == 0) & (g == 0) & (b == 0))
        rs = s.sum(axis=1) >= 12
        i = len(rs) - 1
        while i >= 0 and rs[i]:
            i -= 1
        while i >= 0 and not rs[i]:
            i -= 1
        if i < 0:
            return None
        c = int(y1 - 1 - (y0 + i))
        return c if c >= 8 else None

    def pipe_height(self, img, x0, gtop):
        y0 = max(40, gtop - 100)
        reg = img[y0:gtop, x0:min(256, x0 + 20)].astype(np.int16)
        grey = (reg[..., 0] == reg[..., 1]) & (reg[..., 1] == reg[..., 2]) & (
            (reg[..., 0] == 124) | (reg[..., 0] == 188) | (reg[..., 0] == 252))
        n = grey.shape[0]
        h = np.zeros(grey.shape[1], np.int32)
        alive = np.ones(grey.shape[1], bool)
        for r in range(n - 1, -1, -1):
            alive &= grey[r]
            h += alive
        return int(h.max()) if len(h) else 0

    def profile_max(self, img, x0, gtop):
        x1 = min(256, x0 + PROFILE_LEN)
        y0 = max(40, gtop - 100)
        reg = img[y0:gtop, x0:x1]
        s = solid_mask(reg)
        n = s.shape[0]
        h = np.zeros(s.shape[1], np.int32)
        alive = np.ones(s.shape[1], bool)
        for r in range(n - 1, -1, -1):
            alive &= s[r]
            h += alive
        if len(h) < 10:
            return int(h.max()) if len(h) else 0
        hm = np.min(np.stack([h[i:len(h) - 9 + i] for i in range(10)]), axis=0)
        return int(hm.max())

    def track_goombas(self, img):
        wmax = WIDE_MAX if self.scroll + self.px > WIDE_LO else 20
        cur = ground_goombas(img, 209, wmax)
        new = []
        used = set()
        for g in cur:
            wpos = g[0] + self.scroll
            best = None
            for i, tr in enumerate(self.gtracks):
                if i in used:
                    continue
                if abs(wpos - tr[0]) <= 3:
                    best = i
                    break
            if best is None:
                new.append([wpos, wpos, 1])
            else:
                used.add(best)
                tr = self.gtracks[best]
                new.append([wpos, tr[1], tr[2] + 1])
        self.gtracks = new
        out = []
        for (wpos, w0, age), g in zip(new, cur):
            if age >= 5 and abs(wpos - w0) >= 2:
                out.append(g[0])
        return out

    def air_mover(self, img, front, ds2):
        if self.prev3 is None:
            return None
        xa_ = front + 4
        xb_ = min(256 - ds2, front + 4 + AIR_RANGE)
        if xb_ - xa_ <= 8:
            return None
        a_ = img[168:209, xa_:xb_]
        b_ = self.prev3[168:209, xa_ + ds2:xb_ + ds2]
        diff = (a_ != b_).any(axis=2)
        nd = int(diff.sum())
        if 5 <= nd <= MV_ND_MAX:
            idx = np.nonzero(diff.any(axis=0))[0]
            return int(idx[0]) + 4
        return None

    def step(self, sim, override=None):
        img = sim.screen()
        self.t += 1
        t = self.t
        if t == 1:
            reg = img[130:208, 0:80]
            brick = ((reg[..., 0] == 228) & (reg[..., 1] == 92) & (reg[..., 2] == 16)).sum()
            if brick < 800:
                self.scroll = 3000
                self.mid = True
        ds = 0
        if self.prev_img is not None:
            if np.array_equal(img[40:], self.prev_img[40:]) and not self.waiting:
                self.same += 1
            else:
                self.same = 0
            if self.px >= 100:
                ds = est_scroll(self.prev_img, img)
                self.scroll += ds
        self.ds_hist.append(ds)
        self.ds_hist = self.ds_hist[-3:]
        ds2 = sum(self.ds_hist)
        if self.same > 60:
            self.dead = True
        if self.scroll + self.px > MV_MIN_WX - 150:
            self.mover_x = self.track_goombas(img)
        else:
            self.mover_x = []
        if PLDBG and 770 < self.scroll + self.px < 900:
            gray_ = (img[150:175, :, 0] == 124) & (img[150:175, :, 1] == 124) & (img[150:175, :, 2] == 124)
            cols_ = np.nonzero(gray_.sum(axis=0) >= 8)[0]
            ptop_ = -1
            if len(cols_):
                reg_ = img[110:165, int(cols_.min()):int(cols_.max()) + 1]
                g_ = (reg_[..., 0] == 0) & (reg_[..., 1] == 168) & (reg_[..., 2] == 0)
                gy_ = np.nonzero(g_.any(axis=1))[0]
                ptop_ = int(gy_.min()) + 110 if len(gy_) else 999
            print("PL t=%d wx=%d py=%d top=%d pipex=%s" % (self.t, self.scroll + self.px, self.py, ptop_, (int(cols_.min()), int(cols_.max())) if len(cols_) else None))
        mb = find_mario(img, self.px, self.py)
        py_prev = self.prev_py
        keys = {"Right", "x"}
        if mb is not None:
            xmin, xmax, ymin, ymax = mb
            self.px = (xmin + xmax) // 2
            self.py = ymin
            px = self.px
            scroll = self.scroll
            front = xmax + 1
            gtop = ymin + 16
            brick_only = BRICKONLY_LO < scroll + px < EDGE3_HI and ymin < 135
            sm_ = brick_mask if brick_only else solid_mask   # a koopa head is not ground on the brick row
            fr_ = sm_(img[ymin + 12:min(240, ymin + 22), max(0, px - 3):px + 4])
            rws = np.nonzero(fr_.sum(axis=1) >= 4)[0]
            if (not (len(rws) and rws[0] <= 5)) and brick_only and EDGE3_LO - 30 < scroll + px:
                # toe-on-edge: use the full foot width so that the edge jump can still fire
                fr2_ = sm_(img[ymin + 12:min(240, ymin + 22), xmin + 3:max(xmin + 4, xmax - 2)])
                rws2_ = np.nonzero(fr2_.sum(axis=1) >= 2)[0]
                if len(rws2_) and rws2_[0] <= 5:
                    rws = rws2_
            if len(rws) and rws[0] <= 5:
                gtop = ymin + 12 + int(rws[0])
            if scroll >= 30 and scroll + px > LATE_WX - 150:
                # per-frame flyer tracking (also while airborne) so vy is known when Mario lands
                newtr = []
                for c in green_clusters(img, front - 10, front + FLY_RANGE, 40, gtop + 1):
                    if c[3] >= gtop - 10:
                        continue
                    wxa = c[0] + scroll
                    vy = 0.0
                    for tr in self.ftracks:
                        if abs(tr[0] - wxa) <= 7 and 0 < t - tr[3] <= 3:
                            dv = (c[3] - tr[1]) / float(t - tr[3])
                            vy = 0.6 * dv + 0.4 * tr[2]
                            break
                    newtr.append((wxa, c[3], vy, t))
                self.ftracks = newtr
            bills = find_bills(img)
            for b0, b1 in bills:
                if b1 < px - 4:
                    self.last_pass_t = t
            bill_right = any(b1 > front for b0, b1 in bills)
            on_ground = bool(len(rws) and rws[0] <= 5)
            stat1 = (py_prev is None or abs(self.py - py_prev) <= 2)
            if self.scroll + px > MV_MIN_WX:
                on_ground = on_ground and stat1
            self.on_ground = on_ground
            if self.jump_hold > 0:
                self.jt += 1
                if self.jt >= 5 and on_ground and stat1 and not self.vert:
                    self.jump_hold = 0
            else:
                self.jt = 0
                if on_ground and self.pend is None:
                    self.long_jump = False
            if self.pend is not None and not self.last_z:
                k_, vert_ = self.pend
                self.pend = None
                self.jump_hold = k_ - 1
                keys.add("z")
                self.vert = vert_
                if vert_:
                    keys.discard("Right")
                    keys.discard("x")
            elif self.jump_hold > 0:
                keys.add("z")
                self.jump_hold -= 1
                if self.vert:
                    keys.discard("Right")
                    keys.discard("x")
            elif on_ground:
                bx0 = min(255, front + 1)
                bx1 = min(256, front + 1 + LOOK + 10)
                body = solid_mask(img[ymin + 1:ymin + 15, bx0:bx1])
                if NOTAN and scroll + px > NOTAN_LO:
                    breg = img[ymin + 1:ymin + 15, bx0:bx1]
                    body = body & ~((breg[..., 0] == 240) & (breg[..., 1] == 208) & (breg[..., 2] == 176))
                trig = False
                need = 0
                why = ""
                c0 = None
                h = 0
                hmax = 0
                if body.shape[1] > 10:
                    cs = np.concatenate([np.zeros((body.shape[0], 1), np.int32), np.cumsum(body, axis=1)], axis=1)
                    w = (cs[:, 10:] - cs[:, :-10]) == 10
                    rows_hit = w[:, :LOOK].any(axis=1)
                    if rows_hit.sum() >= (ROWS_LATE if (scroll + px > NOTAN_LO and not (CAN_LO < scroll + px < CAN_HI)) else 6):
                        c0 = int(np.argmax(w[:, :LOOK].any(axis=0)))
                gb = None
                flyers = []
                pipe = False
                if scroll >= 30:
                    for cl in green_clusters(img, front - 10, front + FLY_RANGE, 40, gtop + 1):
                        xa, xb, ya, yb, n = cl
                        if yb < gtop - 10:
                            # on the ground under the brick row the row shields us from flyers/bros above it
                            if not (UR_LO < scroll + px < UR_HI and gtop >= 185 and yb < UR_Y):
                                flyers.append(cl)
                        elif xa - front <= GREEN_LOOK and yb >= ymin + 3 and xb > px:
                            below = img[min(239, yb + 1):min(240, yb + 4), xa:xb + 1]
                            bi = below.astype(np.int16)
                            grey = (bi[..., 0] == bi[..., 1]) & (bi[..., 1] == bi[..., 2]) & (bi[..., 0] >= 120)
                            if int(grey.sum()) < 4:
                                gb = cl
                    if c0 is None:
                        lk_p = PLANT_LOOK if (PLANT_LO < scroll + px < PLANT_HI and gtop >= 185) else LOOK
                        pm = pipe_mask(img[ymin + 1:ymin + 15, bx0:min(256, bx0 + lk_p)])
                        if pm.sum() >= 3:
                            c0 = int(np.argmax(pm.any(axis=0)))
                            pipe = True
                mv_d = None
                lane_only = bool(flyers) and LANE_LO < scroll + px < LANE_HI
                if scroll >= 30 and scroll + px > MV_MIN_WX and self.prev3 is not None and (not flyers or lane_only) \
                        and gb is None:
                    xa_ = front + 4
                    xb_ = min(256 - ds2, front + 4 + MV_RANGE)
                    ya_ = max(40, ymin - 50)
                    if lane_only:
                        ya_ = max(40, ymin - 2)
                    if xb_ - xa_ > 8 and gtop > ya_:
                        a_ = img[ya_:gtop, xa_:xb_]
                        b_ = self.prev3[ya_:gtop, xa_ + ds2:xb_ + ds2]
                        diff = (a_ != b_).any(axis=2)
                        nd = int(diff.sum())
                        self.info["nd"] = nd
                        self.info["ds2"] = ds2
                        if 5 <= nd <= MV_ND_MAX:
                            idx = np.nonzero(diff.any(axis=0))[0]
                            mv_d = int(idx[0]) + 4
                if scroll >= 30 and scroll + px > MV_MIN_WX and not flyers and gb is None:
                    sg = [x0 - front for x0 in self.mover_x if x0 >= front - 2 and x0 - front <= MV_RANGE]
                    if sg:
                        sd = max(0, min(sg))
                        mv_d = sd if mv_d is None else min(mv_d, sd)
                self.info["mv_d"] = mv_d
                self.info["fl"] = [tuple(int(v) for v in c[:4]) for c in flyers]
                self.info["gb"] = gb is not None
                if mv_d is None:
                    self.mv_wait = 0
                near_fly = []
                if scroll + px > LATE_WX:
                    near_fly = [c for c in flyers if 4 <= c[0] - front <= 90 and c[3] > ymin - 90]
                pit_ahead = False
                if (PITNF_LO < scroll + px < PITNF_HI or PITNF2_LO < scroll + px < PITNF2_HI) and gtop >= 200:
                    grow_ = solid_mask(img[min(239, gtop + 1):min(240, gtop + 4), front:min(256, front + 100)]).any(axis=0)
                    pit_ahead = int((~grow_).sum()) >= 8
                if pit_ahead:
                    near_fly = []
                    gb = None
                    if PITNF2_LO < scroll + px < PITNF2_HI:
                        mv_d = None
                if scroll + px < PB_ZONE1:
                    pb_d = [c[0] - front for c in near_fly if c[3] >= ymin - PB_Y1 and c[0] - front <= PB_D1]
                else:
                    pb_d = [c[0] - front for c in near_fly if c[3] >= ymin - PB_Y and c[0] - front <= PB_D]
                if pb_d:
                    self.pb_idle = 0
                else:
                    self.pb_idle += 1
                    if self.pb_idle > 20:
                        self.pb_n = 0
                # --- flyer tracking (world x, bottom y, vy) and danger prediction
                fl_v = []
                for c in flyers:
                    wxa = c[0] + scroll
                    vy = 0.0
                    for tr in self.ftracks:
                        if abs(tr[0] - wxa) <= 7 and tr[3] == t:
                            vy = tr[2]
                            break
                    fl_v.append((c, vy))
                fdanger = None
                if FW_ON and scroll + px > LATE_WX and gtop >= 185 and not pit_ahead and not (NOFW_LO < scroll + px < NOFW_HI):
                    spd = 3.0 if (self.ds_hist and self.ds_hist[-1] >= 2) else 2.2
                    for c, vy in fl_v:
                        d_ = c[0] - front
                        if d_ < -16 or d_ > FW_RANGE:
                            continue
                        if 0 <= d_ < FW_COMMIT and self.ds_hist and self.ds_hist[-1] >= 2:
                            continue   # too close to stop (skid ~40 px): commit to running under it
                        te = max(0, d_) / (spd + 0.5)
                        hi = te + 9.0
                        lo = max(0.0, te - 3.0)
                        if SIN_ON and SIN_LO < scroll + px < SIN_HI:
                            # cannon-zone paratroopa: vertical sinusoid, hold until the flyer is clearly rising, then commit
                            if self.go_t > t or c[3] < SIN_HIGH:
                                continue
                            wxa_ = c[0] + scroll
                            self.bh = [e for e in self.bh if t - e[0] <= 10 and abs(e[2] - wxa_) < 14]
                            self.bh.append((t, c[3], wxa_))
                            old_ = [e for e in self.bh if t - e[0] >= 5]
                            rising = bool(old_) and old_[0][1] - c[3] >= 7
                            if rising and 0 <= d_ <= SIN_GOD:
                                self.go_t = t + SIN_GO
                                continue
                            fdanger = d_ if fdanger is None else min(fdanger, d_)
                            continue
                        if vy > 0.5:
                            eb = min(200.0, c[3] + vy * hi)
                        elif vy < -0.5:
                            eb = c[3] + vy * lo
                        else:
                            eb = c[3]
                        if eb >= ymin - (FW_M_FAST if vy > 1.5 else FW_M):
                            fdanger = d_ if fdanger is None else min(fdanger, d_)
                lowceil = False
                lowfl = [c for c in near_fly if c[3] >= ymin - 30 and c[0] - front <= 60]
                if gb is not None and gb[0] - front <= 60:
                    lowfl.append(gb)
                if FW_ON and CEIL_ON and 1100 < scroll + px < CEIL_HI and lowfl and self.lc_n < LC_MAX and gtop >= 185:
                    cl0 = self.ceil_clear(img, ymin, px, front)
                    lowceil = cl0 is not None and cl0 < FW_CEIL and min(c[0] - front for c in lowfl) > LC_D
                if fdanger is None:
                    self.fw_idle += 1
                    if self.fw_idle > 30:
                        self.fw_n = 0
                else:
                    self.fw_idle = 0
                rk_d = None
                if STOMPZ_LO < scroll + px < STOMPZ_HI and gtop <= 110:
                    # red koopa (same red as Mario's shirt but to the right of him) on the upper platform
                    rreg = img[max(40, ymin - 6):ymin + 14, min(255, front + 2):min(256, front + 70)]
                    rm_ = (rreg[..., 0] == 248) & (rreg[..., 1] == 56) & (rreg[..., 2] == 0)
                    if int(rm_.sum()) >= 6:
                        rk_d = int(np.argmax(rm_.any(axis=0))) + 2
                kz_d = None
                if KZ_ON and KZ_LO < scroll + px < KZ_HI and gtop >= 185 and self.kz_n < KZ_MAX:
                    kzc = [c for c in green_clusters(img, front - 6, front + KZ_RANGE, 150, 212) if c[3] >= 192]
                    if kzc:
                        kz_d = min(c[0] for c in kzc) - front
                        if kz_d < KZ_BEHIND:
                            kz_d = None   # koopa is beside/under us (we stand on it / passed it): do not re-stomp forever
                if scroll < 30:
                    pass
                elif kz_d is not None:
                    # ground koopa walking at us under the low brick row: stand still, hop-stomp it when it is close
                    self.kz_n += 1
                    if kz_d < KZ_MIN and self.kz_back < KZ_BACK_MAX:
                        # landed right beside it: a hop now is hit during the rise -> back off to open a gap first
                        self.kz_back += 1
                        keys.discard("Right")
                        keys.discard("x")
                        keys.add("Left")
                        why = "kback"
                    elif kz_d <= KZ_HOP:
                        trig = True
                        need = 40
                        why = "kstomp d=%d" % kz_d
                    else:
                        keys.discard("Right")
                        keys.discard("x")
                        why = "kwait"
                    if DEBUG and self.kz_n == 1:
                        print("t=%d koopa zone wx=%d d=%d" % (t, scroll + px, kz_d))
                elif PB_LO < scroll + px < PB_HI and gtop >= 185 and pb_d and self.pb_n < PB_MAX:
                    self.pb_n += 1
                    keys.discard("Right")
                    keys.discard("x")
                    keys.add("Left")
                    why = "paraback"
                    if DEBUG and self.pb_n == 1:
                        print("t=%d para back-off wx=%d d=%s" % (t, scroll + px, pb_d))
                elif lowceil:
                    # a low flyer / ground paratroopa under the '?'-block row: back off until the ceiling is gone
                    self.lc_n += 1
                    keys.discard("x")
                    if self.ds_hist and self.ds_hist[-1] >= 2.5:
                        keys.discard("Right")   # coast down to walking pace, then creep up to the flyer
                    why = "lcback"
                    if DEBUG and self.lc_n == 1:
                        print("t=%d lowceil back-off wx=%d low=%s" % (t, scroll + px, [(c[0] - front, c[2], c[3]) for c in lowfl[:2]]))
                elif rk_d is not None and getattr(self, "rw_n", 0) < 150:
                    # red koopa on the upper brick platform: let it come, then stomp it with a vertical hop
                    self.rw_n = getattr(self, "rw_n", 0) + 1
                    if rk_d <= STOMPZ_D:
                        trig = True
                        need = 45
                        why = "stomp redk"
                    else:
                        keys.discard("Right")
                        keys.discard("x")
                        why = "redwait"
                elif scroll + px > LATE_WX and not lowceil and not (SIN_ON and SIN_LO < scroll + px < SIN_HI and fdanger is not None) and min(
                        [c[0] - front for c in near_fly if c[3] >= ymin - (FYZ_M if FYZ_LO < scroll + px < FYZ_HI else 12)]
                        + ([gb[0] - front] if gb is not None else []) + [999]) <= \
                        (PLAT_D if (PLAT_LO - 40 < scroll + px < EDGE3_HI and gtop <= 150) else FLY_JUMP_D):
                    need = 99 if (gtop >= 180 or scroll + px < 1700) else ELEV_NEED
                    if FLY2_LO < scroll + px < FLY2_HI and gtop >= 180:
                        need = FLY2_NEED
                    trig = True
                    why = "flyjump"
                    if PLAT_LO - 40 < scroll + px < EDGE3_HI and gtop <= 150:
                        need = 40
                        why = "stomp plat"
                elif fdanger is not None and self.fw_n < FW_MAX and (mv_d is None or mv_d > 40):
                    self.fw_n += 1
                    keys.discard("Right")
                    keys.discard("x")
                    why = "fwait"
                    ds_last = self.ds_hist[-1] if self.ds_hist else 0
                    lowh = [c[0] - front for c, vy in fl_v if LH_LO <= c[0] - front <= LH_HI and c[2] >= ymin - 50
                            and abs(vy) < 1.0 and c[3] >= ymin - 20]
                    lowc = [c[0] - front for c, vy in fl_v if LH_HI < c[0] - front <= 90 and abs(vy) < 1.0
                            and c[3] >= ymin - 30]
                    if SIN_ON and SIN_LO < scroll + px < SIN_HI:
                        lowh = []
                        lowc = []
                        if fdanger > SIN_CREEP and ds_last < 1.5:
                            keys.add("Right")   # walk up so that the flyer's next rising phase finds us close
                    if lowc and not lowh and ds_last < 1.5 and FW_CREEP:
                        keys.add("Right")
                    if lowh and ds_last <= 1 and FW_JUMP and not lowceil:
                        trig = True
                        need = 99
                        why = "flyjump"
                        keys.add("Right")
                        keys.add("x")
                    elif ds_last >= 1.5 or (fdanger < 16 and not (SIN_ON and SIN_LO < scroll + px < SIN_HI)):
                        keys.add("Left")
                    if DEBUG and self.fw_n == 1:
                        print("t=%d fly wait wx=%d d=%s fl=%s" % (t, scroll + px, fdanger, [(c[0] - front, c[2], c[3], round(v, 1)) for c, v in fl_v[:2]]))
                elif mv_d is not None and not pipe and mv_d <= ((SZ_D if (SZ_LO < scroll + px < 1340 or 1560 < scroll + px < SZ_HI) else HOP_D) if (self.ds_hist and self.ds_hist[-1] < 1.5) else
                                                                (MVZ_D if MVZ_LO < scroll + px < MVZ_HI else
                                                                 (MVZ2_D if MVZ2_LO < scroll + px < MVZ2_HI else HOP_D_FAST))):
                    trig = True
                    if self.ds_hist and self.ds_hist[-1] < 1.5:
                        need = HOP_NEED
                        why = "stomp mv=%d" % mv_d
                    else:
                        need = HOP_NEED_FAST
                        if MVZ_LO < scroll + px < MVZ_HI:
                            need = MVZ_NEED
                        why = "hop mv=%d" % mv_d
                elif mv_d is not None and self.mv_wait < MV_WAIT_MAX and not (MVZ_LO < scroll + px < MVZ_HI) \
                        and not (EDGE3_LO - 40 < scroll + px < EDGE3_HI and gtop <= 150):
                    self.mv_wait += 1
                    keys.discard("Right")
                    keys.discard("x")
                    why = "mvwait"
                elif gb is not None and PLAT_LO - 40 < scroll + px < EDGE3_LO - 40 and gtop <= 150:
                    self.plat_wait += 1
                    keys.discard("x")
                    if self.plat_wait < PLAT_WAIT_MAX:
                        keys.discard("Right")
                    why = "platwait"
                elif gb is not None:
                    need = 90
                    trig = True
                    why = "enemy gb=%s" % (gb,)
                elif c0 is not None:
                    col = bx0 + c0 + 2
                    if pipe:
                        col = bx0 + c0
                    if pipe:
                        h = self.pipe_height(img, bx0 + c0, gtop)
                    else:
                        yy = gtop - 1
                        while yy > 40 and solid_mask(img[yy:yy + 1, col:col + 1])[0, 0]:
                            yy -= 1
                        top = yy + 1
                        h = gtop - top
                    if h <= 0:
                        h = 16
                    need = max(h + 10, 40)
                    if pipe:
                        need = PIPE_NEED
                        if PIPEZ_LO < scroll + px < PIPEZ_HI and gtop >= 185 and c0 >= PIPEZ_MIND:
                            need = PIPEZ_NEED   # stay under the brick row: do not land on the platform
                            if KHIGH_ON:
                                # a ground koopa beyond the pipe: the low hop lands beside it under the low brick row -> high route over the row
                                kh_ = [c for c in green_clusters(img, front + c0 + 8, 256, 150, 212)
                                       if c[3] >= 190 and c[2] >= 165]
                                if KHIGH_ALWAYS or (kh_ and min(c[0] for c in kh_) - front <= KHIGH_D):
                                    need = 99
                                    self.khigh = True
                    hmax = h
                    narrow = False
                    if h >= 14 and not pipe:
                        hmax = max(h, self.profile_max(img, front + 1 + c0, gtop))
                        if hmax > h:
                            need = min(hmax + STAIR_MARGIN, 99)
                    if not pipe and h <= 20 and hmax <= 20:
                        rr = min(239, gtop - 6)
                        row = solid_mask(img[rr:rr + 1, bx0 + c0:min(256, bx0 + c0 + 60)])[0]
                        nz = np.nonzero(~row)[0]
                        wrun = int(nz[0]) if len(nz) else len(row)
                        after = row[wrun:wrun + 6]
                        if wrun <= 22 and len(after) >= 4 and not after.any():
                            narrow = True
                            need = max(need, NARROW_NEED)
                    if END_LO < scroll + px < END_HI and gtop >= 185 and not pipe:
                        need = 99
                    trig = h >= 4
                    if trig and pipe and self.khigh and PIPEZ_LO < scroll + px < PIPEZ_HI and gtop >= 185 and c0 > KH_D:
                        trig = False   # walking-pace high hop: take off close to the pipe so the row's left end is reached while still high
                    why = "wall%s%s h=%d hmax=%d d=%d" % (" pipe" if pipe else "", " narrow" if narrow else "", h, hmax, c0)
                    if not pipe and c0 >= 4:
                        sreg = solid_mask(img[min(239, gtop):min(240, gtop + 3), bx0:bx0 + c0])
                        if sreg.shape[1] >= 3:
                            nog = np.nonzero(~sreg.any(axis=0))[0]
                            if len(nog) >= 3:
                                need = max(need, PITWALL_NEED)
                                why += " pitgap e=%d" % int(nog[0])
                                if int(nog[0]) > PITWALL_E:
                                    trig = False
                else:
                    pl_ = PITLOW_LOOK if PITLOW_LO < scroll + px < PITLOW_HI else PIT_LOOK
                    px0 = min(255, front + pl_)
                    px1 = min(256, front + pl_ + 6)
                    colsolid = solid_mask(img[min(239, gtop):240, px0:px1]).any(axis=0)
                    if px1 > px0 and not colsolid.any():
                        trig = True
                        need = 70
                        why = "pit"
                        far = solid_mask(img[max(40, gtop - 52):max(41, gtop - 3), px0:min(256, px0 + 64)])
                        if int(far.sum()) >= 60:
                            need = PW2_NEED if PW2_LO < scroll + px < PW2_HI else 99
                            why = "pitwall"
                    elif gtop <= 200 and px1 > px0 and scroll + px < LEDGE_MAX_WX:
                        plat = solid_mask(img[gtop:gtop + 3, min(255, front + 2):min(256, front + 12)]).any()
                        if not plat:
                            low = solid_mask(img[188:207, min(255, front):min(256, front + 70)])
                            late_edge = EDGE2_LO < scroll + px < EDGE2_HI and gtop <= 150
                            if late_edge and low.sum() >= 14:
                                sup2 = solid_mask(img[min(239, ymin + 16):min(240, ymin + 19), xmin:xmax + 1]).any(axis=0)
                                if int(sup2.sum()) > EDGE_COLS2:
                                    low = low[:0]
                            if low.sum() >= 14 and not (CANNON_LO < scroll + px < CANNON_HI and gtop >= 150) \
                                    and not (NOJ_LO < scroll + px < NOJ_HI and gtop <= 125):
                                trig = True
                                need = 99
                                why = "ledge"
                if ROWJ_ON and not trig and ROWJ_LO < scroll + px < ROWJ_HI and ROWJ_Y0 <= ymin <= ROWJ_Y1:
                    # arrived on the upper floating-brick row late (lower route): its right end drops us beside the goombas -> take the high jump now
                    trig = True
                    need = 99
                    why = "rowjump"
                if EDGE3_FLO < scroll + px < EDGE3_HI and gtop <= 150 and ymin < 100 and not why.startswith("stomp"):
                    trig = False
                    why = ""
                elif EDGE3_LO < scroll + px < EDGE3_HI and gtop <= 150 and not why.startswith("fly") \
                        and not why.startswith("stomp plat"):
                    y0e = min(239, ymin + 16)
                    sup_e = solid_mask(img[y0e:min(240, y0e + 3), :]).any(axis=0)
                    xe = xmax - 4
                    while xe < 253 and (sup_e[xe] or sup_e[xe + 1] or sup_e[xe + 2]):
                        xe += 1
                    de = xe - xmax
                    if de <= EDGE3_D:
                        trig = True
                        need = 49 if (self.ds_hist and max(self.ds_hist) >= 2.5) else EDGE3_SLOW
                        why = "edge3 d=%d" % de
                    elif trig and (why in ("ledge", "pit", "pitwall") or why.startswith("unstick")):
                        trig = False
                        why = ""
                if trig and EDGE3_LO - 30 < scroll + px < EDGE3_HI and 100 <= ymin and gtop <= 150 \
                        and why.startswith(("stomp", "hop")) and (not why.startswith("stomp plat") or scroll + px > 1950):
                    # lower brick row ends in a pit: a vertical stomp kills the run-up and drops us in the pit -> forward hop
                    need = 71
                    why = "edge3 hop"
                if PITB_LO < scroll + px < PITB_HI and gtop <= 190 and not trig and not why.startswith(("fly", "stomp", "hop")):
                    # floating bricks over the wide pit: full jump when the brick support ends under the front foot
                    y0b = min(239, ymin + 16)
                    sup_b = solid_mask(img[y0b:min(240, y0b + 3), :]).any(axis=0)
                    xb = xmax - 4
                    while xb < 253 and (sup_b[xb] or sup_b[xb + 1] or sup_b[xb + 2]):
                        xb += 1
                    if xb - xmax <= PITB_D:
                        trig = True
                        need = 99
                        why = "brickedge d=%d" % (xb - xmax)
                if RET_LO < scroll + px < RET_HI and gtop <= 150 and not trig and self.ret_n < RET_MAX:
                    rk = [c for c in flyers if -4 <= c[0] - front <= RET_D and c[2] < ymin + 5]
                    if rk:
                        self.ret_n += 1
                        keys.discard("Right")
                        keys.discard("x")
                        keys.add("Left")
                        if DEBUG and self.ret_n == 1:
                            print("t=%d retreat wx=%d fl=%s" % (t, scroll + px, rk[:1]))
                if PW_ON and (not trig) and PW_LO < scroll + px < PW_HI and gtop >= 185 and not self.pw_done:
                    # piranha pipe: stop short of it and start the run-up when the plant has just gone down (down phase ~76 f, up ~116 f)
                    gm_ = (img[165:190, :, 0] == 124) & (img[165:190, :, 1] == 124) & (img[165:190, :, 2] == 124)
                    cols_ = np.nonzero(gm_.sum(axis=0) >= 12)[0]
                    cols_ = cols_[cols_ > front]
                    if len(cols_):
                        x0p = int(cols_.min())
                        dist_ = x0p - front
                        if PW_DMIN <= dist_ <= PW_DMAX:
                            reg_ = img[110:162, x0p:min(256, x0p + 20)]
                            cnt_ = int(((reg_[..., 0] == 0) & (reg_[..., 1] == 168) & (reg_[..., 2] == 0)).sum())
                            if cnt_ < 6:
                                if self.pw_seen_up:
                                    self.pw_done = True
                                    if DEBUG:
                                        print("t=%d plant down -> go wx=%d dist=%d waited=%d" % (t, scroll + px, dist_, self.pw_n))
                            else:
                                self.pw_seen_up = True
                            if not self.pw_done and self.pw_n < PW_MAX:
                                self.pw_n += 1
                                keys.discard("Right")
                                keys.discard("x")
                                if self.ds_hist and self.ds_hist[-1] >= 1:
                                    keys.add("Left")
                                why = "plantw"
                                if DEBUG and self.pw_n == 1:
                                    print("t=%d plant wait start wx=%d dist=%d cnt=%d" % (t, scroll + px, dist_, cnt_))
                if trig and why.startswith("wall") and scroll >= 30:
                    fl = [c for c in flyers if c[1] > px - 10]
                    if hmax >= 40 and fl and "pipe" not in why:
                        if any(c[3] < GO_Y for c in fl) or self.fly_wait >= STAIR_WAIT_MAX:
                            fl = []
                        elif self.fly_wait < STAIR_WAIT_MAX:
                            self.fly_wait += 1
                            keys.discard("Right")
                            keys.discard("x")
                            trig = False
                            fl = []
                            if DEBUG and self.fly_wait == 1:
                                print("t=%d stair wait wx=%d" % (t, scroll + px))
                    if fl and "pipe" not in why and self.fly_wait < FLY_WAIT_MAX:
                        self.fly_wait += 1
                        keys.discard("Right")
                        keys.discard("x")
                        trig = False
                        if DEBUG and self.fly_wait == 1:
                            print("t=%d flyer wait fl=%s wx=%d" % (t, fl, scroll + px))
                if trig and why.startswith("wall") and gtop >= 205 and 100 < scroll + px < 235:
                    ok = (not bill_right) and (t - self.last_pass_t < 100)
                    if not ok:
                        if DEBUG and not self.waiting:
                            print("t=%d gate wait (bills=%s) pos=%d wx=%d" % (t, bills, px, scroll + px))
                        self.waiting = True
                        keys.discard("Right")
                        keys.discard("x")
                        trig = False
                    else:
                        if self.waiting and DEBUG:
                            print("t=%d gate open wx=%d" % (t, scroll + px))
                        self.waiting = False
                else:
                    self.waiting = False
                if trig and why.startswith("wall pipe") and 800 < scroll + px < 900 and gtop >= 185 and c0 is not None \
                        and c0 > 6 and len(self.ds_hist) >= 3 and sum(self.ds_hist) < 8 and getattr(self, "pspd_n", 0) < 40:
                    # piranha pipe: a slow run-up gives a short jump that lands on the plant; keep accelerating first
                    self.pspd_n = getattr(self, "pspd_n", 0) + 1
                    trig = False
                    why = "pipespeed"
                if trig and why.startswith("wall pipe") and PIPEZ_LO < scroll + px < PIPEZ_HI and gtop >= 185 \
                        and getattr(self, "pk_n", 0) < PK_MAX:
                    kc = [c for c in green_clusters(img, front, min(256, front + 125), 130, 212)
                          if c[3] >= PK_Y and c[0] > front + c0 + 20]
                    if kc:
                        self.pk_n = getattr(self, "pk_n", 0) + 1
                        keys.discard("Right")
                        keys.discard("x")
                        trig = False
                        why = "pkwait"
                if trig and why.startswith("wall pipe") and PBILL_LO < scroll + px < PBILL_HI and gtop >= 185:
                    lane = img[100:141, :]
                    dk = ((lane[..., 0] == 0) & (lane[..., 1] == 0) & (lane[..., 2] == 0)).sum(axis=0) >= 2
                    dk[:min(256, front - 8)] = False
                    if dk.any():
                        bill_right = True
                if trig and why.startswith("wall pipe") and PBILL_LO < scroll + px < PBILL_HI and gtop >= 185 \
                        and bill_right and self.bill_n < PBILL_MAX:
                    self.bill_n += 1
                    keys.discard("Right")
                    keys.discard("x")
                    trig = False
                    why = "billwait"
                if (BW_LO < scroll + px < BW_HI or (BWE_ON and 2200 < scroll + px < BW_LO and why.startswith("brickedge"))) \
                        and gtop <= 170 and (not trig or (BWE_ON and gtop >= 130 and scroll + px < 2420 and why.startswith(("brickedge", "pitwall")))) \
                        and self.bw_wait < BW_MAX:
                    bwm = [x0 for x0 in self.mover_x if front - 6 <= x0 <= front + BW_RANGE]
                    amd_b = self.air_mover(img, front, ds2)
                    if amd_b is not None:
                        bwm = bwm + [front + amd_b]
                    if bwm:
                        self.bw_wait += 1
                        keys.discard("Right")
                        keys.discard("x")
                        trig = False
                        why = "bwait"
                        if DEBUG and self.bw_wait == 1:
                            print("t=%d brick wait wx=%d movers=%s" % (t, scroll + px, bwm))
                self.last_why = "%s|trig=%s|mv=%s|gb=%s|fl=%s|nf=%s" % (why, trig, mv_d, None if gb is None else (gb[0] - front, gb[2], gb[3]), [(c[0] - front, c[2], c[3]) for c in flyers[:2]], len(near_fly))
                gated = (not trig) and (why.startswith("wall") or why in ("mvwait", "fwait", "bwait", "paraback", "billwait", "plantwait", "lcback", "pkwait", "redwait", "kwait", "kback", "plantw") or self.waiting)
                wxv = scroll + px
                if wxv > self.best_wx + 2:
                    self.best_wx = wxv
                    self.stuck_t = 0
                    self.stuck_level = 0
                elif gated:
                    self.stuck_t = 0
                else:
                    self.stuck_t += 1
                if self.stuck_t >= (CAN_STUCK if 1655 < wxv < 1700 else STUCK_N) and not trig:
                    self.stuck_t = 0
                    lvl = self.stuck_level % 3
                    self.stuck_level += 1
                    if DEBUG:
                        print("t=%d STUCK level %d wx=%d" % (t, lvl, wxv))
                    if lvl == 0:
                        trig = True
                        need = 70
                        why = "unstick14"
                    elif lvl == 1:
                        self.back_hold = 25
                    else:
                        trig = True
                        need = 99
                        why = "unstick26"
                if trig and FYZ_LO < scroll + px < FYZ_HI and gtop < 185:
                    trig = False
                    why = ""
                ceilcap = False
                if trig and CEIL_ON and 1100 < scroll + px < CEIL_HI and why.startswith(("flyjump", "hop", "stomp", "enemy")):
                    cl_ = self.ceil_clear(img, ymin, px, front)
                    if cl_ is not None and need > cl_ + 4:
                        need = cl_ + 2
                        ceilcap = True
                        if DEBUG and self.nlog < 400:
                            print("t=%d ceiling clear=%d -> need=%d (%s) wx=%d" % (t, cl_, need, why, scroll + px))
                if trig:
                    self.fly_wait = 0
                    k = max(MIN_K, pick_k(need)) if why != "pit" else PIT_K
                    if ceilcap:
                        k = max(2, pick_k(need))
                        if 1200 < scroll + px < 1340:
                            self.walk_n = 26
                    if why.startswith("kstomp"):
                        k = KZ_K
                        self.walk_n = 30
                    if why == "pit" and PK3_LO < scroll + px < PK3_HI:
                        k = PK3_K
                    if why == "pit" and PITB_LO < scroll + px < PITB_HI and gtop <= 190:
                        k = 26
                    if why == "ledge" and LEDGEK_LO < scroll + px < LEDGEK_HI and 120 < gtop <= 175:
                        k = LEDGEK   # ground is only ~20px beyond the floating bricks: a long jump overshoots onto the goombas
                    if why == "pit" and PITLOW_LO < scroll + px < PITLOW_HI:
                        k = PITLOW_K
                    if why.startswith("brickedge") and BE_LO < scroll + px < BE_HI:
                        k = BE_K   # lower floating-brick row: aim at the mid-pit upper row (as the ground route does), not at the ground beyond the goombas
                    if why == "stomp plat" and PLATAIR_ON:
                        self.plat_air = True
                        self.plat_air_n = 0
                    self.long_jump = (why in ("pit", "pitwall", "ledge", "flyjump") or "pitgap" in why
                                      or why.startswith("unstick") or "pipe" in why)
                    if self.last_z:
                        self.pend = (k, why.startswith("stomp"))
                    else:
                        self.jump_hold = k - 1
                        keys.add("z")
                        self.vert = why.startswith("stomp")
                        if self.vert:
                            keys.discard("Right")
                            keys.discard("x")
                    self.nlog += 1
                    if DEBUG and self.nlog < 400:
                        print("t=%d jump k=%d %s pos=(%d,%d) wx=%d khigh=%s ds=%s" % (t, k, why, px, self.py, scroll + px, self.khigh, self.ds_hist))
            if (not on_ground) and self.jump_hold == 0 and AIR_BRAKE and (MV_MIN_WX < scroll + px < LATE_WX or AB2_LO < scroll + px < AB2_HI) \
                    and py_prev is not None and self.py > py_prev + 1 \
                    and ymin + 15 < (207 if AB2_LO < scroll + px < AB2_HI else 197):
                amd = self.air_mover(img, front, ds2)
                in_ab2 = AB2_LO < scroll + px < AB2_HI
                if in_ab2 and AB2_STATIC:
                    # also react to goombas that merely stand / walk on the landing ground (diff detector misses them)
                    sgm = [g[0] - front for g in ground_goombas(img, 209, 52) if g[0] >= front - 2 and g[0] - front <= AB2_PD + 30]
                    if sgm:
                        sm0 = max(0, min(sgm))
                        amd = sm0 if amd is None else min(amd, sm0)
                self.info["amd"] = amd
                vy = self.py - py_prev
                if amd is not None and amd <= (AB2_PD if in_ab2 else PREPRESS_D) and vy >= 3 \
                        and self.py + PREPRESS_F * vy >= 193 and self.py <= 196:
                    self.jump_hold = PREPRESS_K - 1
                    self.jt = 0
                    self.vert = False
                    keys.add("z")
                    if DEBUG and self.nlog < 400:
                        self.nlog += 1
                        print("t=%d prepress amd=%d pos=(%d,%d) wx=%d" % (t, amd, px, self.py, scroll + px))
                elif amd is not None and amd <= AIR_BRAKE_D and (AB2_BRAKE or not in_ab2):
                    keys.discard("Right")
                    keys.discard("x")
                    if amd <= AIR_LEFT_D:
                        keys.add("Left")
                    if DEBUG and in_ab2 and self.nlog < 400:
                        self.nlog += 1
                        print("t=%d ab2 brake amd=%d pos=(%d,%d) wx=%d" % (t, amd, px, self.py, scroll + px))
            if PIPEAIR_ON and (not self.khigh) and (not on_ground) and self.jump_hold == 0 and PIPEZ_LO < scroll + px < PIPEZ_HI + 40 \
                    and py_prev is not None and self.py > py_prev and ymin > 140:
                kc_ = [c for c in green_clusters(img, front, min(256, front + 70), 180, 212) if c[3] >= 188]
                if kc_:
                    keys.discard("Right")
                    keys.discard("x")
                    if min(c[0] for c in kc_) - front <= PIPEAIR_LEFT:
                        keys.add("Left")
            if (not on_ground) and self.jump_hold == 0 and PLAT_LO < scroll + px < PLAT_HI \
                    and py_prev is not None and self.py > py_prev - 1 and ymin < 135:
                kz = [c for c in green_clusters(img, front - 6, front + PLAT_RANGE, 100, 150)
                      if c[3] <= 150 and c[2] >= 100 and c[0] >= front + 2]
                if kz:
                    kd = min(c[0] for c in kz) - front
                    keys.discard("Right")
                    keys.discard("x")
                    if kd <= PLAT_LEFT:
                        keys.add("Left")
                    if DEBUG and self.nlog < 400:
                        self.nlog += 1
                        print("t=%d platbrake kd=%d pos=(%d,%d) wx=%d" % (t, kd, px, self.py, scroll + px))
            if (not on_ground) and self.jump_hold == 0 and EDGE3_LO - 70 < scroll + px < EDGE3_HI \
                    and ymin < 135 and py_prev is not None and self.py >= py_prev:
                sup_a = solid_mask(img[146:149, :]).any(axis=0)
                xe_ = max(0, xmax - 4)
                while xe_ < 253 and (sup_a[xe_] or sup_a[xe_ + 1] or sup_a[xe_ + 2]):
                    xe_ += 1
                de_ = xe_ - xmax
                if 0 < de_ < 130:
                    vy_ = max(self.py - py_prev, 2)
                    drop_ = 145 - (ymin + 16)
                    if drop_ > 0:
                        T_ = drop_ / float(vy_ + 2)
                        s_ = float(self.ds_hist[-1]) if self.ds_hist else 3.0
                        rem_ = de_ - s_ * T_
                        if rem_ < ABRAKE_M:
                            keys.discard("Right")
                            keys.discard("x")
                            if rem_ < 0:
                                keys.add("Left")
                            if DEBUG and self.nlog < 400:
                                self.nlog += 1
                                print("t=%d edgebrake de=%d rem=%.0f pos=(%d,%d) wx=%d" % (t, de_, rem_, px, self.py, scroll + px))
            if HOLD_A_DESC and (not on_ground) and self.jump_hold == 0 and EDGE3_LO - 30 < scroll + px < EDGE3_HI \
                    and ymin < 150 and py_prev is not None and self.py > py_prev:
                keys.add("z")   # hold A while falling: a stomp bounce is higher/longer, giving the range to cross the pit
            if self.plat_air:
                self.plat_air_n += 1
                if on_ground and self.plat_air_n > 4:
                    self.plat_air = False
                elif not on_ground and self.jump_hold == 0:
                    keys.discard("Right")
                    keys.discard("x")
                    keys.add("Left")
        self.prev2_py = py_prev
        self.prev_py = self.py
        pc = getattr(self, "pause_cnt", 0)
        if pc > 0:
            self.pause_cnt = pc - 1
            keys = set()
            self.jump_hold = 0
            self.pend = None
        elif (not getattr(self, "pause_done", False)) and self.scroll + self.px >= PAUSE_WX \
                and getattr(self, "on_ground", False) and self.py > 188 and self.scroll + self.px < PAUSE_WX + 60:
            self.pause_done = True
            self.pause_cnt = PAUSE_N
        if self.walk_n > 0:
            self.walk_n -= 1
            keys.discard("x")
        if WALK_LO < self.scroll + self.px < WALK_HI:
            keys.discard("x")
        if NOX_LO < self.scroll + self.px < NOX_HI:
            keys.discard("x")   # walking-pace hop over the x1790 pipe: shorter flight, lands before the koopa
        if self.back_hold > 0:
            self.back_hold -= 1
            keys.discard("Right")
            keys.discard("z")
            keys.add("Left")
            self.jump_hold = 0
        if override is not None:
            keys = set(override)
        self.last_z = "z" in keys
        if any(a < self.scroll + self.px < b for a, b in _DBG):
            print("D t=%d wx=%d p=(%d,%d) og=%s jh=%d k=%s %s" % (self.t, self.scroll + self.px, self.px, self.py, getattr(self, "on_ground", None), self.jump_hold, "".join(sorted(k[0] for k in keys)), getattr(self, "last_why", "")))
        self.last_why = ""
        sim.step(keys)
        self.hist.append(img)
        self.hist = self.hist[-3:]
        self.prev3 = self.hist[0] if len(self.hist) == 3 else None
        self.prev_img = img


EXIT_ON_DEAD = False


def play(sim):
    p = Pilot()
    while True:
        p.step(sim)
        if p.dead and EXIT_ON_DEAD:
            print("dead/frozen t=%d wx=%d" % (p.t, p.wx()))
            return

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': '"""State assessment for SMB 5-2 (pixels only): cheap heuristics, no keys."""\nimport numpy as np\n\n\ndef assess(frame, ctx):\n    try:\n        f = np.asarray(frame)\n        sky = (f[..., 0] == 104) & (f[..., 1] == 136) & (f[..., 2] == 252)\n        sky_frac = float(sky[40:].mean())\n        red = (f[..., 0] == 248) & (f[..., 1] == 56) & (f[..., 2] == 0)\n        mario = int(red[40:].sum())\n        green = (f[..., 0] == 0) & (f[..., 1] == 168) & (f[..., 2] == 0)\n        ngreen = int(green[40:].sum())\n        stuck = ctx.get("stuck") or {}\n        is_stuck = bool(stuck.get("is")) if isinstance(stuck, dict) else False\n        status = "running"\n        if mario < 6 and sky_frac > 0.2:\n            status = "dead"\n        elif is_stuck:\n            status = "stuck"\n        elif ngreen > 60:\n            status = "danger"\n        return {\n            "status": status,\n            "where": {"hazard": "green_enemy" if ngreen > 60 else None, "confidence": 0.5},\n            "progress": None,\n            "resources": {},\n            "threats": [],\n            "stuck": {"is": is_stuck},\n            "outlook": {"label": "bad" if status in ("dead", "danger") else "ok", "p_through": 0.2},\n            "note": "mario_red=%d green=%d" % (mario, ngreen),\n        }\n    except Exception:\n        return None\n', 'memory.json': '{"human": {}, "past": {}}\n'}

