# ==================================================================================================
# Super Mario Bros. (NES) -- level 8-2  |  超级马里奥兄弟（NES） 关 8-2
# Pass rate on 100 random variants never used before: 27/100; fastest finish 1703 frames (28.38 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 27/100；最快一局 1703 帧（60 帧/秒，约 28.38 秒）。
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
#   hazard-triggered reactive runner (pixel hazards -> (wait,hold) jump macros keyed by kind/world
#   -x); enemy triggers deferred to the pit edge; rear-bill abort of waits; flag-pole routine.
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _solid                         line 137   
#   def _find_mario                    line 149   
#   def _sig                           line 169   
#   def _shift                         line 173   
#   def _pit_dist                      line 184   
#   def _pipe_edge                     line 197   
#   def _hazard                        line 205   
#   class Tracker                      line 239   
#   def _enemy_ahead                   line 310   
#   def _bill_ahead                    line 334   
#   def _bill_behind                   line 357   
#   def _bill_lane                     line 382   
#   def _bill_air                      line 419   
#   def _bill_arc                      line 441   
#   def _adv                           line 468   
#   def run_phase                      line 544   
#   def _air_bill                      line 593   
#   def _bill_near                     line 598   
#   def _green_arc                     line 604   
#   def _koopa_above                   line 619   
#   def _pipe_geom                     line 640   
#   def _plant_state                   line 656   
#   def _overhead                      line 671   
#   def do_jump                        line 686   returns 'landed' | 'dead' | 'skip' | 'done'
#   def _steer                         line 818   
#   def _tall_pipe                     line 876   
#   def _plant_n                       line 897   
#   def _arc_behind                    line 906   moving black object (bill) above Mario's head level, behind / beside him
#   def tall_pipe_jump                 line 925   
#   def pipe_hop                       line 1012  
#   def _sg_green                      line 1101  
#   def _sg_bill                       line 1110  (left, right) of a moving black bullet-bill-sized blob in the stair-top lane (y 158-178), else None
#   def _gl_bill                       line 1151  distance (px) from Mario's rear edge to a moving black blob (ground-lane bullet bill) on the left
#   def _gl_hop                        line 1172  hop in place over a ground-lane bill coming from behind
#   def stair_routine                  line 1185  at the foot of the last staircase: stand until the lane bill has passed and the paratroopa is in a
#   def stair_go                       line 1298  
#   def stair_routine                  line 1342  last staircase: no waiting (the paratroopa kills a standing Mario). bunny-hop up the steps with
#   def _norm                          line 1391  
#   def _lookup                        line 1398  
#   def solve                          line 1425  
#   def play                           line 1433  never exit on a 'dead' verdict (false positives): revive the tracker and keep playing
#   def _play_once                     line 1460  
#   def _finish                        line 1574  
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hazard-triggered reactive runner (pixel hazards -> (wait,hold) jump macros keyed by kind/world-x); enemy triggers deferred to the pit edge; rear-bill abort of waits; flag-pole routine.
import numpy as np

PIT27_H = 24
E21_H = 26
E26_VETO = True
S27_HI = 2722
E26_CAP = 22
W30_H = 36
HP_ON = True
HP_LO = 2735
HP_HI = 2800
HP_D = 10
DEBUG = False
DBG_LO = 1640
DBG_HI = 1740
BRAKE_ON = False
TUNABLE = {"E21_H": (12, 36), "CAPW_H2": (6, 30), "EN_TRIG": (30, 60), "PIT27_H": (14, 30),
           "E26_H": (20, 36), "E26_H2": (20, 36), "SW_H": (24, 36), "SW_W": (12, 48),
           "W26_W": (0, 24), "BILL_TRIG": (30, 60), "HOP_N": (10, 28), "PH_H": (10, 16),
           "G15_W": (0, 12), "CAPW_H": (6, 20)}
BRAKE_LO = 1690
BRAKE_HI = 1790
BRAKE_D = 60
G15_W = 6
SW_W = 36
SW_H = 25
W26_H = 36
W26_W = 12
E26_SPLIT = 2688
E26_H2 = 30
E26_W = 0
E26_H = 31
_HOOK = None
_START_WX = 0
BEHIND_GAP = 11
SAME_OK = True
TALL_BILL = True
TALL_BILL_D = 56
TALL_ARC = True
TALL_FWD = True
TALL_FWD_D = 44
TALL_HOP_N = 20
TABLE = [("green", 2161, 2189, 0, 0, False), ("enemy", 2660, S27_HI, 0, 0, False),
         ("green", 2660, S27_HI, 0, 0, False), ("wall", 2660, S27_HI, 0, 0, False),
         ("enemy", 2590, 2659, E26_W, E26_H, E26_VETO), ("wall", 2890, 2915, 0, 36, False),
         ("enemy", 2725, 2765, 6, 30, False),
         ("wall", 2590, 2612, W26_W, W26_H, False), ("green", 2600, 2640, 6, 10, False),
         ("green", 2715, 2768, 0, 0, False),
         ("pit", 2760, 2790, 0, PIT27_H, False), ("wall", 2735, 2800, 0, PIT27_H, False),
         ("green", 2680, 2710, 0, 30, False), ("wall", 2955, 2985, 6, 30, False),
         ("wall", 3200, 3220, 0, 34, False)]
LOOK = 36
PIT_D = 8
GREEN_D = 60
LATE = 2440
EN_ON = True
EN_RANGE = 90
EN_TRIG = 45
EN_MIN = 6
HB = 6
ARC_RANGE = 180
VETO_MAX = 150
VETO_KINDS = ("wall", "green", "enemy")
ARC_BACK = -6
VETO_CLEAR = 3
BEHIND_FROM = 1990
ARC_BACK2 = 110
ARC2_LO = 2000
ARC2_HI = 2110
PT_LO = 2262
PT_HI = 2312
PT_TAKE = 2294
PIPE_NOT_PIT = True
PIPE_LO = 2150
PIPE_HI = 2300
OV_ON = True
OV_MAXX = 2800
OV_DIV = 3.6
OV_LOW = 40
OV_H1 = 1
BILL_TRIG = 43
USE_BILL_AHEAD = False
BILL_RANGE = 110
LANE_ANY = True
ARC_ANY = False
VETO_OFF = (1380, 1450)
ARC_LO = 1380
ARC_HI = 1760
AIR_BILL = 45
AIR_PRE = 10
BILL_MIN = 8
BILL_W = 0
BILL_H = 36
STILL_FROM = 1600
PIT_DEFER = 60
PIT_DEFER_MIN = 14
GARC_FROM = 2600
GARC_X = 64
GARC_UP = 80
PL_HOP_D = 40
PL_HOP_N = 22
KA_X = 22
KA_UP = 70
PLANT_ON = True
PL_MAX = 200
PL_DOWN = 3
PL_NOSEE = 30
HOP_N = 15
HOP_D = 44
HOP_STILL = True
SLOW_N = 5
SLOW_PX = 6
HOP_COOL = 34
BEHIND_ON = True
BEHIND_MIN = 5
BEHIND_UP = 14
ROUTE = [[184, 'green', 6, 14, 0], [297, 'wall', 0, 16, 0], [331, 'pit', 1, 22, 0], [377, 'wall', -1, 0, 0], [377, 'wall', 6, 10, 0], [591, 'pit', 0, 30, 0], [789, 'pit', 0, 30, 0], [994, 'pit', 0, 30, 0], [1234, 'pit', 0, 30, 0], [1327, 'wall', 9, 22, 0], [1396, 'enemy', 0, 30, 0], [1482, 'wall', 0, 16, 0], [1503, 'green', 0, 14, 0]]

TAIL = [[1581, 'green', G15_W, 22, 0], [1674, 'wall', 3, 22, 0], [1818, 'wall', 0, 16, 0], [1888, 'enemy', 0, 24, 0], [1897, 'wall', 0, 16, 0], [1989, 'wall', 3, 22, 0], [2027, 'bill', 0, 36, 0], [2067, 'green', 18, 22, 0], [2135, 'bill', 0, 36, 0], [2175, 'enemy', 0, E21_H, 0], [2194, 'pit', 0, 14, 0], [2265, 'green', 6, 14, 12], [2289, 'pit', 2, 30, 0], [2362, 'pit', 0, 30, 0]]
TAIL2 = [[2489, 'wall', 0, 36, 0], [2501, 'enemy', 0, 24, 0], [2601, 'wall', 0, 36, 0], [2640, 'enemy', 0, 24, 0], [2682, 'enemy', 0, 24, 0], [2766, 'wall', 0, 16, 0], [2905, 'wall', 12, 36, 0], [2953, 'wall', 3, 16, 0]]
TAIL3 = [[3050, 'wall', 12, W30_H, 0], [3154, 'wall', 6, 16, 0], [3209, 'wall', 24, 36, 0], [3257, 'pit', 0, 30, 0], [3449, 'pole', 0, 40, 0]]
ROUTE = ROUTE + TAIL + TAIL2 + TAIL3

DEFAULT = {"pit": (0, 30, 0), "wall": (0, 16, 0), "green": (6, 14, 0), "pole": (0, 40, 0), "enemy": (0, 30, 0),
           "bill": (BILL_W, BILL_H, 0)}
CANDS = {
    "pit": [(w, H, 0) for H in (30, 22, 38, 14) for w in (0, 1, 2, 3)],
    "wall": [(w, H, 0) for H in (16, 10, 22, 30, 36) for w in (3, 0, 6, 9, 12, 15, 18)],
    "green": [(w, H, 0) for H in (14, 8, 22, 30) for w in (6, 0, 12, 18, 24)],
    "pole": [(0, 40, 0), (0, 30, 0), (6, 40, 0), (12, 36, 0), (0, 22, 0)],
    "enemy": [(0, 30, 0), (0, 36, 0), (0, 24, 0), (0, 16, 0), (0, 10, 0), (0, 4, 0), (4, 30, 0), (8, 30, 0)],
    "bill": [(BILL_W, BILL_H, 0)],
}


def _solid(img, late=False):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    orange = (r == 228) & (g == 92) & (b == 16)
    tan = (r == 240) & (g == 208) & (b == 176)
    black = (r == 0) & (g == 0) & (b == 0)
    if late:
        return orange | tan | black | ((r == 0) & (g == 168) & (b == 0))
    return orange | tan | black


def _find_mario(img, prev):
    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]
    if prev is not None:
        px, py = prev
        x0, x1 = max(0, px - 30), min(256, px + 30)
        y0, y1 = max(40, py - 44), min(240, py + 44)
    else:
        x0, x1, y0, y1 = 0, 256, 40, 240
    sub = (r[y0:y1, x0:x1] == 248) & (g[y0:y1, x0:x1] == 56) & (b[y0:y1, x0:x1] == 0)
    ys, xs = np.nonzero(sub)
    if len(ys) < 4:
        return None
    return (int((xs.min() + xs.max()) // 2 + x0), int(ys.min() + y0))


_SKYU = np.array([104, 136, 252], np.uint8)


def _sig(img):
    return (img[100:236] != _SKYU).any(axis=2).sum(axis=0).astype(np.int16)


def _shift(sig, psig, pd):
    best, bd = 0, 1e18
    for d in range(0, 9):
        e = np.abs(sig[:256 - d] - psig[d:]).mean()
        if d == pd:
            e -= 0.05
        if e < bd - 1e-9:
            best, bd = d, e
    return best


def _pit_dist(S, mx, fy):
    front = mx + 8
    col = S[fy:240, :]
    cols_empty = ~col.any(axis=0)
    seg = cols_empty[max(0, mx - 2):min(256, front + 100)]
    idx = np.nonzero(seg)[0]
    if len(idx):
        edge = max(0, mx - 2) + int(idx[0])
        if edge >= mx - 2:
            return edge - front
    return None


def _pipe_edge(img, e0, fy):
    xa = max(0, min(250, e0))
    xb = min(256, xa + 8)
    G = img[max(40, fy - 12):max(41, fy - 2), xa:xb]
    gm = (G[:, :, 0] == 0) & (G[:, :, 1] == 168) & (G[:, :, 2] == 0)
    return int(gm.sum()) >= 12


def _hazard(img, S, mx, fy, guard, late=False, wx=0):
    front = mx + 8
    xa = min(255, front + 1)
    xb = min(256, front + LOOK)
    pd = _pit_dist(S, mx, fy)
    if pd is not None and pd <= PIT_D:
        if (late or not PIPE_NOT_PIT or not (PIPE_LO < wx < PIPE_HI)
                or not _pipe_edge(img, front + pd + 1, fy)):
            return "pit"
    if not guard:
        band = S[max(0, fy - 14):fy - 1, xa:xb]
        cc = band.sum(axis=0) >= 8
        wide = 0
        run_ = 0
        for v in cc:
            run_ = run_ + 1 if v else 0
            if run_ > wide:
                wide = run_
        if wide >= 10:
            return "wall"
        if late:
            gb = img[max(0, fy - 14):fy - 1, xa:xb]
            gcols = ((gb[:, :, 0] == 0) & (gb[:, :, 1] == 168) & (gb[:, :, 2] == 0)).sum(axis=0) >= 4
            if gcols.sum() >= 6:
                return "wall"
    G = img[max(40, fy - 36):fy + 2, max(0, front - 2):min(256, front + GREEN_D)]
    gm = (G[:, :, 0] == 0) & (G[:, :, 1] == 168) & (G[:, :, 2] == 0)
    if gm.sum() >= 6:
        row = gm[min(gm.shape[0] - 1, max(0, G.shape[0] - 4)), :]
        if row.sum() < 6 or not late:
            return "green"
    return None


class Tracker:
    def __init__(self):
        self.scroll = 0
        self.sig = None
        self.pd = 0
        self.prev = None
        self.same = 0
        self.same_full = 0
        self.ys = ()
        self.last = None
        self.t = 0
        self.lost = 0
        self.dead = False
        self.done = False
        self.pole = None
        self.en = None
        self.enb = False
        self.arc = None
        self.bill = None
        self.billg = None
        self.gy = None
        self.pitd = None
        self.behind = None
        self.garc = False
        self.hopleft = 0
        self.nofreeze = False
        self.cool = 0
        self.wxs = ()
        self.hist = ()

    def still(self):
        return len(self.ys) >= 3 and self.ys[0] == self.ys[1] == self.ys[2]

    def still2(self):
        return len(self.ys) >= 2 and self.ys[-1] == self.ys[-2]

    def wx(self):
        return self.scroll + (self.prev[0] if self.prev else 0)

    def update(self, img):
        sg = _sig(img)
        if self.sig is not None:
            d = _shift(sg, self.sig, self.pd)
            self.scroll += d
            self.pd = d
        self.sig = sg
        if self.last is not None and np.array_equal(img[40:], self.last[40:]):
            self.same += 1
        else:
            self.same = 0
        if self.last is not None and np.array_equal(img, self.last):
            self.same_full += 1
        else:
            self.same_full = 0
        self.last = img
        self.hist = (self.hist + ((self.scroll, img),))[-HB:]
        m = _find_mario(img, self.prev)
        if m is None:
            self.lost += 1
            if self.lost > 3:
                self.prev = None
        else:
            self.lost = 0
            self.prev = m
            self.ys = (self.ys + (m[1],))[-3:]
        self.t += 1
        if self.lost > 8 or (m is not None and m[1] > 224):
            self.dead = True
        return m


def _enemy_ahead(tr, img, m):
    if len(tr.hist) < 3 or m is None:
        return None
    d = tr.scroll - tr.hist[0][0]
    pl = tr.hist[0][1]
    x0 = m[0] + 10
    x1 = min(256 - d, m[0] + 8 + EN_RANGE)
    if x1 <= x0 + 2:
        return None
    y0 = max(40, m[1] - (EN_UP2 if STAIR_LO < tr.wx() < STAIR_HI else 2))
    y1 = min(224, m[1] + 16)
    if y1 <= y0:
        return None
    cur = img[y0:y1, x0:x1]
    ch = (cur != pl[y0:y1, x0 + d:x1 + d]).any(axis=2)
    cols = ch.sum(axis=0)
    idx = np.nonzero(cols >= 2)[0]
    if len(idx) == 0 or cols.sum() < EN_MIN:
        return None
    blk = ch & (cur.sum(axis=2) == 0)
    tr.enb = blk.sum() * 10 >= ch.sum() * 5
    return x0 + int(idx[0]) - (m[0] + 8)


def _bill_ahead(tr, img, m):
    if len(tr.hist) < 3 or m is None:
        return None
    d = tr.scroll - tr.hist[0][0]
    pl = tr.hist[0][1]
    x0 = m[0] + 10
    x1 = min(256 - d, m[0] + 8 + BILL_RANGE)
    if x1 <= x0 + 2:
        return None
    y0 = max(40, m[1] - 2)
    y1 = min(224, m[1] + 16)
    if y1 <= y0:
        return None
    cur = img[y0:y1, x0:x1]
    prv = pl[y0:y1, x0 + d:x1 + d]
    nb = (cur.sum(axis=2) == 0) & (prv.sum(axis=2) != 0)
    cols = nb.sum(axis=0)
    idx = np.nonzero(cols >= 2)[0]
    if len(idx) == 0 or cols.sum() < BILL_MIN:
        return None
    return x0 + int(idx[0]) - (m[0] + 8)


def _bill_behind(tr, img, m):
    if len(tr.hist) < 3 or m is None:
        return None
    d = tr.scroll - tr.hist[0][0]
    pl = tr.hist[0][1]
    x0 = max(0, m[0] - 80)
    x1 = m[0] - BEHIND_GAP
    if x1 <= x0 + 4:
        return None
    y0 = max(40, m[1] - BEHIND_UP)
    y1 = min(224, m[1] + 16)
    if y1 <= y0 or x1 + d > 256:
        return None
    cur = img[y0:y1, x0:x1]
    prv = pl[y0:y1, x0 + d:x1 + d]
    if d == 0:
        nb = (cur != prv).any(axis=2)
    else:
        nb = (cur.sum(axis=2) == 0) & (prv.sum(axis=2) != 0)
    if int(nb.sum()) < BEHIND_MIN:
        return None
    cols = np.nonzero(nb.sum(axis=0) >= 1)[0]
    return int(x1 - (x0 + int(cols[-1])))


def _bill_lane(tr, img, m, gy):
    if len(tr.hist) < 3 or m is None:
        return None
    d = tr.scroll - tr.hist[0][0]
    pl = tr.hist[0][1]
    x0 = m[0] + 10
    x1 = min(256 - d, m[0] + 8 + BILL_RANGE + 20)
    if x1 <= x0 + 2:
        return None
    y0 = max(40, gy - 2)
    y1 = min(224, gy + 16)
    cur = img[y0:y1, x0:x1]
    prv = pl[y0:y1, x0 + d:x1 + d]
    if LANE_ANY:
        nb = (cur != prv).any(axis=2)
    else:
        nb = (cur.sum(axis=2) == 0) & (prv.sum(axis=2) != 0)
    cols = nb.sum(axis=0)
    idx = np.nonzero(cols >= 2)[0]
    if len(idx) == 0 or cols.sum() < BILL_MIN:
        return None
    return x0 + int(idx[0]) - (m[0] + 8)


CJ_LO = 3225
CJ_HI = 3300
CAPW_H = 9
CAPW_SPLIT = 2140
CAPW_H2 = 9
CAPW_LO = 2040
CAPW_HI = 2235
_KIND = ["x"]
DODGE_ON = True
DODGE_D = 46
DODGE_MIN = 6


def _bill_air(tr, img, m):
    if len(tr.hist) < 3 or m is None:
        return None
    d = tr.scroll - tr.hist[0][0]
    pl = tr.hist[0][1]
    x0 = m[0] + 8
    x1 = min(256 - d, m[0] + 8 + DODGE_D)
    if x1 <= x0 + 2:
        return None
    y0 = max(40, m[1] - 16)
    y1 = min(224, m[1] + 8)
    if y1 <= y0:
        return None
    cur = img[y0:y1, x0:x1]
    prv = pl[y0:y1, x0 + d:x1 + d]
    nb = (cur.sum(axis=2) == 0) & (prv.sum(axis=2) != 0)
    if int(nb.sum()) < DODGE_MIN:
        return None
    cols = np.nonzero(nb.sum(axis=0) >= 1)[0]
    return int(x0 + cols[0] - (m[0] + 8))


def _bill_arc(tr, img, m):
    if len(tr.hist) < 3 or m is None:
        return None
    d = tr.scroll - tr.hist[0][0]
    pl = tr.hist[0][1]
    back = ARC_BACK2 if ARC2_LO < tr.wx() < ARC2_HI else ARC_BACK
    x0 = max(0, m[0] - back)
    x1 = min(256 - d, m[0] + 8 + ARC_RANGE)
    if x1 <= x0 + 2:
        return None
    y0 = max(40, m[1] - 62)
    y1 = m[1] - 3
    if y1 <= y0 + 2:
        return None
    cur = img[y0:y1, x0:x1]
    ch = (cur != pl[y0:y1, x0 + d:x1 + d]).any(axis=2)
    if not (ARC_ANY and ARC_LO < tr.wx() < ARC_HI):
        ch = ch & (cur.sum(axis=2) == 0)
    if ch.sum() < 10:
        return None
    cols = ch.sum(axis=0)
    idx = np.nonzero(cols >= 2)[0]
    if len(idx) == 0:
        return None
    return x0 + int(idx[0]) - (m[0] + 8)


def _adv(sim, tr, z, right=True, left=False):
    keys = {"Right", "x"} if right else set()
    if left:
        keys = {"Left", "x"}
    if tr.hopleft > 0:
        tr.hopleft -= 1
        z = 1
        if HOP_STILL and tr.wx() > 2400:
            keys = set()
    if tr.cool > 0:
        tr.cool -= 1
    if z:
        keys.add("z")
    sim.step(keys)
    img = sim.screen()
    m = tr.update(img)
    if _HOOK is not None:
        _HOOK(sim, tr)
    under = False
    fy = 0
    S = None
    tr.pitd = None
    if m is not None:
        fy = min(m[1] + 15, 238)
        S = _solid(img, tr.wx() > LATE)
        if tr.wx() > LATE:
            xs0, xs1 = max(0, m[0] - 5), m[0] + 6
            lime = (img[fy:fy + 2, xs0:xs1, 0] == 184) & (img[fy:fy + 2, xs0:xs1, 1] == 248)
            blk = (S[fy, xs0:xs1] | lime[0]) & (S[fy + 1, xs0:xs1] | lime[1])
            under = blk.sum() >= 7
        else:
            under = S[fy:fy + 2, max(0, m[0] - 5):m[0] + 6].sum() >= 6
    tr.enb = False
    tr.en = _enemy_ahead(tr, img, m) if (EN_ON and under) else None
    tr.arc = _bill_arc(tr, img, m) if (EN_ON and under) else None
    tr.bill = _bill_ahead(tr, img, m) if (EN_ON and under and USE_BILL_AHEAD) else None
    tr.behind = _bill_behind(tr, img, m) if (BEHIND_ON and under and tr.wx() > BEHIND_FROM) else None
    tr.garc = bool(under and tr.wx() > GARC_FROM and _green_arc(img, m))
    tr.wxs = (tr.wxs + (tr.wx(),))[-8:]
    slow = len(tr.wxs) >= 6 and tr.wxs[-1] - tr.wxs[-1 - SLOW_N] <= SLOW_PX
    if (tr.behind is not None and (slow or 2890 < tr.wx() < 2960) and tr.cool == 0 and tr.hopleft == 0
            and tr.behind <= HOP_D):
        tr.hopleft = HOP_N
        tr.cool = HOP_COOL
    tr.billg = None
    if m is not None:
        if under and tr.still2():
            tr.gy = m[1]
        elif EN_ON and tr.gy is not None and tr.wx() > STILL_FROM and (not under or m[1] < tr.gy - 4):
            tr.billg = _bill_lane(tr, img, m, tr.gy)
    if under and tr.en is not None and S is not None:
        tr.pitd = _pit_dist(S, m[0], fy)
    if DEBUG and DBG_LO < tr.wx() < DBG_HI:
        print("D", getattr(sim, "frame", None), tr.wx(), m, "under", under, "bg", tr.billg, "en", tr.en,
              "enb", tr.enb, sorted(keys))
    tr.pole = None
    if m is not None and tr.wx() > 3300:
        L = (img[40:225, :, 0] == 184) & (img[40:225, :, 1] == 248) & (img[40:225, :, 2] == 24)
        cs = L.sum(axis=0)
        px = int(np.argmax(cs))
        if cs[px] >= 70:
            tr.pole = px
            if px - m[0] <= 12 and px - m[0] >= -20:
                tr.done = True
    if tr.wx() > LATE:
        under = under and tr.still()
    if tr.same >= 3 and not under and not tr.nofreeze:
        tr.dead = True
    elif tr.same_full >= 30:
        tr.dead = True
    if tr.dead and tr.pole is not None and m is not None and m[1] < 185 and -24 <= tr.pole - m[0] <= 16:
        tr.dead = False
        tr.done = True
    return img, m, S, fy, under


def run_phase(sim, tr, suppress=0, maxf=700):
    best_wx = tr.wx()
    best_n = 0
    gf = 0
    for n in range(maxf):
        img, m, S, fy, under = _adv(sim, tr, 0)
        if tr.done:
            return ("done",)
        if tr.dead or m is None:
            return ("dead",)
        wx = tr.wx()
        if wx > best_wx + 1:
            best_wx, best_n = wx, n
        elif n - best_n > 30:
            return ("stall",)
        if tr.pole is not None and under and 0 < tr.pole - m[0] <= 44 and tr.same < 3:
            return ("trigger", "pole")
        if _air_bill(tr):
            return ("trigger", "bill")
        if (SG_ON and not _SG["done"] and SG_LO <= wx <= SG_HI and under and m[1] >= 185
                and tr.same < 3):
            return ("trigger", "stair")
        if HP_ON and HP_LO < wx < HP_HI and m[1] >= 188 and S is not None:
            pdh = _pit_dist(S, m[0], fy)
            if pdh is not None and pdh <= HP_D:
                return ("trigger", "pit")
        if suppress > 0:
            suppress -= 1
            continue
        if tr.same >= 3 and not (SAME_OK and under and wx > 2400 and tr.same < 40):
            continue
        gf = gf + 1 if (under and (wx < STILL_FROM or tr.still2())) else 0
        if gf >= 1:
            guard = (tr.t < 90 and m[0] < 100)
            kind = _hazard(img, S, m[0], fy, guard, wx > LATE, wx) if gf >= 2 else None
            if kind != "pit" and tr.bill is not None and tr.bill <= BILL_TRIG and tr.t > 60:
                return ("trigger", "bill")
            if kind != "pit" and tr.en is not None and tr.t > 60:
                if tr.enb and tr.en <= BILL_TRIG:
                    return ("trigger", "bill")
                if not tr.enb and tr.en <= EN_TRIG:
                    defer = (tr.pitd is not None and tr.pitd <= PIT_DEFER and tr.en > PIT_DEFER_MIN)
                    if not defer:
                        return ("trigger", "enemy")
            if kind:
                return ("trigger", kind)
    return ("stall",)


def _air_bill(tr):
    return (tr.billg is not None and tr.billg <= AIR_BILL and tr.gy is not None and len(tr.ys) >= 2
            and tr.ys[-1] > tr.ys[-2] and tr.prev is not None and tr.prev[1] >= tr.gy - AIR_PRE)


def _bill_near(tr):
    if tr.bill is not None and tr.bill <= BILL_TRIG:
        return True
    return tr.en is not None and tr.enb and tr.en <= BILL_TRIG


def _green_arc(img, m):
    if m is None:
        return False
    x0 = max(0, m[0] - 6)
    x1 = min(256, m[0] + GARC_X)
    y0 = max(40, m[1] - GARC_UP)
    y1 = min(238, m[1] + 8)
    if x1 <= x0 or y1 <= y0:
        return False
    G = img[y0:y1, x0:x1]
    gm = (G[:, :, 0] == 0) & (G[:, :, 1] == 168) & (G[:, :, 2] == 0)
    n = int(gm.sum())
    return 8 <= n <= 110


def _koopa_above(img, m):
    if m is None:
        return False
    x0 = max(0, m[0] - KA_X)
    x1 = min(256, m[0] + KA_X)
    y0 = max(40, m[1] - KA_UP)
    y1 = max(y0 + 1, m[1] - 2)
    G = img[y0:y1, x0:x1]
    gm = (G[:, :, 0] == 0) & (G[:, :, 1] == 168) & (G[:, :, 2] == 0)
    pg = _pipe_geom(img, m)
    if pg is not None:
        c0, c1, top = pg
        ya = max(0, top - 32 - y0)
        yb = gm.shape[0]
        xa_ = max(0, c0 - 2 - x0)
        xb_ = max(0, c1 + 3 - x0)
        gm[ya:yb, xa_:xb_] = False
    n = int(gm.sum())
    return 6 <= n <= 90


def _pipe_geom(img, m):
    fy = min(m[1] + 15, 238)
    xa = min(250, m[0] + 6)
    xb = min(256, m[0] + 8 + 56)
    G = img[40:fy, xa:xb]
    gm = (G[:, :, 0] == 0) & (G[:, :, 1] == 168) & (G[:, :, 2] == 0)
    rows = np.nonzero(gm.sum(axis=1) >= 22)[0]
    if len(rows) == 0:
        return None
    top = 40 + int(rows[0])
    cols = np.nonzero(gm[rows[0]:rows[0] + 3].sum(axis=0) >= 2)[0]
    if len(cols) == 0:
        return None
    return (xa + int(cols[0]), xa + int(cols[-1]), top)


def _plant_state(img, m):
    if m is None:
        return None
    pg = _pipe_geom(img, m)
    if pg is None:
        return None
    c0, c1, top = pg
    y0 = max(40, top - 28)
    if top - y0 < 6:
        return None
    box = img[y0:top - 1, c0:c1 + 1]
    ns = (box != _SKYU).any(axis=2).sum()
    return bool(ns >= 12)


def _overhead(img, m):
    x0 = max(0, m[0] - 2)
    x1 = min(256, m[0] + 26)
    y0 = max(40, m[1] - 60)
    y1 = m[1] - 1
    if y1 <= y0:
        return None
    G = img[y0:y1, x0:x1]
    br = (G[:, :, 0] == 228) & (G[:, :, 1] == 92) & (G[:, :, 2] == 16)
    rows = np.nonzero(br.sum(axis=1) >= 8)[0]
    if len(rows) == 0:
        return None
    return int(m[1] - (y0 + int(rows[-1])))


def do_jump(sim, tr, w, H, s=0, nobill=False, veto=False, plant=False, capov=False, until=0, steer=False):
    """returns 'landed' | 'dead' | 'skip' | 'done'"""
    for _ in range(s):
        _adv(sim, tr, 0, right=False)
        if tr.done:
            return "done"
        if tr.dead:
            return "dead"
        if not nobill and _bill_near(tr):
            return do_jump(sim, tr, BILL_W, BILL_H, 0, True, False, False, True)
    if until:
        k = 0
        while tr.wx() < until and k < 40:
            _adv(sim, tr, 0)
            k += 1
            if tr.done:
                return "done"
            if tr.dead:
                return "dead"
        w = 0
    if H == 0:
        for _ in range(8):
            _adv(sim, tr, 0)
            if tr.done:
                return "done"
            if tr.dead:
                return "dead"
        return "skip"
    for _ in range(w):
        _adv(sim, tr, 0)
        if tr.done:
            return "done"
        if tr.dead:
            return "dead"
        if not nobill and _bill_near(tr):
            return do_jump(sim, tr, BILL_W, BILL_H, 0, True, False, False, True)
    if veto and not nobill and not (VETO_OFF[0] < tr.wx() < VETO_OFF[1]):
        n = 0
        clear = 0
        seen = False
        while n < VETO_MAX:
            if tr.arc is not None or tr.garc:
                clear = 0
                seen = True
            else:
                clear += 1
                if clear >= VETO_CLEAR or not seen:
                    break
            n += 1
            _adv(sim, tr, 0, right=False)
            if tr.done:
                return "done"
            if tr.dead:
                return "dead"
            if _bill_near(tr):
                return do_jump(sim, tr, BILL_W, BILL_H, 0, True, False, False, True)
    if plant and not nobill:
        seen_up = False
        down_run = 0
        for n in range(PL_MAX):
            st = _plant_state(tr.last, tr.prev)
            if st is None:
                break
            if st:
                seen_up = True
                down_run = 0
            else:
                down_run += 1
            if seen_up and down_run >= PL_DOWN:
                break
            if (not seen_up) and down_run >= PL_NOSEE:
                break
            if tr.en is not None and not tr.enb and tr.en <= PL_HOP_D and tr.cool == 0:
                for i in range(PL_HOP_N + 22):
                    _adv(sim, tr, 1 if i < PL_HOP_N else 0, right=False)
                    if tr.done:
                        return "done"
                    if tr.dead:
                        return "dead"
                tr.cool = 10
                continue
            if _koopa_above(tr.last, tr.prev):
                _adv(sim, tr, 0, right=False, left=True)
            else:
                _adv(sim, tr, 0)
            if tr.done:
                return "done"
            if tr.dead:
                return "dead"
            if _bill_near(tr):
                return do_jump(sim, tr, BILL_W, BILL_H, 0, True, False, False, True)
    if capov and OV_ON and tr.wx() < OV_MAXX and tr.prev is not None and tr.last is not None:
        c = _overhead(tr.last, tr.prev)
        if c is not None and c < 60:
            H = OV_H1 if c < OV_LOW else max(5, min(H, int(c / OV_DIV)))
            print("OV", tr.wx(), c, H)
    if CAPW_H > 0 and nobill and CAPW_LO < tr.wx() < CAPW_HI and _KIND[0] != "pit" and H > CAPW_H:
        print("CAPW", tr.wx(), H, "->", CAPW_H)
        H = CAPW_H if tr.wx() < CAPW_SPLIT else CAPW_H2
    if tr.wx() > 3000:
        print("JMP", tr.wx(), H, (tr.wxs[-1] - tr.wxs[-2]) if len(tr.wxs) >= 2 else None, tr.prev)
    air = 0
    for i in range(130):
        if i >= H and not nobill and _air_bill(tr):
            return "airbill"
        rt, lf = True, False
        if (DODGE_ON and 3 <= i < H and not nobill and 1990 < tr.wx() < 2260 and len(tr.ys) >= 2
                and tr.ys[-1] <= tr.ys[-2] and _bill_air(tr, tr.last, tr.prev) is not None):
            print("DODGE", tr.wx(), i)
            H = i
        if steer and i >= 8 and not (i < H and len(tr.ys) >= 2 and tr.ys[-1] <= tr.ys[-2]):
            rt, lf = _steer(tr)
        if (BRAKE_ON and nobill and i >= H and BRAKE_LO < tr.wx() < BRAKE_HI and len(tr.ys) >= 2
                and tr.ys[-1] > tr.ys[-2] and tr.billg is not None and tr.billg <= BRAKE_D):
            rt, lf = False, True
        img, m, S, fy, under = _adv(sim, tr, 1 if i < H else 0, right=rt, left=lf)
        if tr.done:
            return "done"
        if tr.dead or m is None:
            return "dead"
        air += 1
        if under and air >= 6 and tr.same < 3 and (tr.wx() < STILL_FROM or tr.still2()):
            if tr.wx() > 3000:
                print("LAND", tr.wx(), air, tr.prev)
            return "landed"
    return "dead"


ST_ON = False
ST_MIN_RUN = 8


def _steer(tr):
    m = tr.prev
    img = tr.last
    if m is None or img is None or len(tr.ys) < 2 or len(tr.wxs) < 2 or not ST_ON:
        return True, False
    r = 209
    feet = m[1] + 16
    if tr.ys[-1] <= tr.ys[-2] or feet > r + 2 or feet < r - 90:
        return True, False
    S = _solid(img, False)
    row = S[min(239, r + 1)]
    vy = tr.ys[-1] - tr.ys[-2]
    tf = (r - feet) / float(vy + 1)
    vx = max(0, min(4, tr.wxs[-1] - tr.wxs[-2]))
    xc = m[0]
    land = xc + vx * tf
    runs = []
    s0 = None
    for x in range(256):
        if row[x]:
            if s0 is None:
                s0 = x
        elif s0 is not None:
            if x - s0 >= ST_MIN_RUN:
                runs.append((s0, x - 1))
            s0 = None
    if s0 is not None and 256 - s0 >= ST_MIN_RUN:
        runs.append((s0, 255))
    runs = [(a, b) for (a, b) in runs if b > xc - 4]
    if not runs:
        return True, False
    for a, b in runs:
        if a + 2 <= land <= b - 2:
            tgt = (a + b) / 2.0
            if land < tgt - 6:
                return True, False
            if land > tgt + 6:
                return False, True
            return False, False
    a, b = min(runs, key=lambda ab: abs((ab[0] + ab[1]) / 2.0 - land))
    tgt = (a + b) / 2.0
    if tgt > land:
        return True, False
    return False, True


G22_LO = 2225
G22_HI = 2290
G22_H = 8
PIT22_LO = 2185
PIT22_HI = 2232
PIT22_H = 7
TALL_MIN = 50
TALL_FROM = 2440
TALL_FRAC = 0.4
TALL_WAIT = 320


def _tall_pipe(img, m):
    if m is None:
        return None
    g = (img[:, :, 0] == 0) & (img[:, :, 1] == 168) & (img[:, :, 2] == 0)
    x0 = max(0, m[0] - 2)
    cs = g[40:230, x0:].sum(axis=0)
    idx = np.nonzero(cs >= TALL_MIN)[0]
    if len(idx) == 0:
        return None
    a = int(idx[0]) + x0
    rs = g[40:230, max(0, a - 10):min(256, a + 20)].sum(axis=1) >= 9
    top = None
    for r in range(0, len(rs) - 9):
        if rs[r:r + 9].all():
            top = r
            break
    if top is None:
        return None
    return (max(0, a - 4), min(255, a + 14), 40 + top)


def _plant_n(img, pg):
    a, b, top = pg
    y0 = max(40, top - 28)
    if top - 1 <= y0:
        return 0
    box = img[y0:top - 1, a:b + 1]
    return int((box != _SKYU).any(axis=2).sum())


def _arc_behind(tr, back=70, fwd=20):
    """moving black object (bill) above Mario's head level, behind / beside him"""
    m = tr.prev
    img = tr.last
    if len(tr.hist) < 3 or m is None:
        return False
    d = tr.scroll - tr.hist[0][0]
    pl = tr.hist[0][1]
    x0 = max(0, m[0] - back)
    x1 = min(256 - d, m[0] + fwd)
    y0 = max(40, m[1] - 62)
    y1 = m[1] - 3
    if x1 <= x0 + 2 or y1 <= y0 + 2:
        return False
    cur = img[y0:y1, x0:x1]
    ch = (cur != pl[y0:y1, x0 + d:x1 + d]).any(axis=2) & (cur.sum(axis=2) == 0)
    return int(ch.sum()) >= 10


def tall_pipe_jump(sim, tr):
    hist = []
    tr.nofreeze = True
    tr.cool = 600
    tr.hopleft = 0
    for n in range(TALL_WAIT):
        pg = _tall_pipe(tr.last, tr.prev)
        if pg is None:
            break
        bb = _bill_behind(tr, tr.last, tr.prev) if TALL_BILL else None
        fb = TALL_FWD and tr.en is not None and tr.enb and tr.en <= TALL_FWD_D
        if fb or (bb is not None and bb <= TALL_BILL_D and bb >= 2):
            print("TALL bill-hop", tr.wx(), bb, n)
            for i in range(TALL_HOP_N + 26):
                _adv(sim, tr, 1 if i < TALL_HOP_N else 0, right=False)
                if tr.done:
                    return "done"
                if tr.dead:
                    return "dead"
            continue
        c = _plant_n(tr.last, pg)
        hist.append(c)
        ready = tr.hopleft == 0 and tr.still2() and n >= 2
        if ready and c < 14:
            break
        if ready and len(hist) >= 10 and max(hist[-10:]) < 40 and c <= hist[-8] + 2:
            break
        if ready and len(hist) >= 6:
            mx = max(hist[-100:])
            if mx >= 150 and c <= TALL_FRAC * mx and c < hist[-4]:
                break
        _adv(sim, tr, 0, right=False)
        if tr.done:
            return "done"
        if tr.dead:
            return "dead"
    if TALL_ARC:
        k = 0
        clear = 0
        while k < 30 and clear < 3:
            clear = 0 if _arc_behind(tr) else clear + 1
            if clear >= 3:
                break
            k += 1
            _adv(sim, tr, 0, right=False)
            if tr.done:
                return "done"
            if tr.dead:
                return "dead"
        print("TARC", k, getattr(sim, "frame", None))
    tr.nofreeze = False
    tr.hopleft = 0
    tr.cool = 80
    print("TPL end_frame", getattr(sim, "frame", None), len(hist), "".join("U" if c >= 14 else "." for c in hist))
    for i in range(36):
        _adv(sim, tr, 1)
        if tr.done:
            return "done"
        if tr.dead:
            return "dead"
    air = 0
    for i in range(130):
        img, m, S, fy, under = _adv(sim, tr, 0)
        if tr.done:
            return "done"
        if tr.dead or m is None:
            return "dead"
        air += 1
        if under and air >= 6 and tr.same < 3 and tr.still2():
            return "landed"
    return "dead"


PH_D = 14
PH_H = 13
PH_BRAKE = 22
PH_EDGE = 2
PH_SPEED = 2.6
PH_WAIT = 10
PH_WMAX = 130
PH_NOSEE = 60
PH_ND = 1
EN_UP2 = 26
STAIR_LO = 2880
STAIR_HI = 3040


def pipe_hop(sim, tr):
    if tr.prev is None or tr.last is None:
        return None
    pg = _pipe_geom(tr.last, tr.prev)
    if pg is None:
        return None
    c0, c1, top = pg
    p0 = tr.scroll + c0
    p1 = tr.scroll + c1
    print("PIPE", tr.wx(), p0, p1, top)
    for k in range(60):
        dd = p0 - (tr.wx() + 8)
        spd = (tr.wxs[-1] - tr.wxs[-2]) if len(tr.wxs) >= 2 else 3
        if dd <= PH_D and (spd >= PH_SPEED or dd <= 6):
            break
        _adv(sim, tr, 0)
        if tr.done:
            return "done"
        if tr.dead or tr.prev is None:
            return "dead"
    seen_up = False
    nd = 0
    tr.nofreeze = True
    clear = 0
    for _w in range(PH_WMAX):
        pg2 = _pipe_geom(tr.last, tr.prev) if tr.prev is not None else None
        if pg2 is None:
            st = None
        else:
            a_, b_, t_ = pg2
            bx = tr.last[max(40, t_ - 28):max(41, t_ - 2), a_:b_ + 1]
            st = int(((bx[:, :, 0] == 0) & (bx[:, :, 1] == 168) & (bx[:, :, 2] == 0)).sum()) >= 8
        if st:
            seen_up = True
            nd = 0
        else:
            nd += 1
        clear = clear + 1 if tr.arc is None else 0
        if _w >= PH_WAIT and clear >= 3 and (st is None or (seen_up and nd >= PH_ND) or _w >= PH_NOSEE):
            break
        _adv(sim, tr, 0, right=False)
        if tr.done:
            tr.nofreeze = False
            return "done"
        if tr.dead or tr.prev is None:
            tr.nofreeze = False
            return "dead"
    tr.nofreeze = False
    print("PHWAIT", _w, seen_up, tr.wx())
    air = 0
    onpipe = False
    for i in range(70):
        xc = tr.wx()
        brake = xc >= p0 + PH_BRAKE
        _adv(sim, tr, 1 if i < PH_H else 0, right=not brake, left=brake and i >= 6)
        if tr.done:
            return "done"
        if tr.dead or tr.prev is None:
            return "dead"
        air += 1
        if air >= 6 and tr.still2():
            if abs(tr.prev[1] + 15 - top) <= 4:
                onpipe = True
                break
            if tr.prev[1] >= 185:
                break
    if not onpipe:
        return "landed"
    print("ONPIPE", tr.wx(), tr.prev)
    r = do_jump(sim, tr, 0, 32, 0, False, False, False, False, p1 - PH_EDGE, True)
    return r


SG_ON = True
SG_LO = 3136
SG_HI = 3190
SG_STAND = 14       # frames to let Mario come to rest before judging
SG_MAX = 110        # give up waiting after this many standing frames
SG_DX_LO = 42
SG_DX_HI = 62
SG_KY_LO = 80
SG_KY_HI = 150
SG_TB = 90
SG_D2 = 16
SG_H = 16
SG_H2 = 34
_SG = {"done": False}


def _sg_green(img):
    g = (img[40:205, :, 0] == 0) & (img[40:205, :, 1] == 168) & (img[40:205, :, 2] == 0)
    n = int(g.sum())
    if n < 20:
        return None
    ys, xs = np.nonzero(g)
    return (float(xs.mean()), float(ys.mean()) + 40.0)


def _sg_bill(img, prev):
    """(left, right) of a moving black bullet-bill-sized blob in the stair-top lane (y 158-178), else None"""
    a = img[154:182]
    p = prev[154:182]
    d = ((a.sum(axis=2) == 0) & (p.sum(axis=2) != 0))
    cols = np.nonzero(d.sum(axis=0) >= 1)[0]
    if len(cols) < 3:
        return None
    if cols.max() - cols.min() > 24:
        return None
    return (int(cols.min()), int(cols.max()))


NEW_GATE = True
SG_MAX2 = 260
BT_NEAR = -85
BT_DX = 6
BT_HOLD = 6
E_AGE_LO = 5
E_AGE_HI = 14
E_DX = 48
L_AGE_LO = 40
L_AGE_HI = 66
L_DX_LO = 22
L_DX_HI = 42
L_DY = -48
F_AGE = 70
F_AGE2 = 84
NOK_WAIT = 70
PRERUN_ON = True
D2_KOOPA = False
KH_ON = False
KH_DX_LO = -12
KH_DX_HI = 30
KH_DY_LO = -40
KH_DY_HI = 22
GB_D = 44
GB_H = 16
GB_ON = True


def _gl_bill(img, prev, m):
    """distance (px) from Mario's rear edge to a moving black blob (ground-lane bullet bill) on the left"""
    if m is None or prev is None:
        return None
    x0 = max(0, m[0] - 100)
    x1 = max(0, m[0] - 2)
    y0 = max(40, m[1] + 1)
    y1 = min(224, m[1] + 16)
    if x1 <= x0 + 3 or y1 <= y0:
        return None
    a = img[y0:y1, x0:x1]
    p = prev[y0:y1, x0:x1]
    d = (a.sum(axis=2) == 0) & (p.sum(axis=2) != 0)
    cols = np.nonzero(d.sum(axis=0) >= 1)[0]
    if len(cols) < 3 or int(d.sum()) < 6:
        return None
    if cols.max() - cols.min() > 26:
        return None
    return int(x1 - (x0 + int(cols.max())))


def _gl_hop(sim, tr):
    """hop in place over a ground-lane bill coming from behind"""
    for i in range(48):
        _adv(sim, tr, 1 if i < GB_H else 0, right=False)
        if tr.done:
            return "done"
        if tr.dead or tr.prev is None:
            return "dead"
        if i >= 14 and tr.still2() and tr.prev[1] >= 185:
            break
    return None


def stair_routine(sim, tr):
    """at the foot of the last staircase: stand until the lane bill has passed and the paratroopa is in a
    window where the hop lands past it, then hop up and keep going."""
    _SG["done"] = True
    tr.nofreeze = True
    tr.hopleft = 0
    tr.cool = 60
    prev = tr.last
    kyp = []
    why = "timeout"
    passed = False
    d2 = 0
    nstand = 0
    res = None
    kcool = 0
    k_seen = False
    k_age0 = 0
    khist = []
    bthr = 0
    for n in range(SG_STAND + (SG_MAX2 if NEW_GATE else SG_MAX)):
        _adv(sim, tr, 0, right=False)
        if tr.done:
            res = "done"
            break
        if tr.dead or tr.prev is None:
            res = "dead"
            break
        img = tr.last
        nstand += 1
        k = _sg_green(img)
        kyp.append(k)
        b = _sg_bill(img, prev) if tr.pd == 0 else None
        gb = _gl_bill(img, prev, tr.prev) if (GB_ON and tr.pd == 0 and tr.prev[1] >= 185) else None
        prev = img
        if gb is not None and gb <= GB_D:
            print("GB hop", gb, tr.wx(), getattr(sim, "frame", None))
            r_ = _gl_hop(sim, tr)
            if r_:
                res = r_
                break
            prev = tr.last
            continue
        if KH_ON and k is not None and tr.pd == 0 and tr.prev[1] >= 185 and kcool <= 0:
            kdx_ = k[0] - tr.prev[0]
            kdy_ = k[1] - tr.prev[1]
            if KH_DX_LO <= kdx_ <= KH_DX_HI and KH_DY_LO <= kdy_ <= KH_DY_HI:
                print("KH hop", round(kdx_), round(kdy_), tr.wx(), getattr(sim, "frame", None))
                r_ = _gl_hop(sim, tr)
                if r_:
                    res = r_
                    break
                prev = tr.last
                kcool = 24
                continue
        kcool -= 1
        mx = tr.prev[0] if tr.prev is not None else 0
        if NEW_GATE:
            kdx = kdy = None
            fr = getattr(sim, "frame", n)
            if k is not None:
                if not k_seen:
                    k_seen = True
                    k_age0 = fr
                kdx = k[0] - mx
                kdy = k[1] - tr.prev[1]
                khist.append((kdx, kdy))
            if b is not None and BT_NEAR <= (b[0] - mx) < BT_DX:
                bthr = fr + BT_HOLD
            if n < SG_STAND or tr.pd != 0:
                continue
            age = (fr - k_age0) if k_seen else -1
            early = (kdx is not None and E_AGE_LO <= age <= E_AGE_HI and kdx >= E_DX)
            late = (kdx is not None and L_AGE_LO <= age <= L_AGE_HI and L_DX_LO <= kdx <= L_DX_HI
                    and kdy <= L_DY and len(khist) >= 4 and kdy < khist[-4][1])
            force = k_seen and (age >= F_AGE and (fr >= bthr or age >= F_AGE2))
            nok = (not k_seen) and nstand >= NOK_WAIT and fr >= bthr
            if ((early or late) and fr >= bthr) or force or nok:
                why = "gate n%d age%d early%s late%s force%s k=%s,%s" % (n, age, early, late, force, kdx, kdy)
                d2 = SG_D2
                break
            continue
        if n < SG_STAND or tr.pd != 0:
            continue
        tb = 9999.0
        if b is not None:
            if b[0] >= mx - 8:
                passed = True
            else:
                tb = (mx - b[0]) / 1.8
        bill_ok = tb >= SG_TB
        if k is None:
            continue
        dx = k[0] - mx
        ky = k[1]
        if bill_ok and SG_DX_LO <= dx <= SG_DX_HI and SG_KY_LO <= ky <= SG_KY_HI:
            why = "win1 dx%d ky%d tb%d" % (dx, ky, tb)
            d2 = SG_D2 if tb >= 90 else 0
            break
        if passed and tb >= SG_TB and 34 <= dx < SG_DX_LO and 135 <= ky <= 152:
            why = "win2 dx%d ky%d" % (dx, ky)
            d2 = 0
            break
        if n >= SG_STAND + 50 and tb >= 12 and 28 <= dx <= 54 and ky >= 100:
            why = "win3 dx%d ky%d tb%d" % (dx, ky, tb)
            d2 = 0
            break
    tr.nofreeze = False
    if res:
        return res
    print("SG go", why, "wx", tr.wx(), "frame", getattr(sim, "frame", None), "n", nstand, "d2", d2)
    return stair_go(sim, tr, d2)


def stair_go(sim, tr, d2):
    r = do_jump(sim, tr, 0, SG_H, 0, True)
    if r != "landed":
        return r
    if tr.prev is not None and tr.prev[1] < 178 and 3190 < tr.wx() < 3230:
        for _ in range(d2):
            _adv(sim, tr, 0, right=False)
            if tr.done:
                return "done"
            if tr.dead:
                return "dead"
            kk = _sg_green(tr.last) if (D2_KOOPA and tr.prev is not None) else None
            if kk is not None:
                kdx = kk[0] - tr.prev[0]
                kdy = kk[1] - tr.prev[1]
                if -24 <= kdx <= 56 and kdy <= 30:
                    print("D2 break", _, kdx, kdy)
                    break
        if tr.prev is not None and tr.prev[1] < 150 and PRERUN_ON:
            for _k in range(16):
                spd = (tr.wxs[-1] - tr.wxs[-2]) if len(tr.wxs) >= 2 else 0
                pdx = None
                if tr.prev is not None and tr.last is not None:
                    fy_ = min(tr.prev[1] + 15, 238)
                    pdx = _pit_dist(_solid(tr.last, False), tr.prev[0], fy_)
                if pdx is not None and pdx <= 4 and spd >= 1:
                    break
                _adv(sim, tr, 0)
                if tr.done:
                    return "done"
                if tr.dead or tr.prev is None:
                    return "dead"
            print("PRERUN", _k, tr.wx(), tr.prev)
        r = do_jump(sim, tr, 0, SG_H2, 0, True)
        if r == "landed" and tr.prev is not None and tr.prev[1] < 150 and 3225 < tr.wx() < 3300:
            r = do_jump(sim, tr, 0, 30, 0, True)
    return r


CL_HC = 12
CL_H1 = 34
CL_MAXF = 140


def stair_routine(sim, tr):
    """last staircase: no waiting (the paratroopa kills a standing Mario). bunny-hop up the steps with
    Right+x held (z held CL_HC frames per hop), then at the top step run to the pit edge and jump onto
    the column (H CL_H1) and on to the tower (H 30). Oracle: 5/5 local variants."""
    _SG["done"] = True
    tr.nofreeze = False
    tr.hopleft = 0
    tr.cool = 10 ** 9
    air = 0
    n = 0
    while n < CL_MAXF:
        n += 1
        grounded = tr.still2() and tr.prev is not None
        if grounded and air == 0:
            air = 1
        z = 1 if (0 < air <= CL_HC) else 0
        if air > 0:
            air += 1
        _adv(sim, tr, z)
        if tr.done:
            return "done"
        if tr.dead:
            return "dead"
        if air > CL_HC + 2 and tr.still2():
            air = 0
        if tr.prev is not None and tr.prev[1] < 150 and tr.still2():
            break
    for k in range(16):
        if tr.prev is None:
            break
        spd = (tr.wxs[-1] - tr.wxs[-2]) if len(tr.wxs) >= 2 else 0
        fy_ = min(tr.prev[1] + 15, 238)
        pdx = _pit_dist(_solid(tr.last, False), tr.prev[0], fy_)
        if pdx is not None and pdx <= 4 and spd >= 1:
            break
        _adv(sim, tr, 0)
        if tr.done:
            return "done"
        if tr.dead:
            return "dead"
    print("STAIR top", tr.wx(), tr.prev, "n", n, "frame", getattr(sim, "frame", None))
    r = do_jump(sim, tr, 0, CL_H1, 0, True)
    if r == "landed" and tr.prev is not None and tr.prev[1] < 150 and 3225 < tr.wx() < 3300:
        r = do_jump(sim, tr, 0, 30, 0, True)
    if r == "landed":
        return None
    return r


def _norm(ev):
    ev = list(ev)
    while len(ev) < 5:
        ev.append(0)
    return ev


def _lookup(route, kind, wx, used):
    if kind == "bill":
        return DEFAULT["bill"]
    best, bd = None, 1e9
    for i, ev in enumerate(route):
        if i in used or ev[1] != kind:
            continue
        d = abs(ev[0] - wx)
        if d <= 60 and d < bd:
            best, bd = i, d
    if best is not None:
        used.add(best)
        ev = _norm(route[best])
        return (ev[2], ev[3], ev[4])
    bd = 1e9
    for i, ev in enumerate(route):
        if ev[1] != kind:
            continue
        d = abs(ev[0] - wx)
        if d <= 40 and d < bd:
            best, bd = i, d
    if best is not None:
        ev = _norm(route[best])
        return (ev[2], ev[3], ev[4])
    return DEFAULT[kind]


def solve(sim):
    return []


RESULT = {}
_ATT = {}


def play(sim):
    """never exit on a 'dead' verdict (false positives): revive the tracker and keep playing"""
    tr = Tracker()
    tr.update(sim.screen())
    for k in range(60):
        try:
            _play_once(sim, tr)
        except Exception as e:
            print("play stop:", type(e).__name__, getattr(sim, "frame", None))
            return
        if RESULT.get("what") == "done":
            return
        if getattr(sim, "frame", 0) >= 7950:
            return
        print("REVIVE", k, "wx", tr.wx(), "frame", getattr(sim, "frame", None))
        tr.dead = False
        tr.lost = 0
        tr.same = 0
        tr.same_full = 0
        tr.prev = None
        tr.hopleft = 0
        tr.nofreeze = False
        for _i in range(3):
            sim.step({"Right", "x"})
            tr.update(sim.screen())


def _play_once(sim, tr):
    route = ROUTE
    suppress = 0
    used = set()
    log = []
    attempts = _ATT
    for _ in range(400):
        res = run_phase(sim, tr, suppress=suppress)
        suppress = 0
        if res[0] in ("done", "dead"):
            return _finish(sim, tr, res[0], log)
        if res[0] == "stall":
            log.append("S%d" % tr.wx())
            r = do_jump(sim, tr, 0, 20)
            if r in ("dead", "done"):
                return _finish(sim, tr, r, log)
            continue
        kind = res[1]
        if kind == "stair":
            log.append("G%d@%s" % (tr.wx(), getattr(sim, "frame", "")))
            r = stair_routine(sim, tr)
            if r in ("dead", "done"):
                return _finish(sim, tr, r, log)
            suppress = 2
            continue
        if (kind in ("green", "wall", "enemy") and tr.prev is not None and tr.prev[1] < 180
                and PT_LO < tr.wx() < PT_HI):
            kind = "pit"
        if (kind == "green" and G22_LO < tr.wx() < G22_HI and tr.prev is not None
                and tr.prev[1] >= 180):
            log.append("H%d@%s" % (tr.wx(), getattr(sim, "frame", "")))
            r = pipe_hop(sim, tr)
            if r in ("dead", "done"):
                return _finish(sim, tr, r, log)
            if r is not None:
                suppress = 6
                continue
        tbl = None
        for (tk, lo, hi, tw, tH, tv) in TABLE:
            if tk == kind and lo <= tr.wx() <= hi:
                tbl = (tw, tH, tv)
                break
        if tbl is not None and tbl[1] == 0:
            suppress = 2
            continue
        if (tbl is None and kind in ("wall", "green") and tr.wx() > TALL_FROM and tr.prev is not None
                and _tall_pipe(tr.last, tr.prev) is not None):
            tk = ("T", tr.wx() // 60)
            tn = attempts.get(tk, 0)
            attempts[tk] = tn + 1
            log.append("T%d@%s#%d" % (tr.wx(), getattr(sim, "frame", ""), tn))
            if tn >= 2:
                r = None
                for i in range(22 + 14):
                    _adv(sim, tr, 0, right=(i >= 22), left=(i < 22))
                    if tr.done:
                        r = "done"
                        break
                    if tr.dead or tr.prev is None:
                        r = "dead"
                        break
                if r is None:
                    r = do_jump(sim, tr, 0, 36 if tn % 2 == 0 else 30)
            else:
                r = tall_pipe_jump(sim, tr)
            if r in ("dead", "done"):
                return _finish(sim, tr, r, log)
            suppress = 6
            continue
        w, H, s = _lookup(route, kind, tr.wx(), used)
        key = (kind, tr.wx() // 40)
        n = attempts.get(key, 0)
        attempts[key] = n + 1
        if n >= 2:
            pool = [c for c in CANDS[kind] if c[1] > 0]
            w, H, s = pool[(n * 7) % len(pool)]
        if kind == "pit" and PIT22_LO < tr.wx() < PIT22_HI and n < 2:
            w, H, s = 0, PIT22_H, 0
        if kind == "green" and G22_LO < tr.wx() < G22_HI and n < 2:
            H = G22_H
        until = 0
        if (kind == "pit" and PT_LO < tr.wx() < PT_HI and tr.prev is not None and tr.prev[1] < 180):
            until = PT_TAKE
            H = 32
        pl = (kind == "wall" and tr.wx() > LATE and PLANT_ON)
        if pl:
            w = min(w, 6)
        veto = kind in VETO_KINDS and s >= 0
        if tbl is not None:
            w, H, s, veto = tbl[0], tbl[1], 0, tbl[2]
            pl = False
            if kind == "enemy" and E26_SPLIT < tr.wx() < 2715 and H > E26_H2:
                H = E26_H2
            if kind == "enemy" and 2620 < tr.wx() < 2660 and H > E26_CAP:
                H = E26_CAP
        _KIND[0] = kind
        log.append("%s%d:%d/%d/%d@%s" % (kind[0], tr.wx(), w, H, s, getattr(sim, "frame", "")))
        r = do_jump(sim, tr, w, H, max(s, 0), False, veto, pl,
                    kind in ("enemy", "bill"), until, kind == "pit" and 2280 < tr.wx() < 2480)
        while r == "airbill":
            log.append("A%d@%s" % (tr.wx(), getattr(sim, "frame", "")))
            r = do_jump(sim, tr, BILL_W, BILL_H, 0, True, False, False, True)
        if r in ("dead", "done"):
            return _finish(sim, tr, r, log)
        if (r == "landed" and CJ_LO < tr.wx() < CJ_HI and tr.prev is not None and tr.prev[1] < 150
                and tr.wx() > 3000):
            log.append("CJ%d@%s" % (tr.wx(), getattr(sim, "frame", "")))
            r = do_jump(sim, tr, 0, 30, 0, True)
            if r in ("dead", "done"):
                return _finish(sim, tr, r, log)
        if r == "skip":
            suppress = 10


def _finish(sim, tr, what, log):
    RESULT["wx"] = tr.wx()
    RESULT["what"] = what
    RESULT["frame"] = getattr(sim, "frame", None)
    RESULT["log"] = " ".join(log)
    print("play end:", what, "wx", tr.wx(), "frame", getattr(sim, "frame", None), " ".join(log[-8:]))
    if what == "done":
        for _i in range(400):
            keys = {"Right", "x"}
            if (_i % 36) < 20:
                keys.add("z")
            sim.step(keys)
    return what

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    stuck = False\n    try:\n        st = (ctx or {}).get("stuck") or {}\n        stuck = bool(st.get("is")) if isinstance(st, dict) else bool(st)\n    except Exception:\n        pass\n    return {"status": "stuck" if stuck else "running", "where": {"hazard": None, "confidence": 0.0},\n            "progress": None, "resources": {}, "threats": [], "stuck": {"is": stuck},\n            "outlook": {"label": "ok", "p_through": 0.3}, "note": ""}\n', 'memory.json': '{"human":{},"past":{}}\n'}

