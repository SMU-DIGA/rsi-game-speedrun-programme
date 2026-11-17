# ==================================================================================================
# Super Mario Bros. (NES) -- level 5-1  |  超级马里奥兄弟（NES） 关 5-1
# Pass rate on 100 random variants never used before: 99/100; fastest finish 1432 frames (23.87 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 99/100；最快一局 1432 帧（60 帧/秒，约 23.87 秒）。
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
#   reactive pixel runner (Right+run, jump at pits/obstacles/enemies) with per-jump hold/delay lea
#   rned by save/load DFS in solve() and kept in sim.memory keyed by world-x; scenery (white trees
#   ) filtered
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _sky                           line 39    
#   def _tree_cols                     line 44    boolean per x: column belongs to a white 'mushroom' tree (background scenery), pipes excluded
#   class Ctrl                         line 64    
#   def binof                          line 316   
#   def skey                           line 320   
#   class Stats                        line 324   success/failure counts of (position bin, event index, candidate) gathered by the search.
#   def _bin_of_key                    line 361   
#   def build_stats                    line 368   
#   def play                           line 388   
#   class Abort                        line 419   ------------------------------------------------------------------ search (solve)
#   class Searcher                     line 423   
#   def solve                          line 514   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
#
# This policy also reads sim.memory; that data is in 5-1.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 5-1.memory.json。
# ==================================================================================================

# idea: reactive pixel runner (Right+run, jump at pits/obstacles/enemies) with per-jump hold/delay learned by save/load DFS in solve() and kept in sim.memory keyed by world-x; scenery (white trees) filtered
import copy
import numpy as np

FREEZE = 6          # identical frames => dead
FREEZE_WAIT = 90    # ... unless we are deliberately standing still (bill / flyer gate)
LOOK = 34           # px ahead of Mario front to trigger an obstacle jump
LOOK_TALL = 46
PIT_LOOK = 10
HOLD_PIT = 22
H_LOW = 8
H_MID = 10
TOL = 24            # world-x tolerance for memory lookup
DEBUG = False
BILL_WAIT = True
BILL_RANGE = 110
BILL_HITS = 1       # consecutive frames of a moving black blob at head height before standing still
ROBUST = True
MAX_ROUNDS = 14
ROUND_FRAMES = 6000
FINISH_X = 3150
import json as _json
STATS0_JSON = '{"1.0|28,0":[3,0],"11.0|10,0":[0,1],"11.0|12,0":[0,1],"11.0|6,0":[0,1],"11.0|8,0":[0,1],"11.0|8,4":[0,1],"13.0|10,0":[0,1],"13.0|10,5":[0,1],"13.0|12,0":[0,1],"13.0|16,0":[0,1],"13.0|4,0":[0,1],"13.0|6,0":[0,1],"13.0|8,0":[3,1],"13.0|8,4":[0,1],"13.0|8,8":[0,1],"14.0|10,0":[0,3],"14.0|10,5":[0,3],"14.0|12,0":[0,3],"14.0|16,0":[0,3],"14.0|4,0":[0,3],"14.0|6,0":[0,3],"14.0|8,0":[0,3],"14.0|8,4":[0,3],"14.0|8,8":[0,3],"17.0|10,0":[0,2],"17.0|10,5":[0,2],"17.0|12,0":[0,2],"17.0|16,0":[0,2],"17.0|4,0":[0,2],"17.0|6,0":[0,2],"17.0|8,0":[3,2],"17.0|8,4":[0,2],"17.0|8,8":[0,2],"19.0|10,0":[0,2],"19.0|10,5":[0,2],"19.0|12,0":[0,2],"19.0|16,0":[0,2],"19.0|4,0":[0,2],"19.0|6,0":[0,2],"19.0|8,0":[0,2],"19.0|8,4":[0,2],"19.0|8,8":[0,2],"20.0|10,0":[0,1],"20.0|6,0":[0,1],"20.0|8,0":[0,1],"20.0|8,4":[3,0],"21.0|16,0":[0,19],"21.0|20,0":[0,20],"21.0|24,0":[3,19],"21.0|28,0":[0,19],"21.0|28,4":[0,20],"21.1|20,0":[3,0],"21.2|28,0":[3,0],"21.3|28,0":[3,0],"21.4|28,0":[3,0],"21.5|20,0":[0,1],"21.5|28,0":[0,1],"21.5|28,4":[3,0],"22.0|10,0":[0,32],"22.0|10,5":[0,32],"22.0|12,0":[0,32],"22.0|14,0":[0,5],"22.0|16,0":[0,37],"22.0|22,0":[3,5],"22.0|22,4":[0,5],"22.0|28,0":[0,5],"22.0|28,6":[0,5],"22.0|4,0":[0,32],"22.0|6,0":[0,32],"22.0|8,0":[0,32],"22.0|8,4":[0,32],"22.0|8,8":[0,32],"22.1|10,0":[0,6],"22.1|10,5":[0,6],"22.1|12,0":[0,6],"22.1|14,0":[0,6],"22.1|16,0":[0,12],"22.1|22,0":[0,6],"22.1|22,4":[0,6],"22.1|28,0":[0,6],"22.1|28,6":[0,6],"22.1|4,0":[0,6],"22.1|6,0":[0,6],"22.1|8,0":[0,6],"22.1|8,4":[0,6],"22.1|8,8":[0,6],"29.0|10,0":[0,1],"29.0|6,0":[0,1],"29.0|8,0":[0,1],"29.0|8,4":[3,0],"29.1|10,0":[0,3],"29.1|10,5":[0,3],"29.1|12,0":[0,3],"29.1|16,0":[0,3],"29.1|4,0":[0,3],"29.1|6,0":[0,3],"29.1|8,0":[0,3],"29.1|8,4":[0,3],"29.1|8,8":[0,3],"3.0|8,0":[0,1],"34.0|8,0":[0,1],"36.0|8,0":[3,0],"41.0|8,0":[3,0],"44.0|28,0":[3,0],"46.0|22,0":[0,1],"46.0|28,0":[3,0],"49.0|8,0":[3,0],"5.0|8,0":[3,0],"53.0|10,0":[0,1],"53.0|10,5":[0,1],"53.0|12,0":[0,1],"53.0|4,0":[3,0],"53.0|6,0":[0,1],"53.0|8,0":[0,1],"53.0|8,4":[0,1],"54.0|22,0":[3,0],"55.0|10,0":[0,1],"55.0|10,5":[0,1],"55.0|12,0":[0,1],"55.0|16,0":[0,1],"55.0|4,0":[0,1],"55.0|6,0":[0,1],"55.0|8,0":[3,1],"55.0|8,4":[0,1],"55.0|8,8":[0,1],"55.1|8,0":[3,0],"55.2|8,0":[3,0],"55.3|8,0":[3,0],"55.4|8,0":[3,0],"55.5|22,0":[0,1],"55.5|28,0":[3,0],"58.0|22,0":[3,0],"65.0|8,0":[3,0],"68.0|10,0":[3,0],"68.0|8,0":[0,1],"71.0|10,0":[0,1],"71.0|10,5":[0,1],"71.0|12,0":[0,1],"71.0|16,0":[0,1],"71.0|4,0":[0,1],"71.0|6,0":[0,1],"71.0|8,0":[0,1],"71.0|8,4":[0,1],"71.0|8,8":[0,1],"73.0|28,0":[3,0],"73.1|16,0":[0,1],"73.1|22,0":[0,1],"73.1|22,4":[0,1],"73.1|28,0":[0,1],"73.1|28,6":[3,0],"77.0|8,0":[1,0],"78.0|8,0":[2,0],"78.1|8,0":[2,0],"79.0|8,0":[3,0],"79.1|22,0":[1,0],"8.0|8,0":[3,0],"82.0|22,0":[1,0],"83.0|8,0":[3,0],"84.0|8,0":[1,0],"84.1|8,0":[1,0],"84.2|8,0":[1,0],"85.0|22,0":[1,0],"88.0|8,0":[2,0],"88.0|8,4":[0,1],"89.0|10,0":[0,1],"89.0|6,0":[0,1],"89.0|8,0":[0,1],"89.0|8,4":[1,0],"91.0|8,0":[3,0],"91.1|8,0":[2,0],"92.0|8,0":[3,0],"92.1|8,0":[3,0],"93.0|8,0":[3,0],"95.0|16,0":[0,1],"95.0|22,0":[0,1],"95.0|22,4":[1,0],"95.0|28,0":[0,1],"98.0|8,0":[2,0]}'
STATS0 = _json.loads(STATS0_JSON)
FLY_X0 = 2700       # world-x from which winged koopas are watched
FLY_AHEAD = 70      # px ahead of Mario's centre that count as his jump corridor
FLY_UP = 80         # px above his feet
FLY_MAXWAIT = 90    # frames of waiting before giving up
PURGE_BIN = 36    # forget stats from this 32px-bin on (learned with trees seen as walls)

CANDS = {
    "low": [(8, 0), (10, 0), (6, 0), (8, 4), (12, 0), (10, 5), (4, 0), (16, 0), (8, 8)],
    "mid": [(10, 0), (8, 0), (12, 0), (16, 0), (10, 4), (6, 0), (20, 0)],
    "tall": [(28, 0), (20, 0), (28, 4), (24, 0), (16, 0)],
    "pit": [(22, 0), (28, 0), (16, 0), (22, 4), (28, 6), (14, 0)],
}


def _sky(img):
    return (abs(img[:, :, 0].astype(np.int16) - 104) < 6) & (abs(img[:, :, 1].astype(np.int16) - 136) < 6) & \
           (abs(img[:, :, 2].astype(np.int16) - 252) < 6)


def _tree_cols(img):
    """boolean per x: column belongs to a white 'mushroom' tree (background scenery), pipes excluded"""
    sub = img[110:208]
    r, g, b = sub[:, :, 0], sub[:, :, 1], sub[:, :, 2]
    cap = ((r == 252) & (g == 252) & (b == 252)) | ((r == 188) & (g == 188) & (b == 188))
    cc = cap.any(axis=0)
    if not cc.any():
        return None
    P = (r == 124) & (g == 124) & (b == 124)
    isp = P.sum(axis=0) >= 8
    if isp.any():
        k = np.ones(9, dtype=np.int32)
        pd = np.convolve(isp.astype(np.int32), k, mode="same") > 0
        cc = cc & ~pd
    if not cc.any():
        return None
    k = np.ones(3, dtype=np.int32)
    return np.convolve(cc.astype(np.int32), k, mode="same") > 0


class Ctrl:
    def __init__(self):
        self.prev = None
        self.feet_hist = []
        self.held = 0
        self.hold = 0
        self.plan = None
        self.need = None
        self.scroll = 0
        self.band = None
        self.same = 0
        self.lastimg = None
        self.lost = 0
        self.stall = 0
        self.cx = 0
        self.feet = 0
        self.on_ground = True
        self.n = 0
        self.prev_black = None
        self.ds = 0
        self.bill_hits = 0
        self.bill_t = 0
        self.wait_run = 0
        self.cool = 0
        self.wait = False
        self.gate = 0
        self.gated = 0

    def clone(self):
        c = copy.copy(self)
        c.feet_hist = list(self.feet_hist)
        c.plan = list(self.plan) if self.plan else None
        return c

    # ---------------------------------------------------------------- perception
    def observe(self, img):
        self.n += 1
        self.need = None
        if self.lastimg is not None and np.array_equal(img[40:], self.lastimg[40:]):
            self.same += 1
        else:
            self.same = 0
        self.lastimg = img
        # scroll estimate from ground band
        band = img[210:238].astype(np.int16)
        if self.band is not None:
            best, bd = None, 0
            W = band.shape[1]
            for d in range(0, 6):
                diff = np.abs(self.band[:, d:] - band[:, :W - d]).mean()
                if best is None or diff < best - 1e-9:
                    best, bd = diff, d
            self.scroll += bd
            self.ds = bd
            if bd == 0:
                self.stall += 1
            else:
                self.stall = 0
        self.band = band
        lim = FREEZE if not (self.wait or self.gate > 0 or self.bill_t > 0) else FREEZE_WAIT
        if self.same > lim:
            if self.scroll + self.cx >= FINISH_X - 40:
                return "finished"
            return "dead"
        # Mario
        red = (img[:, :, 0] == 248) & (img[:, :, 1] == 56) & (img[:, :, 2] == 0)
        red[:30] = False
        ys, xs = np.nonzero(red)
        if len(ys) and self.prev is not None:
            keep = (abs(xs - self.prev[0]) < 40) & (abs(ys - self.prev[1]) < 60)
            if keep.any():
                ys, xs = ys[keep], xs[keep]
        if len(ys) == 0:
            self.lost += 1
            if self.lost > 10:
                return "dead"
            return None
        self.lost = 0
        cx = (xs.min() + xs.max()) / 2.0
        feet = ys.min() + 15
        self.prev = (cx, feet)
        self.cx, self.feet = cx, feet
        self.feet_hist.append(feet)
        if len(self.feet_hist) > 6:
            self.feet_hist.pop(0)
        lastf = self.feet_hist[-2] if len(self.feet_hist) > 1 else feet
        sky = _sky(img)
        icx, ifeet = int(cx), int(feet)
        # standing: something solid right under the feet and not moving vertically
        under = ~sky[min(239, ifeet):min(240, ifeet + 3), max(0, icx - 3):icx + 4]
        self.on_ground = bool(abs(feet - lastf) < 0.5 and under.sum() >= 5)
        self._bill(img, icx, ifeet)
        if self.gate > 0:
            self.gate -= 1
        if self.held == 0 and self.plan is None and self.on_ground and not self.wait:
            self.need = self._analyse(img, sky, icx, ifeet)
            if self.need is not None and self.need["kind"] != "pit" and self.scroll + cx >= FLY_X0:
                if self._flyer_near(img, icx, ifeet):
                    self.need = None
                    self.gate = 3
                    self.gated += 1
                    if self.gated > FLY_MAXWAIT:
                        self.gate = 0
                else:
                    self.gated = 0
            else:
                self.gated = 0
        if self.scroll + cx >= FINISH_X:
            return "finished"
        return None

    def _flyer_near(self, img, cx, feet):
        """a winged koopa (green blob well above the ground, not a pipe plant) in Mario's jump corridor"""
        x0, x1 = max(0, cx - 12), min(255, cx + FLY_AHEAD)
        y0, y1 = max(40, feet - FLY_UP), max(40, feet - 4)
        if y1 <= y0:
            return False
        reg = img[y0:y1, x0:x1 + 1]
        gm = (reg[:, :, 0] == 0) & (reg[:, :, 1] == 168) & (reg[:, :, 2] == 0)
        if gm.sum() < 6:
            return False
        ys, xs = np.nonzero(gm)
        # ignore green that sits on something solid (ground koopas / plants): bottom of the blob must have sky under it
        by = int(ys.max()) + y0
        if by >= feet - 10:
            # blob reaches Mario's own height (ground koopa / low hop): the normal obstacle logic handles it
            return False
        return True

    def _bill(self, img, icx, ifeet):
        """detect a moving black blob (bullet bill) flying at head height of a standing Mario; then wait."""
        cur = (img[:, :, 0] < 12) & (img[:, :, 1] < 12) & (img[:, :, 2] < 12)
        bill = False
        if BILL_WAIT and self.prev_black is not None and self.on_ground and self.cool == 0:
            ds = self.ds
            W = cur.shape[1]
            new = cur[:, :W - ds] & ~self.prev_black[:, ds:]
            y0, y1 = max(0, ifeet - 34), max(0, ifeet - 16)
            c0, c1 = max(0, icx - 4), min(W - ds, icx + 8 + BILL_RANGE)
            if y1 > y0 and c1 > c0:
                reg = new[y0:y1, c0:c1]
                if reg.sum() >= 10:
                    _, xs2 = np.nonzero(reg)
                    if xs2.mean() + c0 > icx - 12:
                        bill = True
        self.prev_black = cur
        self.bill_hits = self.bill_hits + 1 if bill else 0
        if self.bill_hits >= BILL_HITS:
            self.bill_t = 14
        elif self.bill_t > 0:
            self.bill_t -= 1
        if self.cool > 0:
            self.cool -= 1
        if self.bill_t > 0:
            self.wait_run += 1
            if self.wait_run > 150:
                self.bill_t = 0
                self.cool = 80
                self.wait_run = 0
        else:
            self.wait_run = 0
        self.wait = self.bill_t > 0

    def _analyse(self, img, sky, cx, feet):
        H, W = sky.shape
        front = min(W - 1, cx + 8)
        y0, y1 = max(0, feet - 14), min(H, feet - 1)
        x1 = min(W, front + 64)
        sub = img[y0:y1, front + 1:x1].astype(np.int16)
        white = (sub[:, :, 0] > 240) & (sub[:, :, 1] > 240) & (sub[:, :, 2] > 240)
        tan = (abs(sub[:, :, 0] - 240) < 4) & (abs(sub[:, :, 1] - 208) < 4) & (abs(sub[:, :, 2] - 176) < 4)
        tc = _tree_cols(img)
        if tc is not None:
            tcs = tc[front + 1:x1][None, :]
            gb = ((abs(sub[:, :, 0] - 188) < 4) & (abs(sub[:, :, 1] - 188) < 4) & (abs(sub[:, :, 2] - 188) < 4)) | \
                 ((sub[:, :, 0] < 12) & (sub[:, :, 1] < 12) & (sub[:, :, 2] < 12))
            treepix = gb & tcs
        else:
            treepix = np.zeros_like(white)
        obs = ~sky[y0:y1, front + 1:x1] & ~white & ~tan & ~treepix
        cols = obs.sum(axis=0)
        jump = False
        if (cols[:LOOK] >= 2).any():
            jump = True
        tall_hit = (cols[:LOOK_TALL] >= 9).any()
        if tall_hit:
            jump = True
        # pit
        gy = min(H - 1, feet + 3)
        grow = ~sky[gy, front - 6:min(W, front + 80)]
        ahead = grow[6:6 + PIT_LOOK + 12]
        pit = False
        if len(ahead) >= PIT_LOOK + 4 and (~ahead[PIT_LOOK - 2:PIT_LOOK + 4]).all():
            jump = True
            pit = True
        if not jump:
            return None
        # obstacle height
        hmax = 0
        nzc = np.nonzero(cols)[0]
        if len(nzc):
            ya, yb = max(0, feet - 90), min(H, feet - 1)
            tsub = img[ya:yb, front + 1:x1].astype(np.int16)
            tw = (tsub[:, :, 0] > 240) & (tsub[:, :, 1] > 240) & (tsub[:, :, 2] > 240)
            tt = (abs(tsub[:, :, 0] - 240) < 4) & (abs(tsub[:, :, 1] - 208) < 4) & (abs(tsub[:, :, 2] - 176) < 4)
            if tc is not None:
                tgb = ((abs(tsub[:, :, 0] - 188) < 4) & (abs(tsub[:, :, 1] - 188) < 4) & (abs(tsub[:, :, 2] - 188) < 4)) | \
                      ((tsub[:, :, 0] < 12) & (tsub[:, :, 1] < 12) & (tsub[:, :, 2] < 12))
                tpix = tgb & tc[front + 1:x1][None, :]
            else:
                tpix = np.zeros_like(tw)
            tobs = ~sky[ya:yb, front + 1:x1] & ~tw & ~tt & ~tpix
            c0 = int(nzc[0])
            for c in range(c0, min(c0 + 14, tobs.shape[1])):
                colv = tobs[::-1, c]
                z = np.nonzero(~colv)[0]
                hmax = max(hmax, int(z[0]) if len(z) else len(colv))
            hmax += 1
        if pit and hmax <= 1:
            kind = "pit"
        elif hmax <= 17:
            kind = "low"
        elif hmax <= 25:
            kind = "mid"
        elif hmax <= 49:
            kind = "tall"
        else:
            kind = "tall"
        return {"kind": kind, "wx": int(self.scroll + cx), "h": hmax}

    # ---------------------------------------------------------------- action
    def act(self):
        keys = {"Right", "x"}
        if (self.wait or self.gate > 0) and self.held == 0 and self.plan is None:
            keys.discard("Right")
        if self.plan is not None:
            hold, delay = self.plan
            if delay > 0:
                self.plan[1] = delay - 1
            else:
                self.held = 1
                self.hold = hold
                self.plan = None
        if self.held > 0:
            if self.held < self.hold:
                keys.add("z")
            self.held += 1
            if self.held > 6 and self.on_ground:
                self.held = 0
        return keys


def binof(wx):
    return int(wx) >> 5


def skey(b, u, cand):
    return "%d.%d|%d,%d" % (b, min(u, 6), cand[0], cand[1])


class Stats:
    """success/failure counts of (position bin, event index, candidate) gathered by the search."""

    def __init__(self, d=None):
        self.d = d if d is not None else {}

    def get(self, b, u, cand):
        ok = fail = 0
        for nb in (b - 1, b, b + 1):
            v = self.d.get(skey(nb, u, cand))
            if v:
                ok += v[0]
                fail += v[1]
        return ok, fail

    def score(self, b, u, cand):
        ok, fail = self.get(b, u, cand)
        return (ok + 1.0) / (ok + fail + 2.0)

    def note(self, b, u, cand, good):
        v = self.d.setdefault(skey(b, u, cand), [0, 0])
        v[0 if good else 1] += 1

    def ranked(self, need, u):
        b = binof(need["wx"])
        base = CANDS[need["kind"]]
        sc = [(-self.score(b, u, c), i, c) for i, c in enumerate(base)]
        sc.sort()
        return [c for _, _, c in sc]

    def merge_flat(self, ok_d, fail_d):
        for k, v in (ok_d or {}).items():
            self.d.setdefault(k, [0, 0])[0] += int(v)
        for k, v in (fail_d or {}).items():
            self.d.setdefault(k, [0, 0])[1] += int(v)


def _bin_of_key(k):
    try:
        return int(k.split(".")[0])
    except Exception:
        return 0


def build_stats(mem, purge=False):
    st = Stats()
    try:
        for k, v in STATS0.items():
            st.d[k] = [v[0], v[1]]
    except Exception:
        pass
    try:
        if "sok2" in mem:
            st.merge_flat(mem.get("sok2", {}), mem.get("sfail2", {}))
        elif not STATS0:
            # first use: keep the old (c0-era) knowledge before the first tree cluster only
            ok_d = {k: v for k, v in mem.get("sok", {}).items() if _bin_of_key(k) < PURGE_BIN}
            fl_d = {k: v for k, v in mem.get("sfail", {}).items() if _bin_of_key(k) < PURGE_BIN}
            st.merge_flat(ok_d, fl_d)
    except Exception:
        pass
    return st


def play(sim):
    ctrl = Ctrl()
    try:
        stats = build_stats(sim.memory)
    except Exception:
        stats = build_stats({})
    binc = {}
    dead_n = 0
    while True:
        img = sim.screen()
        st = ctrl.observe(img)
        if st == "dead":
            # a frozen screen may also be a false alarm: keep pressing for a while before giving up
            dead_n += 1
            if dead_n > 240:
                return
            sim.step({"Right", "x"})
            continue
        dead_n = 0
        if ctrl.need is not None and ctrl.plan is None:
            b = binof(ctrl.need["wx"])
            u = binc.get(b, 0)
            cand = stats.ranked(ctrl.need, u)[0]
            binc[b] = u + 1
            ctrl.plan = list(cand)
        if DEBUG and ctrl.n % 10 == 0:
            print(ctrl.n, ctrl.scroll, ctrl.cx, ctrl.feet)
        sim.step(ctrl.act())


# ------------------------------------------------------------------ search (solve)
class Abort(Exception):
    pass


class Searcher:
    def __init__(self, sim, stats):
        self.sim = sim
        self.floor = 0
        self.path = []          # [(wx, cand, bin, u)]
        self.best = []
        self.trials = 0
        self.stats = stats
        self.stop_at = 150
        self.binc = {}

    def run_until(self, ctrl):
        sim = self.sim
        while True:
            if sim.budget_left < self.stop_at:
                return "finished"
            img = sim.screen()
            st = ctrl.observe(img)
            if st:
                return st
            if ctrl.need is not None and ctrl.plan is None:
                return "decision"
            sim.step(ctrl.act())

    def dfs(self, ctrl, depth, binc):
        sim = self.sim
        st = self.run_until(ctrl)
        if st == "dead":
            return False
        if st == "finished":
            return True
        need = ctrl.need
        b = binof(need["wx"])
        u = binc.get(b, 0)
        nbinc = dict(binc)
        nbinc[b] = u + 1
        slot = depth % 10
        sim.save(slot)
        self.floor = max(self.floor, depth - 9)
        snap = ctrl.clone()
        cands = self.stats.ranked(need, u)
        fresh = True
        for ci, cand in enumerate(cands):
            ok0, fail0 = self.stats.get(b, u, cand)
            trusted = ok0 >= 2 and fail0 == 0
            robust = True
            if ROBUST and not trusted:
                for pert in ((cand[0], cand[1] + 2), (max(3, cand[0] - 2), cand[1])):
                    if not fresh:
                        if depth < self.floor:
                            raise Abort()
                        sim.load(slot)
                    fresh = False
                    pc2 = snap.clone()
                    pc2.plan = list(pert)
                    sim.step(pc2.act())
                    self.trials += 1
                    r = self.run_until(pc2)
                    if r == "dead":
                        robust = False
                        break
                if not robust:
                    self.stats.note(b, u, cand, False)
                    continue
            if not fresh:
                if depth < self.floor:
                    raise Abort()
                sim.load(slot)
            fresh = False
            c = snap.clone()
            c.plan = list(cand)
            self.path.append((need["wx"], cand, b, u))
            if len(self.path) > len(self.best):
                self.best = list(self.path)
            self.trials += 1
            sim.step(c.act())
            ok = self.dfs(c, depth + 1, nbinc)
            if ok:
                if sim.budget_left >= self.stop_at:
                    self.stats.note(b, u, cand, True)
                return True
            self.stats.note(b, u, cand, False)
            self.path.pop()
            if sim.budget_left < self.stop_at:
                return True
        return False


LEARN = False       # the learned table is embedded in STATS0; set True to extend it with the save/load DFS


def solve(sim):
    if not LEARN:
        return []
    mem = sim.memory
    stats = build_stats(mem)
    nround = 0
    while sim.budget_left > 2500 and nround < MAX_ROUNDS:
        nround += 1
        sim.reset()
        s = Searcher(sim, stats)
        s.stop_at = max(150, sim.budget_left - ROUND_FRAMES)
        t0 = sim.budget_left
        try:
            ok = s.dfs(Ctrl(), 0, {})
        except Abort:
            ok = None
        except Exception as e:  # BudgetExceeded etc.
            print("search stop:", repr(e))
            ok = None
        used = t0 - sim.budget_left
        reached = s.best[-1][0] if s.best else 0
        done = bool(ok) and sim.budget_left >= s.stop_at
        print("round %d: ok=%s done=%s depth=%d best_wx=%d trials=%d used=%d" % (
            nround, ok, done, len(s.path), reached, s.trials, used), flush=True)
        try:
            mem["sok2"] = {k: v[0] for k, v in stats.d.items()}
            mem["sfail2"] = {k: v[1] for k, v in stats.d.items()}
            if hasattr(mem, "flush"):
                mem.flush()
        except Exception as e:
            print("mem err", e)
    try:
        mem["sok2"] = {k: v[0] for k, v in stats.d.items()}
        mem["sfail2"] = {k: v[1] for k, v in stats.d.items()}
        if hasattr(mem, "flush"):
            mem.flush()
    except Exception as e:
        print("mem err", e)
    import json
    print("STATS " + json.dumps(stats.d, sort_keys=True, separators=(",", ":")))
    return []

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n_SKY = (104, 136, 252)\n\n\ndef assess(frame, ctx):\n    try:\n        f = np.asarray(frame)\n        sky = np.all(f == np.array(_SKY, dtype=np.uint8), axis=2)\n        frac = float(sky.mean())\n        dead = bool(ctx.get("stuck", {}).get("is")) if isinstance(ctx, dict) else False\n        status = "stuck" if dead else "running"\n        return {"status": status, "where": {"hazard": None, "confidence": 0.0},\n                "progress": None, "resources": {}, "threats": [],\n                "stuck": {"is": dead}, "outlook": {"label": "ok", "p_through": 0.3},\n                "note": "sky=%.2f" % frac}\n    except Exception:\n        return None\n'}

