# ==================================================================================================
# Super Mario Bros. (NES) -- level 6-4  |  超级马里奥兄弟（NES） 关 6-4
# Pass rate on 100 random variants never used before: 93/100; fastest finish 1155 frames (19.25 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 93/100；最快一局 1155 帧（60 帧/秒，约 19.25 秒）。
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
#   6-4 closed-loop: scripted start route + podoboo gate at pit 1, predictive firebar/podoboo/flam
#   e rollout planner, Bowser-room state machine (no platform head-bump)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def PIT_TAIL_ON                    line 57    
#   def _mask                          line 75    
#   def _solid7                        line 79    
#   def find_mario7                    line 101   
#   def _comps                         line 125   
#   def _apex                          line 151   
#   def _ta                            line 155   
#   def arc                            line 159   height above takeoff, i frames after the first z frame
#   class C7                           line 169   
#   def gb_up                          line 1094  
#   def _solid                         line 1098  
#   def _hold_for                      line 1102  
#   def find_pods                      line 1112  podoboo-like blobs (red+skin, no block-orange nearby): list of (cx, cy)
#   class Ctl                          line 1140  
#   def solve                          line 1901  
#   def play                           line 1905  
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: 6-4 closed-loop: scripted start route + podoboo gate at pit 1, predictive firebar/podoboo/flame rollout planner, Bowser-room state machine (no platform head-bump)
import json
import numpy as np

TUNABLE = {}   # measured constants: nothing to auto-tune

GREYS = ((188, 188, 188), (124, 124, 124), (252, 252, 252), (228, 92, 16))
RED = (248, 56, 0)
BROWN = (172, 124, 0)

KS = (14, 8, 4)
DROP_LOOK = 56
DROP_X = 1400
DROP_HI = 1850
HOPKS = (6, 14, 8, 3)
PIT_K = 14
PIT_TAKE = 10        # takeoff when gap(right edge -> first void col) <= this
H = 100              # planning horizon (frames)
NOJUMP_H = 56
FORCE_STALL = 420
WALL_LOOK = 70
CEIL_LOOK = 80
CEIL_ON = False
OMEGA = 0.0368
URGENT_K = 43
PRE_WAIT = 130
WALL_PRE = False
VMAX = 3.0
ACC = 0.12
DEC_COAST = 0.09
DEC_BRAKE = 0.16
POD_G = 0.162
BALL_CAP = 30
BR0 = 3.0
BRS = 0.0248
MARGIN = 4.0
AIR_GO_BIAS = 6
GO_BOOST = 4.0
BOOST_LO = 260
BOOST_HI = 897
RELAX_MIN = 0.3
RELAX_T0 = 80
RELAX_SPAN = 280.0
AIR_ACC = 0.1
POST_STOP = False
POST_K = 24
POCKET_DROP = 8.0
ZM_X = 2
ZM_Y = 2
POCKET_MAX = 100
CLOSE_GAP = 48
RUNUP_CHECK = True
WHITE_FIX = False
PIT_TAIL = 0


def PIT_TAIL_ON(wx):
    return 1000 <= wx <= 1262


CENTER_R = 5.0
OFF_FIX = True
RET_FORCED = False
PK_LO = 0
PK_HI = 0
FLAME_VX = 1.0
FLAME_HW = 9.0
FLAME_HH = 5.0
FLAME_POST = 12
POST_AFTER = 0
_HT = [(0, 2), (34, 2), (42, 4), (49, 6), (56, 8), (64, 11), (71, 14), (79, 18), (85, 24), (200, 30)]
_RISE = [0, 0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68]


def _mask(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def _solid7(s, clean=False):
    m = np.zeros(s.shape[:2], bool)
    for c in GREYS:
        m |= _mask(s, c)
    if clean and WHITE_FIX:
        # firebar balls carry white pixels: not terrain
        w = _mask(s, (252, 252, 252))
        if w.any():
            r = _mask(s, RED)
            if r.any():
                n = r.copy()
                for d in (1, 2):
                    n[d:, :] |= r[:-d, :]
                    n[:-d, :] |= r[d:, :]
                n2 = n.copy()
                for d in (1, 2):
                    n2[:, d:] |= n[:, :-d]
                    n2[:, :-d] |= n[:, d:]
                m &= ~(w & n2)
    return m


def find_mario7(s, prev):
    if prev is None:
        gx0, gx1, gy0, gy1 = 0, 256, 40, 200
    else:
        px, py = prev
        gx0, gx1 = max(0, px - 24), min(256, px + 24)
        gy0, gy1 = max(32, py - 30), min(240, py + 30)
    win = s[gy0:gy1, gx0:gx1]
    ys, xs = np.nonzero(_mask(win, BROWN))
    if len(ys) < 3:
        return None
    ys = ys + gy0
    xs = xs + gx0
    cx = int(np.median(xs))
    cy = int(np.median(ys))
    sel = (abs(xs - cx) <= 9) & (abs(ys - cy) <= 12)
    if sel.sum() < 3:
        return None
    x0 = int(xs[sel].min())
    x1 = int(xs[sel].max())
    y1 = int(ys[sel].max())
    return (x0, x1, y1 - 15, y1)


def _comps(m, minarea=5):
    ys, xs = np.nonzero(m)
    if len(ys) > 3000:
        return []
    pts = set(zip(ys.tolist(), xs.tolist()))
    out = []
    while pts:
        p = pts.pop()
        st = [p]
        ay = [p[0]]
        ax = [p[1]]
        while st:
            y, x = st.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.remove(q)
                        st.append(q)
                        ay.append(q[0])
                        ax.append(q[1])
        if len(ay) >= minarea:
            out.append((len(ay), min(ax), max(ax), min(ay), max(ay)))
    return out


def _apex(K):
    return float(np.interp(K, [2, 8, 14], [34, 56, 68]))


def _ta(K):
    return float(np.interp(K, [2, 8, 14], [11, 16, 17]))


def arc(i, K):
    """height above takeoff, i frames after the first z frame"""
    ap = _apex(K)
    ta = _ta(K)
    if i < ta:
        r = _RISE[i] if i < len(_RISE) else 68
        return min(float(r), ap)
    return ap - 5.0 * (i - ta)


class C7:
    def __init__(self):
        self.vmax = VMAX
        self.prev = None
        self.cum = 0
        self.last_img = None
        self.last_feet = []
        self.hold = 0
        self.since = 99
        self.lost = 0
        self.sh_prev = 0
        self.wx = 0
        self.wxh = []
        self.best = -1
        self.stall = 0
        self.bb = None
        self.grounded = False
        self.t = 0
        self.hist = []      # (t, comps)
        self.bhist = []
        self.gap = None
        self._sol = None
        self._voids = []
        self.piv_w = {}
        self.piv_long = {}
        self.piv_hist = {}
        self.piv_changes = {}
        self.dbg = None
        self.last_hit = None
        self.pods = {}
        self.balls = []
        self.nflame = 0
        self.mode = ""
        self.pocket = None
        self.tail = 0
        self.why = None
        self.jt0 = -999
        self.jK = 0
        self.jfeet = 0

    # ---------------- perception ----------------
    def _shift(self, a, b):
        costs = []
        for sh in range(0, 6):
            w = 256 - sh
            costs.append(int((b[100:208, :w] != a[100:208, sh:]).any(-1).sum()))
        best = None
        for sh in range(0, 6):
            key = (costs[sh] + (0 if sh == self.sh_prev else 1e-3), sh)
            if best is None or key < best:
                best = key
        return best[1]

    def observe(self, s):
        s = np.array(s, copy=True)
        self.t += 1
        sh = 0
        if self.last_img is not None:
            sh = self._shift(self.last_img, s)
        self.sh_prev = sh
        self.cum += sh
        self.last_img = s
        bb = find_mario7(s, self.prev)
        self.bb = bb
        if bb is None:
            self.lost += 1
            if self.lost >= 8:
                self.prev = None
            return s
        self.lost = 0
        x0, x1, y0, y1 = bb
        self.prev = ((x0 + x1) // 2, (y0 + y1) // 2)
        feet = y1 + 1
        self.last_feet.append(feet)
        if len(self.last_feet) > 4:
            self.last_feet.pop(0)
        sol = _solid7(s[min(239, feet + 1): min(240, feet + 3), x0:x1 + 1])
        stable = len(self.last_feet) >= 3 and self.last_feet[-1] == self.last_feet[-2] == self.last_feet[-3]
        self.grounded = bool(sol.any()) and stable
        self.wx = self.cum + (x0 + x1) / 2.0
        self.wxh.append(self.wx)
        if len(self.wxh) > 6:
            self.wxh.pop(0)
        if self.wx > self.best:
            self.best = self.wx
            self.stall = 0
        else:
            self.stall += 1
        self.track_hazards(s)
        return s

    def _pkey(self, px, py):
        return (int(round(px / 4.0)), int(round(py / 4.0)))

    def vx(self):
        if len(self.wxh) >= 4:
            return (self.wxh[-1] - self.wxh[-4]) / 3.0
        return 0.0

    def track_hazards(self, s):
        x0, x1, y0, y1 = self.bb
        sub = _mask(s[40:190], RED)
        zx0 = max(0, x0 - ZM_X)
        zx1 = min(256, x1 + ZM_X + 1)
        zy0 = max(0, y0 - ZM_Y - 40)
        zy1 = max(0, y1 + ZM_Y - 40)
        sub[zy0:zy1, zx0:zx1] = False
        cs = []
        for (a, cx0, cx1, cy0, cy1) in _comps(sub):
            cs.append((a, cx0 + self.cum, cx1 + self.cum, cy0 + 40, cy1 + 40))
        self.hist.append((self.t, cs))
        if len(self.hist) > 8:
            self.hist.pop(0)
        old = None
        for back in (4, 3, 2, 1):
            if len(self.hist) > back:
                old = self.hist[-1 - back]
                break
        pod_boxes = []
        t = self.t
        for c in cs:
            a, cx0, cx1, cy0, cy1 = c
            w = cx1 - cx0 + 1
            h = cy1 - cy0 + 1
            cx = (cx0 + cx1) / 2.0
            cy = (cy0 + cy1) / 2.0
            vx = vy = 0.0
            matched = False
            dt_m = 1
            if old is not None:
                dt = t - old[0]
                dt_m = dt
                bd = 1e9
                for o in old[1]:
                    ox = (o[1] + o[2]) / 2.0
                    oy = (o[3] + o[4]) / 2.0
                    d = (ox - cx) ** 2 + (oy - cy) ** 2
                    if d < bd and abs(o[0] - a) <= max(12, a * 0.6):
                        bd = d
                        bo = (ox, oy)
                if bd < (7.0 * dt + 2) ** 2:
                    vx = (cx - bo[0]) / dt
                    vy = (cy - bo[1]) / dt
                    matched = True
            is_pod = a >= 80 and 10 <= w <= 17 and 11 <= h <= 17
            if is_pod:
                pod_boxes.append((cx0, cx1, cy0, cy1))
                key = int(round(cx / 16.0))
                for kk in (key - 1, key, key + 1):
                    if kk in self.pods and abs(self.pods[kk]["cx"] - cx) < 12:
                        key = kk
                        break
                vyi = (vy + POD_G * (dt_m / 2.0)) if matched else None
                prevp = self.pods.get(key)
                if vyi is None and prevp is not None and t - prevp["t"] <= 3:
                    vyi = prevp["vy"] + POD_G * (t - prevp["t"])
                if vyi is None:
                    vyi = 0.0 if cy0 < 150 else -4.0
                self.pods[key] = {"cx": cx, "t": t, "top": float(cy0), "vy": vyi}
        # firebar balls: yellow centre pixels next to red
        red = _mask(s[40:190], RED)
        yel = _mask(s[40:190], (252, 160, 68))
        near = red.copy()
        for d in (1, 2):
            near[d:, :] |= red[:-d, :]
            near[:-d, :] |= red[d:, :]
        near2 = near.copy()
        for d in (1, 2):
            near2[:, d:] |= near[:, :-d]
            near2[:, :-d] |= near[:, d:]
        bm = yel & near2
        bm[zy0:zy1, zx0:zx1] = False
        cur = []
        for (a, cx0, cx1, cy0, cy1) in _comps(bm, 3):
            cx = (cx0 + cx1) / 2.0 + self.cum
            cy = (cy0 + cy1) / 2.0 + 40
            inside = False
            for (px0, px1, py0, py1) in pod_boxes:
                if px0 - 1 <= cx <= px1 + 1 and py0 - 1 <= cy <= py1 + 1:
                    inside = True
                    break
            if not inside:
                cur.append((cx, cy))
        self.bhist.append((t, cur))
        if len(self.bhist) > 8:
            self.bhist.pop(0)
        oldb = None
        for back in (4, 3, 2, 1):
            if len(self.bhist) > back:
                oldb = self.bhist[-1 - back]
                break
        # pivot blocks: orange square with a ball at its centre
        orm = _mask(s[40:190], (228, 92, 16))
        pivs = []
        for (a, ox0, ox1, oy0, oy1) in _comps(orm, 60):
            w = ox1 - ox0 + 1
            h = oy1 - oy0 + 1
            if 12 <= w <= 18 and 12 <= h <= 18:
                bcx = (ox0 + ox1) / 2.0 + self.cum
                bcy = (oy0 + oy1) / 2.0 + 40
                for (cx, cy) in cur:
                    if abs(cx - bcx) <= 4 and abs(cy - bcy) <= 4:
                        pivs.append((bcx, bcy))
                        break
        rows = []
        for (cx, cy) in cur:
            vx = vy = 0.0
            if oldb is not None:
                dt = t - oldb[0]
                bd = 1e9
                bo = None
                for (ox, oy) in oldb[1]:
                    d = (ox - cx) ** 2 + (oy - cy) ** 2
                    if d < bd:
                        bd = d
                        bo = (ox, oy)
                if bo is not None and bd < (2.6 * dt + 1.5) ** 2:
                    vx = (cx - bo[0]) / dt
                    vy = (cy - bo[1]) / dt
            pi = -1
            pd = 56.0
            for i, (px, py) in enumerate(pivs):
                d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
                if d < pd:
                    pd = d
                    pi = i
            if pi < 0 and vx == 0.0 and vy == 0.0 and oldb is not None:
                # fast free-flying flame: wide horizontal match
                dt = t - oldb[0]
                bdx = None
                for (ox, oy) in oldb[1]:
                    if abs(oy - cy) <= 3.0 and abs(ox - cx) <= 6.5 * dt + 2.0 and abs(ox - cx) > 1.0:
                        if bdx is None or abs(ox - cx) < abs(bdx):
                            bdx = cx - ox
                if bdx is not None:
                    vx = bdx / dt
            rows.append((cx, cy, vx, vy, pi, pd))
        # angular velocity per pivot
        wsamp = {}
        for (cx, cy, vx, vy, pi, pd) in rows:
            if pi >= 0 and pd >= 10 and (abs(vx) + abs(vy)) > 0.05:
                px, py = pivs[pi]
                om = ((cx - px) * vy - (cy - py) * vx) / (pd * pd)
                wsamp.setdefault(pi, []).append(om)
        for pi, lst in wsamp.items():
            key = self._pkey(pivs[pi][0], pivs[pi][1])
            if len(lst) >= 2 and key not in self.piv_long:
                om = float(np.median(lst))
                old_om = self.piv_w.get(key)
                self.piv_w[key] = om if old_om is None else 0.7 * old_om + 0.3 * om
        # long-baseline angular velocity per pivot (more accurate)
        angs = {}
        for (cx, cy, vx, vy, pi, pd) in rows:
            if pi >= 0 and pd >= 12:
                px, py = pivs[pi]
                key = self._pkey(px, py)
                angs.setdefault(key, {})[int(round(pd / 8.0))] = float(np.arctan2(cy - py, cx - px))
        for key, d in angs.items():
            hist_k = self.piv_hist.setdefault(key, [])
            if hist_k:
                dl1 = []
                for b_, th in d.items():
                    if b_ in hist_k[-1][1]:
                        dd = th - hist_k[-1][1][b_]
                        dd = (dd + np.pi) % (2 * np.pi) - np.pi
                        dl1.append(dd)
                if len(dl1) >= 2 and abs(float(np.median(dl1))) > 0.1:
                    ch = self.piv_changes.setdefault(key, [])
                    ch.append(t)
                    if len(ch) > 6:
                        ch.pop(0)
            hist_k.append((t, d))
            if len(hist_k) > 26:
                hist_k.pop(0)
            for (t_old, d_old) in hist_k:
                if t - t_old >= 10:
                    dts = t - t_old
                    dl = []
                    for b_, th in d.items():
                        if b_ in d_old:
                            dd = th - d_old[b_]
                            dd = (dd + np.pi) % (2 * np.pi) - np.pi
                            dl.append(dd)
                    if len(dl) >= 2:
                        self.piv_w[key] = float(np.median(dl)) / dts
                        self.piv_long[key] = True
                    break
        balls = []
        nfl = 0
        for (cx, cy, vx, vy, pi, pd) in rows:
            piv = None
            om = None
            pkey = None
            if pi >= 0:
                key = self._pkey(pivs[pi][0], pivs[pi][1])
                om = self.piv_w.get(key)
                if om is None or abs(om) <= 0.006:
                    om = -OMEGA     # unmeasured: bars all turn at OMEGA (observed ccw); opposite spin also modelled
                else:
                    om = OMEGA if om > 0 else -OMEGA
                piv = pivs[pi]
                pkey = key
            if piv is None and abs(vx) >= FLAME_VX and abs(vy) < 1.0:
                nfl += 1
                balls.append((cx - FLAME_HW, cx + FLAME_HW, cy - FLAME_HH, cy + FLAME_HH, vx, vy, piv, om, pkey))
            else:
                balls.append((cx - 3.5, cx + 3.5, cy - 3.5, cy + 3.5, vx, vy, piv, om, pkey))
        self.balls = balls
        self.nflame = nfl

    # ---------------- planning ----------------
    def hazard_boxes(self, nk):
        """arrays of boxes [n, nk, 4] (x0,x1,y0,y1) world coords for k=0..nk-1"""
        ks = np.arange(nk, dtype=float)
        kc = np.minimum(ks, BALL_CAP)
        out = []
        for (x0, x1, y0, y1, vx, vy, piv, om, pkey) in self.balls:
            if piv is not None and om is not None:
                cx = (x0 + x1) / 2.0
                cy = (y0 + y1) / 2.0
                rx = cx - piv[0]
                ry = cy - piv[1]
                if rx * rx + ry * ry < CENTER_R * CENTER_R:
                    continue    # the pivot's own ball sits inside the solid block: not a hazard
                th = om * ks
                br_ = BR0 + BRS * float((rx * rx + ry * ry) ** 0.5)
                if pkey not in self.piv_long and abs(om) > 0.02:
                    # direction not yet confirmed by a long baseline: also model the opposite spin
                    th2 = -om * ks
                    c2_ = np.cos(th2)
                    s2_ = np.sin(th2)
                    nx2 = piv[0] + rx * c2_ - ry * s2_
                    ny2 = piv[1] + rx * s2_ + ry * c2_
                    out.append(np.stack([nx2 - br_, nx2 + br_, ny2 - br_, ny2 + br_], axis=1))
                c_ = np.cos(th)
                s_ = np.sin(th)
                nx = piv[0] + rx * c_ - ry * s_
                ny = piv[1] + rx * s_ + ry * c_
                out.append(np.stack([nx - br_, nx + br_, ny - br_, ny + br_], axis=1))
            else:
                kk_ = ks if (piv is None and abs(vx) >= FLAME_VX) else kc
                out.append(np.stack([x0 + vx * kk_, x1 + vx * kk_, y0 + vy * kk_, y1 + vy * kk_], axis=1))
        for key, p in list(self.pods.items()):
            age = self.t - p["t"]
            if age > 3:
                continue
            kk = ks + age
            top = p["top"] + p["vy"] * kk + 0.5 * POD_G * kk * kk
            top = np.where(top > 178.0, 400.0, top)
            cx = p["cx"]
            out.append(np.stack([np.full(nk, cx - 7.0), np.full(nk, cx + 7.0), top, top + 16.0], axis=1))
        if not out:
            return None
        return np.stack(out, axis=0)

    def ground_profile(self, s, feet):
        """solid-at-ground mask for screen columns"""
        ylo = max(34, feet - 48)
        sol = _solid7(s[ylo:max(190, min(240, feet + (DROP_LOOK if DROP_X <= self.wx <= DROP_HI else 20))), :], True)
        anyc = sol.any(axis=0)
        first = sol.argmax(axis=0)
        return np.where(anyc, feet - (ylo + first), -999.0)

    def fill_pockets(self, g, x0, x1, gh=None, start=None):
        """a drop of >= POCKET_DROP px followed (within POCKET_MAX cols) by a block/ledge back at our level is a
        trap (bars sweep it): mark the lowered columns as void so the planner jumps across from the ledge edge"""
        cm = (x0 + x1) // 2
        if PK_LO <= self.wx <= PK_HI:
            self.pocket = None
            return g
        if gh is None:
            gh = float(g[cm]) if g[cm] > -900 else 0.0
        g = g.copy()
        c = x1 + 1 if start is None else start
        self.pocket = None
        while c < 250:
            if g[c] > -900 and g[c] <= gh - POCKET_DROP:
                n = 0
                while c + n < 256 and -900 < g[c + n] <= gh - POCKET_DROP:
                    n += 1
                if n < 5:
                    c += n
                    continue
                c2 = c + n
                while c2 < 246 and not (g[c2:c2 + 10] >= gh - 4.0).all():
                    c2 += 1
                if c2 < 246 and c2 - c <= POCKET_MAX:
                    g[c:c2] = -999.0
                    self.pocket = (c, c2)
                break
            c += 1
        return g

    def air_decide(self, s):
        """mid-jump: choose go/coast/brake so the rest of the arc avoids the predicted hazards"""
        i = max(0, self.t - self.jt0)
        K = self.jK
        self.tail = PIT_TAIL if (K >= 12 and PIT_TAIL_ON(self.wx)) else 0
        feet0 = self.jfeet
        g = self.ground_profile(s, feet0)
        x0, x1 = self.bb[0], self.bb[1]
        g = self.fill_pockets(g, x0, x1, 0.0, max(0, x0 - 60))
        self._sol = _solid7(s)
        hz = self.hazard_boxes(H + 1)
        cum = self.cum
        best = None
        for mode in ("go", "coast", "brake"):
            f = self.simulate(mode, K, False, hz, g, feet0, cum, None, air0=i)
            if f is None:
                self.mode = "air-" + mode
                return self._air_keys(mode)
            if best is None or f > best[0] + (AIR_GO_BIAS if self.wx > 1450 else 0):
                best = (f, mode)
        self.mode = "air-best-" + best[1]
        return self._air_keys(best[1])

    def _air_keys(self, mode):
        if mode == "go":
            return {"Right", "x"}
        if mode == "coast":
            return set()
        return {"Left"}

    def simulate(self, mode, jumpK, jump_now, hz, g, feet, cum, void_x, take_world=None, take_off=0, air0=None):
        """returns k of first failure or None (ok)"""
        self.why = None
        go_boost = (mode == "go" and air0 is None and BOOST_LO <= self.wx <= BOOST_HI)
        x = self.wx
        v = max(0.0, self.vx())
        air = None
        K = jumpK
        if jump_now:
            air = 0
        if air0 is not None:
            air = air0
        halfw = 6.0
        bx = np.zeros((H + 1, 4))
        fail_k = None
        k_last = None
        gbase = 0.0
        stopped = False
        trig = []
        if mode == "go":
            if take_world is not None:
                trig.append(take_world + cum - take_off)
            for vs in self._voids:
                t_ = vs + cum - PIT_TAKE
                if not any(abs(t_ - q) < 8 for q in trig):
                    trig.append(t_)
            trig = sorted(t for t in trig if t > x + 6.0)
        ti = 0
        for k in range(H + 1):
            h = gbase
            if k > 0:
                if air is None:
                    if mode == "go":
                        v = min(self.vmax, v + ACC)
                    elif mode == "coast":
                        v = max(0.0, v - DEC_COAST)
                    elif mode == "back":
                        v = max(-1.5, v - 0.3)
                    elif mode == "stop":
                        v = max(0.0, v - DEC_BRAKE)
                    else:
                        v = max(-1.5, v - DEC_BRAKE)
                    if mode == "stop" and v > 0:
                        fx = int(x + 8 - cum)
                        if 0 <= fx < 256 and g[fx] > gbase + 4.0 and g[fx] > -900:
                            v = 0.0
                    x += v
                    if mode == "go" and ti < len(trig) and (x + 6.0) >= trig[ti]:
                        air = 0
                        K = PIT_K if (jump_now or air0 is not None) else jumpK
                        ti += 1
                else:
                    if mode == "go":
                        v = min(self.vmax, v + AIR_ACC)
                    elif mode == "brake":
                        v = max(-1.5, v - AIR_ACC)
                    x += v
            if air is not None:
                h = arc(air, K)
                sx0 = max(0, min(255, int(x - 5 - cum)))
                sx1 = max(0, min(255, int(x + 5 - cum)))
                gm = float(g[sx0:sx1 + 1].max())
                blocked = False
                if k > 0 and air >= 2:
                    c0 = max(0, int(x - cum) - 6)
                    c1 = min(255, int(x - cum) + 6)
                    r0 = max(34, int(feet - h - 15))
                    r1 = min(189, int(feet - h - 3))
                    if r1 > r0 and c1 > c0 and self._sol[r0:r1, c0:c1 + 1].any():
                        blocked = True
                if blocked:
                    if fail_k is None:
                        fail_k = k
                        self.why = ("ceil", k, round(x - cum), round(h), round(gm))
                    k_last = k
                    air = None
                    h = max(h, 0.0)
                elif gm > -900 and gm > h + 2 and k > 0:
                    # body hits a taller wall
                    if fail_k is None:
                        fail_k = k
                        self.why = ("wall", k, round(x - cum), round(h), round(gm), air, K)
                    k_last = k
                    air = None
                    h = max(gm, 0.0)
                elif air > _ta(K) + 1 and h <= max(gm, 0.0 if gm <= -900 else gm):
                    if gm <= -900:
                        if fail_k is None:
                            fail_k = k
                            self.why = ("pit", k, round(x - cum), round(h), round(gm))
                        h = 0.0
                    else:
                        h = gm
                        gbase = gm
                    air = None
                    k_last = k
                    if POST_STOP and mode == "go" and not stopped:
                        mode = "stop"
                        stopped = True
                    while ti < len(trig) and trig[ti] < x + 6.0:
                        ti += 1
                else:
                    air += 1
            else:
                sx = int(x - cum)
                if 0 <= sx < 256 and g[sx] <= -900:
                    # walking off an edge (centre over void)
                    if void_x is not None and sx >= void_x:
                        if fail_k is None:
                            self.why = ("edge", k, sx)
                        fail_k = k if fail_k is None else fail_k
                elif 0 <= sx < 256 and g[sx] < gbase - 2.0 and k > 0:
                    # stepping down onto lower floor: body follows the floor
                    gbase = float(g[sx])
                    h = gbase
            bx[k] = (x - halfw + 1, x + halfw - 1, feet - h - 14, feet - h)
        if air is not None:
            kend = H
        elif k_last is not None:
            kend = min(H, k_last + 3 + (POST_K if stopped else 0) + max(POST_AFTER, self.tail, FLAME_POST if self.nflame else 0))
        else:
            kend = NOJUMP_H
        if hz is not None:
            n = hz.shape[0]
            m = bx[None, :, :]
            mg = (MARGIN + (GO_BOOST if go_boost else 0.0)) * min(1.0, max(RELAX_MIN, 1.0 - (self.stall - RELAX_T0) / RELAX_SPAN))
            ov = (hz[:, :, 0] - mg < m[:, :, 1]) & (hz[:, :, 1] + mg > m[:, :, 0]) & \
                 (hz[:, :, 2] - mg < m[:, :, 3]) & (hz[:, :, 3] + mg > m[:, :, 2])
            hit = ov.any(axis=0)
            hit[kend + 1:] = False
            if hit.any():
                kh = int(np.argmax(hit))
                self.last_hit = (kh, int(np.argmax(ov[:, kh])), n, kend)
                fail_k = kh if fail_k is None else min(fail_k, kh)
        if fail_k is not None and fail_k > kend:
            fail_k = None
        return fail_k

    def _hold_for(self, s, cx, feet):
        sol = _solid7(s)
        h = 0
        for c in range(min(255, cx), min(256, cx + 4)):
            hh = 0
            for dy in range(1, 100):
                y = feet - dy
                if y < 32:
                    break
                if sol[y, c]:
                    hh = dy
                elif hh and dy - hh > 3:
                    break
            h = max(h, hh)
        need = h + 12
        k = 30
        for (a, ka), (b, kb) in zip(_HT, _HT[1:]):
            if need <= b:
                k = int(round(ka + (kb - ka) * (need - a) / max(1, b - a))) + 1
                break
        return max(2, k)

    def find_wall(self, sol, x1, feet, s):
        """first solid column ahead at body height within WALL_LOOK px: (screen_x, hold, height) or None"""
        bx0 = x1 + 3
        bx1 = min(256, bx0 + WALL_LOOK)
        by0 = max(32, feet - 16)
        by1 = max(by0 + 1, feet - 1)
        if bx1 <= bx0:
            return None
        m = sol[by0:by1, bx0:bx1]
        cnt = m.sum(axis=0)
        cols = np.nonzero(cnt >= 2)[0]
        if len(cols) == 0:
            return None
        wx = bx0 + int(cols.min())
        h = 0
        for c in range(wx, min(256, wx + 4)):
            hh = 0
            for dy in range(1, 100):
                y = feet - dy
                if y < 32:
                    break
                if sol[y, c]:
                    hh = dy
                elif hh and dy - hh > 3:
                    break
            h = max(h, hh)
        return wx, self._hold_for(s, wx, feet), h

    def zones(self, y0):
        """world-x intervals around floor-level firebar pivots that the bars can sweep (Mario must not wait there)"""
        piv = {}
        for b in self.balls:
            if b[6] is None:
                continue
            px, py = b[6]
            cx = (b[0] + b[1]) / 2.0
            cy = (b[2] + b[3]) / 2.0
            d = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5
            e = piv.setdefault(b[8], [px, py, 0.0])
            e[2] = max(e[2], d)
        zs = []
        for px, py, d in piv.values():
            reach = max(d + 4.0, ZREACH)
            if py < ZFLOOR_Y:
                continue
            zs.append((px - reach - ZPAD, px + reach + ZPAD))
        return zs

    def retreat_action(self, y0):
        """inside a bar's sweep zone with no safe plan: walk left to the nearest bar-free spot and wait there"""
        zs = self.zones(y0)
        x = self.wx
        if not zs or not any(a <= x <= b for a, b in zs):
            return None
        if self.vx() > ZRET_VMAX:
            return None
        t = x
        cand = None
        while t > x - 160:
            if not any(a <= t <= b for a, b in zs):
                cand = t
                break
            t -= 2.0
        if cand is None:
            return None
        lo = cand
        while lo > cand - 160 and not any(a <= lo <= b for a, b in zs):
            lo -= 2.0
        free_w = cand - lo
        tx = cand - (free_w / 2.0 if free_w < 40 else 6.0)
        v = self.vx()
        dist = x - tx
        if dist > 5.0 + (max(0.0, -v) * 3.0):
            return {"Left"}
        if dist < -4.0:
            return {"Right"}
        return set()

    def decide2(self, s):
        x0, x1, y0, y1 = self.bb
        feet = y1 + 1
        sol = _solid7(s)
        self._sol = sol
        g = self.ground_profile(s, feet)
        g = self.fill_pockets(g, x0, x1)
        void_x = None
        run = 0
        for cx in range(x1 + 1, 256):
            if g[cx] <= -900:
                run += 1
                if run >= 3:
                    void_x = cx - 2
                    break
            else:
                run = 0
        gap_p = None if void_x is None else void_x - x1
        self.gap = gap_p
        voids = []
        run = 0
        for cx in range(x1 + 1, 256):
            if g[cx] <= -900:
                run += 1
                if run == 3:
                    voids.append(cx - 2)
            else:
                run = 0
        self._voids = voids
        # ceiling clearance -> max apex
        cx_a = max(0, x0 - 2)
        cx_b = min(256, x1 + CEIL_LOOK)
        band = sol[34:max(35, feet - 18), cx_a:cx_b]
        rows = np.nonzero(band.sum(axis=1) >= 2)[0]
        ceil_y = 34 + int(rows.max()) if len(rows) else 40
        lim = feet - 14 - ceil_y - 3
        if not CEIL_ON:
            lim = 200
        ksafe = [K for K in (14, 8, 4, 2) if _apex(K) <= lim] or [2]
        self.ksafe = ksafe
        hz = self.hazard_boxes(H + 1)
        cum = self.cum
        v = self.vx()
        wall = self.find_wall(sol, x1, feet, s)
        obst = None
        if void_x is not None:
            obst = {"kind": "pit", "x": void_x, "gap": gap_p, "off": PIT_TAKE, "Ks": tuple(ksafe)}
        if wall is not None:
            wx_, wh, hw = wall
            gap_w = wx_ - x1
            if obst is None or gap_w < obst["gap"]:
                need = hw + 4
                inn = 17
                for i, r in enumerate(_RISE):
                    if r >= need:
                        inn = i
                        break
                vr = min(VMAX, (max(v, 0.0) ** 2 + 2 * ACC * max(0.0, gap_w - 6.0)) ** 0.5) if OFF_FIX else 0.0
                off = max(6.0, min(3.2, max(v, 2.0, vr)) * inn)
                kcap = max(2, int(np.interp(max(0, lim), [34, 56, 68], [2, 8, 14])))
                ks_ = []
                for kk in (wh, wh + 4, wh + 10):
                    kk = min(kk, kcap)
                    if kk not in ks_:
                        ks_.append(kk)
                obst = {"kind": "wall", "x": wx_, "gap": gap_w, "off": off, "Ks": tuple(ks_), "tall": hw >= 24}
        stall_force = self.stall > FORCE_STALL
        self.tail = PIT_TAIL if (obst is not None and obst["kind"] == "pit" and PIT_TAIL_ON(self.wx)) else 0
        if obst is not None and obst["gap"] <= obst["off"]:
            needs_run = obst["kind"] == "pit" or obst.get("tall", False)
            if needs_run and v < 2.5 and not stall_force:
                self.mode = "runup"
                if RUNUP_CHECK:
                    rb = []
                    for md in ("back", "coast", "brake"):
                        f = self.simulate(md, 0, False, hz, g, feet, cum, void_x)
                        if f is None:
                            self.mode = "runup-" + md
                            return ({"Left"} if md != "coast" else set()), 0
                        rb.append((f, md))
                    rb.sort(key=lambda r: -r[0])
                    if rb[0][0] > 6:
                        self.mode = "runup-best" + rb[0][1]
                        return ({"Left"} if rb[0][1] != "coast" else set()), 0
                return {"Left"}, 0
            res = []
            for K in obst["Ks"]:
                f = self.simulate("go", K, True, hz, g, feet, cum, None)
                if f is None:
                    self.mode = obst["kind"] + str(K)
                    return {"Right", "x"}, K
                res.append((f, "J", K))
            if stall_force:
                resj = [r for r in res if r[1] == "J"]
                resj.sort(key=lambda r: -r[0])
                self.mode = obst["kind"] + "-stallforce"
                return {"Right", "x"}, resj[0][2]
            for mode in ("coast", "brake"):
                f = self.simulate(mode, 0, False, hz, g, feet, cum, void_x)
                if f is None:
                    self.mode = obst["kind"] + "wait-" + mode
                    return ({"Left"} if mode == "brake" else set()), 0
                res.append((f, mode, 0))
            if ZRET_ON and RET_FORCED:
                try:
                    rt = self.retreat_action(y0)
                except Exception:
                    rt = None
                if rt is not None:
                    self.mode = obst["kind"] + "-zretreat"
                    return rt, 0
            res.sort(key=lambda r: -r[0])
            f, md, K = res[0]
            self.mode = obst["kind"] + "-forced-" + md
            if md == "J":
                return {"Right", "x"}, K
            return ({"Left"} if md == "brake" else set()), 0
        # general
        tw = None
        toff = 0
        tK = KS[0]
        if obst is not None:
            tw = obst["x"]
            toff = obst["off"]
            tK = obst["Ks"][-1]
            for K in obst["Ks"]:
                f = self.simulate("go", K, False, None, g, feet, cum, void_x, tw, toff)
                if f is None:
                    tK = K
                    break
        res = []
        self.last_hit = None
        fg = self.simulate("go", tK, False, hz, g, feet, cum, void_x, tw, toff)
        self.dbg = (fg, self.last_hit, None if obst is None else (obst["kind"], obst["gap"], round(obst["off"], 1)), len(self.balls), self.why, tK)
        far_ok = fg is not None and fg > URGENT_K and not (
            obst is not None and (obst["kind"] == "pit" or WALL_PRE) and obst["gap"] <= PRE_WAIT)
        if fg is None or far_ok or stall_force:
            self.mode = "go" if (fg is None or far_ok) else "go-forced"
            return {"Right", "x"}, 0
        res.append((fg, "go"))
        # hop options (while moving) to get over sweeping bars
        self.hopres = {}
        if tw is None or (obst is not None and obst["gap"] > obst["off"] + 40):
            for K in HOPKS:
                if _apex(K) > lim:
                    continue
                f = self.simulate("go", K, True, hz, g, feet, cum, None)
                self.hopres[K] = f
                if f is None:
                    self.mode = "hop%d" % K
                    return {"Right", "x"}, K
        if ZRET_ON:
            try:
                rt = self.retreat_action(y0)
            except Exception:
                rt = None
            if rt is not None:
                self.mode = "zretreat"
                return rt, 0
        close = (obst is not None and (obst["kind"] == "pit" or obst.get("tall", False))
                 and obst["gap"] < CLOSE_GAP and v < 1.6)
        for mode in (("back", "coast", "brake") if close else ("coast", "brake", "back")):
            f = self.simulate(mode, 0, False, hz, g, feet, cum, void_x)
            if f is None:
                self.mode = mode
                return ({"Left"} if mode in ("brake", "back") else set()), 0
            res.append((f, mode))
        # last resort: a hop even close to an obstacle (the rollout checks the landing)
        for K in (6, 3, 14):
            if _apex(K) > lim:
                continue
            f = self.simulate("go", K, True, hz, g, feet, cum, None)
            if f is None:
                self.mode = "hopL%d" % K
                return {"Right", "x"}, K
        res.sort(key=lambda r: -r[0])
        md = res[0][1]
        self.mode = "best-" + md
        return ({"Right", "x"} if md == "go" else ({"Left"} if md in ("brake", "back") else set())), 0


SKIN = (252, 160, 68)
GREY = ((188, 188, 188), (124, 124, 124), (252, 252, 252))
AXE = (228, 92, 16)

HOLD_PIT = 24
OBS_DIST = 22
PIT_DIST = 2
GATE_ON = True
CORR_ON = True
CORR_LO = 520
CORR_HI = 2400
GATE_LO = 340
GATE_HI = 400
PRE_X = 296
BACK_X = 298
POD_X0 = 415
POD_X1 = 465
EM_LO = 0
EM_HI = 14
GATE_MAX = 1200
AIR_ON = True
AXE_X = 242
JSTATIC = 2
PLAT_M = 8
SLIDE_X0 = 97
DODGE_R_D = 72
GO_BL = 0
GO_BL_MIN = 0
HAM_DX = 17
HAM_T = 40
HAM_HOP_DX = 30
HAM_LEFT_X0 = 98
FL_TTC = 9.0
FL_HOLD = 6
UP_G = [127.0, 128.0, 131.0, 134.0, 139.0, 145.0, 152.0, 157.0]
UP_T = [28.0, 24.0, 20.0, 16.0, 12.0, 8.0, 4.0, 0.0]
DN_G = [127.0, 129.0, 133.0, 137.0, 142.0, 149.0, 156.0, 158.0]
DN_T = [32.0, 36.0, 40.0, 44.0, 48.0, 52.0, 56.0, 60.0]
TU0 = 14.0
TU1 = 42.0
DASH_LAT = 2
DASH_ACC = 0.1991
DASH_M = 4.0
DBAND_FAR = 66
FLC_ON = True
FLC_HOLD = 7
FLC_TTC = 11.0
ZRET_ON = True
ZRET_VMAX = 99.0
ZFLOOR_Y = 138
ZREACH = 48.0
ZPAD = 6.0
DASH_JX = 205
DASH_BR_MAX = 204
DBAND_HI = 30
DBAND_LO = 14
JMIN = 11
JMAX = 58
JHOLD = 16
JX0 = 126          # min screen x0 at takeoff (platform head-bump guard)
JVX = 2.0
RUN_V = 1.8
JSLOW = 22
BACK_D = 60
FAR_D = 90
UNDER_ON = True
UNDER_Y = 134
UNDER_D = 50
UNDER_RUN = 14
RETREAT_D = 62
RETREAT_X0 = 102
UNDER_FAR = 50
UNDER_NEAR = 34
UNDER_MAXWAIT = 700
END_X = 2050
BOWSER_MIN = 60
END_JUMP_D = 30
END_HOLD = 9
END_STATIC = 6
END_GROUND_Y = 150
ROUTE = [["j", 47, 13], ["j", 53, 13], ["j", 101, 6], ["j", 190, 6], ["j", 280, 6], ["w", 370, 10], ["j", 399, 20], ["j", 487, 6], ["j", 579, 6], ["j", 621, 6], ["j", 681, 6], ["j", 741, 6], ["j", 781, 11], ["j", 795, 6], ["j", 841, 6], ["j", 904, 6], ["j", 966, 6], ["j", 1026, 6], ["j", 1086, 6], ["j", 1146, 6], ["j", 1233, 6], ["j", 1264, 3], ["j", 1335, 3], ["j", 1357, 10], ["j", 1414, 10], ["j", 1469, 3], ["j", 1478, 10], ["j", 1542, 10], ["j", 1638, 6], ["j", 1752, 6], ["j", 1790, 6], ["j", 1825, 11], ["j", 1900, 6], ["j", 1990, 10], ["j", 2096, 6], ["w", 2185, 6], ["j", 2211, 3]]


def gb_up(bs):
    return bs[2] <= UNDER_Y


def _solid(s):
    return _mask(s, GREY[0]) | _mask(s, GREY[1]) | _mask(s, GREY[2])


def _hold_for(h, hs=1.0):
    need = h + 12
    k = HOLD_PIT
    for (a, ka), (b, kb) in zip(_HT, _HT[1:]):
        if need <= b:
            k = int(round(ka + (kb - ka) * (need - a) / max(1, b - a))) + 1
            break
    return max(2, int(round(k * hs)))


def find_pods(s, bb, thr=80, w=12):
    """podoboo-like blobs (red+skin, no block-orange nearby): list of (cx, cy)"""
    y0, y1 = 60, 194
    sub = s[y0:y1]
    m = (_mask(sub, RED) | _mask(sub, SKIN)).astype(np.int32)
    o = _mask(sub, AXE).astype(np.int32)
    if bb is not None:
        x0, x1, by0, by1 = bb
        m[max(0, by0 - 8 - y0):max(0, by1 + 4 - y0), max(0, x0 - 3):x1 + 4] = 0
    ii = np.pad(m.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    box = ii[w:, w:] - ii[:-w, w:] - ii[w:, :-w] + ii[:-w, :-w]
    cand = box >= thr
    if not cand.any():
        return []
    mo = np.pad(o, 4)
    io = np.pad(mo.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    W2 = w + 8
    obox = io[W2:, W2:] - io[:-W2, W2:] - io[W2:, :-W2] + io[:-W2, :-W2]
    ok = cand & (obox[:box.shape[0], :box.shape[1]] <= 2)
    ys, xs = np.nonzero(ok)
    out = []
    for y, x in sorted(zip(ys, xs)):
        cx, cy = int(x + w // 2), int(y + w // 2 + y0)
        if all(abs(cx - a) > 12 or abs(cy - b) > 12 for a, b in out):
            out.append((cx, cy))
    return out


class Ctl:
    def __init__(self, route=None):
        self.prev = None
        self.cum = 0
        self.last_img = None
        self.last_feet = []
        self.hold = 0
        self.since = 99
        self.lost = 0
        self.sh_prev = 0
        self.wx = 0
        self.best = -1
        self.stall = 0
        self.bb = None
        self.grounded = False
        self.t = 0
        self.pause_left = 0
        self.route = route
        self.ri = 0
        self.vx = 0.0
        self.gating = False
        self.g_t0 = 0
        self.g_log = []
        self.gate_done = False
        self.g_stage = "brake"
        self.g_keys = set()
        self.pod_vis = False
        self.pod_emerge = -999
        self.h = C7()
        self.sflag = []
        self.in_corr = False
        self.left_corr = False
        self.trig_t = None
        self.after_route = None
        self.end_mode = False
        self.bow_prev = None
        self.end_wait = 0
        self.end_log = []
        self.static_n = 0
        self.retreat = False
        self.under_t = -999
        self.plat = None
        self.hams = []
        self.ham_prev = []
        self.fh = 0
        self.dash = False
        self.dodge_n = 0
        self.bow_dir = 0
        self.flame_air = False

    def bowser_span(self, s):
        sub = s[40:200]
        g = (sub[:, :, 0] == 0) & (sub[:, :, 1] == 168) & (sub[:, :, 2] == 0)
        if int(g.sum()) < BOWSER_MIN:
            return None
        xs = np.nonzero(g.any(axis=0))[0]
        ys = np.nonzero(g.any(axis=1))[0]
        return int(xs.min()), int(xs.max()), int(ys.max()) + 40

    def hammer_threat(self, s, x0, x1, y0, bs=None):
        """grey/white pixels in the open air above Mario's column: (mean x, max y) of the nearest one or None"""
        ya, yb = 52, min(158, y0 + 3)
        if yb <= ya:
            return None
        sub = s[ya:yb, 28:210]
        m = ((sub[:, :, 0] == sub[:, :, 1]) & (sub[:, :, 1] == sub[:, :, 2]) & (sub[:, :, 0] >= 120))
        # floating platform (white dots)
        m[max(0, 92 - ya):max(0, 110 - ya), 128 - 28:172 - 28] = False
        if bs is not None:
            # Bowser's own white spikes/claws: mask his columns
            m[:, max(0, bs[0] - 6 - 28):max(0, bs[1] + 6 - 28)] = False
        cxm = (x0 + x1) / 2.0
        c0 = int(max(28, cxm - HAM_DX)) - 28
        c1 = int(min(210, cxm + HAM_DX)) - 28
        if c1 <= c0:
            return None
        col = m[:, c0:c1]
        if int(col.sum()) < 3:
            return None
        ys, xs = np.nonzero(col)
        return (float(xs.mean() + c0 + 28), float(ys.max() + ya))

    def safe_left(self, x0):
        """False if the current leftward slide would carry Mario into the podoboo lane (x0 < ~96)"""
        v = self.vx
        proj = x0 + (v * abs(v) / 0.26 if v < 0 else 0.0)
        return proj > SLIDE_X0

    def update_hammers(self, s, bs):
        """track grey blobs (hammers) in the open air: self.hams = [(cx, cy, vx, vy)]"""
        sub = s[52:158, 28:210]
        m = ((sub[:, :, 0] == sub[:, :, 1]) & (sub[:, :, 1] == sub[:, :, 2]) & (sub[:, :, 0] >= 120))
        m[92 - 52:110 - 52, 128 - 28:172 - 28] = False
        if bs is not None:
            m[:, max(0, bs[0] - 8 - 28):max(0, bs[1] + 8 - 28)] = False
        ys, xs = np.nonzero(m)
        bl = []
        if len(ys) < 600:
            for x, y in sorted(zip(xs.tolist(), ys.tolist())):
                for b in bl:
                    if abs(b[0] / b[2] - x) <= 9 and abs(b[1] / b[2] - y) <= 9:
                        b[0] += x
                        b[1] += y
                        b[2] += 1
                        break
                else:
                    bl.append([x, y, 1])
        cur = [(b[0] / b[2] + 28, b[1] / b[2] + 52) for b in bl if b[2] >= 4]
        out = []
        for (cx, cy) in cur:
            best = None
            for (px, py, pvx, pvy, age) in self.ham_prev:
                d = abs(px - cx) + abs(py - cy)
                if d <= 14 and (best is None or d < best[0]):
                    best = (d, px, py, pvx, pvy, age)
            if best is None:
                out.append((cx, cy, 0.0, 0.0, 0))
            else:
                rvx, rvy = cx - best[1], cy - best[2]
                if best[5] == 0:
                    out.append((cx, cy, rvx, rvy, 1))
                else:
                    out.append((cx, cy, 0.5 * rvx + 0.5 * best[3], 0.5 * rvy + 0.5 * best[4], 2))
        self.ham_prev = out
        self.hams = out

    def ham_choose(self, x0, x1, y0, left_ok, right_ok):
        """pick Left/Right/stay minimising predicted hammer overlaps; None = no threat"""
        cxm = (x0 + x1) / 2.0
        th = []
        for (cx, cy, vx, vy, age) in self.hams:
            if age == 0 or vy < 1.5:
                continue
            t = (y0 - 3 - cy) / vy
            if t < -4 or t > HAM_T:
                continue
            th.append((cx + vx * max(t, 0.0), max(t, 0.0)))
        if not th:
            return None
        v0 = self.vx
        best = None
        for name, a in (("S", 0.0), ("L", -0.1), ("R", 0.1)):
            if (name == "L" and not left_ok) or (name == "R" and not right_ok):
                continue
            sc = 0.0
            for (xh, t) in th:
                if name == "S":
                    xm = cxm + v0 * min(t, 12) * 0.5
                else:
                    xm = cxm + v0 * t + 0.5 * a * t * t
                    vmax_d = 3.0 * (1 if a > 0 else -1)
                    if abs(v0 + a * t) > 3.0:
                        xm = cxm + vmax_d * t
                sc += max(0.0, 1.0 - abs(xh - xm) / 17.0)
            if name == "S":
                sc *= 0.999
            if best is None or sc < best[0] - 1e-9:
                best = (sc, name)
        if best is None:
            return None
        return best

    def ham_dir(self, x0, x1, y0):
        """nearest falling-hammer threat: (hit_x, t_hit) or None"""
        cxm = (x0 + x1) / 2.0
        best = None
        for (cx, cy, vx, vy, age) in self.hams:
            if age == 0 or vy < 1.5:
                continue
            t = (y0 - 3 - cy) / vy
            if t < -4 or t > HAM_T:
                continue
            xh = cx + vx * max(t, 0.0)
            if abs(xh - cxm) < HAM_DX and (best is None or t < best[1]):
                best = (xh, t)
        return best

    def bow_phase(self, gb):
        """frames since Bowser's hop started (None = on the ground), from his green-bottom y and direction"""
        if gb >= 156:
            return None
        if self.bow_dir < 0:
            return float(np.interp(gb, UP_G, UP_T))
        return float(np.interp(gb, DN_G, DN_T))

    def dash_ok(self, x0, x1, bs, tc, v0):
        """simulate a full-speed run from now: does Mario pass beneath the hopping Bowser?"""
        if bs[1] > DASH_BR_MAX:
            return False, 0, 0
        v = max(v0, 0.0)
        xa = float(x1)
        xb = float(x0)
        n1 = None
        n2 = None
        for n in range(1, 100):
            if n > DASH_LAT:
                v = min(3.0, v + DASH_ACC)
            xa += v
            xb += v
            bl = bs[0] - 0.25 * n
            br = bs[1] - 0.25 * n
            if n1 is None and xa >= bl - 1:
                n1 = n
            if xb > br + DASH_M:
                n2 = n
                break
        if n2 is None or n1 is None:
            return False, 0, 0
        return (tc + n1 >= TU0 and tc + n2 <= TU1), n1, n2

    def ball_dodge(self, y0, y1):
        h = self.h
        if h.bb is None:
            return None
        nk = 32
        hz = h.hazard_boxes(nk)
        if hz is None:
            return None
        ks = np.arange(nk, dtype=float)
        mg = 2.0
        v0 = h.vx()
        res = []
        for name in ("run", "stand", "back"):
            v = v0
            x = h.wx
            xs = np.zeros(nk)
            for k in range(nk):
                if k > 0:
                    if name == "run":
                        v = min(3.0, v + 0.12)
                    elif name == "stand":
                        v = max(0.0, v - 0.15)
                    else:
                        v = max(-1.5, v - 0.16)
                    x += v
                xs[k] = x
            ov = (hz[:, :, 0] - mg < xs[None, :] + 5.0) & (hz[:, :, 1] + mg > xs[None, :] - 5.0) & \
                 (hz[:, :, 2] - mg < y1 + 1) & (hz[:, :, 3] + mg > y0)
            hit = ov.any(axis=0)
            fk = int(np.argmax(hit)) if hit.any() else 99
            res.append((fk, name))
        if res[0][0] >= 99 or res[0][0] > 24:
            self.dodge_n = max(0, self.dodge_n - 1)
            return None
        for fk, name in res[1:]:
            if fk >= 99:
                return {"stand": set(), "back": {"Left"}}[name]
        best = max(res, key=lambda r: r[0])
        if best[1] == "run":
            return None
        return {"stand": set(), "back": {"Left"}}[best[1]]

    def bowser_policy(self, s, bs, x0, x1, keys):
        d = bs[0] - x1
        gb = bs[2]
        pg = self.bow_prev
        if pg is not None:
            if gb < pg:
                self.bow_dir = -1
            elif gb > pg:
                self.bow_dir = 1
        self.bow_prev = gb
        tc = self.bow_phase(gb)
        self.end_wait += 1
        if x1 >= bs[0] - 3:
            # already level with / under him: keep going if he is up, else back off
            if tc is not None and gb <= 142:
                self.dash = True
                return keys
            return {"Left"} if x0 > RETREAT_X0 else set()
        if tc is not None:
            ok, n1, n2 = self.dash_ok(x0, x1, bs, tc, self.vx)
            if ok:
                self.dash = True
                self.end_log.append((self.t, "dash", d, gb, round(tc), n1, n2, round(self.vx, 1)))
                return keys
        if d > DBAND_FAR:
            return keys
        vxx = self.vx
        if vxx > 0.8 and d < 10 + 9 * vxx:
            # carrying speed toward him: brake so we stop inside the waiting band
            return {"Left"} if x0 > RETREAT_X0 else set()
        if d > DBAND_HI:
            return {"Right"} if vxx < 1.5 else set()
        if d < DBAND_LO:
            if x0 <= RETREAT_X0:
                return set()
            return {"Left"} if self.safe_left(x0) else set()
        return set()

    def end_act(self, s):
        """bridge room: keep clear of Bowser, run under him when he is airborne, or jump over him from the
        right of the floating platform (x0 >= JX0, moving right) so the head never bumps it"""
        keys = {"Right", "x"}
        try:
            self.update_hammers(s, self.bowser_span(s))
        except Exception as ex:
            self.sflag.append(repr(ex))
            self.hams = []
        if self.dash:
            bsd = self.bowser_span(s)
            if self.bb is None or bsd is None or self.bb[0] > bsd[1] + 3 or self.bb[1] > AXE_X:
                self.dash = False
            else:
                self.end_log.append((self.t, "dashing", bsd[0] - self.bb[1], bsd[2]))
                if self.grounded and self.since > 1 and self.bb[1] >= DASH_JX and self.bb[0] > bsd[1] + 2:
                    hd = self.reactive(s)
                    if hd > 0:
                        self.end_log.append((self.t, "dashjump", hd))
                        self.hold = hd - 1
                        if self.hold == 0:
                            self.since = 0
                        return {"Right", "x", "z"}
                return keys
        if self.fh > 0:
            self.fh -= 1
            return {"z"}
        if not (self.grounded and self.since > 1):
            return set() if self.flame_air else keys
        self.flame_air = False
        x0, x1, y0, y1 = self.bb
        bs = self.bowser_span(s)
        hold = 0
        # lobbed fireballs / podoboos: pick run / stand / back so Mario's ground box avoids their predicted paths
        try:
            bd = self.ball_dodge(y0, y1)
            if bd is not None and self.dodge_n < 90:
                self.dodge_n += 1
                self.end_log.append((self.t, "bdodge", sorted(bd)))
                return set(bd)
        except Exception as ex:
            self.sflag.append(repr(ex))
        # low flame coming at body height: short hop over it
        try:
            for b in self.h.balls:
                if b[6] is not None or abs(b[4]) < 1.0 or abs(b[5]) >= 1.0:
                    continue
                cy = (b[2] + b[3]) / 2.0
                if not (y0 - 4 <= cy <= y1 + 6):
                    continue
                dx = b[0] - (self.cum + x1)
                closing = -b[4] + max(self.vx, 0.0)
                if dx < -14 or closing < 0.8 or b[4] > 0:
                    continue
                ttc = max(dx, 0.0) / closing
                if ttc <= FL_TTC:
                    self.end_log.append((self.t, "flame", round(dx), round(cy), round(ttc, 1)))
                    self.fh = FL_HOLD - 1
                    self.flame_air = True
                    return {"z"}
        except Exception as ex:
            self.sflag.append(repr(ex))
        # hammer flying toward us at body height: short hop
        try:
            for (hcx, hcy, hvx, hvy, hage) in self.hams:
                if hage == 0 or abs(hvy) >= 2.6 or hvx > -0.8:
                    continue
                if not (y0 - 12 <= hcy <= y1 + 2):
                    continue
                dxh = hcx - 6 - x1
                if -10 <= dxh <= HAM_HOP_DX + 2 * (-hvx):
                    self.end_log.append((self.t, "hhop", round(dxh)))
                    self.fh = FL_HOLD - 1
                    self.flame_air = True
                    return {"z"}
        except Exception as ex:
            self.sflag.append(repr(ex))
        # falling hammers: step sideways out of their predicted landing column
        hm = None
        try:
            if not (bs is not None and x1 >= bs[0] - 3):
                right_ok = bs is None or (bs[0] - x1) > DODGE_R_D
                left_ok = x0 > HAM_LEFT_X0 and self.safe_left(x0)
                hm = self.ham_choose(x0, x1, y0, left_ok, right_ok)
        except Exception as ex:
            self.sflag.append(repr(ex))
        if hm is not None and hm[0] > 0.0:
            self.end_log.append((self.t, "ham", round(hm[0], 2), hm[1]))
            if hm[1] == "L":
                return {"Left", "x"}
            if hm[1] == "R":
                return {"Right", "x"}
            if not self.safe_left(x0):
                return {"Right"}
        if x1 >= AXE_X:
            # overshot the axe (landed beyond it at the screen edge): walk back left onto it
            return {"Left"}
        if bs is None or bs[1] <= x1 + 2 or bs[0] <= x0 - 4:
            self.bow_prev = None
            hold = self.reactive(s)
            if hold > 0:
                keys.add("z")
                self.hold = hold - 1
                if self.hold == 0:
                    self.since = 0
            return keys
        return self.bowser_policy(s, bs, x0, x1, keys)
        d = bs[0] - x1
        gb = bs[2]
        vyb = 0 if self.bow_prev is None else gb - self.bow_prev
        self.bow_prev = gb
        self.end_wait += 1
        bow_up = gb <= UNDER_Y
        bow_ground = gb >= END_GROUND_Y - 4
        if self.end_wait > UNDER_MAXWAIT and d <= JMAX + 10 and d >= 12:
            # patience exhausted: commit a jump from the best position
            self.end_log.append((self.t, "force", d, gb))
            keys = set(keys)
            keys.add("z")
            self.hold = JHOLD - 1
            return keys
        sticky = (self.t - self.under_t <= 36) and gb <= UNDER_Y + 6 and self.vx >= 1.5
        if x1 >= bs[0] - 3 or (bow_up and (sticky or (self.vx >= 2.5 and d <= UNDER_RUN and vyb <= 0 and gb >= 122))):
            if not sticky:
                self.under_t = self.t
            self.end_log.append((self.t, "under", d, gb, round(self.vx, 1)))
            return keys
        plat_ok = True
        try:
            sub = s[97:105, 100:236]
            col = (((sub[:, :, 0] == 248) & (sub[:, :, 1] == 56) & (sub[:, :, 2] == 0)) |
                   ((sub[:, :, 0] == 252) & (sub[:, :, 1] == 160) & (sub[:, :, 2] == 68))).sum(axis=0)
            pc = np.nonzero(col >= 4)[0]
            if len(pc) >= 12:
                px1 = int(pc.max()) + 100
                plat_ok = (x0 + 10 * max(self.vx, 2.0)) >= px1 + PLAT_M
                self.plat = px1
        except Exception as ex:
            self.sflag.append(repr(ex))
        if bow_ground and JMIN <= d <= JMAX and x0 >= JX0 and self.vx >= JVX and plat_ok:
            self.end_log.append((self.t, "jover", d, gb, round(self.vx, 1), x0))
            keys = set(keys)
            keys.add("z")
            self.hold = JHOLD - 1
            self.retreat = False
            return keys
        if self.retreat:
            if d >= RETREAT_D or x0 <= RETREAT_X0:
                self.retreat = False
            else:
                self.end_log.append((self.t, "retreat", d, gb))
                return {"Left"} if self.safe_left(x0) else {"Right"}
        if d < JMIN:
            if bow_up and x1 >= bs[0] - 3:
                return keys                      # already beneath him: keep going
            if x0 <= RETREAT_X0:
                return set()                     # podoboo lane behind us: stand
            self.retreat = True
            self.end_log.append((self.t, "back", d, gb, round(self.vx, 1)))
            return {"Left"} if self.safe_left(x0) else {"Right"}
        if bow_up and d <= UNDER_D:
            # Bowser airborne right in front and we are too slow to pass beneath: do not run into him
            self.end_log.append((self.t, "brake", d, gb))
            if x0 <= RETREAT_X0:
                return set()
            return {"Left"} if self.vx > 1.0 else set()
        if bow_ground and x0 < JX0:
            # the platform-safe takeoff column is x0>=JX0: only start the run-up when Bowser is far enough right
            go_bl = max(GO_BL_MIN, GO_BL - self.end_wait // 10)
            if bs[0] < go_bl:
                self.end_log.append((self.t, "wait", d, bs[0], x0))
                return set() if self.safe_left(x0) else {"Right"}
        # approach (run) toward the jump window
        return keys

    def _shift(self, a, b):
        costs = []
        for sh in range(0, 6):
            w = 256 - sh
            costs.append(int((b[130:240, :w] != a[130:240, sh:]).any(-1).sum()))
        best = None
        for sh in range(0, 6):
            key = (costs[sh] + (0 if sh == self.sh_prev else 1e-3), sh)
            if best is None or key < best:
                best = key
        return best[1]

    def observe(self, s):
        s = np.array(s, copy=True)
        sh = 0
        if self.last_img is not None:
            sh = self._shift(self.last_img, s)
        self.sh_prev = sh
        self.cum += sh
        self.last_img = s
        bb = find_mario7(s, self.prev)
        if bb is None and self.prev is not None:
            bb = find_mario7(s, None)
        self.bb = bb
        if bb is None:
            self.lost += 1
            if self.lost >= 3:
                self.prev = None
            return s
        self.lost = 0
        x0, x1, y0, y1 = bb
        self.prev = ((x0 + x1) // 2, (y0 + y1) // 2)
        feet = y1 + 1
        self.last_feet.append(feet)
        if len(self.last_feet) > 4:
            self.last_feet.pop(0)
        sol = _solid(s[min(239, feet): min(240, feet + 3), x0:x1 + 1])
        stable = len(self.last_feet) >= 2 and self.last_feet[-1] == self.last_feet[-2]
        self.grounded = bool(sol.any()) and stable
        nwx = self.cum + (x0 + x1) // 2
        self.vx = 0.6 * self.vx + 0.4 * (nwx - self.wx)
        self.wx = nwx
        if self.wx > self.best:
            self.best = self.wx
            self.stall = 0
        else:
            self.stall += 1
        return s

    def reactive(self, s):
        x0, x1, y0, y1 = self.bb
        feet = y1 + 1
        right = x1 + 2
        sol = _solid(s)
        for dx in range(2, 14, 2):
            cx = right + dx
            if cx >= 256:
                break
            col = sol[min(239, feet): min(240, feet + 3), cx]
            if col.size and not col.any():
                if dx <= PIT_DIST + 2:
                    rise = None
                    for lx in range(right + 4, min(256, right + 100), 2):
                        c2 = sol[max(40, feet - 40):240, lx]
                        if c2.any():
                            rise = feet - (max(40, feet - 40) + int(np.argmax(c2)))
                            break
                    if rise is None:
                        return HOLD_PIT
                    return max(6, _hold_for(max(rise, 0) + 14))
                break
        bx0 = right + 1
        bx1 = min(256, right + 1 + OBS_DIST)
        if bx1 > bx0:
            by0 = max(32, feet - 14)
            by1 = max(by0 + 1, feet - 1)
            box = sol[by0:by1, bx0:bx1]
            if box.sum() >= 4:
                cols = np.nonzero(box.any(axis=0))[0]
                cx = bx0 + int(cols.min())
                h = 0
                for dy in range(1, 90):
                    y = feet - dy
                    if y < 40:
                        break
                    if sol[y, cx:cx + 3].any():
                        h = dy
                    elif h and dy - h > 3:
                        break
                return _hold_for(h)
        return 0

    def gate_step(self, s):
        """before the first pit: brake, stand, and depart when the podoboo has just emerged.
        returns True while still waiting"""
        pods = [p for p in find_pods(s, self.bb) if POD_X0 <= p[0] + self.cum <= POD_X1]
        vis = bool(pods)
        if vis and not self.pod_vis:
            self.pod_emerge = self.t
            self.g_log.append(("em", self.t, pods[:2]))
        self.pod_vis = vis
        if self.g_stage == "brake":
            if self.vx < 0.25 or self.t - self.g_t0 > 60:
                self.g_stage = "back"
                self.g_log.append(("stopped", self.t, self.wx, round(self.vx, 2)))
            self.g_keys = {"Left"}
            return True
        if self.g_stage == "back":
            if self.wx <= BACK_X and abs(self.vx) < 0.25:
                self.g_stage = "wait"
                self.g_log.append(("back", self.t, self.wx))
            self.g_keys = {"Left"} if self.wx > BACK_X else set()
            return True
        self.g_keys = set()
        since = self.t - self.pod_emerge
        if vis and EM_LO <= since <= EM_HI:
            self.g_log.append(("go", self.t, since, self.wx))
            return False
        if self.t - self.g_t0 > GATE_MAX:
            self.g_log.append(("timeout", self.t))
            return False
        return True

    def corr_flame(self, h):
        """a low horizontal flame closing on Mario's body: hop straight up (no direction keys) so it passes beneath"""
        x0, x1, y0, y1 = h.bb
        v = h.vx()
        for b in h.balls:
            if b[6] is not None or abs(b[4]) < 1.0 or abs(b[5]) >= 1.0 or b[4] > 0:
                continue
            cy = (b[2] + b[3]) / 2.0
            if not (y0 - 4 <= cy <= y1 + 6):
                continue
            dx = b[0] - (h.cum + x1)
            if dx < -14:
                continue
            closing = -b[4] + max(v, 0.0)
            if closing < 0.8:
                continue
            if max(dx, 0.0) / closing <= FLC_TTC:
                self.end_log.append((self.t, "cflame", round(dx), round(cy), round(v, 1)))
                return True
        return False

    def c7_act(self, s):
        h = self.h
        keys = {"Right", "x"}
        if h.bb is None:
            if h.hold > 0:
                keys.add("z")
                h.hold -= 1
            return keys
        airborne = h.hold > 0 or (not h.grounded and 0 <= h.t - h.jt0 <= 70)
        if airborne and AIR_ON:
            z = h.hold > 0
            if z:
                h.hold -= 1
                if h.hold == 0:
                    h.since = 0
            try:
                dk = h.air_decide(s)
            except Exception as ex:
                self.sflag.append(repr(ex))
                dk = {"Right", "x"}
            keys = set(dk)
            if z:
                keys.add("z")
            return keys
        if h.hold > 0:
            keys.add("z")
            h.hold -= 1
            if h.hold == 0:
                h.since = 0
            return keys
        h.since += 1
        if not (h.grounded and h.since > 1):
            return keys
        try:
            if FLC_ON and self.corr_flame(h):
                keys, hold = set(), FLC_HOLD
            else:
                keys, hold = h.decide2(s)
        except Exception as ex:
            self.sflag.append(repr(ex))
            return {"Right", "x"}
        if hold > 0:
            keys = set(keys)
            keys.add("z")
            h.hold = hold - 1
            h.jt0 = h.t
            h.jK = hold
            h.jfeet = h.bb[3] + 1
            if h.hold == 0:
                h.since = 0
        return keys

    def act(self, s):
        s = self.observe(s)
        self.t += 1
        try:
            self.h.observe(s)
        except Exception as ex:
            self.sflag.append(repr(ex))
        keys = {"Right", "x"}
        if self.bb is None:
            if self.hold > 0:
                keys.add("z")
                self.hold -= 1
            return keys
        if self.gating:
            if self.gate_step(s):
                self.stall = 0
                return set(self.g_keys)
            self.gating = False
            self.gate_done = True
            return keys
        if (GATE_ON and not self.gate_done and self.route is not None and self.ri < len(self.route)
                and self.route[self.ri][0] == "w" and GATE_LO <= self.route[self.ri][1] <= GATE_HI
                and self.wx >= PRE_X):
            self.gating = True
            self.g_stage = "brake"
            self.g_t0 = self.t
            self.g_keys = {"Left"}
            self.pod_vis = False
            self.pod_emerge = -999
            self.g_log.append(("pre", self.t, self.wx, round(self.vx, 2)))
            return {"Left"}
        if self.route is not None and self.ri < len(self.route) and self.route[self.ri][0] == "w" \
                and self.route[self.ri][1] <= self.wx and GATE_LO <= self.route[self.ri][1] <= GATE_HI:
            self.g_log.append(("trig", self.t, self.wx, round(self.vx, 2)))
        if self.pause_left > 0:
            self.pause_left -= 1
            self.stall = 0
            return set()
        if self.hold > 0:
            keys.add("z")
            self.hold -= 1
            if self.hold == 0:
                self.since = 0
            return keys
        if self.wx > END_X and (self.end_mode or self.bowser_span(s) is not None):
            if not self.end_mode:
                self.end_log.append((self.t, "end", round(self.wx, 1)))
            self.end_mode = True
            self.since += 1
            return self.end_act(s)
        if CORR_ON and self.gate_done and CORR_LO <= self.wx < CORR_HI:
            self.in_corr = True
            return self.c7_act(s)
        if self.in_corr:
            self.in_corr = False
            self.left_corr = True
            if self.after_route is not None:
                self.route = list(self.after_route)
                self.ri = 0
            elif self.route is not None:
                while self.ri < len(self.route) and self.route[self.ri][1] < self.wx + 2:
                    self.ri += 1
            self.g_log.append(("leave", self.t, round(self.wx, 1), self.ri))
        self.since += 1
        if not (self.grounded and self.since > 1):
            return keys
        hold = 0
        if self.route is not None and self.ri < len(self.route):
            while self.ri < len(self.route) and self.route[self.ri][1] < self.wx - 60:
                self.ri += 1
            fire = False
            ev = None
            if self.ri < len(self.route):
                ev = self.route[self.ri]
                if ev[1] <= self.wx and (len(ev) < 4 or self.trig_t is None or self.t - self.trig_t >= ev[3]):
                    fire = True
            if fire:
                self.ri += 1
                if ev[0] == "w":
                    if GATE_LO <= ev[1] <= GATE_HI and self.trig_t is None:
                        self.trig_t = self.t
                    self.pause_left = int(ev[2])
                    return set()
                hold = int(ev[2])
            if self.stall > 70:
                self.route = None
        else:
            hold = self.reactive(s)
        if hold > 0:
            keys.add("z")
            self.hold = hold - 1
            if self.hold == 0:
                self.since = 0
        return keys


RING = []
SLOG = []


def solve(sim):
    return []


def play(sim):
    route = [tuple(x) for x in ROUTE]
    ctl = Ctl(route)
    ctl.after_route = []
    try:
        bb0 = find_mario7(sim.screen(), None)
    except Exception:
        bb0 = None
    f0 = 0
    if bb0 is not None and not (36 <= bb0[0] <= 48):
        f0 = 100
    if f0 > 60:
        # practice mode (checkpoint): world x unknown; assume late corridor
        ctl.cum = 1900
        ctl.h.cum = 1900
        ctl.gate_done = True
        ctl.route = None
        print("practice start f0", f0)
    frozen = 0
    last = None
    step = 0
    try:
        for step in range(7900):
            s = sim.screen()
            if last is not None and np.array_equal(s[40:], last[40:]):
                frozen += 1
            else:
                frozen = 0
            last = s
            if frozen > 90:
                break
            try:
                keys = ctl.act(s)
            except Exception as ex:
                print("act exception", repr(ex))
                keys = {"Right", "x"}
            if step % 240 == 0:
                print("f", step, "wx", ctl.wx, "bb", ctl.bb, "g", ctl.grounded, "ri", ctl.ri, "gating", ctl.gating,
                      "mode", ctl.h.mode, "pocket", ctl.h.pocket, "route" if ctl.route is not None else "reactive")
            try:
                hh = ctl.h
                if frozen < 2:
                    RING.append((step, round(ctl.wx), ctl.bb, hh.mode, None if hh.dbg is None else hh.dbg[:3],
                                 sorted(keys), round(hh.vx(), 2),
                                 [(round((b[0] + b[1]) / 2), round((b[2] + b[3]) / 2), round(b[4], 1))
                                  for b in hh.balls if b[6] is None][:3], getattr(hh, "hopres", None)))
                if len(RING) > 60:
                    RING.pop(0)
                if step % 12 == 0 and 1100 < ctl.wx < 1520 and ctl.stall > 30:
                    pv = {}
                    for b in hh.balls:
                        if b[6] is not None and b[8] not in pv:
                            cx = (b[0] + b[1]) / 2.0
                            cy = (b[2] + b[3]) / 2.0
                            pv[b[8]] = (round(b[6][0]), round(b[6][1]), round(float(np.degrees(np.arctan2(cy - b[6][1], cx - b[6][0])))),
                                        round(b[7] or 0, 3), b[8] in hh.piv_long)
                    SLOG.append((step, round(ctl.wx), hh.mode, None if hh.dbg is None else (hh.dbg[0], hh.dbg[1], hh.dbg[4]), ctl.stall,
                                 list(pv.values())))
                    if len(SLOG) > 40:
                        SLOG.pop(0)
            except Exception:
                pass
            sim.step(keys)
    finally:
        print("end step", step, "wx", ctl.wx, "frozen", frozen, "flags", ctl.sflag[:3])
        print("gate log", [e for e in ctl.g_log if e[0] in ("go", "trig", "leave", "timeout", "pre")][:12])
        print("end log", ctl.end_log[-6:])
        for r in SLOG[-18:]:
            print("S", r)
        for r in RING[-6:]:
            print("R", r[:2], r[3], r[4], r[5], r[6])

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    st = ctx.get("stuck") if isinstance(ctx, dict) else None\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": bool(st and st.get("is"))},\n            "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}'}

