# ==================================================================================================
# Super Mario Bros. (NES) -- level 2-2  |  超级马里奥兄弟（NES） 关 2-2
# Pass rate on 100 random variants never used before: 40/100; fastest finish 2265 frames (37.75 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 40/100；最快一局 2265 帧（60 帧/秒，约 37.75 秒）。
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
#   closed-loop underwater swimmer (pixel perception + per-frame MPC over tap plans + viability-ma
#   p goal cost); enemy-hit cost discounted with look-ahead time, stronger mid-depth preference
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def eq                             line 55    
#   def terrain_mask                   line 65    
#   def end_signature                  line 74    final screen: wall top-right, ledge row, bricks right of the pipe, open water on the left.
#   def find_mario                     line 84    
#   def clusters                       line 118   
#   def viability                      line 156   D[c, r] = distance (in 4px rows) from cell to the nearest row from which a rightward
#   class Agent                        line 188   
#   def play                           line 577   
#   def solve                          line 610   
# ==================================================================================================

# idea: closed-loop underwater swimmer (pixel perception + per-frame MPC over tap plans + viability-map goal cost); enemy-hit cost discounted with look-ahead time, stronger mid-depth preference
import time
import numpy as np

H, W = 240, 256
G = 0.055
V_TAP = -1.66
VMAX = 5.2
SEG = 8
NSEG = 10
HOR = SEG * NSEG
MW, MH = 12.0, 14.0
CY_MIN, CY_MAX = 36.0, 200.5
NP = 1 << NSEG
NC, NR = 64, 42

# tunables (module level so that probes can override them)
DISC = 30.0        # time constant of enemy-cost discount (frames)
DISC_MIN = 1.0
CENTER_W = 0.02
TOP_W = 0.12
BOT_W = 0.06
PH = 1.0           # blooper horizontal pursuit speed assumed (px/frame)
PV = 0.9           # blooper vertical pursuit speed assumed
BM = 3.0           # extra blooper margin
LANE_W = 0.3
LANE_X0 = 1300.0
LANE_X1 = 1800.0
LANE_LO = 95.0
LANE_HI = 165.0
VW_SUM = 0.25
VW_END = 2.0
ESC_W = 0.25
STALL_ON = False
STALL_PEN = 10.0

PL = ((np.arange(NP)[:, None] >> np.arange(NSEG)[None, :]) & 1).astype(bool)
TAPSTEP = np.zeros((NP, HOR), bool)
for _j in range(HOR):
    TAPSTEP[:, _j] = PL[:, _j // SEG] & (_j % 2 == 0)
TT = np.arange(1, HOR + 1, dtype=np.float64)          # time of step j (frames ahead)
MARG = 1.5 + 0.10 * TT                                  # enemy uncertainty margin

# viability grid geometry
_cc = np.arange(NC)
_rr = np.arange(NR)
VX = _cc * 4 + 2
VY = 40 + _rr * 4 + 2
VXL = np.clip(VX - 8, 0, W)
VXR = np.clip(VX + 8, 0, W)
VYT = np.clip(VY - 9, 0, H)
VYB = np.clip(VY + 9, 0, H)


def eq(sc, c):
    return (sc[:, :, 0] == c[0]) & (sc[:, :, 1] == c[1]) & (sc[:, :, 2] == c[2])


RED = (248, 56, 0)
BRN = (172, 124, 0)
SKN = (252, 160, 68)
WATER = (0, 88, 248)


def terrain_mask(sc, coral=True):
    T = eq(sc, (0, 168, 0)) | eq(sc, (184, 248, 184))
    if coral:
        # seaweed is solid in this game (Mario gets stuck on it)
        T = T | eq(sc, (248, 88, 152)) | eq(sc, (228, 0, 88))
    T[:41] = False
    return T


def end_signature(T):
    """final screen: wall top-right, ledge row, bricks right of the pipe, open water on the left."""
    a = T[44:92, 205:250].mean()
    b = T[148:158, 192:250].mean()
    c = T[100:140, 222:250].mean()
    d = T[112:140, 200:214].mean()
    e = T[100:140, 150:180].mean()
    return (a > 0.6 and b > 0.6 and c > 0.5 and d > 0.4 and e < 0.08)


def find_mario(sc, pred):
    red = eq(sc, RED)
    red[:33] = False
    cand = None
    if pred is not None:
        x0 = max(0, int(pred[0]) - 18); x1 = min(W, int(pred[0]) + 19)
        y0 = max(33, int(pred[1]) - 18); y1 = min(H, int(pred[1]) + 19)
        ys, xs = np.nonzero(red[y0:y1, x0:x1])
        if len(ys):
            xs = xs + x0; ys = ys + y0
            cand = (xs, ys)
    if cand is None:
        ys, xs = np.nonzero(red)
        if len(ys) == 0:
            return None
        cand = (xs, ys)
        if pred is None:
            pred = (48.0, 75.0)
    xs, ys = cand
    d = (xs - pred[0]) ** 2 + (ys - pred[1]) ** 2
    k = int(np.argmin(d))
    cx, cy = int(xs[k]), int(ys[k])
    sel = (np.abs(xs - cx) <= 9) & (np.abs(ys - cy) <= 9)
    rcx = float(xs[sel].mean()); rcy = float(ys[sel].mean())
    a0 = max(0, int(rcx) - 9); a1 = min(W, int(rcx) + 10)
    b0 = max(33, int(rcy) - 10); b1 = min(H, int(rcy) + 11)
    sub = sc[b0:b1, a0:a1]
    m = eq(sub, RED) | eq(sub, BRN) | eq(sub, SKN)
    yy, xx = np.nonzero(m)
    bx0 = int(xx.min()) + a0; bx1 = int(xx.max()) + a0
    by0 = int(yy.min()) + b0; by1 = int(yy.max()) + b0
    return ((bx0 + bx1) / 2.0, (by0 + by1) / 2.0, bx0, bx1, by0, by1)


def clusters(mask):
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return []
    ids = np.unique((ys >> 2) * 64 + (xs >> 2)).tolist()
    cells = set(ids)
    seen = set()
    out = []
    for c in ids:
        if c in seen:
            continue
        stack = [c]
        seen.add(c)
        comp = []
        while stack:
            u = stack.pop()
            comp.append(u)
            uy, ux = divmod(u, 64)
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if ux + dx < 0 or ux + dx > 63:
                        continue
                    v = (uy + dy) * 64 + ux + dx
                    if v in cells and v not in seen:
                        seen.add(v)
                        stack.append(v)
        cy_ = [u // 64 for u in comp]; cx_ = [u % 64 for u in comp]
        X0, X1 = min(cx_) * 4, max(cx_) * 4 + 4
        Y0, Y1 = min(cy_) * 4, max(cy_) * 4 + 4
        sub = mask[Y0:Y1, X0:X1]
        n = int(sub.sum())
        if n < 14:
            continue
        yy, xx = np.nonzero(sub)
        out.append((int(xx.min()) + X0, int(yy.min()) + Y0, int(xx.max()) + X0, int(yy.max()) + Y0, n))
    return out


def viability(S):
    """D[c, r] = distance (in 4px rows) from cell to the nearest row from which a rightward
    swim path through the visible terrain exists. S is the integral image of terrain."""
    s = (S[VYB[None, :], VXR[:, None]] - S[VYT[None, :], VXR[:, None]]
         - S[VYB[None, :], VXL[:, None]] + S[VYT[None, :], VXL[:, None]])
    free = (s == 0)
    anyfree = free.any(axis=1)
    nz = np.nonzero(anyfree)[0]
    if len(nz) == 0:
        return np.zeros((NC, NR))
    c0 = int(nz[-1])
    viable = np.zeros((NC, NR), bool)
    viable[c0] = free[c0]
    for c in range(c0 - 1, -1, -1):
        nxt = viable[c + 1]
        d = nxt.copy()
        d[:-1] |= nxt[1:]                 # sink one cell
        if c % 4 != 0:
            d[1:] |= nxt[:-1]             # rise one cell (3 of 4 columns)
        if c % 2 == 1:
            d[:-2] |= nxt[2:]             # sink two cells (every other column)
        viable[c] = free[c] & d
    idx = np.where(viable, _rr[None, :], -1000)
    left = np.maximum.accumulate(idx, axis=1)
    idx2 = np.where(viable, _rr[None, :], 1000)
    right = np.minimum.accumulate(idx2[:, ::-1], axis=1)[:, ::-1]
    D = np.minimum(_rr[None, :] - left, right - _rr[None, :]).astype(np.float64)
    D[~viable.any(axis=1)] = 0.0
    D = np.minimum(D, 40.0)
    return D


class Agent:
    def __init__(self):
        self.pred = (48.0, 75.0)
        self.cy_prev = None
        self.cx_prev = None
        self.vy = 0.0
        self.last_tap = False
        self.vm = 1.0
        self.scE = 0.0
        self.prevNW = None
        self.sc_int = 0
        self.tracks = []
        self.frame = 0
        self.lost = 0
        self.mbox = None
        self.info = ""
        self.nosafe = 0
        self.tcomp = 0.0
        self.since_tap = 9
        self.nostroll = 0
        self.nexc = 0
        self.end_mode = False
        self.end_stall = 0
        self.sig_count = 0
        self.ground_mode = False
        self.sky_count = 0
        self.gframe = 0
        self.stuck = 0
        self.esc_target = None
        self.esc_scroll = 0
        self.esc_age = 0
        self.nesc = 0
        self.adv = True
        self.scx = 0
        self.xw = 0.0
        self.stall_age = 99
        self.nstall = 0

    # ---- scroll estimate ----
    def scroll(self, sc):
        nw = (np.abs(sc[42:206].astype(np.int16) - np.array(WATER, np.int16)).sum(axis=2) > 40)
        if self.prevNW is not None and nw.sum() > 120:
            p = self.prevNW
            errs = []
            for s in range(0, 4):
                if s == 0:
                    errs.append(int((p != nw).sum()))
                else:
                    errs.append(int((p[:, s:] != nw[:, :W - s]).sum()))
            order = np.argsort(errs)
            b = int(order[0])
            if errs[int(order[1])] - errs[b] > 5:
                self.sc_int = b
            # else keep previous
        self.prevNW = nw
        self.scE = 0.85 * self.scE + 0.15 * self.sc_int

    # ---- enemies ----
    def enemies(self, sc, mb):
        m = eq(sc, (252, 252, 252)) | eq(sc, (188, 188, 188)) | eq(sc, RED)
        m[:41] = False
        m[208:] = False
        if mb is not None:
            m[max(0, mb[2] - 3):mb[3] + 4, max(0, mb[0] - 3):mb[1] + 4] = False
        cl = clusters(m)
        new = []
        used = set()
        for (x0, y0, x1, y1, n) in cl:
            cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
            best = None; bd = 1e9
            for i, t in enumerate(self.tracks):
                if i in used:
                    continue
                d = abs(t["cx"] - cx) + abs(t["cy"] - cy)
                if d < bd:
                    bd = d; best = i
            if best is not None and bd < 20:
                used.add(best)
                t = self.tracks[best]
                a = 0.6 if t["age"] < 3 else 0.3
                vx = (1 - a) * t["vx"] + a * (cx - t["cx"])
                vy = (1 - a) * t["vy"] + a * (cy - t["cy"])
                age = t["age"] + 1
            else:
                vx, vy, age = -(self.scE + 1.2), 0.0, 0
            new.append({"cx": cx, "cy": cy, "w": x1 - x0 + 1, "h": y1 - y0 + 1, "vx": vx, "vy": vy, "age": age})
        self.tracks = new
        return new

    # ---- planning ----
    def plan(self, sc, cx, cy, enemies, tg=None):
        t0 = time.time()
        T = terrain_mask(sc)
        S = np.zeros((H + 1, W + 1), np.int32)
        S[1:, 1:] = np.cumsum(np.cumsum(T, axis=0, dtype=np.int32), axis=1)
        D = viability(S)
        tap = TAPSTEP.copy()
        if self.last_tap:
            tap[:, 0] = False
        y = np.full(NP, cy)
        vy = np.full(NP, self.vy)
        Y = np.empty((HOR, NP))
        for j in range(HOR):
            vy = np.where(tap[:, j], V_TAP, np.minimum(vy + G, VMAX))
            y = y + vy
            lo = y < CY_MIN
            hi = y > CY_MAX
            y = np.where(lo, CY_MIN, np.where(hi, CY_MAX, y))
            vy = np.where(lo | hi, 0.0, vy)
            Y[j] = y
        Y = Y[1::2]
        TTs = TT[1::2]
        MARGs = MARG[1::2]
        WT = (DISC_MIN + (1.0 - DISC_MIN) * np.exp(-(TTs - 1.0) / DISC))[:, None]
        # horizontal rollout with blocking: Mario advances unless a wall is in the way
        ns = Y.shape[0]
        v0 = min(1.75, max(0.3, self.vm))
        Xs = np.empty((ns, NP))
        Xc = np.full(NP, float(cx))
        hwb = MW / 2 - 1.0
        hhb = MH / 2 - 1.0
        nblk = np.zeros(NP)
        for s in range(ns):
            v = min(1.75, v0 + 0.12 * 2 * s)
            Xn = Xc + 2.0 * v
            xl = np.clip(np.round(Xn - hwb), 0, W).astype(np.int64)
            xr = np.clip(np.round(Xn + hwb), 0, W).astype(np.int64)
            yt = np.clip(np.round(Y[s] - hhb), 0, H).astype(np.int64)
            yb = np.clip(np.round(Y[s] + hhb), 0, H).astype(np.int64)
            sm = S[yb, xr] - S[yt, xr] - S[yb, xl] + S[yt, xl]
            blk = sm > 0
            Xc = np.where(blk, Xc, Xn)
            nblk += blk
            Xs[s] = Xc
        ev = self.scE

        def evalX(X, stall):
            cost = np.zeros(NP)
            cost += 0.5 * (1.75 * TTs[-1] - (X[-1] - cx))
            if stall:
                cost += STALL_PEN
            return self.evalcost(cost, X, Y, S, D, enemies, TTs, MARGs, WT, ev)

        c_adv = evalX(Xs, False)
        if STALL_ON and tg is None and self.frame > 100:
            tt = 2.0 * (np.arange(ns) + 1.0)
            tmax = v0 / 0.08
            tc = np.minimum(tt, tmax)
            Xst = np.repeat((cx + v0 * tc - 0.04 * tc * tc)[:, None], NP, axis=1)
            c_st = evalX(Xst, True)
        else:
            c_st = None
        if tg is not None:
            c_adv = c_adv + ESC_W * np.abs(Y - tg).sum(axis=0)
        if c_st is not None and float(c_st.min()) < float(c_adv.min()):
            k = int(np.argmin(c_st))
            self.adv = False
            cbest = float(c_st[k])
        else:
            k = int(np.argmin(c_adv))
            self.adv = True
            cbest = float(c_adv[k])
        self.tcomp += time.time() - t0
        return bool(tap[k, 0]), cbest, k, Y[:, k]

    def evalcost(self, cost, X, Y, S, D, enemies, TTs, MARGs, WT, ev):
        for (g, wgt) in ((-1.0, 80.0), (3.0, 6.0), (9.0, 1.2)):
            hw = MW / 2 + g
            hh = MH / 2 + g
            xl = np.clip(np.round(X - hw), 0, W).astype(np.int64)
            xr = np.clip(np.round(X + hw), 0, W).astype(np.int64)
            yt = np.clip(np.round(Y - hh), 0, H).astype(np.int64)
            yb = np.clip(np.round(Y + hh), 0, H).astype(np.int64)
            s = S[yb, xr] - S[yt, xr] - S[yb, xl] + S[yt, xl]
            cost += wgt * (s > 0).sum(axis=0)
        # goal-directed viability cost
        cols = np.clip((X // 4).astype(np.int64), 0, NC - 1)
        rows = np.clip(((Y - 40.0) // 4).astype(np.int64), 0, NR - 1)
        dv = D[cols, rows]
        cost += VW_SUM * dv.sum(axis=0) + VW_END * dv[-1]
        # enemies
        for e in enemies:
            T2 = TTs[:, None]
            lin = e["cx"] + (e["vx"] + ev) * T2
            if e["h"] >= 18:
                # blooper: chases Mario -> blend of linear drift and pursuit of this plan's own track
                ex = e["cx"] + 0.5 * (e["vx"] + ev) * T2 + 0.5 * np.clip(X - e["cx"], -PH * T2, PH * T2)
                dxa = np.abs(ex - X) - (e["w"] + MW) / 2.0
                if dxa.min() > 45:
                    continue
                ey = (e["cy"] + 0.3 * e["vy"] * T2) + 0.7 * np.clip(Y - e["cy"], -PV * T2, PV * T2)
                dy = np.abs(ey - Y) - (e["h"] + MH) / 2.0
                c = np.maximum(dxa, dy) - (MARGs[:, None] + BM)
            else:
                dxa = np.abs(lin - X) - (e["w"] + MW) / 2.0
                if dxa.min() > 40:
                    continue
                ey = e["cy"] + e["vy"] * T2
                dy = np.abs(ey - Y) - (e["h"] + MH) / 2.0
                c = np.maximum(dxa, dy) - MARGs[:, None]
            hit = c < 0
            cost += 400.0 * (hit * WT).sum(axis=0)
            cost += 25.0 * (np.exp(-np.maximum(c, 0) / 9.0) * WT).sum(axis=0)
        if LANE_W > 0:
            xwt = self.xw + (X - X[0:1] * 0 - self.cx_prev)
            msk = (xwt >= LANE_X0) & (xwt <= LANE_X1)
            if msk.any():
                out = np.maximum(0.0, LANE_LO - Y) + np.maximum(0.0, Y - LANE_HI)
                cost += LANE_W * (msk * out).sum(axis=0)
        cost += CENTER_W * np.abs(Y - 115.0).sum(axis=0)
        cost += TOP_W * np.maximum(0.0, 80.0 - Y).sum(axis=0)       # avoid being pinned at the surface
        cost += BOT_W * np.maximum(0.0, Y - 165.0).sum(axis=0)
        return cost

    def find_escape(self, sc, bx0, bx1, by0, by1, cy):
        """Mario is pressed against a wall on his right: pick an altitude that clears it."""
        T = terrain_mask(sc)
        x0 = min(W - 1, bx1 + 1)
        x1 = min(W, bx1 + 6)
        A = T[:, x0:x1].any(axis=1)
        lo = max(41, by0 - 1)
        hi = min(H - 1, by1 + 2)
        rows = np.nonzero(A[lo:hi + 1])[0]
        if len(rows) == 0:
            return None
        r0 = int(rows[0]) + lo
        wt = r0
        while wt - 1 >= 41 and A[wt - 1]:
            wt -= 1
        wb = int(rows[-1]) + lo
        while wb + 1 < 208 and A[wb + 1]:
            wb += 1
        up_t = wt - 10.0
        dn_t = wb + 10.0
        up_ok = up_t >= 40.5
        dn_ok = dn_t <= 196.0
        if not up_ok and not dn_ok:
            return None
        up_c = 1.4 * (cy - up_t) if up_ok else 1e9
        dn_c = 1.0 * (dn_t - cy) if dn_ok else 1e9
        # mid-air obstacles: if the wall hangs from the top there is only one way
        if up_c < dn_c:
            return max(41.0, up_t)
        return dn_t

    def act(self, sc):
        try:
            return self._act(sc)
        except Exception as e:  # never crash the policy: fall back to a crude altitude hold
            self.nexc = getattr(self, "nexc", 0) + 1
            if self.nexc < 5:
                print("EXC in act:", repr(e))
            keys = {"Right", "x"}
            y = self.cy_prev if self.cy_prev is not None else 100.0
            tap = (y + 8 * self.vy > 100) and not self.last_tap
            self.last_tap = tap
            if tap:
                keys.add("z")
            return keys

    def _act(self, sc):
        self.frame += 1
        # ---- after the pipe: overworld stairs + flagpole -> run right and spam jumps ----
        if self.end_mode and not self.ground_mode:
            if eq(sc[44:200], (104, 136, 252)).mean() > 0.3:
                self.sky_count += 1
            else:
                self.sky_count = 0
            if self.sky_count >= 3:
                self.ground_mode = True
                print("GROUND MODE at frame", self.frame)
        if self.ground_mode:
            self.gframe += 1
            keys = {"Right", "x"}
            if self.gframe % 24 < 20:
                keys.add("z")
            self.info = "f%d GROUND" % self.frame
            return keys
        mar = find_mario(sc, self.pred)
        self.scroll(sc)
        self.scx += self.sc_int
        keys = {"Right", "x"}
        if mar is None:
            self.lost += 1
            self.enemies(sc, None)
            if self.end_mode:
                self.last_tap = False
                return keys
            tap = (self.frame % 6 == 0)
            self.last_tap = tap
            if tap:
                keys.add("z")
            return keys
        self.lost = 0
        cx, cy, bx0, bx1, by0, by1 = mar
        self.mbox = (bx0, bx1, by0, by1)
        if self.cy_prev is not None:
            dy = cy - self.cy_prev
            vm_ = V_TAP if self.last_tap else min(self.vy + G, VMAX)
            self.vy = 0.5 * vm_ + 0.5 * dy
            dx = cx - self.cx_prev
            self.vm = 0.92 * self.vm + 0.08 * (dx + self.sc_int)
        self.cy_prev = cy
        self.cx_prev = cx
        self.xw = self.scx + cx - 8.0
        self.pred = (cx, cy + self.vy)

        # ---- end phase detection ----
        if not self.end_mode and self.frame > 1200:
            T = terrain_mask(sc, False)
            if end_signature(T):
                self.sig_count += 1
            else:
                self.sig_count = 0
            if self.sc_int == 0 and cx >= 150:
                self.end_stall += 1
            else:
                self.end_stall = 0
            if self.sig_count >= 3 or self.end_stall >= 25:
                self.end_mode = True
                print("END MODE at frame", self.frame, "sig", self.sig_count, "stall", self.end_stall, "m=(%.0f,%.0f)" % (cx, cy))
        if self.end_mode:
            tap = False
            if cx < 188 and cy + 10 * self.vy > 124 and not self.last_tap:
                tap = True
            self.last_tap = tap
            if tap:
                keys.add("z")
            self.info = "f%d END m=(%.0f,%.0f) vy=%.2f" % (self.frame, cx, cy, self.vy)
            return keys

        en = self.enemies(sc, self.mbox)

        # ---- wall escape: stalled against terrain -> go around it (over/under) ----
        if self.sc_int == 0 and cx >= 108 and self.frame > 60 and self.stall_age > 25:
            self.stuck += 1
        else:
            self.stuck = 0
        if self.esc_target is not None:
            self.esc_scroll += self.sc_int
            self.esc_age += 1
            if self.esc_scroll >= 28 or self.esc_age > 90:
                self.esc_target = None
        if self.esc_target is None and self.stuck >= 4:
            tg = self.find_escape(sc, bx0, bx1, by0, by1, cy)
            if tg is not None:
                self.esc_target = tg
                self.esc_scroll = 0
                self.esc_age = 0
                self.nesc += 1
        if self.esc_target is not None:
            tg = self.esc_target
            tap, cst, k, path = self.plan(sc, cx, cy, en, tg=tg)
            self.last_tap = tap
            if tap:
                keys.add("z")
            self.info = "f%d ESC tg=%.0f m=(%.0f,%.0f) vy=%.2f nesc=%d" % (self.frame, tg, cx, cy, self.vy, self.nesc)
            return keys

        tap, cst, k, path = self.plan(sc, cx, cy, en)
        if cst >= 400:
            self.nosafe += 1
        if not self.adv:
            self.stall_age = 0
            self.nstall += 1
            keys = set()
        else:
            self.stall_age += 1
        if self.sc_int == 0 and cx >= 108 and self.stall_age > 25:
            self.nostroll += 1
        else:
            self.nostroll = 0
        if self.nostroll > 160:
            ph = (self.nostroll - 160) % 140
            if ph < 70:
                tap = (not self.last_tap)
            else:
                tap = False
            if self.nostroll % 140 == 0:
                print("ESCAPE mode", self.info)
        self.last_tap = tap
        if tap:
            keys.add("z")
        self.info = "f%d m=(%.0f,%.0f) vy=%.2f vm=%.2f sc=%.2f en=%d cost=%.0f nosafe=%d nstall=%d tplan=%.1fms" % (
            self.frame, cx, cy, self.vy, self.vm, self.scE, len(en), cst, self.nosafe, self.nstall,
            1000 * self.tcomp / max(1, self.frame))
        return keys


def play(sim):
    ag = Agent()
    n = 0
    prev = None
    same = 0
    t0 = time.time()
    while n < 7900:
        try:
            if sim.budget_left <= 0:
                return
            sc = sim.screen()
        except Exception:
            return
        if prev is not None and np.array_equal(sc, prev):
            same += 1
            if same > 400 and not ag.end_mode:
                print("frozen end at", n, ag.info)
                return
        else:
            same = 0
        prev = sc
        keys = ag.act(sc)
        if n % 100 == 0:
            print(ag.info, "wall=%.1fs" % (time.time() - t0))
        try:
            sim.step(keys)
        except Exception:
            print("step exception at", n, ag.info)
            return
        n += 1
    print("play end", n, ag.info)


def solve(sim):
    sim.reset()
    return []

