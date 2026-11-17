# ==================================================================================================
# Super Mario Bros. (NES) -- level 3-4  |  超级马里奥兄弟（NES） 关 3-4
# Pass rate on 100 random variants never used before: 45/100; fastest finish 2416 frames (40.27 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 45/100；最快一局 2416 帧（60 帧/秒，约 40.27 秒）。
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
#   r13_c0 chain (L0, learned firebar gates L1, hardened pillar hopper P, Bowser L4) with much lon
#   ger patience (no early forced jumps), taller hold for high ledges, and reachability-gated hops
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   class BudgetExceededLeg            line 6     
#   def _mod                           line 10    
#   def shift_of                       line 16    
#   def mario_left                     line 26    
#   class Track                        line 35    wraps the real sim: tracks world scroll, raises BudgetExceededLeg at the leg boundary
#   def mods                           line 1977  
#   def run_leg                        line 1986  
#   def run_ped                        line 2001  
#   def chain                          line 2017  
#   def learn_here                     line 2031  
#   def solve                          line 2057  
#   def play                           line 2116  
#
# This policy also reads sim.memory; that data is in 3-4.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 3-4.memory.json。
# ==================================================================================================

# idea: r13_c0 chain (L0, learned firebar gates L1, hardened pillar hopper P, Bowser L4) with much longer patience (no early forced jumps), taller hold for high ledges, and reachability-gated hops
import types
import numpy as np


class BudgetExceededLeg(Exception):
    pass


def _mod(name, src):
    m = types.ModuleType(name)
    exec(compile(src, name, "exec"), m.__dict__)
    return m


def shift_of(ga, gb):
    W = ga.shape[1]
    best, bk = None, 0
    for k in range(0, 7):
        mm = int((gb[:, :W - k] != ga[:, k:]).sum())
        if best is None or mm < best - 8:
            best, bk = mm, k
    return bk


def mario_left(s):
    m = (s[:, :, 0] == 172) & (s[:, :, 1] == 124) & (s[:, :, 2] == 0)
    m[:40] = False
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None, None
    return int(xs.min()), int(ys.max())


class Track:
    """wraps the real sim: tracks world scroll, raises BudgetExceededLeg at the leg boundary"""

    def __init__(self, sim, rec=None):
        self.sim = sim
        self.KEYS = sim.KEYS
        self.W = 0
        self.cur = sim.screen()
        self.prev_gray = self.cur[40:232].astype(np.int16).sum(axis=2)
        self.mxl, self.fy = mario_left(self.cur)
        self.wx = self.mxl if self.mxl is not None else 0
        self.still = 0
        self.limit = None
        self.mem = None
        self.handoff = False
        self.skip = False
        self.slack = 60
        self.mode = "ground"
        self.xst = 0
        self.lastwx = self.wx
        self.base_frame = 0
        self.nsteps = 0
        self.rec = rec
        self.hold_ok = 0

    @property
    def memory(self):
        return self.mem if self.mem is not None else self.sim.memory

    def resync(self, W):
        self.W = W
        self.cur = self.sim.screen()
        self.prev_gray = self.cur[40:232].astype(np.int16).sum(axis=2)
        ml, fy = mario_left(self.cur)
        if ml is not None:
            self.mxl, self.fy = ml, fy
            self.wx = W + ml
        self.lastwx = self.wx
        self.still = 0
        self.xst = 0

    @property
    def budget_left(self):
        return self.sim.budget_left

    @property
    def frame(self):
        return self.nsteps - self.base_frame

    def screen(self):
        return self.cur

    def save(self, slot):
        return self.sim.save(slot)

    def load(self, slot):
        return self.sim.load(slot)

    def step(self, keys, *a, **k):
        self.sim.step(set(keys), *a, **k)
        self.nsteps += 1
        s = self.sim.screen()
        g = s[40:232].astype(np.int16).sum(axis=2)
        self.W += shift_of(self.prev_gray, g)
        self.prev_gray = g
        self.cur = s
        ml, fy = mario_left(s)
        if ml is not None:
            self.mxl = ml
            self.wx = self.W + ml
            self.still = self.still + 1 if fy == self.fy else 0
            self.fy = fy
            self.xst = self.xst + 1 if self.wx == self.lastwx else 0
            self.lastwx = self.wx
        if self.rec is not None:
            self.rec(self)
        if self.limit is not None and self.wx >= self.limit and (
                (self.mode == "stand" and self.xst >= 3 and self.still >= 3) or
                (self.mode == "ground" and self.still >= 2) or self.wx >= self.limit + self.slack):
            raise BudgetExceededLeg("leg boundary")

    def new_leg(self, limit, slack=60, mode="ground"):
        self.base_frame = self.nsteps
        self.limit = limit
        self.slack = slack
        self.mode = mode

L0_SRC = r'''
import numpy as np

GRAYS = [(188, 188, 188), (252, 252, 252), (124, 124, 124)]
RUN = {"Right", "x"}
START_DELAY = 24
HOLD1 = 18
HOLD2 = 15
JUMP_D = 8
PODO_V = 5.0
TOP_OK = 158
HIDDEN_OK = 22


def masks(s):
    r = s[:, :, 0]
    g = s[:, :, 1]
    b = s[:, :, 2]
    brown = (r == 172) & (g == 124) & (b == 0)
    gray = np.zeros(s.shape[:2], bool)
    for c in GRAYS:
        gray |= (r == c[0]) & (g == c[1]) & (b == c[2])
    piv = (r == 228) & (g == 92) & (b == 16)
    ri = r.astype(np.int16)
    gi = g.astype(np.int16)
    bi = b.astype(np.int16)
    fire = (ri > 200) & (gi < 180) & (bi < 90)
    fire[:40] = False
    piv[:40] = False
    return brown, gray, fire, piv


def find_mario(brown):
    ys, xs = np.nonzero(brown[40:, :])
    if len(xs) == 0:
        return None
    return (float(xs.mean()), int(ys.max()) + 41, int(xs.min()), int(xs.max()), int(ys.min()) + 40)


def npivots(piv):
    xs = np.nonzero(piv.any(axis=0))[0]
    if len(xs) == 0:
        return 0
    c = 1
    for i in range(1, len(xs)):
        if xs[i] - xs[i - 1] > 12:
            c += 1
    return c


def ground_cols(gray):
    return gray[164:176].any(axis=0)


def next_gap(gc, x0):
    n = len(gc)
    x = x0 + 1
    while x < n and gc[x]:
        x += 1
    if x >= n - 2:
        return None
    g0 = x
    while x < n and not gc[x]:
        x += 1
    return (g0, x - 1)


def podoboo_top(fire, piv, m, gap):
    f = fire & ~piv
    a = max(gap[0], 0)
    b = min(gap[1] + 4, 256)
    sub = f[100:240, a:b].copy()
    if m is not None and m[3] + 2 >= a:
        y0 = max(m[4] - 3 - 100, 0)
        y1 = max(m[1] + 2 - 100, 0)
        x0 = 0
        x1 = max(m[3] + 2 - a, 0)
        sub[y0:y1, x0:x1] = False
    ys, xs = np.nonzero(sub)
    if len(ys) < 50:
        return None
    pts = set(zip(ys.tolist(), xs.tolist()))
    best = None
    while pts:
        p = pts.pop()
        st = [p]
        n = 1
        y_lo = y_hi = p[0]
        x_lo = x_hi = p[1]
        while st:
            y, x = st.pop()
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.remove(q)
                        st.append(q)
                        n += 1
                        if q[0] < y_lo:
                            y_lo = q[0]
                        if q[0] > y_hi:
                            y_hi = q[0]
                        if q[1] < x_lo:
                            x_lo = q[1]
                        if q[1] > x_hi:
                            x_hi = q[1]
        w = x_hi - x_lo + 1
        h = y_hi - y_lo + 1
        if n >= 50 and 8 <= w <= 18 and h <= 18 and n >= 0.5 * w * h:
            t = y_lo + 100
            if best is None or t < best:
                best = t
    return best


class Podo:
    def __init__(self, unseen_ok):
        self.top = None
        self.prev_top = None
        self.last_seen = -999
        self.last_top = None
        self.dir = 0
        self.ever = False
        self.obs = 0
        self.unseen_ok = unseen_ok

    def update(self, frame, top):
        self.obs += 1
        self.prev_top = self.top
        self.top = top
        if top is not None:
            gap_t = frame - self.last_seen
            if self.last_top is not None and gap_t <= 3:
                if top > self.last_top:
                    self.dir = 1
                elif top < self.last_top:
                    self.dir = -1
            elif self.ever and gap_t > 3 and self.dir == 1 and self.last_top is not None and self.last_top >= 195 and top >= 195:
                self.dir = -1
            self.last_seen = frame
            self.last_top = top
            self.ever = True

    def safe_at(self, frame, delta):
        if not self.ever:
            return self.obs >= self.unseen_ok
        if self.top is not None:
            if self.dir != 1:
                return False
            y_pred = self.top + PODO_V * delta
            if y_pred < TOP_OK:
                return False
            t_vanish = max((232 - self.top) / PODO_V, 0)
            hidden = delta - t_vanish
            return hidden <= HIDDEN_OK
        hidden_now = frame - self.last_seen
        if self.dir == -1:
            return False
        return hidden_now + delta <= HIDDEN_OK


def play(sim):
    podo = Podo(30)
    prev_hash = None
    same = 0
    jump_t = 99
    mode = 'run'
    stage = 1
    land_frame = None
    d_prev = None
    ts_b = None
    n = 0
    while sim.budget_left > 0 and n < 1700:
        n += 1
        s = sim.screen()
        h = hash(s.tobytes())
        same = same + 1 if h == prev_hash else 0
        prev_hash = h
        if same >= 12 and sim.frame > 30:
            return
        brown, gray, fire, piv = masks(s)
        m = find_mario(brown)
        frame = sim.frame
        if frame < START_DELAY:
            sim.step(set())
            continue
        keys = set(RUN)
        state = 'run'
        if m is None:
            sim.step(keys)
            continue
        cx, feet, left, right, top = m
        on_ground = bool(gray[feet:feet + 2, left:right + 1].any())
        gc = ground_cols(gray)
        if jump_t < 99:
            jump_t += 1
        if on_ground and jump_t > 3:
            jump_t = 99
        gap = next_gap(gc, right) if feet >= 150 else None
        patient = frame < 700
        if stage == 1 and ts_b is None and npivots(piv) >= 2:
            ts_b = frame
        if gap is not None and on_ground:
            podo.update(frame, podoboo_top(fire, piv, m, gap))
        hold = HOLD1 if stage == 1 else HOLD2
        d = None
        if not on_ground:
            state = 'air'
            d_prev = None
            if jump_t < 99 and jump_t < hold:
                keys.add('z')
        elif feet >= 150 and gap is not None:
            d = gap[0] - right - 1
            vel = (d_prev - d) if d_prev is not None else 0
            if stage == 1:
                if mode == 'retreat':
                    keys = {"Left"}
                    state = 'retreat d=%d' % d
                    if d >= 90 or cx < 22:
                        mode = 'run'
                elif d < 160:
                    if d <= JUMP_D:
                        keys.add('z')
                        jump_t = 0
                        state = 'JUMP1'
                    elif d <= 40:
                        bar_ok = ts_b is None or 36 <= (frame + d / 3.0 + 42 - ts_b) % 169 <= 110
                        if not (podo.safe_at(frame, d / 3.0) and bar_ok) and patient:
                            keys = {"Left"}
                            mode = 'retreat'
                            state = 'brake d=%d' % d
                        else:
                            state = 'approach d=%d' % d
            else:
                if land_frame is None:
                    land_frame = frame
                give_up = (frame - land_frame) > 70 or not patient
                if jump_t < 99:
                    keys.add('z')
                    state = 'JUMP2 cont'
                elif d <= (12 if vel < 1.0 else JUMP_D) and (podo.safe_at(frame, 1) or give_up or vel > 1.4):
                    keys.add('z')
                    jump_t = 0
                    state = 'JUMP2'
                else:
                    delta = 3 + max(d - JUMP_D, 0) / 1.7
                    if podo.safe_at(frame, delta) or give_up:
                        state = 'go d=%d' % d
                    else:
                        if vel > 1.2 and d < 45:
                            keys = {"Left"}
                            state = 'stop-L'
                        elif d > 60:
                            keys = {"Right"}
                            state = 'hold-R'
                        else:
                            keys = set()
                            state = 'hold'
            d_prev = d
        if (not on_ground) and jump_t < 99 and jump_t >= 2:
            if stage == 1:
                stage = 2
                podo = Podo(60)
                land_frame = None
            elif podo.obs > 0:
                podo = Podo(60)
                land_frame = None
        try:
            sim.step(keys)
        except Exception as e:
            if 'Budget' in type(e).__name__:
                raise
            return
'''

L1_SRC = r'''
import numpy as np

RUN = frozenset({"Right", "x"})
RJ = frozenset({"Right", "x", "z"})
LEFT = frozenset({"Left"})
NOP = frozenset()

DEBUG = {"log": None}
PITHOLD = 5
MARGIN = 14.0


def mario_pos(s):
    reg = s[40:232]
    m = (reg[:, :, 0] == 172) & (reg[:, :, 1] == 124) & (reg[:, :, 2] == 0)
    ys, xs = np.nonzero(m)
    if len(xs) < 4:
        return None
    return float(xs.mean()), int(ys.max()) + 40, int(ys.min()) + 40


def is_brick(px):
    return ((px[..., 0] == px[..., 1]) & (px[..., 1] == px[..., 2]) & (px[..., 0] > 100))


def ground_row(s, fy):
    return is_brick(s[min(fy + 1, 232):min(fy + 9, 240)]).any(axis=0)


def obstacle_ahead(s, mx, fy):
    x0 = int(mx) + 8
    reg = s[max(fy - 13, 0):fy - 1, x0:x0 + 4].astype(int)
    gray = (reg[..., 0] == reg[..., 1]) & (reg[..., 1] == reg[..., 2]) & (reg[..., 0] > 100)
    blk = (reg[..., 0] == 228) & (reg[..., 1] == 92) & (reg[..., 2] == 16)
    return int((gray | blk).sum()) >= 6


def edge_dist(s, mx, fy):
    row = ground_row(s, fy)
    x0 = int(mx) + 7
    x = x0
    while x < 255 and row[x]:
        x += 1
    return x - x0


def block_info(s, mx=None):
    reg = s[172:198].astype(int)
    m = (reg[:, :, 0] == 228) & (reg[:, :, 1] == 92) & (reg[:, :, 2] == 16)
    cols = m.sum(axis=0)
    xs_all = np.nonzero(cols)[0]
    if len(xs_all) == 0:
        return None
    clusters = []
    start = xs_all[0]
    prev = xs_all[0]
    for x in list(xs_all[1:]) + [10 ** 6]:
        if x - prev > 3:
            clusters.append((start, prev))
            start = x
        prev = x
    for a, b_ in clusters:
        cnt = int(cols[a:b_ + 1].sum())
        if cnt < 100 or b_ - a < 10:
            continue
        sub = m[:, a:b_ + 1]
        ys, xs = np.nonzero(sub)
        cx = float(xs.mean()) + a
        if mx is not None and cx < mx - 70:
            continue
        return cx, float(ys.mean()) + 172
    return None


def lava_cols(s):
    m = (s[222:240, :, 0] == 248) & (s[222:240, :, 1] == 56) & (s[222:240, :, 2] == 0)
    return np.nonzero(m.sum(axis=0) > 5)[0]


_YY, _XX = np.mgrid[0:240, 0:256]


def arm_angle(s, cx, cy):
    c = s.astype(int)
    m = ((c[:, :, 0] == 252) & (c[:, :, 1] == 160) & (c[:, :, 2] == 68)) | \
        ((c[:, :, 0] == 248) & (c[:, :, 1] == 56) & (c[:, :, 2] == 0))
    dx = _XX - cx
    dy = _YY - cy
    r = np.hypot(dx, dy)
    k = m & (r > 12) & (r < 50) & (_YY <= 214) & (_YY >= 140)
    k &= ~((np.abs(dx) < 10) & (np.abs(dy) < 10))
    n = int(k.sum())
    if n < 6:
        return None
    ang = np.arctan2(-dy[k], dx[k])
    sx = np.cos(ang).sum()
    sy = np.sin(ang).sum()
    return float(np.degrees(np.arctan2(sy, sx)) % 360.0)


def default_params(mem):
    p = {"JD": 52, "HOLD": 6, "W": None}
    try:
        q = mem.get("gate")
        if q:
            p.update(q)
    except Exception:
        pass
    return p


def stage_a(sim, st, J=6):
    while st["f"] < 1500:
        s = sim.screen()
        mp = mario_pos(s)
        f = st["f"]
        if mp is None:
            st["lost"] = st.get("lost", 0) + 1
            sim.step(RJ if st["lost"] % 2 else RUN)
            st["f"] += 1
            continue
        mx, fy, top = mp
        row = ground_row(s, fy)
        under = bool(row[max(0, int(mx) - 3):int(mx) + 4].any())
        if st.get("prev_fy") is not None and fy == st["prev_fy"]:
            st["still"] += 1
        else:
            st["still"] = 0
        st["prev_fy"] = fy
        grounded = under and st["still"] >= 2 and (f >= 4 or st.get("handoff"))
        b = block_info(s, mx)
        if grounded and b is not None and fy >= 206 and 30 <= b[0] - mx <= 140 and (f > 60 or st.get("handoff")):
            return True
        if grounded:
            st["hold"] = 0
            st["kind"] = None
            d = edge_dist(s, mx, fy)
            if d <= J and d < 200:
                st["kind"] = "pit"
                st["hold"] = 1
                k = RJ
            elif obstacle_ahead(s, mx, fy):
                st["kind"] = "obs"
                st["hold"] = 1
                k = RJ
            else:
                k = RUN
        else:
            if f < 4 and not st.get("handoff"):
                k = RJ
                st["kind"] = "start"
            elif st.get("kind") == "pit":
                st["hold"] += 1
                k = RJ if st["hold"] <= PITHOLD else RUN
                lc = lava_cols(s)
                if b is not None and len(lc) and lc.max() + 5 < mx:
                    k = LEFT
            else:
                k = RJ
        sim.step(k)
        st["f"] += 1
    return False


def dash(sim, JD, HOLD=99, maxf=140, check=None):
    n = 0
    phase = 0
    prev_fy = None
    still = 0
    air_n = 0
    prev_scr = None
    frozen = 0
    last_dx = None
    nomp = 0
    while n < maxf:
        s = sim.screen()
        mp = mario_pos(s)
        if mp is None:
            nomp += 1
            if check is not None and nomp >= 8:
                return n, "dead"
            sim.step(RUN)
            n += 1
            continue
        nomp = 0
        mx, fy, top = mp
        b = block_info(s, mx)
        if phase >= 1 and b is not None:
            if last_dx is not None and (b[0] - mx) - last_dx > 40:
                b = None
            else:
                last_dx = b[0] - mx
        elif phase == 0 and b is not None:
            last_dx = b[0] - mx
        if check is not None:
            if prev_scr is not None and np.array_equal(s[40:215], prev_scr):
                frozen += 1
            else:
                frozen = 0
            prev_scr = s[40:215].copy()
            if frozen >= 6:
                return n, ("ok" if (last_dx is not None and last_dx <= -40) else "dead")
            r = check(s, mx, fy, b, phase, last_dx)
            if r is not None:
                return n, r
        if phase == 0:
            k = RUN
            if b is not None and b[0] - mx <= JD:
                phase = 1
                k = RJ
                air_n = 0
        elif phase == 1:
            air_n += 1
            k = RJ if air_n <= HOLD else RUN
            if prev_fy is not None and fy == prev_fy:
                still += 1
            else:
                still = 0
            if air_n > 8 and still >= 2 and ground_row(s, fy)[max(0, int(mx) - 3):int(mx) + 4].any():
                phase = 2
                k = RUN
        else:
            k = RUN
            if obstacle_ahead(s, mx, fy):
                k = RJ
        prev_fy = fy
        sim.step(k)
        n += 1
    return n, None


def to_wait_state(sim, maxn=150, dxmin=62):
    n = 0
    prev = None
    stat = 0
    while n < maxn:
        s = sim.screen()
        mp = mario_pos(s)
        if mp is None:
            return n, False
        b = block_info(s, mp[0])
        if b is None:
            return n, False
        mx = mp[0]
        dxb = b[0] - mx
        if prev is not None and abs(mx - prev) < 0.05:
            stat += 1
        else:
            stat = 0
        prev = mx
        if stat >= 3 and dxb >= dxmin - 6:
            return n, True
        sim.step(LEFT if dxb < dxmin else NOP)
        n += 1
    return n, False


def play(sim, maxf=1500):
    mem = getattr(sim, "memory", None)
    p = default_params(mem if mem is not None else {})
    st = {"f": 0, "still": 0, "airhops": 0, "prev_fy": None, "handoff": bool(getattr(sim, "handoff", False))}
    f = 0
    W = p["W"]
    JD = p["JD"]
    if not getattr(sim, "skip", False):
        stage_a(sim, st)
        f = st["f"]
        n, ok = to_wait_state(sim)
        f += n
    Wset = set(W) if W is not None else None
    waited = 0
    while f < maxf:
        s = sim.screen()
        mp = mario_pos(s)
        if mp is None:
            break
        b = block_info(s, mp[0])
        if b is None:
            break
        th = arm_angle(s, b[0], b[1])
        go = True
        if Wset is not None:
            go = th is not None and (int(th // 2) % 180) in Wset
        if go:
            break
        sim.step(NOP)
        f += 1
        waited += 1
    n, _ = dash(sim, JD, p["HOLD"], maxf=maxf - f)
    f += n
    while f < maxf:
        sim.step(RUN)
        f += 1


def dash_check(s, mx, fy, b, phase=0, last_dx=None):
    if fy >= 232:
        return "dead"
    if phase >= 1:
        if b is not None and b[0] - mx <= -50:
            return "ok"
        if b is None and last_dx is not None and last_dx <= -30:
            return "ok"
    return None


def build_good(trials):
    pts = sorted((a, ok) for a, ok, n in trials)
    good = []
    if len(pts) < 4:
        return good
    for bin_ in range(180):
        c = bin_ * 2 + 1.0
        near = [ok for a, ok in pts if min((c - a) % 360, (a - c) % 360) <= MARGIN]
        if len(near) >= 4 and all(near):
            good.append(bin_)
    return good
'''

L4_SRC = r'''
import numpy as np

GRAYS = [(188, 188, 188), (124, 124, 124), (252, 252, 252)]
BROWN = (172, 124, 0)
ORANGE = [(248, 56, 0), (252, 160, 68)]
LOG = False
FACE = 8
WAIT_DIST = 24
GO_TOP = 128
GO_LO = 116
GO_DIST = 36
GO_X0_MAX = 190

_TAB = {
    3: [0, 0, 5, 10, 14, 19, 22, 25, 27, 29, 30, 30, 30, 29, 27, 25, 22, 18, 14, 9, 4, 0],
    6: [0, 0, 5, 10, 14, 19, 23, 26, 30, 33, 36, 37, 38, 39, 38, 37, 36, 34, 31, 27, 23, 18, 13, 8, 3, 0],
    9: [0, 0, 5, 10, 14, 19, 23, 26, 30, 34, 37, 40, 42, 44, 45, 46, 46, 45, 43, 41, 38, 35, 31, 26, 21, 15, 10, 5, 0],
    12: [0, 0, 5, 10, 14, 19, 23, 26, 30, 34, 37, 40, 43, 46, 48, 50, 51, 52, 52, 51, 50, 48, 45, 42, 38, 33, 28, 23, 18, 13, 8, 3, 0],
    16: [0, 0, 5, 10, 14, 19, 23, 26, 30, 34, 37, 40, 43, 46, 48, 51, 53, 55, 57, 58, 58, 58, 57, 56, 54, 51, 47, 43, 38, 33, 27, 22, 17, 12, 7, 2, 0],
    26: [0, 0, 5, 10, 14, 19, 23, 26, 30, 34, 37, 40, 43, 46, 48, 51, 53, 55, 57, 58, 60, 61, 62, 63, 63, 64, 64, 64, 64, 63, 62, 60, 57, 54, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0],
}
HOLDS = [3, 6, 9, 12, 16, 26]


def up_at(H, t):
    tab = _TAB[H]
    if t < 0:
        return 0
    if t >= len(tab):
        return 0
    return tab[t]


def _solid_mask(s):
    m = np.zeros(s.shape[:2], bool)
    for c in GRAYS:
        m |= np.all(s == c, axis=2)
    m[:40] = False
    return m


def _find_mario(s, last):
    m = np.all(s == BROWN, axis=2)
    m[:40] = False
    rc = m.sum(axis=1)
    m[rc > 14] = False
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    ref = last[0] if last is not None else 119
    near = np.abs(xs - ref) < 30
    if not near.any():
        return None
    ys, xs = ys[near], xs[near]
    mx = np.median(xs)
    sel = np.abs(xs - mx) < 10
    ys, xs = ys[sel], xs[sel]
    return int(xs.mean()), int(ys.min()), int(ys.max())


def _green_mask(s):
    r = s[:, :, 0].astype(int)
    g = s[:, :, 1].astype(int)
    b = s[:, :, 2].astype(int)
    m = (g > r + 40) & (g > b + 40)
    m[:40] = False
    return m


def _blobs(s, mx, feet):
    m = np.zeros(s.shape[:2], bool)
    for c in ORANGE:
        m |= np.all(s == c, axis=2)
    m[:44] = False
    m[min(200, feet + 3):] = False
    m[:, :mx + 8] = False
    ys, xs = np.where(m)
    if len(ys) == 0:
        return []
    order = np.argsort(xs)
    xs, ys = xs[order], ys[order]
    out = []
    start = 0
    for i in range(1, len(xs) + 1):
        if i == len(xs) or xs[i] - xs[i - 1] > 3:
            bx, by = xs[start:i], ys[start:i]
            out.append([int(bx.min()), int(bx.max()), int(by.min()), int(by.max()), len(bx)])
            start = i
    return out


def collide(flames, mx, ytop, feet, H, delay, T=46, margin=2):
    ml, mr = mx - 6 - margin, mx + 7 + margin
    for t in range(T):
        up = up_at(H, t - delay) if H else 0
        top, bot = ytop - up - margin, feet - up + margin
        for (x0, x1, y0, y1, vx) in flames:
            fx0 = x0 + vx * t
            fx1 = x1 + vx * t
            if fx1 >= ml and fx0 <= mr and y1 >= top and y0 <= bot:
                return t
    return None


def play(sim):
    try:
        _play(sim)
    except Exception as e:
        if 'Budget' in type(e).__name__ or getattr(sim, 'budget_left', 1) <= 0:
            raise
        i = 0
        while True:
            keys = {"Right", "x"}
            if i % 40 < 14:
                keys.add("z")
            sim.step(keys)
            i += 1


def _play(sim):
    last = None
    prev_y = None
    z_prev = False
    jump_frames = 0
    cur_hold = 26
    lost = 0
    f = 0
    still = 0
    prev_s = None
    prev_fl = []
    prev_S = None
    prev_mx = None
    stuck = 0
    bz_wait = 0
    x0s = []
    tops = []
    go = False
    go_f = 0
    while True:
        s = sim.screen()
        if prev_s is not None and np.array_equal(s, prev_s):
            still += 1
            if still >= 40:
                return
        else:
            still = 0
        prev_s = s
        mar = _find_mario(s, last)
        if mar is None:
            lost += 1
            keys = {"Right", "x"}
            if lost % 2 == 0:
                keys.add("z")
            sim.step(keys)
            f += 1
            continue
        lost = 0
        mx, ytop, feet = mar
        last = mar
        S = _solid_mask(s)
        x1 = mx + 7
        H, W = S.shape
        below = S[min(feet + 1, H - 1):min(feet + 3, H), max(mx - 3, 0):mx + 4].any()
        vy0 = (prev_y is not None and prev_y == feet)
        grounded = below and vy0
        prev_y = feet

        S2 = S.copy()
        S2[1:] |= S[:-1]
        S2[:-1] |= S[1:]

        def gap_at(col):
            if col >= W:
                return False
            return not S2[feet + 1:min(feet + 44, 232), max(col - 2, 0):col + 3].any()

        def wall_h(col):
            if col >= W:
                return 0
            h = 0
            y = feet - 1
            while y > 40 and S2[y, col]:
                h += 1
                y -= 1
                if h >= 80:
                    break
            return h

        green = _green_mask(s)
        gys, gxs = np.where(green)
        bow = None
        if len(gxs) > 20:
            bow = (int(gxs.min()), int(gxs.max()), int(gys.min()), int(gys.max()))
        if bow is not None:
            tops.append(bow[2])
            tops = tops[-8:]
            x0s.append(bow[0])
            x0s = x0s[-14:]
        else:
            tops = []
            x0s = []
        bdist = None
        if bow is not None:
            bdist = (bow[0] - FACE) - x1

        raw = _blobs(s, mx, feet)
        flames = []
        for b in raw:
            x0, xe, y0, y1, n = b
            if n < 8 or (y1 - y0) > 15 or (xe - x0) > 30:
                continue
            if x0 >= 225 and xe <= 251 and y0 >= 124 and y1 <= 142:
                continue
            if bow is not None and xe >= bow[0] - 8 and x0 <= bow[1] + 14 and y1 >= bow[2] - 4:
                continue
            vx = -3.0
            best = None
            for p in prev_fl:
                if abs(p[2] - y0) <= 2 and -9 <= x0 - p[0] <= 1 and (best is None or abs(p[2] - y0) < best[0]):
                    best = (abs(p[2] - y0), x0 - p[0])
            if best is not None and xe < 254:
                vx = float(best[1])
            flames.append((x0, xe, y0, y1, vx))
        prev_fl = [(b[0], b[1], b[2], b[3]) for b in flames]

        want = False
        why = ''
        hold_pref = [26]
        if grounded:
            for d in range(2, 7):
                if gap_at(x1 + d):
                    want = True
                    why = 'gap'
                    hold_pref = [12, 16, 9, 26]
                    break
            if not want:
                wh = 0
                for d in range(2, 14):
                    wh = max(wh, wall_h(x1 + d))
                if wh > 4:
                    want = True
                    why = 'wall%d' % wh
                    hold_pref = [h for h in HOLDS if max(_TAB[h]) >= wh + 6] or [26]
                    if wh <= 20:
                        hold_pref = [h for h in hold_pref if h <= 12] or hold_pref

        keys = {"Right", "x"}
        in_bz = bow is not None and bow[1] > mx and bdist is not None and bdist < 75
        use_hold = None

        if prev_S is not None and grounded and not in_bz and not go and np.array_equal(S, prev_S) and mx == prev_mx:
            stuck += 1
        else:
            stuck = 0
        prev_S = S
        prev_mx = mx
        if stuck >= 8 and grounded and not want:
            want = True
            why = 'stuck'
            hold_pref = [16, 26] if (stuck // 8) % 2 else [26]

        stay_hit = collide(flames, mx, ytop, feet, 0, 0) if (flames and grounded) else None
        if grounded and not go:
            if want and not in_bz:
                chosen = None
                for h in hold_pref:
                    if collide(flames, mx, ytop, feet, h, 0) is None:
                        chosen = h
                        break
                use_hold = chosen if chosen is not None else hold_pref[0]
            elif stay_hit is not None and stay_hit <= 16:
                chosen = None
                for h in HOLDS:
                    if collide(flames, mx, ytop, feet, h, 0) is None:
                        chosen = h
                        break
                if chosen is None:
                    chosen = 6
                use_hold = chosen
                want = True
                why = 'fire(t%d,H%d)' % (stay_hit, chosen)
            elif want and in_bz:
                want = False

        if go:
            go_f += 1
            keys = {"Right", "x"}
            want = False
            use_hold = None
            if bow is None or mx - 6 > bow[1] + 1 or go_f > 70:
                go = False
                why = 'go-done'
            else:
                why = 'go'
        elif in_bz:
            why = (why + ' bz%d' % bdist).strip()
            keys = set()
            if grounded and not want:
                rising = len(tops) >= 4 and tops[-1] <= tops[-4] - 3
                bz_wait += 1
                x0max = GO_X0_MAX if bz_wait < 150 else GO_X0_MAX + 20
                vxb = (x0s[-1] - x0s[-12]) / 11.0 if len(x0s) >= 12 else 0.0
                x0_pred = bow[0] + min(0.0, vxb) * 30
                if GO_LO <= bow[2] <= GO_TOP and rising and bdist <= GO_DIST and x0_pred <= x0max:
                    go = True
                    go_f = 0
                    keys = {"Right", "x"}
                    why += ' GO'
                elif bdist > WAIT_DIST:
                    keys = {"Right"}
                elif bdist < 5:
                    keys = {"Left"}
        if want and use_hold is None:
            use_hold = hold_pref[0]

        if bow is not None and f % 20 == 0:
            print("L4 f", f, "mx", mx, "feet", feet, "bow", bow, "bdist", bdist, "in_bz", in_bz, "go", go, "bzw", bz_wait,
                  "tops", tops[-4:], "x0s", x0s[-3:], "fl", [(b[0], b[2], b[4]) for b in flames][:3], "gr", grounded, "want", want, why)

        if jump_frames == 0:
            z = bool(grounded and want and not z_prev and not go)
            if z:
                jump_frames = 1
                cur_hold = use_hold
        else:
            jump_frames += 1
            if jump_frames > cur_hold or (jump_frames > 6 and grounded):
                z = False
                jump_frames = 0
            else:
                z = True
        if in_bz and not go and jump_frames > 0:
            keys = set()
        if z:
            keys.add("z")
        z_prev = z
        sim.step(keys)
        f += 1
'''

P_SRC = r'''
import math
import numpy as np

GRAYS = [(188, 188, 188), (124, 124, 124), (252, 252, 252)]
RUN = frozenset({"Right", "x"})
RJ = frozenset({"Right", "x", "z"})
LEFT = frozenset({"Left"})
WALK = frozenset({"Right"})
NOP = frozenset()
LOGF = True
EXIT_BOWSER = False
FLAME_V = -1.45
STEER = False
FWD = False
XM = 0          # extra horizontal flame margin (timing slack ~2-5 frames)
WN1 = 900       # patience (frames waiting at one edge) before lax rules (level timer expires ~frame 6000!)
WN2 = 600
LATE = 5500

_RUN = {
    3: [0, 0, 6, 12, 18, 23, 27, 30, 32, 33, 34, 34, 33, 31, 29, 26, 22, 17, 12, 6, 0],
    4: [0, 0, 6, 12, 18, 23, 28, 33, 36, 39, 41, 42, 42, 41, 40, 38, 35, 31, 26, 21, 15, 8, 3, 0],
    6: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 45, 47, 49, 49, 49, 48, 46, 44, 40, 36, 31, 26, 21, 16, 11, 6, 1, 0],
    8: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 50, 53, 54, 55, 56, 55, 54, 51, 48, 44, 40, 35, 30, 25, 20, 15, 10, 5, 0],
    10: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 57, 60, 61, 62, 61, 60, 58, 56, 52, 48, 43, 38, 33, 28, 23, 18, 13, 8, 3, 0],
    20: [0, 0, 6, 12, 18, 23, 28, 33, 38, 42, 47, 51, 54, 58, 61, 65, 68, 70, 73, 75, 77, 79, 81, 81, 81, 81, 79, 76, 73, 69, 64, 59, 54, 49, 44, 39, 34, 29, 24, 19, 14, 9, 4, 0],
}

_t20 = _RUN[20]
_t10 = _RUN[10]
_RUN[14] = _t20[:14] + [v_ + 14 for v_ in _t10[10:]] + [0]


def _interp(pts):
    xs_ = [p[0] for p in pts]
    ys_ = [p[1] for p in pts]
    return [int(round(float(np.interp(t_, xs_, ys_)))) for t_ in range(xs_[-1] + 1)]


# measured standing jump (z held 10 frames): rise above takeoff vs frames since press
_RUN[100] = _interp([(0, 0), (2, 0), (4, 19), (9, 40), (14, 48), (19, 40), (24, 17), (27, 0)])

# standing jump with z held ~26 frames (apex ~64px, airtime ~45 frames; measured in the Bowser leg)
_RUN[26] = [0, 0, 5, 10, 14, 19, 23, 26, 30, 34, 37, 40, 43, 46, 48, 51, 53, 55, 57, 58, 60, 61, 62, 63, 63, 64, 64, 64, 64, 63, 62, 60, 57, 54, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5, 0]

PG = 0.12      # podoboo gravity (px/frame^2)
PV0 = -5.1     # podoboo launch velocity
PHID = 18      # hidden frames before relaunch (measured >=24 in every observed cycle; randomised, so keep conservative)
PCYC = 110
HTAB = (3, 4, 6, 8, 10, 14, 20)


def up_at(H, t):
    tab = _RUN[H]
    if t < 0 or t >= len(tab) - (3 if H == 14 else 0):
        return 0
    return tab[t]


def air_path(H, vv):
    """cumulative horizontal displacement per frame since the jump press (same model as air_dist)"""
    T = flight_len(H)
    fast = vv >= 1.95
    cap = 3.0 if fast else 1.55
    v = max(vv, 0.0) if fast else min(max(vv, 0.0), cap)
    acc = 0.05
    if fast and vv < 2.8:
        cap = min(3.0, vv + 0.35)
        acc = 0.03
    xs = [0.0]
    x = 0.0
    for t in range(T + 4):
        if t < T and v < cap:
            v = min(cap, v + acc)
        x += v
        xs.append(x)
    return xs


def pod_clear(pdb, W, frame, cen0, feet, H, vv, margin=3):
    """True if a jump of hold H taken now (running at vv) never overlaps any tracked podoboo (ballistic model)"""
    xs = air_path(H, vv)
    T = flight_len(H)
    for e in pdb.d:
        ex = e["x"]
        if ex < W + cen0 - 40 or ex > W + cen0 + 200:
            continue
        for t in range(0, T + 3):
            xm = W + cen0 + xs[min(t, len(xs) - 1)]
            if abs(xm - ex) > 22:
                continue
            y = pdb.pred(e, frame, t)
            if y is None:
                return False
            if y >= 900:
                continue
            up = up_at(H, t)
            mt = feet - 15 - up - margin
            mb = feet - up + margin
            if y + 15 >= mt and y <= mb:
                return False
    return True


def flight_len(H):
    return len(_RUN[H]) - (3 if H == 14 else 0)


def air_dist(H, vv, hor, t_start=0):
    """horizontal distance covered from frame t_start of a jump of hold H until landing"""
    T = flight_len(H)
    fast = vv >= 1.95
    cap = 3.0 if fast else 1.55
    v = max(vv, 0.0) if fast else min(max(vv, 0.0), cap)
    acc = 0.05
    if fast and vv < 2.8:
        cap = min(3.0, vv + 0.35)    # measured: slower take-offs barely accelerate in the air
        acc = 0.03
    x = 0.0
    for t in range(t_start, T):
        if hor == 'R':
            v = min(cap, v + acc) if v < cap else v
        else:
            v = max(v - 0.12, 0.0)
        x += v
    return x


def masks(s):
    r = s[:, :, 0]
    g = s[:, :, 1]
    b = s[:, :, 2]
    brown = (r == 172) & (g == 124) & (b == 0)
    gray = np.zeros(s.shape[:2], bool)
    for c in GRAYS[:2]:
        gray |= (r == c[0]) & (g == c[1]) & (b == c[2])
    ri = r.astype(np.int16)
    gi = g.astype(np.int16)
    bi = b.astype(np.int16)
    fire = (ri > 200) & (gi < 180) & (bi < 90)
    fire[:40] = False
    brown[:40] = False
    return brown, gray, fire


def find_mario(brown):
    ys, xs = np.nonzero(brown)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def blobs(fire, mario):
    m = fire.copy()
    m[:56] = False
    m[207:] = False
    if mario:
        x0, x1, y0, y1 = mario
        m[max(0, y0 - 3):y1 + 4, max(0, x0 - 3):x1 + 4] = False
    seen = np.zeros_like(m)
    out = []
    ys, xs = np.nonzero(m)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if seen[y, x]:
            continue
        st = [(y, x)]
        seen[y, x] = True
        pts = []
        while st:
            cy, cx = st.pop()
            pts.append((cy, cx))
            for dy in (-2, -1, 0, 1, 2):
                for dx in (-2, -1, 0, 1, 2):
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < 240 and 0 <= nx < 256 and m[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        st.append((ny, nx))
        if len(pts) < 14:
            continue
        a = np.array(pts)
        w = int(a[:, 1].max() - a[:, 1].min() + 1)
        h = int(a[:, 0].max() - a[:, 0].min() + 1)
        out.append((int(a[:, 1].min()), int(a[:, 1].max()), int(a[:, 0].min()), int(a[:, 0].max()), len(pts), w, h))
    return out


class PodoDB:
    """world-x keyed podoboo tracker with a ballistic + hidden-interval cycle model"""

    def __init__(self):
        self.d = []
        self.hid = PHID
        self.gaps = []   # (world gap x, first frame seen)

    def note_gap(self, gx, frame):
        for g in self.gaps:
            if abs(g[0] - gx) < 20:
                return
        self.gaps.append((gx, frame))

    def gap_age(self, gx, frame):
        for g in self.gaps:
            if abs(g[0] - gx) < 20:
                return frame - g[1]
        return 0

    def update(self, frame, W, pods):
        seen = set()
        for b in pods:
            wx = W + (b[0] + b[1]) / 2.0
            top = float(b[2])
            e = None
            bd = 14.0
            for q in self.d:
                if abs(q["x"] - wx) < bd:
                    bd = abs(q["x"] - wx)
                    e = q
            if e is None:
                e = {"x": wx, "y": top, "vy": None, "f": frame, "hidf": None, "vis": frame, "h": [(frame, top)]}
                self.d.append(e)
            else:
                if e["hidf"] is not None or frame - e["vis"] > 3:
                    if e["hidf"] is not None:
                        hobs = frame - e["hidf"]
                        if 14 <= hobs <= 60:
                            self.hid = min(self.hid, max(hobs - 4, 14))
                    e["hidf"] = None
                    e["h"] = [(frame, top)]
                    e["vy"] = PV0 if top >= 170 else None
                else:
                    e["h"].append((frame, top))
                    e["h"] = e["h"][-4:]
                    if len(e["h"]) >= 2:
                        f0, y0 = e["h"][0]
                        f1, y1 = e["h"][-1]
                        e["vy"] = (y1 - y0) / float(max(1, f1 - f0)) + PG * 0.5 * (f1 - f0) * 0.0
                e["x"] = 0.7 * e["x"] + 0.3 * wx
            e["y"] = top
            e["f"] = frame
            e["vis"] = frame
            seen.add(id(e))
        for e in self.d:
            if id(e) in seen:
                continue
            if e["hidf"] is None and frame - e["vis"] == 1:
                if e["y"] >= 176 and (e["vy"] is None or e["vy"] > 0):
                    e["hidf"] = frame
        self.d = [e for e in self.d if not (e["hidf"] is None and frame - e["vis"] > 200)]

    def pred(self, e, frame, dt):
        """predicted top y at frame+dt; 999 = hidden; None = unknown"""
        target = frame + dt
        if e["hidf"] is not None:
            t_re = e["hidf"] + self.hid
            if target < t_re:
                return 999.0
            if frame > t_re + 3:
                return None
            y, vy, t = 192.0, PV0, t_re
        else:
            if e["vy"] is None:
                return None
            # velocity measured at the middle of the history window: advance to the last sample
            y, vy, t = e["y"], e["vy"], e["f"]
            if frame - e["vis"] > 60:
                return None
        guard = 0
        while t < target and guard < 400:
            guard += 1
            t += 1
            y += vy
            vy += PG
            if vy > 0 and y >= 190:
                t_re = t + self.hid
                if target < t_re:
                    return 999.0
                y, vy, t = 192.0, PV0, t_re
        return y

    def safe(self, ga, gb, frame, t0, t1, B, lax=False, unk_ok=False):
        near = [e for e in self.d if ga - 20 <= e["x"] <= gb + 20]
        if not near:
            if unk_ok and self.gap_age((ga + gb) / 2.0, frame) <= 0:
                return True
            return self.gap_age((ga + gb) / 2.0, frame) >= (125 if not lax else 60)
        t0 = int(max(t0, 0))
        t1 = int(max(t1, t0))
        for e in near:
            for dt in range(t0, t1 + 1, 2):
                y = self.pred(e, frame, dt)
                if y is None:
                    if lax and e["hidf"] is not None:
                        continue
                    return False
                if y < B - 16:
                    return False
        return True


SCEN = ((1.0, 1.0), (1.35, 1.0), (1.0, 0.8), (0.75, 1.0), (1.0, 1.15))


def flame_hit(flames, mx0, mx1, feet, vx, H, T=40, margin=2, td=0, dcap=None):
    """robust version: worst case over speed scenarios, margin + 3"""
    best = None
    for kf, km in SCEN:
        fl2 = [(a, b, c, d, e * kf) for (a, b, c, d, e) in flames]
        r = _flame_hit(fl2, mx0, mx1, feet, vx * km, H, T, margin + 3, td, dcap)
        if r is not None and (best is None or r < best):
            best = r
    return best


def _flame_hit(flames, mx0, mx1, feet, vx, H, T=40, margin=2, td=0, dcap=None):
    """first t at which Mario (x speed vx, jump hold H or None, takeoff delay td) overlaps a flame, else None"""
    for t in range(T):
        up = up_at(H, int(t - td)) if H else 0
        top = feet - 15 - up - margin
        bot = feet - up + margin
        dx = vx * max(t - td, 0) if dcap is not None else vx * t
        if dcap is not None:
            dx = min(dx, dcap)
        ml = mx0 + dx - margin - XM
        mr = mx1 + dx + margin + XM
        for (fx0, fx1, fy0, fy1, fvx) in flames:
            a0 = fx0 + fvx * t
            a1 = fx1 + fvx * t
            if a1 >= ml and a0 <= mr and fy1 >= top and fy0 <= bot:
                return t
    return None


def wall_ahead_f(gray, x1, base):
    """a wall (gray block above floor level) within 40px ahead"""
    try:
        reg = gray[max(base - 24, 0):base - 6, x1 + 2:min(x1 + 42, 256)]
        return bool(reg.any())
    except Exception:
        return False


def _run_time(dist, v0):
    x = 0.0
    v = max(v0, 0.0)
    t = 0
    while x < dist and t < 200:
        v = min(3.0, v + 0.065)
        x += v
        t += 1
    return float(t)


def t_edge(d, vv, go_flag, RW):
    """(frames until takeoff at the edge, window start offset, window end offset) relative to takeoff"""
    dd = max(d - 4, 0)
    if vv >= 1.0:
        return _run_time(dd, vv), 3.0, 16.0
    RWe = 16 if RW < 50 else RW + 6
    if d < (10 if RW < 50 else RW):
        back = (RWe - d) / 1.3 + 3.0
        dist = float(RWe)
    else:
        back = 0.0
        dist = float(dd)
    return back + _run_time(dist, 0.3), -1.0, 20.0


def play(sim):
    prev_fl = []
    jt = 99
    H = 6
    jkeys = RJ
    airL = 0
    pdb = PodoDB()
    gap_id = None
    base = None
    prev_feet = None
    prev_wx = None
    v = 0.0
    wait_n = 0
    n = 0
    last_fvx = -1.5
    go_flag = False
    backoff = False
    vhist = []
    last_left_n = -99
    fprev2 = None
    jinfo = None
    jland = None
    vtake = 2.0
    while sim.budget_left > 0 and n < 7500:
        n += 1
        s = sim.screen()
        if EXIT_BOWSER:
            gm_ = (s[40:, :, 1].astype(np.int16) > s[40:, :, 0].astype(np.int16) + 40) & (s[40:, :, 1].astype(np.int16) > s[40:, :, 2].astype(np.int16) + 40)
            if int(gm_.sum()) > 60:
                return
        brown, gray, fire = masks(s)
        m = find_mario(brown)
        if m is None:
            sim.step(RUN)
            continue
        x0, x1, y0, y1 = m
        feet = y1 + 1
        wx = sim.wx
        vraw = 0.0
        if prev_wx is not None:
            vraw = float(wx - prev_wx)
            v = 0.5 * v + 0.5 * vraw
        prev_wx = wx
        vhist.append(vraw)
        del vhist[:-5]
        stable = prev_feet is not None and prev_feet == feet
        prev_feet = feet
        sup = bool(gray[feet:feet + 2, x0:x1 + 1].any())
        if jt < 99:
            jt += 1
        grounded = sup and stable and (jt > 3)
        if grounded:
            if jt < 99 and jinfo is not None:
                print("P land dx", wx - jinfo[0], "after", jt, "frames; jump v", jinfo[1], "H", jinfo[2], "x1", x1, "d", jinfo[3])
                jinfo = None
            jland = None
            jt = 99
            base = feet
            airL = 0
        bl = blobs(fire, m)
        wx_scrolling = x0 >= 108
        ispod = [(b[6] >= 13 and 10 <= b[5] <= 20 and b[5] - b[6] <= 3) for b in bl]
        pods = [b for b, ip in zip(bl, ispod) if ip]
        pdb.update(sim.frame, sim.W, pods)
        flames = []
        for b, ip in zip(bl, ispod):
            if (not ip) and b[5] >= 9 and b[5] >= b[6] + 2 and b[3] < 192 and b[2] > 70 and (base is not None and base < 180):
                vxf = last_fvx - (vraw if wx_scrolling else 0.0)
                for p in prev_fl:
                    if abs(p[2] - b[2]) <= 6 and -8 <= b[0] - p[0] <= 0:
                        vxf = float(b[0] - p[0])
                        if -6.5 <= vxf <= -0.4 and b[1] < 250 and b[0] > 3:
                            last_fvx = 0.7 * last_fvx + 0.3 * (vxf + (vraw if wx_scrolling else 0.0))
                flames.append((b[0], b[1], b[2], b[3], FLAME_V))     # world speed is constant; reference frame = current screen
        prev_fl = [(f[0], f[1], f[2], f[3]) for f in flames]
        keys = RUN
        why = ""
        if jt < 99:
            # ---- in-air flame dodge: change the hold / air-brake if a visible flame will hit the current arc
            if flames and base is not None and base < 200 and 2 <= jt < 50:
                try:
                    holding = jt < H
                    right_now = (airL == 0) and (jt < H or True)
                    vcur = max(min(vtake, 3.4), 0.0) if airL == 0 else max(v, 0.0)
                    vx_cur = vcur if airL == 0 else 0.0
                    chit = flame_hit(flames, x0, x1, base + 1, vx_cur, H, 64, 2, -jt)
                    if chit is not None and chit <= 52:
                        rowb = gray[min(base + 4, 239):min(base + 12, 240), :].any(axis=0)
                        cen0 = (x0 + x1) / 2.0
                        opts = []
                        for h_ in (HTAB if holding else (H,)):
                            if holding and h_ < jt:
                                continue
                            for hor in ('R',):
                                if airL > 0:
                                    # standing flame-hop: keep it vertical (no forward drift into the gap)
                                    if flame_hit(flames, x0, x1, base + 1, 0.0, h_, 64, 2, -jt) is None:
                                        opts.append((abs(h_ - H), h_, 'V'))
                                    continue
                                vxo = vcur if hor == 'R' else vcur * 0.4
                                if flame_hit(flames, x0, x1, base + 1, vxo, h_, 64, 2, -jt) is not None:
                                    continue
                                cen = cen0 + air_dist(h_, vcur, hor, jt)
                                c_ = int(cen)
                                if c_ - 3 < 0 or c_ + 4 > 256:
                                    continue
                                if int(rowb[c_ - 3:c_ + 4].sum()) < 5:
                                    continue
                                opts.append((abs(h_ - H) + (0 if hor == 'R' else 2), h_, hor))
                        if opts:
                            opts.sort()
                            _, h_, hor = opts[0]
                            if holding:
                                H = h_
                            if hor == 'V':
                                pass
                            elif hor == 'L':
                                jkeys = frozenset({"z"})
                                airL = 40
                            else:
                                jkeys = RJ
                                airL = 0
                            print("P dodge jt", jt, "chit", chit, "->", h_, hor, "flames", [(f[0], f[2], round(f[4], 1)) for f in flames])
                except Exception as e_:
                    print("dodge err", repr(e_))
            if jland is not None and airL == 0 and base is not None and 1 <= jt < H and H < 14:
                # in-flight landing check: extend the hold while z is still held if the predicted landing is short
                try:
                    gx1s = jland[0] - sim.W
                    lo_ = gx1s + 10
                    cen_ = (x0 + x1) / 2.0
                    vl_ = vhist[-2:]
                    vc_ = max(min(vtake, 3.4), 0.0)     # air-pose sprite jitter makes in-flight speed unreliable
                    pred_ = cen_ + air_dist(H, vc_, 'R', jt)
                    if pred_ < lo_:
                        for h_ in HTAB:
                            if h_ <= H or h_ > 14:
                                continue
                            p2_ = cen_ + air_dist(h_, vc_, 'R', jt)
                            if p2_ >= lo_ and ((not flames) or flame_hit(flames, x0, x1, base + 1, vc_, h_, 40, 2, -jt) is None):
                                print("P holdext jt", jt, "H", H, "->", h_, "pred", round(pred_, 1), "lo", lo_, "v", round(vc_, 2))
                                H = h_
                                break
                except Exception as e_:
                    print("holdext err", repr(e_))
            if jt < H:
                keys = jkeys
            else:
                keys = RUN
            if jt >= H and airL > 0:
                keys = LEFT if v > 0.4 else NOP
            elif jt < H and airL > 0 and jkeys is not RJ and "Right" not in jkeys:
                keys = frozenset({"z", "Left"}) if v > 0.4 else frozenset({"z"})
            why = "air"
            if (jt >= max(H, 6) and airL == 0 and base is not None and base < 200 and v > 1.0 and jt < 60
                    and fprev2 is not None and feet > fprev2 and feet <= base):
                cxm = min((x0 + x1) // 2, 255)
                if bool(gray[base + 2:base + 8, cxm].any()):
                    rowa = gray[min(base + 4, 239):min(base + 12, 240), :].any(axis=0)
                    ca = x1 + 1
                    while ca < 256 and rowa[ca]:
                        ca += 1
                    if ca < 236:
                        c3a = ca
                        while c3a < 256 and not rowa[c3a]:
                            c3a += 1
                        te2 = max(ca - x1 - 5, 0) / 2.8 + 5.0
                        if True:
                            keys = LEFT
                            why = "air-brake"
            if STEER and jland is not None and airL == 0 and base is not None and jt < 70:
                # closed-loop landing steering: pick R / coast / Left each frame so the predicted landing is mid-pillar
                gx1s = jland[0] - sim.W
                lo_ = gx1s + 6
                hi_ = (jland[1] - sim.W - 3) if jland[1] is not None else (gx1s + 44)
                mid_ = (lo_ + hi_) / 2.0
                cen_ = (x0 + x1) / 2.0
                trem = max(flight_len(H) + 1 - jt, 1)
                vc = vhist[-1] * 0.5 + vhist[-2] * 0.5 if len(vhist) >= 2 else max(v, 0.0)
                vc = max(vc, 0.0)
                sc_list = []
                for nm_, acc_, fl_ in (("R", 0.1, None), ("N", 0.0, None), ("L", -0.16, -1.0)):
                    vq = vc
                    cq = cen_
                    for _ in range(trem):
                        vq = vq + acc_
                        if vq > 3.0:
                            vq = 3.0
                        if fl_ is not None and vq < fl_:
                            vq = fl_
                        cq += vq
                    sc_list.append((abs(cq - mid_) if lo_ <= cq <= hi_ else abs(cq - mid_) + 30.0, nm_, cq))
                best_ = min(sc_list)
                pick = best_[1]
                for sc0, nm0, cq0 in sc_list:
                    if nm0 == "R" and sc0 <= best_[0] + 5.0:
                        pick = "R"
                holdz = jt < H
                if pick == "R":
                    keys = RJ if holdz else RUN
                elif pick == "N":
                    keys = frozenset({"x", "z"}) if holdz else frozenset({"x"})
                else:
                    keys = frozenset({"x", "z", "Left"}) if holdz else frozenset({"x", "Left"})
                why = "steer-" + pick
        elif base is not None:
            B = base
            row = gray[min(B + 4, 239):min(B + 12, 240), :].any(axis=0)
            gx0 = None
            gx1 = 255
            c = x1 + 1
            while c < 256 and row[c]:
                c += 1
            if c < 254:
                gx0 = c
                while c < 256 and not row[c]:
                    c += 1
                gx1 = c - 1 if c < 256 else 255
            d = (gx0 - x1 - 1) if gx0 is not None else 999
            gid = int(sim.W + gx0) if gx0 is not None else None
            if gid is not None and (gap_id is None or abs(gid - gap_id) > 14):
                wait_n = 0
            if gid is not None:
                gap_id = gid
                if gx1 < 236:
                    pdb.note_gap(gid + (gx1 - gx0) / 2.0, sim.frame)
            nxt = None
            c3s = None
            if gx0 is not None:
                ga = sim.W + gx0
                gb = sim.W + gx1
                if gx1 < 236:
                    c2 = gx1 + 1
                    while c2 < 256 and row[c2]:
                        c2 += 1
                    if c2 < 232:
                        c3 = c2
                        while c3 < 256 and not row[c3]:
                            c3 += 1
                        if c3 < 250:
                            pdb.note_gap(sim.W + (c2 + c3) / 2.0, sim.frame)
                            nxt = (sim.W + c2, sim.W + c3, c2 - gx0)
                            c3s = c2
                        else:
                            c3s = None
            rise = 0
            if gx0 is not None and gx1 < 250:
                col = min(gx1 + 8, 255)
                colm = gray[100:236, col]
                ys = np.nonzero(colm)[0]
                if len(ys):
                    rise = B - (100 + int(ys[0]))
            wh = 0
            dw = 999
            lim = gx0 if gx0 is not None else 256
            for cw in range(x1 + 2, min(lim, x1 + 70)):
                if gray[B - 8, cw]:
                    dw = cw - x1 - 1
                    yy = B - 1
                    while yy > 60 and gray[yy, cw]:
                        wh += 1
                        yy -= 1
                    break
            vh = sorted(vhist[-3:])
            vmed = vh[len(vh) // 2] if vh else 0.0
            vv = max(min(vmed, 3.2), 0.0)
            vh2 = vhist[-4:]
            vf = max(min(sum(vh2) / float(max(len(vh2), 1)), 3.4), 0.0)
            vm = min(vf + 0.2, 3.4) if vf > 1.0 else vf
            if vhist:
                vm = min(vm, max(vhist[-2:]) + 0.1, max(vhist[-1], 0.0) + 0.35)
                vv = min(vv, max(vhist[-1], 0.0) + 0.6)
            bk_ok = bool(gray[B + 2:B + 10, max(x0 - 26, 0)].any() and gray[B + 2:B + 10, max(x0 - 14, 0)].any())
            slide = min(v, vraw)
            cl_ = x0 - 1
            while cl_ >= 0 and row[cl_]:
                cl_ -= 1
            left_edge = cl_ + 1
            avail = (gx0 - left_edge - 16) if gx0 is not None else 26
            RW = 50 if rise >= 34 else int(min(max(avail, 26), 40))
            runlen = (gx0 - left_edge) if gx0 is not None else 999
            short_rw = bool(rise < 34 and runlen < 46)
            if short_rw:
                RW = max(int(runlen) - 8, 6)      # pillar too short for a long run-up: do not dither back and forth
            if backoff and (d >= RW + 4 or not bk_ok):
                backoff = False
            want = None
            if gx0 is not None and d <= 4 and d < 999:
                if rise >= 34:
                    want = "gapH"
                elif rise >= 12:
                    want = "gapM"
                else:
                    want = "gap"
            elif wh > 5 and dw <= (44 if wh > 30 else 26):
                want = "wall"
            if want == "gapH" or (want is None and gx0 is not None and rise >= 34):
                Hn = 10 if rise >= 40 else 8
            elif want == "gapM" or (want is None and gx0 is not None and rise >= 12):
                Hn = 6
            elif want == "wall":
                Hn = 10 if wh <= 36 else 20
            else:
                Hn = 3 if vv >= 2.4 else 6
                if n - last_left_n <= 8 and vv < 2.8:
                    Hn = 10
                if want == "gap" and vv < 1.6:
                    Hn = 14
            fh = None
            if flames:
                fh = flame_hit(flames, x0, x1, feet, 0.0, None, 30)
            go_ok = True
            if want in ("gap", "gapM", "gapH"):
                need = 1.0 if short_rw else 2.0
                forced = vv >= 2.3 and want in ("gap", "gapM")   # running fast: braking at the edge only makes the hop fall short
                if vv >= need:
                    pass
                elif not pdb.safe(ga, gb, sim.frame, 3, 24, B, wait_n >= WN1 or sim.frame > LATE):
                    go_ok = False
                elif nxt is not None and wait_n < WN2 and sim.frame <= LATE and not pdb.safe(nxt[0], nxt[1], sim.frame, nxt[2] / 2.4 - 2, nxt[2] / 2.4 + 20, B, wait_n >= WN1, True):
                    go_ok = False
                if vv < need and bk_ok:
                    backoff = True
                if backoff:
                    go_ok = False
            if want in ("gap", "gapM", "gapH") and go_ok:
                # choose the hold: landing must reach the next pillar, flames must miss the arc
                sv = max(vm, 2.4) if vm >= 1.95 else max(vm, 1.75)
                if want == "gap":
                    cl = [Hn, 10, 14, 6, 8, 4, 3, 20]
                elif want == "gapH":
                    cl = [Hn, 10, 8, 14]
                else:
                    cl = [Hn, 8, 10]
                chosen = None
                fallback = None
                best_sc = None
                podrej = []
                fb2 = None
                fb3 = None
                fb_sc = 0.0
                for h_ in cl:
                    lok = True
                    sc_ = 0.0
                    if want == "gap" and rise < 12:
                        cen = (x0 + x1) / 2.0 + air_dist(h_, vm, 'R', 0)
                        lo = gx1 + 10
                        hi = (c3s - 3) if (c3s is not None) else (gx1 + 46)
                        lok = lo <= cen <= hi
                        tgt = (gx1 + 1 + c3s - 1) / 2.0 if c3s is not None else gx1 + 24
                        sc_ = abs(cen - tgt)
                    fok = (not flames) or flame_hit(flames, x0, x1, feet, sv, h_, min(44, flight_len(h_) + 6), 3, 0) is None
                    pc_ = True
                    if want in ("gap", "gapM", "gapH") and pdb.d:
                        try:
                            pc_ = pod_clear(pdb, sim.W, sim.frame, (x0 + x1) / 2.0, feet, h_, vm)
                        except Exception as e_:
                            pc_ = True
                            print("pod_clear err", repr(e_))
                        if not pc_:
                            fok = False
                            podrej.append(h_)
                    if lok:
                        frel = (not flames) or _flame_hit(flames, x0, x1, feet, sv, h_, min(44, flight_len(h_) + 6), 2, 0) is None
                        if pc_ and frel and fb2 is None:
                            fb2 = h_
                        if frel and fb3 is None:
                            fb3 = h_
                    if lok and fok:
                        if want != "gap" or rise >= 12:
                            chosen = h_
                            break
                        if best_sc is None or sc_ < best_sc - 0.5:
                            best_sc = sc_
                            chosen = h_
                    if lok and (fallback is None or sc_ < fb_sc - 0.5):
                        fallback = h_
                        fb_sc = sc_
                if fb2 is not None:
                    fallback = fb2
                elif fb3 is not None:
                    fallback = fb3
                if want == "gap" and rise < 12 and LOGF:
                    print("P hopsel", "x0", x0, "x1", x1, "gx0", gx0, "gx1", gx1, "c3s", c3s, "vv", round(vv, 2), "vm", round(vm, 2),
                          "dist", {h_: round(air_dist(h_, vv, 'R', 0), 1) for h_ in (3, 6, 10, 14)},
                          "chosen", chosen, "fb", fallback, "Hn", Hn, "podrej", podrej)
                if chosen is not None:
                    Hn = chosen
                elif fallback is not None:
                    if wait_n >= WN1 or forced or (short_rw and wait_n >= 80):
                        Hn = fallback
                    else:
                        go_ok = False
                else:
                    # no hold reaches the next pillar at this speed: back up for a longer run-up if possible
                    if want == "gap" and rise < 12 and bk_ok and wait_n < WN1 and not forced:
                        go_ok = False
                        backoff = True
                    elif flames and (not forced) and flame_hit(flames, x0, x1, feet, sv, Hn, 44, 2, 0) is not None and wait_n < WN1:
                        go_ok = False
                    elif want == "gap" and rise < 12 and vv < 2.2:
                        if wait_n < WN1 + 300 and not forced:
                            go_ok = False
                            if bk_ok:
                                backoff = True
                        else:
                            Hn = 14
            if want and go_ok:
                jt = 0
                H = Hn
                jkeys = RJ
                keys = RJ
                airL = 0
                jinfo = (wx, round(vraw, 2), Hn, d)
                vtake = vm
                jland = None
                if want == "gap" and rise < 12 and gx0 is not None:
                    jland = (sim.W + gx1, (sim.W + c3s) if c3s is not None else None)
                why = "JUMP %s H%d rise%d v%.1f d%d" % (want, Hn, rise, vv, d)
                wait_n = 0
                go_flag = False
            elif want in ("gap", "gapM", "gapH"):
                wait_n += 1
                keys = LEFT if (vv > 0.8 or (backoff and bk_ok)) else NOP
                why = "wait-gap"
            elif gx0 is not None and d < 80:
                te, u0, u1 = t_edge(d, vv, go_flag, RW)
                late = sim.frame > LATE
                lax_ = wait_n >= WN1 or late
                ok = pdb.safe(ga, gb, sim.frame, te + u0, te + u1, B, lax_)
                if (not ok) and go_flag and vv >= 0.2:
                    ok = pdb.safe(ga, gb, sim.frame, te + u0 - 8, te + u1 + 10, B, lax_)
                if ok and nxt is not None and wait_n < WN2 and not late and not (go_flag and vv >= 0.5):
                    t2 = te + nxt[2] / 2.8
                    ok = pdb.safe(nxt[0], nxt[1], sim.frame, t2 - 2, t2 + 20, B, lax_, True)
                if ok and flames and wait_n < 450 and sim.frame < 4200:
                    sv = max(vv, 2.2)
                    for hq_ in (4,):
                        if flame_hit(flames, x0, x1, feet, sv, hq_, 64, 2, int(max(d - 4, 0) / sv)) is not None:
                            ok = False
                            break
                if backoff:
                    keys = LEFT if bk_ok else NOP
                    why = "backoff"
                elif go_flag and rise >= 34 and vv > 1.2 and d <= 70:
                    keys = RUN
                    why = "commit-run d%d" % d
                elif ok and (not go_flag) and vv <= 2.1 and d < RW and bk_ok and (d < 10 or RW >= 50):
                    keys = LEFT
                    why = "runway d%d" % d
                elif ok:
                    keys = RUN
                    why = "go d%d te%.0f" % (d, te)
                    if vv <= 2.1 and d >= RW:
                        go_flag = True
                else:
                    go_flag = False
                    wait_n += 1
                    if d <= 72:
                        if vv > 0.9 and d <= 6.0 * vv * vv + 14:
                            keys = LEFT
                        elif d < RW + 2:
                            keys = LEFT if (bk_ok and vv <= 0.9) else NOP
                        else:
                            keys = NOP
                        why = "hold d%d" % d
                    else:
                        keys = WALK
                        why = "approach"
            if slide < -0.8 and not why.startswith("JUMP"):
                keys = WALK if (not bk_ok or slide < -1.3) else keys
                if keys is WALK:
                    why = "stop-slide"
            if FWD and flames and vv >= 0.8 and (not want) and not why.startswith("JUMP"):
                # running into a ground-lane flame: jump forward over it (timed so the arc clears the flame)
                fr = flame_hit(flames, x0, x1, feet, vv, None, 40, 3)
                if fr is not None and fr <= 30:
                    vj = 1.55 if vv < 2.2 else vv
                    for h_ in (6, 8, 10, 4, 14):
                        dd_ = air_dist(h_, vv, 'R', 0)
                        if gx0 is not None and d < dd_ + 10:
                            continue
                        if wall_ahead_f(gray, x1, base):
                            continue
                        if flame_hit(flames, x0, x1, feet, vj, h_, 44, 3, 0) is None:
                            jt = 0
                            H = h_
                            jkeys = RJ
                            keys = RJ
                            airL = 0
                            jinfo = (wx, round(vraw, 2), h_, d)
                            why = "JUMP flame-fwd H%d fr%d v%.1f" % (h_, fr, vv)
                            break
            runf_ok = False
            if flames and vv >= 1.5 and gx0 is not None and d < 120 and not want:
                # running at a gap: the flame reaches us (relative speed ~ v+1.45) only after the take-off edge -> keep running,
                # the gap-jump selection checks the flame against the whole arc
                fr_ = flame_hit(flames, x0, x1, feet, vv, None, 40, 2)
                t_e = max(d, 0) / max(vv, 1.0)
                runf_ok = fr_ is None or fr_ >= t_e - 1
            if fh is not None and (not want or not go_ok) and fh <= 45 and not why.startswith("JUMP") and not runf_ok:
                best = None
                drift_ok = (gx0 is None or d >= 40) and vv < 0.7
                for td_ in range(0, min(fh, 34) + 1):
                    for h_ in (100, 26):
                        if flame_hit(flames, x0, x1, feet, 0.9 if drift_ok else 0.0, h_, 80, 2, td_, 12.0 if drift_ok else None) is None:
                            best = (td_, 10 if h_ == 100 else 26)
                            break
                    if best:
                        break
                if vv > 0.7 and fh > 14:
                    keys = LEFT
                    why = "fl-brake fh%d" % fh
                elif best is not None and best[0] == 0:
                    jkeys = frozenset({"z", "Right"}) if drift_ok else frozenset({"z"})
                    airL = 40
                    jt = 0
                    H = best[1]
                    keys = jkeys
                    why = "flame-hop fh%d H%d" % (fh, H)
                elif best is None and fh <= 6:
                    jkeys = frozenset({"z"})
                    airL = 40
                    jt = 0
                    H = 26
                    keys = jkeys
                    why = "flame-hop! fh%d" % fh
                else:
                    keys = NOP
                    why = "fl-wait fh%d td%s" % (fh, None if best is None else best[0])
        if LOGF and (why.startswith("JUMP") or why.startswith("flame") or n % 10 == 0):
            ents = [(int(e["x"] - sim.W), int(e["y"]), "H" if e["hidf"] is not None else ("%.1f" % e["vy"] if e["vy"] is not None else "?")) for e in pdb.d if -30 < e["x"] - sim.W < 290]
            print("P", n, "wx", wx, "x", x0, "f", feet, "v%.1f" % v, why, "".join(sorted(k[0] for k in keys)), "pods", ents, "hid", pdb.hid,
                  "F", [(f[0], f[2], round(f[4], 1)) for f in flames])
        if "Left" in keys:
            last_left_n = n
        fprev2 = feet
        sim.step(keys)
'''


GATE_DEFAULT = {'JD': 52, 'HOLD': 8, 'W': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 21, 22, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 168, 171, 172, 173, 177, 178, 179]}

B1 = (458, "stand")
PED = [(905, "ground"), (1080, "ground"), (1335, "ground")]
B4 = (1778, "ground")
LEARN_FLOOR = 2200

_MODS = {}


def mods():
    if not _MODS:
        _MODS["L0"] = _mod("L0", L0_SRC)
        _MODS["L1"] = _mod("L1", L1_SRC)
        _MODS["L4"] = _mod("L4", L4_SRC)
        _MODS["P"] = _mod("P", P_SRC)
    return _MODS


def run_leg(tr, name, lim):
    M = mods()
    if lim:
        tr.new_leg(lim[0], 60, lim[1])
    else:
        tr.new_leg(None)
    try:
        M[name].play(tr)
        print("leg", name, "returned at step", tr.nsteps, "wx", tr.wx)
    except BudgetExceededLeg:
        print("leg", name, "handoff at step", tr.nsteps, "wx", tr.wx, "W", tr.W)
        return True
    return False


def run_ped(tr, k, gates):
    g = gates.get("g%d" % (k + 1))
    if k == 0 and not g:
        g = GATE_DEFAULT
    if not g or not g.get("W"):
        print("ped", k + 1, "no gate -> go immediately")
        g = {"JD": 52, "HOLD": 8, "W": None}
    tr.mem = {"gate": g}
    tr.handoff = k > 0
    tr.skip = False
    r = run_leg(tr, "L1", PED[k])
    tr.mem = None
    tr.handoff = False
    return r


def chain(sim, gates):
    tr = Track(sim)
    run_leg(tr, "L0", B1)
    for k in range(3):
        if sim.budget_left <= 0:
            return tr
        run_ped(tr, k, gates)
    for name, lim in (("P", B4), ("L4", None)):
        if sim.budget_left <= 0:
            break
        run_leg(tr, name, lim)
    return tr


def learn_here(sim, tr, L1, JD=52, HOLD=8):
    sim.save(3)
    W0 = tr.W
    trials = []
    t = 0
    sim.save(2)
    while t < 175 and sim.budget_left > LEARN_FLOOR:
        s = sim.screen()
        mp0 = L1.mario_pos(s)
        b = L1.block_info(s, mp0[0] if mp0 else None)
        th = L1.arm_angle(s, b[0], b[1]) if b else None
        if th is not None:
            n, r = L1.dash(sim, JD, HOLD, maxf=120, check=L1.dash_check)
            trials.append((round(th, 1), 1 if r == "ok" else 0, n))
            sim.load(2)
        for _ in range(3):
            sim.step(L1.NOP)
        sim.save(2)
        t += 3
    sim.load(3)
    tr.resync(W0)
    good = L1.build_good(trials)
    print("learn: trials", len(trials), "ok", sum(x[1] for x in trials), "good bins", len(good), "left", sim.budget_left)
    return {"JD": JD, "HOLD": HOLD, "W": sorted(good)}, trials


def solve(sim):
    mem = sim.memory
    try:
        have = dict(mem.get("gates") or {})
    except Exception:
        have = {}
    if all(("g%d" % i) in have and have["g%d" % i].get("W") for i in (1, 2, 3)):
        print("solve: gates already in memory, skipping learning", {k: len(v["W"]) for k, v in have.items()})
        return []
    M = mods()
    L1 = M["L1"]
    gates = dict(have)
    try:
        sim.reset()
        tr = Track(sim)
        run_leg(tr, "L0", B1)
        for k in range(3):
            tr.new_leg(None)
            tr.handoff = k > 0
            tr.skip = False
            tr.mem = None
            st = {"f": 0, "still": 0, "airhops": 0, "prev_fy": None, "handoff": k > 0}
            ok = L1.stage_a(tr, st)
            n, ok2 = L1.to_wait_state(tr)
            print("ped", k + 1, "gate reached", ok, st["f"], "wait", ok2, n, "wx", tr.wx, "step", tr.nsteps)
            if not ok or not ok2:
                break
            if ("g%d" % (k + 1)) in gates and gates["g%d" % (k + 1)].get("W"):
                g = gates["g%d" % (k + 1)]
            else:
                cfgs = [(52, 8)] if k < 2 else [(52, 8), (58, 8)]
                best = None
                for (JD, HOLD) in cfgs:
                    if best is not None and sim.budget_left < LEARN_FLOOR + 3700:
                        break
                    g1, trials = learn_here(sim, tr, L1, JD, HOLD)
                    print("ped", k + 1, "cfg", JD, HOLD, "bins", len(g1["W"]))
                    if best is None or len(g1["W"]) > len(best["W"]):
                        best = g1
                g = best
            gates["g%d" % (k + 1)] = g
            mem["gates"] = gates
            tr.mem = {"gate": g}
            tr.skip = True
            tr.handoff = k > 0
            run_leg(tr, "L1", PED[k])
            tr.skip = False
            tr.mem = None
    except Exception as e:
        print("solve exception", type(e).__name__, e)
    mem["gates"] = gates
    try:
        mem.flush()
    except Exception:
        pass
    print("solve done, budget left", sim.budget_left, "gates", {k: len(v["W"]) for k, v in gates.items()})
    return []


def play(sim):
    try:
        gates = dict(sim.memory.get("gates") or {})
    except Exception:
        gates = {}
    f0 = getattr(sim, "frame", 0)
    try:
        s0 = sim.screen()
        ml0, fy0 = mario_left(s0)
        dig = (ml0, fy0)
        at_start = ml0 is not None and 36 <= ml0 <= 48 and 108 <= fy0 <= 118
        if (not at_start) and f0 == 0:
            f0 = 1000      # Mario not at the segment-start spot: started mid-level (practice checkpoint)
    except Exception:
        dig = -1
    print("play start frame", f0, "digits", dig)
    try:
        if f0 > 100:
            # practice mode (mid-level start): pillar hopper until Bowser appears, then Bowser leg
            M = mods()
            M["P"].EXIT_BOWSER = True
            tr = Track(sim)
            run_leg(tr, "P", None)
            run_leg(tr, "L4", None)
        else:
            chain(sim, gates)
    except Exception as e:
        print("play end:", type(e).__name__, e)

