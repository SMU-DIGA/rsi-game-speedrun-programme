# ==================================================================================================
# Super Mario Bros. (NES) -- level 3-3  |  超级马里奥兄弟（NES） 关 3-3
# Pass rate on 100 random variants never used before: 72/100; fastest finish 991 frames (16.52 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 72/100；最快一局 991 帧（60 帧/秒，约 16.52 秒）。
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
#   reactive vision policy + best-first save/load DFS over perturbation windows (score = progress 
#   - V*frames, branch&bound vs stored incumbent trace); play() replays the best stored trace with
#    resync, fal
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def n_for_apex                     line 18    
#   def n_for_stand                    line 30    
#   def n_for_range                    line 37    
#   def _eq                            line 44    
#   def find_mario                     line 48    
#   def lift_mask                      line 74    Moving lift platforms: wide bars made of dithered red/tan pixels.
#   class Percept                      line 97    
#   def percept                        line 101   
#   def scroll_est                     line 130   
#   def enemy_feature                  line 151   horizontal distance (px, clipped) from Mario's right edge to the nearest enemy pixel in front of him
#   class Policy                       line 163   Stateful closed-loop controller. act(screen) -> set of keys.
#   def _build_cands                   line 463   
#   def encode_keys                    line 503   
#   def decode_keys                    line 510   
#   def solve_dfs                      line 519   
#   def play                           line 713   
#   def run_win                        line 811   
#   def _clear                         line 834   
#   def macro                          line 855   
#   def beam_section                   line 873   Beam search over macro actions from the current state (incumbent index i0) looking for a
#   def solve                          line 972   Time-synchronous beam over macro actions. Baseline action = closed-loop Policy(KNOWN); in slow
#   def _match_states                  line 1153  strict equivalence of live state with incumbent states lo..hi -> index array (absolute)
#   def solve_v2                       line 1164  
#   def solve_v1                       line 1316  
#   def solve_pass                     line 1452  
#
# This policy also reads sim.memory; that data is in 3-3.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 3-3.memory.json。
# ==================================================================================================

# idea: reactive vision policy + best-first save/load DFS over perturbation windows (score = progress - V*frames, branch&bound vs stored incumbent trace); play() replays the best stored trace with resync, falling back to the policy
import copy
import time
import numpy as np

RED = (248, 56, 0)
BROWN = (172, 124, 0)
TAN = (252, 160, 68)
GREEN = (184, 248, 24)      # mushroom-tree cap (solid)
HL = (240, 208, 176)        # brick highlight / goomba feet / rope
KGREEN = (0, 168, 0)        # koopa
ORANGE = (228, 92, 16)      # brick / trunk (trunk is scenery!)

APEX = [(2, 34), (4, 42), (6, 49), (8, 56), (10, 62), (12, 67), (14, 71), (17, 77), (20, 81), (30, 86)]
RANGE = [(2, 63), (4, 69), (6, 81), (8, 87), (10, 96), (12, 102), (14, 111), (17, 120), (20, 129), (30, 140)]


def n_for_apex(a):
    for n, ap in APEX:
        if ap >= a:
            return n
    return 40


USE_BRAKE = False
USE_LIFTWALL = True
STAND = [(8, 44), (12, 52), (16, 56), (20, 58), (30, 62)]


def n_for_stand(a):
    for n, ap in STAND:
        if ap >= a:
            return n
    return 30


def n_for_range(r):
    for n, rg in RANGE:
        if rg >= r:
            return max(n, 4)
    return 30


def _eq(s, c):
    return (s[..., 0] == c[0]) & (s[..., 1] == c[1]) & (s[..., 2] == c[2])


def find_mario(s, prev):
    if prev is None:
        xa, xb, ya, yb = 0, 256, 32, 240
    else:
        cx = (prev[0] + prev[1]) // 2
        cy = (prev[2] + prev[3]) // 2
        xa, xb = max(0, cx - 40), min(256, cx + 41)
        ya, yb = max(32, cy - 56), min(240, cy + 57)
    sub = s[ya:yb, xa:xb]
    ys, xs = np.nonzero(_eq(sub, BROWN))
    if len(xs) < 2:
        return None
    mx = int(np.median(xs))
    keep = np.abs(xs - mx) <= 10
    xs, ys = xs[keep], ys[keep]
    if len(xs) < 2:
        return None
    bx0, bx1 = max(0, xs.min() - 3), min(sub.shape[1], xs.max() + 4)
    by0, by1 = max(0, ys.min() - 3), min(sub.shape[0], ys.max() + 4)
    reg = sub[by0:by1, bx0:bx1]
    mm = _eq(reg, BROWN) | _eq(reg, RED) | _eq(reg, TAN)
    ys, xs = np.nonzero(mm)
    return (int(xs.min()) + bx0 + xa, int(xs.max()) + bx0 + xa,
            int(ys.min()) + by0 + ya, int(ys.max()) + by0 + ya)


def lift_mask(rt):
    """Moving lift platforms: wide bars made of dithered red/tan pixels."""
    redm = rt[1]
    rt = rt[0].astype(np.int16)
    rt[:32] = 0
    rd = redm.astype(np.int16)
    rd[:32] = 0
    cs = np.concatenate([np.zeros((rt.shape[0], 1), np.int16), np.cumsum(rt, axis=1, dtype=np.int16)], axis=1)
    cr = np.concatenate([np.zeros((rd.shape[0], 1), np.int16), np.cumsum(rd, axis=1, dtype=np.int16)], axis=1)
    W = 32
    win = cs[:, W:] - cs[:, :-W]
    winr = cr[:, W:] - cr[:, :-W]
    hit = ((win >= 17) & (winr >= 8)).astype(np.int16)
    ch = np.concatenate([np.zeros((hit.shape[0], 1), np.int16), np.cumsum(hit, axis=1, dtype=np.int16)], axis=1)
    n = hit.shape[1]
    xs = np.arange(256)
    A = np.maximum(0, xs - W + 1)
    B = np.minimum(n - 1, xs)
    valid = B >= A
    L = ((ch[:, B + 1] - ch[:, A]) > 0) & valid[None, :]
    return L & (rt > 0)


class Percept:
    pass


def percept(s):
    p = Percept()
    p.green = _eq(s, GREEN)
    hl = _eq(s, HL)
    red = _eq(s, RED)
    tan = _eq(s, TAN)
    kg = _eq(s, KGREEN)
    orange = _eq(s, ORANGE)
    p.lift = lift_mask((red | tan, red))
    c = np.concatenate([np.zeros((1, 256), np.int16), np.cumsum(hl.astype(np.int16), axis=0, dtype=np.int16)], axis=0)
    H = 240
    ys = np.arange(H)
    a = np.maximum(0, ys - 8)
    b = np.minimum(H, ys + 9)
    cnt = c[b] - c[a]
    rope = hl & (cnt >= 14)
    hlm = hl & ~rope
    g = p.green | hlm | p.lift
    brickrows = np.zeros((H, 1), bool)
    brickrows[207:] = True
    g = g | (orange & brickrows)
    g[:32] = False
    p.ground = g
    en = hlm | (red & ~p.lift) | kg
    en[:44] = False
    p.enemy = en
    return p


def scroll_est(prev, cur):
    a = prev[40::2, :, :].astype(np.int16).sum(axis=2)
    b = cur[40::2, :, :].astype(np.int16).sum(axis=2)
    nb = a.shape[0] // 8
    a = a[:nb * 8]
    b = b[:nb * 8]
    D = np.zeros((6, nb))
    for k in range(0, 6):
        d = np.abs(b[:, :256 - k] - a[:, k:])
        for j in range(nb):
            D[k, j] = d[j * 8:(j + 1) * 8].mean()
    act = (a.reshape(nb, 8, 256).max(axis=(1, 2)) > 60) | (b.reshape(nb, 8, 256).max(axis=(1, 2)) > 60)
    votes = []
    for j in range(nb):
        if act[j] and D[:, j].max() - D[:, j].min() > 1.0:
            votes.append(int(np.argmin(D[:, j])))
    if not votes:
        return 0
    return int(np.median(votes))


def enemy_feature(en, x0, x1, y0, y1):
    """horizontal distance (px, clipped) from Mario's right edge to the nearest enemy pixel in front of him"""
    ya, yb = max(44, y0 - 16), min(240, y1 + 8)
    xa, xb = min(255, x1 + 2), min(256, x1 + 90)
    if yb <= ya or xb <= xa:
        return 99
    cols = np.nonzero(en[ya:yb, xa:xb].any(axis=0))[0]
    if not len(cols):
        return 99
    return int(max(-12, min(90, int(cols[0]) + xa - x1)))


class Policy:
    """Stateful closed-loop controller. act(screen) -> set of keys."""
    efeat = 99

    def __init__(self, overrides=None):
        self.prev = None
        self.jump_t = None
        self.hold_n = 10
        self.fr = 0
        self.prev_y1 = None
        self.prev_s = None
        self.prev_cx = None
        self.wx = 0.0
        self.v = 0.0
        self.lost = 0
        self.y1 = 0
        self.on_ground = True
        self.pert = {}
        self.jn_pending = None
        self.trig_wait = None
        self.brake_left = None
        self.pend_hold = None
        self.overrides = overrides or []
        self.active_ov = None
        self.verbose = True
        self.rec = []

    def set_pert(self, p):
        self.pert = dict(p) if p else {}
        self.jn_pending = self.pert.get("jumpnow")
        self.trig_wait = None
        self.brake_left = None
        self.script = self.pert.get("script")
        self.sp = 0
        self.sc = 0

    script = None
    sp = 0
    sc = 0
    takeoff_y = -999

    def _start_jump(self, hold, keys):
        self.jump_t = 0
        self.hold_n = max(2, int(hold))
        self.takeoff_y = self.y1
        keys.add("z")

    def act(self, s):
        self.fr += 1
        fr = self.fr
        base = {"Right", "x"}
        mb = find_mario(s, self.prev)
        if mb is None and self.prev is not None:
            mb = find_mario(s, None)
        if mb is None:
            self.lost += 1
            self.prev = None
            self.jump_t = None
            self.prev_y1 = None
            self.prev_s = s.copy()
            self.prev_cx = None
            return set(base)
        self.lost = 0
        self.prev = mb
        x0, x1, y0, y1 = mb
        self.y1 = y1
        cxm = (x0 + x1) / 2.0
        if self.prev_s is not None and self.prev_cx is not None:
            dx = scroll_est(self.prev_s, s) + (cxm - self.prev_cx)
            self.wx += dx
            self.v = dx
        self.prev_s = s.copy()
        self.prev_cx = cxm
        if self.overrides:
            hit = None
            for i, ov in enumerate(self.overrides):
                if ov[0] - 3 <= self.wx <= ov[1]:
                    hit = i
                    break
            if hit is not None and hit != self.active_ov and not (self.on_ground and self.jump_t is None):
                hit = self.active_ov
            if hit != self.active_ov:
                self.active_ov = hit
                self.set_pert(self.overrides[hit][2] if hit is not None else {})
        pert = self.pert
        P = percept(s)
        self.efeat = enemy_feature(P.enemy, x0, x1, y0, y1)
        G = P.ground
        vy = 0 if self.prev_y1 is None else y1 - self.prev_y1
        self.prev_y1 = y1
        ya_, yb_ = min(239, y1 + 1), min(240, y1 + 4)
        on_ground = bool(G[ya_:yb_, max(0, x0):x1 + 1].any()) and vy >= -2
        self.on_ground = on_ground
        if self.script is not None and self.sp < len(self.script):
            kind, n = self.script[self.sp]
            if kind == "L":
                sk = {"Left"}
            elif kind == "R":
                sk = {"Right", "x"}
            elif kind == "J":
                sk = {"Right", "x", "z"}
            else:
                sk = set()
            if kind == "J" and self.sc == 0:
                self.takeoff_y = y1
            self.sc += 1
            if self.sc >= n:
                self.sp += 1
                self.sc = 0
                if kind == "J":
                    self.jump_t = 4
                    self.hold_n = 0
            return sk
        keys = set(base)
        trigger = False
        tgt = None
        gapN = None
        gap = False
        farenemy = False
        wall_mode = None
        if x1 < 236 and fr >= 22:
            gap = not G[ya_:min(240, y1 + 4), x1 + 2:min(256, x1 + 6)].any()
            if gap:
                seg = G[max(32, y1 - 4):min(240, y1 + 75), min(255, x1 + 4):min(256, x1 + 130)]
                cols = np.nonzero(seg.any(axis=0))[0]
                if len(cols):
                    gx0 = int(cols[0])
                    top = int(np.nonzero(seg[:, gx0])[0][0]) + max(32, y1 - 4)
                    d = top - (y1 + 1)
                    run = 0
                    while gx0 + run < seg.shape[1] and seg[:, gx0 + run].any():
                        run += 1
                    if d >= 6:
                        reach = 3.0 * (5.0 * d) ** 0.5
                        if gx0 + 4 <= 0.8 * reach:
                            gap = False
                    gapN = n_for_range(gx0 + 4 + min(run, 24) * 0.5 + 12)
                    if d < -6:
                        gapN = max(4, n_for_apex(-d + 8), n_for_range((gx0 + 16) / 0.78))
                else:
                    gapN = 12
                if pert.get("nojump"):
                    gap = False
            WM = P.green
            zone = WM[max(32, y1 - 70):max(33, y1 - 2), x1 + 3:min(256, x1 + 64)]
            wall = False
            ceil = False
            ab = WM[max(32, y0 - 40):max(33, y0 - 1), max(0, x0 - 2):x1 + 3]
            if ab.size and ab.any():
                ceil = True
            if zone.size:
                cols = np.nonzero(zone.any(axis=0))[0]
                if len(cols):
                    c0 = int(cols[0])
                    rows = np.nonzero(zone[:, c0:c0 + 6].any(axis=1))[0]
                    top = int(rows[0]) + max(32, y1 - 70)
                    h = y1 - top
                    need = min(56, max(18, 0.75 * h + 5)) * pert.get("early", 1.0)
                    if c0 <= need and not ceil and not USE_BRAKE:
                        wall = True
                        tgt = top
                    elif c0 <= need and not ceil and h >= 14:
                        if self.v > 1.5:
                            if c0 >= 0.65 * need or h < 20:
                                wall = True
                                tgt = top
                            else:
                                wall_mode = "brake"
                        elif c0 <= 14:
                            if c0 >= 5:
                                wall_mode = "stand"
                                tgt = top
                            else:
                                wall_mode = "back"
            if not wall and USE_LIFTWALL:
                lz = P.lift[max(32, y1 - 70):max(33, y1 - 2), x1 + 3:min(256, x1 + 64)]
                lab = P.lift[max(32, y0 - 40):max(33, y0 - 1), max(0, x0 - 2):x1 + 3]
                if lz.size and lz.any() and lab.size and lab.any() and on_ground:
                    wall_mode = "back"
                elif lz.size and lz.any() and not (lab.size and lab.any()):
                    cols = np.nonzero(lz.any(axis=0))[0]
                    c0 = int(cols[0])
                    rows = np.nonzero(lz[:, c0:c0 + 6].any(axis=1))[0]
                    top = int(rows[0]) + max(32, y1 - 70)
                    h = y1 - top
                    need = min(56, max(18, 0.75 * h + 5))
                    if h >= 20 and c0 <= need + 6:
                        if self.v > 1.5:
                            if c0 >= 0.65 * need:
                                wall = True
                                tgt = top
                            else:
                                wall_mode = "brake"
                        elif c0 < 5:
                            wall_mode = "back"
                        elif c0 <= 18:
                            tgt = top
                            if abs(self.v) < 0.5:
                                wall_mode = "stand"
                            elif self.v < 0:
                                wall_mode = "settleR"
                            else:
                                wall_mode = "brake"
            ea_, eb_ = max(32, y1 - 14), max(33, y1 - 2)
            edist = int(28 * pert.get("early", 1.0))
            en = bool(P.enemy[ea_:eb_, x1 + 4:min(256, x1 + 4 + edist)].any()) and not pert.get("nojump")
            trigger = gap or wall or en
            if tgt is not None:
                r0, r1 = max(32, tgt - 22), max(33, tgt - 1)
            else:
                r0, r1 = max(32, y1 - 18), max(33, y1 - 2)
            farenemy = bool(P.enemy[r0:r1, x1 + 4:min(256, x1 + 110)].any())
        hold_add = pert.get("hold", 0)

        def compute_hold():
            if tgt is not None:
                hn = n_for_apex(y1 - tgt + 6)
            elif gap and gapN is not None:
                hn = gapN
            else:
                hn = 8
            if farenemy:
                hn = max(hn, 12)
            return hn + hold_add

        if self.jump_t is not None:
            if on_ground and self.jump_t >= 3:
                self.jump_t = None
            else:
                self.jump_t += 1
                if self.jump_t <= self.hold_n:
                    keys.add("z")
        elif self.brake_left is not None:
            if self.brake_left > 0:
                self.brake_left -= 1
                keys.discard("Right")
                keys.discard("x")
                keys.add("Left")
            else:
                self.brake_left = None
                self._start_jump(self.pend_hold if self.pend_hold is not None else 8, keys)
        elif on_ground and self.jn_pending:
            self._start_jump(self.jn_pending, keys)
            self.jn_pending = None
        elif on_ground and wall_mode == "stand":
            self._start_jump(n_for_stand(y1 - tgt + 5) + hold_add, keys)
        elif on_ground and wall_mode in ("brake", "back"):
            keys.discard("Right")
            keys.discard("x")
            keys.add("Left")
        elif on_ground and wall_mode == "settleR":
            keys.discard("x")
        elif on_ground and trigger:
            if self.trig_wait is None:
                self.trig_wait = int(pert.get("delay", 0))
            if self.trig_wait > 0:
                self.trig_wait -= 1
            else:
                hn = compute_hold()
                self.trig_wait = None
                if pert.get("brake"):
                    self.pend_hold = hn
                    self.brake_left = int(pert["brake"]) - 1
                    keys.discard("Right")
                    keys.discard("x")
                    keys.add("Left")
                else:
                    self._start_jump(hn, keys)
        elif not trigger:
            self.trig_wait = None
        # air steering while descending
        if not on_ground and vy > 0 and fr >= 22:
            mcols = slice(max(0, x0 + 2), max(1, x1 - 1))
            below = G[min(239, y1 + 1):min(240, y1 + 70), mcols]
            rows_ = np.nonzero(below.any(axis=1))[0]
            if len(rows_):
                lt = int(rows_[0]) + y1 + 1
                ahead = G[max(32, lt - 4):lt + 6, min(255, x1 + 4):min(256, x1 + 12)].any()
                same_level = abs((lt - 1) - self.takeoff_y) <= 4
                if not ahead and lt - y1 < 40 and not same_level:
                    keys.discard("Right")
                    keys.discard("x")
                    keys.add("Left")
            else:
                behind = G[min(239, y1 + 1):min(240, y1 + 70), max(0, x0 - 60):max(1, x0 - 2)].any()
                ahead = G[min(239, y1 + 1):min(240, y1 + 70), min(255, x1 + 2):min(256, x1 + 80)].any()
                if behind and not ahead:
                    keys.discard("Right")
                    keys.add("Left")
        return keys


WIN = 60
MINPROG = 25
VREF = 3.0          # px/frame used to turn progress into a time-equivalent score
VBOUND = 3.3        # optimistic speed for branch-and-bound
NKEEP = 5           # feasible candidates to compare per node
TIER = 11           # evaluate at least this many cands before stopping on NKEEP


def _build_cands():
    c = [
        {},
        {"hold": 4}, {"hold": -3}, {"hold": 8}, {"hold": -5},
        {"delay": 3}, {"delay": 6}, {"delay": 10},
        {"early": 1.4}, {"early": 0.7},
        {"nojump": True},
        {"jumpnow": 6}, {"jumpnow": 10}, {"jumpnow": 14}, {"jumpnow": 20},
    ]
    for b in (3, 4, 5, 6, 7, 8, 9, 10, 12):
        for n in (14, 18, 26):
            c.append({"script": [["L", b], ["J", n]]})
    c += [
        {"brake": 8}, {"brake": 14}, {"brake": 8, "hold": 6}, {"brake": 14, "hold": 6},
        {"delay": 3, "hold": 4}, {"delay": 6, "hold": 4}, {"delay": 6, "hold": -3},
    ]
    for B in (10, 20, 30):
        c.append({"script": [["L", B]]})
    for W in (12, 30):
        c.append({"script": [["W", W]]})
    for B in (12, 24, 36):
        for R in (0, 10, 20):
            for N in (10, 16, 24):
                sc = []
                if B:
                    sc.append(["L", B])
                if R:
                    sc.append(["R", R])
                sc.append(["J", N])
                c.append({"script": sc})
    return c


CANDS = _build_cands()
NKEEP = 6
TIER = 9

KEYBITS = {"Left": 1, "Right": 2, "Up": 4, "Down": 8, "z": 16, "x": 32}


def encode_keys(ks):
    m = 0
    for k in ks:
        m |= KEYBITS.get(k, 0)
    return m


def decode_keys(m):
    return {k for k, b in KEYBITS.items() if m & b}


KNOWN = [[1351.5, 1377.0, {"script": [["L", 24], ["R", 10], ["J", 10]]}], [1428.0, 1455.5, {"jumpnow": 6}],
         [1488.0, 1525.5, {"script": [["W", 30]]}], [1525.5, 1674.0, {"hold": 4}], [1674.0, 1848.0, {"hold": 4}],
         [1848.0, 1971.5, {"hold": 4}], [1971.5, 2145.0, {"hold": -3}], [2145.0, 2361.0, {"hold": 4}]]


def solve_dfs(sim):
    t0 = time.time()
    mem = sim.memory
    inc_frames = len(mem.get("tr_keys") or [])
    inc_wx = float(mem.get("tr_wx") or 2361.0)
    FIN_WX = inc_wx - 6.0
    best_frames = inc_frames if inc_frames > 0 else 10 ** 9
    # need to beat incumbent by a small margin
    best_frames = best_frames - 2 if inc_frames > 0 else best_frames
    print("incumbent frames", inc_frames, "wx", inc_wx, "FIN_WX", FIN_WX)
    sim.reset()
    pol = Policy()
    pol.verbose = False
    RING = [4, 5, 6, 7, 8, 9]
    ANCH = [0, 1, 2, 3]
    ring_owner = {}
    anch_owner = {}
    cnt = {"ring": 0, "anch": 0}
    stack = []
    best = {"frames": None, "keys": None, "rec": None}

    def give_ring(e):
        sl = RING[cnt["ring"] % len(RING)]
        cnt["ring"] += 1
        old = ring_owner.get(sl)
        if old is not None:
            old["slot"] = None
        ring_owner[sl] = e
        e["slot"] = sl
        sim.save(sl)

    def give_anchor(e):
        sl = ANCH[cnt["anch"] % len(ANCH)]
        cnt["anch"] += 1
        old = anch_owner.get(sl)
        if old is not None:
            old["anchor"] = None
        anch_owner[sl] = e
        e["anchor"] = sl
        sim.save(sl)

    def make_entry(p, keys_list, ov, rec):
        e = {"slot": None, "anchor": None, "pol": copy.deepcopy(p), "attempt": 0,
             "ov": ov, "keys": keys_list, "rec": list(rec or []), "ranked": None, "rpos": 0}
        stack.append(e)
        give_ring(e)
        if (len(stack) - 1) % 5 == 0:
            give_anchor(e)
        return e

    def restore(i):
        e = stack[i]
        if e["slot"] is not None:
            sim.load(e["slot"])
            return True
        j = i
        while j >= 0 and stack[j]["slot"] is None and stack[j]["anchor"] is None:
            j -= 1
        if j < 0:
            return False
        a = stack[j]
        sim.load(a["slot"] if a["slot"] is not None else a["anchor"])
        for k in range(j + 1, i + 1):
            for ks in stack[k]["keys"]:
                sim.step(ks)
        give_ring(e)
        return True

    def run_window(p, pert, n):
        p.set_pert(pert)
        p.overrides = KNOWN if pert.get("ovs") else []
        p.active_ov = None
        p.rec = []
        wx0 = p.wx
        i = 0
        kl = []
        while i < n + 60:
            if sim.budget_left < 5:
                return False, p.wx, kl
            s = sim.screen()
            keys = p.act(s)
            kl.append(frozenset(keys))
            p.rec.append((int(p.y1), int((p.prev[0] + p.prev[1]) // 2) if p.prev is not None else 0,
                          int(p.wx), int(bool(p.on_ground)), int(round(p.v))))
            sim.step(keys)
            i += 1
            if p.lost >= 3 or p.y1 >= 236:
                return False, p.wx, kl
            if i >= n and p.on_ground and p.jump_t is None and p.lost == 0:
                break
        else:
            return False, p.wx, kl
        return (p.wx - wx0) >= MINPROG, p.wx, kl

    def bound_ok(e):
        p = e["pol"]
        return p.fr + max(0.0, FIN_WX - p.wx) / VBOUND < best_frames

    def expand(idx):
        e = stack[idx]
        results = []
        for ci, pert in enumerate(CL):
            if sim.budget_left < 300 or time.time() - t0 > 1500:
                break
            if not restore(idx):
                break
            p = copy.deepcopy(e["pol"])
            ok, wxb, kl = run_window(p, pert, WIN)
            if ok:
                fr = len(kl)
                results.append((wxb - VREF * fr - (0 if ci == 0 else 6), ci))
                if ci == 0 and not any("z" in k for k in kl):
                    break   # plain running window: nothing to choose between
            if len(results) >= NKEEP and ci >= TIER:
                break
        results.sort(key=lambda r: -r[0])
        e["ranked"] = [ci for _, ci in results]
        e["rpos"] = 0
        return results

    first_log = [0]
    make_entry(pol, [], None, [])
    rounds = 0
    try:
        while sim.budget_left > 300 and time.time() - t0 < 1500 and stack:
            rounds += 1
            idx = len(stack) - 1
            top = stack[idx]
            if not bound_ok(top):
                stack.pop()
                continue
            if top["ranked"] is None:
                res = expand(idx)
                if rounds % 3 == 0 or len(res) <= 1:
                    print("expand depth", idx, "wx", int(top["pol"].wx), "fr", top["pol"].fr,
                          "ranked", [(int(sc), ci) for sc, ci in res][:6], "budget_left", sim.budget_left)
            if top["rpos"] >= len(top["ranked"]):
                stack.pop()
                continue
            ci = top["ranked"][top["rpos"]]
            top["rpos"] += 1
            if not restore(idx):
                stack.pop()
                continue
            p = copy.deepcopy(top["pol"])
            pert = CL[ci]
            wx_a = p.wx
            ok, wx_b, kl = run_window(p, pert, WIN)
            if not ok:
                continue
            if p.fr + max(0.0, FIN_WX - p.wx) / VBOUND >= best_frames:
                continue
            if pert.get("ovs"):
                ovl = [list(o) for o in KNOWN if o[1] >= wx_a - 3 and o[0] <= wx_b]
            elif pert:
                ovl = [[float(wx_a), float(wx_b), dict(pert)]]
            else:
                ovl = []
            p.overrides = []
            p.active_ov = None
            ne = make_entry(p, kl, ovl, p.rec)
            if p.wx >= FIN_WX:
                bk, br = [], []
                for e in stack:
                    bk += e["keys"]
                    br += e["rec"]
                if len(bk) < best_frames:
                    best_frames = len(bk)
                    best["frames"], best["keys"], best["rec"] = len(bk), bk, br
                    best["ov"] = [o for e in stack if e["ov"] for o in e["ov"]]
                    print("FINISH-ish path: frames", len(bk), "wx", int(p.wx), "budget_left", sim.budget_left,
                          "t", int(time.time() - t0))
                stack.pop()
    except Exception as e:
        print("solve stopped:", type(e).__name__, str(e)[:100])
    print("rounds", rounds, "best frames", best["frames"], "budget_left", sim.budget_left)
    try:
        if best["keys"]:
            bk, br = best["keys"], best["rec"]
            mem["tr_keys"] = [encode_keys(k) for k in bk]
            mem["tr_y"] = [int(r[0]) for r in br]
            mem["tr_x"] = [int(r[1]) for r in br]
            mem["tr_w"] = [int(r[2]) for r in br]
            mem["tr_g"] = [int(r[3]) for r in br]
            mem["tr_v"] = [int(r[4]) for r in br]
            mem["tr_wx"] = float(br[-1][2])
            mem["overrides"] = best.get("ov", [])
            print("stored NEW trace", len(bk), "frames, wx", br[-1][2])
        mem.flush()
    except Exception as e:
        print("memory error", e)
    return []


def play(sim):
    mem = sim.memory if hasattr(sim, "memory") else {}
    trk = mem.get("tr_keys") or []
    ty = mem.get("tr_y") or []
    tx = mem.get("tr_x") or []
    ovs = [tuple(o) for o in (mem.get("overrides") or KNOWN)]
    pol = Policy(ovs)
    pol.verbose = False
    replaying = len(trk) > 0
    diverge = 0
    i = 0
    finished = False
    resyncs = 0
    TW = np.array(mem.get("tr_w") or [0], dtype=np.int32)
    TY = np.array(ty or [0], dtype=np.int32)
    TX = np.array(tx or [0], dtype=np.int32)
    TG = np.array(mem.get("tr_g") or [0], dtype=np.int32)
    TV = np.array(mem.get("tr_v") or [0], dtype=np.int32)
    TE = np.array(mem.get("tr_e") or [], dtype=np.int32)
    have_e = len(TE) == len(trk) and len(trk) > 0
    emm = 0
    cool = 0
    print("play: stored trace frames", len(trk), "wx", mem.get("tr_wx"))
    try:
        for _ in range(sim.MAX_FRAMES):
            s = sim.screen()
            kp = pol.act(s)
            if replaying and i < len(trk):
                if pol.prev is not None and i < len(ty):
                    cx = int((pol.prev[0] + pol.prev[1]) // 2)
                    if abs(int(pol.y1) - ty[i]) > 3 or abs(cx - tx[i]) > 3 or \
                            (len(TW) == len(trk) and abs(int(pol.wx) - int(TW[i])) > 4):
                        diverge += 1
                    else:
                        diverge = 0
                keys = decode_keys(trk[i])
                if have_e and i < len(TE) and pol.prev is not None:
                    ea, eb = int(pol.efeat), int(TE[i])
                    if min(ea, eb) < 80 and abs(ea - eb) > 8:
                        emm += 1
                    else:
                        emm = 0
                    if emm >= 3:
                        replaying = False
                        emm = 0
                        cool = pol.fr + 25
                        _clear(pol)
                        keys = kp
                        print("enemy mismatch at replay frame", i, "wx", int(pol.wx), "now", ea, "rec", eb)
                i += 1
                if diverge >= 2:
                    replaying = False
                    cool = pol.fr + 12
                    _clear(pol)
                    print("replay diverged at frame", i, "wx", int(pol.wx), "y1", pol.y1)
                    keys = kp
            else:
                if replaying:
                    replaying = False
                    finished = True
                    print("replay finished at frame", i, "wx", int(pol.wx))
                keys = kp
                if (not finished) and len(trk) > 20 and pol.prev is not None and pol.on_ground \
                        and pol.jump_t is None and len(TW) == len(trk) and pol.fr >= cool and pol.efeat >= 45:
                    lo = max(0, i - 120)
                    cond = (np.abs(TW[lo:] - int(pol.wx)) <= 8) & (np.abs(TY[lo:] - int(pol.y1)) <= 2) & \
                           (TG[lo:] == 1) & (np.abs(TV[lo:] - int(round(pol.v))) <= 1) & \
                           (np.abs(np.arange(lo, len(TW)) - pol.fr) <= 80) & \
                           (np.abs(TX[lo:] - int((pol.prev[0] + pol.prev[1]) // 2)) <= 4)
                    idx = np.nonzero(cond)[0]
                    if len(idx):
                        j = int(lo + idx[np.argmin(np.abs(TW[lo + idx] - int(pol.wx)))])
                        if j + 1 < len(trk):
                            replaying = True
                            diverge = 0
                            resyncs += 1
                            keys = decode_keys(trk[j])
                            i = j + 1
                            print("re-synced to trace frame", j, "wx", int(pol.wx), "n", resyncs)
            if pol.fr % 200 == 0:
                print("f", pol.fr, "wx", int(pol.wx), "y1", pol.y1, "replaying", replaying)
            sim.step(keys)
    except Exception as e:
        print("play ended:", type(e).__name__, str(e)[:100], "frame", pol.fr)


# ---------------------------------------------------------------------------
# solve(): one pass along the stored incumbent trace; at grounded boundaries before obstacles,
# try perturbed policy windows and splice in any that reach an equivalent incumbent state earlier.
ORDER = [{"hold": 4}, {"hold": -3}, {"jumpnow": 6}, {"early": 1.4}, {"early": 0.7}, {"hold": 8},
         {"delay": 3}, {"jumpnow": 10}, {"hold": -5}, {"jumpnow": 14}, {"nojump": True}, {},
         {"script": [["W", 12]]}, {"script": [["L", 6], ["J", 18]]}, {"script": [["L", 4], ["J", 14]]},
         {"delay": 6}, {"brake": 8}]


CL = [{"ovs": True}] + [dict(c) for c in ORDER if c] + [c for c in CANDS[15:]]


def run_win(sim, p, pert, n):
    p.set_pert(pert)
    p.rec = []
    wx0 = p.wx
    i = 0
    kl = []
    while i < n + 60:
        if sim.budget_left < 5:
            return False, kl
        s = sim.screen()
        keys = p.act(s)
        kl.append(frozenset(keys))
        p.rec.append((int(p.y1), int((p.prev[0] + p.prev[1]) // 2) if p.prev is not None else 0,
                      int(p.wx), int(bool(p.on_ground)), int(round(p.v))))
        sim.step(keys)
        i += 1
        if p.lost >= 3 or p.y1 >= 236:
            return False, kl
        if i >= n and p.on_ground and p.jump_t is None and p.lost == 0:
            return (p.wx - wx0) >= MINPROG, kl
    return False, kl


def _clear(p):
    p.jump_t = None
    p.brake_left = None
    p.trig_wait = None
    p.jn_pending = None
    p.script = None
    p.pert = {}
    p.active_ov = None


_R = frozenset({"Right", "x"})
_RJ = frozenset({"Right", "x", "z"})
_LF = frozenset({"Left"})
_NO = frozenset()
_RN = frozenset({"Right"})
NA = 7
MLEN = 6
BW = 4
MING = 4


def macro(ai, t, K, i0, L):
    if ai == 0:
        return [frozenset(decode_keys(K[min(len(K) - 1, i0 + t + q)])) for q in range(L)]
    if ai == 1:
        return [_R] * L
    if ai == 2:
        return [_RJ] * L
    if ai == 3:
        return [_RJ] * 4 + [_R] * (L - 4)
    if ai == 4:
        return [_RJ] * 2 + [_R] * (L - 2)
    if ai == 5:
        return [_LF] * L
    if ai == 6:
        return [_NO] * L
    return [_RN] * L


def beam_section(sim, pol, i0, end, K, O, cap):
    """Beam search over macro actions from the current state (incumbent index i0) looking for a
    state equal to a later incumbent state reached with fewer frames."""
    OY, OX, OW, OG, OV = O
    N = len(K)
    L = MLEN
    sim.save(0)
    base = copy.deepcopy(pol)
    _clear(base)
    beam = [{"slot": 0, "pol": base, "keys": [], "rec": [], "t": 0, "m": i0}]
    pools = [[0, 1, 2], [3, 4, 5]]
    cur = 0
    best = None
    b_start = sim.budget_left
    maxlev = (end - i0) // L + 4
    stop = False
    for lev in range(maxlev):
        free = list(pools[1 - cur])
        cands = []
        for nd in beam:
            for ai in range(NA):
                if b_start - sim.budget_left > cap or sim.budget_left < 600:
                    stop = True
                    break
                sim.load(nd["slot"])
                p = copy.deepcopy(nd["pol"])
                ks = macro(ai, nd["t"], K, i0, L)
                rc = []
                alive = True
                for k in ks:
                    s = sim.screen()
                    p.act(s)
                    cxv = int((p.prev[0] + p.prev[1]) // 2) if p.prev is not None else 0
                    rc.append((int(p.y1), cxv, int(p.wx), int(bool(p.on_ground)), int(round(p.v))))
                    sim.step(k)
                    if p.lost >= 3 or p.y1 >= 236:
                        alive = False
                        break
                if not alive or p.prev is None:
                    continue
                t = nd["t"] + L
                cxv = int((p.prev[0] + p.prev[1]) // 2)
                wx = int(p.wx)
                y1 = int(p.y1)
                lo, hi = i0, min(N, i0 + t + 150)
                dist = np.abs(OW[lo:hi] - wx) + np.abs(OY[lo:hi] - y1)
                scv = np.arange(lo, hi) - 2.0 * dist
                m = float(scv.max())
                child = {"slot": None, "pol": p, "keys": nd["keys"] + ks, "rec": nd["rec"] + rc, "t": t, "m": m,
                         "sig": (wx // 4, y1 // 4, bool(p.on_ground))}
                if p.on_ground:
                    strict = ((OG[lo:hi] == 1) & (np.abs(OV[lo:hi] - int(round(p.v))) <= 1) &
                              (np.abs(OY[lo:hi] - y1) <= 1) & (np.abs(OW[lo:hi] - wx) <= 3) &
                              (np.abs(OX[lo:hi] - cxv) <= 3))
                    sidx = np.nonzero(strict)[0]
                    if len(sidx):
                        js = int(lo + sidx[-1])
                        gain = (js + 1) - (i0 + t)
                        if gain >= MING and (best is None or gain > best["gain"]):
                            sim.save(8)
                            best = {"gain": gain, "j": js, "keys": child["keys"], "rec": child["rec"], "p": p, "t": t}
                            print("  beam candidate: lev", lev, "t", t, "matches inc", js, "gain", gain)
                # insert into next beam
                same = None
                for c in cands:
                    if c["sig"] == child["sig"]:
                        same = c
                        break
                if same is not None:
                    if m > same["m"]:
                        child["slot"] = same["slot"]
                        sim.save(child["slot"])
                        cands.remove(same)
                        cands.append(child)
                elif free:
                    child["slot"] = free.pop()
                    sim.save(child["slot"])
                    cands.append(child)
                else:
                    worst = min(cands, key=lambda c: c["m"])
                    if m > worst["m"]:
                        child["slot"] = worst["slot"]
                        sim.save(child["slot"])
                        cands.remove(worst)
                        cands.append(child)
            if stop:
                break
        if stop or not cands:
            break
        cands.sort(key=lambda c: -c["m"])
        beam = cands
        cur = 1 - cur
        if best is not None and best["j"] >= end - 3:
            break
    print("  beam done: levels", lev + 1, "best", None if best is None else (best["gain"], best["j"]),
          "used", b_start - sim.budget_left)
    return best


def solve(sim):
    """Time-synchronous beam over macro actions. Baseline action = closed-loop Policy(KNOWN); in slow
    parts of the incumbent (found from its trace) extra macro actions are tried. Progress is measured as
    the index of the nearest incumbent state (position based, time independent)."""
    t0 = time.time()
    mem = sim.memory
    N = len(mem.get("tr_keys") or [])
    if N < 100 or not (mem.get("tr_y") and len(mem.get("tr_y")) == N):
        print("no stored trace")
        return []
    OY = np.array(mem["tr_y"], dtype=np.int32)
    OX = np.array(mem["tr_x"], dtype=np.int32)
    OW = np.array(mem["tr_w"], dtype=np.int32)
    END_WX = int(OW[N - 1]) - 2
    slow = np.zeros(N, bool)
    for k in range(0, N - 30):
        if OW[k + 30] - OW[k] < SLOWTH:
            slow[k:k + 30] = True
    zones = []
    k = 0
    while k < N:
        if slow[k]:
            a = k
            while k < N and slow[k]:
                k += 1
            if k - a >= 30 and a > 20:
                zones.append((int(OW[a]) - 6, int(OW[min(N - 1, k)]) + 6))
        else:
            k += 1
    print("incumbent", N, "frames; slow wx zones", zones, "slow frames", int(slow.sum()))

    def in_zone(wx):
        for a, b in zones:
            if a <= wx <= b:
                return True
        return False

    sim.reset()
    pol = Policy([tuple(o) for o in KNOWN])
    pol.verbose = False
    L = MLEN
    pools = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    sim.save(0)
    beam = [{"slot": 0, "pol": pol, "keys": [], "rec": [], "t": 0, "j": 0, "m": 0.0}]
    cur = 0
    finished = None
    maxlev = (N + 250) // L
    lean = False
    prev_beam = None
    retried = False
    nfull = 0
    for lev in range(maxlev):
        if time.time() - t0 > 1550:
            print("wall-clock stop")
            break
        jbest = max(nd["j"] for nd in beam)
        reserve = int((N - jbest) * 1.3) + 200
        if sim.budget_left < reserve + 400:
            if not lean:
                print("lean mode from lev", lev, "budget_left", sim.budget_left, "jbest", jbest)
            lean = True
        pool = pools[(cur + 1) % 3]
        free = list(pool)
        cands = []
        force_full = False
        if lean:
            beam = beam[:1]
        for nd in beam:
            full = (not lean) and (in_zone(nd["pol"].wx) or force_full)
            ais = list(range(NA)) if full else [0]
            ai_i = 0
            while ai_i < len(ais):
                ai = ais[ai_i]
                ai_i += 1
                if sim.budget_left < 30:
                    break
                sim.load(nd["slot"])
                p = copy.deepcopy(nd["pol"])
                ks = macro(ai, nd["t"], [0], 0, L) if ai else None
                rc = []
                kk = []
                alive = True
                fin_q = None
                for q in range(L):
                    s = sim.screen()
                    kp = p.act(s)
                    if p.lost >= 3 or p.y1 >= 236:
                        alive = False
                        break
                    if p.prev is not None and p.wx >= END_WX and p.on_ground and nd["t"] + q > 100:
                        fin_q = q
                        break
                    keys = kp if ai == 0 else ks[q]
                    cxv = int((p.prev[0] + p.prev[1]) // 2) if p.prev is not None else 0
                    rc.append((int(p.y1), cxv, int(p.wx), int(bool(p.on_ground)), int(round(p.v)), int(p.efeat)))
                    kk.append(frozenset(keys))
                    sim.step(keys)
                if fin_q is not None:
                    tt = nd["t"] + fin_q
                    if finished is None or tt < finished["t"]:
                        finished = {"t": tt, "keys": nd["keys"] + kk, "rec": nd["rec"] + rc}
                        print("FINISHED path at t", tt, "lev", lev, "budget_left", sim.budget_left)
                    continue
                if not alive or p.prev is None:
                    if ai == 0 and len(ais) == 1 and not lean:
                        ais = list(range(NA))   # policy action died: try all the extra macros
                        nfull += 1
                    continue
                if ai:
                    _clear(p)
                wx = int(p.wx)
                y1 = int(p.y1)
                lo, hi = max(0, nd["j"] - 30), min(N, nd["j"] + 130)
                dist = np.abs(OW[lo:hi] - wx) + np.abs(OY[lo:hi] - y1)
                scv = np.arange(lo, hi) - 2.0 * dist
                b = int(np.argmax(scv))
                child = {"slot": None, "pol": p, "keys": nd["keys"] + kk, "rec": nd["rec"] + rc,
                         "t": nd["t"] + L, "j": lo + b, "m": float(scv[b]),
                         "sig": (wx // 4, y1 // 4, bool(p.on_ground))}
                same = None
                for c in cands:
                    if c["sig"] == child["sig"]:
                        same = c
                        break
                if same is not None:
                    if child["m"] > same["m"]:
                        child["slot"] = same["slot"]
                        sim.save(child["slot"])
                        cands.remove(same)
                        cands.append(child)
                elif free:
                    child["slot"] = free.pop()
                    sim.save(child["slot"])
                    cands.append(child)
                else:
                    worst = min(cands, key=lambda c: c["m"])
                    if child["m"] > worst["m"]:
                        child["slot"] = worst["slot"]
                        sim.save(child["slot"])
                        cands.remove(worst)
                        cands.append(child)
                # a dead policy child -> also try the extra actions for this node
                if ai_i == 1 and len(ais) == 1 and False:
                    pass
            if len(ais) == 1 and not any(True for c in cands):
                pass
        if finished is not None:
            break
        if not cands:
            print("beam died at lev", lev, "t", beam[0]["t"], "wx", int(beam[0]["pol"].wx))
            break
        cands.sort(key=lambda c: -c["m"])
        prev_beam = beam
        beam = cands[:BW3]
        cur = (cur + 1) % 3
        if lev % 20 == 0:
            print("lev", lev, "t", beam[0]["t"], "wx", int(beam[0]["pol"].wx), "j", beam[0]["j"],
                  "nbeam", len(beam), "budget_left", sim.budget_left, "lean", lean)
    print("beam finished:", None if finished is None else finished["t"], "incumbent", N,
          "budget_left", sim.budget_left, "t", int(time.time() - t0))
    try:
        if finished is not None and finished["t"] <= N - 5:
            fk, fr = finished["keys"], finished["rec"]
            mem["tr_keys"] = [encode_keys(k) for k in fk]
            mem["tr_y"] = [int(r[0]) for r in fr]
            mem["tr_x"] = [int(r[1]) for r in fr]
            mem["tr_w"] = [int(r[2]) for r in fr]
            mem["tr_g"] = [int(r[3]) for r in fr]
            mem["tr_v"] = [int(r[4]) for r in fr]
            mem["tr_e"] = [int(r[5]) for r in fr]
            print("stored NEW trace", len(fk), "frames")
        mem.flush()
    except Exception as e:
        print("memory error", e)
    return []


SLOWTH = 45
BW3 = 3


def _match_states(OY, OX, OW, OG, OV, lo, hi, pol):
    """strict equivalence of live state with incumbent states lo..hi -> index array (absolute)"""
    if hi <= lo:
        return np.zeros(0, np.int64)
    cx = int((pol.prev[0] + pol.prev[1]) // 2)
    c = ((OG[lo:hi] == 1) & (np.abs(OV[lo:hi] - int(round(pol.v))) <= 1) &
         (np.abs(OY[lo:hi] - int(pol.y1)) <= 1) & (np.abs(OW[lo:hi] - int(pol.wx)) <= 3) &
         (np.abs(OX[lo:hi] - cx) <= 3))
    return lo + np.nonzero(c)[0]


def solve_v2(sim):
    t0 = time.time()
    mem = sim.memory
    trk = list(mem.get("tr_keys") or [])
    N = len(trk)
    if N < 100 or not (mem.get("tr_y") and len(mem.get("tr_y")) == N):
        print("no stored trace")
        return []
    OY = np.array(mem["tr_y"], dtype=np.int32)
    OX = np.array(mem["tr_x"], dtype=np.int32)
    OW = np.array(mem["tr_w"], dtype=np.int32)
    OG = np.array(mem["tr_g"], dtype=np.int32)
    OV = np.array(mem["tr_v"], dtype=np.int32)
    O = (OY, OX, OW, OG, OV)
    slow = np.zeros(N, bool)
    for k in range(0, N - 30):
        if OW[k + 30] - OW[k] < 50:
            slow[k:k + 30] = True
    raw = []
    k = 0
    while k < N:
        if slow[k]:
            a = k
            while k < N and slow[k]:
                k += 1
            if raw and a - raw[-1][1] < 25:
                raw[-1][1] = k
            else:
                raw.append([a, k])
        else:
            k += 1
    secs = []
    CH = 150
    for a, b in raw:
        if b - a < 40 or a <= 20:
            continue
        a0 = max(25, a - 10)
        b0 = min(N - 40, b + 10)
        while a0 < b0 - 30:
            e0 = min(b0, a0 + CH)
            secs.append((a0, e0))
            a0 = e0
    print("incumbent", N, "frames; sections", secs)
    sim.reset()
    pol = Policy([tuple(o) for o in KNOWN])
    pol.verbose = False
    out_keys = []
    out_rec = []
    i = 0
    cs = 0
    mode = "replay"
    div = 0
    aborted = False
    gains = 0
    nres = 0
    ndiv = 0
    try:
        while True:
            if sim.budget_left < 10 or time.time() - t0 > 1600:
                aborted = True
                break
            if mode == "replay" and i >= N:
                break
            if len(out_keys) > N + 150:
                aborted = True
                break
            if mode == "replay":
                while cs < len(secs) and i > secs[cs][1] - 30:
                    cs += 1
                if (cs < len(secs) and i >= secs[cs][0] and pol.prev is not None and pol.on_ground
                        and abs(pol.y1 - int(OY[i])) <= 1 and not (trk[i] & 16)):
                    s_, e_ = secs[cs]
                    rem = sum(b - a for a, b in secs[cs:])
                    avail = sim.budget_left - (N - i) - 700
                    cap = int(avail * (e_ - s_) / max(1, rem))
                    cs += 1
                    if cap >= 500:
                        print("section", s_, e_, "start at inc", i, "cap", cap, "budget_left", sim.budget_left)
                        sim.save(9)
                        best = beam_section(sim, pol, i, e_, trk, O, cap)
                        if best is not None:
                            sim.load(8)
                            out_keys += best["keys"]
                            out_rec += best["rec"]
                            pol = best["p"]
                            _clear(pol)
                            gains += best["gain"]
                            print("SPLICE inc", i, "->", best["j"] + 1, "gain", best["gain"], "total", gains)
                            i = best["j"] + 1
                            div = 0
                            continue
                        else:
                            sim.load(9)
            s = sim.screen()
            kp = pol.act(s)
            if pol.lost >= 3 or pol.y1 >= 236:
                print("died/lost at out frame", len(out_keys), "mode", mode)
                aborted = True
                break
            cx = int((pol.prev[0] + pol.prev[1]) // 2) if pol.prev is not None else 0
            if mode == "replay":
                keys = decode_keys(trk[i])
                if pol.prev is not None and (abs(int(pol.y1) - int(OY[i])) > 5 or abs(cx - int(OX[i])) > 5):
                    div += 1
                else:
                    div = 0
                if div >= 4:
                    print("replay diverged at inc", i, "out frame", len(out_keys), "-> policy mode")
                    mode = "policy"
                    ndiv += 1
                    _clear(pol)
                    keys = kp
            else:
                keys = kp
                if pol.prev is not None and pol.on_ground and pol.jump_t is None:
                    idx = _match_states(OY, OX, OW, OG, OV, max(0, i - 10), min(N, i + 60), pol)
                    if len(idx):
                        j = int(idx[np.argmin(np.abs(idx - i))])
                        mode = "replay"
                        nres += 1
                        i = j
                        div = 0
                        keys = decode_keys(trk[i])
                        print("resync to inc", j, "at out frame", len(out_keys))
                if mode == "policy" and pol.wx >= OW[N - 1] - 2 and pol.on_ground:
                    print("policy mode reached incumbent end wx at out frame", len(out_keys))
                    break
            out_keys.append(frozenset(keys))
            out_rec.append((int(pol.y1), cx, int(pol.wx), int(bool(pol.on_ground)), int(round(pol.v))))
            sim.step(keys)
            i += 1
    except Exception as e:
        print("pass stopped:", type(e).__name__, str(e)[:100], "at inc", i)
    complete = not aborted
    newN = len(out_keys)
    print("pass done: complete", complete, "new length", newN, "incumbent", N, "gains", gains, "resyncs", nres,
          "divs", ndiv, "budget_left", sim.budget_left, "t", int(time.time() - t0))
    try:
        if complete and newN <= N - 3:
            mem["tr_keys"] = [encode_keys(k) for k in out_keys]
            mem["tr_y"] = [int(r[0]) for r in out_rec]
            mem["tr_x"] = [int(r[1]) for r in out_rec]
            mem["tr_w"] = [int(r[2]) for r in out_rec]
            mem["tr_g"] = [int(r[3]) for r in out_rec]
            mem["tr_v"] = [int(r[4]) for r in out_rec]
            print("stored NEW trace", newN, "frames")
        mem.flush()
    except Exception as e:
        print("memory error", e)
    return []


def solve_v1(sim):
    t0 = time.time()
    mem = sim.memory
    trk = list(mem.get("tr_keys") or [])
    N = len(trk)
    if N < 100 or not (mem.get("tr_y") and len(mem.get("tr_y")) == N):
        print("no stored trace")
        return []
    OY = np.array(mem["tr_y"], dtype=np.int32)
    OX = np.array(mem["tr_x"], dtype=np.int32)
    OW = np.array(mem["tr_w"], dtype=np.int32)
    OG = np.array(mem["tr_g"], dtype=np.int32)
    OV = np.array(mem["tr_v"], dtype=np.int32)
    O = (OY, OX, OW, OG, OV)
    # slow sections of the incumbent
    slow = np.zeros(N, bool)
    for k in range(0, N - 30):
        if OW[k + 30] - OW[k] < 50:
            slow[k:k + 30] = True
    secs = []
    k = 0
    while k < N:
        if slow[k]:
            a = k
            while k < N and slow[k]:
                k += 1
            b = k
            if secs and a - secs[-1][1] < 25:
                secs[-1][1] = b
            else:
                secs.append([a, b])
        else:
            k += 1
    secs = [(max(25, a - 10), min(N - 40, b + 10)) for a, b in secs if b - a >= 40 and a > 20]
    print("incumbent", N, "frames; slow sections", secs)
    sim.reset()
    pol = Policy()
    pol.verbose = False
    out_keys = []
    out_rec = []
    i = 0
    cs = 0
    verify_until = -1
    pend = None
    div = 0
    aborted = False
    gains = 0
    try:
        while i < N:
            if sim.budget_left < 10 or time.time() - t0 > 1600:
                break
            while cs < len(secs) and i > secs[cs][1] - 30:
                cs += 1
            if (pend is None and cs < len(secs) and i >= secs[cs][0] and pol.prev is not None and pol.on_ground
                    and abs(pol.y1 - int(OY[i])) <= 1 and not (trk[i] & 16)):
                s_, e_ = secs[cs]
                rem = sum(b - a for a, b in secs[cs:])
                avail = sim.budget_left - (N - i) - 500
                cap = int(avail * (e_ - s_) / max(1, rem))
                cs += 1
                if cap >= 600:
                    print("section", s_, e_, "start at inc", i, "cap", cap, "budget_left", sim.budget_left)
                    sim.save(9)
                    pol0 = copy.deepcopy(pol)
                    best = beam_section(sim, pol, i, e_, trk, O, cap)
                    if best is not None:
                        sim.load(8)
                        pend = {"i0": i, "k0": len(out_keys), "pol0": pol0}
                        out_keys += best["keys"]
                        out_rec += best["rec"]
                        pol = best["p"]
                        _clear(pol)
                        gains += best["gain"]
                        print("SPLICE inc", i, "->", best["j"] + 1, "gain", best["gain"], "total", gains)
                        i = best["j"] + 1
                        verify_until = i + 45
                        div = 0
                        continue
                    else:
                        sim.load(9)
            if pend is not None and i >= verify_until:
                pend = None
            s = sim.screen()
            pol.act(s)
            keys = decode_keys(trk[i])
            cx = int((pol.prev[0] + pol.prev[1]) // 2) if pol.prev is not None else 0
            if pol.prev is not None and (abs(int(pol.y1) - int(OY[i])) > 5 or abs(cx - int(OX[i])) > 5):
                div += 1
            else:
                div = 0
            if div >= 4:
                if pend is not None:
                    print("splice verification FAILED at inc", i, "-> revert to", pend["i0"])
                    sim.load(9)
                    del out_keys[pend["k0"]:]
                    del out_rec[pend["k0"]:]
                    pol = pend["pol0"]
                    i = pend["i0"]
                    pend = None
                    div = 0
                    secs = [sc for sc in secs if sc[0] > i + 10]
                    cs = 0
                    continue
                print("incumbent replay diverged at", i, "- abort pass")
                aborted = True
                break
            out_keys.append(frozenset(keys))
            out_rec.append((int(pol.y1), cx, int(pol.wx), int(bool(pol.on_ground)), int(round(pol.v))))
            sim.step(keys)
            i += 1
    except Exception as e:
        print("pass stopped:", type(e).__name__, str(e)[:100], "at inc", i)
    complete = (i >= N) and not aborted
    if not complete and not aborted and out_keys:
        for k in range(i, N):
            out_keys.append(frozenset(decode_keys(trk[k])))
            out_rec.append((int(OY[k]), int(OX[k]), int(OW[k]), int(OG[k]), int(OV[k])))
        complete = True
    newN = len(out_keys)
    print("pass done: complete", complete, "new length", newN, "incumbent", N, "budget_left", sim.budget_left,
          "t", int(time.time() - t0))
    try:
        if complete and newN <= N - 3:
            mem["tr_keys"] = [encode_keys(k) for k in out_keys]
            mem["tr_y"] = [int(r[0]) for r in out_rec]
            mem["tr_x"] = [int(r[1]) for r in out_rec]
            mem["tr_w"] = [int(r[2]) for r in out_rec]
            mem["tr_g"] = [int(r[3]) for r in out_rec]
            mem["tr_v"] = [int(r[4]) for r in out_rec]
            print("stored NEW trace", newN, "frames")
        mem.flush()
    except Exception as e:
        print("memory error", e)
    return []


def solve_pass(sim):
    t0 = time.time()
    mem = sim.memory
    trk = list(mem.get("tr_keys") or [])
    N = len(trk)
    if N < 100 or not (mem.get("tr_y") and len(mem.get("tr_y")) == N):
        print("no stored trace")
        return []
    OY = np.array(mem["tr_y"], dtype=np.int32)
    OX = np.array(mem["tr_x"], dtype=np.int32)
    OW = np.array(mem["tr_w"], dtype=np.int32)
    OG = np.array(mem["tr_g"], dtype=np.int32)
    OV = np.array(mem["tr_v"], dtype=np.int32)
    print("incumbent", N, "frames")
    sim.reset()
    pol = Policy()
    pol.verbose = False
    out_keys = []
    out_rec = []
    i = 0
    last_b = -1000
    verify_until = -1
    pend = None
    div = 0
    banned = set()
    gain_total = 0
    aborted = False
    nb = 0
    try:
        while i < N:
            if sim.budget_left < 10 or time.time() - t0 > 1600:
                break
            # ---- boundary: try to improve the next obstacle
            if (pend is None and i - last_b >= 60 and i >= 25 and N - i > 45 and i not in banned
                    and pol.prev is not None and pol.on_ground and abs(pol.y1 - int(OY[i])) <= 1
                    and not (trk[i] & 16) and any(k & 16 for k in trk[i:i + 30])):
                last_b = i
                rem_events = max(1, (N - i) // 75)
                allow = (sim.budget_left - (N - i) - 500) / rem_events
                if allow >= 200:
                    nb += 1
                    sim.save(0)
                    base = copy.deepcopy(pol)
                    _clear(base)
                    best = None
                    spent = 0
                    for pert in ORDER:
                        if spent > allow or sim.budget_left < (N - i) + 400:
                            break
                        sim.load(0)
                        p = copy.deepcopy(base)
                        b0 = sim.budget_left
                        ok, kl = run_win(sim, p, pert, WIN)
                        spent += b0 - sim.budget_left
                        if not ok or p.prev is None:
                            continue
                        cxp = int((p.prev[0] + p.prev[1]) // 2)
                        lo, hi = i + 8, min(N, i + len(kl) + 80)
                        if hi <= lo:
                            continue
                        cond = ((OG[lo:hi] == 1) & (np.abs(OY[lo:hi] - p.y1) <= 1) &
                                (np.abs(OW[lo:hi] - int(p.wx)) <= 3) &
                                (np.abs(OV[lo:hi] - int(round(p.v))) <= 1) &
                                (np.abs(OX[lo:hi] - cxp) <= 3))
                        idx = np.nonzero(cond)[0]
                        if not len(idx):
                            continue
                        j = int(lo + idx[np.argmin(np.abs(OW[lo + idx] - int(p.wx)))])
                        gain = (j + 1 - i) - len(kl)
                        if gain >= 3 and (best is None or gain > best["gain"]):
                            sim.save(1)
                            best = {"gain": gain, "j": j, "kl": kl, "p": p, "pert": pert}
                    if best is not None:
                        sim.load(1)
                        pend = {"i0": i, "k0": len(out_keys), "pol0": base}
                        out_keys += best["kl"]
                        out_rec += best["p"].rec
                        pol = best["p"]
                        _clear(pol)
                        i = best["j"] + 1
                        verify_until = i + 45
                        last_b = i
                        div = 0
                        gain_total += best["gain"]
                        print("SPLICE at inc", pend["i0"], "->", i, "gain", best["gain"], "pert", best["pert"],
                              "total", gain_total, "budget_left", sim.budget_left)
                        continue
                    else:
                        sim.load(0)
            if pend is not None and i >= verify_until:
                pend = None
            s = sim.screen()
            pol.act(s)
            keys = decode_keys(trk[i])
            cx = int((pol.prev[0] + pol.prev[1]) // 2) if pol.prev is not None else 0
            if pol.prev is not None and (abs(int(pol.y1) - int(OY[i])) > 5 or abs(cx - int(OX[i])) > 5):
                div += 1
            else:
                div = 0
            if div >= 4:
                if pend is not None:
                    print("splice verification FAILED at inc", i, "-> revert to", pend["i0"])
                    sim.load(0)
                    del out_keys[pend["k0"]:]
                    del out_rec[pend["k0"]:]
                    pol = pend["pol0"]
                    gain_total = 0 if False else gain_total
                    banned.add(pend["i0"])
                    i = pend["i0"]
                    last_b = i
                    pend = None
                    div = 0
                    # gain accounting is recomputed from lengths at the end
                    continue
                print("incumbent replay diverged at", i, "- abort pass")
                aborted = True
                break
            out_keys.append(frozenset(keys))
            out_rec.append((int(pol.y1), cx, int(pol.wx), int(bool(pol.on_ground)), int(round(pol.v))))
            sim.step(keys)
            i += 1
    except Exception as e:
        print("pass stopped:", type(e).__name__, str(e)[:100], "at inc", i)
    complete = (i >= N) and not aborted
    if not complete and not aborted and out_keys:
        # budget ran out mid-pass: append incumbent remainder
        for k in range(i, N):
            out_keys.append(frozenset(decode_keys(trk[k])))
            out_rec.append((int(OY[k]), int(OX[k]), int(OW[k]), int(OG[k]), int(OV[k])))
        complete = True
    newN = len(out_keys)
    print("pass done: complete", complete, "new length", newN, "incumbent", N, "boundaries", nb,
          "budget_left", sim.budget_left, "t", int(time.time() - t0))
    try:
        if complete and newN <= N - 3:
            mem["tr_keys"] = [encode_keys(k) for k in out_keys]
            mem["tr_y"] = [int(r[0]) for r in out_rec]
            mem["tr_x"] = [int(r[1]) for r in out_rec]
            mem["tr_w"] = [int(r[2]) for r in out_rec]
            mem["tr_g"] = [int(r[3]) for r in out_rec]
            mem["tr_v"] = [int(r[4]) for r in out_rec]
            print("stored NEW trace", newN, "frames")
        mem.flush()
    except Exception as e:
        print("memory error", e)
    return []

