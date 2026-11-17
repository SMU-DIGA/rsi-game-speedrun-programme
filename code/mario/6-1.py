# ==================================================================================================
# Super Mario Bros. (NES) -- level 6-1  |  超级马里奥兄弟（NES） 关 6-1
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1294 frames (21.57 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1294 帧（60 帧/秒，约 21.57 秒）。
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
#   reactive vision runner (pits/blocks/enemies) + solve() does a save/load leg-by-leg search that
#    records a world-x keyed jump route, play() follows the route (reactive fallback)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _mask                          line 20    
#   def _sky                           line 24    
#   def _hold_for                      line 31    
#   def find_mario                     line 54    
#   class Ctl                          line 76    
#   def _cands                         line 254   
#   def _run_leg                       line 264   
#   def solve                          line 283   
#   def _load_route                    line 337   
#   def play                           line 353   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
#
# This policy also reads sim.memory; that data is in 6-1.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 6-1.memory.json。
# ==================================================================================================

# idea: reactive vision runner (pits/blocks/enemies) + solve() does a save/load leg-by-leg search that records a world-x keyed jump route, play() follows the route (reactive fallback)
import json
import numpy as np

SKY = ((0, 0, 0), (104, 136, 252))
ORANGE = (228, 92, 16)
TAN = (240, 208, 176)
SKIN = (252, 160, 68)
RED = (248, 56, 0)
BROWN = (172, 124, 0)

HOLD_PIT = 28
OBS_DIST = 26
PIT_DIST = 8
LEG = 90
_HT = [(0, 2), (34, 2), (42, 4), (49, 6), (56, 8), (64, 11), (71, 14), (79, 18), (85, 24), (200, 30)]
SOLID = (ORANGE, TAN, RED, SKIN)


def _mask(s, c):
    return (s[:, :, 0] == c[0]) & (s[:, :, 1] == c[1]) & (s[:, :, 2] == c[2])


def _sky(s):
    m = np.zeros(s.shape[:2], bool)
    for c in SKY:
        m |= _mask(s, c)
    return m


def _hold_for(s, cx, feet, hs=1.0):
    h = 0
    for c in range(min(255, cx), min(256, cx + 4)):
        hh = 0
        for dy in range(1, 100):
            y = feet - dy
            if y < 32:
                break
            col = (int(s[y, c, 0]), int(s[y, c, 1]), int(s[y, c, 2]))
            if col in SOLID:
                hh = dy
            elif hh and dy - hh > 3:
                break
        h = max(h, hh)
    need = h + 12
    k = HOLD_PIT
    for (a, ka), (b, kb) in zip(_HT, _HT[1:]):
        if need <= b:
            k = int(round(ka + (kb - ka) * (need - a) / max(1, b - a))) + 1
            break
    return max(2, int(round(k * hs)))


def find_mario(s, prev):
    if prev is None:
        gx0, gx1, gy0, gy1 = 0, 100, 40, 240
    else:
        px, py = prev
        gx0, gx1 = max(0, px - 24), min(256, px + 24)
        gy0, gy1 = max(32, py - 30), min(240, py + 30)
    win = s[gy0:gy1, gx0:gx1]
    sub = _mask(win, RED) | _mask(win, BROWN)
    ys, xs = np.nonzero(sub)
    if len(ys) < 4:
        return None
    ys = ys + gy0
    xs = xs + gx0
    cx = int(np.median(xs))
    cy = int(np.median(ys))
    sel = (abs(xs - cx) <= 10) & (abs(ys - cy) <= 12)
    if sel.sum() < 4:
        return None
    return (int(xs[sel].min()), int(xs[sel].max()), int(ys[sel].min()), int(ys[sel].max()))


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
        self.dead = False
        self.leg_t = 0
        self.forced_done = False
        self.pause_left = 0
        self.events = []
        self.route = route
        self.ri = 0
        self.t = 0
        self.info = ""

    def snap(self):
        d = dict(self.__dict__)
        d["last_feet"] = list(self.last_feet)
        d["events"] = list(self.events)
        return d

    def restore(self, d):
        self.__dict__.update(d)
        self.last_feet = list(d["last_feet"])
        self.events = list(d["events"])

    # ---------------- perception ----------------
    def _shift(self, a, b):
        costs = []
        for sh in range(0, 6):
            w = 256 - sh
            costs.append(int((b[209:240, :w] != a[209:240, sh:]).any(-1).sum()))
        srt = sorted(costs)
        if srt[1] - srt[0] < 12:   # ambiguous (pit / uniform ground): use the upper band too
            for sh in range(0, 6):
                w = 256 - sh
                costs[sh] += int((b[100:208, :w] != a[100:208, sh:]).any(-1).sum()) * 0.5
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
        bb = find_mario(s, self.prev)
        self.bb = bb
        if bb is None:
            self.lost += 1
            if self.lost >= 4:
                self.dead = True
            return s
        self.lost = 0
        x0, x1, y0, y1 = bb
        self.prev = ((x0 + x1) // 2, (y0 + y1) // 2)
        if y1 >= 233:
            self.dead = True
        feet = y1 + 1
        self.last_feet.append(feet)
        if len(self.last_feet) > 4:
            self.last_feet.pop(0)
        under = ~_sky(s[min(239, feet + 1): min(240, feet + 3), x0:x1 + 1])
        stable = len(self.last_feet) >= 3 and self.last_feet[-1] == self.last_feet[-2] == self.last_feet[-3]
        self.grounded = bool(under.any()) and stable
        self.wx = self.cum + (x0 + x1) // 2
        if self.wx > self.best:
            self.best = self.wx
            self.stall = 0
        else:
            self.stall += 1
        return s

    # ---------------- reactive proposal ----------------
    def reactive(self, s, P):
        x0, x1, y0, y1 = self.bb
        feet = y1 + 1
        right = x1 + 2
        sky = _sky(s)
        od = P.get("obs_dist", OBS_DIST)
        pd = P.get("pit_dist", PIT_DIST)
        hs = P.get("hs", 1.0)
        for dx in range(2, 14, 2):
            cx = right + dx
            if cx >= 256:
                break
            col = sky[min(239, feet + 1):, cx]
            if col.size and col.all():
                if dx <= pd + 2:
                    return max(2, int(round(HOLD_PIT * hs)))
                break
        bx0 = right + 1
        bx1 = min(256, right + 1 + od)
        if bx1 > bx0:
            by0 = max(32, feet - 16)
            by1 = max(by0 + 1, feet - 1)
            box = s[by0:by1, bx0:bx1]
            m = _mask(box, ORANGE) | _mask(box, TAN) | _mask(box, RED)
            if m.sum() >= 3:
                cols = np.nonzero(m.any(axis=0))[0]
                return _hold_for(s, bx0 + int(cols.min()), feet, hs)
        return 0

    # ---------------- act ----------------
    def act(self, s, P=None):
        P = P or {}
        s = self.observe(s)
        self.t += 1
        keys = {"Right", "x"}
        if self.bb is None:
            if self.hold > 0:
                keys.add("z")
                self.hold -= 1
            return keys
        # search-time pause
        pz = P.get("pause", 0)
        if pz and self.leg_t == 0:
            self.events.append(("w", self.wx, pz))
        if pz and self.leg_t < pz:
            return set()
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
        self.since += 1
        if not (self.grounded and self.since > 1):
            return keys
        hold = 0
        if self.route is not None and self.ri < len(self.route):
            while self.ri < len(self.route) and self.route[self.ri][1] < self.wx - 60:
                self.ri += 1
            if self.ri < len(self.route) and self.route[self.ri][1] <= self.wx:
                ev = self.route[self.ri]
                self.ri += 1
                if ev[0] == "w":
                    self.pause_left = int(ev[2])
                    return set()
                hold = int(ev[2])
            if self.stall > 70:
                self.route = None
        else:
            f = P.get("force")
            if f and not self.forced_done and self.leg_t >= f[0]:
                self.forced_done = True
                hold = f[1]
            else:
                hold = self.reactive(s, P)
        if hold > 0:
            self.events.append(("j", self.wx, hold))
            keys.add("z")
            self.hold = hold - 1
            if self.hold == 0:
                self.since = 0
        return keys


def _cands():
    c = [{}, {"obs_dist": 14, "pit_dist": 4}, {"obs_dist": 40, "pit_dist": 14}, {"hs": 0.6}, {"hs": 1.5}]
    for t in (0, 4, 8, 12, 16, 22, 30):
        for k in (3, 8, 14, 22, 30):
            c.append({"force": (t, k)})
    for p in (10, 25):
        c.append({"pause": p})
    return c


def _run_leg(sim, ctl, P, target, maxf=140):
    ctl.leg_t = 0
    ctl.forced_done = False
    ctl.stall = 0
    keys_log = []
    for i in range(maxf):
        keys = ctl.act(sim.screen(), P)
        sim.step(keys)
        keys_log.append(keys)
        ctl.leg_t += 1
        if ctl.dead:
            return "dead", keys_log
        if ctl.stall > 45:
            return "stall", keys_log
        if ctl.wx >= target and ctl.hold == 0 and ctl.grounded:
            return "ok", keys_log
    return "timeout", keys_log


def solve(sim):
    sim.reset()
    cands = _cands()
    ctl = Ctl()
    # warm up one frame so tracker has state
    slot = 0
    sim.save(slot)
    stack = [{"slot": 0, "snap": ctl.snap(), "wx": 0, "ci": 0, "nev": 0}]
    events = []
    best_route = []
    best_wx = 0
    end_hist = []
    try:
        while sim.budget_left > 200:
            e = stack[-1]
            if e["ci"] >= len(cands):
                if len(stack) == 1:
                    print("search exhausted at wx", e["wx"])
                    break
                stack.pop()
                continue
            P = cands[e["ci"]]
            e["ci"] += 1
            sim.load(e["slot"])
            ctl.restore(e["snap"])
            ev0 = len(ctl.events)
            st, _ = _run_leg(sim, ctl, P, e["wx"] + LEG)
            if st == "ok":
                ns = (e["slot"] + 1) % 10
                sim.save(ns)
                stack.append({"slot": ns, "snap": ctl.snap(), "wx": ctl.wx, "ci": 0})
                if len(stack) > 9:
                    stack.pop(0)
                if ctl.wx > best_wx:
                    best_wx = ctl.wx
                    best_route = [list(x) for x in ctl.events]
                    sim.memory["route"] = json.dumps(best_route)
                print("leg ok wx", ctl.wx, "cand", P, "frame", sim.frame)
            else:
                print("leg fail", st, "from wx", e["wx"], "cand", P, "->", ctl.wx)
                if st == "stall":
                    end_hist.append(ctl.wx)
                    if len(end_hist) >= 6 and len(set(end_hist[-6:])) == 1 and best_wx >= end_hist[-1] - 40:
                        print("level end reached (flag stall) at", end_hist[-1])
                        break
                else:
                    end_hist.append(-1)
    except Exception as ex:
        print("solve exception", repr(ex))
    print("best_wx", best_wx)
    print("ROUTE", json.dumps(best_route))
    return []


def _load_route(sim):
    try:
        r = sim.memory.get("route")
        if r:
            return [tuple(x) for x in json.loads(r)]
    except Exception:
        pass
    try:
        r = sim.mem.past.get("route")
        if r:
            return [tuple(x) for x in r]
    except Exception:
        pass
    return None


def play(sim):
    route = _load_route(sim)
    ctl = Ctl(route)
    frozen = 0
    for step in range(7900):
        s = sim.screen()
        if ctl.last_img is not None and np.array_equal(s[40:], ctl.last_img[40:]):
            frozen += 1
        else:
            frozen = 0
        if frozen > 90:
            break
        keys = ctl.act(s, {})
        if step % 60 == 0:
            print("f", step, "wx", ctl.wx, "bb", ctl.bb, "g", ctl.grounded, "ri", ctl.ri, "route" if ctl.route is not None else "reactive")
        sim.step(keys)
    print("end step", step, "wx", ctl.wx, "frozen", frozen)

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    st = ctx.get("stuck") if isinstance(ctx, dict) else None\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": bool(st and st.get("is"))},\n            "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {"route": [["j", 47, 11], ["j", 61, 26], ["j", 302, 28], ["j", 452, 3], ["j", 480, 3], ["j", 531, 3], ["j", 608, 28], ["j", 896, 28], ["j", 1070, 3], ["j", 1124, 3], ["j", 1146, 3], ["j", 1167, 28], ["j", 1371, 25], ["j", 1400, 3], ["j", 1412, 3], ["j", 1587, 14], ["j", 1651, 3], ["j", 1920, 3], ["j", 1974, 3], ["j", 1993, 3], ["j", 2002, 14], ["j", 2064, 28], ["j", 2254, 3], ["j", 2308, 3], ["j", 2329, 3], ["j", 2345, 3], ["j", 2359, 3], ["j", 2612, 28], ["j", 2741, 3], ["j", 2761, 3], ["j", 2771, 28], ["j", 2952, 3]]}}\n'}

