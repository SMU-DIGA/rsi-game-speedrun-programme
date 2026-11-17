# ==================================================================================================
# Super Mario Bros. (NES) -- level 7-4  |  超级马里奥兄弟（NES） 关 7-4
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1226 frames (20.43 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1226 帧（60 帧/秒，约 20.43 秒）。
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
#   c6 reactive pixel bot (run right, jump pits/walls, steer height tiers) + bridge treated as flo
#   or + Bowser dodge (run under him when he is high, hop fire)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def hold_for_height                line 22    
#   def goal_tier                      line 29    
#   def find_mario                     line 36    
#   def solid_mask                     line 45    
#   def surfaces                       line 58    tops of solid runs in column slab x2..x2+4 between rows lo..hi with free space above.
#   class Bot                          line 68    
#   def play                           line 322   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: c6 reactive pixel bot (run right, jump pits/walls, steer height tiers) + bridge treated as floor + Bowser dodge (run under him when he is high, hop fire)
import numpy as np

PIT_DEPTH = 64
WALL_MARGIN = 10
WALL_TRIG = 14
LEDGE_TRIG = 20
APEX = [(2, 34), (4, 42), (6, 49), (8, 56), (11, 64), (14, 71), (18, 79), (24, 85), (34, 90)]

# (scroll_from, scroll_to, goal feet-y)  floor=208 mid=160 upper=96 ; None = free
SCHED = [(0, 340, -1), (340, 700, 208), (700, 930, 160), (930, 1700, 96), (1700, 1960, 160), (1960, 2400, 96)]
# lava-section jump holds by scroll
LAVA_HOLDS = [(100, 200, 26), (200, 300, 34)]
BRIDGE_S = 2850      # scroll from which the bridge (red under white chain) counts as floor
BOW_SAFE = 136       # Bowser green bottom <= this: Mario (top ~147) can run under him
BOW_NEAR = 60        # wait distance when Bowser is low
FIRE_AHEAD = 36
FIRE_HOLD = 12
DEBUG = True


def hold_for_height(h):
    for k, a in APEX:
        if a >= h:
            return k
    return 34


def goal_tier(s):
    for a, b, g in SCHED:
        if a <= s < b:
            return g
    return None


def find_mario(im):
    m = (im[:, :, 0] == 172) & (im[:, :, 1] == 124) & (im[:, :, 2] == 0)
    m[:40] = False
    ys, xs = np.nonzero(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def solid_mask(im):
    r = im[:, :, 0].astype(np.int16); g = im[:, :, 1].astype(np.int16); b = im[:, :, 2].astype(np.int16)
    gray = (np.abs(r - g) < 6) & (np.abs(g - b) < 6) & (r > 100)
    e = np.zeros_like(gray)
    e[:, 2:-2] = gray[:, 2:-2] & gray[:, 1:-3] & gray[:, :-4] & gray[:, 3:-1] & gray[:, 4:]
    s = e.copy()
    for k in (1, 2):
        s[:, k:] |= e[:, :-k]
        s[:, :-k] |= e[:, k:]
    s[:40] = False
    return s


def surfaces(sup, x2, lo, hi):
    """tops of solid runs in column slab x2..x2+4 between rows lo..hi with free space above."""
    v = sup[:, x2:x2 + 4].any(axis=1)
    out = []
    for y in range(max(lo, 60), min(hi, 238)):
        if v[y] and not v[y - 1] and not v[max(41, y - 14):y].any():
            out.append(y)
    return out


class Bot:
    def __init__(self):
        self.prev = None
        self.still = 0
        self.jump_t = 0
        self.in_air = False
        self.left_ground = False
        self.fy_hist = []
        self.scroll = 0
        self.dead = False
        self.hold = 34
        self.pending = None
        self.hold_fn = self.default_hold
        self.boost = 0
        self.nframes = 0
        self.mark_n = 0
        self.mark_s = 0
        self.fight_seen = False
        self.dbg = ""

    def default_hold(self, sc):
        for a, b, h in LAVA_HOLDS:
            if a <= sc < b:
                return h
        return 0

    def update(self, im):
        if self.prev is not None and np.array_equal(self.prev, im):
            self.still += 1
        else:
            self.still = 0
            if self.prev is not None:
                a = self.prev[40:200].astype(np.int16)
                b = im[40:200].astype(np.int16)
                best = (1e18, 0)
                for d in range(0, 6):
                    if d == 0:
                        sad = np.abs(a - b).sum()
                    else:
                        sad = np.abs(a[:, d:] - b[:, :-d]).sum() * (256.0 / (256 - d))
                    if sad < best[0]:
                        best = (sad, d)
                self.scroll += best[1]
        self.prev = im
        if self.still >= 3:
            self.dead = True
        self.nframes += 1
        if self.nframes - self.mark_n >= 40:
            if self.scroll - self.mark_s < 6 and not self.fight_seen:
                self.boost = min(self.boost + 10, 60)
            elif self.scroll - self.mark_s > 40:
                self.boost = 0
            self.mark_n = self.nframes
            self.mark_s = self.scroll

    def decide(self, im):
        sup = solid_mask(im)
        if self.scroll >= BRIDGE_S:
            red = (im[163:183, :, 0] == 248) & (im[163:183, :, 1] == 56) & (im[163:183, :, 2] == 0)
            sup[163:183] |= red
        m = find_mario(im)
        keys = {"Right", "x"}
        if m is None:
            return keys, "nomario"
        x0, x1, y0, y1 = m
        fy = y1 + 1
        mx = (x0 + x1) // 2
        front = x1
        G = goal_tier(self.scroll)

        def supp(x, y):
            if x < 0 or x > 255 or y > 237:
                return False
            return bool(sup[y:y + 2, x].any())

        on_ground = supp(mx, fy) or supp(x0, fy) or supp(x1, fy)
        self.fy_hist.append(fy)
        if len(self.fy_hist) > 4:
            self.fy_hist.pop(0)
        grounded = on_ground and len(self.fy_hist) >= 2 and self.fy_hist[-1] == self.fy_hist[-2]

        # ---- Bowser (green shell) on the bridge
        bow = None
        if self.scroll >= BRIDGE_S - 100:
            gm = (im[60:200, :, 0] == 0) & (im[60:200, :, 1] == 168) & (im[60:200, :, 2] == 0)
            if gm.sum() >= 30:
                ys, xs = np.nonzero(gm)
                bow = (int(xs.min()), int(xs.max()), int(ys.min()) + 60, int(ys.max()) + 60)
                self.fight_seen = True

        if bow is not None:
            return self.bridge_decide(im, keys, bow, x0, x1, y0, fy, grounded)

        # ---- edge / pit ahead
        edge = None   # (dx, kind, cand or None)
        for dx in range(1, 40):
            x = front + dx
            if x > 255:
                break
            if sup[fy:min(239, fy + 10), x:x + 3].any():
                continue
            if self.scroll >= BRIDGE_S and sup[fy:min(239, fy + 10), x:x + 9].any():
                continue   # tiny notch at the bridge start: ignore
            lava = bool(((im[205:240, x:x + 3, 0] == 248) & (im[205:240, x:x + 3, 1] == 56)).any())
            drops = []
            for x3 in range(x, min(252, x + 56), 4):
                drops += surfaces(sup, x3, fy + 8, fy + 84)
            drop_y = min(drops) if drops else None
            cands = []
            seen = set()
            for x2 in range(x, min(250, x + 100), 2):
                for yy in surfaces(sup, x2, fy - 70, fy + 70):
                    if yy >= 200 and fy < 170:
                        continue
                    key = yy // 8
                    if key not in seen:
                        seen.add(key)
                        cands.append((x2 - x, yy))
            best = None
            if cands:
                if G is None or G == -1:
                    best = min(cands, key=lambda c: (c[0]))
                else:
                    best = min(cands, key=lambda c: (abs(c[1] - G), c[0]))
            if lava or drop_y is None:
                edge = (dx, "pit", best)
            else:
                if best is not None and G != -1:
                    if G is None:
                        if best[1] < drop_y - 24:
                            edge = (dx, "cand", best)
                    else:
                        ref = abs(drop_y - G)
                        if abs(best[1] - G) < ref or (abs(best[1] - G) == ref and best[1] < drop_y - 6):
                            edge = (dx, "cand", best)
            break
        # ---- wall ahead
        wall = None
        for dx in range(1, max(14, WALL_TRIG + 4)):
            x = front + dx
            if x > 255:
                break
            if sup[max(40, fy - 12):fy - 3, x].any():
                y = fy - 4
                gap = 0
                top = y
                while y > 40 and gap < 5:
                    if sup[y, x:x + 6].any():
                        top = y
                        gap = 0
                    else:
                        gap += 1
                    y -= 1
                wall = (dx, fy - top)
                break
        # ---- ledge overhead (from the floor)
        ledge = None
        if fy >= 200:
            for dx in range(1, 24):
                x = front + dx
                if x > 255:
                    break
                if sup[fy - 12:fy - 3, x].any():
                    break
                ys = surfaces(sup, x, fy - 56, fy - 12)
                if ys:
                    ledge = (dx, fy - ys[0])
                    break

        why = ""
        if self.in_air or not grounded:
            if self.jump_t > 0 and self.jump_t < self.hold:
                keys.add("z")
                self.jump_t += 1
            if not grounded:
                self.left_ground = True
            elif self.left_ground:
                self.in_air = False
                self.jump_t = 0
            why = "air"
        if grounded and not self.in_air:
            kind = None
            if wall is not None and wall[0] <= WALL_TRIG:
                kind = "wall"
            elif edge is not None and edge[0] <= 6:
                kind = edge[1]
            elif ledge is not None and ledge[0] <= LEDGE_TRIG:
                ly = fy - ledge[1]
                if G is not None and G != -1 and abs(ly - G) < abs(fy - G):
                    kind = "ledge"
            if kind is not None:
                h = self.hold_fn(self.scroll) if self.pending is None else self.pending
                if h == 0:
                    if kind == "wall":
                        h = hold_for_height(wall[1] + WALL_MARGIN + self.boost)
                    elif kind == "ledge":
                        h = hold_for_height(ledge[1] + WALL_MARGIN + self.boost)
                    else:
                        h = 34
                        c = edge[2]
                        if c is not None:
                            gapw, ty = c
                            rise = fy - ty
                            h = max(hold_for_height(max(rise, 0) + 8), min(34, 6 + gapw // 4))
                self.pending = None
                self.hold = h
                keys.add("z")
                self.in_air = True
                self.jump_t = 1
                self.left_ground = False
                why = "jump %s S=%d fy=%d G=%s hold=%d edge=%s" % (kind, self.scroll, fy, G, h, edge)
        elif not grounded and not self.in_air:
            self.in_air = True
            self.jump_t = 0
            self.left_ground = True
        return keys, why

    def bridge_decide(self, im, keys, bow, x0, x1, y0, fy, grounded):
        """Mario on the bridge with Bowser in view."""
        front = x1
        dxb = bow[0] - front
        low = bow[3] > BOW_SAFE
        why = "bow dxb=%d bot=%d" % (dxb, bow[3])
        if self.in_air or not grounded:
            if self.jump_t > 0 and self.jump_t < self.hold:
                keys.add("z")
                self.jump_t += 1
            if not grounded:
                self.left_ground = True
            elif self.left_ground:
                self.in_air = False
                self.jump_t = 0
            return keys, why + " air"
        self.in_air = False
        # fire ahead in body band (between Mario and Bowser)
        xe = min(255, front + FIRE_AHEAD, bow[0] - 6) if dxb > 0 else min(255, front + FIRE_AHEAD)
        fire = False
        if xe > front + 2:
            band = im[max(41, fy - 26):fy - 1, front + 1:xe]
            hz = ((band[:, :, 0] == 248) & (band[:, :, 1] == 56)) | ((band[:, :, 0] == 252) & (band[:, :, 1] == 160))
            fire = hz.sum() >= 6
        if fire:
            keys.add("z")
            self.in_air = True
            self.jump_t = 1
            self.hold = FIRE_HOLD
            self.left_ground = False
            return keys, "jump fire " + why
        if low and 0 < dxb < BOW_NEAR:
            keys.discard("Right")      # wait for Bowser to rise
            return keys, "wait " + why
        return keys, why


def play(sim):
    bot = Bot()
    n = 0
    nlog = 0
    while n < 8000:
        im = sim.screen()
        bot.update(im)
        if bot.dead:
            print("dead/frozen at frame", sim.frame, "scroll", bot.scroll)
            return
        keys, why = bot.decide(im)
        if why.startswith("jump") and nlog < 80:
            nlog += 1
            print("f", sim.frame, why)
        if DEBUG and bot.fight_seen and n % 3 == 0 and nlog < 200:
            nlog += 1
            print("  f", sim.frame, "S", bot.scroll, why, sorted(keys))
        sim.step(keys)
        n += 1

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    """Cheap judgement: Bowser\'s green shell in view = danger zone (final bridge)."""\n    try:\n        g = (frame[60:200, :, 0] == 0) & (frame[60:200, :, 1] == 168) & (frame[60:200, :, 2] == 0)\n        bow = int(g.sum()) >= 30\n    except Exception:\n        bow = False\n    stuck = bool((ctx or {}).get("stuck", {}) and (ctx or {}).get("stuck", {}).get("is"))\n    status = "stuck" if stuck else ("danger" if bow else "running")\n    return {"status": status, "where": {"hazard": "bowser" if bow else None, "confidence": 0.6},\n            "progress": None, "resources": {}, "threats": ["bowser"] if bow else [], "stuck": {"is": stuck},\n            "outlook": {"label": "bad" if bow else "ok", "p_through": 0.7 if bow else 0.9}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

