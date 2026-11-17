# ==================================================================================================
# Super Mario Bros. (NES) -- level 4-4  |  超级马里奥兄弟（NES） 关 4-4
# Pass rate on 100 random variants never used before: 86/100; fastest finish 1191 frames (19.85 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 86/100；最快一局 1191 帧（60 帧/秒，约 19.85 秒）。
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
#   replay the verified deterministic 4-4 route (RLE trace), but closed-loop: per-frame vision gua
#   rd dodges podoboos / fireballs, sync waits and Bowser handling are reactive
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def dec                            line 7     
#   def rle_dec                        line 25    
#   def mario_box                      line 39    
#   def bowser_box                     line 48    
#   def estimate                       line 57    
#   def components                     line 72    8-connected components of a boolean mask -> list of (x0,y0,x1,y1,count)
#   def long_runs                      line 104   
#   def find_fire                      line 122   return list of (kind, x0, y0, x1, y1) of podoboo / fireball sprites (screen coords)
#   class Bot                          line 147   
#   def play                           line 344   
#   def _stuck                         line 480   
#   def solve                          line 488   
# ==================================================================================================

# idea: replay the verified deterministic 4-4 route (RLE trace), but closed-loop: per-frame vision guard dodges podoboos / fireballs, sync waits and Bowser handling are reactive
import numpy as np

KEYBIT = {"Left": 1, "Right": 2, "Up": 4, "Down": 8, "z": 16, "x": 32}


def dec(v):
    return frozenset(k for k, b in KEYBIT.items() if v & b)


RUN = frozenset({"Right", "x"})
JUMP = frozenset({"Right", "x", "z"})
IDLE = frozenset()
LEFT = frozenset({"Left"})
LEFTRUN = frozenset({"Left", "x"})
UNSTICK = [(frozenset({"Left", "z"}), 24), (frozenset({"Right", "z"}), 24), (frozenset({"z"}), 24),
           (frozenset({"Left"}), 30), (frozenset({"Left", "x", "z"}), 30), (frozenset({"Right"}), 20)]
WALK = frozenset({"Right"})

SYNC_IDX = 1021
GUARD_FROM = 880
PREFIX_RLE = "c36s12c36s20c364s7c17s7c29s22c38B70c308s12c48s12c24A32c28B30c27"


def rle_dec(s):
    vals = []
    i = 0
    while i < len(s):
        v = ord(s[i]) - 65
        i += 1
        j = i
        while j < len(s) and s[j].isdigit():
            j += 1
        vals.extend([v] * int(s[i:j]))
        i = j
    return vals


def mario_box(s):
    sub = s[40:]
    m = (sub[:, :, 0] == 172) & (sub[:, :, 1] == 124) & (sub[:, :, 2] == 0)
    ys, xs = np.nonzero(m)
    if len(ys) < 6:
        return None
    return (int(xs.min()), int(ys.min()) + 40, int(xs.max()), int(ys.max()) + 40, len(ys))


def bowser_box(s):
    sub = s[88:172]
    m = (sub[:, :, 0] == 0) & (sub[:, :, 1] == 168) & (sub[:, :, 2] == 0)
    ys, xs = np.nonzero(m)
    if len(ys) < 110:
        return None
    return (int(xs.min()), int(ys.min()) + 88, int(xs.max()), int(ys.max()) + 88, len(ys))


def estimate(a, b):
    # scroll estimate from the static ceiling band only (sprites below would corrupt it)
    ga = a[33:47].sum(axis=2, dtype=np.int16)
    gb = b[33:47].sum(axis=2, dtype=np.int16)
    d0 = float(np.abs(gb - ga).mean())
    if d0 < 0.05:
        return 0
    best, bd = 0, d0
    for dx in range(1, 6):
        d = float(np.abs(gb[:, :-dx] - ga[:, dx:]).mean())
        if d < bd - 1e-6:
            best, bd = dx, d
    return best


def components(mask):
    """8-connected components of a boolean mask -> list of (x0,y0,x1,y1,count)"""
    ys, xs = np.nonzero(mask)
    pts = set(zip(ys.tolist(), xs.tolist()))
    out = []
    while pts:
        p = pts.pop()
        stack = [p]
        x0 = x1 = p[1]
        y0 = y1 = p[0]
        n = 1
        while stack:
            y, x = stack.pop()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    q = (y + dy, x + dx)
                    if q in pts:
                        pts.discard(q)
                        stack.append(q)
                        n += 1
                        if q[1] < x0:
                            x0 = q[1]
                        if q[1] > x1:
                            x1 = q[1]
                        if q[0] < y0:
                            y0 = q[0]
                        if q[0] > y1:
                            y1 = q[0]
        out.append((x0, y0, x1, y1, n))
    return out


def long_runs(mask, L=20):
    Hh, Ww = mask.shape
    c = np.pad(np.cumsum(mask.astype(np.int32), axis=1), ((0, 0), (1, 0)))
    run = (c[:, L:] - c[:, :-L]) == L
    cov = np.zeros((Hh, Ww + 1), np.int32)
    rows, starts = np.nonzero(run)
    np.add.at(cov, (rows, starts), 1)
    np.add.at(cov, (rows, starts + L), -1)
    return (np.cumsum(cov, axis=1)[:, :Ww] > 0) & mask


# measured jump arc (feet y relative to take-off; z held 26 frames, running right), sampled every 3 frames
JA_T = [3 * k for k in range(0, 18)]
JA_Y = [0, -6, -23, -38, -51, -61, -70, -77, -82, -85, -86, -81, -69, -54, -39, -22, -8, 0]
Y_TOP = 48
Y_BOT = 206


def find_fire(s, mb):
    """return list of (kind, x0, y0, x1, y1) of podoboo / fireball sprites (screen coords)"""
    sub = s[Y_TOP:Y_BOT].astype(np.int16)
    red = (sub[:, :, 0] == 248) & (sub[:, :, 1] == 56) & (sub[:, :, 2] == 0)
    org = (sub[:, :, 0] == 252) & (sub[:, :, 1] == 160) & (sub[:, :, 2] == 68)
    wht = (sub[:, :, 0] == 252) & (sub[:, :, 1] == 252) & (sub[:, :, 2] == 252)
    fire = red | org
    fire &= ~long_runs(fire, 26)
    if mb is not None:
        x0, y0, x1, y1 = mb[0], mb[1], mb[2], mb[3]
        fire[max(0, y0 - 2 - Y_TOP):max(0, y1 + 2 - Y_TOP + 1), max(0, x0 - 2):x1 + 3] = False
    out = []
    for (cx0, cy0, cx1, cy1, n) in components(fire):
        w = cx1 - cx0 + 1
        h = cy1 - cy0 + 1
        if n < 12:
            continue
        nw = int(wht[cy0:cy1 + 1, cx0:cx1 + 1].sum())
        if 9 <= w <= 18 and 9 <= h <= 18 and nw >= 5:
            out.append(("pod", cx0, cy0 + Y_TOP, cx1, cy1 + Y_TOP))
        elif 16 <= w <= 40 and 3 <= h <= 10 and w >= 1.6 * h:
            out.append(("fire", cx0, cy0 + Y_TOP, cx1, cy1 + Y_TOP))
    return out


class Bot:
    def __init__(self, sim):
        self.sim = sim
        self.tr = rle_dec(PREFIX_RLE)
        self.i = 0
        self.prev = None
        self.scroll = 0
        self.mhist = []     # (world x0, y1) per frame
        self.tracks = []    # podoboo tracks: dict(cx, cy, vy, ay, age)
        self.ftracks = []
        self.alltracks = []
        self.floor = 160
        self.log_n = 0
        self.nguard = 0

    # ---- observation
    def observe(self, s):
        if self.i >= GUARD_FROM and self.prev is not None:
            self.scroll += estimate(self.prev, s)
        self.prev = s
        self.mb = mario_box(s)
        if self.mb is not None:
            self.mhist.append((self.scroll + self.mb[0], self.mb[3]))
            if len(self.mhist) > 8:
                self.mhist.pop(0)
        self.fires = []
        if self.i >= GUARD_FROM:
            try:
                self.fires = find_fire(s, self.mb)
            except Exception:
                self.fires = []
        self._track()

    def _track(self):
        new = []
        for f in self.fires:
            if f[0] != "pod":
                continue
            cx = self.scroll + (f[1] + f[3]) / 2.0
            cy = (f[2] + f[4]) / 2.0
            best = None
            for t in self.alltracks:
                d = abs(t["cx"] - cx) + abs(t["cy"] - cy) * 0.5
                if d < 14 and (best is None or d < best[0]):
                    best = (d, t)
            if best is not None:
                t = best[1]
                vy = cy - t["cy"]
                ay = vy - t["vy"] if t["age"] >= 1 else 0.0
                ay = max(-0.6, min(0.6, ay))
                still = t.get("still", 0) + 1 if (abs(vy) < 1.5 and abs(cx - t["cx"]) < 1.5) else 0
                new.append(dict(cx=cx, cy=cy, vy=vy, ay=0.5 * ay + 0.5 * t["ay"], age=t["age"] + 1,
                                w=f[3] - f[1] + 1, h=f[4] - f[2] + 1, still=still))
                self.alltracks.remove(t)
            else:
                new.append(dict(cx=cx, cy=cy, vy=0.0, ay=0.0, age=0, w=f[3] - f[1] + 1, h=f[4] - f[2] + 1, still=0))
        # a sprite that has not moved for 20+ frames is scenery (the axe), not a podoboo
        self.alltracks = new
        self.tracks = [t for t in new if t["still"] < 20]
        # horizontal fire (Bowser breath): only moving ones count
        fnew = []
        for f in self.fires:
            if f[0] != "fire":
                continue
            cx = self.scroll + (f[1] + f[3]) / 2.0
            cy = (f[2] + f[4]) / 2.0
            best = None
            for t in self.ftracks:
                d = abs(t["cx"] - cx) + abs(t["cy"] - cy) * 2
                if d < 8 and (best is None or d < best[0]):
                    best = (d, t)
            if best is not None:
                t = best[1]
                vx = cx - t["cx"]
                fnew.append(dict(cx=cx, cy=cy, vx=vx, age=t["age"] + 1, w=f[3] - f[1] + 1, h=f[4] - f[2] + 1))
                self.ftracks.remove(t)
            else:
                fnew.append(dict(cx=cx, cy=cy, vx=0.0, age=0, w=f[3] - f[1] + 1, h=f[4] - f[2] + 1))
        self.ftracks = [t for t in fnew]

    def mario_vw(self):
        if len(self.mhist) >= 5:
            return max(-3.0, min(3.0, (self.mhist[-1][0] - self.mhist[-5][0]) / 4.0))
        if len(self.mhist) >= 3:
            return max(-3.0, min(3.0, (self.mhist[-1][0] - self.mhist[-3][0]) / 2.0))
        return 0.0

    def grounded(self):
        if len(self.mhist) < 3:
            return False
        ys = [h[1] for h in self.mhist[-3:]]
        return max(ys) - min(ys) <= 1

    # ---- guard
    def jump_safe(self, v_min=0.0):
        """would a full-hold running jump taken right now hit a podoboo / fire?"""
        return self.threat_time(JUMP, T=44, arc=True) == 999

    def threat_time(self, keys, T=18, arc=False):
        """first frame t at which Mario (doing `keys` from now) would overlap a podoboo / fire, or 999"""
        fts = [f for f in self.ftracks if f["vx"] <= -0.8 and f["age"] >= 1]
        if self.mb is None or not (self.tracks or fts):
            return 999
        v0 = self.mario_vw()
        mx0 = self.scroll + self.mb[0]
        mw = self.mb[2] - self.mb[0] + 1
        mh = self.mb[3] - self.mb[1] + 1
        my1 = float(self.mb[3])
        base1 = my1
        air = not self.grounded()
        vy = 0.0
        if air and len(self.mhist) >= 2:
            vy = float(self.mhist[-1][1] - self.mhist[-2][1])
        g = 0.25 if "z" in keys else 0.6
        right = "Right" in keys
        left = "Left" in keys
        run = "x" in keys
        vmax = 2.5 if run else 1.5
        x = float(mx0)
        v = v0
        for t in range(1, T + 1):
            if right and not left:
                v = min(vmax, v + (0.1 if v >= 0 else 0.25))
            elif left and not right:
                v = max(-vmax, v - (0.1 if v <= 0 else 0.25))
            elif air or arc:
                pass
            else:
                v = v * 0.93 if abs(v) > 0.1 else 0.0
            x += v
            if arc:
                my1 = base1 + float(np.interp(t, JA_T, JA_Y))
            elif air:
                vy += g
                my1 += vy
                if my1 > self.floor:
                    my1 = float(self.floor)
                    vy = 0.0
            ymid = my1 - mh / 2.0
            for p in self.tracks:
                tt = min(t, 14)
                py = p["cy"] + p["vy"] * tt + 0.5 * p["ay"] * tt * tt
                if t > 14:
                    # beyond the horizon: podoboo could be anywhere in a vertical band around its predicted spot
                    if abs((x + mw / 2.0) - p["cx"]) < (mw / 2.0 + p["w"] / 2.0 + 3) and \
                            abs(ymid - py) < (mh / 2.0 + p["h"] / 2.0 + 6):
                        return t
                    continue
                vpad = 5 + (12 if p["age"] < 3 else 0)    # fresh tracks have no velocity estimate yet
                if abs((x + mw / 2.0) - p["cx"]) < (mw / 2.0 + p["w"] / 2.0 + 3) and \
                        abs(ymid - py) < (mh / 2.0 + p["h"] / 2.0 + vpad):
                    return t
            for f in fts:
                fx = f["cx"] + f["vx"] * t
                if abs((x + mw / 2.0) - fx) < (mw / 2.0 + f["w"] / 2.0 + 2) and \
                        abs(ymid - f["cy"]) < (mh / 2.0 + f["h"] / 2.0 + 2):
                    return t
        return 999

    def guard(self, want):
        """returns keys to press (maybe overriding `want`)"""
        if self.i < GUARD_FROM or self.mb is None:
            return want
        if self.mb[0] >= 200:
            return want     # already past Bowser, next to the axe: nothing left to dodge (and the axe looks like a podoboo)
        gr = self.grounded()
        if gr and self.mb[3] >= 150:
            self.floor = self.mb[3]
        if gr and "z" in want:
            return want
        t0 = self.threat_time(want)
        if t0 == 999:
            return want
        best = (t0, want)
        if gr:
            cands = (IDLE, LEFT, RUN, frozenset({"Left", "x"}))
        else:
            base = frozenset(k for k in want if k not in ("Right", "Left"))
            cands = (base, base | {"Left"}, base | {"Right"})
        for c in cands:
            if c == want:
                continue
            tc = self.threat_time(c)
            if tc > best[0]:
                best = (tc, c)
            if tc == 999:
                break
        self.nguard += 1
        if self.log_n < 8:
            self.log_n += 1
            print("GUARD f=%d i=%d want=%s -> %s (t %d -> %d) tracks=%s" % (
                getattr(self.sim, "frame", 0), self.i, sorted(want), sorted(best[1]), t0, best[0],
                [(round(p["cx"] - self.scroll), round(p["cy"]), round(p["vy"], 1)) for p in self.tracks]
                + [("F", round(p["cx"] - self.scroll), round(p["cy"]), round(p["vx"], 1)) for p in self.ftracks]))
        return best[1]


def play(sim):
    B = Bot(sim)
    tr = B.tr
    jump_left = 0
    wait_n = 0
    ys = []
    synced = False
    sync_on = False
    sync_n = 0
    sync_hist = []
    pos_hist = []
    back_left = 0
    bmode = False
    retreat = False
    last_pos = None
    stable_n = 0
    macro_n = 0
    macro_i = 0
    macro_key = IDLE
    try:
        while True:
            s = sim.screen()
            B.observe(s)
            mb = B.mb
            bb = bowser_box(s) if B.i >= GUARD_FROM else None
            pos_hist.append(None if mb is None else (mb[0], mb[3]))
            want = None
            adv = False
            # stuck-at-the-axe detector: if Mario stands still for 40 frames near the right edge, try macros
            cur = None if mb is None else (mb[0], mb[3])
            if cur is not None and last_pos is not None and abs(cur[0] - last_pos[0]) <= 1 and abs(cur[1] - last_pos[1]) <= 1:
                stable_n += 1
            else:
                stable_n = 0
            last_pos = cur
            if macro_n > 0:
                macro_n -= 1
                want = macro_key
            elif bmode and cur is not None and cur[0] >= 200 and stable_n >= 40:
                macro_key, macro_n = UNSTICK[macro_i % len(UNSTICK)]
                macro_i += 1
                stable_n = 0
                want = macro_key
                print("UNSTICK macro", macro_i, sorted(macro_key), macro_n, "at", cur)
            if want is not None:
                pass
            elif back_left > 0:
                back_left -= 1
                want = LEFT
            elif B.i >= len(tr) and not bmode and len(pos_hist) >= 90 and _stuck(pos_hist):
                back_left = 35
                del pos_hist[:]
                want = LEFT
            else:
                if B.i == SYNC_IDX and not synced:
                    synced = True
                    sync_on = True
                if sync_on:
                    reg = s[120:168, 120:190]
                    m = (((reg[:, :, 0] == 248) & (reg[:, :, 1] == 56) & (reg[:, :, 2] == 0))
                         | ((reg[:, :, 0] == 252) & (reg[:, :, 1] == 160) & (reg[:, :, 2] == 68)))
                    c = int(m.sum())
                    sync_hist.append(c)
                    sync_n += 1
                    if (len(sync_hist) >= 16 and c <= 40 and max(sync_hist[-16:-3]) >= 100) or sync_n >= 400:
                        sync_on = False
                    else:
                        want = IDLE
                if bb is not None and mb is not None:
                    bmode = True
                far = not bmode
                if want is not None:
                    pass
                elif far and B.i < len(tr):
                    want = dec(tr[B.i])
                    adv = True
                elif mb is None:
                    want = RUN
                else:
                    ys.append(mb[3])
                    grounded = len(ys) >= 3 and max(ys[-3:]) - min(ys[-3:]) <= 1
                    if grounded and mb[2] >= 236 and mb[3] <= 150:
                        want = LEFT
                    elif jump_left > 0:
                        jump_left -= 1
                        want = JUMP
                    else:
                        want = RUN
                        if bb is not None:
                            gap = bb[0] - mb[2]
                            v = B.mario_vw()
                            if retreat:
                                if gap >= 80 or mb[0] < 100 or gap < -6:
                                    retreat = False
                                elif grounded:
                                    want = LEFTRUN
                            if retreat:
                                pass
                            elif grounded and -6 < gap <= 66:
                                go = False
                                if gap <= 24:
                                    go = v >= 2.0
                                    if not go and mb[0] >= 100:
                                        retreat = True
                                        want = LEFTRUN
                                    elif not go:
                                        go = True
                                else:
                                    go = v >= 2.1 and (gap <= 27 or B.jump_safe())
                                if go:
                                    jump_left = 26
                                    want = JUMP
                                    wait_n = 0
                                    if B.log_n < 60:
                                        B.log_n += 1
                                        print("BOWSER JUMP f=%d gap=%d v=%.2f bx=%d mx=%d" % (
                                            getattr(sim, "frame", 0), gap, v, bb[0], mb[0]))
                            elif grounded and gap <= -6 and mb[2] >= 190:
                                jump_left = 14
                                want = JUMP
            final = B.guard(want)
            if final != want and B.grounded():
                adv = False
            if adv:
                B.i += 1
            else:
                B.i += 0
            # keep a frame counter even when the trace is paused so GUARD_FROM logic works
            if not adv and B.i < GUARD_FROM:
                pass
            sim.step(final)
    except Exception as ex:
        print("play end:", type(ex).__name__, ex, "frame", getattr(sim, "frame", None), "nguard", B.nguard)
        return


def _stuck(pos_hist):
    recent = [p for p in pos_hist[-90:] if p is not None]
    if len(recent) < 30:
        return True
    return (max(p[0] for p in recent) - min(p[0] for p in recent) < 12 and
            max(p[1] for p in recent) - min(p[1] for p in recent) < 30)


def solve(sim):
    return []

