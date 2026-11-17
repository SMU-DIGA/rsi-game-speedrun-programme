# ==================================================================================================
# Super Mario Bros. (NES) -- level 2-1  |  超级马里奥兄弟（NES） 关 2-1
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1389 frames (23.15 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1389 帧（60 帧/秒，约 23.15 秒）。
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
#   replay a deterministic learned prefix trace (reaches the tall hard-block tower of 2-1), then c
#   losed-loop springboard finisher (jump left onto the spring, re-press A on landing -> super bou
#   nce over the
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def dec                            line 13    
#   def prefix                         line 17    
#   def find_tower                     line 25    left x of the 2-block-wide column of hard blocks (rows 90..140 fully non-sky), or None
#   def mario_bbox                     line 45    
#   def seq_spring                     line 57    
#   class Runner                       line 62    
#   def play                           line 183   
#   def solve                          line 191   
# ==================================================================================================

# idea: replay a deterministic learned prefix trace (reaches the tall hard-block tower of 2-1), then closed-loop springboard finisher (jump left onto the spring, re-press A on landing -> super bounce over the tower -> flag)
import numpy as np

RLE = ("9:27,14:14,10:7,14:3,13:21,9:4,13:10,9:25,13:7,9:19,8:6,13:14,9:8,8:1,9:14,13:24,9:35,13:24,9:13,13:24,9:41,"
       "13:24,9:4,13:24,9:4,13:14,9:4,8:8,9:7,13:24,9:4,13:24,9:48,13:30,9:35,13:15,9:15,13:15,9:70,13:17,9:23,13:17,"
       "9:23,13:30,9:36,13:34,9:20,13:20,9:35,13:15,9:21,13:19,8:20,9:3,13:7,9:20,13:20,9:55,13:25,9:25,13:15,9:10")
KEYBITS = {"Right": 1, "Left": 2, "z": 4, "x": 8, "Up": 16, "Down": 32}
T0 = 1220         # frames of the prefix trace that are replayed
SKY = (104, 136, 252)
PS = [35, 34, 36, 33, 35, 34, 36, 33]


def dec(v):
    return {k for k, b in KEYBITS.items() if v & b}


def prefix():
    out = []
    for tok in RLE.split(","):
        v, n = tok.split(":")
        out += [int(v)] * int(n)
    return out


def find_tower(s):
    """left x of the 2-block-wide column of hard blocks (rows 90..140 fully non-sky), or None"""
    band = s[90:140].astype(np.int16)
    sky = (band[..., 0] == SKY[0]) & (band[..., 1] == SKY[1]) & (band[..., 2] == SKY[2])
    frac = 1.0 - sky.mean(axis=0)
    col = frac > 0.92
    x = 0
    while x < 256:
        if col[x]:
            x2 = x
            while x2 < 256 and col[x2]:
                x2 += 1
            if x2 - x >= 26:
                return x
            x = x2
        else:
            x += 1
    return None


def mario_bbox(s, xt):
    m = (s[..., 0] == 248) & (s[..., 1] == 56) & (s[..., 2] == 0)
    m[:40] = False
    m[226:] = False
    if xt is not None:
        m[168:214, max(0, xt - 34):max(0, xt - 12)] = False
    ys, xs = np.nonzero(m)
    if len(ys) < 5:
        return None
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def seq_spring(p):
    q = [{"Left", "z"}] * 6 + [{"z"}] * 6 + [set()] * (p - 12) + [{"Right", "z"}] * 24
    return q


class Runner:
    def __init__(self, sim, log=True):
        self.sim = sim
        self.log = log
        self.pre = prefix()[:T0]
        self.n = 0
        self.last = None
        self.same = 0
        self.q = []
        self.prev_box = None
        self.stable = 0
        self.attempt = 0
        self.stall = 0
        self.lastx = None
        self.phase = "replay"
        self.events = []

    def say(self, *a):
        if self.log:
            print("[run]", self.n, *a, flush=True)

    def step(self, keys):
        self.sim.step(keys)
        self.n += 1

    def finisher_keys(self, s):
        xt = find_tower(s)
        box = mario_bbox(s, xt)
        if box is None:
            self.prev_box = None
            self.stable = 0
            return {"Right", "x"}
        if self.prev_box is not None and box[2] == self.prev_box[2] and box[3] == self.prev_box[3]:
            self.stable += 1
        else:
            self.stable = 0
        self.prev_box = box
        x0, x1, y0, y1 = box
        grounded = self.stable >= 2 and y1 >= 203
        if xt is None:
            # tower not visible (already passed / not yet there): just run, hop when stalled
            if self.lastx is not None and x1 == self.lastx and grounded:
                self.stall += 1
            else:
                self.stall = 0
            self.lastx = x1
            if self.stall > 6:
                self.q = [{"Right", "x", "z"}] * 20
                self.stall = 0
            return {"Right", "x"}
        if x0 > xt + 34:
            # past the tower: run to the pole
            if self.lastx is not None and x1 == self.lastx and grounded:
                self.stall += 1
            else:
                self.stall = 0
            self.lastx = x1
            if self.stall > 6:
                self.q = [{"Right", "x", "z"}] * 20
                self.stall = 0
            return {"Right", "x"}
        if not grounded:
            k = {"Right", "x"}
            if x0 < xt + 30:
                k.add("z")
            return k
        if x0 >= xt - 14:
            # on the tower side of the spring, standing: press against the wall, then do the spring trick
            p = PS[self.attempt % len(PS)]
            self.attempt += 1
            self.say("spring attempt", self.attempt, "p", p, "xt", xt, "box", box)
            self.q = [{"Right"}] * 8 + seq_spring(p)
            return {"Right"}
        if x0 >= xt - 36:
            # standing on / next to the spring: bounce with a fresh press
            self.say("on spring zone", xt, box)
            self.q = [{"Right", "z"}] * 40
            return {"Right"}
        # left of the spring
        if xt - x1 <= 52:
            self.say("hop toward spring", xt, box)
            self.q = [{"Right", "x", "z"}] * 26
            return {"Right", "x", "z"}
        return {"Right", "x"}

    def run(self, maxf):
        sim = self.sim
        pre = self.pre
        while self.n < maxf:
            if sim.budget_left <= 0:
                return "budget"
            s = sim.screen()
            if self.last is not None and np.array_equal(s, self.last):
                self.same += 1
            else:
                self.same = 0
            self.last = s
            if self.same >= 45:
                return "frozen"
            if self.n < len(pre):
                self.step(dec(pre[self.n]))
                continue
            if self.phase == "replay":
                self.phase = "finish"
                self.say("prefix done; tower", find_tower(s), "mario", mario_bbox(s, find_tower(s)))
            if self.q:
                keys = self.q.pop(0)
                # keep tracking stability even while queued
                xt = find_tower(s)
                box = mario_bbox(s, xt)
                self.prev_box = box
                self.stable = 0
                self.step(keys)
                continue
            keys = self.finisher_keys(s)
            self.step(keys)
            if self.n % 100 == 0:
                self.say("status")
        return "maxf"


def play(sim):
    r = Runner(sim, log=False)
    try:
        r.run(7900)
    except Exception as e:
        print("play end:", repr(e))


def solve(sim):
    return []

