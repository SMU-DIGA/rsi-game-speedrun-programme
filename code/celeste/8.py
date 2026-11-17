# ==================================================================================================
# Celeste Classic (PICO-8) -- room 8  |  蔚蓝经典版（PICO-8） 房间 8
# Pass rate on 100 random variants never used before: 100/100; fastest finish 89 frames (2.97 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 89 帧（30 帧/秒，约 2.97 秒）。
#
# What this file is / 这是什么:
#   A game-playing policy written by an AI agent (Claude Sonnet). Below this header the code is the
#   agent's own file, unchanged (only local paths are masked). The agent's comments are kept.
#   这是 AI 代理（Claude Sonnet）写的通关策略。本说明以下是代理写的原文件，未改动（只隐去了本地路径），代理自己的注释都保留。
#
# How it is run / 怎么运行:
#   The referee calls play(sim) once on a fresh random variant. Every frame the policy may read
#   sim.screen() -- the RGB pixels of the game window, nothing else (no game memory, no positions) --
#   and must call sim.step(keys) to hold a set of keys for one frame (keys: Left / Right / Up / Down, z = jump, x = dash).
#   Death or finishing is judged by the referee, not reported to the policy.
#   裁判在一个新的随机变体上调用一次 play(sim)。策略每一帧只能用 sim.screen() 读取游戏画面像素（读不到游戏内存和坐标），
#   用 sim.step(keys) 按住一组按键走一帧。死亡还是通关由裁判判定，不告诉策略。
#
# The agent's summary of its approach / 代理自己对思路的概括:
#   waypoint-guided offline beam search (save/load macros) -> key trace stored in TRACE; play() sy
#   ncs on run-start then replays
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _skin                          line 23    
#   def _hair                          line 31    
#   def _enc                           line 37    
#   def _dec                           line 45    
#   def _sync                          line 49    
#   def _macros                        line 58    
#   class _Node                        line 76    
#   def _path                          line 83    
#   def _score                         line 92    
#   def search                         line 99    
#   def solve                          line 182   
#   def play                           line 203   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: waypoint-guided offline beam search (save/load macros) -> key trace stored in TRACE; play() syncs on run-start then replays
import math
import numpy as np
try:
    from PIL import Image
except Exception:
    Image = None

WAYPOINTS = [(60, 108, 5), (100, 96, 9), (100, 60, 8), (51, 36, 5), (22, -6, 5)]
DASH_FROM = 3
RADIUS = 5.0
PREFIX = ""      # tokens of already-solved stage(s) (comma separated)
TRACE = "R,R,R,R,R,R,Rz,Rz,Rz,R,R,R,R,R,R,R,R,R,R,R,R,R,R,R,Rz,Rz,Rz,R,R,R,R,R,R,R,R,R,R,R,R,L,L,,L,L,L,L,L,L,L,L,L,LUx,LU,LU,LU,LU,LU,LU,LU,,,,,,,LUx,LU,LU,LU,LU,LU,LU,LU,,,,,,,Lz,Lz"           # final full trace (tokens) used by play()
SYNC_X = 14.0
_W = 5
_H = 3
_HD = 8
MAXLAYERS = 120
_N = {"L": "Left", "R": "Right", "U": "Up", "D": "Down", "z": "z", "x": "x"}
DIRS8 = [("Up",), ("Right", "Up"), ("Right",), ("Right", "Down"), ("Down",), ("Left", "Down"), ("Left",), ("Left", "Up")]


def _skin(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()), len(xs))


def _hair(img, s):
    x, y = int(s[0]), int(s[1])
    w = img[max(0, y - 8):y + 3, max(0, x - 7):x + 8]
    return int(((w[:, :, 0] == 255) & (w[:, :, 1] == 0) & (w[:, :, 2] == 77)).sum())


def _enc(keys):
    s = ""
    for k, c in (("Left", "L"), ("Right", "R"), ("Up", "U"), ("Down", "D"), ("z", "z"), ("x", "x")):
        if k in keys:
            s += c
    return s


def _dec(tok):
    return {_N[c] for c in tok}


def _sync(sim, maxn=250):
    for i in range(maxn):
        sim.step({"Right"})
        s = _skin(sim.screen())
        if s is not None and s[0] >= SYNC_X and 100 < s[1] < 120 and s[2] >= 5:
            return i, s
    return None, None


def _macros(has_dash):
    ms = []
    for hx in (None, "Left", "Right"):
        for z in (0, 1):
            k = set()
            if hx:
                k.add(hx)
            if z:
                k.add("z")
            ms.append([set(k) for _ in range(_H)])
    if has_dash:
        for d in DIRS8:
            first = set(d) | {"x"}
            rest = set(d)
            ms.append([first] + [set(rest) for _ in range(_HD - 1)])
    return ms


class _Node:
    __slots__ = ("par", "mac", "x", "y", "dash", "sc", "slot", "idx")

    def __init__(self, par, mac, x, y, dash, sc, slot, idx):
        self.par, self.mac, self.x, self.y, self.dash, self.sc, self.slot, self.idx = par, mac, x, y, dash, sc, slot, idx


def _path(node):
    out = []
    while node is not None and node.mac is not None:
        out.append(node.mac)
        node = node.par
    out.reverse()
    return [k for m in out for k in m]


def _score(x, y, dash, idx):
    if idx >= len(WAYPOINTS):
        return -1000.0 * idx
    wx, wy, _r = WAYPOINTS[idx]
    return -1000.0 * idx + math.hypot(x - wx, y - wy) - (6.0 if dash else 0.0)


def search(sim):
    sim.reset()
    n, s = _sync(sim)
    print("sync after", n, s, "frame", sim.frame, flush=True)
    if PREFIX:
        for tok in PREFIX.split(","):
            sim.step(_dec(tok))
    img = sim.screen()
    s = _skin(img)
    print("start skin", s, "frame", sim.frame, flush=True)
    sim.save(0)
    idx0 = 0
    root = _Node(None, None, s[0], s[1], _hair(img, s) >= 4, 0.0, 0, idx0)
    beam = [root]
    cur_base, nxt_base = 0, 5
    best = root
    pre = [{c for c in t} for t in []]
    for layer in range(MAXLAYERS):
        kids = {}
        free = list(range(nxt_base, nxt_base + _W))
        for par in beam:
            for mac in _macros(par.dash and par.idx >= DASH_FROM):
                if sim.budget_left < 80:
                    raise RuntimeError("budget")
                sim.load(par.slot)
                for ks in mac:
                    sim.step(ks)
                img = sim.screen()
                s = _skin(img)
                if s is None or s[2] < 3:
                    if par.y < 12:
                        return _path(_Node(par, mac, 0, 0, 0, 0, 0, 0)), True, img
                    continue
                if par.y < 20 and s[1] > par.y + 40:
                    return _path(_Node(par, mac, 0, 0, 0, 0, 0, 0)), True, img
                if s[1] > 124:
                    continue
                dash = _hair(img, s) >= 4
                idx = par.idx
                while idx < len(WAYPOINTS) and math.hypot(s[0] - WAYPOINTS[idx][0], s[1] - WAYPOINTS[idx][1]) < WAYPOINTS[idx][2]:
                    idx += 1
                sc = _score(s[0], s[1], dash, idx)
                key = (int(round(s[0])), int(round(s[1])), dash, idx)
                old = kids.get(key)
                if old is not None and old.sc <= sc:
                    continue
                if old is not None:
                    slot = old.slot
                elif free:
                    slot = free.pop()
                else:
                    ncls = sum(1 for k in kids.values() if k.dash == dash)
                    cap = 3 if dash else 2
                    pool = [k for k in kids.values() if (k.dash != dash) == (ncls < cap)] or list(kids.values())
                    worst = max(pool, key=lambda k: k.sc)
                    if worst.sc <= sc and ncls >= cap:
                        continue
                    if worst.dash == dash and worst.sc <= sc:
                        continue
                    slot = worst.slot
                    del kids[(int(round(worst.x)), int(round(worst.y)), worst.dash, worst.idx)]
                sim.save(slot)
                kids[key] = _Node(par, mac, s[0], s[1], dash, sc, slot, idx)
                if idx >= len(WAYPOINTS):
                    pass
        if not kids:
            print("beam empty at layer", layer)
            break
        beam = sorted(kids.values(), key=lambda k: k.sc)
        cur_base, nxt_base = nxt_base, cur_base
        if beam[0].sc < best.sc:
            best = beam[0]
        if layer % 4 == 0 or beam[0].idx != getattr(search, "_lastidx", -1):
            print("layer", layer, "used", sim.budget - sim.budget_left, "best",
                  [(round(b.x), round(b.y), int(b.dash), b.idx) for b in beam[:2]], flush=True)
        search._lastidx = beam[0].idx
        if beam[0].idx >= len(WAYPOINTS):
            sim.load(beam[0].slot)
            return _path(beam[0]), False, sim.screen()
    sim.load(best.slot)
    return _path(best), False, sim.screen()


def solve(sim):
    if TRACE:
        return []
    try:
        trace, ok, img = search(sim)
    except Exception as e:
        print("search ended:", type(e).__name__, e)
        return []
    try:
        Image.fromarray(img).resize((512, 512), Image.NEAREST).save("end.png")
    except Exception:
        pass
    print("FINISHED" if ok else "STAGE END", "len", len(trace))
    enc = ",".join(_enc(k) for k in trace)
    if PREFIX:
        enc = PREFIX + "," + enc
    print("TRACE=" + repr(enc))
    sim.memory["trace"] = enc
    return []


def play(sim):
    enc = TRACE
    toks = enc.split(",") if enc else []
    try:
        n, s = _sync(sim)
        print("sync", n, s, "frame", sim.frame)
        for j, tok in enumerate(toks):
            sim.step(_dec(tok))
            if j % 6 == 5:
                s = _skin(sim.screen())
                print("j", j, "frame", sim.frame, None if s is None else (round(s[0], 1), round(s[1], 1), s[2]))
        for _ in range(20):
            sim.step(set())
    except Exception as e:
        print("play ended:", type(e).__name__, e)

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    m = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n    n = int(m.sum())\n    if n == 0:\n        return {"status": "running", "where": {"hazard": None, "confidence": 0.3}, "progress": None,\n                "resources": {}, "threats": [], "stuck": {"is": False},\n                "outlook": {"label": "ok", "p_through": 0.8}, "note": "no skin pixels (dash flash or transition)"}\n    ys, xs = np.nonzero(m)\n    y = float(ys.mean())\n    prog = float(max(0.0, min(1.0, 1.0 - y / 120.0)))\n    danger = y > 121\n    return {"status": "danger" if danger else "running", "where": {"hazard": "pit" if danger else None, "confidence": 0.5},\n            "progress": prog, "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "bad" if danger else "ok", "p_through": 0.3 if danger else 0.9}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

