# ==================================================================================================
# Celeste Classic (PICO-8) -- room 6  |  蔚蓝经典版（PICO-8） 房间 6
# Pass rate on 100 random variants never used before: 100/100; fastest finish 52 frames (1.73 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 52 帧（30 帧/秒，约 1.73 秒）。
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
#   offline beam search (save/load, geodesic heuristic from pixels) for a key trace; play() waits 
#   for spawn landing then replays it with position logging
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _skin                          line 18    
#   def _hair                          line 26    
#   def _enc                           line 32    
#   def _dec                           line 40    
#   def _build_dist                    line 44    
#   def _dval                          line 75    
#   def _macros                        line 81    
#   def _sync                          line 99    Hold Right from step 0 (ignored during the spawn animation); the first frame the sprite
#   class _Node                        line 110   
#   def _path                          line 117   
#   def search                         line 126   
#   def solve                          line 190   
#   def play                           line 206   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: offline beam search (save/load, geodesic heuristic from pixels) for a key trace; play() waits for spawn landing then replays it with position logging
import heapq
import numpy as np

_SYNC_X = 15.5           # sprite x (centroid) at which the run-right sync fires
_W = 5                   # beam width (2*_W <= 10 save slots)
_H = 3                   # frames per ordinary macro
_HD = 4                  # frames per dash macro
_MAXLAYERS = 70
TRACE = "Rx,R,R,Ux,U,U,U,z,z,z,,,,R,R,R,RUx,RU,RU,RU,Ux,U,U,U,,,,RUx,RU,RU,RU,,,,z,z,z,,Rz,,z,z,z,,Rz,,,,,z,z,z"
_PREV2 = "Rx,R,R,R,Ux,U,U,U,z,z,z,,,,R,R,R,RUx,RU,RU,RU,Ux,U,U,U,,,,RUx,RU,RU,RU,,,,z,z,z,,,,z,z,z,,,,,,,z,z,z"
_PREV_TRACE = "Rx,R,R,R,Ux,U,U,U,z,z,z,,,,R,R,R,R,R,R,R,R,R,RUx,RU,RU,RU,Ux,U,U,U,,,,L,L,L,z,z,z,,,,,,,z,z,z,,,,,,,z,z,z,,,,,,,R,R,R,Lz,Lz,Lz,,,,,,,R,R,R,z,z,z,L,L,L,z,z,z,Ux,U,U,U"
_OLD = ""              # filled in after search: tokens like "R+z/R+z/..." per frame, ';' not used
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


def _build_dist(img):
    sol = np.zeros((128, 128), bool)
    for c in ((41, 173, 255), (255, 241, 232)):
        sol |= (img == np.array(c, np.uint8)).all(2)
    cs = sol.reshape(32, 4, 32, 4).sum((1, 3)) >= 6
    sp = (img == np.array((194, 195, 199), np.uint8)).all(2).reshape(32, 4, 32, 4).sum((1, 3)) >= 2
    blocked = cs | sp
    dist = np.full((32, 32), 1e9)
    pq = []
    for cx in range(32):
        if not blocked[0, cx]:
            dist[0, cx] = 0
            pq.append((0.0, 0, cx))
    heapq.heapify(pq)
    while pq:
        d, cy, cx = heapq.heappop(pq)
        if d > dist[cy, cx]:
            continue
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < 32 and 0 <= nx < 32 and not blocked[ny, nx]:
                    nd = d + (1.414 if dy and dx else 1.0)
                    if nd < dist[ny, nx]:
                        dist[ny, nx] = nd
                        heapq.heappush(pq, (nd, ny, nx))
    return dist


def _dval(dist, x, y):
    cx = min(31, max(0, int(x) // 4))
    cy = min(31, max(0, int(y) // 4))
    return float(dist[max(0, cy - 1):cy + 2, max(0, cx - 1):cx + 2].min())


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


def _sync(sim, maxn=250):
    """Hold Right from step 0 (ignored during the spawn animation); the first frame the sprite
    has clearly run right of the spawn x defines the common start state of every variant."""
    for i in range(maxn):
        sim.step({"Right"})
        s = _skin(sim.screen())
        if s is not None and s[0] >= _SYNC_X and 80 < s[1] < 100:
            return i, s
    return None, None


class _Node:
    __slots__ = ("par", "mac", "x", "y", "dash", "sc", "slot")

    def __init__(self, par, mac, x, y, dash, sc, slot):
        self.par, self.mac, self.x, self.y, self.dash, self.sc, self.slot = par, mac, x, y, dash, sc, slot


def _path(node):
    out = []
    while node is not None and node.mac is not None:
        out.append(node.mac)
        node = node.par
    out.reverse()
    return [k for m in out for k in m]


def search(sim):
    sim.reset()
    n, s = _sync(sim)
    print("sync after", n, "steps", s, flush=True)
    img = sim.screen()
    dist = _build_dist(img)
    s = _skin(img)
    print("base skin", s, "dist", _dval(dist, s[0], s[1]), "frame", sim.frame, flush=True)
    sim.save(0)
    root = _Node(None, None, s[0], s[1], True, 0.0, 0)
    beam = [root]
    cur_base, nxt_base = 0, 5
    best_seen = (1e9, None)
    for layer in range(_MAXLAYERS):
        kids = {}      # key -> node
        free = list(range(nxt_base, nxt_base + _W))
        for par in beam:
            for mac in _macros(par.dash):
                if sim.budget_left < 60:
                    raise RuntimeError("budget")
                sim.load(par.slot)
                for ks in mac:
                    sim.step(ks)
                img = sim.screen()
                s = _skin(img)
                if s is None or s[2] < 3:
                    # finished (left through top) or dead
                    if par.y < 30:
                        trace = _path(_Node(par, mac, 0, 0, 0, 0, 0))
                        return trace, True
                    continue
                if par.y < 30 and s[1] > par.y + 25:
                    trace = _path(_Node(par, mac, 0, 0, 0, 0, 0))
                    return trace, True
                hn = _hair(img, s)
                dash = hn >= 4
                sc = _dval(dist, s[0], s[1]) - (1.5 if dash else 0.0)
                key = (int(round(s[0])), int(round(s[1])), dash)
                old = kids.get(key)
                if old is not None and old.sc <= sc:
                    continue
                if old is not None:
                    slot = old.slot
                elif free:
                    slot = free.pop()
                else:
                    worst = max(kids.values(), key=lambda k: k.sc)
                    if worst.sc <= sc:
                        continue
                    slot = worst.slot
                    del kids[(int(round(worst.x)), int(round(worst.y)), worst.dash)]
                sim.save(slot)
                kids[key] = _Node(par, mac, s[0], s[1], dash, sc, slot)
        if not kids:
            print("beam empty at layer", layer)
            break
        beam = sorted(kids.values(), key=lambda k: k.sc)
        cur_base, nxt_base = nxt_base, cur_base
        if beam[0].sc < best_seen[0]:
            best_seen = (beam[0].sc, beam[0])
        print("layer", layer, "used", sim.budget - sim.budget_left, "best", [(round(b.x), round(b.y), b.dash, round(b.sc, 1)) for b in beam[:3]], flush=True)
    return (_path(best_seen[1]) if best_seen[1] else []), False


def solve(sim):
    if TRACE:
        return []
    try:
        trace, ok = search(sim)
    except Exception as e:
        print("search ended:", type(e).__name__, e)
        return []
    print("FINISHED" if ok else "NOT FINISHED", "len", len(trace))
    enc = ",".join(_enc(k) for k in trace)
    print("TRACE=" + repr(enc))
    sim.memory["trace"] = enc
    sim.memory["ok"] = bool(ok)
    return []


def play(sim):
    enc = TRACE
    try:
        if not enc:
            enc = sim.memory.get("trace", "")
    except Exception:
        pass
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
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    m = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n    n = int(m.sum())\n    if n == 0:\n        return {"status": "dead", "where": {"hazard": None, "confidence": 0.3}, "progress": None,\n                "resources": {}, "threats": [], "stuck": {"is": False},\n                "outlook": {"label": "bad", "p_through": 0.2}, "note": "no player pixels"}\n    ys, xs = np.nonzero(m)\n    prog = float(max(0.0, min(1.0, 1.0 - ys.mean() / 120.0)))\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.5}, "progress": prog,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.8}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

