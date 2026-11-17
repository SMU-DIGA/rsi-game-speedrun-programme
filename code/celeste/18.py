# ==================================================================================================
# Celeste Classic (PICO-8) -- room 18  |  蔚蓝经典版（PICO-8） 房间 18
# Pass rate on 100 random variants never used before: 94/100; fastest finish 183 frames (6.10 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 94/100；最快一局 183 帧（30 帧/秒，约 6.10 秒）。
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
#   sync on spawn landing, replay beam-searched balloon-chain route; re-sync the replay pointer on
#    observed dash-refill events (platform landing, balloons B5..B1)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _look                          line 24    
#   def q_ok                           line 39    
#   def solve                          line 43    
#   def play                           line 47    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: sync on spawn landing, replay beam-searched balloon-chain route; re-sync the replay pointer on observed dash-refill events (platform landing, balloons B5..B1)
import numpy as np

SK = np.array((255, 204, 170))
HAIR = np.array((255, 0, 77))
ML = 3
M = {"N": (), "U": ("Up",), "L": ("Left",), "R": ("Right",), "Z": ("z",),
     "RX": ("Right", "x"), "UX": ("Up", "x"), "LUX": ("Left", "Up", "x"),
     "RZ": ("Right", "z"), "RUX": ("Right", "Up", "x")}
# first macro of the searched trace ("U", a no-op) dropped
TRACE = ("RX N Z N N R R R R N Z UX N R R R RUX R N N N UX N Z N UX N L R UX N N N LUX N N N N L LUX "
         "N L L L UX N N L L L L N RZ N N R UX N Z N").split()
REF = []
for _t in TRACE:
    for _ in range(ML):
        REF.append(M[_t])
# events: (frame count after which the refill is seen in the reference, kind)
EVENTS = [(66, "land"), (86, "b"), (95, "b"), (128, "b"), (161, "b"), (173, "b")]
_SYNC_STABLE = 2
_WAIT_MAX = 6      # frames to wait for an expected event once the pointer reached it
_EARLY_MAX = 14    # accept an event up to this many frames before the reference time


def _look(s):
    m = (np.abs(s.astype(int) - SK).sum(2) < 6)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    x0, y0 = int(xs.min()), int(ys.min())
    h = (np.abs(s[max(0, y0 - 3):y0 + 2, max(0, x0 - 3):x0 + 6].astype(int) - HAIR).sum(2) < 6).sum()
    return (x0, y0, len(ys), int(h > 0))


_DASH3 = 132   # REF index of the first frame of the dash after the B3 pickup
_YD3 = 47      # skin y at which that dash was issued in the reference
_WX = 33       # skin x at which the player touches the cross wall (wall jump possible)


def q_ok(p):
    return p is not None


def solve(sim):
    return []


def play(sim):
    st = 0
    p = None
    for i in range(200):
        sim.step(set())
        p = _look(sim.screen())
        if p is not None and p[1] == 99 and p[2] >= 12:
            st += 1
            if st >= _SYNC_STABLE:
                break
        else:
            st = 0
    print("sync", i, p, "frame", sim.frame)
    ptr = 0          # reference frames executed
    ev = 0           # next expected event
    prev = p
    waited = 0
    total = 0
    extra3 = 0
    wj_n = 0
    wj_last_y = 0
    trail = []
    last_keys = ()
    while ptr < len(REF) and total < 400:
        total += 1
        # position-triggered dash after the B3 pickup (ev>=4 means event 3 seen)
        if ev == 4 and 128 <= ptr < _DASH3 + 1 and prev is not None and q_ok(prev):
            if ptr < _DASH3 and prev[1] >= _YD3:
                ptr = _DASH3
        if ev == 4 and ptr == _DASH3 and prev is not None and prev[1] < _YD3 and extra3 < 6:
            extra3 += 1
            keys = ("Left",)
            last_keys = keys
            sim.step(set(keys))
            q = _look(sim.screen())
            if q is not None:
                prev = q
            continue
        # closed-loop wall jump off the left side of the spiked cross (B3 -> B2 leg)
        if ev == 4 and 153 <= ptr < 159:
            if wj_n >= 12 or (wj_n >= 2 and prev is not None and prev[1] < wj_last_y):
                ptr = 159
        if ev == 4 and 153 <= ptr < 159:
            wj_last_y = prev[1] if prev is not None else 0
            keys = ("Right", "z") if wj_n % 2 == 0 else ("Right",)
            wj_n += 1
        # expected event reached but not yet seen: hold
        elif ev < len(EVENTS) and ptr >= EVENTS[ev][0] and waited < _WAIT_MAX:
            waited += 1
            keys = tuple(k for k in last_keys if k in ("Left", "Right"))
        else:
            keys = REF[ptr]
            ptr += 1
        last_keys = keys
        sim.step(set(keys))
        s = sim.screen()
        q = _look(s)
        if q is not None and prev is not None and prev[3] == 0 and q[3] == 1 and ev < len(EVENTS):
            E, kind = EVENTS[ev]
            ok = (q[1] < 90) if kind == "b" else (q[1] >= 104 and q[0] >= 100)
            if ok and ptr >= E - _EARLY_MAX:
                print("event", ev, kind, "at ptr", ptr, "expected", E, "pos", q[:2], "frame", sim.frame)
                ptr = E
                ev += 1
                waited = 0
        if ev < len(EVENTS) and ptr > EVENTS[ev][0] + 0 and waited >= _WAIT_MAX:
            print("event", ev, "MISSED at ptr", ptr, "pos", q)
            ev += 1
            waited = 0
        if q is not None:
            prev = q
        if 120 <= ptr <= 166:
            trail.append((ptr, q[:2] if q else None, q[3] if q else None))
        if total % 20 == 0:
            print(total, ptr, q)
        if ptr == 166 and trail:
            print("trail", trail)
            trail = []
    print("trace done", sim.frame)
    for j in range(60):
        sim.step({"Up"})
        if j % 10 == 0:
            print("post", j, _look(sim.screen()))

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {}, "progress": None, "resources": {}, "threats": [],\n            "stuck": {"is": False}, "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

