# ==================================================================================================
# Celeste Classic (PICO-8) -- room 9  |  蔚蓝经典版（PICO-8） 房间 9
# Pass rate on 100 random variants never used before: 100/100; fastest finish 87 frames (2.90 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 87 帧（30 帧/秒，约 2.90 秒）。
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
#   hold Right+z through the spawn, sync on the first frame the sprite visibly jumps, then replay 
#   a pre-searched open-loop trace (spring -> 2 balloons -> dash up out the top)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _frames                        line 11    
#   def _skin_full                     line 18    
#   def _skin                          line 26    
#   def _balloons                      line 35    
#   def _p                             line 44    
#   def play                           line 50    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hold Right+z through the spawn, sync on the first frame the sprite visibly jumps, then replay a pre-searched open-loop trace (spring -> 2 balloons -> dash up out the top)
import sys
import numpy as np

# run-length trace: (keys, frames); frame 0 of the trace is the first controlled frame (Right+z already sent during sync)
TRACE = [(('Right', 'z'), 12), (('Right',), 5), (('Right', 'x'), 1), (('Right',), 17), (('Left',), 12), ((), 2), (('Up', 'x'), 1), (('Up',), 5), (('Left', 'Up', 'x'), 1), (('Left', 'Up'), 5), (('Left', 'Up', 'x'), 1), (('Left', 'Up'), 5), (('Left', 'Up', 'x'), 1), (('Left', 'Up'), 3), (('Left', 'Up', 'x'), 1), (('Left', 'Up'), 3), (('Up', 'x'), 1), (('Up',), 3), (('Left', 'Up', 'x'), 1), (('Left', 'Up'), 5), (('z',), 2), ((), 12)]

DIAG = False


def _frames():
    out = []
    for keys, n in TRACE:
        out.extend([set(keys)] * n)
    return out


def _skin_full(s):
    m = (s[:, :, 0] == 255) & (s[:, :, 1] == 204) & (s[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(len(xs))


def _skin(s):
    sub = s[84:112, 8:48]
    m = (sub[:, :, 0] == 255) & (sub[:, :, 1] == 204) & (sub[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()) + 8, int(ys.min()) + 84, int(len(xs))


def _balloons(s):
    m = (s[:, :, 0] == 255) & (s[:, :, 1] == 0) & (s[:, :, 2] == 77)
    out = []
    for x0, x1 in ((26, 48), (58, 80)):
        ys, xs = np.nonzero(m[18:52, x0:x1])
        out.append(None if len(xs) == 0 else (int(xs.min()) + x0, int(ys.min()) + 18, int(len(xs))))
    return out


def _p(*a):
    if DIAG:
        print(*a)
        sys.stdout.flush()


def play(sim):
    frames = _frames()
    sync_keys = {"Right", "z"}
    started = False
    pre = []
    t = 0
    for t in range(150):
        sim.step(sync_keys)
        s = sim.screen()
        sk = _skin(s)
        pre.append(sk)
        if sk is not None and sk[2] >= 10 and sk[0] >= 20 and sk[1] <= 97:
            started = True
            break
    _p("SYNC t=%d started=%s pre_tail=%s balloons=%s" % (t, started, pre[-4:], _balloons(s)))
    start = 2 if started else 0
    for i in range(start, len(frames)):
        sim.step(frames[i])
        if DIAG and (i in (8, 20, 30, 34, 38, 42, 46, 50, 56, 62, 68, 74, 80) or i % 10 == 0):
            s = sim.screen()
            _p("  i=%d skin=%s balloons=%s" % (i, _skin_full(s), _balloons(s)))

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    m = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n    n = int(m.sum())\n    ys, xs = np.nonzero(m)\n    y = int(ys.min()) if n else None\n    st = "running"\n    prog = None\n    if y is not None:\n        prog = max(0.0, min(1.0, (100 - y) / 100.0))\n        if y > 110:\n            st = "danger"\n    return {"status": st, "where": {"hazard": None, "confidence": 0.3}, "progress": prog,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.8}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

