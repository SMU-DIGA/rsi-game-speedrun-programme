# ==================================================================================================
# Celeste Classic (PICO-8) -- room 10  |  蔚蓝经典版（PICO-8） 房间 10
# Pass rate on 100 random variants never used before: 100/100; fastest finish 128 frames (4.27 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 128 帧（30 帧/秒，约 4.27 秒）。
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
#   sync on the spawn landing (skin px count drop), then replay an offline beam-searched key trace
#    (waypoint-guided: column top, then shaft)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _decode                        line 8     
#   def _skin                          line 15    
#   def play                           line 25    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: sync on the spawn landing (skin px count drop), then replay an offline beam-searched key trace (waypoint-guided: column top, then shaft)
import numpy as np

_TRACE = "Rx R R R z z z z z z z z Rx R R R z z z z z z z z z z z z Rz Rz Rz Rz RUx RU RU RU RU RU z z z z z z z z z z z z Lz Lz Lz Lz - z z z Lz Lz Lz Lz L Lz Lz Lz Lz Lz Lz Lz Lz Lz Lz Lz Rz Rz Rz Rz DLx DL DL DL DL DL - - - - z z z z Lz Lz Lz Lz Lz Lz Lz Lz LUx LU LU LU LU z z z z z z z z L Lz Lz Lz - z z z Ux U U U U U U U U U U U U U"
_NAMES = {"L": "Left", "R": "Right", "U": "Up", "D": "Down", "z": "z", "x": "x"}


def _decode():
    out = []
    for tok in _TRACE.split(" "):
        out.append(frozenset() if tok == "-" else frozenset(_NAMES[c] for c in tok))
    return out


def _skin(s):
    m = (s[:, :, 0] == 255) & (s[:, :, 1] == 204) & (s[:, :, 2] == 170)
    n = int(m.sum())
    if n == 0:
        return 0, 0, 0
    ys = np.nonzero(m.any(axis=1))[0]
    xs = np.nonzero(m.any(axis=0))[0]
    return n, int(ys[0]), int(xs[0])


def play(sim):
    trace = _decode()
    sq = False
    stable = 0
    for _ in range(200):
        n, y, x = _skin(sim.screen())
        if n:
            if n <= 9 and y >= 105:
                sq = True
            if n >= 11 and y == 107 and x == 10:
                stable += 1
                if sq or stable >= 5:
                    break
            else:
                stable = 0
        sim.step(set())
    for k in trace:
        sim.step(set(k))
    # keep idling a little in case the room change is late
    for _ in range(5):
        sim.step(set())

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    """Cheap judge: skin pixels present -> running; absent -> unknown (dash/death/new room)."""\n    try:\n        m = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n        n = int(m.sum())\n        ys = None\n        if n:\n            ys = int(m.any(axis=1).nonzero()[0][0])\n    except Exception:\n        return None\n    status = "running" if n else "danger"\n    return {"status": status, "where": {"hazard": None, "confidence": 0.3},\n            "progress": None if ys is None else max(0.0, min(1.0, 1 - ys / 128.0)),\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.7}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

