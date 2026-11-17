# ==================================================================================================
# Celeste Classic (PICO-8) -- room 25  |  蔚蓝经典版（PICO-8） 房间 25
# Pass rate on 100 random variants never used before: 100/100; fastest finish 103 frames (3.43 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 103 帧（30 帧/秒，约 3.43 秒）。
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
#   sync on first Right+z jump response, then replay a beam-searched (BFS-field guided) 101-frame 
#   trace
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _frames                        line 8     
#   def _skin                          line 16    
#   def play                           line 24    
#   def solve                          line 36    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: sync on first Right+z jump response, then replay a beam-searched (BFS-field guided) 101-frame trace
import numpy as np

_KEY = {"R": "Right", "L": "Left", "U": "Up", "D": "Down", "z": "z", "x": "x"}
RLE = [("LRxz", 1), ("DRxz", 1), ("LR", 1), ("DRxz", 1), ("DRz", 1), ("LUxz", 1), ("Uxz", 1), ("RUz", 1), ("DLRz", 2), ("Rx", 1), ("U", 1), ("Rx", 1), ("DLUz", 1), ("", 1), ("DLxz", 1), ("DRz", 1), ("DR", 1), ("DRx", 1), ("Lx", 1), ("", 1), ("LUxz", 1), ("LUx", 1), ("DLRUxz", 1), ("LUxz", 1), ("DRUx", 1), ("LUx", 2), ("Lx", 1), ("DUx", 1), ("Dx", 1), ("", 1), ("Dz", 2), ("x", 1), ("LUxz", 1), ("Rx", 1), ("LRUxz", 1), ("DLR", 1), ("DRU", 1), ("LRUxz", 1), ("DLx", 1), ("DLU", 1), ("Lz", 1), ("DUx", 1), ("Dxz", 1), ("x", 1), ("RUz", 1), ("DLRxz", 1), ("DLU", 1), ("Dx", 1), ("DLRUz", 1), ("LU", 1), ("DRxz", 1), ("xz", 1), ("DRxz", 1), ("DLz", 1), ("DLR", 1), ("DRUz", 1), ("DLz", 1), ("LRx", 1), ("LRUz", 1), ("DUz", 1), ("DRUxz", 2), ("DL", 1), ("LRUx", 1), ("DUx", 1), ("DLRxz", 1), ("Ux", 1), ("DRUxz", 1), ("LUxz", 1), ("DR", 1), ("LRUx", 1), ("LRUxz", 1), ("LRUx", 1), ("LUxz", 1), ("Dxz", 1), ("DU", 1), ("LU", 1), ("LRUz", 1), ("x", 1), ("DRU", 1), ("DLUz", 1), ("DLU", 1), ("DLUz", 1), ("DLz", 1), ("L", 1), ("DLUz", 1), ("U", 1), ("LRUx", 1), ("LRxz", 1), ("DL", 1), ("Uz", 1), ("z", 1), ("RU", 1), ("DL", 1), ("LUxz", 1), ("Uxz", 1), ("DLU", 1), ("DRUz", 1)]


def _frames():
    out = []
    for ks, n in RLE:
        k = frozenset(_KEY[c] for c in ks)
        out.extend([k] * n)
    return out


def _skin(s):
    m = (s[:, :, 0] == 255) & (s[:, :, 1] == 204) & (s[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), len(xs)


def play(sim):
    for i in range(160):
        sim.step({"Right", "z"})
        b = _skin(sim.screen())
        if b is not None and b[0] >= 36 and b[4] >= 10:
            break
    for k in _frames():
        sim.step(set(k))
    for i in range(40):
        sim.step({"Up"})


def solve(sim):
    return []

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    s = np.asarray(frame)\n    m = (s[:, :, 0] == 255) & (s[:, :, 1] == 204) & (s[:, :, 2] == 170)\n    n = int(m.sum())\n    ys = np.nonzero(m.any(axis=1))[0]\n    y = int(ys.min()) if len(ys) else None\n    status = "dead" if n >= 25 else "running"\n    prog = None if y is None else max(0.0, min(1.0, 1.0 - y / 128.0))\n    return {"status": status, "where": {"hazard": None, "confidence": 0.3}, "progress": prog,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n'}

