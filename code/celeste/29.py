# ==================================================================================================
# Celeste Classic (PICO-8) -- room 29  |  蔚蓝经典版（PICO-8） 房间 29
# Pass rate on 100 random variants never used before: 100/100; fastest finish 77 frames (2.57 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 77 帧（30 帧/秒，约 2.57 秒）。
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
#   hold Right+z through spawn, sync on the first jump response (y drops after landing), replay a 
#   Go-Explore-found 78-frame route, then hold Up
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _ks                            line 10    
#   def _skin                          line 14    
#   def solve                          line 22    
#   def play                           line 26    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hold Right+z through spawn, sync on the first jump response (y drops after landing), replay a Go-Explore-found 78-frame route, then hold Up
import numpy as np

_KN = {1: "Left", 2: "Right", 4: "Up", 8: "Down", 16: "z", 32: "x"}
_TR = [18, 42, 10, 10, 1, 1, 1, 1, 1, 1, 1, 1, 18, 18, 18, 18, 18, 18, 18, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 38, 6, 6, 6, 6,
       18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 18, 2, 2, 2, 24, 24, 8, 8, 8, 8, 8, 8, 38, 6, 6, 20, 20, 20, 20, 36, 4, 4,
       9, 9, 9, 9, 9, 17, 17, 17, 41, 9, 9]


def _ks(m):
    return {v for b, v in _KN.items() if m & b}


def _skin(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(len(xs))


def solve(sim):
    return []


def play(sim):
    last = None
    for _ in range(200):
        sim.step(_ks(_TR[0]))
        s = _skin(sim.screen())
        if s is not None and s[0] <= 12:
            if last is not None and last >= 90 and s[1] <= 88 and s[2] >= 10:
                break
            last = s[1]
    for m in _TR[1:]:
        sim.step(_ks(m))
    for _ in range(40):
        sim.step({"Up"})

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.9}, "note": ""}\n'}

