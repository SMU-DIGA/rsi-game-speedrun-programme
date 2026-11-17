# ==================================================================================================
# Celeste Classic (PICO-8) -- room 30  |  蔚蓝经典版（PICO-8） 房间 30
# Pass rate on 100 random variants never used before: 100/100; fastest finish 55 frames (1.83 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 55 帧（30 帧/秒，约 1.83 秒）。
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
#   hold Right through the spawn, sync on first run response (skin x>=14), replay a pre-searched j
#   ump/dash route, then hold Up
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _parse                         line 10    
#   def _skin                          line 17    
#   def solve                          line 25    
#   def play                           line 29    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hold Right through the spawn, sync on first run response (skin x>=14), replay a pre-searched jump/dash route, then hold Up
import numpy as np

_M = {"L": "Left", "R": "Right", "U": "Up", "D": "Down", "z": "z", "x": "x"}
_ROUTE = ("R R R z z z Rx Rx Rx Rx - - - - RUx RUx RUx RUx - - - - z z LUx LUx LUx LUx "
          "- - - - Ux Ux Ux Ux - - - - - - - - z z z")
_TAIL = 40


def _parse(s):
    out = []
    for tok in s.split():
        out.append(set() if tok == "-" else {_M[c] for c in tok})
    return out


def _skin(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(len(xs))


def solve(sim):
    return []


def play(sim):
    n = 0
    trig = []
    for n in range(150):
        sim.step({"Right"})
        o = _skin(sim.screen())
        trig.append(o)
        if o is not None and o[0] >= 14 and o[2] >= 7 and 81 <= o[1] <= 84:
            break
    print("sync", n, trig[-4:])
    for k in _parse(_ROUTE):
        sim.step(k)
    for _ in range(_TAIL):
        sim.step({"Up"})

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n'}

