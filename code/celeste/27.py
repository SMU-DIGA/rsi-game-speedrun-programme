# ==================================================================================================
# Celeste Classic (PICO-8) -- room 27  |  蔚蓝经典版（PICO-8） 房间 27
# Pass rate on 100 random variants never used before: 100/100; fastest finish 215 frames (7.17 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 215 帧（30 帧/秒，约 7.17 秒）。
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
#   sync on first full-sprite stand frame (x=2,y=115,skin>=11) after the spawn, then replay a save
#   /load-searched jump/dash route through the ring corridor (30/30 explore, 20/20 test)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _parse                         line 15    
#   def _skin                          line 19    
#   def solve                          line 27    
#   def play                           line 31    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: sync on first full-sprite stand frame (x=2,y=115,skin>=11) after the spawn, then replay a save/load-searched jump/dash route through the ring corridor (30/30 explore, 20/20 test)
import numpy as np

_M = {"L": "Left", "R": "Right", "U": "Up", "D": "Down", "z": "z", "x": "x"}
_ROUTE = (
    "Rz Rz R R R R R R R R RUx RU RU RU - - - - - - - - - - - - - - - - Ux U U U - - L L R R R "
    "R z z R R R R R R R R R R - - - - z - - - - - - - R R R R R R R R - - - R R - - - - "
    "- - - - - - Lx L L L R R R R R R R - - L L R R R R RUx RU RU RU Lx L L L - - - - Ux U U U "
    "- - L L L L Ux U U U - - - Lx L L L L z z - - Lx L L Lx L L Rx R R - - L L - - Lx L L Rx R "
    "R Lx L L Rx R R L L Lx L L Lx L L z z - - Ux U U U - - - - LUx LU LU LU Rx R R - - z z Lx "
    "L L - - z z Lx L L "
)


def _parse(s):
    return [frozenset() if t == "-" else frozenset({_M[c] for c in t}) for t in s.split()]


def _skin(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(len(xs))


def solve(sim):
    return []


def play(sim):
    for n in range(150):
        sim.step(frozenset())
        o = _skin(sim.screen())
        if o is not None and o[0] == 2 and o[1] == 115 and o[2] >= 11:
            break
    for k in _parse(_ROUTE) + [frozenset({"Up"})] * 40:
        sim.step(k)

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "good", "p_through": 0.95}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

