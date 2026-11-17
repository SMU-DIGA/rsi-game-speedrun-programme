# ==================================================================================================
# Celeste Classic (PICO-8) -- room 16  |  蔚蓝经典版（PICO-8） 房间 16
# Pass rate on 100 random variants never used before: 100/100; fastest finish 129 frames (4.30 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 129 帧（30 帧/秒，约 4.30 秒）。
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
#   hold Right through spawn, sync on first frame skin xmax>=8 (robust vs spawn flicker), then rep
#   lay searched macro (deterministic physics)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _bb                            line 9     
#   def solve                          line 17    
#   def play                           line 21    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hold Right through spawn, sync on first frame skin xmax>=8 (robust vs spawn flicker), then replay searched macro (deterministic physics)
import numpy as np

SK = np.array((255, 204, 170))
_KM = {'L': 'Left', 'R': 'Right', 'U': 'Up', 'D': 'Down', 'z': 'z', 'x': 'x'}
PATH = [('Rz', 8), ('Rz', 4), ('Rz', 2), ('', 9), ('R', 11), ('Rz', 3), ('R', 4), ('z', 14), ('R', 2), ('z', 11), ('Ux', 1), ('', 3), ('Rz', 9), ('Lx', 1), ('L', 2), ('Rz', 3), ('LU', 14), ('Lz', 3), ('LUz', 4), ('L', 2), ('Lz', 5), ('Rz', 3), ('Lz', 7)]


def _bb(s):
    m = (np.abs(s.astype(int) - SK).sum(2) < 6)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), len(ys)


def solve(sim):
    return []


def play(sim):
    synced = False
    for i in range(200):
        sim.step({"Right"})
        b = _bb(sim.screen())
        if b is not None and b[2] >= 8 and 94 <= b[1] <= 102:
            synced = True
            break
    print("sync", synced, i, b)
    for keys, n in PATH:
        ks = {_KM[c] for c in keys}
        for _ in range(n):
            sim.step(ks)
    print("path done", sim.frame)
    for j in range(40):
        sim.step({"Left", "Up"})

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {}, "progress": None, "resources": {}, "threats": [],\n            "stuck": {"is": False}, "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

