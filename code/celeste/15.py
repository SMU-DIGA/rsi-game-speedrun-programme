# ==================================================================================================
# Celeste Classic (PICO-8) -- room 15  |  蔚蓝经典版（PICO-8） 房间 15
# Pass rate on 100 random variants never used before: 100/100; fastest finish 71 frames (2.37 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 71 帧（30 帧/秒，约 2.37 秒）。
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
#   hold Right through spawn, sync when sprite has moved right (x>=54), then replay a beam-searche
#   d key trace (found offline with save/load)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _sync_ok                       line 9     
#   def solve                          line 15    
#   def play                           line 19    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hold Right through spawn, sync when sprite has moved right (x>=54), then replay a beam-searched key trace (found offline with save/load)
import numpy as np

SK = np.array((255, 204, 170))

TRACE = [["Right", "x"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["z"], ["z"], ["z"], ["Right", "Up", "x"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Left", "x"], ["Left"], ["Left"], ["Right", "Up", "x"], ["Right"], ["Right"], ["Right"], ["Right"], [], ["z"], ["z"], ["Left", "Up", "x"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], [], [], [], ["Left", "z"], ["Left", "z"], ["Left", "z"], ["Left"], ["Left", "z"], ["Left", "z"], ["Left", "z"], ["Left"], ["Left"], ["Left", "z"], ["Left", "z"], ["Left", "z"], ["Up", "x"], [], [], [], [], [], [], [], [], ["z"], ["z"], [], ["z"], ["z"], [], [], [], [], [], [], [], [], [], [], []]


def _sync_ok(s):
    m = (np.abs(s.astype(int) - SK).sum(2) < 6)
    ys, xs = np.where(m)
    return len(ys) >= 9 and xs.min() >= 54 and 104 <= ys.min() <= 109


def solve(sim):
    return []


def play(sim):
    synced = False
    for i in range(150):
        sim.step({"Right"})
        if _sync_ok(sim.screen()):
            synced = True
            break
    print("sync", synced, i)
    for k in TRACE:
        sim.step(set(k))
    for _ in range(60):
        sim.step({"Up"})

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {}, "progress": None, "resources": {}, "threats": [],\n            "stuck": {"is": False}, "outlook": {"label": "ok", "p_through": 0.9}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

