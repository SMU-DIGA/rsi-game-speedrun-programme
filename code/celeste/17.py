# ==================================================================================================
# Celeste Classic (PICO-8) -- room 17  |  蔚蓝经典版（PICO-8） 房间 17
# Pass rate on 100 random variants never used before: 100/100; fastest finish 97 frames (3.23 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 97 帧（30 帧/秒，约 3.23 秒）。
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
#   wait for stable spawn stand (18,107), replay beam-searched trace (offline save/load beam searc
#   h with waypoints)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _look                          line 8     
#   def solve                          line 16    
#   def play                           line 20    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: wait for stable spawn stand (18,107), replay beam-searched trace (offline save/load beam search with waypoints)
import numpy as np

SK = np.array((255, 204, 170))
TRACE = [["Right"], ["Right"], ["Right"], ["Right"], ["Right", "x"], ["Right"], ["Right"], ["Right"], [], [], [], [], [], [], [], [], ["Right", "x"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], [], [], [], [], [], [], [], [], ["Right", "z"], ["Right"], ["Right"], ["Right"], [], [], [], [], ["Right", "x"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Down", "x"], ["Down"], ["Down"], ["Down"], ["Left", "z"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Left"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right", "z"], ["Right"], ["Right"], ["Right"], [], [], [], [], ["Up", "x"], ["Up"], ["Up"], ["Up"], ["Up"], ["Up"], ["Up"], ["Up"], ["Right", "z"], ["Right"], ["Right"], ["Right"], [], [], [], [], [], [], [], [], [], [], [], [], [], []]


def _look(s):
    m = (np.abs(s.astype(int) - SK).sum(2) < 6)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), len(ys))


def solve(sim):
    return []


def play(sim):
    run = 0
    p = None
    i = 0
    for i in range(200):
        sim.step(set())
        p = _look(sim.screen())
        if p is not None and p[0] == 18 and p[1] == 107 and p[2] >= 11:
            run += 1
        else:
            run = 0
        if run >= 3:
            break
    print("sync", i, p, flush=True)
    for k in TRACE:
        sim.step(set(k))
    print("trace done", sim.frame, flush=True)
    for j in range(60):
        sim.step(set())

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {}, "progress": None, "resources": {}, "threats": [],\n            "stuck": {"is": False}, "outlook": {"label": "ok", "p_through": 0.9}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

