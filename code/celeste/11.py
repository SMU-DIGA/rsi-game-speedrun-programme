# ==================================================================================================
# Celeste Classic (PICO-8) -- room 11  |  蔚蓝经典版（PICO-8） 房间 11
# Pass rate on 100 random variants never used before: 99/100; fastest finish 91 frames (3.03 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 99/100；最快一局 91 帧（30 帧/秒，约 3.03 秒）。
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
#   hold Right through spawn, sync on first movement response, replay beam-searched key trace (fou
#   nd offline)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _pos                           line 9     
#   def solve                          line 17    
#   def play                           line 21    
#   def _play                          line 29    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: hold Right through spawn, sync on first movement response, replay beam-searched key trace (found offline)
import numpy as np

SK = np.array((255, 204, 170))

TRACE = [[["Right", "Up"], 5], [["Right"], 2], [["Right", "z"], 5], [["Right", "Up", "z"], 4], [["Right"], 7], [["Right", "z"], 7], [["Right"], 5], [["Down", "Right", "z"], 3], [["Left", "Up", "x", "z"], 1], [["Left", "Up"], 5], [["Up", "z"], 2], [[], 5], [["Right", "Up"], 5], [["Right", "z"], 7], [["Up", "z"], 3], [["Up", "x", "z"], 1], [["Up"], 5], [["Down", "Right"], 4], [["Right"], 2], [["Right", "Up", "z"], 5], [["Right", "Up"], 2], [["Up", "x", "z"], 1], [["Up"], 5]]


def _pos(s):
    m = (np.abs(s.astype(int) - SK).sum(2) < 6)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), len(ys))


def solve(sim):
    return []


def play(sim):
    log = []
    try:
        _play(sim, log)
    finally:
        print("LOG", log)


def _play(sim, log):
    synced = False
    xs = []
    for i in range(150):
        sim.step({"Right"})
        p = _pos(sim.screen())
        log.append(p)
        xs.append(p[0] if p is not None and p[2] >= 5 and p[1] in (114, 115, 116) else -99)
        if len(xs) >= 3 and p is not None and p[2] >= 9 and p[1] in (114, 115):
            x3, x2, x1 = xs[-1], xs[-2], xs[-3]
            if x3 in (16, 17) and x3 - 3 <= x2 < x3 and x2 - 3 <= x1 < x2 and x1 >= 11:
                synced = True
                break
    print("sync", synced, i, p)
    flat = []
    for k, n in TRACE:
        flat += [k] * n
    for k in flat[2:]:
        sim.step(set(k))
        log.append(_pos(sim.screen()))
    for _ in range(40):
        sim.step({"Up"})
        log.append(_pos(sim.screen()))

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {}, "progress": None, "resources": {}, "threats": [],\n            "stuck": {"is": False}, "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

