# ==================================================================================================
# Celeste Classic (PICO-8) -- room 22  |  蔚蓝经典版（PICO-8） 房间 22
# Pass rate on 100 random variants never used before: 100/100; fastest finish 142 frames (4.73 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 142 帧（30 帧/秒，约 4.73 秒）。
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
#   sync on first run response (skin x>=24), then replay a pre-searched open-loop trace (chest -> 
#   orb -> 2 dashes -> wall jumps)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _trace                         line 19    
#   def _skin                          line 26    
#   def solve                          line 35    
#   def play                           line 39    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: sync on first run response (skin x>=24), then replay a pre-searched open-loop trace (chest -> orb -> 2 dashes -> wall jumps)
import numpy as np

_SK = np.array([255, 204, 170])

R, Z, X, U, N = "Right", "z", "x", "Up", None
# (keys, count) run-length encoded trace, found offline with save/load beam search
_RLE = [
    (("Right", "z"), 3), (("Right", "Up", "x"), 1), (("Right", "Up"), 4), (("Right", "x"), 1),      # to the chest
    (("Right",), 7), (("Right", "x"), 1), (("Right",), 12),
    ((), 46),                                                                                        # chest opening (frozen)
    (("z",), 12), ((), 16),                                                                          # jump, touch orb, freeze
    (("Right", "Up", "x"), 1), (("Right", "Up"), 4), ((), 2),                                        # dash 1
    (("Up", "x"), 1), (("Up",), 4),                                                                  # dash 2
    (("Right",), 6), (("Right", "z"), 2), (("Right",), 4), (("Right", "z"), 2), (("Right",), 4),    # wall jumps
]


def _trace():
    out = []
    for k, c in _RLE:
        out += [set(k)] * c
    return out


def _skin(sim):
    s = sim.screen()
    m = (s == _SK).all(axis=2)
    p = np.argwhere(m)
    if len(p) == 0:
        return None
    return int(p[:, 1].min()), int(p[:, 0].min()), len(p)


def solve(sim):
    return []


def play(sim):
    n = 0
    synced = False
    while n < 160:
        sim.step({"Right"})
        n += 1
        a = _skin(sim)
        if a and a[0] >= 22 and a[2] >= 8:
            synced = True
            break
    print("sync", synced, n, flush=True)
    for k in _trace():
        sim.step(set(k))
    print("trace done", _skin(sim), flush=True)
    # fall back: keep climbing
    for i in range(300):
        sim.step({"Up"})

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.3}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.7}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

