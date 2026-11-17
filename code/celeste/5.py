# ==================================================================================================
# Celeste Classic (PICO-8) -- room 5  |  蔚蓝经典版（PICO-8） 房间 5
# Pass rate on 100 random variants never used before: 100/100; fastest finish 72 frames (2.40 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 72 帧（30 帧/秒，约 2.40 秒）。
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
#   evolved (offline, save/load hill-climb) macro trace for room 5, replayed from a fixed delay af
#   ter the spawn-landing is detected on screen (skin-pixel count drop); stable-standing fallback
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _skin                          line 12    
#   def _frames                        line 20    
#   def play                           line 29    
#   def solve                          line 72    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: evolved (offline, save/load hill-climb) macro trace for room 5, replayed from a fixed delay after the spawn-landing is detected on screen (skin-pixel count drop); stable-standing fallback
import numpy as np

TRACE = [["R", 3], ["Rz", 1], ["R", 12], ["ULz", 2], ["Rz", 7], ["Uzx", 1], ["Ux", 1], ["UL", 7], ["", 3],
         ["URz", 15], ["URz", 1], ["Rx", 1], ["URz", 3], ["Uzx", 2], ["URz", 3], ["U", 8], ["", 10]]
_NAMES = {"L": "Left", "R": "Right", "U": "Up", "D": "Down", "z": "z", "x": "x"}
STAND_X, STAND_Y = 12.1, 92.1
LAND_DELAY = 5     # idle frames after the landing detection before the trace starts (4 works, 3 does not)
MAX_WAIT = 90


def _skin(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()), len(xs))


def _frames():
    out = []
    for ks, n in TRACE:
        keys = {_NAMES[c] for c in ks}
        for _ in range(n):
            out.append(keys)
    return out


def play(sim):
    frames = _frames()
    try:
        prev = 0
        stable = 0
        landed = False
        for i in range(MAX_WAIT):
            sim.step(set())
            s = _skin(sim.screen())
            c = 0 if s is None else s[2]
            if i >= 1 and s is not None and prev >= 10 and c <= 8 and s[1] > 88:
                landed = True
                for _ in range(LAND_DELAY):
                    sim.step(set())
                break
            prev = c
            if (s is not None and c >= 11 and abs(s[0] - STAND_X) < 0.5 and abs(s[1] - STAND_Y) < 0.5):
                stable += 1
                if stable >= 2 and i > 22:
                    break
            else:
                stable = 0
        print("start trace at frame", sim.frame, "landed", landed)
        miss = 0
        for j, k in enumerate(frames):
            sim.step(set(k))
            s = _skin(sim.screen())
            if s is None:
                miss += 1
                if miss >= 12:
                    print("player gone at trace frame", j, "frame", sim.frame)
                    break
            else:
                miss = 0
                if j % 5 == 0:
                    print("j", j, "frame", sim.frame, "skin", round(s[0], 1), round(s[1], 1))
        print("trace done at frame", sim.frame)
        for _ in range(20):
            sim.step(set())
    except Exception as e:
        print("play ended:", type(e).__name__, e)


def solve(sim):
    return []

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    m = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n    n = int(m.sum())\n    if n == 0:\n        return {"status": "dead", "where": {"hazard": None, "confidence": 0.3}, "progress": None,\n                "resources": {}, "threats": [], "stuck": {"is": False},\n                "outlook": {"label": "bad", "p_through": 0.3}, "note": "no player pixels"}\n    ys, xs = np.nonzero(m)\n    prog = float(max(0.0, min(1.0, 1.0 - ys.mean() / 100.0)))\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.5}, "progress": prog,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.8}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

