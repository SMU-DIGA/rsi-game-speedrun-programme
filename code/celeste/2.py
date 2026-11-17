# ==================================================================================================
# Celeste Classic (PICO-8) -- room 2  |  蔚蓝经典版（PICO-8） 房间 2
# Pass rate on 100 random variants never used before: 100/100; fastest finish 96 frames (3.20 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 96 帧（30 帧/秒，约 3.20 秒）。
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
#   beam-searched (offline, save/load) key trace for room 2, replayed from the detected "standing"
#    state; skin-pixel tracking for stand detection
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _decode                        line 9     
#   def _skin                          line 16    
#   def play                           line 24    
#   def solve                          line 58    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: beam-searched (offline, save/load) key trace for room 2, replayed from the detected "standing" state; skin-pixel tracking for stand detection
import numpy as np

TRACE = "Rx,R,R,R,z,z,z,z,,z,z,z,,,,,z,z,z,z,,z,z,z,,z,z,z,,,,,z,z,z,z,,,,,R,R,R,R,,,,,R,R,R,R,,,,,RUx,RU,RU,RU,z,z,z,z,,,,,,,,,R,R,R,R,z,z,z,z,,,,,z,z,z,z,,,,"
_READY_WAIT = 5
_NAMES = {"L": "Left", "R": "Right", "U": "Up", "D": "Down", "z": "z", "x": "x"}


def _decode():
    out = []
    for tok in TRACE.split(","):
        out.append({_NAMES[c] for c in tok})
    return out


def _skin(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()), len(xs))


def play(sim):
    keys = _decode()
    hist = []
    stand = None
    prev = None
    try:
        for i in range(300):
            sim.step(set())
            s = _skin(sim.screen())
            hist.append(s)
            if (s is not None and prev is not None and s[2] >= 12 and abs(s[1] - 116.08) < 0.3 and
                    abs(s[0] - 12.08) < 0.3 and abs(s[0] - prev[0]) < .01 and abs(s[1] - prev[1]) < .01):
                stand = i
                break
            if prev is not None and s is not None and prev[2] >= 12 and s[2] <= 10 and prev[1] > 112:
                # landing event of the spawn animation: control starts ~5 frames later
                for _ in range(_READY_WAIT):
                    sim.step(set())
                stand = i
                break
            prev = s
        print("stand wait", stand, "frame", sim.frame, hist[-1])
        for j, k in enumerate(keys):
            sim.step(set(k))
            if j % 40 == 0:
                s = _skin(sim.screen())
                print("j", j, "frame", sim.frame, "skin", None if s is None else (round(s[0]), round(s[1])))
        print("trace done at frame", sim.frame)
        for _ in range(30):
            sim.step(set())
    except Exception as e:
        print("play ended:", type(e).__name__, e)


def solve(sim):
    return []

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    m = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n    n = int(m.sum())\n    if n == 0:\n        return {"status": "dead", "where": {"hazard": None, "confidence": 0.3}, "progress": None,\n                "resources": {}, "threats": [], "stuck": {"is": False},\n                "outlook": {"label": "bad", "p_through": 0.2}, "note": "no player pixels"}\n    ys, xs = np.nonzero(m)\n    prog = float(max(0.0, min(1.0, 1.0 - ys.mean() / 120.0)))\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.5}, "progress": prog,\n            "resources": {}, "threats": [], "stuck": {"is": bool(ctx.get("stuck", {}) and ctx["stuck"].get("is"))}\n            if isinstance(ctx.get("stuck"), dict) else {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.8}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

