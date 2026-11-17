# ==================================================================================================
# Celeste Classic (PICO-8) -- room 20  |  蔚蓝经典版（PICO-8） 房间 20
# Pass rate on 100 random variants never used before: 100/100; fastest finish 115 frames (3.83 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 115 帧（30 帧/秒，约 3.83 秒）。
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
#   wait for the crumble-floor collapse (sync event), then replay a pre-searched balloon/dash rout
#   e (B1->B2->under lip->B4->B3->top corridor->exit)
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _obs                           line 8     
#   def solve                          line 16    
#   def play                           line 20    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: wait for the crumble-floor collapse (sync event), then replay a pre-searched balloon/dash route (B1->B2->under lip->B4->B3->top corridor->exit)
import numpy as np

FULL = [["z"], ["z"], ["Right", "Up", "x"], ["Right", "Up"], ["Right", "Up"], ["Right", "Up"], [], ["z"], ["z"], [], [], ["Right", "Up", "x"], ["Right", "Up"], ["Right", "Up"], ["Right", "Up"], ["Up", "x"], ["Up"], ["Up"], ["Up"], [], [], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Up", "Right", "x"], ["Up", "Right"], ["Up", "Right"], ["Up", "Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Right"], ["Up", "x"], ["Up"], ["Up"], ["Up"], ["z"], ["z"], [], [], ["Left", "Up", "x"], ["Left", "Up"], ["Left", "Up"], ["Left", "Up"], [], [], ["Up", "x"], ["Up"], ["Up"], ["Up"], ["Up", "x"], ["Up"], ["Up"], ["Up"], ["Left", "x"], ["Left"], ["Left"], ["Left"], ["Left", "Up", "x"], ["Left", "Up"], ["Left", "Up"], ["Left", "Up"], [], ["Up", "x"], ["Up"], ["Up"], ["Up"], [], [], ["z"], ["z"], [], [], ["Up", "x"], ["Up"], ["Up"], ["Up"]]
TAIL = 30


def _obs(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(len(xs))


def solve(sim):
    return []


def play(sim):
    prev_y = None
    gap = 0
    trig = False
    for _ in range(400):
        sim.step(set())
        gap += 1
        o = _obs(sim.screen())
        if o is None:
            continue
        x, y, n = o
        if n >= 11 and 85 <= y <= 86 and prev_y is not None and prev_y <= 84 and gap <= 3:
            trig = True
            break
        prev_y = y
        gap = 0
    if not trig:
        return
    for f in FULL:
        sim.step(set(f))
    for _ in range(TAIL):
        sim.step({"Up"})

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'def assess(frame, ctx):\n    return {"status": "running", "where": {"hazard": None, "confidence": 0.0}, "progress": None,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.5}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

