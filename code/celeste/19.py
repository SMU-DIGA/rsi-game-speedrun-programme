# ==================================================================================================
# Celeste Classic (PICO-8) -- room 19  |  蔚蓝经典版（PICO-8） 房间 19
# Pass rate on 100 random variants never used before: 100/100; fastest finish 65 frames (2.17 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 65 帧（30 帧/秒，约 2.17 秒）。
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
#   hold Right+z through spawn, sync on first jump response (x>=4,y<=112), replay beam-searched 2-
#   frame-macro trace
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _look                          line 8     
#   def _macros                        line 16    
#   def solve                          line 30    
#   def play                           line 34    
# ==================================================================================================

# idea: hold Right+z through spawn, sync on first jump response (x>=4,y<=112), replay beam-searched 2-frame-macro trace
import numpy as np

SK = np.array((255, 204, 170))
TRACE = [13, 0, 17, 1, 8, 8, 8, 8, 9, 0, 0, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 17, 0, 0, 1, 0, 0, 0, 0]


def _look(s):
    m = (np.abs(s.astype(int) - SK).sum(2) < 6)
    ys, xs = np.where(m)
    if len(ys) == 0:
        return None
    return (int(xs.min()), int(ys.min()), len(ys))


def _macros():
    M = []
    for dk in ([], ["Left"], ["Right"]):
        d = set(dk)
        M.append([d, d])
        M.append([d | {"z"}, d | {"z"}])
        M.append([d, d | {"z"}])
        M.append([d | {"z"}, d])
    dirs = [["Up"], ["Down"], ["Left"], ["Right"], ["Up", "Left"], ["Up", "Right"], ["Down", "Left"], ["Down", "Right"]]
    for k in dirs:
        M.append([set(k) | {"x"}] + [set(k)] * 3)
    return M


def solve(sim):
    return []


def play(sim):
    p = None
    synced = False
    for i in range(200):
        sim.step({"Right", "z"})
        p = _look(sim.screen())
        if p and p[2] >= 11 and p[0] >= 4 and p[1] <= 112:
            synced = True
            break
    print("sync", synced, i, p)
    M = _macros()
    for mi in TRACE:
        for ks in M[mi]:
            sim.step(set(ks))
    print("trace done", sim.frame)
    for j in range(60):
        sim.step({"Up"})

