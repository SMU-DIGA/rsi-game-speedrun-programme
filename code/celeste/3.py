# ==================================================================================================
# Celeste Classic (PICO-8) -- room 3  |  蔚蓝经典版（PICO-8） 房间 3
# Pass rate on 100 random variants never used before: 100/100; fastest finish 130 frames (4.33 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 130 帧（30 帧/秒，约 4.33 秒）。
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
#   beam-searched (offline, save/load, potential-guided) per-frame key demo for room 3, followed c
#   losed-loop with skin-pixel position re-alignment
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def skin                           line 13    
#   def keys_of                        line 23    
#   def expected_before                line 27    
#   def play                           line 35    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: beam-searched (offline, save/load, potential-guided) per-frame key demo for room 3, followed closed-loop with skin-pixel position re-alignment
import numpy as np

KEYN = [(1, 'Left'), (2, 'Right'), (4, 'Up'), (8, 'Down'), (16, 'z'), (32, 'x')]
READY = (11, 107)
# entry t = (x, y, keymask) : keymask pressed on frame t, (x, y) = skin position observed AFTER that frame (None = hidden)
DEMO = [[11, 107, 18], [12, 103, 18], [14, 100, 18], [16, 98, 18], [None, None, 38], [None, None, 6], [21, 95, 6], [24, 92, 6], [29, 87, 6], [None, None, 6], [35, 81, 6], [36, 80, 0], [37, 77, 0], [40, 75, 0], [43, 75, 0], [45, 73, 0], [47, 71, 0], [49, 71, 0], [51, 71, 0], [53, 71, 0], [53, 71, 0], [53, 73, 0], [53, 73, 0], [53, 75, 0], [53, 77, 0], [53, 79, 0], [53, 82, 0], [52, 87, 33], [52, 87, 1], [50, 90, 1], [48, 93, 1], [46, 96, 1], [44, 99, 1], [42, 102, 1], [None, None, 34], [38, 108, 2], [39, 106, 2], [42, 103, 2], [43, 99, 2], [45, 96, 2], [47, 92, 2], [49, 90, 2], [51, 87, 2], [53, 84, 2], [55, 82, 2], [57, 80, 2], [59, 78, 2], [61, 76, 2], [63, 76, 2], [65, 74, 2], [67, 74, 2], [69, 74, 2], [71, 76, 2], [None, None, 38], [None, None, 6], [76, 78, 6], [None, None, 6], [82, 69, 6], [None, None, 6], [None, None, 6], [90, 60, 0], [92, 58, 0], [95, 56, 0], [98, 54, 0], [100, 52, 0], [102, 52, 0], [104, 52, 0], [106, 52, 0], [108, 52, 0], [108, 52, 0], [108, 52, 0], [108, 54, 0], [107, 56, 17], [107, 58, 17], [105, 55, 17], [103, 52, 17], [101, 49, 0], [99, 47, 0], [99, 45, 0], [99, 43, 0], [99, 41, 1], [99, 41, 1], [97, 39, 1], [95, 39, 1], [93, 39, 0], [91, 39, 0], [91, 41, 1], [89, 41, 1], [87, 43, 0], [87, 43, 0], [None, None, 37], [None, None, 5], [None, None, 5], [None, None, 5], [80, 37, 5], [75, 35, 5], [73, 32, 5], [68, 30, 0], [67, 28, 0], [64, 26, 0], [61, 24, 0], [59, 22, 0], [57, 22, 0], [55, 20, 0], [53, 20, 0], [51, 20, 2], [52, 22, 2], [52, 22, 2], [54, 24, 2], [56, 24, 0], [56, 26, 0], [57, 29, 1], [55, 31, 1], [55, 35, 16], [55, 31, 16], [56, 28, 2], [56, 25, 2], [58, 23, 2], [60, 21, 2], [None, None, 38], [None, None, 6], [None, None, 6], [67, 12, 6], [73, 12, 6], [74, 8, 6], [80, 6, 6], [79, 4, 0], [82, 1, 0], [85, 0, 0], [None, None, 0]]
N = len(DEMO)
TOL = 3
KEYS0 = {'Right', 'z'}


def skin(im):
    sk = (im[:, :, 0] == 255) & (im[:, :, 1] == 204) & (im[:, :, 2] == 170)
    ys, xs = np.nonzero(sk)
    if len(xs) == 0:
        return None
    if xs.max() - xs.min() > 6 or ys.max() - ys.min() > 4:
        return None
    return (int(xs.min()), int(ys.min()))


def keys_of(mask):
    return {n for b, n in KEYN if mask & b}


def expected_before(i):
    # position the skin should have before pressing demo frame i
    if i == 0:
        return READY
    d = DEMO[i - 1]
    return None if d[0] is None else (d[0], d[1])


def play(sim):
    # 1) wait for the spawn animation to finish (skin stable at the start spot)
    # Hold the demo's first keys (Right+z) through the spawn animation: they are ignored until the
    # player becomes controllable, then the jump starts at once (same as demo frames 0..1).
    # The first observation of the post-jump position (12,103) means demo frame 1 was just played.
    cur = 0
    p = None
    for _ in range(150):
        sim.step(KEYS0)
        p = skin(sim.screen())
        if p is not None and p[0] >= 12 and 99 <= p[1] <= 104:
            cur = 2
            break
    print('started at frame', sim.frame, p, 'cur', cur)
    if cur == 0:
        # fallback: wait until the skin is stable at the start spot
        n = 0
        lp = None
        for _ in range(150):
            sim.step(set())
            p = skin(sim.screen())
            n = n + 1 if (p is not None and p == lp and p[1] >= 106) else 0
            lp = p
            if n >= 3:
                break
    # 2) follow the demo, realigning on the observed position
    ndiv = 0
    try:
        for t in range(900):
            if cur >= N:
                sim.step({'Up'})
                continue
            p = skin(sim.screen())
            e = expected_before(cur)
            if p is not None and e is not None and abs(p[0] - e[0]) + abs(p[1] - e[1]) > TOL:
                # look for a nearby demo frame matching the observation
                best, bi = 1e9, cur
                for i in range(max(0, cur - 4), min(N - 1, cur + 8) + 1):
                    ei = expected_before(i)
                    if ei is None:
                        continue
                    c = abs(p[0] - ei[0]) + abs(p[1] - ei[1]) + 0.6 * abs(i - cur)
                    if c < best:
                        best, bi = c, i
                if ndiv < 12:
                    ndiv += 1
                    print('div t', t, 'cur', cur, 'p', p, 'exp', e, '-> ', bi, 'cost', round(best, 1))
                if best <= TOL + 3:
                    cur = bi
            sim.step(keys_of(DEMO[cur][2]))
            cur += 1
    except Exception as ex:
        print('end', type(ex).__name__, ex, 'cur', cur, 'frame', sim.frame)

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    sk = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n    ys, xs = np.nonzero(sk)\n    if len(xs) == 0 or xs.max() - xs.min() > 6 or ys.max() - ys.min() > 4:\n        return {"status": "running", "where": {"hazard": None, "confidence": 0.3}, "progress": None,\n                "resources": {}, "threats": [], "stuck": {"is": False},\n                "outlook": {"label": "ok", "p_through": 0.7}, "note": "player not visible (dash/spawn/death)"}\n    y = int(ys.min())\n    prog = max(0.0, min(1.0, (107 - y) / 110.0))\n    danger = y > 112\n    return {"status": "danger" if danger else "running", "where": {"hazard": "pit" if danger else None, "confidence": 0.6},\n            "progress": prog, "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "bad" if danger else "good", "p_through": 0.3 if danger else 0.8}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

