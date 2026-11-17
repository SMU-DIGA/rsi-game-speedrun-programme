# ==================================================================================================
# Celeste Classic (PICO-8) -- room 7  |  蔚蓝经典版（PICO-8） 房间 7
# Pass rate on 100 random variants never used before: 99/100; fastest finish 132 frames (4.40 s at 30 fps).
# 在 100 个从未用过的随机变体上通过 99/100；最快一局 132 帧（30 帧/秒，约 4.40 秒）。
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
#   closed-loop cloud-riding bot: static white mask + moving cloud tracking, jump/dash up cloud ro
#   ws
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _static                        line 15    
#   def cmask                          line 24    
#   def skin                           line 28    
#   def find_clouds                    line 36    
#   def solve                          line 74    
#   def play                           line 78    
#   def _play                          line 85    
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: closed-loop cloud-riding bot: static white mask + moving cloud tracking, jump/dash up cloud rows
import numpy as np

WH = (255, 241, 232)
LI = (194, 195, 199)
STATIC_B64 = "eNrtkzGOgzAQRb8EkkuKbehAORmSuUM60J4gV3JJsYfwEZwuVWYxM+CxVxu09WakGJ7G3//bwcBaVRwcthrWn4kvfnvFmHiEVBOHAPzGMr+Pjzpxvfe130semG0cFgmzc5rf8votU4+J80zYEtQgihxHTkDPOM6k3dSKJdudl4Kdck8J2H1PAHFPb+x+JKigzs1hMD5xt565eaj+2mtUP555o/UB/aH/3PR1qa+03v3wR+GPP/qf6s/8db9zA6zWLyPs1r9cdj2fPnPKd71u+tCDHokN1XiqPmaxN7P8KY6X64g4hmc29xvH8LJdL9sMjWxMPoIg+vst1xOJXrY7cww4vf3kf5Qvuck5lHymL/3O2Oa82JO8ZR7KOX38cg5417v+c83FlSDKr0yY8ivl2/zKOeRX9gsfBVev5n8DfhSJOQ=="
_S = None
_VERBOSE = True
_JUMP_TOL = 10
_MAXGAP = 36
_DASH_AT = 5
_JUMP_TOL_DASH = 22


def _static():
    global _S
    if _S is None:
        import base64, zlib
        raw = zlib.decompress(base64.b64decode(STATIC_B64))
        _S = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))[:128 * 128].reshape(128, 128).astype(bool)
    return _S


def cmask(img):
    return ((img == WH).all(2) | (img == LI).all(2))


def skin(img):
    m = (img[:, :, 0] == 255) & (img[:, :, 1] == 204) & (img[:, :, 2] == 170)
    ys, xs = np.nonzero(m)
    if len(xs) == 0:
        return None
    return (float(xs.mean()), float(ys.mean()), len(xs))


def find_clouds(img):
    S = _static()
    m = cmask(img) & ~S
    m[:36] = False
    runs = []  # (y, x0, x1)
    for y in range(36, 128):
        xs = np.nonzero(m[y])[0]
        if len(xs) < 7:
            continue
        s = xs[0]
        p = xs[0]
        for x in list(xs[1:]) + [9999]:
            if x != p + 1:
                if p - s + 1 >= 7:
                    runs.append((y, int(s), int(p)))
                s = x
            p = x
    # union by overlap in consecutive rows
    comps = []
    for r in runs:
        placed = False
        for c in comps:
            last = c[-1]
            if last[0] == r[0] - 1 and r[1] <= last[2] + 1 and r[2] >= last[1] - 1:
                c.append(r)
                placed = True
                break
        if not placed:
            comps.append([r])
    out = []
    for c in comps:
        if len(c) >= 3:
            x0 = min(r[1] for r in c)
            x1 = max(r[2] for r in c)
            out.append([c[0][0] - 1, x0, x1])
    return out


def solve(sim):
    return []


def play(sim, stop=None):
    try:
        return _play(sim, stop)
    except Exception as e:
        print("play ended:", type(e).__name__, e)


def _play(sim, stop=None):
    L, R, U, D = "Left", "Right", "Up", "Down"
    t = 0
    prev = None
    last_good = None
    air_t = -1       # frames since jump press, -1 = not in jump
    need_dash = False
    target = None
    tcx = 0.0
    zprev = False
    feet0 = 0.0
    fell = False
    prev_clouds = {}
    vrow = {39: 0.65, 63: -0.65, 87: 0.65, 103: -0.65}
    hist = []
    log = []
    while t < 1500:
        img = sim.screen()
        sk = skin(img)
        if sk is not None and sk[2] >= 9:
            pos = (sk[0], sk[1])
            last_good = pos
        else:
            pos = None
        cl = find_clouds(img)
        # row velocity estimate
        for c in cl:
            top = c[0]
            cc = (c[1] + c[2]) / 2.0
            if top in prev_clouds:
                for pc in prev_clouds[top]:
                    d = cc - pc
                    if abs(d) < 1.6 and abs(c[2] - c[1]) >= 13:
                        vrow[top] = 0.7 * vrow.get(top, d) + 0.3 * d
        prev_clouds = {}
        for c in cl:
            prev_clouds.setdefault(c[0], []).append((c[1] + c[2]) / 2.0)
        keys = set()
        # dead reckoning of the target cloud's centre (the player sprite occludes it)
        if air_t >= 0 and target is not None:
            tv = vrow.get(target[0], 0.0)
            tcx += tv
            for c in cl:
                if abs(c[0] - target[0]) <= 1 and abs(c[2] - c[1]) >= 13:
                    ccx = (c[1] + c[2]) / 2.0
                    if abs(ccx - tcx) < 5:
                        tcx = ccx
        if pos is None:
            # dashing / dead: keep last intention
            if air_t >= 0:
                air_t += 1
            sim.step(set())
            zprev = False
            t += 1
            continue
        cx, cy = pos
        feet = cy + 4.0
        hist.append((cx, cy))
        vy = hist[-1][1] - hist[-2][1] if len(hist) >= 2 else 0
        if air_t >= 0 and (vy >= 0.5 or air_t >= (20 if need_dash else 15)):
            fell = True
        # grounded on a cloud?
        sup = None
        for c in cl:
            if abs(c[0] - feet) <= 2.6 and c[1] - 3 <= cx <= c[2] + 3 and abs(c[2] - c[1]) >= 12 and vy > -0.5:
                sup = c
        onground = sup is not None or (feet >= 119 and cx < 33 and abs(vy) < 0.1)
        if air_t >= 0 and not fell:
            onground = False
        if stop is not None and stop(sim, cl, pos, onground, sup):
            return log
        if onground and air_t > 3:
            air_t = -1
            need_dash = False
            target = None
        if onground and air_t < 0 and sup is not None and sup[0] <= 42:
            # final stage: on the top cloud row -> jump + dash up through the roof
            if 54 <= cx <= 66:
                log.append((sim.frame, round(cx, 1), 'FINAL'))
                print("FINAL at frame", sim.frame, round(cx, 1))
                for k in ([set()] if zprev else []) + [{"z"}] * 5 + [{U, "x"}] + [set()] * 25:
                    sim.step(k)
                return log
            if cx < 54:
                kk = {R} if cx < sup[2] - 3 else set()
            else:
                kk = {L} if cx > sup[1] + 4 else set()
            sim.step(kk)
            zprev = False
            t += 1
            continue
        if onground and air_t < 0 and not zprev:
            # choose next row up
            cands = [c for c in cl if feet - _MAXGAP <= c[0] <= feet - 8 and abs(c[2] - c[1]) >= 12]
            if cands:
                best = None
                tops = sorted(set(c[0] for c in cands))
                for rowtop in tops:  # highest row first
                    bb = None
                    for c in cands:
                        if c[0] != rowtop:
                            continue
                        cc = (c[1] + c[2]) / 2.0
                        pred = cc + vrow.get(c[0], 0) * 12
                        dx = pred - cx
                        if c[0] <= 42 and pred > 60:
                            continue
                        if feet - c[0] > 18 and 44 <= cx <= 86:
                            continue  # dash-up from under the spike block is unsafe
                        if bb is None or abs(dx) < abs(bb[0]):
                            bb = (dx, c)
                    if bb is None:
                        continue
                    tl = _JUMP_TOL_DASH if feet - bb[1][0] > 18 else _JUMP_TOL
                    if abs(bb[0]) <= tl:
                        best = bb
                        break
                tol = 1e9
                if best and abs(best[0]) <= tol:
                    target = best[1]
                    gap = feet - target[0]
                    need_dash = gap > 18
                    feet0 = feet
                    air_t = 0
                    fell = False
                    tcx = (target[1] + target[2]) / 2.0
                    log.append((sim.frame, round(cx, 1), round(feet, 1), 'jump', target, round(gap, 1)))
        if air_t >= 0:
            keys = set()
            dash_at = _DASH_AT if (feet0 - target[0]) <= 28 else 5
            if need_dash:
                if air_t < dash_at:
                    keys.add("z")
            elif air_t < 14:
                keys.add("z")
            if need_dash and air_t == dash_at:
                keys = {U, "x"}
            else:
                dx = tcx - cx
                if dx > 2 and not (cx > 99 and feet < 56):
                    keys.add(R)
                elif dx < -2 and not (cx < 28 and feet < 56):
                    keys.add(L)
            air_t += 1
            if air_t > 70:
                air_t = -1
        else:
            # idle on ground: ride/wait
            pass
        if _VERBOSE and (t % 10 == 0 or air_t == 1):
            print("f", sim.frame, "pos", round(cx, 1), round(cy, 1), "gr", bool(onground), "air", air_t, sorted(keys),
                  "tgt", (None if target is None else target[:1]), "cl", [tuple(c) for c in cl if c[0] < feet + 4 and c[0] > feet - 40])
        prev = pos
        sim.step(keys)
        zprev = "z" in keys
        t += 1
    return log

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    m = (frame[:, :, 0] == 255) & (frame[:, :, 1] == 204) & (frame[:, :, 2] == 170)\n    n = int(m.sum())\n    if n == 0:\n        return {"status": "running", "where": {"hazard": None, "confidence": 0.3}, "progress": None,\n                "resources": {}, "threats": [], "stuck": {"is": False},\n                "outlook": {"label": "ok", "p_through": 0.7}, "note": "no skin pixels (dash)"}\n    ys, xs = np.nonzero(m)\n    prog = float(max(0.0, min(1.0, 1.0 - ys.mean() / 120.0)))\n    status = "dead" if n > 40 else "running"\n    return {"status": status, "where": {"hazard": None, "confidence": 0.5}, "progress": prog,\n            "resources": {}, "threats": [], "stuck": {"is": False},\n            "outlook": {"label": "ok", "p_through": 0.8}, "note": ""}\n', 'memory.json': '{"human": {}, "past": {}}\n'}

