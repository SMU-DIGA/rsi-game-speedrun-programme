# ==================================================================================================
# Super Mario Bros. (NES) -- level 8-4  |  超级马里奥兄弟（NES） 关 8-4
# Pass rate on 100 random variants never used before: 71/100; fastest finish 3683 frames (61.38 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 71/100；最快一局 3683 帧（60 帧/秒，约 61.38 秒）。
#
# What this file is / 这是什么:
#   A game-playing policy written by an AI agent (Claude Sonnet). Below this header the code is the
#   agent's own file, unchanged (only local paths are masked). The agent's comments are kept.
#   这是 AI 代理（Claude Sonnet）写的通关策略。本说明以下是代理写的原文件，未改动（只隐去了本地路径），代理自己的注释都保留。
#
# How it is run / 怎么运行:
#   The referee calls play(sim) once on a fresh random variant. Every frame the policy may read
#   sim.screen() -- the RGB pixels of the game window, nothing else (no game memory, no positions) --
#   and must call sim.step(keys) to hold a set of keys for one frame (keys: Left / Right / Up / Down, z = jump (A), x = run (B)).
#   Death or finishing is judged by the referee, not reported to the policy.
#   裁判在一个新的随机变体上调用一次 play(sim)。策略每一帧只能用 sim.screen() 读取游戏画面像素（读不到游戏内存和坐标），
#   用 sim.step(keys) 按住一组按键走一帧。死亡还是通关由裁判判定，不告诉策略。
#
# The agent's summary of its approach / 代理自己对思路的概括:
#   PB reactive runner (pillar-pipe entry, float-pipe, oracle water, bro hop) + Bowser type switch
#   : early left drift -> dash under, else jump over.
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _masks                         line 34    
#   def _solid                         line 42    
#   def _teal                          line 50    
#   def _log                           line 54    
#   def _floating                      line 59    
#   class Policy                       line 78    
#   def _edge_gap                      line 411   
#   def _is_water                      line 429   
#   class Swimmer                      line 436   underwater section: hold Right, stroke (A pulses) while a block is ahead of the body
#   def _wkeys                         line 488   
#   class EndCtl                       line 511   
#   class BroCtl                       line 607   castle Hammer Bro (teal sprite standing on the floor): hop + steer in the air to stomp it
#   class PodGate                      line 826   lava pit guarded by a podoboo (fixed world x, jumps out of the lava): back off, wait until it is gone, then ru
#   def play                           line 890   
#
# The dict _RSIGSR_FILES at the end holds the agent's other files: memory.json (data its play() reads,
# e.g. recorded key sequences) and assess.py (its own progress check). 文件末尾的 _RSIGSR_FILES 是代理的其它文件：
# memory.json（play() 读取的数据，例如录下的按键序列）和 assess.py（它自己的进度判断）。
# ==================================================================================================

# idea: PB reactive runner (pillar-pipe entry, float-pipe, oracle water, bro hop) + Bowser type switch: early left drift -> dash under, else jump over.
import numpy as np

_SH_MARGIN_PIPE = 56
_TEST_LOOP2 = True
_GEO_FRAMES = 70
_GEO_LO = 5
_GEO_HI = 10
_T2_SCROLL = 2560
_LEAD_MID = 26
_EH_SCROLL = 2462
_COYOTE = True
_EDGE_HOLD = 18
_SH_K = 300
_SH_K2 = 600
_SH_MARGIN = 14
_PIPE_HOLD = 9
_WALL_HOLD = 40
_DEFAULT_TEST = False
_LOG = True
_STEPDOWN = False
_WAIT_H2 = True
_TEST_PILLAR = True
_TEST_TALL = False
_PIPE_HOLD_TALL = 16

_PATTERN = ["Left"] * 5 + ["Down"] * 10
for _i in range(4):
    _PATTERN += ["Left"] * 2 + ["Down"] * 10
for _i in range(8):
    _PATTERN += ["Right"] * 2 + ["Down"] * 10


def _masks(img):
    r = img[:, :, 0].astype(np.int16); g = img[:, :, 1].astype(np.int16); b = img[:, :, 2].astype(np.int16)
    grey = (r == g) & (g == b) & (r >= 120)
    brown = (r == 172) & (g == 124) & (b == 0)
    skin = (r == 252) & (g == 160) & (b == 68)
    return grey, brown, skin


def _solid(grey, y0, y1, x0, x1):
    H, W = grey.shape
    y0 = max(0, y0); y1 = min(H, y1); x0 = max(0, x0); x1 = min(W, x1)
    if y1 <= y0 or x1 <= x0:
        return 0
    return int(grey[y0:y1, x0:x1].sum())


def _teal(img):
    return (img[:, :, 0] == 0) & (img[:, :, 1] == 136) & (img[:, :, 2] == 136)


def _log(*a):
    if _LOG:
        print(*a, flush=True)


def _floating(grey):
    g = grey[56:205, :]
    anyc = g.any(axis=0)
    first = np.argmax(g, axis=0)
    cand = np.nonzero(anyc & (first > 20) & (first < 72))[0]
    xs = []; tops = []
    for c in cand:
        col = g[:, c]
        f = int(first[c])
        rest = col[f:]
        nz = np.nonzero(~rest)[0]
        run = int(nz[0]) if len(nz) else len(rest)
        if 28 <= run <= 80 and f + run < 140:
            xs.append(int(c)); tops.append(f)
    if len(xs) < 20:
        return None
    return min(xs), max(xs), 56 + int(np.median(tops))


class Policy:
    def __init__(self):
        self.pending = False
        self.mx = None; self.fy = None; self.prev = None; self.scroll = 0
        self.fr = 0
        self.jump_left = 0; self.prev_z = False; self.jt = 0
        self.pstate = 0; self.pw = 0; self.pj = 0; self.pd = 0; self.pfy0 = 0
        self.pskip = -1; self.ref = None; self.pidx = 0
        self.entered = False; self.ent_fr = None; self.nok = 0
        self.events = []
        self.fpe = False; self.fpn = 0; self.fpw = 0; self.fpnone = 0
        self.post = False; self.fdfr = None; self.jrel = None; self.jfy0 = 0
        self.postw = False

    def track(self, img):
        grey, brown, skin = _masks(img)
        m = brown | skin
        m[:48] = False
        ys, xs = np.nonzero(m)
        if len(ys) == 0:
            return False, grey
        bm = brown.copy(); bm[:48] = False
        bys, bxs = np.nonzero(bm)
        if self.mx is not None and len(bys):
            kb = (abs(bxs - self.mx) < 40) & (abs(bys - self.fy) < 40)
            bys = bys[kb]; bxs = bxs[kb]
        if len(bys) >= 3:
            x0 = bxs.min() - 3; x1 = bxs.max() + 3; y0 = bys.min() - 10; y1 = bys.max() + 1
            sel = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
            if sel.sum() >= 3:
                ys = ys[sel]; xs = xs[sel]
                self.mx = int(round((xs.min() + xs.max()) / 2)); self.fy = int(ys.max())
                return self._scroll(img, grey)
        if self.mx is not None:
            k = (abs(xs - self.mx) < 40) & (abs(ys - self.fy) < 40)
            if k.sum() >= 3:
                ys = ys[k]; xs = xs[k]
        self.mx = int(round((xs.min() + xs.max()) / 2)); self.fy = int(ys.max())
        return self._scroll(img, grey)

    def _scroll(self, img, grey):
        g = img[60:200, :, 1].astype(np.int16)
        sc = 0
        if self.prev is not None:
            best = None
            for s in range(0, 5):
                d = np.abs(g[:, 128:250 - s] - self.prev[:, 128 + s:250]).mean()
                if best is None or d < best[0] - 1e-9:
                    best = (d, s)
            sc = best[1]
        self.scroll += sc
        self.prev = g
        return True, grey

    def _geo_keys(self, grey, mx, fy):
        row = grey[min(239, fy + 2)]
        if not row[max(0, min(255, mx))]:
            return None
        x0 = mx
        while x0 > 0 and row[x0 - 1]:
            x0 -= 1
        x1 = mx
        while x1 < 255 and row[x1 + 1]:
            x1 += 1
        if not (24 <= x1 - x0 + 1 <= 44):
            return None
        pm = getattr(self, "_gprev", None)
        self._gprev = (x0, mx)
        v = 0.0 if pm is None else (pm[0] - x0) + (mx - pm[1])
        dx = mx - x0
        if v > 0.5:
            return {"Left"}
        if v < -0.5:
            return {"Right"}
        if dx < _GEO_LO:
            return {"Right"}
        if dx > _GEO_HI:
            return {"Left"}
        return {"Down"}

    def _fp_keys(self, fp, mx, fy, ground):
        xmin, xmax, pt = fp
        if self.fpe:
            self.fpn += 1
            return {"Down"}
        on_top = abs(fy - pt) <= 3 and ground and xmin - 3 <= mx <= xmax + 3
        if on_top:
            self.fpw += 1
            if mx < xmin + _FP_ENTER_DX and self.fpw < 40:
                return {"Right"}
            self.fpe = True
            if self.fdfr is None:
                self.fdfr = self.fr
            _log("FLOAT PIPE: on top, Down. mx", mx, "xmin", xmin, "pt", pt)
            return {"Down"}
        if fy < pt + 4 and xmin - 45 <= mx <= xmax + 2:
            if mx > xmax - 4:
                return {"Left", "x"}
            return {"Right", "x"}
        return None

    def act(self, img):
        fr = self.fr
        self.fr += 1
        ok, grey = self.track(img)
        keys = {"Right", "x"}
        if fr < 56:
            keys = {"Right"} if fr < 12 else {"Right", "x"}
            if 40 <= fr < 56:
                keys = keys | {"z"}
            return keys
        if self.pstate == 3 and not ok:
            self.nok += 1
            if self.nok >= 4 and not self.entered:
                self.entered = True; self.ent_fr = fr; self.pstate = 0
                self.events.append(("entered", fr, self.scroll, self.pidx))
                _log("ENTERED (mario lost) at frame", fr, "scroll", self.scroll, "pipe", self.pidx)
            elif self.nok >= 4 and self.entered:
                self.pstate = 0; self.pskip = self.scroll + 60
                self.events.append(("entered2", fr, self.scroll, self.pidx))
                _log("ENTERED AGAIN (mario lost) at frame", fr, "scroll", self.scroll, "pipe", self.pidx)
        elif ok:
            self.nok = 0
        if not ok:
            return keys if self.pstate != 3 else {"Down"}
        mx, fy = self.mx, self.fy
        ground = _solid(grey, fy + 1, fy + 3, mx - 3, mx + 3) > 0
        if ground:
            self.cyo = 3
        elif getattr(self, "cyo", 0) > 0 and fy == getattr(self, "pfy_", None):
            self.cyo -= 1
        else:
            self.cyo = 0
        self.pfy_ = fy
        grounded = ground or self.cyo > 0
        d0 = None; h = 0; h2 = 0
        if fy > 40:
            # post-water: hammers (grey/white sprites) fly at head height -> only count foot-level obstacles
            band = grey[max(0, fy - (9 if self.postw else 26)):fy - 3, mx + 8:mx + 118]
            cols = band.any(axis=0)
            if cols.any():
                d0 = int(np.argmax(cols))
                xx = mx + 8 + d0 + 1
                colv = grey[48:fy - 3, min(xx, 255)][::-1]
                nz = np.nonzero(~colv)[0]
                h = int(nz[0]) + 3 if len(nz) else 100
                for dx in (1, 4, 8):
                    colv = grey[48:fy - 3, min(mx + 8 + d0 + dx, 255)][::-1]
                    nz = np.nonzero(~colv)[0]
                    h2 = max(h2, int(nz[0]) + 3 if len(nz) else 100)
        lead = 12 if h <= 20 else (_LEAD_MID if h <= 36 else 34)
        self.dbg = (ground, d0, h, h2, fy, mx)
        teal = _teal(img)
        wait = False
        if d0 is not None and (h >= 30 or (_WAIT_H2 and self.entered and h2 >= 28)) and d0 < 100:
            tw = teal[60:max(61, fy - 12), max(0, mx - 4):mx + 8 + d0 + 44]
            if tw.sum() > 0:
                wait = True
        if wait and d0 > lead:
            keys = set()
        fp = _floating(grey)
        if fp is None:
            self.fpnone += 1
            if self.fpnone > 40:
                self.fpe = False; self.fpw = 0
        else:
            self.fpnone = 0
        if fp is not None and self.pstate == 0:
            fk = self._fp_keys(fp, mx, fy, ground)
            if fk is not None:
                self.prev_z = "z" in fk
                return fk
        if self.pstate == 0 and not self.pending:
            self.holdback = False
            if (ground and d0 is not None and 22 <= h2 <= 52 and d0 <= (22 if h2 >= 40 else 28)
                    and self.jump_left <= 0 and not self.prev_z and self.scroll > self.pskip):
                x0 = min(255, mx + 8 + d0)
                reg = img[max(0, fy - 24):max(1, fy - 4), x0:min(256, x0 + 14)]
                ispipe = reg.size > 0 and float((reg == 252).all(axis=2).mean()) >= 0.25
                if ispipe:
                    idx = self.pidx
                    test = _DEFAULT_TEST or (_TEST_PILLAR and fy < 190 and not self.entered)
                    if _TEST_LOOP2 and self.fdfr is not None and self.scroll > _T2_SCROLL and h2 < 40 and not self.postw:
                        test = True
                    self.pidx += 1
                    self.phold = _PIPE_HOLD if h2 < 40 else _PIPE_HOLD_TALL
                    if test:
                        tw = teal[60:max(61, fy - 12), max(0, mx - 4):mx + 8 + d0 + 44]
                        self.pfy0 = fy
                        if tw.sum() > 0:
                            self.pstate = 1; self.pw = 0
                        else:
                            self.pstate = 2; self.pj = 0
                        _log("pipe", idx, "TEST f", fr, "scroll", self.scroll, "fy", fy)
                    else:
                        self.pskip = self.scroll + 60
                        _log("pipe", idx, "skip f", fr, "scroll", self.scroll)
        if self.pstate == 1:
            keys = set(); self.pw += 1
            x0 = mx + 8 + (d0 if d0 is not None else 20)
            tw = teal[60:max(61, fy - 12), max(0, mx - 4):x0 + 44]
            if tw.sum() == 0:
                self.pstate = 2; self.pj = 0
            elif self.pw > 400:
                self.pstate = 0; self.pskip = self.scroll + 80
        if self.pstate == 2:
            self.pj += 1
            keys = {"Right", "x", "z"} if self.pj <= self.phold else {"Right"}
            if self.pj > self.phold + 3 and ground:
                if fy < self.pfy0 - 12:
                    self.pstate = 3; self.pd = 0; self.ref = img[48:].astype(np.int16)
                    _log("landed on pipe top f", fr, "fy", fy, "scroll", self.scroll)
                elif self.pj > self.phold + 10:
                    self.pstate = 0; self.pskip = self.scroll + 60
            if self.pstate == 2 and self.pj > 70:
                self.pstate = 0; self.pskip = self.scroll + 60
        if self.pstate == 3:
            geo = self.fdfr is not None
            gk = None
            if geo and self.pd < _GEO_FRAMES:
                gk = self._geo_keys(grey, mx, fy)
            if gk is not None:
                keys = gk
            else:
                pi = max(0, self.pd - _GEO_FRAMES) if geo else self.pd
                keys = {_PATTERN[min(pi, len(_PATTERN) - 1)]}
            self.pd += 1
            if self.pd >= len(_PATTERN) + (_GEO_FRAMES if geo else 0):
                self.pstate = 0; self.pskip = self.scroll + 60
        self.post = self.fdfr is not None and _SH_K <= self.fr - self.fdfr <= _SH_K2
        if self.pstate == 0 and not getattr(self, "holdback", False):
            if (grounded if _COYOTE else ground) and self.jump_left <= 0 and not self.prev_z:
                wall = d0 is not None and d0 <= lead and not (self.postw and max(h, h2) <= 8)
                edge = _solid(grey & (img[:, :, 0] < 230), fy + 1, fy + 30, mx + 6, mx + 11) < 3
                if edge and (_STEPDOWN or self.post or self.postw) and fy < 190 and int((img[min(239, fy + 31):min(240, fy + 58), max(0, mx + 6):min(256, mx + 11), 0] == 188).sum()) >= 8:
                    edge = False
                bro = False
                flame = False
                if self.postw and not (wall or edge) and not getattr(self, "noflame", False):
                    reg = img[max(0, fy - 30):fy + 4, mx + 6:min(256, mx + 6 + _BRO_D)]
                    bowser_here = int(((img[40:215, :, 0] == 0) & (img[40:215, :, 1] == 168) & (img[40:215, :, 2] == 0)).sum()) >= 90
                    if (not bowser_here) and reg.size and int(((reg[:, :, 0] == 228) & (reg[:, :, 1] == 92) & (reg[:, :, 2] == 16)).sum()) >= 25:
                        bro = True
                        _log("BRO jump fr", fr, "mx", mx, "fy", fy)
                    rf = img[max(0, fy - 18):max(1, fy - 3), mx + 8:min(256, mx + 8 + _FL_D)]
                    bw = getattr(self, "bowser", None)
                    if bw is not None and bowser_here and bw[0] <= 90:
                        rf = rf[:0]
                    if rf.size and int(((rf[:, :, 0] == 248) & (rf[:, :, 1] == 56) & (rf[:, :, 2] == 0)).sum()) >= 8:
                        bro = True; flame = True
                        _log("FLAME jump fr", fr, "mx", mx, "fy", fy)
                if wall or edge or bro:
                    self.jwhy = (fr, bool(wall), bool(edge), bool(bro), bool(flame))
                    eh = _EDGE_HOLD if (self.fdfr is not None and self.scroll >= _EH_SCROLL and not self.postw) else 40
                    self.pitj = None
                    if self.postw and edge and not wall:
                        eh = _PIT_EH
                        if not bro:
                            self.pitj = 0
                    self.jump_left = eh if (edge or not wall) else _WALL_HOLD
                    if wall and not edge and self.postw:
                        self.jump_left = _WALL_HOLD_PW if max(h, h2) <= 36 else _WALL_HOLD_PW + 6
                    if bro:
                        self.jump_left = _FL_HOLD if flame else _BRO_HOLD
                    self.jt = 0
                    self.jfy0 = fy
                    mg = _SH_MARGIN
                    if wall and d0 is not None:
                        x0 = min(255, mx + 8 + d0)
                        reg = img[max(0, fy - 24):max(1, fy - 4), x0:min(256, x0 + 14)]
                        if reg.size > 0 and float((reg == 252).all(axis=2).mean()) >= 0.25:
                            mg = _SH_MARGIN_PIPE
                    self.jrel = (min(100, max(h, h2)) + mg) if (wall and self.post) else None
            self.jt += 1
            if self.jump_left > 0 and self.jt > 5 and ground:
                self.jump_left = 0
            if self.jump_left > 0 and self.jrel is not None and self.jt > 3 and (self.jfy0 - fy) >= self.jrel:
                self.jump_left = 0
            if self.jump_left > 0:
                keys = keys | {"z"}
                self.jump_left -= 1
        if self.postw and ground and self.jump_left <= 0 and "z" not in keys and self.pstate == 0:
            gm = ((img[40:215, mx:, 0] == 0) & (img[40:215, mx:, 1] == 168) & (img[40:215, mx:, 2] == 0))
            if int(gm.sum()) >= 90:
                ys_, xs_ = np.nonzero(gm)
                bx0 = int(xs_.min()) - 6; bot = int(ys_.max()) + 40
                air = (fy - bot) >= _BOW_AIR
                self.bowser = (bx0, bot, air)
                if getattr(self, "dash_left", 0) > 0:
                    self.dash_left -= 1                        # committed dash under him
                elif bx0 <= _BOW_GAP:
                    if air and bx0 <= _BOW_NEAR + 4:
                        self.dash_left = 34                    # hovering high and we are at his lip: dash under him
                    elif _BOW_MODE == 1:
                        # creep up to the lip of his shadow, wait until he hovers high, then dash
                        keys = {"Left"} if bx0 < _BOW_BACK else ({"Right"} if bx0 > _BOW_NEAR else set())
                    elif _BOW_MODE == 2 and bx0 >= _BOW_BACK:
                        keys = {"Right", "x", "z"}             # full-speed full jump over him towards the axe
                        self.jump_left = 40; self.jt = 0; self.jfy0 = fy; self.jrel = None
        pj = getattr(self, "pitj", None)
        if pj is not None and self.pstate == 0:
            self.pitj = pj + 1
            if _SLOW_A <= pj < _SLOW_A + _SLOW_L:
                keys = (set(keys) - {"Right"}) | {"Left"}
        self.prev_z = "z" in keys
        return keys


_HL_N0 = 12
_HL_L = 10
_HL_COND = True
_POD_K2 = 0.06
_POD_TC = 50
_POD_LO2 = 112.5
_POD_LO_W = 105.0
_POD_HI2 = 150
_SLOW_A = 2
_SLOW_L = 0
_BOW_MODE = 0    # 0 = just run, 1 = wait for Bowser to hop high then dash, 2 = jump over him
_BOW_GAP = 58
_BOW_BACK = 6
_BOW_NEAR = 12
_BOW_AIR = 31
_WALL_HOLD_PW = 12
_FL_D = 46
_FL_HOLD = 12
_BRO_D = 36
_BRO_HOLD = 6
_FP_ENTER_DX = 5
_MACRO_GAP = 6
_MACRO_SKIP = 13


def _edge_gap(grey, mx, fy):
    row = grey[min(239, fy + 2)]
    x = int(mx)
    if x < 0 or x > 255 or not row[x]:
        return None
    while x < 255 and row[x]:
        x += 1
    return x - int(mx)


MACRO = [[["Right", "x"], 13], [["Right", "x", "z"], 25], [["Right", "x"], 6], [["Left", "x"], 1], [["Right", "x"], 7], [["Left", "x"], 4], [["Left", "x", "z"], 2], [["Right", "x", "z"], 10], [["Right", "x"], 5], [["Left", "x"], 1], [["Right", "x"], 5], [["Left", "x"], 4], [["Right", "x"], 2], [["Right", "x", "z"], 1], [["Left", "x", "z"], 4]]
_MACRO_FRAMES = 78
_MK = []
for _k, _n in MACRO:
    _MK += [set(_k)] * _n
_MK = _MK[:_MACRO_FRAMES]


def _is_water(img):
    return int(((img[::4, ::4, 0] == 0) & (img[::4, ::4, 1] == 88) & (img[::4, ::4, 2] == 248)).sum()) > 400


_SW_LOOK = 12


class Swimmer:
    """underwater section: hold Right, stroke (A pulses) while a block is ahead of the body"""
    def __init__(self):
        self.n = 0

    def act(self, img):
        self.n += 1
        r = img[:, :, 0]; g = img[:, :, 1]; b = img[:, :, 2]
        solid = ((r == 0) & (g == 168) & (b == 0)) | ((r == 184) & (g == 248) & (b == 184))
        brown = (r == 172) & (g == 124) & (b == 0)
        ys, xs = np.nonzero(brown[24:, :])
        keys = {"Right"}
        if len(ys) == 0:
            return keys
        ys = ys + 24
        mx = int((xs.min() + xs.max()) // 2); fy = int(ys.max()); top = int(ys.min())
        col = solid[:, mx + 6:mx + 6 + _SW_LOOK].any(axis=1)
        runs = []
        y = 34
        while y < 236:
            if col[y]:
                y += 1
                continue
            y0 = y
            while y < 236 and not col[y]:
                y += 1
            if y - y0 >= 18:
                runs.append((y0, y - 1))
        cy = (top + fy) / 2.0
        stroke = False
        info = None
        if runs:
            best = min(runs, key=lambda r: 0 if r[0] <= cy <= r[1] else min(abs(cy - r[0]), abs(cy - r[1])))
            info = best
            mid = (best[0] + best[1]) / 2.0
            tgt = min(max(mid + 8, best[0] + 17), best[1] - 1)
            if fy > tgt:
                stroke = True
        else:
            stroke = True
        if stroke and self.n % 4 < 2:
            keys.add("z")
        if _LOG and self.n % 20 == 0:
            _log("SWIM", self.n, "mx", mx, "fy", fy, "gap", info)
        return keys


# water route found offline by save/load lookahead (10-frame chunks: Right held, A strokes `d` of every 4 frames)
_WCH = ("R0 R0 R0 R0 R0 R0 R0 R1 R1 R1 R0 R0 R1 R0 R0 R0 R0 R1 R1 R1 R1 R0 R0 R1 R0 R0 R0 R0 R0 R0 R0 R1 R0 R0 R0 R0 R0 R0 R0 N0 "
        "R0 R0 R0 R0 R0 R1 R0 R0 R0 R0 R0 R0 R1 R0 R0 R0 R0 R1 R1 R1 R0 R0 R0 R1 R0 R0 R1 R1 R0 R4 R0 R0 R0 R0 R0 R1 R0 R4 R0 R0 R0 R0 R0 R0 R0 R0").split()


def _wkeys(wt):
    c = wt // 10
    i = wt % 10
    p = _WCH[min(c, len(_WCH) - 1)]
    k = set()
    if p[0] == "R":
        k.add("Right")
    elif p[0] == "L":
        k.add("Left")
    if c >= len(_WCH):
        k = {"Right"}
    elif (i % 4) < int(p[1]):
        k.add("z")
    return k


_E_WT = 684      # from here on (and once Mario hugs the end wall) the last firebar is handled reactively
_E_MX = 165
_E_HOVER = 70    # feet-y of the waiting altitude (top of the cavity)
_E_PIPE = 140    # feet-y of the exit-pipe mouth
_E_WAIT_MAX = 420


class EndCtl:
    def __init__(self):
        self.on = False; self.phase = 0; self.n = 0; self.hist = []; self.tw = 0

    def _mario(self, img):
        r = img[:, :, 0]; g = img[:, :, 1]; b = img[:, :, 2]
        brown = (r == 172) & (g == 124) & (b == 0)
        ys, xs = np.nonzero(brown[40:, :])
        if len(ys) == 0:
            return None
        ys = ys + 40
        return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())

    def act(self, img, wt):
        mb = self._mario(img)
        if mb is None:
            return {"Right"}
        bx0, bx1, by0, by1 = mb
        mx = (bx0 + bx1) // 2; fy = by1
        if not self.on:
            if wt >= _E_WT and mx >= _E_MX:
                self.on = True
                _log("ENDCTL on wt", wt, "mx", mx, "fy", fy)
            else:
                return None
        self.n += 1
        r = img[:, :, 0]; g = img[:, :, 1]; b = img[:, :, 2]
        fire = (r == 248) & (g == 56) & (b == 0)
        fire[:40] = False
        fire[by0 - 6:by1 + 5, max(0, bx0 - 6):bx1 + 7] = False
        box = int(fire[max(40, fy - 12):152, max(0, mx - 30):mx + 16].sum())
        near = int(fire[max(40, fy - 6):fy + 24, max(0, mx - 22):mx + 16].sum())
        box = box if box >= 12 else 0
        near = near if near >= 8 else 0
        self.hist.append(box > 0)
        self.hist = self.hist[-16:]
        k = {"Right"}
        if self.phase == 0:
            self.tw += 1
            if fy > _E_HOVER and (self.n % 4) < 2:
                k.add("z")
            was = any(self.hist[:-1])
            if (box == 0 and was) or self.tw > _E_WAIT_MAX:
                self.phase = 1
                _log("ENDCTL descend wt", wt, "fy", fy, "tw", self.tw)
        else:
            if near > 0 and fy < _E_PIPE - 24:
                self.phase = 0; self.tw = 0
                _log("ENDCTL abort wt", wt, "fy", fy, "near", near)
                k.add("z")
            elif fy >= _E_PIPE:
                if fy > _E_PIPE + 3 and (self.n % 4) < 2:
                    k.add("z")
        if _LOG and self.n % 15 == 0:
            _log("ENDCTL wt", wt, "ph", self.phase, "mx", mx, "fy", fy, "box", box, "near", near)
        return k


_PRACTICE = False
_PRACTICE_OFF = False
_BRO_JD = 70     # hop as soon as the bro is this close (it is first seen at dx~58)
_BRO_ZH = 12    # z hold frames of the hop (searched: 11 and 13 fail, 12 passes 13/14 variants)
_BRO_TOL = 5
_BRO_T99 = 97     # bro sprite pixel count >= 97 at some point of the approach = the "slow approach" bro variant
_BRO_PITWALK = False   # tried (t12): late short hops miss the bro (1/5)
_BRO_PITD = 150
_BRO_JD_W = 24
_BRO_ZH_W = 14
_BOW_ON = True
_BOW_ALWAYS_GO = True
_PIT_EH = 40
_BRO_DELAY = 0
_BOW_DEC_T = 21   # Bowser type decided at this many frames after he appears: early left drift (d=bx0-mx<=66) -> dash under, else jump over
_BOW_B_D = 66
_BOW_DASH = True# dash-under / retreat variants tested: hammers kill the retreating Mario, standing jumps fail
_BOW_STALL = 3
_BOW_LEDGE = True
_BOW_LEDGE_D = 14
_BOW_PASS = 40      # dash ends when Bowser's left edge is this far behind Mario
_BOW_SAFE_Y = 138   # Bowser green bottom above this y: Mario (head y~144) fits under him
_BOW_SLOPE = 1.1
_BOW_RISE = 10
_BOW_SPAN = 44
_BOW_START = 4
_BOW_KEEP = 95
_BOW_HOLD = 120    # wait (no keys) once Bowser is closer than this and cannot be passed
_BOW_AIRD = 10
_BOW_WAIT_AIR = False   # True = run under hovering Bowser: tried (t6), died under him 2/3; keep the jump-over rule
_BOW_J = 52
_BOW_JS = 44    # jump trigger for the "S" Bowser phase (d>=_BOW_SD at the first trigger frame): 0/4 died at J=50, J 42/44 passed both checkpoints
_BOW_SD = 170    # bx0 of Bowser's green at the first trigger frame (<=170 = "S" phase, >=171 = normal phase)
_BOW_JN = 50
_N_DASH = False
_BOW_H = 28


class BroCtl:
    """castle Hammer Bro (teal sprite standing on the floor): hop + steer in the air to stomp it"""
    def __init__(self):
        self.on = False; self.n = 0; self.cool = 0; self.cnt = 0
        self.bow = False; self.bn = -1; self.tmax = 0

    @staticmethod
    def find(img, mx):
        t = _teal(img)
        t[:183] = False
        t[:, :max(0, mx - 30)] = False
        ys, xs = np.nonzero(t)
        if len(ys) < 25 or ys.max() < 195:
            return None
        return int(np.median(xs)), int(ys.min()), int(ys.max()), len(ys)

    def bowser(self, img, pol, keys):
        """Bowser bridge: run right; jump over him when his green body is _BOW_J px ahead (searched: J=50, H>=24)"""
        if pol.mx is None:
            return keys
        g = (img[40:215, :, 0] == 0) & (img[40:215, :, 1] == 168) & (img[40:215, :, 2] == 0)
        gg = g[:, max(0, pol.mx):]
        if not self.bow and int(gg.sum()) >= 90:
            self.bow = True
            _log("BOWSER phase start mx", pol.mx, "fy", pol.fy)
        if not self.bow:
            return keys
        k = {"Right", "x"}
        xs = np.nonzero(g.any(axis=0))[0]
        gy = np.nonzero(g[:, max(0, pol.mx):].any(axis=1))[0]
        y1 = int(gy.max()) + 40 if len(gy) else None
        if y1 is not None and int(g.sum()) >= 60:
            self.bgmax = max(getattr(self, "bgmax", 0), y1)
        air = y1 is not None and y1 < getattr(self, "bgmax", 0) - _BOW_AIRD
        self.bt = getattr(self, "bt", 0) + 1
        if self.bt % 3 == 0 and len(xs):
            _log("BOW", self.bt, "mx", pol.mx, "fy", pol.fy, "bx0", int(xs.min()), "y1", y1, "bgmax", getattr(self, "bgmax", 0), "air", air, "bn", self.bn)
        if self.bn < 0 and int(g.sum()) >= 60 and len(xs):
            bx0 = int(xs.min())
            d = bx0 - pol.mx
            py = getattr(self, "py1", None)
            rising = py is not None and y1 is not None and y1 < py
            if y1 is not None:
                self.py1 = y1
            if getattr(self, "btype", None) is None:
                if self.bt >= _BOW_DEC_T:
                    self.btype = "B" if d <= _BOW_B_D else "A"
                    _log("BOWSER type", self.btype, "d", d, "bt", self.bt)
                else:
                    pol.jump_left = 0; pol.prev_z = False
                    return {"Right", "x"}
            use_dash = _BOW_DASH and self.btype == "B"
            if use_dash and getattr(self, "dash", False):
                if (not air) and d > 0:
                    self.dash = False          # he landed in front of us: back to the jump-over rule
                    _log("BOWSER dash aborted (landed ahead) mx", pol.mx, "bx0", bx0)
                if not self.dash:
                    pass
                elif d < -_BOW_PASS or y1 is None:
                    self.dash = False
                    self.bn = 0
                    _log("BOWSER passed -> jump for the axe mx", pol.mx, "bx0", bx0)
                else:
                    pol.jump_left = 0; pol.prev_z = False
                    return {"Right", "x"}
            elif use_dash and air and d <= _BOW_HOLD:
                okf = (_BOW_SAFE_Y - y1) * _BOW_SLOPE + (_BOW_RISE if rising else 0)
                ext = (bx0 + _BOW_SPAN - pol.mx) / 3.0 + (_BOW_START if self.bt - getattr(self, "lastgo", 0) > 3 else 0)
                if ext <= okf or _BOW_ALWAYS_GO:
                    self.dash = True
                    _log("BOWSER dash under mx", pol.mx, "bx0", bx0, "y1", y1, "okf", okf, "ext", ext)
                    return {"Right", "x"}
                pol.jump_left = 0; pol.prev_z = False
                if d <= _BOW_HOLD:
                    # Bowser is up but we cannot get through in time: brake (running momentum slides ~35 px) and wait for him to land
                    # retreat so that a running start (>=~40 px) is available when he lands
                    if d < _BOW_KEEP and pol.mx > 40:
                        return {"Left"}
                    return set()
                return {"Right", "x"}
            elif d <= _BOW_J and (getattr(self, "bsub", None) is None or d <= (_BOW_JS if self.bsub == "S" else _BOW_JN)):
                if getattr(self, "bsub", None) is None:
                    # Bowser phase subclass: he is 1-2 frames "ahead" (d>=49 at the first trigger frame) -> jump later
                    self.bsub = "S" if (d + pol.mx) <= _BOW_SD else "N"
                    _log("BOWSER sub", self.bsub, "d", d, "bt", self.bt)
                    if self.bsub == "N" and _N_DASH:
                        self.btype = "B"          # experimental: dash under the hovering Bowser for the normal phase too
                        return {"Right", "x"}
                    if (d > _BOW_JS) if self.bsub == "S" else (d > _BOW_JN):
                        pol.jump_left = 0; pol.prev_z = False
                        return {"Right", "x"}
                self.hn = 0
                self.bn = 0
                _log("BOWSER jump mx", pol.mx, "fy", pol.fy, "bx0", bx0, "y1", y1, "air", air)
        if self.bn > _BOW_H:
            # landed against the axe ledge (Right held, no movement): jump again
            if pol.mx == getattr(self, "lmx", None) and pol.fy == getattr(self, "lfy", None):
                self.stall = getattr(self, "stall", 0) + 1
            else:
                self.stall = 0
            if self.stall >= _BOW_STALL:
                self.stall = 0; self.bn = 0
                _log("BOWSER blocked -> re-jump mx", pol.mx, "fy", pol.fy)
        self.lmx = pol.mx; self.lfy = pol.fy
        dbg = getattr(pol, "dbg", None)
        if (_BOW_LEDGE and (self.bn < 0 or self.bn > _BOW_H) and dbg and dbg[0] and dbg[1] is not None
                and dbg[1] <= _BOW_LEDGE_D and pol.fy >= 156 and not pol.prev_z and pol.mx > 150):
            self.bn = 0
            _log("BOWSER ledge ahead -> jump mx", pol.mx, "d0", dbg[1], "h", dbg[2])
        if self.bn >= 0:
            self.bn += 1
            if self.bn <= _BOW_H:
                k.add("z")
        pol.jump_left = 0; pol.prev_z = "z" in k
        return k

    @staticmethod
    def _pod(img, mx):
        m = (img[:, :, 0] == 248) & (img[:, :, 1] == 56) & (img[:, :, 2] == 0)
        m[:60] = False; m[203:] = False; m[:, :mx + 14] = False
        ys, xs = np.nonzero(m)
        if len(ys) < 80 or len(ys) > 112 or ys.max() - ys.min() > 17 or xs.max() - xs.min() > 15:
            return None
        return int(ys.min())

    def _danger(self):
        ps = self.ps
        if len(ps) < 2:
            _log("POD? no samples", ps)
            return False
        n1, y1 = ps[-1]
        n0, y0 = ps[0]
        for (na, ya) in ps:
            if n1 - na >= 3:
                n0, y0 = na, ya
        if n1 - n0 < 2:
            return False
        na = (n1 + n0) / 2.0 - (y1 - y0) / (2.0 * _POD_K2 * (n1 - n0))
        apex = y1 - _POD_K2 * (n1 - na) ** 2
        yc = apex + _POD_K2 * (_POD_TC - na) ** 2
        lo = _POD_LO_W if getattr(self, "walk_h", False) else _POD_LO2
        dg = lo < yc < _POD_HI2
        _log("POD? samples", ps[0], ps[-1], "na", round(na, 1), "apex", round(apex, 1), "yc", round(yc, 1), "walk", getattr(self, "walk_h", False), "danger", dg)
        return dg

    def act(self, img, pol, keys):
        if _BOW_ON:
            if self.bow or (pol.pstate == 0 and pol.mx is not None):
                keys = self.bowser(img, pol, keys)
            if self.bow:
                return keys
        if pol.mx is None or pol.pstate != 0:
            return keys
        mx, fy = pol.mx, pol.fy
        ground = pol.dbg[0] if getattr(pol, "dbg", None) else False
        if self.cool > 0:
            self.cool -= 1
        b = self.find(img, mx)
        if b is not None and not self.on and self.cool <= 0:
            self.tmax = max(self.tmax, b[3])
        walk = self.tmax >= _BRO_T99
        if _BRO_PITWALK and not self.on and getattr(pol, "de", None) is not None and pol.de <= _BRO_PITD:
            walk = True     # pit right behind the bro: stomp late and short instead of a long hop that lands at the lava edge at full speed
        jd = _BRO_JD_W if walk else _BRO_JD
        zh = _BRO_ZH_W if walk else _BRO_ZH
        if not self.on:
            if (_GATE_ON and getattr(pol, "de", None) is not None and pol.de <= 30 and b is not None and b[0] - mx < 0
                    and not getattr(pol, "noflame", False)):
                return keys                   # bro is behind us and a pit is at our feet: never hop in place there
            if b is None or not ground or self.cool > 0 or b[0] - mx > jd:
                if walk and b is not None and b[0] - mx < 110:
                    return {"Right"}          # this bro variant needs a slow approach + late hop
                return keys
            if self.cnt == 0 and getattr(self, "dl", 0) < _BRO_DELAY and not walk:
                self.dl = getattr(self, "dl", 0) + 1
                _log("BRO delay", self.dl, "mx", mx, "bro", b)
                return set()
            self.on = True; self.n = 0; self.cnt += 1
            _log("BRO hop start mx", mx, "fy", fy, "bro", b, "dx", b[0] - mx, "walk", walk, "tmax", self.tmax)
        self.n += 1
        k = {"Right"} if walk else {"x", "Right"}
        if b is not None:
            err = b[0] - mx
            if err < 2:
                k.discard("Right")
            if err < -8:
                k.add("Left")
        if self.n <= zh:
            k.add("z")
        if _HL_COND:
            if self.n == 1:
                self.ps = []; self.hl = None; self.walk_h = bool(walk)
            py = self._pod(img, mx)
            if py is not None:
                self.ps.append((self.n, py))
            if self.n == _HL_N0 and self.hl is None:
                self.hl = self._danger()
        else:
            self.hl = True
        if _HL_L > 0 and self.hl and _HL_N0 <= self.n < _HL_N0 + _HL_L:
            k.discard("Right"); k.add("Left")
        if self.n % 5 == 0:
            _log("BRO hop n", self.n, "mx", mx, "fy", fy, "bro", b)
        if self.n > 8 and ground:
            self.on = False; self.cool = 40
            _log("BRO landed n", self.n, "mx", mx, "fy", fy, "bro", b)
        pol.jump_left = 0; pol.prev_z = "z" in k
        return k


_GATE_ON = False   # A/B vs archived c6 (10 variants): c6 5/10, with the gate 1-2/10 -> the gate/bro-hop interplay hurts
_G_BACK = 22     # frames of backing off Left when a podoboo guards the pit
_G_ABSENT = 3    # frames the podoboo must be gone before the run-up starts
_G_DE = 45      # pit-edge distance (px) at which the gate engages
_G_MAXWAIT = 300
_G_BRO_AWARE = False   # skipping the back-off when a bro is behind made Mario slide into the pit (t9: 4/5 deaths)
_G_BRO_AFTER = 20


class PodGate:
    """lava pit guarded by a podoboo (fixed world x, jumps out of the lava): back off, wait until it is gone, then run up and jump"""
    def __init__(self):
        self.st = 0; self.n = 0; self.absent = 0; self.cnt = 0; self.seen = False; self.hop = False

    @staticmethod
    def pod(img, pol):
        fm = (img[:, :, 0] == 248) & (img[:, :, 1] == 56) & (img[:, :, 2] == 0)
        fm[:60] = False; fm[209:] = False
        fm[max(0, pol.fy - 20):pol.fy + 3, max(0, pol.mx - 10):pol.mx + 11] = False
        ys, xs = np.nonzero(fm)
        if len(ys) < 12 or pol.fy > 215:
            return None
        bb = (int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max()))
        if (bb[3] - bb[2] < 13 and bb[3] < 185) or bb[1] - bb[0] > 26:
            return None          # bridge girders / flames / lava glow are not a podoboo (14x16 sprite)
        return bb

    def act(self, img, pol, keys, ground, de):
        if pol.mx is None:
            return keys
        p = self.pod(img, pol)
        if self.st == 0:
            if (de is not None and de <= _G_DE and p is not None and p[1] > pol.mx
                    and pol.pstate == 0 and self.cnt < 6):
                self.st = 1; self.n = 0; self.cnt += 1; self.absent = 0
                _log("GATE engage fr", pol.fr, "mx", pol.mx, "de", de, "pod", p)
            else:
                return keys
        self.n += 1
        if self.st == 1:
            pol.jump_left = 0; pol.prev_z = False
            if self.n % 4 == 0 or p is not None:
                _log("GATE", self.st, self.n, "mx", pol.mx, "fy", pol.fy, "pod", p)
            if self.n <= _G_BACK and self.hop:
                k2 = set(keys); k2.discard("Right"); k2.add("Left")      # keep the bro-hop's A (height) but shed speed in the air
                return k2
            if self.n <= _G_BACK:
                t = _teal(img); t[:183] = False
                tx = np.nonzero(t.any(axis=0))[0]
                if _G_BRO_AWARE and self.n > _G_BRO_AFTER and len(tx) >= 8 and int(tx.min()) < pol.mx:
                    pass                    # a hammer bro is behind us: backing off would walk into it
                else:
                    return {"Left"}
            if p is None:
                self.absent += 1
            else:
                self.absent = 0
            if self.absent >= _G_ABSENT or self.n > _G_MAXWAIT:
                self.st = 2; self.n = 0
                _log("GATE go fr", pol.fr, "mx", pol.mx)
            return set()
        if self.st == 2:
            if self.n > 80 or (de is not None and de > 60 and self.n > 40):
                self.st = 0
            if self.n % 4 == 0:
                _log("GATE go", self.n, "mx", pol.mx, "fy", pol.fy, "de", de, "pod", p, "keys", sorted(keys),
                     "dbg", getattr(pol, "dbg", None), "pst", pol.pstate, "jl", pol.jump_left, "pz", pol.prev_z)
            if self.n <= 40 and ground and (p is None or True):
                pass
            return keys
        return keys


def play(sim):
    pol = Policy()
    broc = BroCtl()
    gate = PodGate()
    _i0 = sim.screen()
    _bm = (_i0[48:, :, 0] == 172) & (_i0[48:, :, 1] == 124) & (_i0[48:, :, 2] == 0)
    _by, _bx = np.nonzero(_bm)
    _real = len(_by) > 0 and 40 <= (_bx.min() + _bx.max()) / 2 <= 56 and 100 <= _by.max() + 48 <= 120
    if (not _real) and not _PRACTICE_OFF:
        _log("PRACTICE start (Mario not at level start)")
        pol.fr = 100; pol.postw = True; pol.entered = True; pol.fdfr = -10 ** 6
    elif _PRACTICE:
        pol.fr = 100; pol.postw = True; pol.entered = True; pol.fdfr = -10 ** 6
    swim = Swimmer()
    endc = EndCtl()
    wt = 0
    still = 0
    last = None
    queue = []
    mdone = False
    for fr in range(7990):
        img = sim.screen()
        if last is not None and np.array_equal(img, last):
            still += 1
        else:
            still = 0
        last = img
        if still > 90:
            _log("frozen at", fr, "scroll", pol.scroll)
            return
        if _is_water(img):
            if wt % 50 == 0:
                _log("WATER wt", wt, "fr", fr)
            pol.postw = True
            ek = endc.act(img, wt)
            sim.step(ek if ek is not None else _wkeys(wt))
            wt += 1
            continue
        if queue:
            pol.track(img)
            keys = queue.pop(0)
        else:
            pol.noflame = gate.st != 0
            keys = pol.act(img)
            if (not mdone) and pol.entered and pol.pstate == 0 and 1650 <= pol.fr <= 1900 and pol.mx is not None:
                g = _edge_gap(_masks(img)[0], pol.mx, pol.fy)
                if pol.fy is not None and 158 <= pol.fy <= 162 and g is not None and g <= _MACRO_GAP:
                    mdone = True
                    _log("MACRO start at", fr, "pol.fr", pol.fr, "gap", g)
                    queue = list(_MK[_MACRO_SKIP:])
                    keys = queue.pop(0)
            if pol.postw:
                keys = broc.act(img, pol, keys)
        if pol.postw and pol.mx is not None:
            fm = (img[:, :, 0] == 248) & (img[:, :, 1] == 56) & (img[:, :, 2] == 0)
            fm[:60] = False; fm[200:] = False
            fm[max(0, pol.fy - 20):pol.fy + 3, max(0, pol.mx - 10):pol.mx + 11] = False
            fys, fxs = np.nonzero(fm)
            gr = _masks(img)[0]
            row = gr[214]
            de = None
            for xx in range(pol.mx + 4, 256):
                if not row[xx]:
                    de = xx - pol.mx
                    break
            if len(fys) and fr % 3 == 0:
                _log("PD", fr, "mx", pol.mx, "fy", pol.fy, "de", de, "fire", len(fys),
                     (int(fxs.min()), int(fxs.max()), int(fys.min()), int(fys.max())),
                     "jl", pol.jump_left, "why", getattr(pol, "jwhy", None))
            pol.de = de
            if _GATE_ON and not broc.bow and not queue:
                gate.hop = broc.on
                keys = gate.act(img, pol, keys, bool(pol.dbg[0]) if getattr(pol, "dbg", None) else False, de)
        if fr % 200 == 0:
            _log("f", fr, "scroll", pol.scroll, "mx", pol.mx, "fy", pol.fy, "pstate", pol.pstate, "pidx", pol.pidx, "fpe", pol.fpe)
        sim.step(keys)

# ==== rsigsr bundle: the agent's other files (assess.py, memory.json), kept with this solver ====
_RSIGSR_FILES = {'assess.py': 'import numpy as np\n\n\ndef assess(frame, ctx):\n    try:\n        f = np.asarray(frame)\n        brown = (f[48:, :, 0] == 172) & (f[48:, :, 1] == 124) & (f[48:, :, 2] == 0)\n        st = (ctx or {}).get("stuck") or {}\n        stuck = bool(st.get("is")) if isinstance(st, dict) else False\n        status = "dead" if brown.sum() == 0 else ("stuck" if stuck else "running")\n        return {"status": status, "where": {}, "progress": None, "resources": {}, "threats": [],\n                "stuck": {"is": stuck}, "outlook": {"label": "ok", "p_through": 0.2}, "note": ""}\n    except Exception:\n        return None\n', 'memory.json': '{"human": {}, "past": {}}\n'}

