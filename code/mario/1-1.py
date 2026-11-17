# ==================================================================================================
# Super Mario Bros. (NES) -- level 1-1  |  超级马里奥兄弟（NES） 关 1-1
# Pass rate on 100 random variants never used before: 100/100; fastest finish 1340 frames (22.33 s at 60 fps).
# 在 100 个从未用过的随机变体上通过 100/100；最快一局 1340 帧（60 帧/秒，约 22.33 秒）。
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
#   PB geometric vision bot + a "v2" variant (wall scan looks at ALL step candidates so pipes behi
#   nd a low obstacle are not masked, earlier stuck-jump, earlier retreat) A/B-tested against the 
#   PB behaviour
#
# Main parts (line numbers refer to the code below this header) / 主要部分（行号指下面原文件的行号）:
#   def _sky                           line 20    
#   def _is_gameplay                   line 24    
#   def _find_mario                    line 29    
#   def _est_scroll                    line 45    Horizontal scroll (px, 0..5) between two frames from the ground rows (bricks have seams).
#   def _scan_pit                      line 66    Distance (px) from Mario's front to the next pit (>=4 columns of sky in the ground rows).
#   def _scan_walls                    line 84    All sharp steps up in the contiguous non-sky column height above the ground row (pipe,
#   class St                           line 115   
#   def _say                           line 146   
#   def _snap                          line 151   
#   def _unsnap                        line 159   
#   def decide                         line 167   
#   def run                            line 306   
#   def _load_ovr                      line 432   
#   def _load_ver                      line 443   
#   def play                           line 450   
#   def _mean                          line 459   
#   def solve                          line 463   Learning phase. A/B test: version 0 (PB behaviour, baseline, ovr never modified) against
#
# This policy also reads sim.memory; that data is in 1-1.memory.json next to this file.
# 这个策略还会读取 sim.memory，数据在同目录的 1-1.memory.json。
# ==================================================================================================

# idea: PB geometric vision bot + a "v2" variant (wall scan looks at ALL step candidates so pipes behind a low obstacle are not masked, earlier stuck-jump, earlier retreat) A/B-tested against the PB behaviour (v0) inside solve(); the winner version + learned world-x jump overrides are stored in sim.memory for play()
import sys
import time
import copy
import random
import numpy as np

# ---------------------------------------------------------------------------
# Palette facts (measured earlier): sky (104,136,252); Mario cap/overalls (248,56,0);
# ground top row y=208, ground bricks occupy y=208..239; small Mario standing: red rows end at y=206.
# Bush/hill slopes have single-column steps <= 5 px, goomba edge step ~10 px, pipes/stairs >= 16.
# After death / flagpole the emulator freezes: the screen stops changing.
# ---------------------------------------------------------------------------
GROUND_RED_YMAX = 206
SHORT_HOLD = 16
FULL_HOLD = 44
FINISH_WX = 3040


def _sky(img):
    return (img[:, :, 0] == 104) & (img[:, :, 1] == 136) & (img[:, :, 2] == 252)


def _is_gameplay(img):
    s = _sky(img[40:200])
    return float(s.mean()) > 0.30


def _find_mario(img, last_x):
    R = (img[:, :, 0] == 248) & (img[:, :, 1] == 56) & (img[:, :, 2] == 0)
    R[:32] = False
    if not R.any():
        return None
    if last_x is not None:
        lo = max(0, last_x - 30)
        hi = min(R.shape[1], last_x + 45)
        win = np.zeros_like(R)
        win[:, lo:hi] = R[:, lo:hi]
        if win.any():
            R = win
    ys, xs = np.nonzero(R)
    return int(xs.min()), int(xs.max()), int(ys.min()), int(ys.max())


def _est_scroll(prev, cur):
    """Horizontal scroll (px, 0..5) between two frames from the ground rows (bricks have seams)."""
    P = prev[212:236, :, 1].astype(np.int16)
    C = cur[212:236, :, 1].astype(np.int16)
    W = P.shape[1]
    res = []
    for s in range(0, 6):
        if s == 0:
            m = np.count_nonzero(P != C) / float(P.size)
        else:
            m = np.count_nonzero(P[:, s:] != C[:, :W - s]) / float(P.shape[0] * (W - s))
        res.append(m)
    s = int(np.argmin(res))
    if res[s] > 0.3:
        return 0
    for k in range(s):
        if res[k] <= res[s] + 0.002:
            return k
    return s


def _scan_pit(sky, front, L=120):
    """Distance (px) from Mario's front to the next pit (>=4 columns of sky in the ground rows)."""
    x0 = front + 1
    x1 = min(256, x0 + L)
    if x1 - x0 < 4:
        return None
    col = sky[216, x0:x1] & sky[224, x0:x1] & sky[232, x0:x1]
    run = 0
    for i in range(len(col)):
        if col[i]:
            run += 1
            if run >= 4:
                return i - 3
        else:
            run = 0
    return None


def _scan_walls(sky, front, hm, L=112):
    """All sharp steps up in the contiguous non-sky column height above the ground row (pipe,
    stairs, enemy standing on the floor, flag pole), nearest first.
    Returns a list of (distance, height above Mario's feet)."""
    x0 = front + 4
    x1 = min(256, x0 + L)
    if x1 - x0 < 6 or x0 < 1:
        return []
    sub = sky[30:208, x0 - 1:x1]
    f = sub[::-1]
    anys = f.any(axis=0)
    h = np.where(anys, f.argmax(axis=0), f.shape[0]).astype(int)
    step = h[1:] - h[:-1]
    rel = h[1:] - hm
    cand = np.nonzero((step >= 7) & (rel >= 8))[0]
    out = []
    last = -100
    for ci in cand:
        i = int(ci)
        if i - last < 4 and out:
            continue
        last = i
        x_edge = x0 + i
        H = int(h[1 + i:1 + i + 15].max()) - hm
        d = x_edge - (front + 1)
        out.append((d, H))
        if len(out) >= 6:
            break
    return out


class St(object):
    def __init__(self):
        self.frame = 0          # gameplay frames
        self.prev = None
        self.scroll = 0
        self.same = 0
        self.last_x = None
        self.missing = 0
        self.prev_fy = None
        self.still = 0
        self.jumping = False
        self.jf = 0
        self.hold = FULL_HOLD
        self.seen_up = False
        self.seen_down = False
        self.a_prev = False
        self.best_wx = -1000
        self.stuck = 0
        self.retreat = 0
        self.nret = 0
        self.boost = 0
        self.max_wx = 0
        self.njumps = 0
        self.log = None
        self.fy = 0
        self.wx = 0
        self.used = set()
        self.ovr = {}
        self.ver = 0


def _say(st, msg):
    if st.log is not None:
        st.log(msg)


def _snap(st):
    c = copy.copy(st)
    c.used = set(st.used)
    c.log = None
    c.ovr = None
    return c


def _unsnap(c, ovr, log):
    s = copy.copy(c)
    s.used = set(c.used)
    s.ovr = ovr
    s.log = log
    return s


def decide(st, img):
    sky = _sky(img)
    if st.prev is not None:
        st.scroll += _est_scroll(st.prev, img)
        if np.array_equal(st.prev, img):
            st.same += 1
        else:
            st.same = 0
    st.prev = img

    ver = st.ver
    stuck_jump = 3 if ver >= 2 else 4
    retreat_thr = 45 if ver >= 2 else 90

    keys = {"Right", "x"}
    m = _find_mario(img, st.last_x)
    if m is None:
        st.missing += 1
        if st.a_prev and st.missing < 30:
            keys.add("z")
        st.a_prev = "z" in keys
        return keys
    st.missing = 0

    xmin, xmax, ymin, ymax = m
    st.last_x = xmin
    front = xmax + 1
    fy = ymax
    wx = st.scroll + xmin - 2
    st.fy = fy
    st.wx = wx
    if wx > st.max_wx:
        st.max_wx = wx

    dy = 0 if st.prev_fy is None else fy - st.prev_fy
    st.prev_fy = fy
    st.still = st.still + 1 if dy == 0 else 0

    # progress / stuck bookkeeping
    if wx >= st.best_wx + 3:
        st.best_wx = wx
        st.stuck = 0
    else:
        st.stuck += 1

    # jump phase tracking
    if st.jumping:
        st.jf += 1
        if dy < 0:
            st.seen_up = True
        if dy > 0 and st.seen_up:
            st.seen_down = True
        if st.jf >= 5 and not st.seen_up:
            st.jumping = False
        elif st.seen_down and st.still >= 2:
            st.jumping = False
        elif st.jf > 130:
            st.jumping = False

    # retreat (fallback: back off to get a run-up)
    if st.retreat > 0:
        st.retreat -= 1
        if st.retreat == 0:
            st.best_wx = wx
            st.stuck = 0
        st.a_prev = False
        return {"Left"}

    hm = max(0, GROUND_RED_YMAX - fy)
    a = False
    if st.jumping:
        a = (not st.seen_down) and st.jf < st.hold
    else:
        grounded = st.still >= 2
        if grounded and st.frame > 60 and st.stuck >= retreat_thr:
            st.retreat = 25 + 10 * min(st.nret, 5)
            st.nret += 1
            st.boost = min(40, 8 * st.nret)
            _say(st, "f=%d RETREAT n=%d wx=%d" % (st.frame, st.nret, wx))
            st.a_prev = False
            return {"Left"}
        if grounded and not st.a_prev:
            want = None
            why = ''
            ovr = st.ovr or {}
            suppress = False
            for k, v in ovr.items():
                if v < 0 and k <= wx < k - v:
                    suppress = True
            # learned overrides: forced jumps keyed by world x
            for k in sorted(ovr):
                v = ovr[k]
                if v > 0 and k not in st.used and wx >= k:
                    st.used.add(k)
                    if wx <= k + 48:
                        want = int(v)
                        why = 'ovr@%d' % k
                        break
            if want is None and not suppress:
                dp = _scan_pit(sky, front)
                pit_seen = dp is not None
                if pit_seen and dp <= 26 + st.boost // 2:
                    want = FULL_HOLD
                    why = 'pit d=%d' % dp
                if want is None:
                    ws = _scan_walls(sky, front, hm)
                    if ver < 2:
                        ws = ws[:1]
                    for (d, H) in ws:
                        if H >= 26:
                            T = 30 + 0.62 * H + st.boost
                            if d <= T:
                                want = FULL_HOLD
                                why = 'wall d=%d H=%d' % (d, H)
                        else:
                            T = 40 + st.boost // 2
                            if d <= T:
                                want = FULL_HOLD if pit_seen else SHORT_HOLD
                                why = 'low d=%d H=%d' % (d, H)
                        if want is not None:
                            break
                if want is None and st.frame > 40 and st.stuck >= stuck_jump:
                    want = FULL_HOLD
                    why = 'stuck=%d' % st.stuck
            if want is not None:
                st.jumping = True
                st.jf = 0
                st.hold = want
                st.seen_up = False
                st.seen_down = False
                st.njumps += 1
                a = True
                _say(st, "f=%d JUMP h=%d [%s] wx=%d mx=%d fy=%d" % (st.frame, want, why, wx, xmin, fy))
    if a:
        keys.add("z")
    st.a_prev = a
    return keys


def run(sim, ovr, search, deadline, log, rng, step_cap, ver=0):
    st = St()
    st.ver = ver
    st.ovr = ovr
    st.log = log
    ckpts = []
    ck_n = 0
    fails = {}
    trial = set()
    deaths = 0
    ep_best = -1
    good_ovr = dict(ovr)
    steps = 0
    intro = 0
    last_ck_wx = -1000
    result = 'cap'
    fz = 6 if search else 90
    death_wxs = []
    if search:
        try:
            sim.save(0)
            ckpts.append({'slot': 0, 'wx': 0, 'n': 0, 'st': _snap(st)})
            ck_n = 1
        except Exception:
            search = False
            fz = 90
    while steps < step_cap:
        if deadline is not None and steps % 8 == 0 and time.time() > deadline:
            result = 'deadline'
            break
        img = sim.screen()
        if img.shape[1] != 256 and img.shape[1] % 256 == 0:
            k = img.shape[1] // 256
            img = img[::k, ::k]
        playing = False
        if _is_gameplay(img):
            playing = True
            # checkpoint (state corresponds to the screen we are looking at, before deciding)
            if (search and st.still >= 2 and not st.jumping and st.retreat == 0 and st.missing == 0
                    and st.frame > 5 and st.wx >= last_ck_wx + 40 and st.same == 0):
                slot = ck_n % 10
                ckpts = [c for c in ckpts if c['slot'] != slot]
                try:
                    sim.save(slot)
                    ckpts.append({'slot': slot, 'wx': st.wx, 'n': ck_n, 'st': _snap(st)})
                    ck_n += 1
                    last_ck_wx = st.wx
                    if st.wx > ep_best:
                        ep_best = st.wx
                        good_ovr = dict(ovr)
                except Exception:
                    pass
            keys = decide(st, img)
            ended = st.same >= fz
            early = search and st.missing == 0 and st.fy >= 226
            if ended or early:
                if not search:
                    result = 'frozen'
                    break
                if ended and st.wx >= FINISH_WX:
                    result = 'finished'
                    break
                # ---- death: backtrack to an earlier checkpoint and perturb ----
                deaths += 1
                dwx = st.wx
                death_wxs.append(int(dwx))
                if not ckpts:
                    result = 'nockpt'
                    break
                cands = [i for i, c in enumerate(ckpts) if c['wx'] <= dwx - 40]
                idx = cands[-1] if cands else 0
                cid = ckpts[idx]['n']
                fails[cid] = fails.get(cid, 0) + 1
                idx = max(0, idx - (fails[cid] - 1) // 4)
                c = ckpts[idx]
                ckpts = ckpts[:idx + 1]
                last_ck_wx = c['wx']
                try:
                    sim.load(c['slot'])
                except Exception:
                    result = 'loadfail'
                    break
                st = _unsnap(c['st'], ovr, log)
                for kk in list(ovr.keys()):
                    if c['wx'] < kk <= dwx + 60 and (kk in trial or rng.random() < 0.25):
                        del ovr[kk]
                        trial.discard(kk)
                lo = c['wx'] + 10
                hi = max(lo + 10, dwx + 4)
                n_add = 1 + (1 if rng.random() < 0.4 else 0)
                for _ in range(n_add):
                    kk = int(rng.uniform(lo, hi))
                    if rng.random() < 0.3:
                        v = -rng.choice([8, 14, 20, 28])
                    else:
                        v = rng.choice([8, 12, 18, 26, 34, 44])
                    ovr[kk] = v
                    trial.add(kk)
                if log is not None:
                    log("DEATH #%d at wx=%d -> ckpt wx=%d (fails=%d) ovr=%d" % (
                        deaths, dwx, c['wx'], fails[cid], len(ovr)))
                continue
        else:
            keys = {"Right", "x"}
            st.prev = None
            st.same = 0
            intro += 1
            if intro > 1500:
                result = 'intro-timeout'
                break
        try:
            sim.step(set(keys))
        except Exception as e:  # BudgetExceeded / game over
            result = 'exc:' + type(e).__name__
            break
        steps += 1
        if playing:
            st.frame += 1
            if log is not None and st.frame % 200 == 0:
                log("f=%d wx=%d maxwx=%d fy=%d still=%d jumping=%s stuck=%d scroll=%d" % (
                    st.frame, st.wx, st.max_wx, st.fy, st.still, st.jumping, st.stuck, st.scroll))
    return {"result": result, "frames": st.frame, "steps": steps, "intro": intro,
            "max_wx": st.max_wx, "last_wx": st.wx, "deaths": deaths, "death_wxs": death_wxs,
            "good_ovr": good_ovr, "ovr": ovr}


def _load_ovr(sim):
    ovr = {}
    try:
        raw = sim.memory.get('ovr') or {}
        for k, v in raw.items():
            ovr[int(k)] = int(v)
    except Exception:
        pass
    return ovr


def _load_ver(sim):
    try:
        return int(sim.memory.get('ver', 0))
    except Exception:
        return 0


def play(sim):
    ovr = _load_ovr(sim)
    ver = _load_ver(sim)
    try:
        run(sim, ovr, False, None, None, None, 7990, ver)
    except Exception:
        pass


def _mean(xs):
    return float(sum(xs)) / max(1, len(xs))


def solve(sim):
    """Learning phase. A/B test: version 0 (PB behaviour, baseline, ovr never modified) against
    version 2 (all-candidate wall scan, earlier stuck-jump / retreat; its own ovr is learned by
    checkpoint backtracking). Choose version 2 only if it is clean (no deaths) in its last 3
    episodes and faster than the baseline. Result stored in sim.memory['ver'] / ['ovr']."""
    t0 = time.time()
    deadline = t0 + 1250
    rng = random.Random(12345)

    def log(msg):
        print(msg)
        sys.stdout.flush()

    mem = sim.memory
    base = _load_ovr(sim)
    base_ver = _load_ver(sim)
    ovr2 = dict(base)
    clean0 = []
    hist2 = []          # list of (clean, frames)
    episodes = 0
    use2 = False
    try:
        for step in range(9):
            if time.time() > deadline - 160 or sim.budget_left < 2600:
                break
            ver = 0 if step in (1, 5) else 2
            if episodes > 0:
                sim.reset()
            episodes += 1
            cap = int(min(3500, sim.budget_left - 5))
            if ver == 0:
                ovr = dict(base)
            else:
                ovr = dict(ovr2)
            info = run(sim, ovr, True, deadline, log, rng, cap, ver)
            clean = (info['result'] == 'finished' and info['deaths'] == 0)
            log("EPISODE %d ver=%d: result=%s clean=%s frames=%d steps=%d maxwx=%d deaths=%d deathwx=%s nov=%d budget_left=%d t=%.0f" % (
                episodes, ver, info['result'], clean, info['frames'], info['steps'], info['max_wx'],
                info['deaths'], info['death_wxs'][:20], len(ovr), sim.budget_left, time.time() - t0))
            if ver == 0:
                if clean:
                    clean0.append(info['frames'])
            else:
                hist2.append((clean, info['frames']))
                if info['result'] == 'finished':
                    ovr2 = dict(ovr)       # learned adjustments (if any) after a finished episode
                # on failure keep the previous ovr2 (trial entries of a failed search are discarded)
            try:
                mem.setdefault('ab', [])
                mem['ab'].append({'ver': ver, 'res': info['result'], 'clean': bool(clean),
                                  'frames': info['frames'], 'deaths': info['deaths']})
            except Exception:
                pass
            if info['result'] == 'deadline':
                break
            last3 = hist2[-3:]
            if len(last3) >= 3 and all(c for c, _ in last3) and (clean0 or step >= 4):
                break
        # ---- decision ----
        last3 = hist2[-3:]
        base0 = _mean(clean0) if clean0 else 1349.0
        if len(last3) >= 3 and all(c for c, _ in last3):
            m2 = _mean([f for _, f in last3])
            log("DECISION: v2 mean=%.1f vs v0 mean=%.1f (clean0=%s)" % (m2, base0, clean0))
            if m2 <= base0 - 4:
                use2 = True
        else:
            log("DECISION: v2 not validated hist2=%s clean0=%s" % (hist2, clean0))
    except Exception as e:
        log("solve exception: %r" % (e,))
    try:
        if use2:
            mem['ver'] = 2
            mem['ovr'] = {str(k): int(v) for k, v in ovr2.items()}
        else:
            mem['ver'] = base_ver if base_ver in (0,) else 0
            mem['ovr'] = {str(k): int(v) for k, v in base.items()}
        log("FINAL ver=%s ovr=%s" % (mem['ver'], mem['ovr']))
        try:
            mem.flush()
        except Exception:
            pass
    except Exception:
        pass
    return []

