"""Iris / privire - toata logica bazata pe punctele de iris din MediaPipe Face Landmarker.

Modelul face_landmarker.task scoate 478 de puncte: ultimele 10 sunt irisul (468-477).
  - 468 = centrul irisului drept, inelul 469-472
  - 473 = centrul irisului stang, inelul 474-477

Doua lucruri:
  1) PRIVIRE: pozitia irisului IN ochi -> incotro se uita soferul, independent de cap.
  2) OCHI INCHISI (eye_openness): deschiderea ochiului / diametrul irisului, PER OCHI.

CALIBRARE "scanare completa" (GazeCalib): invatam din fata soferului cat de departe isi
misca ochii in fiecare directie (dreapta/stanga/sus/jos fata de centru). Pragul devine
PERSONAL, iar semnul stanga/dreapta se afla singur (fara GAZE_H_SIGN manual).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

# --- puncte (indici MediaPipe Face Mesh 478) ---
EYES = [
    {"corners": (33, 133), "lids": (159, 145), "iris": 468, "ring": [469, 470, 471, 472]},   # dreapta
    {"corners": (362, 263), "lids": (386, 374), "iris": 473, "ring": [474, 475, 476, 477]},  # stanga
]

SACCADE_THRESH = 0.08
GAZE_FRAC = 0.40           # cat din drumul pana la "privire completa laterala" = deja "in alta parte"

# --- ochi inchisi din iris ---
EYE_OPEN_CLOSED = 0.45

# --- ferestre de timp (secunde) ---
VATS_WINDOW = 30.0
VATS_SUM_ALERT = 10.0
LONG_GLANCE_S = 3.0
MIRROR_MIN_S = 0.25
MIRROR_MAX_S = 1.5
MIRROR_DEBOUNCE_S = 0.8
MIRROR_PEAK_FRAC = 0.55    # abaterea laterala trebuie sa fie macar atat din raza personala
ONROAD_WINDOW = 30.0
SACCADE_WINDOW = 3.0
FIX_WINDOW = 1.5
FIXATION_MOVE = 0.03
HYPNOSIS_S = 8.0
HEAT_WINDOW = 15.0
GRID = 5
GAZE_RANGE = 0.35


@dataclass
class GazeCalib:
    """Neutrul si raza de miscare a privirii, invatate la scanarea completa."""
    h0: float = 0.5           # iris centrat orizontal (priveste drept)
    v0: float = 0.5           # iris centrat vertical
    h_half: float = 0.15      # cat de mult se misca orizontal pana la "complet lateral"
    v_half: float = 0.15      # idem vertical
    h_sign: int = 1           # +1 daca h creste cand priveste la dreapta (aflat automat)
    v_sign: int = 1           # +1 daca v creste cand priveste in jos

    def offsets(self, h, v):
        """Abaterea NORMALIZATA (0 = centrat, 1 = privire completa in acea directie)."""
        nx = (h - self.h0) / (self.h_half + 1e-6)
        ny = (v - self.v0) / (self.v_half + 1e-6)
        return nx, ny


def gaze_ratios(xy):
    """(h, v) = pozitia medie a irisului in ochi, 0..1 (0.5 = centrat). None daca lipsesc punctele."""
    if len(xy) < 478:
        return None
    hs, vs = [], []
    for e in EYES:
        (ca, cb), (lt, lb), iris = e["corners"], e["lids"], e["iris"]
        left, right = min(xy[ca][0], xy[cb][0]), max(xy[ca][0], xy[cb][0])
        top, bottom = min(xy[lt][1], xy[lb][1]), max(xy[lt][1], xy[lb][1])
        ix, iy = xy[iris][0], xy[iris][1]
        if right - left > 1e-3:
            hs.append((ix - left) / (right - left))
        if bottom - top > 1e-3:
            vs.append((iy - top) / (bottom - top))
    if not hs or not vs:
        return None
    return float(np.mean(hs)), float(np.mean(vs))


def eye_openness(xy):
    """Per ochi: deschiderea verticala / diametrul irisului. (dreapta, stanga) sau None."""
    if len(xy) < 478:
        return None
    out = []
    for e in EYES:
        lt, lb = e["lids"]
        center = np.array(xy[e["iris"]])
        lid = float(np.hypot(xy[lt][0] - xy[lb][0], xy[lt][1] - xy[lb][1]))
        diam = 2.0 * float(np.mean([np.hypot(xy[r][0] - center[0], xy[r][1] - center[1])
                                    for r in e["ring"]]))
        out.append(lid / diam if diam > 1e-3 else 0.0)
    return out[0], out[1]


def direction(h, v, cal: GazeCalib):
    """Eticheta privirii din raportul de iris + calibrarea personala. '' = pe drum."""
    nx, ny = cal.offsets(h, v)
    if abs(nx) < GAZE_FRAC and abs(ny) < GAZE_FRAC:
        return ""
    if abs(nx) >= abs(ny):
        return "dreapta" if nx * cal.h_sign > 0 else "stanga"
    return "jos" if ny * cal.v_sign > 0 else "sus"


def is_away(h, v, cal: GazeCalib):
    nx, ny = cal.offsets(h, v)
    return abs(nx) >= GAZE_FRAC or abs(ny) >= GAZE_FRAC


def _cell(dh, dv):
    def c(x):
        return int(np.clip((x + GAZE_RANGE) / (2 * GAZE_RANGE) * GRID, 0, GRID - 1))
    return c(dh), c(dv)


class GazeTracker:
    """Metrici de privire pentru SOFER (o singura persoana), pe ferestre de timp."""

    def __init__(self, cal: GazeCalib | None = None):
        self.cal = cal or GazeCalib()
        self.smooth = deque(maxlen=3)
        self.samples = deque()
        self.ep = None
        self.last_mirror_t = -1e9
        self.episodes = deque()
        self.mirrors = deque()
        self.fix_since = None
        self.prev = None
        self.saccades = deque()
        self.blinks = deque()

    def note_blink(self, t):
        self.blinks.append(t)
        if self.ep is not None:
            self.ep["blink"] = True

    def update(self, t, gaze, head_away, eyes_open):
        cal = self.cal
        has_gaze = gaze is not None and eyes_open
        if has_gaze:
            self.smooth.append(gaze)
            sh = float(np.mean([g[0] for g in self.smooth]))
            sv = float(np.mean([g[1] for g in self.smooth]))
            dh, dv = sh - cal.h0, sv - cal.v0
            nx, ny = cal.offsets(sh, sv)
            gaze_away = abs(nx) >= GAZE_FRAC or abs(ny) >= GAZE_FRAC
        else:
            self.smooth.clear()
            dh = dv = 0.0
            gaze_away = False
        away = bool(head_away or gaze_away)
        on_road = not away

        # --- episod in afara drumului ---
        if away:
            if self.ep is None:
                self.ep = {"start": t, "peak_h": 0.0, "peak_v": 0.0, "n": 0,
                           "blink": not eyes_open, "head": head_away}
            self.ep["peak_h"] = max(self.ep["peak_h"], abs(dh))
            self.ep["peak_v"] = max(self.ep["peak_v"], abs(dv))
            self.ep["n"] += 1
            self.ep["head"] = self.ep["head"] or head_away
            if not eyes_open:
                self.ep["blink"] = True
        elif self.ep is not None:
            ep, dur = self.ep, t - self.ep["start"]
            self.episodes.append((t, dur))
            is_mirror = (MIRROR_MIN_S <= dur < MIRROR_MAX_S and ep["n"] >= 2
                         and not ep["blink"] and not ep["head"]
                         and ep["peak_h"] > cal.h_half * MIRROR_PEAK_FRAC
                         and ep["peak_v"] < cal.v_half * GAZE_FRAC)
            if is_mirror and t - self.last_mirror_t > MIRROR_DEBOUNCE_S:
                self.mirrors.append(t)
                self.last_mirror_t = t
            self.ep = None

        away_sec = (t - self.ep["start"]) if self.ep is not None else 0.0

        self.samples.append((t, dh, dv, on_road, has_gaze))
        while self.samples and t - self.samples[0][0] > max(ONROAD_WINDOW, HEAT_WINDOW):
            self.samples.popleft()
        while self.episodes and t - self.episodes[0][0] > VATS_WINDOW:
            self.episodes.popleft()
        while self.mirrors and t - self.mirrors[0] > 60.0:
            self.mirrors.popleft()
        while self.blinks and t - self.blinks[0] > 60.0:
            self.blinks.popleft()

        if has_gaze and self.prev is not None:
            if math.hypot(dh - self.prev[0], dv - self.prev[1]) > SACCADE_THRESH:
                self.saccades.append(t)
        if has_gaze:
            self.prev = (dh, dv)
        while self.saccades and t - self.saccades[0] > SACCADE_WINDOW:
            self.saccades.popleft()
        saccade_rate = len(self.saccades) / SACCADE_WINDOW

        recent = [(hh, vv) for tt, hh, vv, _, hg in self.samples if hg and t - tt <= FIX_WINDOW]
        if len(recent) >= 5:
            spread = math.hypot(float(np.std([r[0] for r in recent])),
                                float(np.std([r[1] for r in recent])))
            moving = spread > FIXATION_MOVE
        else:
            moving = True
        if has_gaze and on_road and not moving:
            self.fix_since = self.fix_since or t
        else:
            self.fix_since = None
        fixation_s = (t - self.fix_since) if self.fix_since is not None else 0.0
        blink_rate_min = len(self.blinks)
        hypnosis = fixation_s >= HYPNOSIS_S and blink_rate_min < 8 and saccade_rate < 0.5

        onroad_t = total_t = 0.0
        prev_t = None
        for tt, _, _, orr, _ in self.samples:
            if prev_t is not None and t - tt <= ONROAD_WINDOW:
                dt = tt - prev_t
                total_t += dt
                if orr:
                    onroad_t += dt
            prev_t = tt
        on_road_ratio = (onroad_t / total_t) if total_t > 0 else 1.0

        vats_s = sum(d for _, d in self.episodes) + away_sec
        vats_alert = vats_s >= VATS_SUM_ALERT

        grid = np.zeros((GRID, GRID))
        for tt, hh, vv, _, hg in self.samples:
            if hg and t - tt <= HEAT_WINDOW:
                cx, cy = _cell(hh, vv)
                grid[cy, cx] += 1
        total = grid.sum()
        if total > 0:
            p = grid[grid > 0] / total
            entropy = abs(float(-(p * np.log(p)).sum() / math.log(GRID * GRID)))
        else:
            entropy = 0.0

        return {
            "direction": direction(dh + cal.h0, dv + cal.v0, cal) if has_gaze
                         else ("cap intors" if head_away else ""),
            "dh": round(dh, 3), "dv": round(dv, 3),
            "on_road": on_road, "away_sec": round(away_sec, 2),
            "on_road_ratio": round(on_road_ratio, 3),
            "vats_s": round(vats_s, 1), "vats_alert": vats_alert,
            "long_glance": away_sec >= LONG_GLANCE_S,
            "mirror_checks": len(self.mirrors),
            "saccade_rate": round(saccade_rate, 2),
            "fixation_s": round(fixation_s, 1), "hypnosis": hypnosis,
            "entropy": round(entropy, 2),
            "heat": grid, "heat_max": float(grid.max()),
        }
