"""Analiza offline a debug_log.csv (format vechi sau nou) + evaluarea etichetelor de test (tastele 0-6).

Nu modifica niciun fisier al proiectului. Rulare din folderul proiectului:
    uv run --with pandas python analyze_log.py [debug_log.csv] [calib.json]
(`--with pandas` foloseste pandas doar pentru aceasta rulare, fara sa-l adauge in pyproject.toml)
"""
import json
import sys

import warnings

import numpy as np
import pandas as pd

try:
    import gaze  # doar codul vechi (module separate) are gaze.py -> replay-ul din sectiunea 5
except ImportError:
    gaze = None

warnings.filterwarnings("ignore", category=RuntimeWarning)
LOG = sys.argv[1] if len(sys.argv) > 1 else "debug_log.csv"
CALIB = sys.argv[2] if len(sys.argv) > 2 else "calib.json"
d_all = pd.read_csv(LOG)
d = d_all[d_all.face == 1].reset_index(drop=True) if "face" in d_all else d_all   # log nou: si cadre fara fata
cal = json.load(open(CALIB))
dt = d.t.diff().clip(upper=0.2).fillna(0)          # acelasi plafon ca BlinkTracker
dur_s = d.t.iloc[-1] - d.t.iloc[0]


def episodes(mask):
    """[(start, durata)] pentru secventele True consecutive."""
    out, st = [], None
    for m, t in zip(mask, d.t):
        if m and st is None:
            st = t
        elif not m and st is not None:
            out.append((st, t - st))
            st = None
    return out


def debounce(raw, n):
    c, run = np.zeros(len(raw), bool), 0
    for i, v in enumerate(raw):
        run = run + 1 if v else 0
        c[i] = run >= n
    return c


print(f"== 1. TIMP: {len(d)} cadre cu fata, {dur_s:.0f} s, FPS median {d.fps[d.fps > 0].median():.1f}, "
      f"pauze >0.5 s (fata pierduta): {(d.t.diff() > 0.5).sum()}")

print("\n== 2. OCHI INCHISI")
o = (d.open_r + d.open_l) / 2
if cal.get("eye_open_closed") and cal.get("eye_open_ref"):      # calib.json vechi (v9)
    print(f"prag calibrat eye_open_closed={cal['eye_open_closed']:.3f}  eye_open_ref={cal['eye_open_ref']:.3f}"
          f"  -> prag = {cal['eye_open_closed'] / cal['eye_open_ref'] * 100:.0f}% din deschiderea normala")
elif "eye" in cal:                                               # calib.json nou (v10)
    e = cal["eye"]
    print(f"calibrare ochi v10: eyeBlink deschis {e['bs_open']} inchis {e['bs_closed']}  geometrie: {e.get('use_geom')}"
          f"  deschis {e.get('open')} inchis {e.get('closed')}")
bs_closed = (d.blink_l > 0.5) & (d.blink_r > 0.5)
print(f"deschidere mediana: ochi deschisi (eyeBlink<0.3) {o[(d.blink_l < 0.3) & (d.blink_r < 0.3)].median():.3f} | "
      f"clipit clar (eyeBlink>0.5 ambii) {o[bs_closed].median():.3f}")
print(f"cadre 'closed' din log: {int(d.closed.sum())}, dintre care cu eyeBlink>0.5 pe ambii ochi: "
      f"{int((bs_closed & (d.closed == 1)).sum())}")
for name, raw, n in [("CURENT (coloana closed)", d.closed.values == 1, 1),
                     ("iris ambii < 0.45", ((d.open_r < 0.45) & (d.open_l < 0.45)).values, 2),
                     ("eyeBlink ambii > 0.45 (1 cadru)", ((d.blink_l > 0.45) & (d.blink_r > 0.45)).values, 1)]:
    c = debounce(raw, n)
    e = episodes(c)
    du = np.array([x[1] for x in e]) if e else np.zeros(1)
    print(f"  {name:34s} episoade {len(e):4d} ({len(e) / (dur_s / 60):4.1f}/min)  >=1 s: {int((du >= 1).sum())}"
          f"  PERCLOS {(dt * c).sum() / dt.sum() * 100:4.1f}%")
print("inchideri >= 1 s din log (microsomn raportat) si ce spun blendshape-urile:")
for st, du in episodes(d.closed.values == 1):
    if du >= 1.0:
        w = d[(d.t >= st) & (d.t < st + du)]
        print(f"  t={st:7.1f}s  {du:.2f}s  eyeBlink L/R={w.blink_l.mean():.2f}/{w.blink_r.mean():.2f}"
              f"  deschidere={((w.open_r + w.open_l) / 2).mean():.2f}")
ld_thr = cal.get("look_down_thr", cal.get("eye", {}).get("look_down_thr", "NECALIBRAT -> 0.45"))
print(f"looking_down activ {d.looking_down.mean() * 100:.0f}% din cadre (prag eyeLookDown "
      f"{ld_thr}), eyeLookDown median pe drum "
      f"{d.look_down[d.zone == 'drum'].median():.2f}")

print("\n== 3. DEPENDENTA DE POZA CAPULUI")
print(f"corelatie roll~yaw {np.corrcoef(d.roll, d.yaw)[0, 1]:+.2f}   deschidere~|yaw| {np.corrcoef(o, d.yaw.abs())[0, 1]:+.2f}")
for lo, hi in [(-90, -38), (-38, -25), (-25, -10), (-10, 10), (10, 25), (25, 40), (40, 90)]:
    w = d[(d.yaw >= lo) & (d.yaw < hi)]
    if len(w) < 20:
        continue
    bad = ((w.gaze_h < 0) | (w.gaze_h > 1) | (w.gaze_v < 0) | (w.gaze_v > 1)).mean()
    print(f"  yaw [{lo:+3d},{hi:+3d}) n={len(w):5d}  roll {w.roll.median():+5.1f}  deschidere {((w.open_r + w.open_l) / 2).median():.2f}"
          f"  iris h/v in afara [0,1]: {bad * 100:4.1f}%")

print("\n== 4. ZONE BRUTE (inainte de confirmare)")
runs, cur, st = [], None, None
for z, t in zip(d.zone, d.t):
    if z != cur:
        if cur is not None:
            runs.append((cur, st, t - st))
        cur, st = z, t
if cur is not None:
    runs.append((cur, st, d.t.iloc[-1] - st))
r = pd.DataFrame(runs, columns=["zone", "start", "dur"])
s = r.groupby("zone").dur.agg(["count", "median", "max"]).join(d.groupby("zone").zone_dist.median())
s[">=confirm"] = r[r.dur >= 0.35].groupby("zone").size()
print(s.fillna(0).round(2).to_string())

if gaze is None:
    print("\n== 5. (replay-ul vechi sarit: gaze.py nu exista; codul nou e intr-un singur fisier)")
else:
    print("\n== 5. REPLAY OGLINZI (gaze.ZoneModel + gaze.GazeTracker din proiect)")
orig_scale = dict(gaze.ZONE_SCALE) if gaze else {}


def replay(scale, validate):
    gaze.ZONE_SCALE.clear()
    gaze.ZONE_SCALE.update(scale)
    zones = {k: list(v) for k, v in cal["zones"].items()}
    if validate:      # iris imposibil la calibrare (in afara 0.05..0.95) -> zona se clasifica doar din cap
        for v in zones.values():
            if v[2] is not None and not (0.05 <= v[2] <= 0.95 and 0.05 <= v[3] <= 0.95):
                v[2] = v[3] = None
    zm = gaze.ZoneModel(zones)
    gt = gaze.GazeTracker(gaze.GazeCalib(cal["gaze_h0"], cal["gaze_v0"]))
    n_orig, last_m, dwell, n_dwell, last_d, lst = 0, gt.last_mirror_t, None, 0, -1e9, []
    for row in d.itertuples():
        g = None if (np.isnan(row.gaze_h) or row.closed) else (row.gaze_h, row.gaze_v)
        if validate and g is not None and (not (0 <= g[0] <= 1 and 0 <= g[1] <= 1) or abs(row.yaw) > 25):
            g = None  # iris invalid sau cap prea intors -> doar capul
        zone, _ = zm.classify(row.yaw, row.pitch, g, prefer=gt.zone)
        prev = gt.zone
        gt.update(row.t, g, eyes_open=not row.closed, zone=zone)
        if gt.last_mirror_t != last_m:
            n_orig, last_m = n_orig + 1, gt.last_mirror_t
        if gt.zone != prev:  # contor propus: orice privire confirmata in oglinda, 0.25-2 s, indiferent unde merge dupa
            if prev in gaze.MIRROR_ZONES and dwell is not None:
                du = gt.cand_since - dwell
                if gaze.MIRROR_MIN_S <= du <= gaze.MIRROR_ZONE_MAX_S and row.t - last_d > gaze.MIRROR_DEBOUNCE_S:
                    n_dwell, last_d = n_dwell + 1, row.t
                    lst.append(f"{dwell:.1f}s {prev} {du:.2f}s")
            dwell = gt.cand_since if gt.zone in gaze.MIRROR_ZONES else None
    gaze.ZONE_SCALE.clear()
    gaze.ZONE_SCALE.update(orig_scale)
    return n_orig, n_dwell, lst


for label, scale, val in [] if gaze is None else [
        ("V0 cod curent", orig_scale, False),
        ("V1 iris h=0.25 v=0.35", {**orig_scale, "h": 0.25, "v": 0.35}, False),
        ("V2 V1 + validare iris", {**orig_scale, "h": 0.25, "v": 0.35}, True),
        ("V3 V2 + cap yaw=10 pitch=8", {"yaw": 10.0, "pitch": 8.0, "h": 0.25, "v": 0.35}, True)]:
    n_o, n_d, lst = replay(scale, val)
    print(f"  {label:28s} contor actual: {n_o:2d}   contor per-privire: {n_d:2d}")
    if lst:
        print("     " + "; ".join(lst))


# ---------------------------------------------------------------------------------------------------
LABELS = {1: "oglinda_stanga", 2: "oglinda_dreapta", 3: "oglinda_mijloc", 4: "ochi inchisi", 5: "cascat", 6: "vorbit"}
if "label" in d_all and (d_all.label > 0).any():
    print("\n== 6. ETICHETE DE TEST (tastele 0-6) vs DETECTIE")
    seg, cur, st = [], None, None
    for lab, t in zip(d_all.label, d_all.t):
        if lab != cur:
            if cur:
                seg.append((cur, st, t))
            cur, st = lab, t
    if cur:
        seg.append((cur, st, d_all.t.iloc[-1]))
    rows = []
    for lab, a, b in seg:
        w = d_all[(d_all.t >= a) & (d_all.t <= b + 0.5)]       # +0.5 s: confirmarea zonei vine cu intarziere
        if lab in (1, 2, 3):
            hit = (w.zone_conf == LABELS[lab]).any()
            other = sorted(set(w.zone_conf.dropna()) - {LABELS[lab], "drum"})
            rows.append((LABELS[lab], f"{a:.1f}", f"{b - a:.1f}", "DA" if hit else "nu", ", ".join(other)))
        elif lab == 4:
            hit = (w.closed == 1).any()
            rows.append((LABELS[lab], f"{a:.1f}", f"{b - a:.1f}", "DA" if hit else "nu",
                         f"inchis {(w.closed == 1).mean() * 100:.0f}% din segment"))
        elif lab in (5, 6):
            hit = (w.yawning == 1).any()
            ok = hit if lab == 5 else not hit
            rows.append((LABELS[lab], f"{a:.1f}", f"{b - a:.1f}", "DA" if ok else "nu",
                         "cascat detectat" if hit else "fara cascat"))
    print(pd.DataFrame(rows, columns=["eticheta", "start_s", "durata_s", "corect", "detalii"]).to_string(index=False))
    for name in LABELS.values():
        r = [x for x in rows if x[0] == name]
        if r:
            print(f"  {name:16s} corect {sum(x[3] == 'DA' for x in r)}/{len(r)}")
    if "ms_face" in d_all:
        tot = d_all[["ms_camera", "ms_face", "ms_hands", "ms_draw"]].sum(axis=1)
        print(f"timp/cadru (fara camera): median {(tot - d_all.ms_camera).median():.1f} ms, "
              f"p95 {(tot - d_all.ms_camera).quantile(.95):.1f} ms; FPS median {d_all.fps[d_all.fps > 0].median():.1f}")
