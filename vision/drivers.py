"""Driver profiles + the app window (Tkinter): camera + warnings on the left, drivers panel on the right.

profiles.json (saved locally, holds the face prints):
    {"last": "Andrei", "drivers": {"Andrei": {"glasses": true, "faces": [[128 numbers], ...]}}}
Each driver has his own calibration: calib_<name>.json (no driver -> calib.json).
The driver is recognized automatically (faceid.py); the panel adds drivers or picks one by hand.

The window runs inside the camera loop: panel.show(frame) every frame returns the key pressed. Clicks are
queued as events for face_basic (panel.pop_events()):
  ("select", name)     picked by hand: load his calibration (calibrate if he has none)
  ("add", name)        new driver: calibrate, then learn his face
  ("calibrate", name)  Recalibrate button
  ("enroll", name)     Re-learn face button
"""

import json
import os
import re
import tkinter as tk
from tkinter import messagebox

import cv2

PROFILES_FILE = "profiles.json"

# ---------- look: same colors as the dashboard (static/style.css) ----------
BG = "#f3f4f6"          # --bg          window background
CARD = "#ffffff"        # --surface     panels
BORDER = "#d1d5db"      # --border      panel / button borders
CARD_HI = "#e5e7eb"     # --row-border  separators, list selection, bars
TEXT = "#000000"        # --text
MUTED = "#4d4d4d"       # text at 70% (the dashboard uses opacity 0.7 for secondary text)
ACCENT = "#000000"      # --primary     primary buttons
ACCENT_HOVER = "#0a365c"  # --primary-hover
OK = "#15803d"          # --ok
WARN = "#b45309"        # --warn
BAD = "#b91c1c"         # --danger
INFO = "#0a365c"        # info alerts (dark blue, like --primary-hover)
DISABLED = "#9ca3af"    # grayed-out buttons
FONT = "Arial"          # body font of the dashboard
PANEL_W = 300           # width of the panel, pixels

LEVEL_LOOK = {           # alert level -> (label, color) for the warnings row under the image
    "critical": ("CRITICAL", BAD),
    "warning": ("WARNING", WARN),
    "info": ("INFO", INFO),
}
WARN_TEXT = {"glasses_missing": "Put your glasses on!"}   # nicer text for some alert codes
MAX_WARNINGS = 3          # most important warnings shown under the image (all of them are in state.json)

STATUS_LOOK = {          # how -> (pill text, color)
    "recognized": ("RECOGNIZED", OK),
    "picked by hand": ("PICKED BY HAND", ACCENT),
    "new driver": ("NEW DRIVER", ACCENT),
    "last used": ("IDENTIFYING...", MUTED),
    "unknown face": ("UNKNOWN FACE - NOT MONITORED", WARN),
    "deleted": ("NO DRIVER", MUTED),
    "no profile": ("NO PROFILE - ADD A DRIVER", WARN),
}


def calib_file(name):
    """Calibration file of a driver ("calib.json" when no driver is selected)."""
    if not name:
        return "calib.json"
    return f"calib_{re.sub(r'[^A-Za-z0-9_-]', '_', name)}.json"


def load_profiles():
    try:
        with open(PROFILES_FILE) as f:
            p = json.load(f)
        if isinstance(p.get("drivers"), dict):
            return {"last": p.get("last"), "drivers": p["drivers"]}
    except (OSError, ValueError, AttributeError):
        pass
    return {"last": None, "drivers": {}}


def save_profiles(p):
    tmp = PROFILES_FILE + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump(p, f, indent=2)
        os.replace(tmp, PROFILES_FILE)
    except OSError as e:
        print(f"Cannot save {PROFILES_FILE}: {e}")


def _button(parent, text, command, kind="secondary"):
    """Dashboard-style button: primary = black, secondary = white with a border, danger = red text.
    _enable(b, False) grays it out and ignores clicks."""
    bg, fg, hover = {"primary": (ACCENT, "white", ACCENT_HOVER),
                     "secondary": (CARD, TEXT, BG),
                     "danger": (CARD, BAD, BG)}[kind]
    b = tk.Label(parent, text=text, bg=bg, fg=fg, font=(FONT, 9, "bold"), padx=12, pady=6, cursor="hand2",
                 highlightthickness=0 if kind == "primary" else 1, highlightbackground=BORDER)
    b.look, b.enabled = (bg, fg), True
    b.bind("<Enter>", lambda e: b.enabled and b.config(bg=hover))
    b.bind("<Leave>", lambda e: b.enabled and b.config(bg=bg))
    b.bind("<Button-1>", lambda e: b.enabled and command())
    return b


def _enable(b, on):
    if on != b.enabled:
        b.enabled = on
        b.config(bg=b.look[0] if on else CARD, fg=b.look[1] if on else DISABLED,
                 cursor="hand2" if on else "arrow")


def _card(parent, title=None):
    outer = tk.Frame(parent, bg=CARD, padx=12, pady=8, highlightthickness=1, highlightbackground=BORDER)
    outer.pack(fill="x", padx=12, pady=(0, 8))
    if title:                                    # like .field-label: bold, uppercase, black
        tk.Label(outer, text=title.upper(), bg=CARD, fg=TEXT, font=(FONT, 8, "bold")).pack(anchor="w")
    return outer


class DriverPanel:
    def __init__(self, title="Driver Recognition"):
        self.profiles = load_profiles()
        last, names = self.profiles["last"], sorted(self.profiles["drivers"])
        # start with the last driver (or the first saved one); recognition then switches to whoever it sees
        self.active = last if last in names else (names[0] if names else None)
        self.how = "last used" if self.active else "no profile"   # also "recognized", "picked by hand", ...
        self.events = []
        self.alive = True
        self.glasses_now = None
        self._names = []                  # listbox row -> driver name
        self._match = (None, None)
        self._alerts = None

        self.root = tk.Tk()
        self.root.title(title)
        self.root.geometry("+0+0")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)
        self.quit = False
        self.root.protocol("WM_DELETE_WINDOW", lambda: setattr(self, "quit", True))   # X = same as q
        self.key = 255                          # last key pressed (255 = none, like cv2.waitKey)
        self.root.bind("<Key>", self._on_key)

        # camera image + warnings (left) | panel (right)
        left = tk.Frame(self.root, bg=BG)
        left.pack(side="left", anchor="n")
        self.photo = None
        self.video = tk.Label(left, bd=0, bg="black")
        self.video.pack(anchor="n")
        self.video.bind("<Button-1>", lambda e: self.root.focus_set())   # click the image = keys work again
        tk.Label(left, text="WARNINGS", bg=BG, fg=TEXT, font=(FONT, 9, "bold")).pack(
            anchor="w", padx=12, pady=(10, 4))
        self.alerts_row = tk.Frame(left, bg=BG, width=620, height=30)   # set to the image width later
        self.alerts_row.pack(anchor="w", padx=12, pady=(0, 10))
        self.alerts_row.pack_propagate(False)                           # never wider than the image
        side = tk.Frame(self.root, bg=BG)
        side.pack(side="left", fill="y")
        tk.Frame(side, bg=BG, width=PANEL_W, height=0).pack()          # fixes the panel width

        tk.Label(side, text="DRIVER MONITOR", bg=BG, fg=TEXT, font=(FONT, 11, "bold")).pack(
            anchor="w", padx=14, pady=(12, 6))

        # --- active driver card: avatar + name + status pill + match bar ---
        card = _card(side)
        top = tk.Frame(card, bg=CARD)
        top.pack(fill="x")
        self.avatar = tk.Canvas(top, width=46, height=46, bg=CARD, highlightthickness=0)
        self.avatar.pack(side="left")
        names = tk.Frame(top, bg=CARD)
        names.pack(side="left", padx=10, fill="x", expand=True)
        self.name_lbl = tk.Label(names, bg=CARD, fg=TEXT, font=(FONT, 15, "bold"), anchor="w")
        self.name_lbl.pack(fill="x")
        self.pill = tk.Label(names, font=(FONT, 7, "bold"), padx=6, pady=1)
        self.pill.pack(anchor="w", pady=(2, 0))
        tk.Label(card, text="Face match", bg=CARD, fg=MUTED, font=(FONT, 8)).pack(anchor="w", pady=(10, 2))
        self.bar = tk.Canvas(card, height=8, bg=CARD_HI, highlightthickness=0)
        self.bar.pack(fill="x")
        self.status_lbl = tk.Label(card, bg=CARD, fg=MUTED, font=(FONT, 8), anchor="w")
        self.status_lbl.pack(fill="x", pady=(4, 0))
        tk.Frame(card, bg=CARD_HI, height=1).pack(fill="x", pady=8)                 # separator

        # --- glasses (in the driver card) ---
        row = tk.Frame(card, bg=CARD)
        row.pack(fill="x")
        tk.Label(row, text="Glasses", bg=CARD, fg=MUTED, font=(FONT, 9)).pack(side="left", padx=(0, 8))
        self.dot = tk.Canvas(row, width=10, height=10, bg=CARD, highlightthickness=0)
        self.dot.pack(side="left")
        self.now_lbl = tk.Label(row, bg=CARD, fg=TEXT, font=(FONT, 10))
        self.now_lbl.pack(side="left", padx=6)
        self.glasses_var = tk.BooleanVar()
        self.glasses_chk = tk.Checkbutton(card, text="Wears glasses (warn if missing)",
                                          variable=self.glasses_var, command=self._glasses_changed,
                                          bg=CARD, fg=TEXT, activebackground=CARD, activeforeground=TEXT,
                                          selectcolor=CARD, font=(FONT, 9), bd=0, highlightthickness=0)
        self.glasses_chk.pack(anchor="w", pady=(4, 0))

        # --- drivers card: list + actions ---
        dcard = _card(side, "Drivers")
        self.listbox = tk.Listbox(dcard, height=3, bg=CARD, fg=TEXT, font=(FONT, 10), bd=0,
                                  highlightthickness=0, selectbackground=CARD_HI, selectforeground=TEXT,
                                  activestyle="none", exportselection=False)
        self.listbox.pack(fill="x", pady=(6, 6))
        self.listbox.bind("<Double-Button-1>", lambda e: self._use())
        row = tk.Frame(dcard, bg=CARD)
        row.pack(fill="x")
        _button(row, "Use selected", self._use).pack(side="left")
        _button(row, "Delete", self._delete, "danger").pack(side="right")
        row = tk.Frame(dcard, bg=CARD)
        row.pack(fill="x", pady=(6, 0))
        self.recal_btn = _button(row, "Recalibrate", self._recalibrate)
        self.recal_btn.pack(side="left", fill="x", expand=True)
        tk.Frame(row, bg=CARD, width=6).pack(side="left")
        self.relearn_btn = _button(row, "Re-learn face",
                                   lambda: self.active and self.events.append(("enroll", self.active)))
        self.relearn_btn.pack(side="left", fill="x", expand=True)

        # --- new driver card ---
        ncard = _card(side, "New driver")
        self.name_entry = tk.Entry(ncard, bg=CARD, fg=TEXT, insertbackground=TEXT, relief="flat",
                                   font=(FONT, 10), highlightthickness=1, highlightbackground=BORDER,
                                   highlightcolor=ACCENT)
        self.name_entry.pack(fill="x", pady=(6, 6), ipady=4)
        self.name_entry.bind("<Return>", lambda e: self._add())
        _button(ncard, "+  Add and calibrate", self._add, "primary").pack(fill="x")

        tk.Label(side, text="q - quit      f - face      c - cancel calibration", bg=BG, fg=MUTED,
                 font=(FONT, 8)).pack(side="bottom", pady=8)

        self._refresh()
        self.set_alerts([])
        self.tick(False)

    # ---------- called from the camera loop ----------
    def tick(self, glasses_now=None):
        """Process clicks / redraws (once per frame)."""
        if not self.alive:
            return
        if glasses_now is not None and glasses_now != self.glasses_now:
            self.glasses_now = glasses_now
            self.now_lbl.config(text="Detected now" if glasses_now else "Not detected")
            self.dot.delete("all")
            self.dot.create_oval(1, 1, 9, 9, fill=OK if glasses_now else DISABLED, outline="")
        try:
            self.root.update()
        except tk.TclError:                       # window destroyed
            self.alive = False

    def show(self, frame):
        """Show the camera frame, process clicks, return the key pressed (255 = none)."""
        ok, buf = cv2.imencode(".ppm", frame)              # PPM: Tk reads it natively (fast)
        if ok and self.alive:
            if self.photo is None:
                self.photo = tk.PhotoImage(data=buf.tobytes())
                self.video.config(image=self.photo)
                self.alerts_row.config(width=frame.shape[1] - 24)
            else:
                self.photo.configure(data=buf.tobytes())
        self.tick()
        key, self.key = self.key, 255
        return ord("q") if self.quit or not self.alive else key

    def set_status(self, text):
        if text != self.status_lbl.cget("text"):
            self.status_lbl.config(text=text)

    def set_match(self, score, threshold):
        """Face-match bar: score 0..1, a black tick at the threshold."""
        key = (None if score is None else round(score, 2), threshold)
        if key == self._match:
            return
        self._match = key
        self.bar.update_idletasks()
        w = max(self.bar.winfo_width(), 10)
        self.bar.delete("all")
        if score is not None:
            s = min(max(score, 0.0), 1.0)
            self.bar.create_rectangle(0, 0, int(w * s), 8, fill=OK if score >= threshold else WARN, outline="")
        tx = int(w * threshold)
        self.bar.create_line(tx, 0, tx, 8, fill=TEXT, width=2)

    def set_alerts(self, items):
        """Warnings under the image. items: [(alert code, "info" / "warning" / "critical"), ...]."""
        items = list(items)[:MAX_WARNINGS]
        if items == self._alerts:
            return
        self._alerts = items
        for wdg in self.alerts_row.winfo_children():
            wdg.destroy()
        if not items:
            tk.Label(self.alerts_row, text="\u25cf  No warnings", bg=BG, fg=OK, font=(FONT, 10)).pack(side="left")
            return
        for code, level in items:
            label, col = LEVEL_LOOK.get(level, LEVEL_LOOK["info"])
            text = WARN_TEXT.get(code, code.replace("_", " ").capitalize())
            chip = tk.Frame(self.alerts_row, bg=col)
            chip.pack(side="left", padx=(0, 6), fill="y")
            tk.Label(chip, text=label, bg=col, fg="white", font=(FONT, 7, "bold"), padx=6).pack(side="left")
            tk.Label(chip, text=text, bg=CARD, fg=TEXT, font=(FONT, 9, "bold"), padx=8).pack(
                side="left", fill="y", padx=(0, 2), pady=2)

    def pop_events(self):
        ev, self.events = self.events, []
        return ev

    @property
    def wears_glasses(self):
        """True if the active driver's profile says he wears glasses."""
        return bool(self.active and self.profiles["drivers"].get(self.active, {}).get("glasses"))

    def has_face(self, name):
        return bool(name and self.profiles["drivers"].get(name, {}).get("faces"))

    def save_faces(self, name, faces):
        if name in self.profiles["drivers"]:
            self.profiles["drivers"][name]["faces"] = faces
            save_profiles(self.profiles)

    def set_recognized(self, name):
        """Recognition result: a name, or None = matches nobody.
        True if the active driver changed (load his calibration)."""
        if name is None:
            if self.how != "unknown face":
                self.active, self.how = None, "unknown face"     # not monitored, no glasses check
                self._refresh()
            return False
        changed = name != self.active
        if changed or self.how != "recognized":
            self.active, self.how = name, "recognized"
            self.profiles["last"] = name
            save_profiles(self.profiles)
            self._refresh()
        return changed

    def close(self):
        if self.alive:
            self.alive = False
            self.root.destroy()

    # ---------- buttons / keys ----------
    def _on_key(self, e):
        if e.widget is self.name_entry:                   # typing a name: not a command
            return
        if e.char and len(e.char) == 1:
            self.key = ord(e.char) & 0xFF

    def _selected(self):
        sel = self.listbox.curselection()
        return self._names[sel[0]] if sel and sel[0] < len(self._names) else None

    def _use(self):
        name = self._selected()
        if name and (name != self.active or self.how == "recognized"):
            self._set_active(name, "picked by hand")
            self.events.append(("select", name))

    def _add(self):
        name = self.name_entry.get().strip()
        if not name:
            return
        if name in self.profiles["drivers"]:
            messagebox.showinfo("Drivers", f"'{name}' already exists.")
            return
        self.profiles["drivers"][name] = {"glasses": bool(self.glasses_now)}   # what the camera sees now
        self.name_entry.delete(0, "end")
        self.root.focus_set()                             # keys (q / f / c) work again
        self._set_active(name, "new driver")
        self.events.append(("add", name))

    def _delete(self):
        name = self._selected()
        if not name:
            return
        self.profiles["drivers"].pop(name, None)
        try:
            os.remove(calib_file(name))
        except OSError:
            pass
        if name == self.active:
            self.active = None
            self.how = "deleted" if self.profiles["drivers"] else "no profile"
            self.profiles["last"] = None
            self.events.append(("select", None))
            self.key = ord("c")                           # cancels his calibration if one is running
        save_profiles(self.profiles)
        self._refresh()

    def _recalibrate(self):
        self.events.append(("calibrate", self.active))

    def _glasses_changed(self):
        if self.active:
            self.profiles["drivers"][self.active]["glasses"] = bool(self.glasses_var.get())
            save_profiles(self.profiles)
            self._refresh()

    def _set_active(self, name, how):
        self.active, self.how = name, how
        self.profiles["last"] = name
        save_profiles(self.profiles)
        self._refresh()

    def _refresh(self):
        self._names = sorted(self.profiles["drivers"])
        self.listbox.delete(0, "end")
        for i, n in enumerate(self._names):
            d = self.profiles["drivers"][n]
            self.listbox.insert("end", f"  {n}" + ("  ·  glasses" if d.get("glasses") else ""))
            if n == self.active:
                self.listbox.itemconfig(i, fg=OK)
        self.listbox.selection_clear(0, "end")
        if self.active in self._names:
            self.listbox.selection_set(self._names.index(self.active))

        self.name_lbl.config(text=self.active or ("Unknown" if self.how == "unknown face" else "No driver"))
        text, col = STATUS_LOOK.get(self.how, (self.how.upper(), MUTED))
        self.pill.config(text=text, bg=col, fg="white")
        self.avatar.delete("all")
        self.avatar.create_oval(2, 2, 44, 44, fill=col if self.active else CARD_HI, outline="")
        initials = "".join(p[0] for p in (self.active or "?").split()[:2]).upper()
        self.avatar.create_text(23, 23, text=initials, fill="white" if self.active else MUTED,
                                font=(FONT, 14, "bold"))
        self.glasses_var.set(self.wears_glasses)
        self.glasses_chk.config(state="normal" if self.active else "disabled")
        _enable(self.recal_btn, bool(self.active))        # only for a saved driver
        _enable(self.relearn_btn, bool(self.active))
