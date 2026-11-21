const $ = id => document.getElementById(id);
const UI_LANG = document.body.dataset.lang === "ro" ? "ro-RO" : "en-US";

const STATE = {
  ok:     { label: "OK",   sub: "Driver looks attentive." },
  warn:   { label: "Warning", sub: "Signs of fatigue or distraction." },
  danger: { label: "Danger",  sub: "Immediate attention needed." },
};

const SUGGESTIONS = {
  ok:     "Attention level optimal. Keep your eyes on the road ahead.",
  warn:   "Signs of drowsiness or distraction detected. Refocus forward or take a short break.",
  danger: "Critical attention drop. Please pull over safely as soon as possible."
};

const NAMES = {
  microsleep: "Microsleep",
  drowsy: "Drowsiness",
  yawn: "Yawning",
  nod: "Head nodding",
  distracted: "Looking away",
  phone: "Phone use",
  eyerub: "Eye rubbing",
  noface: "Driver not visible"
};

const DELAYS = {
  afterCarSpeaks: 600,
  alertCooldown:  4000,
};

let suggestUntil = 0;          // keep the suggestion visible until this time (ms)
const SUGGEST_HOLD_MS = 8000;
function showSuggestion(on) {
  $("guidance").hidden = !on;
  if (!on) $("suggestion-text").textContent = "";
}
function pushSuggestion(text) {
  suggestUntil = Date.now() + SUGGEST_HOLD_MS;
  showSuggestion(true);
  $("suggestion-text").textContent = text;
}

// Set to false once the LLM notifications cover spoken warnings, so the car does not
// read the rule-based alert text AND the LLM message for the same situation.
const RULE_BASED_VOICE = false;
let lastAssistantSpeechAt = 0;

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let lastAlertSpeechAt = 0;

// ---- live events ----
const src = new EventSource("/events");
src.onmessage = e => {
  const ev = JSON.parse(e.data);
  if (ev.type === "metrics") onMetrics(ev);
  else if (ev.type === "alert") onAlert(ev);
  else if (ev.type === "action") onAction(ev);
  else if (ev.type === "assistant") onAssistant(ev);
};

// number -> rounded degrees; anything missing or non-numeric -> "-"
const deg = v => (v == null || Number.isNaN(Number(v))) ? "-" : Math.round(Number(v)) + "°";

let activeFlags = [];
const flagSince = {}, flagLast = {};   // flag -> when it became active / when it was last seen
const FLAG_GRACE_MS = 1500;      // a flag may drop out for up to this long without restarting its timer (detector flicker)
function onMetrics({ metrics: m, flags, state, t, yawns, source, vehicle }) {
  activeFlags = flags;
  const nowMs = Date.now();
  flags.forEach(f => { if (!(f in flagSince)) flagSince[f] = nowMs; flagLast[f] = nowMs; });
  Object.keys(flagSince).forEach(f => {
    if (!flags.includes(f) && nowMs - flagLast[f] > FLAG_GRACE_MS) { delete flagSince[f]; delete flagLast[f]; }
  });
  renderVehicle(vehicle);
  updateActionRows(vehicle);
  handleAsleep(flags);
  $("hero").className = "panel status-panel state-" + state;
  $("hero-state").textContent = STATE[state].label;
  $("hero-sub").textContent = flags.length ? flags.map(f => NAMES[f]).join(", ") : STATE[state].sub;
  const problem = state !== "ok" || flags.length > 0 || Date.now() < suggestUntil;
  showSuggestion(problem);
  if (problem && !$("suggestion-text").textContent) $("suggestion-text").textContent = SUGGESTIONS[state === "ok" ? "warn" : state];

  // live values
  const noFace = m.face === false;
  const f = (v, fn) => (noFace || v == null ? "-" : fn(v));
  const yn = v => (v ? "Yes" : "No");
  $("m-ear").textContent = f(m.ear, v => Math.round(Math.max(0, Math.min(100, 100 - v * 25))) + "%");
  $("m-perclos").textContent = f(m.perclos, v => Math.round(v * 100) + "%");
  $("m-blinks").textContent = f(m.blink_rate, v => Math.round(v) + "/min" + (m.blink_ms == null ? "" : ", " + Math.round(m.blink_ms) + " ms"));
  $("m-eyerub").textContent = f(m.eyerub, yn);
  // yaw / pitch: also accept the raw detector names (head_yaw / head_pitch) if the server passes them through
  $("m-yaw").textContent = noFace ? "-" : deg(m.yaw ?? m.head_yaw);
  $("m-pitch").textContent = noFace ? "-" : deg(m.pitch ?? m.head_pitch);
  $("m-face").textContent = noFace ? "No" : "Yes";
  $("m-mar").textContent = noFace ? "-" : (m.mar != null ? m.mar.toFixed(2) : (m.yawning ? "Open" : "Closed"));
  $("m-phone").textContent = f(m.phone, yn);

  // session statistics
  stats.yawns5 = yawns;
  if (stats.lastT != null) {
    const dt = Math.min(t - stats.lastT, 2);
    stats.totT += dt;
    if (state === "ok") stats.okT += dt;
  }
  stats.lastT = t;
  if (m.level != null) stats.maxLevel = Math.max(stats.maxLevel, m.level);
  renderCounts();
}

function onAlert(a) {
  stats.alerts++; stats.counts[a.kind] = (stats.counts[a.kind] || 0) + 1;
  $("s-alerts").textContent = stats.alerts;
  renderCounts();
  pushSuggestion(a.text);
  const row = addLog(a.t, NAMES[a.kind] || a.kind, a.severity, a.text);
  if (a.severity >= 2) {
    const count = a.severity >= 3 ? 3 : 2;
    const fire = () => { if (playBeeps(count)) { row.beeps = count; renderActions(row); } };   // shows up in "Car actions"
    if (BEEP_AFTER_MS <= 0) fire();
    else {
      const arrived = Date.now();
      setTimeout(() => {
        // still there, and it did not go away for longer than FLAG_GRACE_MS in between
        const since = flagSince[a.kind];
        if (since != null && since <= arrived + 1500) fire();
        else console.log("[beep] skipped: the problem did not last");
      }, BEEP_AFTER_MS);
    }
  }

  const now = Date.now();
  if (RULE_BASED_VOICE && a.severity >= 2 && (now - lastAlertSpeechAt) > DELAYS.alertCooldown) {
    lastAlertSpeechAt = now;
    say(a.text, a.lang);
  }
}

function onAction(a) {
  stats.maxLevel = Math.max(stats.maxLevel, a.level);
  renderCounts();
  setSiren("action", a.actions.includes("alarm"));
  if (a.level < 2) return;
  const sev = a.level >= 3 ? 3 : 2;
  pushSuggestion(a.text);
  addLog(a.t, "Intervention (Level " + a.level + ")", sev, a.text);

  if (RULE_BASED_VOICE && (a.level === 2 || a.level === 4)) {
    say(a.text, a.lang);
  }
}

// What the copilot was reacting to (llm.py "trigger") -> text for the Detection column
const TRIGGER_NAMES = {
  drowsy: "Drowsiness", distracted: "Looking away", phone: "Phone use",
  yawning: "Yawning", hypnosis: "Highway hypnosis", no_face: "Driver not visible",
};
const triggerLabel = t => TRIGGER_NAMES[t] || (t ? t.charAt(0).toUpperCase() + t.slice(1).replace(/_/g, " ") : "Situation");

// LLM notification pushed by the server (no driver input needed):
//   { type: "assistant", t, text, lang?, severity?, trigger?, why?: [..], reason? }  severity 1 = info, 2 = warning, 3 = danger
//   trigger = what it reacts to, e.g. "phone", "drowsy" or an outside event like "pedestrian"
function onAssistant(a) {
  const sev = a.severity || 2;
  addLog(a.t || Date.now() / 1000, triggerLabel(a.trigger), sev, a.text, a.why, a.reason);
  pushSuggestion(a.text);

  const now = Date.now();
  if (sev >= 2 && (now - lastAssistantSpeechAt) > DELAYS.alertCooldown) {
    lastAssistantSpeechAt = now;
    say(a.text, a.lang);
  }
}

function addLog(t, what, sev, text, why, reason) {
  // optional second/third line under the message: the variables (and the model's reason) behind it
  const detail =
    (why && why.length ? `<div class="log-why">Based on: ${why.map(esc).join(" · ")}</div>` : "") +
    (reason ? `<div class="log-why">Reason: ${esc(reason)}</div>` : "");
  const tr = document.createElement("tr");
  tr.className = "new";
  tr.innerHTML =
    `<td>${new Date(t * 1000).toLocaleTimeString()}</td><td>${esc(what)}</td><td class="sev-${sev}">${sev}</td>` +
    `<td>${esc(text)}${detail}</td><td class="act-cell">-</td>`;
  $("log-body").prepend(tr);
  return trackActions(tr.querySelector(".act-cell"));
}

// ---- car actions shown in the log ----
// Each log row keeps watching the vehicle for ACTION_WINDOW_MS and lists what the car did meanwhile
// (the same values as the Vehicle summary tab).
const ACTION_WINDOW_MS = 12000;
const THROTTLE_NORMAL = 0.3;        // normal throttle of the simulated vehicle
let lastVehicle = null;
const actionRows = [];

function trackActions(cell) {
  const r = { cell, born: Date.now(), peakBrake: 0, minThrottle: null,
              hazard: false, seat: false, wheel: false, cruiseOff: false, siren: false };
  collectActions(r, lastVehicle);
  renderActions(r);
  actionRows.push(r);
  return r;
}

function collectActions(r, v) {
  if (v) {
    if (typeof v.brake === "number") r.peakBrake = Math.max(r.peakBrake, v.brake);
    if (typeof v.throttle === "number") r.minThrottle = r.minThrottle == null ? v.throttle : Math.min(r.minThrottle, v.throttle);
    if (v.hazard_lights) r.hazard = true;
    if (v.seat_vibration) r.seat = true;
    if (v.wheel_vibration) r.wheel = true;
    if (v.cruise_control === false) r.cruiseOff = true;
  }
  if (sirenWanted) r.siren = true;
}

function renderActions(r) {
  const out = [];
  if (r.siren) out.push("Alarm siren");
  if (r.beeps) out.push("Beeps \u00d7" + r.beeps);
  if (r.peakBrake > 0.02) out.push("Braking " + Math.round(r.peakBrake * 100) + "%");
  if (r.minThrottle != null) {
    if (r.minThrottle < 0.02) out.push("Throttle cut");
    else if (r.minThrottle < THROTTLE_NORMAL * 0.85) out.push("Throttle reduced");
  }
  if (r.hazard) out.push("Hazard lights");
  if (r.seat) out.push("Seat vibration");
  if (r.wheel) out.push("Wheel vibration");
  if (r.cruiseOff) out.push("Cruise control off");
  r.cell.innerHTML = out.length ? out.map(t => `<span class="act">${esc(t)}</span>`).join("") : "-";
}

function updateActionRows(v) {
  if (v) lastVehicle = v;
  const now = Date.now();
  for (let i = actionRows.length - 1; i >= 0; i--) {
    const r = actionRows[i];
    collectActions(r, v);
    renderActions(r);
    if (now - r.born > ACTION_WINDOW_MS) actionRows.splice(i, 1);   // window over: the row keeps its last text
  }
}

// ---- car voice (speech out) ----
function say(text, lang, onend) {
  if (!text || !$("voice-toggle").checked || !("speechSynthesis" in window)) {
    if (onend) setTimeout(onend, DELAYS.afterCarSpeaks);
    return;
  }
  const u = new SpeechSynthesisUtterance(text);
  u.rate = 0.75;      // 0.1 to 10, normal = 1
  u.pitch = 1.0;     // 0 to 2
  u.volume = 1.0;    // 0 to 1
  u.lang = lang ? (lang === "ro" ? "ro-RO" : "en-US") : UI_LANG;
  const voices = speechSynthesis.getVoices();
  const v = voices.find(x => x.lang === u.lang && /Google|Microsoft|Natural/i.test(x.name)) || voices.find(x => x.lang === u.lang);
  if (v) u.voice = v;
  if (onend) u.onend = () => setTimeout(onend, DELAYS.afterCarSpeaks);
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ---- alarm siren: police-style two-tone beeps ----
// Plays while the driver looks asleep (see ASLEEP_FLAGS) or when the server sends an "alarm" action.
// Browsers only allow sound after the first click or key press on the page.
const SIREN_HI = 960, SIREN_LO = 720;   // Hz, the two tones
const SIREN_STEP_MS = 380;              // one beep every 380 ms, alternating high / low (about 1.6 beeps per second)
const SIREN_VOLUME = 0.14;              // 0..1
const ASLEEP_FLAGS = ["microsleep"];    // flags from the server that count as "asleep"
let ac = null, sirenTimer = null, sirenOsc = null, sirenGain = null, sirenWanted = false, asleepOffTimer = null;
const sirenSources = { action: false, asleep: false };

function audioCtx() {
  ac = ac || new (window.AudioContext || window.webkitAudioContext)();
  if (ac.state === "suspended") ac.resume();
  return ac;
}
// Browsers block speech AND beeps until the user has clicked or pressed a key on the page.
// The first click anywhere unlocks both, plays a short confirmation beep and hides the prompt button.
let soundReady = false;
function unlockSound() {
  if (soundReady) return;
  soundReady = true;
  try {
    const ctx = audioCtx(), t = ctx.currentTime;
    const o = ctx.createOscillator(), g = ctx.createGain();
    o.type = "sine"; o.frequency.value = 660;
    g.gain.setValueAtTime(0.0001, t);
    g.gain.exponentialRampToValueAtTime(0.12, t + 0.02);
    g.gain.exponentialRampToValueAtTime(0.0001, t + 0.25);
    o.connect(g).connect(ctx.destination); o.start(t); o.stop(t + 0.3);
  } catch {}
  try {
    if ("speechSynthesis" in window) { speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance("")); }
  } catch {}
  if ($("sound-btn")) $("sound-btn").hidden = true;
}
["pointerdown", "keydown"].forEach(ev => document.addEventListener(ev, unlockSound));

function startSiren() {
  if (sirenTimer) return;
  try {
    const ctx = audioCtx();
    sirenGain = ctx.createGain(); sirenGain.gain.value = 0;
    sirenOsc = ctx.createOscillator(); sirenOsc.type = "square";
    sirenOsc.connect(sirenGain).connect(ctx.destination); sirenOsc.start();
    let hi = false;
    const beep = () => {
      hi = !hi;
      const t = ctx.currentTime, d = SIREN_STEP_MS / 1000;
      sirenOsc.frequency.setValueAtTime(hi ? SIREN_HI : SIREN_LO, t);
      sirenGain.gain.cancelScheduledValues(t);
      sirenGain.gain.setValueAtTime(0.0001, t);
      sirenGain.gain.exponentialRampToValueAtTime(SIREN_VOLUME, t + 0.03);
      sirenGain.gain.setValueAtTime(SIREN_VOLUME, t + d - 0.09);
      sirenGain.gain.exponentialRampToValueAtTime(0.0001, t + d - 0.03);
    };
    beep();
    sirenTimer = setInterval(beep, SIREN_STEP_MS);
  } catch {}
}

function stopSiren() {
  if (!sirenTimer) return;
  clearInterval(sirenTimer); sirenTimer = null;
  try { sirenOsc.stop(); sirenOsc.disconnect(); sirenGain.disconnect(); } catch {}
  sirenOsc = sirenGain = null;
}

function setSiren(source, on) {
  sirenSources[source] = on;
  const wanted = sirenSources.action || sirenSources.asleep;
  const was = sirenWanted;
  sirenWanted = wanted;
  if (wanted && !was && source === "asleep") {
    addLog(Date.now() / 1000, "Driver asleep", 3, "Driver appears asleep. Alarm siren started.");
  }
  const allowed = $("alarm-toggle") ? $("alarm-toggle").checked : true;
  (wanted && allowed) ? startSiren() : stopSiren();
}
if ($("alarm-toggle")) $("alarm-toggle").onchange = () => setSiren("action", sirenSources.action);

// every metrics event with an "asleep" flag keeps the siren going; it stops about 2.5 s after the flag is gone
function handleAsleep(flags) {
  if (!flags.some(f => ASLEEP_FLAGS.includes(f))) return;
  setSiren("asleep", true);
  clearTimeout(asleepOffTimer);
  asleepOffTimer = setTimeout(() => setSiren("asleep", false), 2500);
}

// ---- alert beeps: short beeps for any warning/danger alert (the siren above is only for "asleep") ----
const BEEP_HZ = 880, BEEP_VOLUME = 0.12;
const BEEP_AFTER_MS = 3000;   // the problem must last this long, without a break, before the beeps start (0 = at once)
const BEEP_GAP_MS = 2500;     // minimum time between two beep bursts
let lastBeepAt = 0;
function playBeeps(count) {
  if ($("alarm-toggle") && !$("alarm-toggle").checked) { console.log("[beep] skipped: Alarm sound is off"); return false; }
  if (Date.now() - lastBeepAt < BEEP_GAP_MS) { console.log("[beep] skipped: too soon after the last one"); return false; }
  if (sirenTimer) { console.log("[beep] skipped: the siren is already playing"); return false; }
  try {
    const ctx = audioCtx();                                  // also resumes a suspended context
    if (ctx.state !== "running") { console.log("[beep] skipped: audio is locked, click the page once"); return false; }
    lastBeepAt = Date.now();
    for (let i = 0; i < count; i++) {
      const t = ctx.currentTime + 0.02 + i * 0.22;
      const o = ctx.createOscillator(), g = ctx.createGain();
      o.type = "square"; o.frequency.value = BEEP_HZ;
      g.gain.setValueAtTime(0.0001, t);
      g.gain.exponentialRampToValueAtTime(BEEP_VOLUME, t + 0.015);
      g.gain.setValueAtTime(BEEP_VOLUME, t + 0.10);
      g.gain.exponentialRampToValueAtTime(0.0001, t + 0.14);
      o.connect(g).connect(ctx.destination); o.start(t); o.stop(t + 0.16);
    }
    return true;
  } catch (e) { console.log("[beep] failed:", e); return false; }
}

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}


// ---- session summary ----
const stats = { start: Date.now(), alerts: 0, maxLevel: 0, okT: 0, totT: 0, lastT: null, counts: {}, yawns5: 0 };
const pad = n => String(n).padStart(2, "0");
setInterval(() => {
  const s = Math.floor((Date.now() - stats.start) / 1000);
  $("s-time").textContent = pad(Math.floor(s / 60)) + ":" + pad(s % 60);
}, 1000);

// the "This session" column of the driver summary
function renderCounts() {
  const n = k => stats.counts[k] || 0;
  $("c-perclos").textContent = `Drowsy: ${n("drowsy")} / Microsleep: ${n("microsleep")} `;
  $("c-eyerub").textContent = `${n("eyerub")} `;
  $("c-yaw").textContent = `Looking away: ${n("distracted")} `;
  $("c-pitch").textContent = `Nodding: ${n("nod")} `;
  $("c-face").textContent = `Face not visible: ${n("noface")} `;
  $("c-mar").textContent = `Yawning: ${n("yawn")}  / ${stats.yawns5} in last 5 min`;
  $("c-phone").textContent = `${n("phone")} `;
  $("c-alerts").textContent = `Highest level: ${stats.maxLevel}`;
  $("s-ok").textContent = stats.totT ? Math.round(100 * stats.okT / stats.totT) + "%" : "-";
}
renderCounts();

// ---- summary tabs (Driver / Vehicle) ----
document.querySelectorAll(".tab").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll(".tab").forEach(b => {
      const on = b === btn;
      b.classList.toggle("on", on);
      b.setAttribute("aria-selected", on);
      $("tab-" + b.dataset.tab).hidden = !on;
    });
  };
});

// ---- vehicle summary ----
// Reads `vehicle` from the "metrics" event:
//   { speed_kmh, throttle (0..1), brake (0..1), hazard_lights, seat_vibration, wheel_vibration, cruise_control }
// Missing values show "-".
const pct   = v => Math.round(v * 100) + "%";
const onOff = v => (v ? "On" : "Off");
const activeOff = v => (v ? "Active" : "Off");
const VEHICLE_FIELDS = [
  ["v-speed",    "speed_kmh",       v => Math.round(v) + " km/h"],
  ["v-throttle", "throttle",        pct],
  ["v-brake",    "brake",           pct],
  ["v-hazard",   "hazard_lights",   onOff],
  ["v-seat",     "seat_vibration",  activeOff],
  ["v-wheel",    "wheel_vibration", activeOff],
  ["v-cruise",   "cruise_control",  onOff],
];
function renderVehicle(v) {
  VEHICLE_FIELDS.forEach(([id, key, fmt]) => {
    const val = v ? v[key] : null;
    $(id).textContent = val == null ? "-" : fmt(val);
  });
}