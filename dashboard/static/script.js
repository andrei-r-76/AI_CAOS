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

function onMetrics({ metrics: m, flags, state, t, yawns, source, vehicle }) {
  renderVehicle(vehicle);
  $("hero").className = "panel status-panel state-" + state;
  $("hero-state").textContent = STATE[state].label;
  $("hero-sub").textContent = flags.length ? flags.map(f => NAMES[f]).join(", ") : STATE[state].sub;
  // Live Suggestion is only shown while there is a problem (or shortly after an alert/intervention)
  const problem = state !== "ok" || flags.length > 0 || Date.now() < suggestUntil;
  showSuggestion(problem);
  if (problem && !$("suggestion-text").textContent) $("suggestion-text").textContent = SUGGESTIONS[state === "ok" ? "warn" : state];

  // live values
  const noFace = m.face === false;
  const f = (v, fn) => (noFace || v == null ? "-" : fn(v));
  const yn = v => (v ? "Yes" : "No");
  $("m-ear").textContent = f(m.ear, v => v.toFixed(2));
  $("m-perclos").textContent = f(m.perclos, v => Math.round(v * 100) + "%");
  $("m-blinks").textContent = f(m.blink_rate, v => Math.round(v) + "/min" + (m.blink_ms == null ? "" : ", " + Math.round(m.blink_ms) + " ms"));
  $("m-eyerub").textContent = f(m.eyerub, yn);
  $("m-yaw").textContent = f(m.yaw, v => Math.round(v) + "°");
  $("m-pitch").textContent = f(m.pitch, v => Math.round(v) + "°");
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
  renderCounts();
}

function onAlert(a) {
  stats.alerts++; stats.counts[a.kind] = (stats.counts[a.kind] || 0) + 1;
  $("s-alerts").textContent = stats.alerts;
  if (m.level != null) stats.maxLevel = Math.max(stats.maxLevel, m.level);
  renderCounts();
  pushSuggestion(a.text);
  addLog(a.t, NAMES[a.kind] || a.kind, a.severity, a.text);

  const now = Date.now();
  if (RULE_BASED_VOICE && a.severity >= 2 && (now - lastAlertSpeechAt) > DELAYS.alertCooldown) {
    lastAlertSpeechAt = now;
    say(a.text, a.lang);
  }
}

function onAction(a) {
  stats.maxLevel = Math.max(stats.maxLevel, a.level);
  renderCounts();
  alarm(a.actions.includes("alarm"));
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
  addMsg("ai", a.text);
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
  $("log-body").insertAdjacentHTML(
    "afterbegin",
    `<tr class="new"><td>${new Date(t * 1000).toLocaleTimeString()}</td><td>${esc(what)}</td><td class="sev-${sev}">${sev}</td><td>${esc(text)}${detail}</td></tr>`
  );
}

// ---- car voice (speech out) ----
function say(text, lang, onend) {
  if (!text || !$("voice-toggle").checked || !("speechSynthesis" in window)) {
    if (onend) setTimeout(onend, DELAYS.afterCarSpeaks);
    return;
  }
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang ? (lang === "ro" ? "ro-RO" : "en-US") : UI_LANG;
  if (onend) u.onend = () => setTimeout(onend, DELAYS.afterCarSpeaks);
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ---- helpers (plain-text chat matching log rows) ----
function addMsg(who, text) {
  const chat = $("chat");
  const sender = "Assistant";
  chat.insertAdjacentHTML(
    "beforeend",
    `<div class="chat-row"><span class="chat-meta">${sender}</span><span class="chat-text">${esc(text)}</span></div>`
  );
  while (chat.children.length > 30) chat.firstChild.remove();
  chat.scrollTop = chat.scrollHeight;
}

let ac, tick, osc;
function alarm(on) {
  try {
    if (on && !tick) {
      ac = ac || new AudioContext();
      const gain = ac.createGain(); gain.gain.value = 0;
      osc = ac.createOscillator(); osc.type = "square"; osc.frequency.value = 880;
      osc.connect(gain).connect(ac.destination); osc.start();
      let loud = false;
      tick = setInterval(() => { loud = !loud; gain.gain.value = loud ? 0.08 : 0; }, 300);
    } else if (!on && tick) { clearInterval(tick); tick = null; osc.stop(); }
  } catch {}
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