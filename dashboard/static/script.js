const $ = id => document.getElementById(id);
const UI_LANG = document.body.dataset.lang === "ro" ? "ro-RO" : "en-US";

const STATE = {
  ok:     { label: "Alert",   sub: "Driver looks attentive." },
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

const SEV = { 1: "Info", 2: "Warning", 3: "Danger" };

const DELAYS = {
  afterMicPrompt: 350,
  afterCarSpeaks: 600,
  alertCooldown:  4000,
  micHoldTime:    400,
};

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));
let lastAlertSpeechAt = 0;

// ---- live events ----
const src = new EventSource("/events");
src.onopen = () => { $("conn-light").className = "dot"; $("conn-text").textContent = "Live"; };
src.onerror = () => { $("conn-light").className = "dot off"; $("conn-text").textContent = "Reconnecting…"; };
src.onmessage = e => {
  const ev = JSON.parse(e.data);
  if (ev.type === "metrics") onMetrics(ev);
  else if (ev.type === "alert") onAlert(ev);
  else if (ev.type === "action") onAction(ev);
};

function onMetrics({ metrics: m, flags, state, t, yawns, source, vehicle }) {
  latest.driver = m; latest.state = state; latest.flags = flags;
  renderVehicle(vehicle);
  $("hero").className = "panel status-panel state-" + state;
  $("hero-state").textContent = STATE[state].label;
  $("hero-sub").textContent = flags.length ? flags.map(f => NAMES[f]).join(", ") : STATE[state].sub;
  if (!flags.length) $("suggestion-text").textContent = SUGGESTIONS[state];

  // live values
  const noFace = m.face === false;
  const f = (v, fn) => (noFace || v == null ? "—" : fn(v));
  const yn = v => (v ? "Yes" : "No");
  $("m-ear").textContent = f(m.ear, v => v.toFixed(2));
  $("m-perclos").textContent = f(m.perclos, v => Math.round(v * 100) + "%");
  $("m-blinks").textContent = f(m.blink_rate, v => Math.round(v) + "/min" + (m.blink_ms == null ? "" : ", " + Math.round(m.blink_ms) + " ms"));
  $("m-eyerub").textContent = f(m.eyerub, yn);
  $("m-yaw").textContent = f(m.yaw, v => Math.round(v) + "°");
  $("m-pitch").textContent = f(m.pitch, v => Math.round(v) + "°");
  $("m-face").textContent = noFace ? "No" : "Yes";
  $("m-mar").textContent = f(m.mar, v => v.toFixed(2));
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
  renderCounts();
  $("suggestion-text").textContent = a.text;
  addMsg("car", a.text);
  addLog(a.t, NAMES[a.kind] || a.kind, a.severity, a.text);

  const now = Date.now();
  if (a.severity >= 2 && (now - lastAlertSpeechAt) > DELAYS.alertCooldown) {
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
  $("suggestion-text").textContent = a.text;
  addMsg("car", a.text);
  addLog(a.t, "Intervention (Level " + a.level + ")", sev, a.text);

  if (a.level === 2) {
    say(a.text, a.lang, () => listen(DELAYS.afterCarSpeaks));
  } else if (a.level === 4) {
    say(a.text, a.lang);
  }
}

function addLog(t, what, sev, text) {
  $("log-body").insertAdjacentHTML(
    "afterbegin",
    `<tr><td>${new Date(t * 1000).toLocaleTimeString()}</td><td>${esc(what)}</td><td class="sev-${sev}">${SEV[sev]}</td><td>${esc(text)}</td></tr>`
  );
}

// ---- car voice (speech out) ----
function say(text, lang, onend) {
  if (!$("voice-toggle").checked || !("speechSynthesis" in window)) {
    if (onend) setTimeout(onend, DELAYS.afterCarSpeaks);
    return;
  }
  const u = new SpeechSynthesisUtterance(text);
  u.lang = lang ? (lang === "ro" ? "ro-RO" : "en-US") : UI_LANG;
  if (onend) u.onend = () => setTimeout(onend, DELAYS.afterCarSpeaks);
  speechSynthesis.cancel();
  speechSynthesis.speak(u);
}

// ---- talk to the car (speech in) ----
const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
let rec = null;
let micGranted = false;
let micBusy = false;

function setMic(on, hintText) {
  $("mic").classList.toggle("listening", on);
  $("mic-hint").textContent = hintText || (on ? "Listening…" : "Tap or press Space to talk");
}

async function ensureMicPermission() {
  if (micGranted) return true;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    addMsg("car", "Microphone blocked. Open via http://localhost:5000.");
    return false;
  }
  try {
    setMic(false, "Requesting mic permission…");
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    stream.getTracks().forEach(track => track.stop());
    micGranted = true;
    setMic(false, "Preparing microphone…");
    await sleep(DELAYS.afterMicPrompt);
    return true;
  } catch {
    micGranted = false;
    setMic(false, "Mic blocked — allow microphone in browser address bar");
    addMsg("car", "Microphone permission was blocked. Click the lock icon next to the URL bar and set Microphone to Allow.");
    return false;
  }
}

async function listen(initialDelayMs = 0) {
  if (!SR) {
    addMsg("car", "Voice input requires Microsoft Edge or Chrome.");
    return;
  }
  if (rec || micBusy) return;
  micBusy = true;

  try {
    if (initialDelayMs > 0) {
      setMic(false, "Get ready to speak…");
      await sleep(initialDelayMs);
    }

    const allowed = await ensureMicPermission();
    if (!allowed) return;

    speechSynthesis.cancel();
    setMic(false, "Starting mic…");
    await sleep(DELAYS.micHoldTime);

    rec = new SR();
    rec.lang = UI_LANG;
    rec.interimResults = false;

    rec.onstart = () => setMic(true, "Listening… speak now");
    rec.onend = () => { rec = null; setMic(false); };
    rec.onerror = e => {
      rec = null;
      setMic(false);
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        micGranted = false;
        addMsg("car", "Microphone blocked in browser permissions.");
      } else if (!["no-speech", "aborted"].includes(e.error)) {
        addMsg("car", "I couldn't hear you (" + e.error + ").");
      }
    };
    rec.onresult = async e => {
      const text = e.results[0][0].transcript;
      addMsg("me", text);
      await sleep(250);
      try {
        const r = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, context: { driver: latest.driver, vehicle: latest.vehicle, state: latest.state, flags: latest.flags } })
        });
        const reply = (await r.json()).reply;
        addMsg("car", reply);
        say(reply);
      } catch {
        addMsg("car", "Cannot reach the server.");
      }
    };

    rec.start();
  } catch {
    rec = null;
    setMic(false);
  } finally {
    micBusy = false;
  }
}

$("mic").onclick = () => (rec ? rec.stop() : listen(0));
document.addEventListener("keydown", e => {
  if (e.code === "Space" && e.target === document.body) {
    e.preventDefault();
    rec ? rec.stop() : listen(0);
  }
});

// ---- helpers (plain-text chat matching log rows) ----
function addMsg(who, text) {
  const chat = $("chat");
  const sender = who === "me" ? "Driver" : "System";
  chat.insertAdjacentHTML(
    "beforeend",
    `<div class="chat-row${who === "me" ? " me" : ""}"><span class="chat-meta">${sender}</span><span class="chat-text">${esc(text)}</span></div>`
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
  $("s-ok").textContent = stats.totT ? Math.round(100 * stats.okT / stats.totT) + "%" : "—";
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
// Missing values show "—".
const latest = { driver: null, vehicle: null, state: null, flags: [] };
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
  latest.vehicle = v || null;
  VEHICLE_FIELDS.forEach(([id, key, fmt]) => {
    const val = v ? v[key] : null;
    $(id).textContent = val == null ? "—" : fmt(val);
  });
}