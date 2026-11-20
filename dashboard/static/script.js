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
  phone: "Phone use"
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

function onMetrics({ metrics: m, flags, state }) {
  $("hero").className = "panel status-panel state-" + state;
  $("hero-state").textContent = STATE[state].label;
  $("hero-sub").textContent = flags.length ? flags.map(f => NAMES[f]).join(", ") : STATE[state].sub;

  if (!flags.length) {
    $("suggestion-text").textContent = SUGGESTIONS[state];
  }

  $("m-perclos").textContent = Math.round(m.perclos * 100) + "%";
  $("m-ear").textContent = m.ear.toFixed(2);
  $("m-phone").textContent = m.phone ? "Yes" : "No";
  $("m-yaw").textContent = Math.round(m.yaw) + "°";
  $("m-pitch").textContent = Math.round(m.pitch) + "°";
  $("m-mar").textContent = m.mar.toFixed(2);
}

function onAlert(a) {
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
          body: JSON.stringify({ message: text })
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

// ---- demo drawer ----
$("demo-toggle").onclick = () => { $("demo").hidden = !$("demo").hidden; };
document.querySelectorAll(".scen").forEach(b => b.onclick = () => {
  document.querySelectorAll(".scen").forEach(x => x.classList.toggle("on", x === b));
  fetch("/api/scenario", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name: b.dataset.name }) });
});