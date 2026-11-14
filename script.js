// ── STATE ─────────────────────────────────────────────────────────────────────
let audioCtx, analyser, micStream, animId;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let countdownTimer = null;

const RECORD_SECONDS = 5;

// ── DOM HELPERS ───────────────────────────────────────────────────────────────
const $  = id => document.getElementById(id);
const statusDot  = () => $("src-dot");
const statusText = () => $("status-text");

function setStatus(msg, state="idle") {
    statusText().textContent = msg;
    const dot = statusDot();
    dot.className = "status-dot";
    if (state === "live")    dot.classList.add("live");
    if (state === "rec")     dot.classList.add("recording");
    if (state === "ok")      dot.classList.add("ok");
    if (state === "err")     dot.classList.add("err");
}

// ── INIT VISUALIZERS ──────────────────────────────────────────────────────────
function initFreqGrid() {
    const grid = $("freq-grid");
    if (!grid || grid.children.length > 0) return;
    const bands = ["Sub","Bass","Low-Mid","Mid","High-Mid","High","Air"];
    bands.forEach(b => {
        const col = document.createElement("div");
        col.className = "freq-col";
        col.innerHTML = `<div class="freq-bar-wrap"><div class="freq-bar" id="fb-${b}"></div></div><span>${b}</span>`;
        grid.appendChild(col);
    });
}

// ── START MIC ─────────────────────────────────────────────────────────────────
async function startListening() {
    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        audioCtx  = new (window.AudioContext || window.webkitAudioContext)();
        analyser  = audioCtx.createAnalyser();
        analyser.fftSize = 2048;
        audioCtx.createMediaStreamSource(micStream).connect(analyser);

        initFreqGrid();
        drawVisualizers();
        setStatus("Mic active — click Record 5s", "live");

        $("btn-start").disabled  = true;
        $("btn-record").disabled = false;
        $("btn-stop").disabled   = false;
    } catch(e) {
        setStatus("Mic denied: " + e.message, "err");
    }
}

// ── DRAW WAVEFORM + FREQ BARS ─────────────────────────────────────────────────
function drawVisualizers() {
    const canvas = $("waveform");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    const bufLen = analyser.frequencyBinCount;
    const timeBuf = new Uint8Array(bufLen);
    const freqBuf = new Uint8Array(bufLen);

    const freqRanges = [
        [20,80],[80,250],[250,500],[500,2000],[2000,4000],[4000,8000],[8000,20000]
    ];
    const bandIds = ["Sub","Bass","Low-Mid","Mid","High-Mid","High","Air"];
    const SR = audioCtx.sampleRate;

    function loop() {
        animId = requestAnimationFrame(loop);
        analyser.getByteTimeDomainData(timeBuf);
        analyser.getByteFrequencyData(freqBuf);

        // Waveform
        ctx.clearRect(0,0,W,H);
        ctx.strokeStyle = isRecording ? "#ff4444" : "#00d4aa";
        ctx.lineWidth = 2;
        ctx.beginPath();
        const sliceW = W / bufLen;
        let x = 0;
        for (let i = 0; i < bufLen; i++) {
            const v = timeBuf[i] / 128.0;
            const y = v * H / 2;
            i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
            x += sliceW;
        }
        ctx.stroke();

        // dB meter
        const rms = Math.sqrt(timeBuf.reduce((s,v)=>s+((v-128)/128)**2,0)/bufLen);
        const db  = 20 * Math.log10(rms + 1e-9);
        const pct = Math.max(0, Math.min(100, (db + 60) / 60 * 100));
        const dbEl = $("db-val"); if(dbEl) dbEl.textContent = db.toFixed(1) + " dB";
        const fillEl = $("db-fill"); if(fillEl) fillEl.style.width = pct + "%";

        // Freq bands
        const binHz = SR / analyser.fftSize;
        freqRanges.forEach(([lo,hi], i) => {
            const lo_bin = Math.floor(lo / binHz);
            const hi_bin = Math.min(Math.ceil(hi / binHz), bufLen - 1);
            let sum = 0;
            for (let b = lo_bin; b <= hi_bin; b++) sum += freqBuf[b];
            const avg = sum / (hi_bin - lo_bin + 1);
            const el = $("fb-" + bandIds[i]);
            if (el) el.style.height = (avg / 255 * 100) + "%";
        });
    }
    loop();
}

// ── RECORD 5 SECONDS (AUTO-STOP) ─────────────────────────────────────────────
function startRecording() {
    if (!micStream || isRecording) return;

    recordedChunks = [];
    isRecording = true;

    // Prefer wav-capable mimeType, fallback to webm
    let mimeType = "audio/webm";
    if (MediaRecorder.isTypeSupported("audio/wav")) mimeType = "audio/wav";
    else if (MediaRecorder.isTypeSupported("audio/ogg")) mimeType = "audio/ogg";

    mediaRecorder = new MediaRecorder(micStream, { mimeType });
    mediaRecorder.ondataavailable = e => { if (e.data.size > 0) recordedChunks.push(e.data); };
    mediaRecorder.onstop = () => { isRecording = false; processRecording(mimeType); };
    mediaRecorder.start();

    $("btn-record").disabled = true;
    setStatus(`Recording… 5s`, "rec");

    // Countdown
    let remaining = RECORD_SECONDS;
    countdownTimer = setInterval(() => {
        remaining--;
        if (remaining > 0) {
            setStatus(`Recording… ${remaining}s`, "rec");
            $("btn-record").textContent = `⏺ ${remaining}s`;
        } else {
            clearInterval(countdownTimer);
            countdownTimer = null;
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();   // triggers onstop → processRecording
            }
            $("btn-record").textContent = "⏺ Record 5s";
        }
    }, 1000);
}

// ── PROCESS RECORDING ─────────────────────────────────────────────────────────
function processRecording(mimeType) {
    if (recordedChunks.length === 0) {
        setStatus("No audio captured.", "err");
        $("btn-record").disabled = false;
        return;
    }

    setStatus("Analysing…", "live");
    showResult("thinking");

    const blob = new Blob(recordedChunks, { type: mimeType });
    const ext  = mimeType.includes("wav") ? ".wav" : mimeType.includes("ogg") ? ".ogg" : ".webm";
    const formData = new FormData();
    formData.append("audio", blob, "recording" + ext);

    fetch("/predict", { method: "POST", body: formData })
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                setStatus("Error: " + data.error, "err");
                showResult("error", data.error);
            } else {
                setStatus("Detection complete ✓", "ok");
                showResult(data.result, data.confidence, data.emergency_prob);
                if (data.result === "Emergency") logAlert(data.confidence);
            }
            $("btn-record").disabled = false;
        })
        .catch(err => {
            setStatus("Network error: " + err.message, "err");
            showResult("error", err.message);
            $("btn-record").disabled = false;
        });
}

// ── SHOW RESULT ───────────────────────────────────────────────────────────────
function showResult(type, conf, emergencyProb) {
    const box   = $("result-box");
    const icon  = $("result-icon");
    const label = $("result-label");
    const confEl= $("result-conf");

    box.className = "result-box";

    if (type === "Emergency") {
        box.classList.add("emergency");
        icon.textContent  = "🚨";
        label.textContent = "EMERGENCY DETECTED";
        confEl.textContent = `Confidence: ${conf}% (emergency prob: ${emergencyProb}%)`;
    } else if (type === "Normal") {
        box.classList.add("normal");
        icon.textContent  = "✅";
        label.textContent = "Normal Sound";
        confEl.textContent = `Confidence: ${conf}% (emergency prob: ${emergencyProb}%)`;
    } else if (type === "thinking") {
        icon.textContent  = "⏳";
        label.textContent = "Analysing audio…";
        confEl.textContent = "";
    } else {
        box.classList.add("error");
        icon.textContent  = "⚠️";
        label.textContent = "Detection Error";
        confEl.textContent = conf || "";
    }
}

// ── STOP ALL ──────────────────────────────────────────────────────────────────
function stopAll() {
    if (countdownTimer) { clearInterval(countdownTimer); countdownTimer = null; }
    if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
    if (animId) { cancelAnimationFrame(animId); animId = null; }
    if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
    if (audioCtx)  { audioCtx.close(); audioCtx = null; }

    isRecording = false;
    $("btn-start").disabled  = false;
    $("btn-record").disabled = true;
    $("btn-record").textContent = "⏺ Record 5s";
    $("btn-stop").disabled   = true;
    setStatus("Stopped. Click Start Mic to begin again.", "idle");
}

// ── CONTACTS ──────────────────────────────────────────────────────────────────
function addContact() {
    const name  = $("inp-name").value.trim();
    const phone = $("inp-num").value.trim();
    if (!name || !phone) { alert("Enter name and phone number."); return; }
    fetch("/add_contact", {
        method:"POST",
        headers:{"Content-Type":"application/json"},
        body: JSON.stringify({name, phone})
    })
    .then(r => r.json())
    .then(d => {
        $("inp-name").value = ""; $("inp-num").value = "";
        renderContacts(d.contacts);
    });
}

function renderContacts(contacts) {
    const list = $("contacts-list");
    if (!contacts || contacts.length === 0) {
        list.innerHTML = '<p class="empty-msg">No contacts added yet.</p>'; return;
    }
    list.innerHTML = contacts.map((c,i) =>
        `<div class="contact-item">
            <span class="contact-name">${c.name}</span>
            <span class="contact-num mono">${c.phone}</span>
         </div>`
    ).join("");
}

function logAlert(conf) {
    const list = $("alert-list");
    const empty = list.querySelector(".empty-msg");
    if (empty) empty.remove();
    const now = new Date().toLocaleTimeString();
    const item = document.createElement("div");
    item.className = "alert-item";
    item.innerHTML = `<span class="alert-time mono">${now}</span>
                      <span class="alert-msg">🚨 Emergency detected — ${conf}% confidence</span>`;
    list.prepend(item);
}

// ── INIT ──────────────────────────────────────────────────────────────────────
window.addEventListener("DOMContentLoaded", () => {
    fetch("/get_contacts").then(r=>r.json()).then(renderContacts);
    initFreqGrid();
});