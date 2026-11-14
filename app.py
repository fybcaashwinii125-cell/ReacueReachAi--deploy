import os, json, pickle, tempfile, wave
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

MODEL_PATH     = "model.pkl"
SCALER_PATH    = "scaler.pkl"
THRESHOLD_PATH = "threshold.pkl"
CONTACTS_PATH  = "contacts.json"

model = scaler = None
EMERGENCY_THRESHOLD = 0.35  # default, overridden from file

def load_model():
    global model, scaler, EMERGENCY_THRESHOLD
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        with open(MODEL_PATH,"rb")  as f: model  = pickle.load(f)
        with open(SCALER_PATH,"rb") as f: scaler = pickle.load(f)
        if os.path.exists(THRESHOLD_PATH):
            with open(THRESHOLD_PATH,"rb") as f: EMERGENCY_THRESHOLD = pickle.load(f)
        print(f"Model loaded. Emergency threshold: {EMERGENCY_THRESHOLD}")
    else:
        print("WARNING: Run model_training.py first.")

# ── FEATURE EXTRACTION (identical to training) ────────────────────────────────

def extract_features(y, sr):
    y  = y.astype(np.float32)
    N  = len(y)
    n_fft=512; hop=256; n_mels=26; n_mfcc=13
    fmin,fmax=20,sr//2

    def hz_to_mel(h): return 2595*np.log10(1+h/700)
    def mel_to_hz(m): return 700*(10**(m/2595)-1)

    mel_lo,mel_hi=hz_to_mel(fmin),hz_to_mel(fmax)
    mel_pts=np.linspace(mel_lo,mel_hi,n_mels+2)
    hz_pts=mel_to_hz(mel_pts)
    bin_pts=np.floor((n_fft+1)*hz_pts/sr).astype(int)
    fbank=np.zeros((n_mels,n_fft//2+1))
    for m in range(1,n_mels+1):
        lo,cen,hi=bin_pts[m-1],bin_pts[m],bin_pts[m+1]
        for k in range(lo,cen):
            if cen!=lo: fbank[m-1,k]=(k-lo)/(cen-lo)
        for k in range(cen,hi):
            if hi!=cen: fbank[m-1,k]=(hi-k)/(hi-cen)

    win=np.hanning(n_fft); frames=[]
    for start in range(0,N-n_fft,hop):
        frame=y[start:start+n_fft]*win
        spec=np.abs(np.fft.rfft(frame))**2
        mel=np.dot(fbank,spec); mel=np.where(mel==0,1e-10,mel)
        frames.append(np.log(mel))

    if not frames: return np.zeros(30)
    frames=np.array(frames)
    dct_mat=np.cos(np.pi*np.arange(n_mfcc)[:,None]*(2*np.arange(n_mels)+1)[None,:]/(2*n_mels))
    mfccs=frames@dct_mat.T
    mfcc_mean=np.mean(mfccs,axis=0); mfcc_std=np.std(mfccs,axis=0)

    full_spec=np.abs(np.fft.rfft(y*np.hanning(N),n=N))**2
    freqs=np.fft.rfftfreq(N,1/sr); eps=1e-10
    centroid=np.sum(freqs*full_spec)/(np.sum(full_spec)+eps)
    bandwidth=np.sqrt(np.sum(((freqs-centroid)**2)*full_spec)/(np.sum(full_spec)+eps))
    cumsum=np.cumsum(full_spec); rolloff_idx=np.searchsorted(cumsum,0.85*cumsum[-1])
    rolloff=freqs[min(rolloff_idx,len(freqs)-1)]
    zcr=np.mean(np.abs(np.diff(np.sign(y))))/2
    rms=np.sqrt(np.mean(y**2))
    def band_energy(lo,hi):
        mask=(freqs>=lo)&(freqs<hi); return np.mean(full_spec[mask]) if mask.any() else 0.0
    b_sub=band_energy(20,250); b_mid=band_energy(250,2000)
    b_high=band_energy(2000,6000); b_treble=band_energy(6000,sr//2)
    total=np.sum(full_spec)+eps
    extra=np.array([centroid/sr,bandwidth/sr,rolloff/sr,zcr,rms,
                    b_sub/total,b_mid/total,b_high/total,b_treble/total])
    feat=np.concatenate([mfcc_mean,mfcc_std[:4],extra])
    if len(feat)<30: feat=np.pad(feat,(0,30-len(feat)))
    return feat[:30]

def read_webm_audio(path):
    import soundfile as sf
    y, sr = sf.read(path)
    if y.ndim == 2:
        y = y.mean(axis=1)
    return y.astype(np.float32), sr
    try:
        with wave.open(path,'rb') as wf:
            sr  = wf.getframerate()
            raw = wf.readframes(wf.getnframes())
            y   = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() == 2:
                y = y.reshape(-1,2).mean(axis=1)
            return y, sr
    except Exception:
        pass
    try:
        from scipy.io import wavfile
        sr, data = wavfile.read(path)
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.float32 or data.dtype == np.float64:
            data = data.astype(np.float32)
        if data.ndim == 2:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr
    except Exception as e:
        raise ValueError(f"Could not read audio: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if model is None or scaler is None:
        return jsonify({"error":"Model not loaded. Run model_training.py first."}),500
    if "audio" not in request.files:
        return jsonify({"error":"No audio file."}),400

    audio_file = request.files["audio"]
    suffix = ".wav" if (audio_file.filename or "").endswith(".wav") else ".webm"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        audio_file.save(tmp.name); tmp_path = tmp.name

    try:
        y, sr = read_webm_audio(tmp_path)

        if len(y) == 0:
            return jsonify({"error":"Empty audio received."}), 400

        feat   = extract_features(y, sr)
        feat_s = scaler.transform([feat])
        proba  = model.predict_proba(feat_s)[0]

        # Use asymmetric threshold — lower bar for emergency detection
        emergency_prob = float(proba[1])  # index 1 = emergency class
        label = 1 if emergency_prob >= EMERGENCY_THRESHOLD else 0
        result = "Emergency" if label == 1 else "Normal"
        conf   = round(emergency_prob * 100, 1) if label == 1 else round((1 - emergency_prob) * 100, 1)

        return jsonify({
            "result":  result,
            "confidence": conf,
            "label":   label,
            "emergency_prob": round(emergency_prob * 100, 1)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route("/add_contact", methods=["POST"])
def add_contact():
    data  = request.get_json()
    name  = data.get("name","").strip()
    phone = data.get("phone","").strip()
    if not name or not phone:
        return jsonify({"error":"Name and phone required."}),400
    contacts = []
    if os.path.exists(CONTACTS_PATH):
        with open(CONTACTS_PATH) as f: contacts = json.load(f)
    contacts.append({"name":name,"phone":phone})
    with open(CONTACTS_PATH,"w") as f: json.dump(contacts,f,indent=2)
    return jsonify({"message":f"Contact {name} added.","contacts":contacts})

@app.route("/get_contacts", methods=["GET"])
def get_contacts():
    if os.path.exists(CONTACTS_PATH):
        with open(CONTACTS_PATH) as f: return jsonify(json.load(f))
    return jsonify([])

if __name__ == "__main__":
    load_model()
    app.run(debug=True)