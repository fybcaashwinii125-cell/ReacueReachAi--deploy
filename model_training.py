import os
import numpy as np
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import csv
from scipy.io import wavfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("static", exist_ok=True)

# ── ESC-50 CONFIG ─────────────────────────────────────────────────────────────
# ESC-50 folder structure:
#   ESC-50-master/
#     audio/       ← all .wav files
#     meta/esc50.csv
#
# Emergency categories we care about (ESC-50 category names):
EMERGENCY_CATEGORIES = {
    "siren",
    "glass_breaking",
    "gunshot_or_gunfire",
    "screaming",
    "car_horn",
    "engine",        # car crash / vehicle
    "crackling_fire",
    "fireworks",
}

# All other ESC-50 categories → Normal
# You can adjust ESC50_PATH if your folder is named differently
ESC50_PATH = "C:/Users/Hp/Downloads/ESC-50-master"
AUDIO_DIR  = os.path.join(ESC50_PATH, "audio")
META_CSV   = os.path.join(ESC50_PATH, "meta", "esc50.csv")

SR_TARGET  = 22050
DURATION   = 5   # ESC-50 clips are 5 seconds

# ── FEATURE EXTRACTION (identical to app.py — do not change) ─────────────────

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


def load_wav(filepath):
    """Load wav, resample to SR_TARGET if needed, return float32 mono."""
    sr, data = wavfile.read(filepath)

    # Convert to float32
    if data.dtype == np.int16:
        data = data.astype(np.float32) / 32768.0
    elif data.dtype == np.int32:
        data = data.astype(np.float32) / 2147483648.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        data = data.astype(np.float32)
    else:
        data = data.astype(np.float32)

    # Stereo → mono
    if data.ndim == 2:
        data = data.mean(axis=1)

    # Resample to SR_TARGET if needed (simple linear interp)
    if sr != SR_TARGET:
        orig_len = len(data)
        new_len  = int(orig_len * SR_TARGET / sr)
        data     = np.interp(
            np.linspace(0, orig_len - 1, new_len),
            np.arange(orig_len),
            data
        ).astype(np.float32)

    return data.astype(np.float32), SR_TARGET


# ── READ ESC-50 META CSV ──────────────────────────────────────────────────────

print(f"Reading ESC-50 metadata from: {META_CSV}")
if not os.path.exists(META_CSV):
    print(f"ERROR: Could not find {META_CSV}")
    print("Make sure ESC-50-master folder is in the same directory as this script.")
    exit(1)

file_labels = {}   # filename → (category, label 0/1)
category_counts = {}

with open(META_CSV, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        fname    = row["filename"]
        category = row["category"].strip().lower()
        # Normalize common variations
        category = category.replace(" ", "_").replace("-", "_")
        is_emergency = int(category in EMERGENCY_CATEGORIES)
        file_labels[fname] = (category, is_emergency)
        category_counts[category] = category_counts.get(category, 0) + 1

print(f"\nFound {len(file_labels)} files in metadata.")
print(f"Emergency categories matched: {[c for c in EMERGENCY_CATEGORIES if c in category_counts]}")

# ── EXTRACT FEATURES ──────────────────────────────────────────────────────────

print(f"\nExtracting features from audio files...")
features, labels = [], []
skipped = 0
emergency_count = 0
normal_count = 0

audio_files = os.listdir(AUDIO_DIR) if os.path.exists(AUDIO_DIR) else []
print(f"Audio files found: {len(audio_files)}")

for fname, (category, label) in file_labels.items():
    fpath = os.path.join(AUDIO_DIR, fname)
    if not os.path.exists(fpath):
        skipped += 1
        continue
    try:
        y, sr = load_wav(fpath)
        # Pad or trim to DURATION seconds
        target_len = SR_TARGET * DURATION
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)))
        else:
            y = y[:target_len]
        feat = extract_features(y, sr)
        features.append(feat)
        labels.append(label)
        if label == 1: emergency_count += 1
        else: normal_count += 1
    except Exception as e:
        skipped += 1
        print(f"  Skipped {fname}: {e}")

print(f"\nLoaded: {len(features)} samples")
print(f"  Emergency: {emergency_count}")
print(f"  Normal:    {normal_count}")
print(f"  Skipped:   {skipped}")

if len(features) == 0:
    print("ERROR: No features extracted. Check your ESC-50 path.")
    exit(1)

# ── HANDLE CLASS IMBALANCE ────────────────────────────────────────────────────
# ESC-50 has ~10 emergency categories out of 50 → imbalanced
# We use class_weight to compensate

X     = np.array(features)
y_arr = np.array(labels)

ratio = normal_count / max(emergency_count, 1)
print(f"\nClass ratio (normal/emergency): {ratio:.1f}x")
print("Using class_weight to handle imbalance...")

# ── SPLIT & SCALE ─────────────────────────────────────────────────────────────

X_train,X_test,y_train,y_test = train_test_split(
    X, y_arr, test_size=0.2, random_state=42, stratify=y_arr)
scaler    = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s  = scaler.transform(X_test)

# ── TRAIN ─────────────────────────────────────────────────────────────────────

print("\nTraining Logistic Regression...")
lr = LogisticRegression(
    max_iter=1000, random_state=42,
    class_weight="balanced"   # auto-handles imbalance
)
lr.fit(X_train_s, y_train)
lr_acc = accuracy_score(y_test, lr.predict(X_test_s))
print(f"  Accuracy: {lr_acc*100:.2f}%")

print("Training Random Forest...")
rf_base = RandomForestClassifier(
    n_estimators=300,
    max_depth=25,
    random_state=42,
    class_weight="balanced",
    min_samples_split=4,
    n_jobs=-1
)
rf = CalibratedClassifierCV(rf_base, cv=3, method="sigmoid")
rf.fit(X_train_s, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_s))
print(f"  Accuracy: {rf_acc*100:.2f}%")

best_model = rf if rf_acc >= lr_acc else lr
best_name  = "Random Forest" if rf_acc >= lr_acc else "Logistic Regression"
print(f"\nBest model: {best_name} ({max(lr_acc, rf_acc)*100:.2f}%)")
print(classification_report(y_test, best_model.predict(X_test_s),
                             target_names=["Normal", "Emergency"]))

# ── SAVE THRESHOLD ────────────────────────────────────────────────────────────
EMERGENCY_THRESHOLD = 0.35
with open("threshold.pkl", "wb") as f:
    pickle.dump(EMERGENCY_THRESHOLD, f)
print(f"Emergency threshold: {EMERGENCY_THRESHOLD}")

# ── CHART ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(
    ["Logistic Regression", "Random Forest"],
    [lr_acc*100, rf_acc*100],
    color=["#378ADD", "#1D9E75"], width=0.5, zorder=3
)
ax.set_ylim(0, 110)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Model Accuracy — ESC-50 Dataset", fontweight="bold")
ax.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)
for bar, acc in zip(bars, [lr_acc*100, rf_acc*100]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"{acc:.1f}%", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("static/accuracy_chart.png", dpi=120)
plt.close()
print("Chart saved → static/accuracy_chart.png")

# ── SAVE MODEL ────────────────────────────────────────────────────────────────

with open("model.pkl",  "wb") as f: pickle.dump(best_model, f)
with open("scaler.pkl", "wb") as f: pickle.dump(scaler, f)
print("Saved: model.pkl + scaler.pkl + threshold.pkl")
print("\nDone! Run: python app.py")