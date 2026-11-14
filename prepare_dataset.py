import os
import pandas as pd
import shutil

BASE_PATH = r"C:\Users\Hp\Downloads\ESC-50-master\ESC-50-master"
AUDIO_PATH = os.path.join(BASE_PATH, "audio")
CSV_PATH = os.path.join(BASE_PATH, "meta", "esc50.csv")

OUTPUT_PATH = "dataset"

emergency_classes = ["siren", "glass_breaking", "gun_shot"]
normal_classes = ["dog", "rain", "footsteps"]

os.makedirs(os.path.join(OUTPUT_PATH, "emergency"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_PATH, "normal"), exist_ok=True)

df = pd.read_csv(CSV_PATH)

for _, row in df.iterrows():
    file = row["filename"]
    category = row["category"]

    src = os.path.join(AUDIO_PATH, file)

    if category in emergency_classes:
        shutil.copy(src, os.path.join(OUTPUT_PATH, "emergency", file))

    elif category in normal_classes:
        shutil.copy(src, os.path.join(OUTPUT_PATH, "normal", file))

print("✅ Dataset Ready!")