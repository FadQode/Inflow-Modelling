import os
import shutil
import pandas as pd
import torchaudio
from glob import glob

# ======================= KONFIGURASI =======================
INDO_DIR = "indo_wave"
TESS_DIR = "TESS/TESS Toronto emotional speech set data"
OUTPUT_DIR = "processed_audio"
CSV_OUTPUT = "dataset.csv"

# Label mapping: semua emosi disatukan ke label global
LABEL_MAP = {
    "neutral": "neutral",
    "happy": "happy",
    "surprised": "surprised",
    "disgust": "disgust",
    "sad": "sad",
    "angry": "angry",
    "fear": "fear"
}

# IndoWave: kode emosi ke label
INDO_LABEL_MAP = {
    '01': 'neutral',
    '02': 'happy',
    '03': 'surprised',
    '04': 'disgust',
    '05': 'sad'
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
data = []

def normalize_audio(src_path, dst_path, target_sr=16000):
    wav, sr = torchaudio.load(src_path)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    torchaudio.save(dst_path, wav, target_sr)

# ======================= PROSES INDO WAVE =======================
print("Processing Indo Wave Sentiment...")
for wav_path in glob(os.path.join(INDO_DIR, "IndoWaveSentiment", "**", "*.wav"), recursive=True):
    fname = os.path.basename(wav_path)
    parts = fname.split("-")
    if len(parts) < 4:
        continue

    emotion_code = parts[1]  # bagian ke-2 (01-05)
    label = INDO_LABEL_MAP.get(emotion_code)
    mapped_label = LABEL_MAP.get(label)
    if mapped_label is None:
        continue

    filename = f"indo_{mapped_label}_{fname}"
    out_path = os.path.join(OUTPUT_DIR, filename)
    normalize_audio(wav_path, out_path)
    data.append({"path": out_path, "emotion": mapped_label})

# ======================= PROSES TESS =======================
print("Processing TESS...")
for emotion_folder in os.listdir(TESS_DIR):
    emotion_name = emotion_folder.split("_")[-1].lower()  # OAF_angry → angry
    mapped_label = LABEL_MAP.get(emotion_name)
    if mapped_label is None:
        continue

    folder_path = os.path.join(TESS_DIR, emotion_folder)
    for wav_path in glob(os.path.join(folder_path, "*.wav")):
        fname = os.path.basename(wav_path)
        filename = f"tess_{mapped_label}_{fname}"
        out_path = os.path.join(OUTPUT_DIR, filename)
        normalize_audio(wav_path, out_path)
        data.append({"path": out_path, "emotion": mapped_label})

# ======================= SIMPAN DATASET =======================
df = pd.DataFrame(data)
if df.empty:
    print("❌ Tidak ada data yang berhasil diproses. Cek struktur folder dan label mapping.")
else:
    df.to_csv(CSV_OUTPUT, index=False)
    print(f"✅ Dataset berhasil dibuat: {CSV_OUTPUT}")
    print(f"Total file: {len(df)} | Label distribusi:\n{df['emotion'].value_counts()}")
