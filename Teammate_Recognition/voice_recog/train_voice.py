from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import pickle
import numpy as np

# 🎯 Path to dataset and output
DATASET_DIR = Path("voice_recog/voice_dataset")
MODEL_PATH = "models/voice_embeddings.pkl"

# 🎤 Load encoder
encoder = VoiceEncoder()
voice_embeddings = {}

# 🔍 Loop through all speaker folders
for speaker_dir in DATASET_DIR.iterdir():
    if not speaker_dir.is_dir():
        continue

    print(f"\n🎙️ Processing speaker: {speaker_dir.name}")
    embeddings = []
    file_count = 0

    # 🌀 Support ANY number of WAV files
    for audio_file in speaker_dir.glob("*"):
        if audio_file.suffix.lower() not in [".wav", ".mp3"]:
            continue

        try:
            wav = preprocess_wav(audio_file)
            emb = encoder.embed_utterance(wav)
            embeddings.append(emb)
            file_count += 1
            print(f"  ✅ Loaded: {audio_file.name}")
        except Exception as e:
            print(f"  ⚠️ Skipped {audio_file.name}: {e}")

    if embeddings:
        avg_embedding = np.mean(embeddings, axis=0)
        voice_embeddings[speaker_dir.name] = avg_embedding
        print(f"🎉 Finished {file_count} files for {speaker_dir.name}")
    else:
        print(f"❌ No valid audio found for: {speaker_dir.name}")

# 💾 Save to pickle
with open(MODEL_PATH, "wb") as f:
    pickle.dump(voice_embeddings, f)

print(f"\n✅ Voice embeddings saved to: {MODEL_PATH}")
