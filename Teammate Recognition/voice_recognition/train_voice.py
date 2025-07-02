from resemblyzer import VoiceEncoder, preprocess_wav
import os
import pickle
from pathlib import Path

DATASET_PATH = "voice_dataset"
MODEL_PATH = "../models/voice_embeddings.pkl"
encoder = VoiceEncoder()
embeddings = {}
for person in os.listdir(DATASET_PATH):
    files = [str(p) for p in Path(DATASET_PATH, person).rglob("*.wav")]
    waves = [preprocess_wav(f) for f in files]
    embed = encoder.embed_utterance(waves[0])  # Use first file or average multiple
    embeddings[person] = embed

with open(MODEL_PATH, "wb") as f:
    pickle.dump(embeddings, f)

print("✅ Voice embeddings saved.")
