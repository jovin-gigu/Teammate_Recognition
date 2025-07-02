from resemblyzer import VoiceEncoder, preprocess_wav
import sounddevice as sd
import numpy as np
import tempfile
from pathlib import Path
import scipy.io.wavfile as wav

def record_audio(duration=4, sample_rate=16000):
    print("🎤 Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1)
    sd.wait()
    path = Path(tempfile.gettempdir()) / "temp_voice.wav"
    wav.write(path, sample_rate, audio)
    return path

def recognize_from_microphone(voice_data, threshold=0.65):
    encoder = VoiceEncoder()
    input_path = record_audio()
    wav_data = preprocess_wav(input_path)
    input_embedding = encoder.embed_utterance(wav_data)

    best_match = None
    highest_similarity = -1

    for name, stored_embedding in voice_data.items():
        sim = np.dot(input_embedding, stored_embedding) / (
            np.linalg.norm(input_embedding) * np.linalg.norm(stored_embedding)
        )
        print(f"🔎 {name}: {sim:.3f}")
        if sim > highest_similarity:
            highest_similarity = sim
            best_match = name

    if highest_similarity >= threshold:
        print(f"✅ Voice match: {best_match} (confidence: {highest_similarity:.2f})")
        return best_match
    else:
        print(f"⚠️ No strong match (best: {best_match}, score: {highest_similarity:.2f})")
        return None
