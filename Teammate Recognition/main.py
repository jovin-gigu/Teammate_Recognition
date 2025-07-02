# main.py

import pickle
import numpy as np
#from face_recognition.recognize_face import recognize_from_camera
from voice_recognition.recognize_voice import recognize_from_microphone
'''
# Load face and voice encodings
with open("models/face_encodings.pkl", "rb") as f:
    face_data = pickle.load(f)
'''
with open("models/voice_embeddings.pkl", "rb") as f:
    voice_data = pickle.load(f)
'''
# Recognize face
face_result = recognize_from_camera(face_data)
'''
# Recognize voice
voice_result = recognize_from_microphone(voice_data)

# Final decision
if face_result == voice_result:
    print(f"✅ Identity Verified: {face_result}")
else:
    print(f"⚠️ Mismatch Detected!\nFace: {face_result}\nVoice: {voice_result}")
