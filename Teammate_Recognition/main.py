import sys
import os
sys.path.append(os.path.dirname(__file__))
import pickle
import os
from face_recog.recognize_face import recognize_from_camera, recognize_from_image
from voice_recog.recognize_voice import recognize_from_microphone

# Load models
with open("models/face_encodings.pkl", "rb") as f:
    face_data = pickle.load(f)

with open("models/voice_embeddings.pkl", "rb") as f:
    voice_data = pickle.load(f)

print("==== TEAMMATE RECOGNITION SYSTEM ====")
print("Select mode (face / voice / both): ", end="", flush=True)
mode = input().strip().lower()


face_result = None
voice_result = None

# FACE RECOGNITION
if mode == "face" or mode == "both":
    print("Face input type? (image/camera): ", end="", flush=True)
    face_input = input().strip().lower()

    if face_input == "image":
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            face_result = recognize_from_image(face_data, image_path)
        else:
            print("[ERROR] Image path does not exist.")
    elif face_input == "camera":
        face_result = recognize_from_camera(face_data)
    else:
        print("[ERROR] Invalid face input option.")

# VOICE RECOGNITION
if mode == "voice" or mode == "both":
    voice_result = recognize_from_microphone(voice_data)

# === FINAL OUTPUT ===
print("\n=== FINAL RESULT ===")

if mode == "face":
    if face_result:
        print(f"🎥 Face recognized as: {face_result}")
    else:
        print("❌ No face recognized.")

elif mode == "voice":
    if voice_result:
        print(f"🎙️ Voice recognized as: {voice_result}")
    else:
        print("❌ No voice recognized.")

elif mode == "both":
    if not face_result and not voice_result:
        print("❌ Neither face nor voice could be recognized.")
    elif not face_result:
        print(f"⚠️ Voice matched ({voice_result}), but no face could be recognized.")
    elif not voice_result:
        print(f"⚠️ Face matched ({face_result}), but no voice could be recognized.")
    elif face_result != voice_result:
        print(f"❌ Mismatch detected!\nFace: {face_result}\nVoice: {voice_result}")
    else:
        print(f"✅ Identity VERIFIED: {face_result}")
