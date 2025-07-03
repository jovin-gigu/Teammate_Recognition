import sys
import os
sys.path.append(os.path.dirname(__file__))
import pickle
import os
from face_recog.recognize_face import recognize_from_camera, recognize_from_image
from voice_recog.recognize_voice import recognize_from_microphone

VOICE_STRICT_THRESHOLD = 0.65
VOICE_SOFT_THRESHOLD = 0.60


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
face_sim = 0
voice_sim = 0

# FACE RECOGNITION
if mode == "face" or mode == "both":
    print("Face input type? (image/camera): ", end="", flush=True)
    face_input = input().strip().lower()

    if face_input == "image":
        image_path = input("Enter image path: ").strip()
        if os.path.exists(image_path):
            face_result, face_sim = recognize_from_image(face_data, image_path)
        else:
            print("[ERROR] Image path does not exist.")
    elif face_input == "camera":
        face_result, face_sim = recognize_from_camera(face_data)
    else:
        print("[ERROR] Invalid face input option.")

# VOICE RECOGNITION
if mode == "voice" or mode == "both":
    voice_result, voice_sim = recognize_from_microphone(voice_data)

# === FINAL OUTPUT ===
print("\n=== FINAL RESULT ===")

if mode == "face":
    if face_result:
        print(f"🎥 Face recognized as: {face_result} (Similarity: {face_sim:.2f}%)")
    else:
        print("❌ No face recognized.")

elif mode == "voice":
    if voice_result:
        print(f"🎙️ Voice recognized as: {voice_result} (Similarity: {voice_sim:.2f}%)")
    else:
        print("❌ No voice recognized.")

elif mode == "both":
    if not face_result and not voice_result:
        print("❌ Neither face nor voice could be recognized.")
    elif not face_result:
        print(f"⚠️ Voice matched: {voice_result} ({voice_sim:.2f}%)\n❌ No face recognized.")
    elif not voice_result:
        if voice_sim >= VOICE_SOFT_THRESHOLD and face_result:
            print(f"✅ Face matched: {face_result} ({face_sim:.2f}%)")
            print(f"💬 Voice almost matched: Similarity {voice_sim:.2f}% — accepted based on face 🥰")
            voice_result = face_result  # treat as a pass
        else:
            print(f"⚠️ Face matched: {face_result} ({face_sim:.2f}%)\n❌ No voice recognized.")

    elif face_result != voice_result:
        # 💫 If face is confident and voice is close but name mismatch, still give soft pass
        if face_result and voice_sim >= VOICE_SOFT_THRESHOLD and voice_result:
            print(f"💡 Possible identity: Face = {face_result}, Voice = {voice_result}")
            print(f"🤔 Voice might've been off — but your face is speaking loud and clear 🥰")
            print(f"   - Face Similarity: {face_sim:.2f}%\n   - Voice Similarity: {voice_sim:.2f}%")
        else:
            print(f"❌ Mismatch Detected!")
            print(f"   - Face: {face_result or 'None'} ({face_sim:.2f}%)")
            print(f"   - Voice: {voice_result or 'None'} ({voice_sim:.2f}%)")


    else:
        print(f"✅ Identity VERIFIED: {face_result}")
        print(f"   - Face Similarity: {face_sim:.2f}%\n   - Voice Similarity: {voice_sim:.2f}%")