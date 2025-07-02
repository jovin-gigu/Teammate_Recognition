import face_recognition
import cv2
import numpy as np
import os

def recognize_from_image(face_data, image_path):
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print("[ERROR] No face in image.")
        return None

    return match_face(encodings[0], face_data)

def recognize_from_camera(face_data):
    cap = cv2.VideoCapture(0)
    print("[INFO] Press Q to quit after detection.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert frame to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect face locations
        face_locations = face_recognition.face_locations(rgb_frame)

        # Extract encodings based on those locations
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if encodings:
            result = match_face(encodings[0], face_data)
            cap.release()
            cv2.destroyAllWindows()
            return result

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return None

def match_face(encoding, face_data, threshold=0.6):
    for name, known_encoding in face_data.items():
        dist = np.linalg.norm(encoding - known_encoding)
        if dist < threshold:
            print(f"✅ Face matched: {name} (distance: {dist:.2f})")
            return name

    print("[INFO] No face match.")
    return None
