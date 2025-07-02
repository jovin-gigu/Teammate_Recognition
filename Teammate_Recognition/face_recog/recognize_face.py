import face_recognition
import cv2
import numpy as np
import os

def recognize_from_image(face_data, image_path):
    """
    Recognize a face from a static image.
    Returns: (matched_name, similarity_percent)
    """
    image = face_recognition.load_image_file(image_path)
    encodings = face_recognition.face_encodings(image)

    if not encodings:
        print("[ERROR] No face in image.")
        return None, None

    return match_face(encodings[0], face_data)

def recognize_from_camera(face_data):
    """
    Recognize a face from a live webcam feed with bounding boxes and labels.
    Returns: (matched_name, similarity_percent)
    """
    cap = cv2.VideoCapture(0)
    print("[INFO] Press Q to quit after detection.")

    found_match = None
    found_sim = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for (top, right, bottom, left), encoding in zip(face_locations, encodings):
            name, sim = match_face(encoding, face_data)

            # ✅ Custom color logic
            if sim > 70:
                color = (0, 255, 0)       # Green
            elif sim > 40:
                color = (0, 255, 255)     # Yellow
            else:
                color = (0, 0, 255)       # Red

            # Draw box and label
            label = f"{name or 'Unknown'} ({sim:.1f}%)"
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if not found_match and name:
                found_match = name
                found_sim = sim

        cv2.imshow("Camera", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return found_match, found_sim


def match_face(encoding, face_data, threshold=0.6):
    """
    Compare input encoding to known faces.
    Returns: (best_match_name, similarity_percent)
    """
    best_match = None
    best_dist = float("inf")

    for name, known_encoding in face_data.items():
        dist = np.linalg.norm(encoding - known_encoding)
        if dist < best_dist:
            best_match = name
            best_dist = dist

    similarity = max(0.0, 1.0 - best_dist) * 100  # approx. similarity %

    if best_dist < threshold:
        print(f"✅ Face matched: {best_match} (distance: {best_dist:.2f}, similarity: {similarity:.2f}%)")
        return best_match, similarity
    else:
        print("[INFO] No confident face match.")
        return None, similarity
