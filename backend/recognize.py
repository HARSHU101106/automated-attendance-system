import cv2
import face_recognition
import pickle
import os
import numpy as np
from attendance import mark_attendance

print("🚀 Starting Attendance System...")

# Load encodings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoding_path = os.path.join(BASE_DIR, "..", "encodings", "encodings.pkl")

with open(encoding_path, "rb") as f:
    data = pickle.load(f)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cam.isOpened():
    print("❌ Camera error")
    exit()

# To prevent repeated marking
marked_names = set()

while True:
    ret, frame = cam.read()

    if not ret or frame is None:
        print("⚠️ Camera error, retrying...")
        continue

    # Resize for performance
    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for encoding, face in zip(encodings, faces):

        name = "Unknown"

        if len(data["encodings"]) > 0:

            # ✅ Distance-based matching
            face_distances = face_recognition.face_distance(
                data["encodings"], encoding
            )

            best_match_index = np.argmin(face_distances)

            # ✅ Confidence threshold
            if face_distances[best_match_index] < 0.5:
                name = data["names"][best_match_index]

                print(f"Detected: {name}, Distance: {face_distances[best_match_index]:.2f}")

                # ✅ Avoid duplicate marking
                if name not in marked_names:
                    mark_attendance(name)
                    marked_names.add(name)

        # Draw rectangle
        top, right, bottom, left = face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        # Show name
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(1)

    # Press ESC to exit
    if key == 27:
        break

cam.release()
cv2.destroyAllWindows()