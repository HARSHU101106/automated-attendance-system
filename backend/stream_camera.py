import cv2
import face_recognition
import pickle
import numpy as np
import streamlit as st
import os
from attendance import mark_attendance

st.title("🎥 Live Attendance Camera")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
encoding_path = os.path.join(BASE_DIR, "..", "encodings", "encodings.pkl")

with open(encoding_path, "rb") as f:
    data = pickle.load(f)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

frame_placeholder = st.empty()

marked_names = set()

while True:
    ret, frame = cam.read()

    if not ret:
        st.error("Camera error")
        break

    frame = cv2.resize(frame, (640, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    faces = face_recognition.face_locations(rgb)
    encodings = face_recognition.face_encodings(rgb, faces)

    for encoding, face in zip(encodings, faces):
        name = "Unknown"

        if len(data["encodings"]) > 0:
            distances = face_recognition.face_distance(data["encodings"], encoding)
            best_match = np.argmin(distances)

            if distances[best_match] < 0.5:
                name = data["names"][best_match]

                if name not in marked_names:
                    mark_attendance(name)
                    marked_names.add(name)

        top, right, bottom, left = face

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    frame_placeholder.image(frame, channels="BGR")

cam.release()