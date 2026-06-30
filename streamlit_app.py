import streamlit as st
import pandas as pd
import os
import subprocess
from datetime import datetime
from streamlit_autorefresh import st_autorefresh
import cv2
import dlib
import face_recognition_models
import pickle
import numpy as np
import time

# ------------------ FACE RECOGNITION (direct dlib, no source build) ------------------
# Using dlib + face_recognition_models directly avoids the `face_recognition`
# wrapper, which would force dlib to compile from source and fail on Streamlit
# Cloud. The models match what encode.py used, so encodings.pkl stays compatible.
@st.cache_resource
def _load_face_models():
    detector = dlib.get_frontal_face_detector()
    pose_predictor = dlib.shape_predictor(
        face_recognition_models.pose_predictor_five_point_model_location()
    )
    encoder = dlib.face_recognition_model_v1(
        face_recognition_models.face_recognition_model_location()
    )
    return detector, pose_predictor, encoder


def face_locations(rgb):
    detector, _, _ = _load_face_models()
    return [
        (max(d.top(), 0), d.right(), d.bottom(), max(d.left(), 0))
        for d in detector(rgb, 1)
    ]


def face_encodings(rgb, locations):
    _, pose_predictor, encoder = _load_face_models()
    encs = []
    for (top, right, bottom, left) in locations:
        rect = dlib.rectangle(left, top, right, bottom)
        shape = pose_predictor(rgb, rect)
        encs.append(np.array(encoder.compute_face_descriptor(rgb, shape, 1)))
    return encs


def face_distance(known_encodings, encoding):
    if len(known_encodings) == 0:
        return np.empty(0)
    return np.linalg.norm(np.asarray(known_encodings) - encoding, axis=1)


if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Attendance System",
    layout="wide",
    page_icon="🎓"
)

# Auto refresh every 5 sec
st_autorefresh(interval=5000, key="refresh")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "excel", "attendance.xlsx")

# ------------------ FUNCTIONS ------------------
def load_data():
    if os.path.exists(FILE_PATH):
        return pd.read_excel(FILE_PATH)
    return pd.DataFrame(columns=["Name", "Date", "Time"])

def get_today_count(df):
    today = datetime.now().strftime("%Y-%m-%d")
    return len(df[df["Date"] == today])

# ------------------ SIDEBAR ------------------
st.sidebar.title("📌 Menu")

option = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Start Attendance", "Analytics"]
)

# ------------------ LOAD DATA ------------------
df = load_data()

# ------------------ DASHBOARD ------------------
if option == "Dashboard":

    st.title("🎓 Automated Attendance System")

    col1, col2, col3 = st.columns(3)

    col1.metric("👨‍🎓 Total Records", len(df))
    col2.metric("📅 Today's Attendance", get_today_count(df))
    col3.metric("🕒 Last Updated", datetime.now().strftime("%H:%M:%S"))

    st.divider()

    st.subheader("📊 Attendance Records")

    # 🔍 Search filter
    search = st.text_input("🔍 Search by Name")

    if search:
        filtered_df = df[df["Name"].str.contains(search, case=False)]
    else:
        filtered_df = df
    st.button("🔄 Refresh Data")
    df = load_data()
    st.dataframe(filtered_df, width="stretch")

    # 📥 Download button
    if os.path.exists(FILE_PATH):
        with open(FILE_PATH, "rb") as f:
            st.download_button(
                "📥 Download Excel",
                f,
                file_name="attendance.xlsx"
            )

# ------------------ START ATTENDANCE ------------------
elif option == "Start Attendance":

    st.title("🎥 Smart Attendance System")
    st.info("📸 Allow camera access in your browser, then take a photo to mark attendance.")

    encoding_path = os.path.join(BASE_DIR, "encodings", "encodings.pkl")

    if not os.path.exists(encoding_path):
        st.error("❌ Encodings file not found. Please generate encodings first (run encode.py).")
    else:
        with open(encoding_path, "rb") as f:
            data = pickle.load(f)

        # Browser-based camera works both locally and on Streamlit Cloud
        photo = st.camera_input("Take a photo to mark attendance")

        if photo is not None:
            from backend.attendance import mark_attendance

            # Decode the captured photo into an OpenCV image
            file_bytes = np.asarray(bytearray(photo.getvalue()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = face_locations(rgb)
            encodings = face_encodings(rgb, faces)

            if len(faces) == 0:
                st.warning("😕 No face detected. Please try again.")

            for encoding, (top, right, bottom, left) in zip(encodings, faces):

                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                confidence_text = ""

                if len(data["encodings"]) > 0:
                    distances = face_distance(data["encodings"], encoding)
                    best_match = np.argmin(distances)

                    if distances[best_match] < 0.5:
                        name = data["names"][best_match]
                        confidence = (1 - distances[best_match]) * 100
                        confidence_text = f"{confidence:.1f}%"
                        color = (0, 255, 0)  # Green for known

                        if mark_attendance(name):
                            st.success(f"✅ Attendance marked for {name} ({confidence_text})")
                        else:
                            st.info(f"ℹ️ {name} already marked today ({confidence_text})")
                    else:
                        st.warning("⚠️ Face detected but not recognized.")

                # 🎨 Draw clean UI box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.rectangle(frame, (left, top - 30), (right, top), color, -1)

                label = f"{name} {confidence_text}"
                cv2.putText(frame, label, (left + 5, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            st.image(frame, channels="BGR", caption="Result")

# ------------------ ANALYTICS ------------------
elif option == "Analytics":

    st.title("📊 Attendance Analytics")

    if df.empty:
        st.warning("No data available")
    else:
        df["Date"] = pd.to_datetime(df["Date"])

        # 📅 Attendance per day
        daily = df.groupby(df["Date"].dt.date).count()["Name"]

        st.subheader("📅 Daily Attendance")
        st.line_chart(daily)

        # 👥 Attendance per student
        student_count = df["Name"].value_counts()

        st.subheader("👨‍🎓 Student Attendance Count")
        st.bar_chart(student_count)