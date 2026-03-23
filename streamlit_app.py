import streamlit as st
import pandas as pd
import os
import subprocess
from datetime import datetime
#from streamlit_autorefresh import st_autorefresh
import cv2
import face_recognition
import pickle
import numpy as np
import time
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# ------------------ CONFIG ------------------
st.set_page_config(
    page_title="Attendance System",
    layout="wide",
    page_icon="🎓"
)

# Auto refresh every 5 sec
#st_autorefresh(interval=5000, key="refresh")

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
    st.warning("⚠️ Camera feature works only in local system")

    st.info("💻 Please run this project locally to use face recognition")

    st.markdown("""
    ### 🔧 How to use locally:
    1. Open terminal  
    2. Run: `py -3.12 -m streamlit run streamlit_app.py`  
    3. Click Start Camera  
    """)

    col1, col2 = st.columns(2)

    # START BUTTON
    if col1.button("🚀 Start Camera"):
        st.session_state.camera_on = True

    # STOP BUTTON
    if col2.button("🛑 Stop Camera"):
        st.session_state.camera_on = False

    frame_placeholder = st.empty()

    #import cv2
    #import face_recognition
    import numpy as np
    import pickle
    import time

    if st.session_state.camera_on:

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        encoding_path = os.path.join(BASE_DIR, "encodings", "encodings.pkl")

        with open(encoding_path, "rb") as f:
            data = pickle.load(f)

        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        frame_placeholder = st.empty()
        marked_names = set()

        st.success("📸 Camera Running...")

        while st.session_state.camera_on:

            ret, frame = cam.read()
            if not ret:
                st.error("Camera error")
                break

            frame = cv2.resize(frame, (640, 480))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            faces = face_recognition.face_locations(rgb)
            encodings = face_recognition.face_encodings(rgb, faces)

            for encoding, (top, right, bottom, left) in zip(encodings, faces):

                name = "Unknown"
                color = (0, 0, 255)  # Red for unknown
                confidence_text = ""

                if len(data["encodings"]) > 0:
                    distances = face_recognition.face_distance(data["encodings"], encoding)
                    best_match = np.argmin(distances)

                    if distances[best_match] < 0.5:
                        name = data["names"][best_match]
                        confidence = (1 - distances[best_match]) * 100
                        confidence_text = f"{confidence:.1f}%"
                        color = (0, 255, 0)  # Green for known

                        # Mark attendance only once
                        if name not in marked_names:
                            from backend.attendance import mark_attendance
                            mark_attendance(name)
                            marked_names.add(name)

                            # ✅ AUTO STOP AFTER MARKING
                            st.success(f"✅ Attendance marked for {name}")
                            time.sleep(2)
                            st.session_state.camera_on = False
                            break

                # 🎨 Draw clean UI box
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                # Background for text
                cv2.rectangle(frame, (left, top - 30), (right, top), color, -1)

                label = f"{name} {confidence_text}"

                cv2.putText(frame, label, (left + 5, top - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            frame_placeholder.image(frame, channels="BGR")

            if not st.session_state.camera_on:
                break

            time.sleep(0.03)

        cam.release()
        cv2.destroyAllWindows()
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
