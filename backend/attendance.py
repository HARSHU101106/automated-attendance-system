import pandas as pd
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILE_PATH = os.path.join(BASE_DIR, "..", "excel", "attendance.xlsx")

def mark_attendance(name):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    # Create file if not exists
    if not os.path.exists(FILE_PATH):
        df = pd.DataFrame(columns=["Name", "Date", "Time"])
        df.to_excel(FILE_PATH, index=False)

    df = pd.read_excel(FILE_PATH)

    # ✅ CHECK: already marked TODAY?
    df["Date"] = df["Date"].astype(str)
    already_marked = ((df["Name"] == name) & (df["Date"] == date)).any()

    if already_marked:
        print(f"⚠️ {name} already marked today")
        return False
    else:
        new_row = {"Name": name, "Date": date, "Time": time}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        df.to_excel(FILE_PATH, index=False)

        print(f"✅ Attendance marked: {name}")
        return True