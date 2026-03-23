from flask import Flask, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/attendance')   # 👈 THIS IS YOUR URL
def attendance():
    try:
        df = pd.read_excel("../excel/attendance.xlsx")
        return jsonify(df.to_dict(orient="records"))
    except:
        return jsonify([])

if __name__ == "__main__":
    app.run(debug=True)