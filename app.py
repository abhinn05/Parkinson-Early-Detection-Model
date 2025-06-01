from flask import Flask, Response, request, jsonify, render_template
from flask_cors import CORS
import cv2
import joblib
import pandas as pd
import numpy as np
import threading
import time
import os

from combined import record_audio, record_video_and_compute_blink_rate, extract_audio_features

app = Flask(__name__)
CORS(app)

# Constants
AUDIO_FILENAME = "recorded_audio.wav"
VIDEO_FILENAME = "recorded_video.avi"
DURATION = 10
CSV_FILE = "test_cases.csv"
camera = cv2.VideoCapture(0)

# Load models
audio_model = joblib.load("parkinson_model.pkl")
audio_scaler = joblib.load("scaler.pkl")
feature_names = joblib.load("feature_names.pkl")

age_blink_model = joblib.load("age_blink_combined_model.pkl")
age_blink_scaler = joblib.load("age_blink_combined_scaler.pkl")


# ---------------- Live video stream ----------------
def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# ---------------- Predict Route ----------------
@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['age'])
    except:
        return jsonify({'error': 'Invalid age input'}), 400

    # Blink rate container (shared between thread and main)
    blink_rate_container = [None]

    audio_thread = threading.Thread(target=record_audio, args=(AUDIO_FILENAME, DURATION))
    video_thread = threading.Thread(target=record_video_and_compute_blink_rate, args=(VIDEO_FILENAME, DURATION, blink_rate_container))

    audio_thread.start()
    video_thread.start()
    audio_thread.join()
    video_thread.join()

    blink_rate = blink_rate_container[0]
    audio_df = extract_audio_features(AUDIO_FILENAME)

    if audio_df.isnull().values.any():
        audio_df.fillna(0, inplace=True)

    # Make predictions
    try:
        audio_input = audio_scaler.transform(audio_df[feature_names])
        audio_proba = audio_model.predict_proba(audio_input)[0][1]
    except Exception as e:
        print(f"[ERROR] Audio prediction failed: {e}")
        audio_proba = np.nan

    try:
        blink_rate = blink_rate if blink_rate is not None else np.nan
        age_blink_input = age_blink_scaler.transform(np.array([[age, blink_rate]]))
        age_blink_proba = age_blink_model.predict_proba(age_blink_input)[0][1]
    except Exception as e:
        print(f"[ERROR] Age/blink prediction failed: {e}")
        age_blink_proba = np.nan

    # Fusion logic
    audio_weight, ab_weight = 0.6, 0.4
    if not np.isnan(audio_proba) and not np.isnan(age_blink_proba):
        fused_proba = audio_weight * audio_proba + ab_weight * age_blink_proba
    elif not np.isnan(audio_proba):
        fused_proba = audio_proba
    elif not np.isnan(age_blink_proba):
        fused_proba = age_blink_proba
    else:
        fused_proba = np.nan

    prediction = (
        "Parkinson's Detected" if fused_proba >= 0.5 else
        "No Parkinson's" if not np.isnan(fused_proba) else
        "Unable to Predict"
    )

        # Optional: get name if it's part of the form
    name = request.form.get('name', 'Unknown')

    return jsonify({
        'name': name,
        'blink_rate': round(blink_rate, 2) if blink_rate is not None else None,
        'audio_proba': round(audio_proba, 4) if not np.isnan(audio_proba) else None,
        'age_proba': round(age_blink_proba, 4) if not np.isnan(age_blink_proba) else None,
        'fused_proba': round(fused_proba, 4) if not np.isnan(fused_proba) else None,
        'prediction': 1 if fused_proba >= 0.5 else 0 if not np.isnan(fused_proba) else -1
    })




# ---------------- Frontend for Live View ----------------
@app.route('/')
def index():
    return render_template('index.html')
        

if __name__ == '__main__':
    app.run(debug=True)
