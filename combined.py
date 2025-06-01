import cv2
import dlib
import numpy as np
import pandas as pd
import sounddevice as sd
import soundfile as sf
import threading
import time
import os
import csv
import parselmouth
from parselmouth.praat import call
from scipy.spatial import distance
import librosa
import librosa.feature
import scipy.stats
import joblib

# --- Global Configuration ---
try:
    os.chdir(r"E:\Parkinson_2")
except FileNotFoundError:
    print("[ERROR] Could not change directory. Please ensure the path is correct.")
    print("Attempting to use current directory for file operations.")
    # Optionally, exit if the directory is critical and not found
    # import sys
    # sys.exit(1)


AUDIO_FILENAME = "recorded_audio.wav"
VIDEO_FILENAME = "recorded_video.avi"
DURATION = 30  # seconds
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
CSV_FILE = "test_cases.csv"

# --- Audio Recording ---
def record_audio(filename=AUDIO_FILENAME, duration=DURATION, samplerate=44100):
    """Records audio for a specified duration and saves it to a file."""
    print("[INFO] Starting audio recording...")
    try:
        audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
        sd.wait()  # Wait until recording is finished
        sf.write(filename, audio_data, samplerate)
        print(f"[INFO] Audio recording complete. Saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Audio recording failed: {e}")

# --- Blink Rate Calculation ---
def calculate_ear(eye):
    """Calculates the Eye Aspect Ratio (EAR) for a single eye."""
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def record_video_and_compute_blink_rate(filename=VIDEO_FILENAME, duration=DURATION, result_container=None):
    """
    Records video, detects blinks, calculates blink rate, and saves the video.
    Stores the blink rate in result_container[0] if provided.
    """
    print("[INFO] Starting video recording and blink detection...")
    
    if not os.path.exists(PREDICTOR_PATH):
        print(f"[ERROR] Dlib shape predictor model not found at {PREDICTOR_PATH}")
        if result_container is not None:
            result_container[0] = -1.0 # Indicate error
        return -1.0

    try:
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to load dlib detector or predictor: {e}")
        if result_container is not None:
            result_container[0] = -1.0
        return -1.0

    (lStart, lEnd) = (42, 48)
    (rStart, rEnd) = (36, 42)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open webcam.")
        if result_container is not None:
            result_container[0] = -1.0
        return -1.0
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    try:
        out = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 20.0, (frame_width, frame_height))
    except Exception as e:
        print(f"[ERROR] Could not create VideoWriter (try 'MJPG' instead of 'XVID' if issues persist): {e}")
        cap.release()
        if result_container is not None:
            result_container[0] = -1.0
        return -1.0


    blink_count = 0
    ear_threshold = 0.21
    ear_consec_frames = 2
    consecutive_frames_below_thresh = 0

    start_time = time.time()
    frames_processed = 0

    print("[INFO] Webcam opened. Recording video...")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARNING] Could not read frame from webcam. Ending video recording.")
            break
        
        if out is not None : out.write(frame)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])

            left_eye = shape_np[lStart:lEnd]
            right_eye = shape_np[rStart:rEnd]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0

            if ear < ear_threshold:
                consecutive_frames_below_thresh += 1
            else:
                if consecutive_frames_below_thresh >= ear_consec_frames:
                    blink_count += 1
                consecutive_frames_below_thresh = 0
        
        frames_processed += 1
        if time.time() - start_time >= duration:
            break

    cap.release()
    if out is not None : out.release()
    cv2.destroyAllWindows()

    actual_duration_seconds = time.time() - start_time
    blink_rate = 0.0
    if actual_duration_seconds > 0:
        blink_rate = blink_count / (actual_duration_seconds / 60.0)

    print(f"[INFO] Video recording complete. Processed {frames_processed} frames in {actual_duration_seconds:.2f}s.")
    print(f"[INFO] Total blinks: {blink_count}. Blink Rate: {blink_rate:.2f} blinks/min. Saved to {filename}")
    
    if result_container is not None:
        result_container[0] = blink_rate
    return blink_rate

# --- Audio Feature Extraction (Robust Version) ---
def extract_audio_features(audio_filepath):
    """Extracts audio features from the given audio file with improved robustness."""
    # Define expected column names for consistency, especially if returning NaNs
    feature_columns = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    default_feature_values = [np.nan] * len(feature_columns)

    try:
        sound = parselmouth.Sound(audio_filepath)
        y, sr = librosa.load(audio_filepath, sr=None) # Load with original sample rate

        # Initialize all features to a default (e.g., np.nan or 0.0)
        fo, fhi, flo = np.nan, np.nan, np.nan
        jitter_local, jitter_abs, rap, ppq, ddp = np.nan, np.nan, np.nan, np.nan, np.nan
        shimmer_local, shimmer_db, apq3, apq5, apq, dda = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
        nhr, hnr = np.nan, np.nan
        rpde, dfa, spread1, spread2, d2, ppe = np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        # Parselmouth features
        try:
            pitch = sound.to_pitch()
            fo = call(pitch, "Get mean", 0, 0, "Hertz")
            fhi = call(pitch, "Get maximum", 0, 0, "Hertz", "Parabolic")
            flo = call(pitch, "Get minimum", 0, 0, "Hertz", "Parabolic")

            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500) # Common pitch floor/ceil
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            jitter_abs = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
            rap = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
            ppq = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
            ddp = rap * 3 if isinstance(rap, (float, int)) else np.nan

            shimmer_local = call([sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            shimmer_db = call([sound, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq3 = call([sound, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq5 = call([sound, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            apq = call([sound, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            dda = apq3 * 3 if isinstance(apq3, (float, int)) else np.nan


            harmonicity_obj = sound.to_harmonicity_cc()
            hnr_val = call(harmonicity_obj, "Get mean", 0, 0)
            if isinstance(hnr_val, (float, int)) and hnr_val != 0:
                 hnr = hnr_val
                 nhr = 1 / (10**(hnr / 10) + 1) if hnr != -np.inf else 1.0 # Avoid division by zero with 10**-inf
            elif isinstance(hnr_val, (float, int)) and hnr_val == 0:
                 hnr = 0.0
                 nhr = 0.5 # 1 / (1+1)
            else: # hnr_val might be nan or other non-numeric
                 hnr = np.nan
                 nhr = np.nan

        except Exception as e_parselmouth:
            print(f"[WARNING] Parselmouth feature extraction failed: {e_parselmouth}. Using defaults for these.")

        # Librosa features
        if len(y) > 0:
            try:
                spectral_flatness_values = librosa.feature.spectral_flatness(y=y)
                if spectral_flatness_values.size > 0 and np.any(spectral_flatness_values > 0):
                    positive_spectral_flatness = spectral_flatness_values[spectral_flatness_values > 0]
                    if positive_spectral_flatness.size > 0:
                         rpde = scipy.stats.entropy(positive_spectral_flatness.flatten())
                
                rms_feat = librosa.feature.rms(y=y)
                if rms_feat.size > 0:
                    dfa = np.std(rms_feat[0])

                mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
                if mfccs.shape[1] > 0: 
                    if mfccs.shape[0] > 1 : spread1 = scipy.stats.skew(mfccs[1,:])
                    if mfccs.shape[0] > 2 : spread2 = scipy.stats.kurtosis(mfccs[2,:])
                    if mfccs.shape[0] > 0 : ppe = np.std(mfccs[0,:]) 
                
                d2_val = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                if d2_val.size > 0:
                    d2 = np.mean(d2_val)
            except Exception as e_librosa:
                print(f"[WARNING] Librosa feature extraction failed: {e_librosa}. Using defaults for these.")
        else:
            print("[WARNING] Audio signal 'y' is empty. Using defaults for Librosa features.")
        
        # Replace any remaining non-numeric results from Praat with NaN if necessary
        current_values = [
            fo, fhi, flo, jitter_local, jitter_abs, rap, ppq, ddp,
            shimmer_local, shimmer_db, apq3, apq5, apq, dda,
            nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe
        ]
        sanitized_values = [v if isinstance(v, (float, int)) else np.nan for v in current_values]


        features_df = pd.DataFrame([sanitized_values], columns=feature_columns)
        return features_df

    except Exception as e_load:
        print(f"[ERROR] Failed to load or process audio {audio_filepath}: {e_load}")
        return pd.DataFrame([default_feature_values], columns=feature_columns)


# --- Main Data Collection and Processing ---
def run_data_collection():
    """Runs the data collection, feature extraction, and saves to CSV."""
    print("[INFO] Starting multimodal data collection and analysis...")
    feature_columns = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
        "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]
    patient_name = input("Enter patient name: ").strip()
    if not patient_name: patient_name = "Unknown"
    
    age_str = input("Enter patient age: ").strip()
    try:
        age = int(age_str)
    except ValueError:
        print("[ERROR] Invalid age. Please enter a valid number.")
        return False # Indicate failure

    blink_rate_container = [None] # Using list to pass by reference to thread

    audio_thread = threading.Thread(target=record_audio, args=(AUDIO_FILENAME, DURATION))
    video_thread = threading.Thread(target=record_video_and_compute_blink_rate, 
                                    args=(VIDEO_FILENAME, DURATION, blink_rate_container))

    audio_thread.start()
    video_thread.start()
    audio_thread.join()
    video_thread.join()

    blink_rate_value = blink_rate_container[0]
    audio_features_df = extract_audio_features(AUDIO_FILENAME)

    print("\n--- Combined Extracted Features ---")
    if audio_features_df is not None and not audio_features_df.empty and not audio_features_df.isnull().values.all():
        print("Audio Features:")
        for col_name in audio_features_df.columns:
            value = audio_features_df[col_name].iloc[0]
            print(f"  {col_name}: {value:.4f}" if isinstance(value, (float, np.number)) and not np.isnan(value) else f"  {col_name}: {value}")
    else:
        print("[ERROR] Audio feature extraction failed or returned no valid data.")
        # Clean up and return even if audio fails, as blink might be useful or vice-versa
        # return False # Indicate failure

    if blink_rate_value is not None and blink_rate_value != -1.0:
        print(f"Blink Rate (blinks per minute): {blink_rate_value:.2f}")
    else:
        print("[ERROR] Blink rate calculation failed or was invalid.")
        blink_rate_value = np.nan # Ensure it's NaN for CSV if it failed

    # Prepare CSV row
    row_data = {
        "name": patient_name,
        "Age": age,
        "Blink Rate": round(blink_rate_value, 2) if pd.notna(blink_rate_value) else np.nan
    }
    
    if audio_features_df is not None and not audio_features_df.empty:
        row_data.update(audio_features_df.iloc[0].to_dict())
    else: # Add empty audio features if extraction failed
        feature_columns = [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
            "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
            "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
            "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
        ]
        for col in feature_columns:
            row_data[col] = np.nan


    # Define fieldnames based on a complete row_data structure, ensuring order
    # This is important if the CSV is being created for the first time.
    ordered_fieldnames = ["name", "Age", "Blink Rate"] + [
        col for col in feature_columns if col in row_data
    ]


    # Write to CSV
    file_exists = os.path.isfile(CSV_FILE)
    try:
        with open(CSV_FILE, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=ordered_fieldnames)
            if not file_exists or os.path.getsize(CSV_FILE) == 0: # Check if file is empty too
                writer.writeheader()
            writer.writerow(row_data)
        print(f"[INFO] Appended results to {CSV_FILE}.")
    except IOError as e:
        print(f"[ERROR] Could not write to CSV file {CSV_FILE}: {e}")
        return False


    # Clean up temporary files
    for f_to_delete in [AUDIO_FILENAME, VIDEO_FILENAME]:
        if os.path.exists(f_to_delete):
            try:
                os.remove(f_to_delete)
                print(f"[INFO] Temporary file {f_to_delete} removed.")
            except OSError as e:
                print(f"[ERROR] Could not remove temporary file {f_to_delete}: {e}")
    return True # Indicate success

# --- Prediction Section ---
def run_prediction():
    """Loads data and models, then performs prediction on the last CSV entry."""
    print("\n--- Running Prediction ---")
    try:
        # Load models and tools
        audio_model = joblib.load('parkinson_model.pkl')
        # Ensure this scaler corresponds to how parkinson_model.pkl was trained
        audio_feature_scaler = joblib.load('scaler.pkl') 
        feature_names_pkl = joblib.load('feature_names.pkl') # These are the expected audio feature columns

        age_blink_model = joblib.load('age_blink_combined_model.pkl')
        age_blink_scaler = joblib.load('age_blink_combined_scaler.pkl')
    except FileNotFoundError as e:
        print(f"[ERROR] Model or scaler file not found: {e}. Cannot proceed with prediction.")
        return
    except Exception as e:
        print(f"[ERROR] Could not load models/scalers: {e}")
        return


    try:
        df = pd.read_csv(CSV_FILE)
        if df.empty:
            print("[ERROR] CSV file is empty. No data to predict.")
            return
    except FileNotFoundError:
        print(f"[ERROR] CSV file {CSV_FILE} not found. Run data collection first.")
        return
    except pd.errors.EmptyDataError:
        print(f"[ERROR] CSV file {CSV_FILE} is empty or corrupted.")
        return
    except Exception as e:
        print(f"[ERROR] Could not read CSV file {CSV_FILE}: {e}")
        return

    # Use the last entry for prediction
    last_entry_index = len(df) - 1
    current_entry = df.iloc[[last_entry_index]] # Keep as DataFrame for consistency

    # Audio features prediction
    try:
        # Ensure all expected feature names from pkl are in the CSV
        missing_audio_features = set(feature_names_pkl) - set(current_entry.columns)
        if missing_audio_features:
            print(f"[ERROR] Missing required audio features in CSV: {missing_audio_features}")
            return

        audio_features_from_csv = current_entry[feature_names_pkl]
        # Handle potential NaNs before scaling: either impute or ensure scaler handles them
        # For now, let's assume scaler and model can handle NaNs or they are imputed before this step.
        # A common strategy is to fill NaNs with the mean of the training set for that feature.
        # If your scaler (e.g. StandardScaler) was fit on data with NaNs, it might error.
        # If NaNs are present, you might need to impute them, e.g., audio_features_from_csv.fillna(training_mean_values, inplace=True)
        if audio_features_from_csv.isnull().values.any():
            print(f"[WARNING] NaN values found in audio features for prediction. Results might be unreliable.")
            # Option: Fill NaNs with 0 or mean if appropriate for your model
            # audio_features_from_csv = audio_features_from_csv.fillna(0) 

        audio_scaled = audio_feature_scaler.transform(audio_features_from_csv)
        audio_proba = audio_model.predict_proba(audio_scaled)[0][1]
    except Exception as e:
        print(f"[ERROR] Failed to process audio features for prediction: {e}")
        audio_proba = np.nan # Indicate failure


    # Age & Blink features prediction
    try:
        current_age = current_entry['Age'].iloc[0]
        current_blink_rate = current_entry['Blink Rate'].iloc[0]

        if pd.isna(current_age) or pd.isna(current_blink_rate):
            print("[ERROR] Age or Blink Rate is NaN in the current CSV entry. Cannot predict age/blink probability.")
            age_blink_proba = np.nan
        else:
            age_blink_input_raw = np.array([[current_age, current_blink_rate]])
            age_blink_input_scaled = age_blink_scaler.transform(age_blink_input_raw)
            age_blink_proba = age_blink_model.predict_proba(age_blink_input_scaled)[0][1]
    except Exception as e:
        print(f"[ERROR] Failed to process age/blink features for prediction: {e}")
        age_blink_proba = np.nan


    # Weighted fusion (handle NaN probabilities by excluding them or assigning zero weight)
    # For simplicity, if one proba is NaN, fused_proba will be NaN unless handled.
    # Here we proceed, and fused_proba will be NaN if any component is NaN.
    # A more robust fusion would check for NaNs and adjust weights or exclude.
    
    audio_model_weight = 0.6 
    age_blink_model_weight = 0.4

    # Handle cases where probabilities might be NaN
    valid_probas = {}
    if pd.notna(audio_proba):
        valid_probas['audio'] = audio_proba
    if pd.notna(age_blink_proba):
        valid_probas['age_blink'] = age_blink_proba
    
    # Simplified fusion: if both are valid, use weights.
    # If only one is valid, use that one. If none, then NaN.
    # This is a basic way to handle missing probabilities; more sophisticated logic might be needed.
    if 'audio' in valid_probas and 'age_blink' in valid_probas:
        fused_proba = (audio_model_weight * valid_probas['audio'] +
                       age_blink_model_weight * valid_probas['age_blink'])
    elif 'audio' in valid_probas:
        print("[INFO] Only audio probability available for fusion.")
        fused_proba = valid_probas['audio']
    elif 'age_blink' in valid_probas:
        print("[INFO] Only age/blink probability available for fusion.")
        fused_proba = valid_probas['age_blink']
    else:
        print("[ERROR] No valid probabilities available for fusion.")
        fused_proba = np.nan


    fused_prediction = 1 if pd.notna(fused_proba) and fused_proba >= 0.5 else (0 if pd.notna(fused_proba) else -1) # -1 for undetermined

    print(f"\n🔍 Prediction for Test Case: {current_entry['name'].iloc[0]} (Row {last_entry_index + 1})")
    print(f" - Audio Probability: {audio_proba:.4f}" if pd.notna(audio_proba) else " - Audio Probability: Error/NaN")
    print(f" - Age & Blink Combined Probability: {age_blink_proba:.4f}" if pd.notna(age_blink_proba) else " - Age & Blink Combined Probability: Error/NaN")
    print(f" - Fused Probability (Audio_wt={audio_model_weight}, AgeBlink_wt={age_blink_model_weight}): {fused_proba:.4f}" if pd.notna(fused_proba) else " - Fused Probability: Error/NaN")
    
    if fused_prediction == 1:
        print(" - Final Prediction: Parkinson's Detected (1)")
    elif fused_prediction == 0:
        print(" - Final Prediction: No Parkinson's (0)")
    else:
        print(" - Final Prediction: Undetermined due to errors.")


if __name__ == '__main__':
    if run_data_collection(): # Only run prediction if data collection was successful
        run_prediction()
    else:
        print("\n[INFO] Data collection was not fully successful. Prediction step skipped.")