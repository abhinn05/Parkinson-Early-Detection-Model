# 🧠 Parkinson's Early Detection Model

An AI-powered, multimodal early detection system for Parkinson's Disease using voice analysis, eye-blink rate from video, and patient age. This project leverages computer vision, signal processing, and machine learning to assist healthcare professionals in early diagnosis of Parkinson’s.

![Parkinson Detection Demo](https://github.com/abhinn05/Parkinson-Early-Detection-Model/assets/demo.gif)

---

## 🚀 Features

- 🎙️ **Voice Analysis:** Uses jitter and shimmer from audio recordings.
- 👁️ **Blink Rate Estimation:** Detects and counts eye blinks from webcam input.
- 🎂 **Age Prediction:** Uses age as a demographic indicator.
- 🧩 **Multimodal Fusion:** Combines all three modalities using weighted probabilities.
- 💻 **Web Dashboard:** Upload media, get instant results, and view reports.

---

## 🧬 Multimodal Architecture

```
           +-----------+       +------------+       +-------------+
Audio ---> | Voice Model|----> |            |       |             |
           +-----------+       |            |       |             |
                               |  Fusion &  |-----> |   Final     |
Video ---> | Blink Model|----> | Prediction |       | Prediction  |
           +-----------+       |            |       |             |
                               +------------+       +-------------+
Age -----> | Age Model |----> 
           +-----------+
```

---

## 📦 Tech Stack

| Layer        | Tech Used                                     |
|--------------|-----------------------------------------------|
| Frontend     | HTML, CSS, JavaScript                         |
| Backend      | Python (Flask), CORS                          |
| Machine Learning | Scikit-learn, OpenCV, dlib, librosa, joblib |
| Deployment   | Localhost / Cloud-Ready                       |

---



## 🧪 How to Run Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/abhinn05/Parkinson-Early-Detection-Model.git
   cd Parkinson-Early-Detection-Model
   ```

2. **Create a Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Start Backend**
   ```bash
   python app.py
   ```

5. **Open Frontend**
   Open `templates/index.html` in your browser and start testing!

---

## 🧠 Model Weights

The repository includes pre-trained models stored in the `/models` directory. These models were trained on extracted features from curated datasets of Parkinson’s and healthy patients.

- `audio_model.pkl`: Trained on jitter/shimmer features
- `blink_model.pkl`: Trained on blink rate per minute
- `age_model.pkl`: Logistic regression based on age

---

## 📈 Example Input & Output

**Input:**
- 10-second webcam video
- User's age

**Output:**
```json
{
  "Name":"Alex",
  "Fused Probability": 0.7002,
  "Prediction":"Parkinson's Detected",
  "Blink Rate": "23 Blinks/min",
  "Audio Probability": 0.6350,
  "Age & Blink Probability": 0.4472
}
```

---

## 🛡️ Disclaimer

This tool is for **research and educational purposes only**. It is **not approved for clinical diagnosis** and should not replace professional medical advice.

---

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-name`
3. Commit your changes.
4. Push to your fork.
5. Open a Pull Request.


## ⭐ If you like this project...

...consider giving it a ⭐ on GitHub and sharing it with friends in AI and healthcare!
