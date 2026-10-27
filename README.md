# Multimodal Attention Drift & Fatigue Detection System

A real-time, computer-vision and machine-learning powered monitoring system that tracks attention levels and fatigue states through a modern, responsive web dashboard.

---

## 🌟 Key Highlights

- **Web Dashboard Interface**: Interactive Streamlit-powered dashboard featuring live video streaming, real-time telemetry gauges, dynamic attention timeline charts, and responsive visual alerts.
- **Computer Vision Pipeline**: High-precision 468-point facial landmark detection via MediaPipe FaceLandmarker for continuous calculation of:
  - **EAR (Eye Aspect Ratio)**: Micro-blink and prolonged eye closure detection.
  - **MAR (Mouth Aspect Ratio)**: Yawning and fatigue marker tracking.
  - **Head Pose Estimation**: Pitch and yaw tracking for focus alignment.
- **Machine Learning Classification**: Multi-Layer Perceptron (MLP) neural classifier trained on the **NHTU Driver Drowsiness Dataset (~66,000 images)** to predict drowsiness probability.
- **Multimodal Behavioral Fusion**: Combines visual biometric indicators with operating system window-switching patterns to calculate composite Attention and Fatigue indices.
- **Automated Session Reporting**: Automatically generates downloadable PDF and CSV performance reports upon session completion.

---

## 🚀 Quick Start

### 1. Prerequisites
- Python 3.10 or higher
- A working webcam

### 2. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/TanishkaVashishth/<REPO_NAME>.git
cd <REPO_NAME>
pip install -r requirements.txt
```

### 3. Launching the Web Dashboard
You can run the web dashboard using any of the following methods:

**Option A — Direct Python Launcher:**
```bash
python run_app.py
```

**Option B — Streamlit Command:**
```bash
python -m streamlit run Project_Submission/deployment/app.py
```

**Option C — Windows 1-Click Runner:**
Double-click `run.bat` in the project root.

The dashboard will open automatically in your browser at `http://localhost:8501`.

---

## 📁 Repository Structure

```
.
├── Project_Submission/
│   ├── deployment/             # Ready-to-deploy web application
│   │   ├── app.py              # Main Streamlit web dashboard application
│   │   ├── vision_module.py    # MediaPipe FaceLandmarker & metric computation
│   │   ├── scoring_engine.py   # Sensor fusion & ML inference engine
│   │   ├── behavioral_module.py# OS window/tab switching tracker
│   │   ├── report_generator.py # PDF/CSV report generation engine
│   │   ├── face_landmarker.task# MediaPipe FaceLandmarker model binary
│   │   ├── fatigue_model.pkl   # Trained MLP classifier weights
│   │   └── reports/            # Output folder for generated session reports
│   ├── code/                   # Core research and development source code
│   │   ├── main.py             # Streamlit application entrypoint
│   │   ├── train_fatigue_model.py # Model training script (NHTU dataset)
│   │   ├── validate_thresholds.py # Threshold tuning and validation tool
│   │   └── requirements.txt    # Component dependency list
│   └── dataset/                # Dataset documentation & links
│       └── dataset_links.txt   # Download source for NHTU DDD dataset
├── run_app.py                  # Convenient Python launcher for the web app
├── run.bat                     # Windows batch runner
├── requirements.txt            # Root Python dependencies
├── .gitignore                  # Git configuration ignoring large datasets & caches
└── README.md                   # Project documentation
```

---

## 🧠 Methodology & Architecture

1. **Facial Feature Extraction (`vision_module.py`)**  
   Frames are captured from the webcam and converted to RGB tensors. MediaPipe extracts facial landmark points around the eyes and lips to calculate EAR and MAR using Euclidean distances.
2. **Behavioral Telemetry (`behavioral_module.py`)**  
   Tracks foreground window changes using OS window APIs (`PyGetWindow` & `psutil`), penalizing rapid tab switching and off-task multitasking.
3. **Sensor Fusion Engine (`scoring_engine.py`)**  
   Fuses facial geometry signals, MLP drowsiness classification probability, and window switching frequency into normalized continuous scores:
   - **Attention Score**: `0.0` (Distracted) to `1.0` (Attentive)
   - **Fatigue Index**: `0.0` (Alert) to `1.0` (Exhausted / Drowsy)
4. **Interactive Alerts & Reporting**  
   Triggers a red-pulse visual vignette and audio warning when fatigue exceeds threshold levels, and compiles the time-series session telemetry into a formatted PDF summary upon clicking **Stop Session**.

---

## 📊 Dataset Note

The trained model utilizes the **NHTU Driver Drowsiness Dataset (NHTU-DDD)**. Due to GitHub's file size limits, the ~2.88 GB raw image dataset is excluded from version control via `.gitignore`. The dataset download instructions and citation links can be found in `Project_Submission/dataset/dataset_links.txt`. The pre-trained model weights (`fatigue_model.pkl`) and landmark tasks are included in the repository so the application works immediately out-of-the-box.

---

## 📜 License & Acknowledgements

Developed as an academic AI & Computer Vision project. Special thanks to the creators of the NHTU Driver Drowsiness Dataset and Google MediaPipe.
