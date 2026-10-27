# Web Application Deployment Instructions

This folder contains the deployable web-based **Multimodal AI Attention & Fatigue Monitor**.

## Prerequisites
Ensure Python 3.10+ and the required packages are installed:
```bash
pip install -r ../../requirements.txt
```

## Running the Web Dashboard
From this directory, run:
```bash
python -m streamlit run app.py
```
Or from the project root:
```bash
python run_app.py
```

## Key Files in This Directory
- **`app.py`**: The main Streamlit web application.
- **`vision_module.py`**: Face mesh landmark detection and EAR/MAR extraction.
- **`scoring_engine.py`**: Sensor fusion logic and neural classifier inference.
- **`behavioral_module.py`**: Window switching and application telemetry.
- **`report_generator.py`**: PDF and CSV report creation engine.
- **`face_landmarker.task`**: MediaPipe FaceLandmarker task asset.
- **`fatigue_model.pkl`**: Pre-trained MLP fatigue classification model.
- **`reports/`**: Directory where session PDF/CSV reports are saved.
