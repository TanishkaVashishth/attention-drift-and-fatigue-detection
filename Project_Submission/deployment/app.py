import streamlit as st
import cv2
import time
import numpy as np
import threading
import winsound
import os
from vision_module import VisionModule
from behavioral_module import BehavioralModule
from scoring_engine import ScoringEngine
from report_generator import ReportGenerator

# Page Config
st.set_page_config(page_title="Multimodal AI Monitor", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS FOR DYNAMIC UI ---
st.markdown("""
<style>
    /* Premium Glassmorphism Theme */
    .main {
        background: radial-gradient(circle at center, #1e293b 0%, #0f172a 100%) !important;
        background-attachment: fixed !important;
    }
    
    [data-testid="stSidebar"] {
        background: rgba(15, 23, 42, 0.8) !important;
        backdrop-filter: blur(10px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        padding: 15px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(5px);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-5px);
        background: rgba(255, 255, 255, 0.07);
        border-color: #38bdf8;
    }

    /* Danger Pulse Vignette */
    @keyframes alertPulse {
        0% { box-shadow: inset 0 0 50px rgba(239, 68, 68, 0); }
        50% { box-shadow: inset 0 0 150px rgba(239, 68, 68, 0.4); }
        100% { box-shadow: inset 0 0 50px rgba(239, 68, 68, 0); }
    }

    .danger-overlay {
        position: fixed;
        top: 0; left: 0; width: 100vw; height: 100vh;
        pointer-events: none;
        z-index: 9999;
    }
    
    .is-fatigued {
        animation: alertPulse 1.5s infinite;
    }

    /* Vibrant Monitoring Bars (Simple version for Streamlit) */
    .monitor-bar {
        height: 10px;
        border-radius: 5px;
        margin-top: 5px;
        transition: width 0.5s ease;
        box-shadow: 0 0 10px rgba(255,255,255,0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize modules in session state
if 'vision' not in st.session_state:
    st.session_state.vision = VisionModule()
    st.session_state.behavior = BehavioralModule()
    st.session_state.scorer = ScoringEngine(st.session_state.vision, st.session_state.behavior)
    st.session_state.reporter = ReportGenerator()
    st.session_state.is_active = False
    st.session_state.last_beep_time = 0
    st.session_state.attn_history = []

def trigger_beep():
    winsound.Beep(1000, 500)

# Sidebar
st.sidebar.title("System Controls")
if st.sidebar.button("CALIBRATE SYSTEM"):
    # This is tricky in Streamlit, we'll just set a flag
    st.session_state.calibrating = True
    st.sidebar.info("Please look at the camera for 5 seconds...")

st.sidebar.markdown("---")
st.sidebar.subheader("Live Telemetry")
ear_val = st.sidebar.empty()
mar_val = st.sidebar.empty()
tab_val = st.sidebar.empty()

st.sidebar.markdown("---")
session_btn_placeholder = st.sidebar.empty()

# Main Dashboard
st.title("🎛️ Multimodal AI Attention Monitor")

col1, col2 = st.columns([2, 1])

with col1:
    video_placeholder = st.empty()

with col2:
    st.subheader("State Analysis")
    status_placeholder = st.empty()
    
    m1, m2 = st.columns(2)
    attn_metric = m1.metric("Attention Score", "0%")
    fatg_metric = m2.metric("Fatigue Index", "0%")
    
    # Placeholder for the danger overlay
    vignette_placeholder = st.empty()
    
    st.markdown("---")
    st.subheader("Attention Timeline")
    chart_placeholder = st.empty()

# Action Logic
if st.session_state.is_active:
    if session_btn_placeholder.button("STOP SESSION", type="primary"):
        st.session_state.is_active = False
        csv_p, txt_p, pdf_p = st.session_state.reporter.generate_report()
        if pdf_p:
            with open(pdf_p, "rb") as f:
                st.sidebar.download_button("📥 DOWNLOAD PDF REPORT", f, file_name=os.path.basename(pdf_p))
else:
    if session_btn_placeholder.button("START SESSION", type="secondary"):
        st.session_state.is_active = True
        st.session_state.reporter = ReportGenerator()
        st.session_state.attn_history = []

# Main Camera Loop (Runs while app is active)
cap = cv2.VideoCapture(0)
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

last_log_time = time.time()

try:
    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        
        # Processing
        landmarks = st.session_state.vision.get_landmarks(frame)
        ear, mar, pose = 0, 0, (1.0, 1.0)
        
        if landmarks:
            ear_l = st.session_state.vision.calculate_ear(landmarks, st.session_state.vision.LEFT_EYE)
            ear_r = st.session_state.vision.calculate_ear(landmarks, st.session_state.vision.RIGHT_EYE)
            ear = (ear_l + ear_r) / 2.0
            mar = st.session_state.vision.calculate_mar(landmarks)
            pose = st.session_state.vision.estimate_head_pose(landmarks, frame_w, frame_h)
            
            # Simple UI overlay (Circles)
            for idx in [33, 133, 362, 263]:
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x*frame_w), int(lm.y*frame_h)), 2, (33, 150, 243), -1)

        st.session_state.behavior.update()
        attn, fatg = st.session_state.scorer.calculate_scores(ear, mar, pose)
        status, color = st.session_state.scorer.get_status()

        # Update Sidebar Telemetry
        ear_val.text(f"EAR (Eye Ratio): {ear:.3f}")
        mar_val.text(f"MAR (Mouth Ratio): {mar:.3f}")
        tab_val.text(f"Windows Switches: {st.session_state.behavior.tab_switch_count}")

        # Update Main UI
        video_placeholder.image(frame, channels="BGR", use_container_width=True)
        
        status_fmt = f'<div style="background-color: rgb({color[2]},{color[1]},{color[0]},0.2); border: 1px solid rgb({color[2]},{color[1]},{color[0]}); padding: 10px; border-radius: 5px; text-align: center; color: rgb({color[2]},{color[1]},{color[0]}); font-weight: bold; font-size: 24px;">{status}</div>'
        status_placeholder.markdown(status_fmt, unsafe_allow_html=True)
        
        attn_metric.metric("Attention Score", f"{int(attn*100)}%", delta=f"{int((attn-0.5)*100)}%")
        fatg_metric.metric("Fatigue Index", f"{int(fatg*100)}%", delta=f"{int((fatg-0.5)*100)}%", delta_color="inverse")

        # Chart update
        if st.session_state.is_active:
            st.session_state.attn_history.append(attn)
            if len(st.session_state.attn_history) > 100:
                st.session_state.attn_history.pop(0)
            chart_placeholder.line_chart(st.session_state.attn_history)
            
            # Log
            if (time.time() - last_log_time) > 1.0:
                st.session_state.reporter.log_data(attn, fatg, status)
                last_log_time = time.time()

        # Beep and Vignette
        if status == "FATIGUED":
            vignette_placeholder.markdown('<div class="danger-overlay is-fatigued"></div>', unsafe_allow_html=True)
            if (time.time() - st.session_state.last_beep_time) > 2.0:
                threading.Thread(target=trigger_beep, daemon=True).start()
                st.session_state.last_beep_time = time.time()
        else:
            vignette_placeholder.empty()

        time.sleep(0.01)
        
except Exception as e:
    st.error(f"Camera Error: {e}")
finally:
    cap.release()
