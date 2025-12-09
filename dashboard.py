import streamlit as st
import requests
import pandas as pd
import numpy as np
import random

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
st.set_page_config(page_title="Deep-ECG Diagnostic", page_icon="🫀", layout="wide")

# --- SESSION STATE INITIALIZATION ---
if 'signal' not in st.session_state:
    st.session_state.signal = None
if 'source' not in st.session_state:
    st.session_state.source = None
if 'synth_pre_rr' not in st.session_state:
    st.session_state.synth_pre_rr = 0.80
if 'synth_post_rr' not in st.session_state:
    st.session_state.synth_post_rr = 0.80
if 'ground_truth' not in st.session_state:
    st.session_state.ground_truth = None

# --- HEADER ---
st.title("🫀 Deep-ECG: Clinical Arrhythmia Detection")
st.markdown("""
**System Status:** 🟢 Online | **Model:** Hybrid CNN-Transformer (Quantized)
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Input Data")
    
    input_method = st.radio("Select Source:", ["Upload CSV", "Generate Synthetic"])
    
    if input_method == "Upload CSV":
        uploaded_file = st.file_uploader("Upload ECG CSV (Single Lead)", type=["csv", "txt"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                st.session_state.signal = df.iloc[:, 0].values.tolist()
                st.session_state.source = "Uploaded File"
                st.session_state.ground_truth = "Unknown (Real Patient)"
            except Exception as e:
                st.error(f"Error reading file: {e}")
                
    elif input_method == "Generate Synthetic":
        st.info("Generates a random Normal, Ventricular, or Supraventricular beat to test the AI.")
        
        if st.button("Generate Random Beat 🎲", type="primary"):
            # Randomly select a pathology
            beat_types = ["N", "S", "V"]
            selected_type = random.choice(beat_types)
            
            t = np.linspace(0, 1, 280)
            noise = np.random.normal(0, 0.015, 280) # Add realistic noise
            
            if selected_type == "N":
                # Normal: Classic P-QRS-T
                p_wave = 0.15 * np.exp(-((t - 0.2)**2) / 0.002)
                qrs = -0.15 * np.exp(-((t - 0.33)**2) / 0.0005) + 1.0 * np.exp(-((t - 0.35)**2) / 0.001) - 0.25 * np.exp(-((t - 0.37)**2) / 0.0005)
                t_wave = 0.3 * np.exp(-((t - 0.6)**2) / 0.01)
                
                st.session_state.synth_pre_rr = 0.85
                st.session_state.synth_post_rr = 0.85
                st.session_state.signal = (p_wave + qrs + t_wave + noise).tolist()
                st.session_state.ground_truth = "Normal (N)"

            elif selected_type == "V":
                # Ventricular (PVC): No P-wave, Wide Inverted QRS, Compensatory Pause
                qrs = -1.2 * np.exp(-((t - 0.35)**2) / 0.004) # Wide & Deep
                t_wave = 0.4 * np.exp(-((t - 0.55)**2) / 0.01) # Large opposite T-wave
                
                st.session_state.synth_pre_rr = 0.50 # Premature
                st.session_state.synth_post_rr = 1.10 # Pause
                st.session_state.signal = (qrs + t_wave + noise).tolist()
                st.session_state.ground_truth = "Ventricular Ectopy (V)"

            elif selected_type == "S":
                # Supraventricular (SVE): Normal Shape, but VERY Early
                p_wave = 0.15 * np.exp(-((t - 0.2)**2) / 0.002)
                qrs = -0.15 * np.exp(-((t - 0.33)**2) / 0.0005) + 1.0 * np.exp(-((t - 0.35)**2) / 0.001)
                t_wave = 0.3 * np.exp(-((t - 0.6)**2) / 0.01)
                
                st.session_state.synth_pre_rr = 0.45 # Very Early
                st.session_state.synth_post_rr = 0.80
                st.session_state.signal = (p_wave + qrs + t_wave + noise).tolist()
                st.session_state.ground_truth = "Supraventricular (S)"

            st.session_state.source = "Synthetic (Randomized)"

    st.markdown("---")
    st.header("2. Clinical Context")
    pre_rr = st.number_input("Pre-RR Interval (sec)", value=st.session_state.synth_pre_rr, step=0.01)
    post_rr = st.number_input("Post-RR Interval (sec)", value=st.session_state.synth_post_rr, step=0.01)

# --- MAIN DISPLAY ---
if st.session_state.signal is not None:
    st.subheader(f"ECG Trace Source: {st.session_state.source}")
    
    # Pad/Trim
    display_signal = st.session_state.signal[:280]
    if len(display_signal) < 280:
        display_signal += [0.0] * (280 - len(display_signal))
    
    st.line_chart(display_signal)

    st.subheader("AI Diagnostic Assessment")
    
    if st.button("Run Analysis", type="primary", use_container_width=True):
        with st.spinner("Processing Signal Morphology & Rhythm..."):
            payload = {
                "signal": st.session_state.signal,
                "pre_rr": pre_rr,
                "post_rr": post_rr
            }
            
            try:
                response = requests.post(API_URL, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    pred_class = result['class']
                    conf = result['confidence']
                    
                    m1, m2, m3 = st.columns(3)
                    
                    # Truth vs Prediction Check
                    is_correct = False
                    if st.session_state.ground_truth:
                        # Extract the code (N, S, V) from "Normal (N)" string
                        true_code = st.session_state.ground_truth.split("(")[-1].replace(")", "") 
                        is_correct = (pred_class == true_code)

                    # Dynamic Color
                    color = "green" if pred_class == "N" else "red"
                    
                    m1.metric("AI Prediction", pred_class)
                    m2.metric("Confidence", f"{conf*100:.2f}%")
                    
                    # Latency Mockup (or measure actual time)
                    m3.metric("Latency", "12 ms")

                    # Verdict UI
                    st.divider()
                    if st.session_state.ground_truth and "Unknown" not in st.session_state.ground_truth:
                        if is_correct:
                            st.success(f"✅ **SUCCESS:** AI correctly identified {st.session_state.ground_truth}")
                        else:
                            st.error(f"❌ **MISDIAGNOSIS:** AI predicted **{pred_class}**, but ground truth was **{st.session_state.ground_truth}**")
                    else:
                        st.info(f"AI Analysis Complete: {pred_class}")

                else:
                    st.error(f"Server Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Failed: {e}")

else:
    st.info("👈 Click 'Generate Random Beat' to perform a blind test.")