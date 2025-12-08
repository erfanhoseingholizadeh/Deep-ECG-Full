import streamlit as st
import requests
import pandas as pd
import numpy as np
import json

# --- CONFIGURATION ---
API_URL = "http://localhost:8000/predict"
st.set_page_config(page_title="Deep-ECG Diagnostic", page_icon="🫀", layout="wide")

# --- HEADER ---
st.title("🫀 Deep-ECG: Clinical Arrhythmia Detection")
st.markdown("""
This system uses a **Hybrid CNN-Transformer** model to detect cardiac irregularities.
**Supported Classes:** Normal (N), Supraventricular (S), Ventricular (V), Fusion (F), Unknown (Q).
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Patient Data")
    pre_rr = st.number_input("Pre-RR Interval (sec)", value=0.80, step=0.01)
    post_rr = st.number_input("Post-RR Interval (sec)", value=0.80, step=0.01)
    
    st.markdown("---")
    st.info("Upload a CSV file containing a single column of ECG signal values (280 samples).")
    uploaded_file = st.file_uploader("Upload ECG CSV", type=["csv", "txt"])

# --- MAIN LOGIC ---
if uploaded_file is not None:
    try:
        # Load Data
        df = pd.read_csv(uploaded_file, header=None)
        signal = df.iloc[:, 0].values.tolist()
        
        # Validation
        if len(signal) < 187: # MIN length for MIT-BIH padding
            st.error(f"Signal too short! Need approx 280 samples. Got {len(signal)}.")
        else:
            # Trim or Pad to 280 for visualization consistency
            display_signal = signal[:280]
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.subheader("ECG Trace")
                st.line_chart(display_signal)
            
            with col2:
                st.subheader("AI Diagnosis")
                if st.button("Run Analysis", type="primary"):
                    with st.spinner("Analyzing Morphology & Rhythm..."):
                        # Payload
                        payload = {
                            "signal": signal, # Send full signal, API handles cropping
                            "pre_rr": pre_rr,
                            "post_rr": post_rr
                        }
                        
                        try:
                            response = requests.post(API_URL, json=payload)
                            
                            if response.status_code == 200:
                                result = response.json()
                                pred_class = result['class']
                                conf = result['confidence']
                                
                                # Dynamic Color Logic
                                color = "green" if pred_class == "N" else "red"
                                
                                st.markdown(f"### Result: :{color}[{pred_class}]")
                                st.metric("Confidence", f"{conf*100:.2f}%")
                                
                                # Interpretation
                                mapping = {'N': 'Normal', 'S': 'Supraventricular', 'V': 'Ventricular', 'F': 'Fusion', 'Q': 'Unknown'}
                                st.caption(f"Interpretation: {mapping.get(pred_class, 'Unknown')}")
                                
                            else:
                                st.error(f"API Error: {response.text}")
                                
                        except Exception as e:
                            st.error(f"Connection Failed. Is api.py running? \n\nError: {e}")

    except Exception as e:
        st.error(f"Error reading file: {e}")

else:
    # Default State
    st.info("👈 Upload a CSV file to begin analysis.")
    
    # Generate Synthetic Example Button
    if st.button("Generate Synthetic Normal Beat"):
        # Create a synthetic "Normal" looking wave (just a sine wave with noise for demo)
        t = np.linspace(0, 1, 280)
        synthetic_signal = (np.sin(2 * np.pi * 5 * t) * np.exp(-3*t)).tolist()
        
        st.line_chart(synthetic_signal)
        
        if st.button("Analyze Synthetic Beat"):
             # Call API with synthetic data
             pass # (Logic would be same as above)