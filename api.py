# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import config
from inference import ArrhythmiaPredictor
import uvicorn

# 1. Define the Input Schema (Data Contract)
class ECGInput(BaseModel):
    signal: List[float]       # The raw heartbeat values
    pre_rr: float             # Time since last beat (seconds)
    post_rr: Optional[float] = None # Time to next beat (optional if Real-Time mode)

# 2. Initialize the App & Model
app = FastAPI(title="Deep-ECG-Full API", version="1.0")
predictor = None

@app.on_event("startup")
def load_model():
    """Load the model once when the server starts (Optimization)"""
    global predictor
    try:
        # Load the quantized model for speed, or fallback to standard
        model_path = config.MODEL_SAVE_PATH
        if "quantized" not in model_path and "quantized" in config.MODEL_SAVE_PATH: 
             # Logic to prefer quantized if available, but for now let's stick to config
             pass
        
        predictor = ArrhythmiaPredictor(model_path)
        print("✅ API Startup: Model Loaded Successfully")
    except Exception as e:
        print(f"❌ API Startup Failed: {e}")

# 3. Define the Prediction Endpoint
@app.post("/predict")
def predict_heartbeat(data: ECGInput):
    if not predictor:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    # Validation for Diagnostic Mode
    if config.DIAGNOSTIC_MODE and data.post_rr is None:
        raise HTTPException(status_code=400, detail="Diagnostic Mode requires 'post_rr'")

    try:
        # Run Inference
        result = predictor.predict(
            signal_window=data.signal,
            pre_rr=data.pre_rr,
            post_rr=data.post_rr
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 4. Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy", "mode": "Diagnostic" if config.DIAGNOSTIC_MODE else "Real-Time"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)