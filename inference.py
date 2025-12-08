# inference.py
import torch
import numpy as np
from scipy.signal import butter, filtfilt
import os
import config
from model import HybridECGNet

class ArrhythmiaPredictor:
    def __init__(self, model_path):
        self.device = config.DEVICE
        self.model = HybridECGNet().to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found at {model_path}. Run main.py first!")
            
        # Load weights (map_location ensures CPU compatibility)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.classes = {0: 'N', 1: 'S', 2: 'V', 3: 'F', 4: 'Q'}
        print(f"✅ Model loaded. Diagnostic Mode: {config.DIAGNOSTIC_MODE}")

    def _preprocess(self, wave, pre_rr, post_rr=None):
        # 1. Bandpass Filter
        nyquist = 0.5 * config.FS
        b, a = butter(2, [0.5/nyquist, 50.0/nyquist], btype='band')
        wave = filtfilt(b, a, wave)
        
        # 2. Z-Score Norm
        wave = (wave - np.mean(wave)) / (np.std(wave) + 1e-6)
        
        # 3. Tensor Setup
        wave_t = torch.tensor(wave, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 4. Rhythm Logic
        if config.DIAGNOSTIC_MODE:
            if post_rr is None: raise ValueError("Diagnostic Mode requires post_rr")
            rhythm_t = torch.tensor([[pre_rr, post_rr]], dtype=torch.float32)
        else:
            rhythm_t = torch.tensor([[pre_rr]], dtype=torch.float32)
            
        return wave_t.to(self.device), rhythm_t.to(self.device)

    def predict(self, signal_window, pre_rr, post_rr=None):
        wave_t, rhythm_t = self._preprocess(signal_window, pre_rr, post_rr)
        
        with torch.no_grad():
            logits = self.model(wave_t, rhythm_t)
            probs = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(probs).item()
            
        return {
            "class": self.classes[pred_idx],
            "confidence": probs[0][pred_idx].item(),
            "probabilities": probs.cpu().numpy().tolist()
        }

if __name__ == "__main__":
    # Test Run
    predictor = ArrhythmiaPredictor(config.MODEL_SAVE_PATH)
    
    # Fake patient data
    fake_signal = np.random.randn(280) 
    result = predictor.predict(fake_signal, pre_rr=0.8, post_rr=0.75)
    
    print(f"\n🩺 Diagnosis: {result['class']} ({result['confidence']:.2%})")