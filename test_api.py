import pytest
from fastapi.testclient import TestClient
from api import app
import config


@pytest.fixture(scope="module")
def client():
    # The 'with' block forces the @app.on_event("startup") to run
    with TestClient(app) as c:
        yield c

def test_health_check(client):
    """
    Verifies the server is running and returns the correct mode.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    
    expected_mode = "Diagnostic" if config.DIAGNOSTIC_MODE else "Real-Time"
    assert response.json()["mode"] == expected_mode

def test_predict_success(client):
    """
    Sends a valid payload and checks for a diagnosis.
    """
    # Create a fake signal (280 zeros)
    fake_signal = [0.0] * 280
    
    payload = {
        "signal": fake_signal,
        "pre_rr": 0.8,
        "post_rr": 0.85
    }
    
    response = client.post("/predict", json=payload)
    
    # Debugging: Print error if it fails
    if response.status_code != 200:
        print(f"DEBUG ERROR: {response.json()}")

    # 1. Check HTTP Status
    assert response.status_code == 200
    
    # 2. Check Response Structure
    data = response.json()
    assert "class" in data
    assert "confidence" in data
    
    # 3. Sanity check
    assert isinstance(data["class"], str)

def test_predict_validation_error(client):
    """
    Sends an incomplete payload (missing signal).
    """
    payload = {
        "pre_rr": 0.8
    }
    
    response = client.post("/predict", json=payload)
    
    # Should return 422 Unprocessable Entity
    assert response.status_code == 422

def test_diagnostic_mode_logic(client):
    """
    If Diagnostic Mode is ON, ensuring post_rr is required.
    """
    if config.DIAGNOSTIC_MODE:
        fake_signal = [0.0] * 280
        # Payload missing 'post_rr'
        payload = {
            "signal": fake_signal,
            "pre_rr": 0.8,
            # "post_rr": 0.85  <-- MISSING
        }
        
        response = client.post("/predict", json=payload)
        
        # and return 400 (Bad Request), not 500.
        assert response.status_code == 400
        assert "Diagnostic Mode requires 'post_rr'" in response.json()["detail"]