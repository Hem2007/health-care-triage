import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import os
import io

from backend.main import app
from src.features import make_features

client = TestClient(app)

def test_feature_function_single_row():
    df = pd.DataFrame([{"subject_id": "TEST_01", "temperature": 37.5}])
    df_features = make_features(df)
    assert 'temperature_is_missing' in df_features.columns
    assert 'subject_id' not in df_features.columns

def test_predict_empty_body():
    response = client.post("/api/predict", json={})
    assert response.status_code == 200
    data = response.json()
    assert "probabilities" in data
    assert "predicted_label" in data

def test_predict_invalid_type():
    response = client.post("/api/predict", json={"temperature": "hot"})
    assert response.status_code == 422

def test_predict_batch_success():
    csv_content = "subject_id,temperature\nSUBJ_1,37.0\nSUBJ_2,38.0"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/api/predict_batch", files=files)
    assert response.status_code == 200
    data = response.json()
    assert "preview" in data
    assert "csv_content" in data
    csv_str = data["csv_content"]
    assert "Predicted Triage" in csv_str
    assert "RED Probability" in csv_str

def test_predict_batch_missing_column():
    csv_content = "temperature\n37.0"
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/api/predict_batch", files=files)
    assert response.status_code == 400
    assert "must contain 'subject_id'" in response.json()["detail"]

def test_parity_with_submission():
    if not os.path.exists("outputs/TEAMID_MM26ML03.csv"):
        pytest.skip("Submission file not found")
        
    sub_df = pd.read_csv("outputs/TEAMID_MM26ML03.csv")
    test_df = pd.read_csv("data/test_pred_ml03.csv")
    
    assert len(sub_df) == len(test_df)
    
    # Process batch via API
    csv_content = test_df.to_csv(index=False)
    files = {"file": ("test.csv", csv_content, "text/csv")}
    response = client.post("/api/predict_batch", files=files)
    
    assert response.status_code == 200
    api_sub = pd.read_csv(io.StringIO(response.json()["csv_content"]))
    
    # Check exact match for labels
    assert list(sub_df["Predicted Triage"]) == list(api_sub["Predicted Triage"])
    
    # Check probabilities within 1e-3
    prob_cols = ['RED Probability', 'YELLOW Probability', 'GREEN Probability', 'BLACK Probability']
    for col in prob_cols:
        diff = np.abs(sub_df[col] - api_sub[col])
        assert diff.max() < 1e-3
