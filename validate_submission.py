import sys
import pandas as pd
import numpy as np

def validate_csv(filepath):
    print(f"Validating {filepath}...")
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        print(f"FAIL: Could not read CSV file: {e}")
        sys.exit(1)
        
    expected_cols = [
        "Patient ID", 
        "Predicted Triage", 
        "RED Probability", 
        "YELLOW Probability", 
        "GREEN Probability", 
        "BLACK Probability"
    ]
    
    if list(df.columns) != expected_cols:
        print(f"FAIL: Columns do not match exactly.\\nExpected: {expected_cols}\\nGot: {list(df.columns)}")
        sys.exit(1)
    print("PASS: Exact column names and order")
    
    if len(df) != 112:
        print(f"FAIL: Expected 112 rows, got {len(df)}")
        sys.exit(1)
    print("PASS: Exactly 112 rows")
    
    if df.isna().sum().sum() > 0:
        print("FAIL: Missing values found in the CSV")
        sys.exit(1)
    print("PASS: No missing values")
    
    valid_labels = {"RED", "YELLOW", "GREEN", "BLACK"}
    invalid_labels = set(df["Predicted Triage"]) - valid_labels
    if invalid_labels:
        print(f"FAIL: Invalid labels found in 'Predicted Triage': {invalid_labels}")
        sys.exit(1)
    print("PASS: Valid predicted labels")
    
    prob_cols = ["RED Probability", "YELLOW Probability", "GREEN Probability", "BLACK Probability"]
    for col in prob_cols:
        if (df[col] < 0).any() or (df[col] > 1).any():
            print(f"FAIL: Probabilities in {col} outside [0, 1]")
            sys.exit(1)
    print("PASS: Probabilities in [0, 1]")
    
    row_sums = df[prob_cols].sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-3):
        bad_idx = np.where(~np.isclose(row_sums, 1.0, atol=1e-3))[0]
        print(f"FAIL: Probabilities do not sum to 1 within 1e-3 on rows: {bad_idx.tolist()}")
        sys.exit(1)
    print("PASS: Probabilities sum to 1")
    
    try:
        test_df = pd.read_csv("data/test_pred_ml03.csv")
        if list(df["Patient ID"]) != list(test_df["subject_id"]):
            print("FAIL: 'Patient ID' column does not exactly match 'subject_id' in data/test_pred_ml03.csv")
            sys.exit(1)
        print("PASS: Patient ID matches test file exactly")
    except Exception as e:
        print(f"WARNING: Could not load test file to check Patient IDs: {e}")
        
    print("\\nSUCCESS: All checks passed!")
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_submission.py <path_to_csv>")
        sys.exit(1)
    validate_csv(sys.argv[1])
