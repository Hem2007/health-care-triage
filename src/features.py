import pandas as pd
import numpy as np

LABELS = ["RED", "YELLOW", "GREEN", "BLACK"]

# RED, YELLOW, GREEN, BLACK
COST_MATRIX = np.array([
    [0, 5, 10, 5],
    [2, 0, 3, 2],
    [1, 1, 0, 1],
    [2, 2, 2, 0]
])

def make_features(df):
    df = df.copy()

    # Drop subject_id and race (design choice)
    for col in ['subject_id', 'race']:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Add _is_missing flags
    vitals = ['temperature', 'heartrate', 'resprate', 'o2sat', 'sbp', 'dbp', 'pain_score']
    for v in vitals:
        if v in df.columns:
            df[f'{v}_is_missing'] = df[v].isna().astype(float)
        else:
            df[f'{v}_is_missing'] = 1.0
            df[v] = np.nan

    # START-style flags
    df['start_resp_high'] = (df.get('resprate', 0) > 30).astype(float)
    df['start_resp_zero'] = (df.get('resprate', 1) == 0).astype(float)
    df['start_hr_zero'] = (df.get('heartrate', 1) == 0).astype(float)
    df['start_sbp_low'] = (df.get('sbp', 100) < 90).astype(float)
    df['start_motor_low'] = (df.get('gcs_motor', 6) < 6).astype(float)

    # Chief complaint flags
    if 'chiefcomplaint' in df.columns:
        cc = df['chiefcomplaint'].fillna('').str.upper()
        df['cc_arrest'] = cc.str.contains('ARREST').astype(float)
        df['cc_unresponsive'] = cc.str.contains('UNRESPONSIVE').astype(float)
        df['cc_code'] = cc.str.contains('CODE').astype(float)
        df['cc_deceas'] = cc.str.contains('DECEAS').astype(float)
        df = df.drop(columns=['chiefcomplaint'])
    else:
        for c in ['cc_arrest', 'cc_unresponsive', 'cc_code', 'cc_deceas']:
            df[c] = 0.0

    # One-hot encoding
    cats = ['gender', 'arrival_transport', 'avpu', 'consciousness_source']
    for cat in cats:
        if cat in df.columns:
            dummies = pd.get_dummies(df[cat], prefix=cat, dummy_na=True).astype(float)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[cat])

    return df
