import nbformat as nbf

nb = nbf.v4.new_notebook()

text_intro = """# MM26ML03: Healthcare Triage Classification
This notebook builds the end-to-end machine learning pipeline for the hackathon."""

code_config = """TEAM_ID = "TEAMID"
import os
os.makedirs("outputs/artifacts", exist_ok=True)
os.makedirs("src", exist_ok=True)
"""

text_features = """## A2. Feature Engineering
The feature engineering logic is saved to `src/features.py` so it can be imported by the API."""

code_features = """%%writefile src/features.py
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
    if 'gcs_motor' not in df.columns:
        df['gcs_motor'] = np.nan
        
    df['start_resp_high'] = (df['resprate'].fillna(0) > 30).astype(float)
    df['start_resp_zero'] = (df['resprate'].fillna(1) == 0).astype(float)
    df['start_hr_zero'] = (df['heartrate'].fillna(1) == 0).astype(float)
    df['start_sbp_low'] = (df['sbp'].fillna(100) < 90).astype(float)
    df['start_motor_low'] = (df['gcs_motor'].fillna(6) < 6).astype(float)
    
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
"""

text_load = """## A1. Load and checks
Loading data and checking basic properties."""

code_load = """import pandas as pd
import numpy as np
from src.features import LABELS, COST_MATRIX, make_features

train_df = pd.read_csv("data/train_ml03.csv")
test_df = pd.read_csv("data/test_pred_ml03.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

train_subjects = set(train_df['subject_id'])
test_subjects = set(test_df['subject_id'])
print(f"Test subject_ids in train: {len(test_subjects.intersection(train_subjects))}")

y = train_df['start_category'].map({l: i for i, l in enumerate(LABELS)}).values
groups = train_df['subject_id'].values

X_train_raw = train_df.drop(columns=['start_category'])
X_test_raw = test_df.copy()

X_train = make_features(X_train_raw)
X_test = make_features(X_test_raw)

# align columns
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0.0)

feature_cols = list(X_train.columns)
"""

text_cv = """## A3. Grouped cross-validation
Because subject_ids repeat, we use StratifiedGroupKFold."""

code_cv = """from sklearn.model_selection import StratifiedGroupKFold

def repeated_group_splits(X, y, groups, n_splits=5, seeds=[42, 43, 44]):
    splits = []
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, val_idx in cv.split(X, y, groups):
            splits.append((train_idx, val_idx))
    return splits, len(seeds)

splits, n_repeats = repeated_group_splits(X_train, y, groups)
"""

text_models = """## A4. Models
Training LightGBM, ExtraTrees, and LogisticRegression."""

code_models = """import lightgbm as lgb
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, accuracy_score
import copy

def get_lgb():
    return lgb.LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=8, 
        min_child_samples=8, subsample=0.8, subsample_freq=1, 
        colsample_bytree=0.8, reg_lambda=1.0, verbosity=-1, random_state=42
    )

def get_et():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('et', ExtraTreesClassifier(n_estimators=300, max_depth=8, min_samples_leaf=4, random_state=42))
    ])

def get_lr():
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(C=0.1, max_iter=2000, random_state=42)) # No multi_class='multinomial' as it's deprecated
    ])

models = {'lightgbm': [], 'extratrees': [], 'logreg': []}
oof_preds = {'lightgbm': np.zeros((len(X_train), 4)), 
             'extratrees': np.zeros((len(X_train), 4)), 
             'logreg': np.zeros((len(X_train), 4))}
test_preds = {'lightgbm': np.zeros((len(X_test), 4)), 
              'extratrees': np.zeros((len(X_test), 4)), 
              'logreg': np.zeros((len(X_test), 4))}

counts = np.zeros(len(X_train))

for i, (train_idx, val_idx) in enumerate(splits):
    X_tr, y_tr = X_train.iloc[train_idx], y[train_idx]
    X_va, y_va = X_train.iloc[val_idx], y[val_idx]
    
    # LGBM
    m_lgb = get_lgb()
    m_lgb.fit(X_tr, y_tr)
    models['lightgbm'].append(m_lgb)
    oof_preds['lightgbm'][val_idx] += m_lgb.predict_proba(X_va)
    test_preds['lightgbm'] += m_lgb.predict_proba(X_test)
    
    # ET
    m_et = get_et()
    m_et.fit(X_tr, y_tr)
    models['extratrees'].append(m_et)
    oof_preds['extratrees'][val_idx] += m_et.predict_proba(X_va)
    test_preds['extratrees'] += m_et.predict_proba(X_test)
    
    # LR
    m_lr = get_lr()
    m_lr.fit(X_tr, y_tr)
    models['logreg'].append(m_lr)
    oof_preds['logreg'][val_idx] += m_lr.predict_proba(X_va)
    test_preds['logreg'] += m_lr.predict_proba(X_test)
    
    counts[val_idx] += 1

# Average OOF and test
for k in oof_preds:
    oof_preds[k] /= counts[:, None]
    test_preds[k] /= len(splits)

ens_oof = (oof_preds['lightgbm'] + oof_preds['extratrees'] + oof_preds['logreg']) / 3
ens_test = (test_preds['lightgbm'] + test_preds['extratrees'] + test_preds['logreg']) / 3

for k in oof_preds:
    print(f"{k} logloss: {log_loss(y, oof_preds[k])}")
print(f"ensemble logloss: {log_loss(y, ens_oof)}")
"""

text_cal = """## A5. Calibration
HONEST evaluation of calibration using StratifiedGroupKFold on OOF."""

code_cal = """from sklearn.metrics import brier_score_loss
import matplotlib.pyplot as plt

def get_calibrator():
    return LogisticRegression(C=1.0, max_iter=2000, random_state=42)

cal_oof = np.zeros_like(ens_oof)

for train_idx, val_idx in splits: # reusing splits
    X_tr_cal = np.log(np.clip(ens_oof[train_idx], 1e-15, 1-1e-15))
    X_va_cal = np.log(np.clip(ens_oof[val_idx], 1e-15, 1-1e-15))
    
    cal = get_calibrator()
    cal.fit(X_tr_cal, y[train_idx])
    cal_oof[val_idx] += cal.predict_proba(X_va_cal)

cal_oof /= counts[:, None]

# Fit final calibrator
final_calibrator = get_calibrator()
final_calibrator.fit(np.log(np.clip(ens_oof, 1e-15, 1-1e-15)), y)

# Apply to test
cal_test = final_calibrator.predict_proba(np.log(np.clip(ens_test, 1e-15, 1-1e-15)))

print(f"Logloss before: {log_loss(y, ens_oof)}, after: {log_loss(y, cal_oof)}")
"""

text_rule = """## A6. Minimum-risk decision
Applying expected cost minimization."""

code_rule = """from sklearn.metrics import f1_score, confusion_matrix

def expected_cost(probs):
    return probs @ COST_MATRIX

def min_risk_predict(probs):
    return expected_cost(probs).argmin(axis=1)

def get_metrics(preds):
    cm = confusion_matrix(y, preds, labels=[0,1,2,3])
    acc = np.trace(cm) / np.sum(cm)
    macro_f1 = f1_score(y, preds, average='macro')
    weighted_f1 = f1_score(y, preds, average='weighted')
    cost = (cm * COST_MATRIX).sum()
    r_as_g = cm[0, 2] # RED(0) as GREEN(2)
    r_as_b = cm[0, 3] # RED(0) as BLACK(3)
    return acc, macro_f1, weighted_f1, cost, r_as_g, r_as_b

results = {}
results['LGB_argmax'] = get_metrics(oof_preds['lightgbm'].argmax(axis=1))
results['Ens_argmax'] = get_metrics(ens_oof.argmax(axis=1))
results['Ens_minrisk'] = get_metrics(min_risk_predict(ens_oof))
results['CalEns_minrisk'] = get_metrics(min_risk_predict(cal_oof))

comparison_table = pd.DataFrame(results, index=['Accuracy', 'Macro-F1', 'Weighted-F1', 'Cost', 'R_as_G', 'R_as_B']).T
print(comparison_table)
"""

text_explain_rule = """By predicting the class that minimizes expected cost (multiplying probabilities by the cost matrix), we explicitly penalize costly mistakes like classifying a RED patient as GREEN. This can slightly lower overall accuracy but significantly reduces total misclassification cost."""

code_final = """from sklearn.metrics import classification_report
import json
import seaborn as sns

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

final_preds = min_risk_predict(cal_oof)
cr = classification_report(y, final_preds, target_names=LABELS, output_dict=True)
cm = confusion_matrix(y, final_preds, labels=[0,1,2,3])

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# Feature importance (LGBM)
imp = np.zeros(len(feature_cols))
for m in models['lightgbm']:
    imp += m.feature_importances_
imp /= len(models['lightgbm'])
top_features = pd.Series(imp, index=feature_cols).sort_values(ascending=False).head(10).to_dict()

# Save results
out_res = {
    'comparison': comparison_table.to_dict(orient='index'),
    'final_metrics': {
        'accuracy': results['CalEns_minrisk'][0],
        'macro_f1': results['CalEns_minrisk'][1],
        'weighted_f1': results['CalEns_minrisk'][2],
        'cost': results['CalEns_minrisk'][3],
    },
    'per_class': cr,
    'confusion_matrix': cm.tolist(),
    'logloss_before': log_loss(y, ens_oof),
    'logloss_after': log_loss(y, cal_oof),
    'class_counts': {LABELS[i]: int(c) for i, c in enumerate(np.bincount(y))},
    'top_features': top_features
}

with open('outputs/results.json', 'w') as f:
    json.dump(out_res, f, indent=2, cls=NpEncoder)
"""

text_save = """## A9. Save Artifacts and Submission"""

code_save = """import joblib
joblib.dump(models, 'outputs/artifacts/models.joblib')
joblib.dump(final_calibrator, 'outputs/artifacts/calibrator.joblib')
joblib.dump(feature_cols, 'outputs/artifacts/feature_cols.joblib')

meta = {
    'LABELS': LABELS,
    'COST_MATRIX': COST_MATRIX.tolist()
}
with open('outputs/artifacts/metadata.json', 'w') as f:
    json.dump(meta, f)

# Submission
sub_preds = min_risk_predict(cal_test)
sub_df = pd.DataFrame({
    'Patient ID': test_df['subject_id'],
    'Predicted Triage': [LABELS[p] for p in sub_preds],
    'RED Probability': cal_test[:, 0].round(4),
    'YELLOW Probability': cal_test[:, 1].round(4),
    'GREEN Probability': cal_test[:, 2].round(4),
    'BLACK Probability': cal_test[:, 3].round(4)
})

# normalize to sum to 1 to fix rounding
prob_cols = ['RED Probability', 'YELLOW Probability', 'GREEN Probability', 'BLACK Probability']
sub_df[prob_cols] = sub_df[prob_cols].div(sub_df[prob_cols].sum(axis=1), axis=0).round(4)

sub_df.to_csv(f'outputs/{TEAM_ID}_MM26ML03.csv', index=False)
sub_df.to_csv(f'{TEAM_ID}_MM26ML03.csv', index=False)
"""

cells = [
    nbf.v4.new_markdown_cell(text_intro),
    nbf.v4.new_code_cell(code_config),
    nbf.v4.new_markdown_cell(text_features),
    nbf.v4.new_code_cell(code_features),
    nbf.v4.new_markdown_cell(text_load),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_markdown_cell(text_cv),
    nbf.v4.new_code_cell(code_cv),
    nbf.v4.new_markdown_cell(text_models),
    nbf.v4.new_code_cell(code_models),
    nbf.v4.new_markdown_cell(text_cal),
    nbf.v4.new_code_cell(code_cal),
    nbf.v4.new_markdown_cell(text_rule),
    nbf.v4.new_code_cell(code_rule),
    nbf.v4.new_markdown_cell(text_explain_rule),
    nbf.v4.new_code_cell(code_final),
    nbf.v4.new_markdown_cell(text_save),
    nbf.v4.new_code_cell(code_save),
]

nb['cells'] = cells

with open('MM26ML03_triage.ipynb', 'w') as f:
    nbf.write(nb, f)

print("Notebook generated.")
