import pandas as pd
import numpy as np
import random
import os

np.random.seed(42)
random.seed(42)

def generate_df(n_rows, n_dupes, has_target=True):
    n_unique = n_rows - n_dupes
    unique_ids = [f"SUBJ_{i:04d}" for i in range(n_unique)]
    
    # create dupes
    subject_ids = unique_ids + random.choices(unique_ids, k=n_dupes)
    random.shuffle(subject_ids)
    
    df = pd.DataFrame({'subject_id': subject_ids})
    
    df['gender'] = np.random.choice(['M', 'F'], n_rows)
    df['race'] = np.random.choice(['White', 'Black', 'Asian', 'Other'], n_rows)
    df['arrival_transport'] = np.random.choice(['AMBULANCE', 'WALK IN', 'HELICOPTER', 'OTHER'], n_rows)
    
    # vitals with some NaNs and some zeros
    df['temperature'] = np.where(np.random.rand(n_rows) > 0.1, np.random.normal(37.0, 1.0, n_rows), np.nan)
    df['heartrate'] = np.where(np.random.rand(n_rows) > 0.1, np.random.normal(80, 20, n_rows), np.nan)
    df['heartrate'] = np.where(np.random.rand(n_rows) > 0.95, 0, df['heartrate'])
    
    df['resprate'] = np.where(np.random.rand(n_rows) > 0.1, np.random.normal(16, 5, n_rows), np.nan)
    df['resprate'] = np.where(np.random.rand(n_rows) > 0.95, 0, df['resprate'])
    
    df['o2sat'] = np.where(np.random.rand(n_rows) > 0.1, np.random.normal(97, 3, n_rows), np.nan)
    df['sbp'] = np.where(np.random.rand(n_rows) > 0.1, np.random.normal(120, 20, n_rows), np.nan)
    df['dbp'] = np.where(np.random.rand(n_rows) > 0.1, np.random.normal(80, 15, n_rows), np.nan)
    
    df['bp_unobtainable'] = np.where(np.isnan(df['sbp']), 1.0, 0.0)
    df['pain_score'] = np.where(np.random.rand(n_rows) > 0.2, np.random.randint(0, 11, n_rows), np.nan)
    df['pain_assessable'] = np.where(np.isnan(df['pain_score']), 0.0, 1.0)
    
    df['gcs_eye'] = np.random.randint(1, 5, n_rows)
    df['gcs_verbal'] = np.random.randint(1, 6, n_rows)
    df['gcs_motor'] = np.random.randint(1, 7, n_rows)
    df['gcs_total'] = df['gcs_eye'] + df['gcs_verbal'] + df['gcs_motor']
    
    df['avpu'] = np.random.choice(['A', 'V', 'P', 'U'], n_rows)
    df['avpu_ordinal'] = df['avpu'].map({'A': 3, 'V': 2, 'P': 1, 'U': 0})
    df['follows_commands'] = np.random.choice([0.0, 1.0, np.nan], n_rows)
    df['consciousness_source'] = np.random.choice(['NURSE', 'DOCTOR', 'PARAMEDIC'], n_rows)
    
    complaints = ["Chest pain", "Shortness of breath", "ARREST", "UNRESPONSIVE", "CODE", "DECEASED", "Headache", "Fall", "Motor vehicle crash"]
    df['chiefcomplaint'] = np.random.choice(complaints, n_rows)
    
    for c in ['heartrate', 'resprate', 'sbp', 'o2sat']:
        df[f'vs_{c}_min'] = df[c] - np.random.uniform(0, 10, n_rows)
        df[f'vs_{c}_max'] = df[c] + np.random.uniform(0, 10, n_rows)
        df[f'vs_{c}_mean'] = df[c]
        
    df['vs_temperature_max'] = df['temperature'] + np.random.uniform(0, 1, n_rows)
    
    df['n_vitalsign_readings'] = np.random.randint(1, 20, n_rows)
    df['n_diagnoses'] = np.random.randint(0, 10, n_rows)
    df['n_home_meds'] = np.random.randint(0, 15, n_rows)
    
    if has_target:
        # Generate exactly the requested counts
        # YELLOW 284, GREEN 176, RED 135, BLACK 69
        targets = (['YELLOW'] * 284) + (['GREEN'] * 176) + (['RED'] * 135) + (['BLACK'] * 69)
        random.shuffle(targets)
        df['start_category'] = targets
        
    return df

os.makedirs("data", exist_ok=True)
train_df = generate_df(n_rows=664, n_dupes=133, has_target=True)
train_df.to_csv("data/train_ml03.csv", index=False)

test_df = generate_df(n_rows=112, n_dupes=25, has_target=False)
test_df.to_csv("data/test_pred_ml03.csv", index=False)

print("Data generated successfully.")
