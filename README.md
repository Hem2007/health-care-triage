# MM26ML03: Healthcare Triage Classification

## 1. Problem and Cost Matrix
The goal is to classify emergency patients into four START triage categories: RED, YELLOW, GREEN, and BLACK. Errors have unequal costs, heavily penalizing undertriage (e.g., predicting GREEN for a RED patient). We use the official cost matrix exactly:

| Actual \ Predicted | RED | YELLOW | GREEN | BLACK |
|---|---|---|---|---|
| **RED** | 0 | 5 | 10 | 5 |
| **YELLOW** | 2 | 0 | 3 | 2 |
| **GREEN** | 1 | 1 | 0 | 1 |
| **BLACK** | 2 | 2 | 2 | 0 |

## 2. Data and Features
- Excluded `subject_id` and `race` (design choice: not a clinical signal).
- Added boolean flags indicating missing values for vital signs (`_is_missing`).
- Added START-style boolean flags (e.g., `resprate > 30`, `resprate == 0`, `heartrate == 0`, `sbp < 90`, `gcs_motor < 6`).
- Preserved zeros and NaNs in numeric vitals, as they are meaningful for specific conditions (like BLACK).

## 3. Validation
We used Repeated StratifiedGroupKFold cross-validation (grouped by `subject_id`). This ensures that multiple rows corresponding to the same patient do not leak across the train/validation splits, providing a more honest evaluation.

## 4. Models
We built a simple ensemble by averaging the probability predictions of three regularized models:
- LightGBM
- ExtraTrees Classifier
- Logistic Regression

## 5. Calibration
To ensure the output probabilities are trustworthy before applying the cost matrix, we trained a Multinomial Logistic Regression calibrator on the Out-Of-Fold probabilities.
- **Log-loss before calibration:** 1.330
- **Log-loss after calibration:** 1.283

## 6. Decision Rule
Instead of picking the class with the highest probability (argmax), we apply the minimum expected risk decision rule using the official cost matrix:
`expected_cost(j) = sum_i P(i) * Cost[i][j]`
We predict the class `j` that minimizes this expected cost.

## 7. Results
This table compares standard `argmax` decisions vs our `min-risk` decision rule (evaluated on Out-Of-Fold cross-validation data):

| Approach | Accuracy | Macro-F1 | Total Cost | RED predicted as GREEN |
|---|---|---|---|---|
| LightGBM argmax | 0.355 | 0.228 | 1261.0 | 24 |
| Ensemble argmax | 0.395 | 0.201 | 1143.0 | 14 |
| Ensemble + min-risk (uncalibrated) | 0.300 | 0.182 | 943.0 | 0 |
| **Ensemble + min-risk (calibrated, FINAL)** | **0.217** | **0.116** | **909.0** | **0** |

## 8. Note on Predicted Labels
**Important:** Because of the minimum-risk decision rule, the predicted label will frequently differ from the class with the highest probability. The model prioritizes safety (lower expected cost) over raw accuracy.

## 9. Limitations
- **Small Dataset:** The model is trained on a very limited number of patients.
- **Cross-Validated Estimates:** The metrics reported are CV estimates rather than performance on a true hidden holdout test score.
- **Data-Collection Artifacts:** Some of the top features (like missingness flags) may reflect hospital data-collection patterns rather than true clinical signs.

## 10. Run the Web Demo
To run the interactive web demonstration locally:
1. Start the backend: `uvicorn backend.main:app --reload --port 8000`
2. Start the frontend (in the `frontend/` directory): `npm run dev`
