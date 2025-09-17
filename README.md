# Heat-Failure-Prediction
Baseline clinical ML pipeline using synthesized MIMIC-III–derived data: descriptive stats, feature engineering, SVMlight export, and model benchmarking (LR/SVM/DT) with cross-validation. This README reflects the code in HW1.py.

## Project Background

### Company context
A regional hospital network aims to reduce 30-day heart-failure (HF) readmissions and ICU escalations. In a value-based care model, the organization benefits from preventing avoidable admissions while improving outcomes. This project’s goal is to flag at-risk patients earlier so clinicians can prioritize follow-ups, adjust meds, and allocate case-management resources.

Earlier risk signals → fewer readmissions, lower cost, better outcomes. A reliable baseline model + clean feature pipeline creates the foundation for future deep-learning work (e.g., temporal sequence models on visit histories).

### Terminology/ metrics/ dimensions

- pid: de-identified patient ID

- event_id: clinical event (diagnosis / lab / med)

- vid: visit index (ordinal; larger = later)

- value: event value (1 in this dataset)

- index_vid: first HF visit for HF patients; last visit for non-HF

- Observation window: visits strictly before index_vid (prevents label leakage)

- Prediction window: post-window horizon where outcomes occur

- Event count: #rows in events.csv per patient

- Encounter count: #unique vid per patient

- Normalization: per-feature min-max (count / feature-wise max)

Data sources: events.csv, hf_events.csv, event_feature_map.csv.

## Executive Summary

### ERD 
<img width="549" height="142" alt="image" src="https://github.com/user-attachments/assets/0c8aa1d5-aad9-4e35-9259-e9bc5bde2be1" />

### High-level findings 

- HF patients show higher utilization before diagnosis (more events and slightly more encounters), signaling elevated risk.

- A simple feature pipeline (counts → per-feature min-max) + linear models already yields useful discrimination.

- From the provided split in HW1.py: logistic regression achieves ~0.86 training accuracy and ~0.69 validation accuracy; 5-fold CV F1 is ~0.73 for SVM (baseline, untuned).

## Insights Deep-Dive
### 1) Descriptive statistics

- Events per patient

 HF: higher average and wider spread (heavier pre-diagnosis utilization)

Non-HF: lower average, narrower spread

- Encounters per patient

HF: more visits on average than non-HF

Interpretation: HF cohort interacts more frequently/intensely with the system pre-diagnosis.

### 2) Feature engineering checks

- Windowing: keep events with vid < index_vid (prevents leakage).

- Aggregation: counts by (pid, feature_id); per-feature min-max normalization.

- SVMlight export: label +1 for HF, -1 otherwise; feature IDs strictly increasing, duplicates merged, zeros/NaNs dropped.

### 3) Modeling & validation

- Models: Logistic Regression, Linear SVM, Decision Tree (max_depth=5), RANDOM_STATE=545510477.

- Illustrative results from HW1.py

LR training accuracy ≈ 0.8563

LR validation accuracy ≈ 0.6937

5-fold CV F1 (SVM) ≈ 0.7258

- Takeaway: Linear baselines are competitive and stable; decision tree offers interpretability but may require tuning/regularization.

## Recommendations
### Clinical Operations / Care Management

- Create a nurse review queue for top-risk patients (threshold tuned to desired precision/recall trade-off).

- Close the loop: integrate risk flags into post-discharge workflows (med rec, follow-up calls, early appointments).

### Data Science / Engineering

- Enrich features: recency/tempo, visit-level aggregates, comorbidity indices.

- Address class imbalance: probability calibration, class weights; explore focal losses later.

- Temporal DL next: sequence models (GRU/LSTM/Transformer) on visit timelines.

- Model governance: monitoring for drift, fairness, and threshold re-tuning.

### Product / Compliance

- Document thresholds & explainability artifacts for clinical sign-off.

- Ensure retention & PHI handling policies when moving to real EHR data.


