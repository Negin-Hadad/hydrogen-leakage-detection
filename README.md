# Hydrogen Leak Detection with XGBoost (Single-Sensor Version)

This repository contains a machine learning pipeline for predicting hydrogen system states using **a single hydrogen sensor** and time information.

The model predicts system safety conditions and recommended actions based on hydrogen concentration over time.

---

## Problem

Hydrogen systems require reliable monitoring to detect hazardous conditions.

In this simplified setup, the system uses:

- a **single hydrogen concentration sensor (ppm)**
- **time (seconds)**

to predict key operational states:

* **leak severity** – hazard level of the system  
* **action plan** – recommended response  

> ⚠️ Note:  
> The previous version included *leak location*, but this has been removed since it cannot be reliably inferred from a single sensor.

---

## Input Features

The model uses only two features:

* `ppm` – hydrogen concentration from a single sensor  
* `time (s)` – elapsed time in seconds  

---

## Dataset

The dataset is provided as an Excel file with two sheets:

Data   → sensor measurements and labels (used for training)  
Des.   → system component descriptions  

Only the **Data** sheet is used during training.

### Feature Transformation

The original dataset contains multiple sensor columns:

- compressor sensor (ppm)
- storage sensor (ppm)
- pipeline sensor (ppm)
- fuelcell sensor (ppm)

These are converted into a single feature:

ppm = mean(all sensor readings)

---

## Methodology

1. Load data  
2. Clean dataset  
3. Create `ppm` feature  
4. Select features: `["time (s)", "ppm"]`  
5. Encode labels  
6. Train/test split  
7. Train XGBoost models  
8. Hyperparameter tuning  
9. Evaluation  
10. Save model  

Targets:
- leak severity  
- action plan  

---

## Using the Model

```python
import joblib
import pandas as pd

bundle = joblib.load("hydrogen_leakage_prediction_model_single_sensor.pkl")

sample = pd.DataFrame([{
    "time (s)": 120,
    "ppm": 18.5
}])

predictions = {}

for target, model in bundle["models"].items():
    features = bundle["selected_feature_sets"][target]
    pred = model.predict(sample[features])[0]
    predictions[target] = bundle["label_encoders"][target].inverse_transform([pred])[0]

print(predictions)
```

---

## Changes from Previous Version

- Removed multi-sensor inputs  
- Removed leak location  
- Uses only `ppm` + `time (s)`  
- Simplified deployment  
