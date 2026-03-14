# Hydrogen Leak Detection with XGBoost

This repository contains a machine learning pipeline for predicting hydrogen system states using sensor measurements.
The model predicts leak detection and response information based on hydrogen concentration sensors deployed in different system components.

The project uses **XGBoost** models with hyperparameter tuning and compares feature configurations to determine whether including time information improves predictions.

---

## Problem

Hydrogen systems rely on multiple sensors to monitor hydrogen concentration.
Given the readings from these sensors, the goal is to predict several operational labels:

* **leak_label** – whether a leak is present
* **leak location** – location of the leak
* **leak severity** – hazard level
* **action plan** – recommended response
* **phase** – system operation phase

Each sample represents a snapshot of sensor measurements.

---

## Input Features

Sensor readings used as predictors:

* `compressor sensor (ppm)`
* `storage sensor (ppm)`
* `pipeline sensor (ppm)`
* `fuelcell sensor (ppm)`

An additional experiment evaluates whether including:

* `time (s)`

improves model performance.

---

## Dataset

The dataset is provided as an Excel file with two sheets:

```
Data   → sensor measurements and labels (used for training)
Des.   → system component descriptions
```

Only the **Data** sheet is used during model training.

---

## Methodology

The training pipeline follows these steps:

1. **Load data**
2. **Clean dataset** (remove missing values)
3. **Encode labels** using `LabelEncoder`
4. **Train/test split** (80/20, stratified by leak label)
5. **Model training using XGBoost**
6. **Hyperparameter tuning** with `RandomizedSearchCV`
7. **Cross-validation**
8. **Model evaluation**
9. **Feature importance analysis**
10. **Export trained model bundle**

A separate model is trained for each predicted label.

---

## Why XGBoost?

XGBoost (Extreme Gradient Boosting) is an ensemble algorithm based on decision trees.

It builds many trees sequentially, where each new tree learns to correct the errors of previous trees.

XGBoost is well suited for **tabular sensor data** because it:

* handles nonlinear relationships well
* is robust to noise
* provides strong predictive performance
* offers built-in feature importance analysis

---

## Experimental Results

The models were evaluated using an 80/20 train–test split with stratification on `leak_label`.
Hyperparameters were tuned using `RandomizedSearchCV` with cross-validation.

A separate XGBoost model was trained for each target variable.

| Target        | Description             | Test Accuracy |
| ------------- | ----------------------- | ------------- |
| leak_label    | Leak detection          | ~0.98         |
| leak location | Location classification | ~0.87         |
| leak severity | Hazard level            | ~0.89         |
| action plan   | Recommended response    | ~0.86         |
| phase         | Operational phase       | ~0.94         |

Feature importance analysis showed that the **pipeline sensor** and **compressor sensor** contribute most to leak detection in most models.

An additional experiment compared two feature sets:

**Baseline**

```
compressor sensor
storage sensor
pipeline sensor
fuelcell sensor
```

**Extended**

```
time (s)
+ sensor measurements
```

The training pipeline automatically selects the configuration that yields the best performance for each target.

Confusion matrices and feature importance plots are generated during training in the notebooks.

## Repository Structure

```
project/
│
├── data/
│   └── H2-Dataset.xlsx
│
├── notebooks/
│   └── hydrogen_leakage_detection.ipynb
│
├── models/
│   └── hydrogen_leakage_prediction_model.pkl
│
├── requirements.txt
└── README.md
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/Negin-Hadad/hydrogen-leakage-detection.git
cd hydrogen-leak-detection
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```
pandas
numpy
scikit-learn
xgboost
matplotlib
joblib
openpyxl
jupyter
```

---

## Training

Run the training notebook:

```
notebooks/hydrogen-leakage-detection.ipynb
```

The notebook will:

* train the models
* compare feature sets (with and without time)
* evaluate performance
* export the trained model bundle (`.pkl`)

---

## Using the Model

Example inference:

```python
import joblib
import pandas as pd

bundle = joblib.load("hydrogen_leakage_prediction_model.pkl")

models = bundle["models"]
encoders = bundle["label_encoders"]

sample = pd.DataFrame([{
    "compressor sensor (ppm)": 20.5,
    "storage sensor (ppm)": 15.2,
    "pipeline sensor (ppm)": 18.9,
    "fuelcell sensor (ppm)": 14.7,
    "time (s)": 120
}])

predictions = {}

for target, model in models.items():
    features = bundle["selected_feature_sets"][target]
    pred = model.predict(sample[features])[0]
    predictions[target] = encoders[target].inverse_transform([pred])[0]

print(predictions)
```

---

## Hardware

Training runs efficiently on **CPU**.  
GPU acceleration is not required for this dataset.


---

## Author

Negin Hadad  
University of Oulu, Oulu, Finland