import joblib
import pandas as pd

bundle = joblib.load("./models/hydrogen_leakage_prediction_model_single_sensor.pkl")

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


# output sample:
# ```
# {'leak severity': 'medium', 'action plan': 'inspect_and_repair'}
# ```
