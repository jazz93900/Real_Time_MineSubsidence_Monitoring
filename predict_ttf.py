import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from risk_engine import classify_risk


# ==========================================
# LOAD MODEL AND SCALERS
# ==========================================

model = load_model("ttf_lstm_model.keras")

feature_scaler = joblib.load(
    "lstm_feature_scaler.pkl"
)

target_scaler = joblib.load(
    "lstm_target_scaler.pkl"
)

print("Model and scalers loaded successfully!")


# ==========================================
# PREDICTION FUNCTION
# ==========================================

WINDOW_SIZE = 30


def predict_ttf(sensor_data):

    if len(sensor_data) < WINDOW_SIZE:
        raise ValueError(
            "At least 30 readings are required."
        )

    # Take the latest 30 readings
    recent_data = sensor_data[
        -WINDOW_SIZE:
    ].copy()

    # Calculate inverse velocity
    recent_data["inverse_velocity"] = (
        1 / recent_data["velocity"]
    )

    # Select features in EXACT training order
    features = recent_data[
        [
            "velocity",
            "deformation",
            "inverse_velocity"
        ]
    ].values

    # Scale using the SAME scaler from training
    features_scaled = feature_scaler.transform(
        features
    )

    # Add batch dimension
    X = np.expand_dims(
        features_scaled,
        axis=0
    )

    # Predict
    prediction_scaled = model.predict(
        X,
        verbose=0
    )

    # Convert prediction back to hours
    prediction = target_scaler.inverse_transform(
        prediction_scaled
    )

    # Raw LSTM prediction
    prediction = float(
        prediction[0][0]
    )

    return prediction


# ==========================================
# TEST PREDICTIONS AT DIFFERENT POINTS
# ==========================================

data = pd.read_csv(
    "all_creep_scenarios.csv"
)

test_scenario = data[
    data["scenario_id"] == 1
].copy()

test_scenario = test_scenario.reset_index(
    drop=True
)


print("\n======================================")
print("TTF PROGRESSION TEST - SCENARIO 1")
print("======================================")


# Points where we want to test
test_points = [
    100,
    300,
    500,
    700,
    900,
    1100,
    len(test_scenario)
]


for point in test_points:

    if point < WINDOW_SIZE:
        continue

    current_data = test_scenario.iloc[
        :point
    ].copy()

    prediction = predict_ttf(
        current_data
    )

    risk = classify_risk(
        prediction
    )

    actual_ttf = current_data[
        "ttf"
    ].iloc[-1]

    print(
        f"\nReading: {point}"
    )

    print(
        f"Actual TTF:     {actual_ttf:.3f} h"
    )

    print(
        f"Predicted TTF:  {prediction:.3f} h"
    )

    print(
        f"Prototype Risk: {risk}"
    )

    print(
        f"Error:          "
        f"{prediction - actual_ttf:+.3f} h"
    )