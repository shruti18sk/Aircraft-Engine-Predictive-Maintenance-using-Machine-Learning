
import streamlit as st
import pandas as pd
import joblib

# 1. Load Model and Features
model = joblib.load("best_rul_model.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# 2. Real Validation Sample
sample_input = {
    "cycle": 1.0000,
    "setting_1": -0.0007,
    "setting_2": -0.0004,
    "setting_3": 100.0000,
    "sensor_2": 641.8200,
    "sensor_3": 1589.7000,
    "sensor_4": 1400.6000,
    "sensor_6": 21.6100,
    "sensor_7": 554.3600,
    "sensor_8": 2388.0600,
    "sensor_9": 9046.1900,
    "sensor_11": 47.4700,
    "sensor_12": 521.6600,
    "sensor_13": 2388.0200,
    "sensor_14": 8138.6200,
    "sensor_15": 8.4195,
    "sensor_17": 392.0000,
    "sensor_20": 39.0600,
    "sensor_21": 23.4190
}

# 3. Page Configuration
st.set_page_config(
    page_title="Aircraft Engine Health Monitoring",
    page_icon="✈️",
    layout="wide"
)

# 4. Dashboard Header
st.title("✈️ Aircraft Engine Health Monitoring System")

st.write(
    "Predict the Remaining Useful Life (RUL) "
    "of an aircraft engine using sensor data."
)

st.divider()

# 5. Sensor Readings
st.subheader("Engine Sensor Readings")

# Initialize session state
for feature in feature_columns:
    if f"input_{feature}" not in st.session_state:
        st.session_state[f"input_{feature}"] = 0.0

# Load Sample Button
if st.button("📥 Load Real Sample Data"):
    for feature in feature_columns:
        st.session_state[f"input_{feature}"] = float(
            sample_input[feature]
        )

    st.session_state["sample_loaded"] = True
    st.rerun()

if st.session_state.get("sample_loaded", False):
    st.success("Real validation sample loaded successfully!")

# 6. Input Form
with st.form("rul_prediction_form"):

    input_values = {}

    for feature in feature_columns:
        input_values[feature] = st.number_input(
            feature,
            key=f"input_{feature}",
            format="%.4f"
        )

    submit = st.form_submit_button("🔍 Predict RUL")

# 7. Prediction
if submit:

    input_df = pd.DataFrame(
        [input_values],
        columns=feature_columns
    )

    prediction = model.predict(input_df)[0]

    st.divider()
    st.subheader("📊 Prediction Results")

    # KPI Cards
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Predicted RUL",
        f"{prediction:.2f} cycles"
    )

    if st.session_state.get("sample_loaded", False):

        actual_rul = 191.0
        error = abs(actual_rul - prediction)

        col2.metric(
            "Actual RUL",
            f"{actual_rul:.2f} cycles"
        )

        col3.metric(
            "Prediction Error",
            f"{error:.2f} cycles"
        )

        st.info(
            "Prediction based on the real validation sample."
        )

        # Actual vs Predicted Chart
        chart_data = pd.DataFrame({
            "RUL Type": ["Actual RUL", "Predicted RUL"],
            "Cycles": [actual_rul, prediction]
        })

        st.subheader("Actual vs Predicted RUL")

        st.bar_chart(
            chart_data,
            x="RUL Type",
            y="Cycles"
        )

    else:
        st.info(
            "Custom sensor input prediction. "
            "Actual RUL is unavailable for this input."
        )

# 8. Footer
st.divider()

st.caption(
    "Aircraft Engine Predictive Maintenance | "
    "Machine Learning Project"
)
