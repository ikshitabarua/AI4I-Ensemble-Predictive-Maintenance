
from pathlib import Path
import math
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Paths and page setup
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

st.set_page_config(
    page_title="AI4I Predictive Maintenance",
    page_icon="⚙️",
    layout="wide",
)

st.title("AI4I Predictive Maintenance — Failure Risk Demo")
st.caption(
    "Educational prototype using the AI4I 2020 dataset. "
    "This is decision support, not an industrially validated safety system."
)

# --------------------------------------------------
# Load saved model and feature information
# --------------------------------------------------
model_path = MODELS / "stacking_ensemble.joblib"
info_path = MODELS / "thresholds_and_features.joblib"

if not model_path.exists():
    st.error(f"Model file not found: {model_path}")
    st.info("Run your training script first.")
    st.stop()

if not info_path.exists():
    st.error(f"Feature information file not found: {info_path}")
    st.stop()

try:
    model = joblib.load(model_path)
    saved_info = joblib.load(info_path)
except Exception as exc:
    st.error(f"Could not load saved model information: {exc}")
    st.stop()


def find_feature_columns(obj):
    """Find saved feature column names without guessing their order."""
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)

    if isinstance(obj, (list, tuple)) and all(
        isinstance(item, str) for item in obj
    ):
        return list(obj)

    if isinstance(obj, dict):
        # Common feature-column keys used by training scripts
        for key in (
            "feature_columns",
            "features",
            "columns",
            "feature_names",
            "X_columns",
        ):
            if key in obj:
                result = find_feature_columns(obj[key])
                if result:
                    return result

        # Check nested dictionaries
        for value in obj.values():
            if isinstance(value, dict):
                result = find_feature_columns(value)
                if result:
                    return result

    return None


def find_thresholds(obj):
    """Find saved model thresholds if the saved structure contains them."""
    if not isinstance(obj, dict):
        return None

    # Thresholds may be stored directly or inside a nested dictionary.
    for key in ("thresholds", "model_thresholds"):
        value = obj.get(key)
        if isinstance(value, dict):
            return value

    if "Stacking Ensemble" in obj:
        value = obj["Stacking Ensemble"]
        if isinstance(value, (int, float)):
            return obj

    for value in obj.values():
        if isinstance(value, dict):
            result = find_thresholds(value)
            if result is not None:
                return result

    return None


feature_columns = find_feature_columns(saved_info)
thresholds = find_thresholds(saved_info)

if not feature_columns:
    st.error(
        "Could not identify the feature columns in "
        "thresholds_and_features.joblib."
    )
    st.write("Saved object type:", type(saved_info).__name__)

    if isinstance(saved_info, dict):
        st.write("Available keys:", list(saved_info.keys()))

    st.info(
        "Please check how ml/train_ensemble.py saves this file. "
        "The dashboard must use exactly the same feature columns "
        "and order as the trained model."
    )
    st.stop()

# Do not silently use 0.5 when the tuned threshold is unavailable.
if thresholds is None or "Stacking Ensemble" not in thresholds:
    st.error(
        "The tuned Stacking Ensemble threshold was not found in "
        "thresholds_and_features.joblib. Check the saved threshold "
        "structure in ml/train_ensemble.py before making predictions."
    )
    if isinstance(saved_info, dict):
        st.write("Available saved keys:", list(saved_info.keys()))
    st.stop()

threshold = float(thresholds["Stacking Ensemble"])

if not math.isfinite(threshold) or not 0 <= threshold <= 1:
    st.error("The saved classification threshold is invalid.")
    st.stop()

# --------------------------------------------------
# Machine input controls
# --------------------------------------------------
st.sidebar.header("Machine operating conditions")

type_value = st.sidebar.selectbox(
    "Machine type",
    ["L", "M", "H"],
    index=1,
)

air = st.sidebar.number_input(
    "Air temperature (K)",
    min_value=250.0,
    max_value=350.0,
    value=300.0,
    step=0.5,
)

process = st.sidebar.number_input(
    "Process temperature (K)",
    min_value=250.0,
    max_value=400.0,
    value=310.0,
    step=0.5,
)

rpm = st.sidebar.number_input(
    "Rotational speed (rpm)",
    min_value=0.0,
    max_value=5000.0,
    value=1500.0,
    step=10.0,
)

torque = st.sidebar.number_input(
    "Torque (Nm)",
    min_value=0.0,
    max_value=200.0,
    value=40.0,
    step=1.0,
)

wear = st.sidebar.number_input(
    "Tool wear (min)",
    min_value=0.0,
    max_value=300.0,
    value=100.0,
    step=1.0,
)

# --------------------------------------------------
# Feature engineering
# Keep these definitions consistent with training.
# --------------------------------------------------
row = {
    "Air temperature": air,
    "Process temperature": process,
    "Rotational speed": rpm,
    "Torque": torque,
    "Tool wear": wear,
    "Temperature Difference": process - air,
    "Mechanical Power _W_": torque * rpm * 2 * math.pi / 60,
    "Torque_RPM": torque * rpm,
    "ToolWear_Torque": wear * torque,
    "ToolWear_RPM": wear * rpm,
    "Type_L": int(type_value == "L"),
    "Type_M": int(type_value == "M"),
    "Type_H": int(type_value == "H"),
}

missing_features = [c for c in feature_columns if c not in row]

if missing_features:
    st.error(
        "Some trained model features cannot be created by the dashboard. "
        "The feature engineering must be aligned with training."
    )
    st.write("Missing features:", missing_features)
    st.stop()

input_df = pd.DataFrame(
    [[row[c] for c in feature_columns]],
    columns=feature_columns,
)

# XGBoost may have been trained using cleaned feature names.
# The selected stacking model must receive the exact expected columns.
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
    if expected != list(input_df.columns):
        st.error(
            "The saved model's expected feature columns do not match "
            "the dashboard feature columns. Check training preprocessing."
        )
        st.write("Model expects:", expected)
        st.write("Dashboard provides:", list(input_df.columns))
        st.stop()

# --------------------------------------------------
# Prediction
# --------------------------------------------------
st.subheader("Failure risk prediction")

if st.button("Predict machine failure risk", type="primary"):
    try:
        probabilities = model.predict_proba(input_df)[0]
        classes = list(model.classes_)

        if 1 not in classes:
            st.error(
                "The trained model does not contain class 1 "
                "(machine failure). Check target-label encoding."
            )
            st.stop()

        failure_probability = float(probabilities[classes.index(1)])
        prediction = int(failure_probability >= threshold)

        if prediction == 1:
            risk = "HIGHER RISK"
            action = (
                "Consider a qualified inspection and review recent "
                "operating history."
            )
        else:
            risk = "LOWER RISK"
            action = (
                "Continue routine monitoring. This prediction does not "
                "guarantee that failure will not occur."
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Predicted failure probability",
            f"{failure_probability:.1%}",
        )
        col2.metric("Threshold used", f"{threshold:.3f}")
        col3.metric("Model decision", risk)

        if prediction == 1:
            st.warning(action)
        else:
            st.info(action)

        st.subheader("Input features sent to model")
        st.dataframe(
            input_df.T.rename(columns={0: "Value"}),
            use_container_width=True,
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

st.divider()
st.caption(
    "The classification threshold should come from the validation "
    "process, not the test set. Probability calibration and industrial "
    "deployment require further validation."
)
