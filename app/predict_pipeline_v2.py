# app/predict_pipeline_v2.py
import os
import pandas as pd
import joblib

MODEL_PATH = os.path.join("models/models_v2", "bail_reckoner_model.pkl")
PREPROCESS_PATH = os.path.join("models/models_v2", "preprocessing_objects.pkl")

class PredictError(Exception):
    pass

class BailPredictor:
    def __init__(self, model_path=MODEL_PATH, preproc_path=PREPROCESS_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        if not os.path.exists(preproc_path):
            raise FileNotFoundError(f"Preprocessing objects not found: {preproc_path}")

        self.model = joblib.load(model_path)
        preproc = joblib.load(preproc_path)
        self.label_encoders = preproc.get("label_encoders", {})
        self.scaler = preproc.get("scaler", None)
        if self.scaler is None:
            raise ValueError("Scaler missing in preprocessing objects")

    def _check_and_transform_cat(self, df, col):
        """Transform a categorical column using saved LabelEncoder.
           If unseen labels exist, raise PredictError listing allowed labels."""
        encoder = self.label_encoders.get(col)
        if encoder is None:
            raise PredictError(f"No encoder available for column '{col}'")

        vals = df[col].astype(str).tolist()
        unseen = [v for v in vals if v not in encoder.classes_]
        if unseen:
            allowed = sorted(list(encoder.classes_))
            raise PredictError(
                f"Unseen label(s) for column '{col}': {unseen}. "
                f"Allowed labels ({len(allowed)}): {allowed}"
            )
        # safe to transform
        df[col] = encoder.transform(df[col].astype(str))
        return df

    def preprocess(self, data):
        """Accept dict or DataFrame with exact feature names:
           ipc_section_count, has_special_law, bail_type, bail_cancellation_case,
           prior_cases, crime_type
        """
        df = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()

        # required columns check
        required = [
            "ipc_section_count", "has_special_law", "bail_type",
            "bail_cancellation_case", "prior_cases_count", "crime_type"
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise PredictError(f"Missing required columns: {missing}")

        # ensure numeric types
        df["ipc_section_count"] = pd.to_numeric(df["ipc_section_count"], errors="coerce").fillna(0.0)
        df["prior_cases_count"] = pd.to_numeric(df["prior_cases_count"], errors="coerce").fillna(0.0)
        df["has_special_law"] = df["has_special_law"].astype(int)
        df["bail_cancellation_case"] = df["bail_cancellation_case"].astype(int)

        # categorical transforms with strict check
        for col in ("bail_type", "crime_type"):
            df = self._check_and_transform_cat(df, col)

        # scale numeric columns using saved scaler (expects same order as training)
        numeric_cols = ["ipc_section_count", "prior_cases_count"]
        df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        # ensure column order matches training order (model expects same feature order)
        # Adjust this list if training used a different order.
        feature_order = ["ipc_section_count", "has_special_law", "bail_type",
                         "bail_cancellation_case", "prior_cases_count", "crime_type"]
        df = df[feature_order]

        return df

    def predict(self, data):
        df = self.preprocess(data)
        pred = self.model.predict(df)[0]
        return int(pred), "Granted" if pred == 1 else "Rejected"

if __name__ == "__main__":
    predictor = BailPredictor()

    # Example input - replace with categories that appear in your training encoders
    sample_input = {
        "ipc_section_count": 3,
        "has_special_law": 1,
        "bail_type": "Not applicable",            # MUST be in encoder.classes_ for 'bail_type'
        "bail_cancellation_case": 0,
        "prior_cases_count": 2,
        "crime_type": "Theft or Robbery"   # THIS caused your error if 'Theft' wasn't seen in training
    }

    try:
        numeric_pred, label = predictor.predict(sample_input)
        print(f"Prediction: {numeric_pred} -> {label}")
    except Exception as e:
        # Friendly, actionable error
        print("Prediction failed:", str(e))
        # If it's a PredictError we already provide details; otherwise print full exception.
