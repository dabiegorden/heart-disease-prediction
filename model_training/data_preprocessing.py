import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class CardiovascularDataPreprocessor:
    """Preprocessor for cardiovascular disease datasets."""

    # Standardized feature order (must match ONNX input)
    FEATURE_NAMES = [
        "age",
        "sex",
        "chestpaintype",
        "restingbps",
        "cholesterol",
        "fastingbloodsugar",
        "restingecg",
        "maxheartrate",
        "exerciseangina",
        "oldpeak",
        "slope",
        "noofmajorvessels"
    ]

    # ---------------------------------------------------------
    # LOAD DATASET 1 (Cardiovascular_Disease_Dataset.xlsx)
    # ---------------------------------------------------------
    @staticmethod
    def load_cardiovascular_dataset(filepath):
        df = pd.read_excel(filepath)

        df.columns = df.columns.str.lower()

        mapping = {
            "age": "age",
            "gender": "sex",
            "chestpain": "chestpaintype",
            "restingbp": "restingbps",
            "serumcholestrol": "cholesterol",
            "fastingbloodsugar": "fastingbloodsugar",
            "restingrelectro": "restingecg",
            "maxheartrate": "maxheartrate",
            "exerciseangia": "exerciseangina",
            "oldpeak": "oldpeak",
            "slope": "slope",
            "noofmajo": "noofmajorvessels",
            "target": "target"
        }

        df = df.rename(columns=mapping)
        df = df.drop(columns=["patientid"], errors="ignore")

        return df

    # ---------------------------------------------------------
    # LOAD DATASET 2 (Heart_Disease x2.xlsx)
    # ---------------------------------------------------------
    @staticmethod
    def load_heart_disease_dataset(filepath):
        df = pd.read_excel(filepath)

        df.columns = df.columns.str.lower()

        mapping = {
            "age": "age",
            "sex": "sex",
            "chest pain type": "chestpaintype",
            "resting bp s": "restingbps",
            "cholesterol": "cholesterol",
            "fasting blood sugar": "fastingbloodsugar",
            "resting ecg": "restingecg",
            "max heart rate": "maxheartrate",
            "exercise angina": "exerciseangina",
            "oldpeak": "oldpeak",
            "st slope": "slope",
            "target": "target"
        }

        df = df.rename(columns=mapping)

        # dataset2 does not contain "noofmajorvessels"
        if "noofmajorvessels" not in df.columns:
            df["noofmajorvessels"] = 0

        return df

    # ---------------------------------------------------------
    # PREPROCESS DATA (Scaling)
    # ---------------------------------------------------------
    @staticmethod
    def preprocess(df, fit_scaler=False, scaler=None):
        df = df.fillna(df.mean(numeric_only=True))

        X = df[CardiovascularDataPreprocessor.FEATURE_NAMES].copy()
        y = df["target"]

        if fit_scaler:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
        else:
            if scaler is None:
                raise ValueError("Scaler is required for transformation.")
            X_scaled = scaler.transform(X)

        return X_scaled, y, scaler

    # ---------------------------------------------------------
    # COMPLETE PIPELINE: Load + Preprocess + Sync Scaling
    # ---------------------------------------------------------
    @staticmethod
    def load_and_prepare(filepath, dataset_type="auto", fit_scaler=False, scaler=None):

        df_raw = pd.read_excel(filepath)
        cols = [c.lower() for c in df_raw.columns]

        # Auto detect dataset
        if dataset_type == "auto":
            dataset_type = "cardiovascular" if "patientid" in cols else "heart"

        if dataset_type == "cardiovascular":
            df = CardiovascularDataPreprocessor.load_cardiovascular_dataset(filepath)
        else:
            df = CardiovascularDataPreprocessor.load_heart_disease_dataset(filepath)

        X, y, scaler = CardiovascularDataPreprocessor.preprocess(
            df, fit_scaler=fit_scaler, scaler=scaler
        )

        return X, y, scaler
