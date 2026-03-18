"""
data_preprocessing.py
=====================
Universal preprocessor for heart / cardiovascular disease datasets.

Design goals
------------
* Accept **any** tabular dataset that contains a binary target column.
* Auto-detect the target column by name (many common spellings supported).
* Build a canonical feature set from whatever columns are present, so that
  two hospitals with different column sets can both be processed without
  code changes.
* Provide a stable ``FEATURE_NAMES`` list *per fitted instance* that is
  saved alongside the scaler and reloaded at inference time.
* Remain backwards-compatible with the original 12-feature ONNX models when
  exactly those columns are present.

Column aliasing
---------------
The alias map translates every known spelling / casing variant into a single
canonical name.  If a column does not appear in the map it is kept as-is
(lowercased, stripped).  This means a completely new column from a new hospital
is still accepted – it just gets its own canonical name.
"""

from __future__ import annotations

import re
import json
import logging
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ALIAS MAP  (raw column name → canonical name)
# Add rows here whenever a new dataset variant is encountered.
# ---------------------------------------------------------------------------
COLUMN_ALIAS_MAP: dict[str, str] = {
    # patient id / index – always dropped
    "patientid":            "__drop__",
    "patient_id":           "__drop__",
    "id":                   "__drop__",

    # ── age ──────────────────────────────────────────────────────────────
    "age":                  "age",

    # ── sex / gender ─────────────────────────────────────────────────────
    "sex":                  "sex",
    "gender":               "sex",

    # ── chest pain ───────────────────────────────────────────────────────
    "chestpain":            "chest_pain_type",
    "chest pain type":      "chest_pain_type",
    "chest_pain_type":      "chest_pain_type",
    "chestpaintype":        "chest_pain_type",
    "cp":                   "chest_pain_type",

    # ── resting blood pressure ───────────────────────────────────────────
    "restingbp":            "resting_bp",
    "restingbps":           "resting_bp",
    "resting bp s":         "resting_bp",
    "resting_bp_s":         "resting_bp",
    "resting_bp":           "resting_bp",
    "trestbps":             "resting_bp",

    # ── cholesterol ──────────────────────────────────────────────────────
    "cholesterol":          "cholesterol",
    "serumcholestrol":      "cholesterol",
    "serum_cholesterol":    "cholesterol",
    "chol":                 "cholesterol",

    # ── fasting blood sugar ──────────────────────────────────────────────
    "fastingbloodsugar":    "fasting_blood_sugar",
    "fasting blood sugar":  "fasting_blood_sugar",
    "fasting_blood_sugar":  "fasting_blood_sugar",
    "fbs":                  "fasting_blood_sugar",

    # ── resting ECG ──────────────────────────────────────────────────────
    "restingrelectro":      "resting_ecg",
    "restingecg":           "resting_ecg",
    "resting ecg":          "resting_ecg",
    "resting_ecg":          "resting_ecg",
    "restecg":              "resting_ecg",

    # ── max heart rate ───────────────────────────────────────────────────
    "maxheartrate":         "max_heart_rate",
    "max heart rate":       "max_heart_rate",
    "max_heart_rate":       "max_heart_rate",
    "thalach":              "max_heart_rate",

    # ── exercise-induced angina ──────────────────────────────────────────
    "exerciseangia":        "exercise_angina",
    "exerciseangina":       "exercise_angina",
    "exercise angina":      "exercise_angina",
    "exercise_angina":      "exercise_angina",
    "exang":                "exercise_angina",

    # ── ST depression (oldpeak) ──────────────────────────────────────────
    "oldpeak":              "oldpeak",

    # ── slope of peak-exercise ST segment ───────────────────────────────
    "slope":                "slope",
    "st slope":             "slope",
    "st_slope":             "slope",

    # ── number of major vessels ──────────────────────────────────────────
    "noofmajorvessels":     "num_major_vessels",
    "noofmajo":             "num_major_vessels",
    "ca":                   "num_major_vessels",
    "num_major_vessels":    "num_major_vessels",

    # ── thalassemia ──────────────────────────────────────────────────────
    "thal":                 "thal",
    "thalassemia":          "thal",

    # ── target ───────────────────────────────────────────────────────────
    "target":               "target",
    "heartdisease":         "target",
    "heart disease":        "target",
    "heart_disease":        "target",
    "output":               "target",
    "diagnosis":            "target",
    "label":                "target",
    "class":                "target",
    "condition":            "target",
    "num":                  "target",
}

# Potential target column names (after aliasing) – first match wins
TARGET_CANDIDATES: list[str] = [
    "target", "heart_disease", "heartdisease", "output",
    "diagnosis", "label", "class", "condition", "num",
]


def _normalise_col(name: str) -> str:
    """Lowercase, strip, collapse whitespace/underscores for lookup."""
    return re.sub(r"[\s_]+", " ", name.strip().lower())


def _alias_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename columns using the alias map.
    Columns not in the map are kept with a normalised (lower, stripped) name.
    Columns mapped to ``__drop__`` are removed.
    """
    rename: dict[str, str] = {}
    drop: list[str] = []

    for col in df.columns:
        key = _normalise_col(col)
        canonical = COLUMN_ALIAS_MAP.get(key, key.replace(" ", "_"))
        if canonical == "__drop__":
            drop.append(col)
        else:
            rename[col] = canonical

    df = df.drop(columns=drop, errors="ignore")
    df = df.rename(columns=rename)
    return df


def _detect_target(df: pd.DataFrame) -> str:
    """Return the target column name, raising ValueError if not found."""
    for candidate in TARGET_CANDIDATES:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not find a target column in dataset. "
        f"Columns present: {df.columns.tolist()}\n"
        f"Add an entry to TARGET_CANDIDATES or rename your target column to 'target'."
    )


def _coerce_target_binary(series: pd.Series) -> pd.Series:
    """
    Make the target strictly 0 / 1.
    Some datasets use >0 = disease; others already use 0/1.
    """
    if series.nunique() == 2 and set(series.dropna().unique()).issubset({0, 1}):
        return series.astype(int)
    # Treat any value > 0 as positive (disease present)
    return (series > 0).astype(int)


# ---------------------------------------------------------------------------
# Main preprocessor class
# ---------------------------------------------------------------------------
class CardiovascularDataPreprocessor:
    """
    Universal heart-disease dataset preprocessor.

    Usage – training
    ----------------
    >>> prep = CardiovascularDataPreprocessor()
    >>> X, y = prep.fit_transform("hospital_a.csv")
    >>> prep.save_state("prep_state.json")

    Usage – inference (same column set as training)
    -----------------------------------------------
    >>> prep = CardiovascularDataPreprocessor.load_state("prep_state.json")
    >>> X, y = prep.transform("hospital_b.csv")   # y may be None

    Usage – multiple datasets (combined training)
    ---------------------------------------------
    >>> prep = CardiovascularDataPreprocessor()
    >>> X1, y1 = prep.fit_transform("ds1.csv")
    >>> X2, y2 = prep.transform("ds2.xlsx")   # aligned to ds1's feature order
    >>> X = np.vstack([X1, X2])
    >>> y = np.concatenate([y1, y2])
    """

    def __init__(self) -> None:
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None   # canonical, ordered
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_transform(
        self,
        source: "str | Path | pd.DataFrame",
        target_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, pd.Series]:
        """
        Fit on this dataset and return (X_scaled, y).

        Parameters
        ----------
        source      : file path (csv / xlsx / xls) or a DataFrame already loaded.
        target_col  : override auto-detection of the target column.
        """
        df = self._load(source)
        df = _alias_columns(df)
        target_col = target_col or _detect_target(df)

        y = _coerce_target_binary(df.pop(target_col))
        X_raw = self._select_features(df, fit=True)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_raw.values).astype(np.float32)
        self._fitted = True
        return X_scaled, y

    def transform(
        self,
        source: "str | Path | pd.DataFrame",
        target_col: Optional[str] = None,
    ) -> Tuple[np.ndarray, Optional[pd.Series]]:
        """
        Transform a new dataset using the already-fitted scaler / feature list.

        Missing features are filled with column mean (computed during fit).
        Extra features that were not seen during fit are ignored.

        Parameters
        ----------
        source      : file path or DataFrame.
        target_col  : override auto-detection; pass ``None`` if no target present.

        Returns
        -------
        X_scaled : np.ndarray
        y        : pd.Series or None
        """
        if not self._fitted:
            raise RuntimeError("Call fit_transform() before transform().")

        df = self._load(source)
        df = _alias_columns(df)

        # Extract target if present
        y: Optional[pd.Series] = None
        if target_col and target_col in df.columns:
            y = _coerce_target_binary(df.pop(target_col))
        else:
            for candidate in TARGET_CANDIDATES:
                if candidate in df.columns:
                    y = _coerce_target_binary(df.pop(candidate))
                    break

        X_raw = self._select_features(df, fit=False)
        X_scaled = self.scaler.transform(X_raw.values).astype(np.float32)
        return X_scaled, y

    def save_state(self, path: "str | Path") -> None:
        """Persist feature list + scaler parameters to a JSON file."""
        if not self._fitted:
            raise RuntimeError("Nothing to save – preprocessor has not been fitted.")
        state = {
            "feature_names": self.feature_names,
            "scaler_mean":   self.scaler.mean_.tolist(),
            "scaler_scale":  self.scaler.scale_.tolist(),
            "col_means":     self._col_means,
        }
        Path(path).write_text(json.dumps(state, indent=2))
        logger.info("Preprocessor state saved → %s", path)

    @classmethod
    def load_state(cls, path: "str | Path") -> "CardiovascularDataPreprocessor":
        """Reconstruct a fitted preprocessor from a saved JSON file."""
        state = json.loads(Path(path).read_text())
        obj = cls()
        obj.feature_names = state["feature_names"]
        obj._col_means     = state["col_means"]

        obj.scaler = StandardScaler()
        obj.scaler.mean_  = np.array(state["scaler_mean"],  dtype=np.float64)
        obj.scaler.scale_ = np.array(state["scaler_scale"], dtype=np.float64)
        obj.scaler.var_   = obj.scaler.scale_ ** 2
        obj.scaler.n_features_in_ = len(obj.feature_names)
        obj._fitted = True
        return obj

    # ------------------------------------------------------------------
    # Convenience: load one or more files and merge
    # ------------------------------------------------------------------

    @classmethod
    def load_and_merge(
        cls,
        *file_paths: "str | Path",
    ) -> Tuple[np.ndarray, pd.Series, "CardiovascularDataPreprocessor"]:
        """
        Load and merge multiple datasets into a single (X, y) pair.

        The *first* file is used to fit the scaler; subsequent files are
        transformed using the same scaler so that the feature space is
        consistent across all datasets.

        Returns
        -------
        X       : np.ndarray  – stacked scaled features
        y       : pd.Series   – stacked targets
        prep    : fitted CardiovascularDataPreprocessor instance
        """
        if not file_paths:
            raise ValueError("Provide at least one file path.")

        prep = cls()
        Xs, ys = [], []

        for i, fp in enumerate(file_paths):
            if i == 0:
                X, y = prep.fit_transform(fp)
            else:
                X, y = prep.transform(fp)
            if y is None:
                raise ValueError(f"Dataset {fp!r} has no target column – cannot use for training.")
            Xs.append(X)
            ys.append(y)

        X_all = np.vstack(Xs)
        y_all = pd.concat(ys, ignore_index=True)
        logger.info(
            "Merged %d datasets → %d samples, %d features",
            len(file_paths), len(y_all), X_all.shape[1],
        )
        return X_all, y_all, prep

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load(source: "str | Path | pd.DataFrame") -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
        raise ValueError(f"Unsupported file format: {suffix!r}")

    def _select_features(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        """
        During *fit*  : discover all numeric columns and store the ordered list.
        During *transform*: align to the stored list (add missing cols as NaN,
                            drop extra cols).
        """
        # Keep only numeric columns
        df = df.select_dtypes(include=[np.number])

        if fit:
            self.feature_names = sorted(df.columns.tolist())
            # Store per-column means for imputing missing cols at transform time
            self._col_means: dict[str, float] = df.mean().to_dict()
            logger.info("Fitted features (%d): %s", len(self.feature_names), self.feature_names)
        else:
            # Add columns present in fit but absent in new dataset
            for col in self.feature_names:
                if col not in df.columns:
                    fill = self._col_means.get(col, 0.0)
                    logger.warning(
                        "Column %r missing in new dataset – filling with training mean %.4f",
                        col, fill,
                    )
                    df[col] = fill
            # Reorder + drop extra columns
            df = df[self.feature_names]

        # Impute missing values with training column means
        for col in df.columns:
            if df[col].isna().any():
                fill = self._col_means.get(col, df[col].mean())
                df[col] = df[col].fillna(fill)

        return df