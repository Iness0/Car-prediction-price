import re
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TRAIN_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_train.csv"
TEST_URL = "https://raw.githubusercontent.com/Murcha1990/MLDS_ML_2022/main/Hometasks/HT1/cars_test.csv"


def _to_float(series: pd.Series) -> pd.Series:
    """Extract first numeric token from a string column."""
    extracted = series.astype(str).str.extract(r"([\d\.]+)", expand=False)
    return pd.to_numeric(extracted, errors="coerce")


def _parse_torque(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Parse torque strings into torque (Nm) and rpm."""

    def convert(val):
        if pd.isna(val):
            return pd.Series({"torque": np.nan, "max_torque_rpm": np.nan})
        s = str(val).lower().replace(",", "")
        numbers = re.findall(r"[0-9]+\.?[0-9]*", s)
        if not numbers:
            return pd.Series({"torque": np.nan, "max_torque_rpm": np.nan})
        torque_val = float(numbers[0])
        if "kgm" in s:
            torque_val *= 9.80665
        rpm_vals = [float(n) for n in numbers[1:]]
        rpm_val = float(np.mean(rpm_vals)) if rpm_vals else np.nan
        return pd.Series({"torque": torque_val, "max_torque_rpm": rpm_val})

    parsed = series.apply(convert)
    return parsed["torque"], parsed["max_torque_rpm"]


def split_name(series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    words = series.fillna("").str.split()
    brand = words.str[0].replace("", "unknown")
    model = words.str[1].replace("", "unknown")
    return brand, model


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lightweight preprocessing: parse strings, add FE, keep consistent schema."""
    df = df.copy()

    numeric_text_cols = ["mileage", "engine", "max_power"]
    for col in numeric_text_cols:
        if col in df.columns:
            df[col] = _to_float(df[col])

    if "torque" in df.columns:
        df["torque"], df["max_torque_rpm"] = _parse_torque(df["torque"])

    if "name" in df.columns:
        df["brand"], df["model"] = split_name(df["name"])

    for col in ["engine", "seats", "year", "km_driven"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "year" in df.columns:
        df["car_age"] = 2024 - df["year"]
    if {"max_power", "engine"}.issubset(df.columns):
        df["power_per_engine"] = df["max_power"] / df["engine"]
    if {"km_driven", "car_age"}.issubset(df.columns):
        df["km_per_year"] = df["km_driven"] / df["car_age"].clip(lower=1)

    return df


def make_pipeline(scale_cat: bool = False) -> Pipeline:
    """Build a full sklearn Pipeline with preprocessing + Ridge model.

    Parameters
    ----------
    scale_cat : bool
        If True, applies StandardScaler(with_mean=False) to OHE outputs
        (useful to shrink large category weights).
    """
    numeric_features = [
        "year",
        "km_driven",
        "mileage",
        "engine",
        "max_power",
        "torque",
        "max_torque_rpm",
        "car_age",
        "power_per_engine",
        "km_per_year",
    ]
    categorical_features = [
        "brand",
        "model",
        "fuel",
        "seller_type",
        "transmission",
        "owner",
        "seats",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("ohe", OneHotEncoder(handle_unknown="ignore", drop="first")),
                        *(
                            [("cat_scaler", StandardScaler(with_mean=False))]
                            if scale_cat
                            else []
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    model = Ridge(alpha=0.1)

    pipeline = Pipeline(
        steps=[
            ("build", FunctionTransformer(build_features, validate=False)),
            ("preprocess", preprocess),
            ("model", model),
        ]
    )
    return pipeline
