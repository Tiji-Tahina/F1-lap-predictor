import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import FEATURES


def encode_features(laps: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    encoders: dict[str, LabelEncoder] = {}

    le_driver = LabelEncoder()
    laps["Driver_enc"] = le_driver.fit_transform(laps["Driver"])
    encoders["Driver"] = le_driver

    le_compound = LabelEncoder()
    laps["Compound_enc"] = le_compound.fit_transform(laps["Compound"])
    encoders["Compound"] = le_compound

    le_gp = LabelEncoder()
    laps["GP_enc"] = le_gp.fit_transform(laps["GP"])
    encoders["GP"] = le_gp

    return laps, encoders


def add_stint(laps: pd.DataFrame) -> pd.DataFrame:
    laps = laps.sort_values(["GP", "Driver", "LapNumber"])
    laps["Stint"] = (
        laps.groupby(["GP", "Driver"])["TyreLife"]
        .transform(lambda s: (s.diff() < 0).cumsum() + 1)
    )
    return laps


def add_lap_frac(laps: pd.DataFrame) -> pd.DataFrame:
    laps["LapFrac"] = laps.groupby(["GP"])["LapNumber"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min() + 1e-6)
    )
    return laps


def build_features(laps: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, dict]:
    laps = add_stint(laps)
    laps = add_lap_frac(laps)
    laps, encoders = encode_features(laps)

    X = laps[FEATURES]
    y = laps["LapTime_s"]

    return X, y, encoders
