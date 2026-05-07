import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from xgboost import XGBRegressor

from src.config import MODEL_PARAMS, VAL_GP


def train_val_split(
    X: pd.DataFrame,
    y: pd.Series,
    gps: pd.Series,
    val_gps: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    val_gps = val_gps or VAL_GP
    train_mask = ~gps.isin(val_gps)
    val_mask = gps.isin(val_gps)

    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]

    return X_train, X_val, y_train, y_val


def build_model() -> XGBRegressor:
    return XGBRegressor(**MODEL_PARAMS)


def train(
    model: XGBRegressor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> XGBRegressor:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )
    return model


def evaluate(
    model: XGBRegressor,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> tuple[pd.Series, float, float]:
    preds = model.predict(X_val)
    mae = mean_absolute_error(y_val, preds)
    rmse = root_mean_squared_error(y_val, preds)
    return preds, mae, rmse
