import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.config import DPI, FEATURES, SHAP_SAMPLE_SIZE


def plot_residuals(
    y_val: pd.Series,
    preds: pd.Series,
    path: str = "residuals.png",
) -> None:
    plt.figure(figsize=(8, 4))
    plt.scatter(y_val, preds, alpha=0.3, s=8, color="steelblue")
    plt.plot(
        [y_val.min(), y_val.max()],
        [y_val.min(), y_val.max()],
        "r--",
        lw=1,
    )
    plt.xlabel("Actual lap time (s)")
    plt.ylabel("Predicted lap time (s)")
    plt.title("Predicted vs Actual — validation races")
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()


def plot_shap(
    model,
    X_val: pd.DataFrame,
    path: str = "shap.png",
) -> None:
    sample = X_val.iloc[:SHAP_SAMPLE_SIZE]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(
        shap_values, sample, feature_names=FEATURES, show=False
    )
    plt.tight_layout()
    plt.savefig(path, dpi=DPI)
    plt.close()
