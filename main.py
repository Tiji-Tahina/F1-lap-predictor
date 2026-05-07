from src.data import prepare_pipeline
from src.features import build_features
from src.models import build_model, evaluate, train_val_split, train
from src.viz import plot_residuals, plot_shap


def main() -> None:
    # 1. Load & clean
    laps = prepare_pipeline()
    print(f"Clean laps: {len(laps)}")

    # 2. Feature engineering
    X, y, encoders = build_features(laps)

    # 3. Train / validation split (by race)
    X_train, X_val, y_train, y_val = train_val_split(X, y, laps["GP"])
    print(f"Train: {len(X_train)}  |  Val: {len(X_val)}")

    # 4. Train
    model = build_model()
    model = train(model, X_train, y_train, X_val, y_val)

    # 5. Evaluate
    preds, mae, rmse = evaluate(model, X_val, y_val)
    print(f"\nValidation MAE:  {mae:.3f}s")
    print(f"Validation RMSE: {rmse:.3f}s")

    # 6. Visualize
    plot_residuals(y_val, preds)
    plot_shap(model, X_val)
    print("\nSaved: residuals.png, shap.png")


if __name__ == "__main__":
    main()
