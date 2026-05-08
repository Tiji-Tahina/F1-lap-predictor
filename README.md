# F1 Lap Predictor

Predicts Formula 1 lap times using XGBoost regression on [fastf1](https://github.com/theOehrly/Fast-F1) telemetry data.

## How it works

The pipeline loads race sessions via fastf1, cleans laps (removes pit laps, outliers), engineers features (tyre compound encoding, stint detection, fuel-load proxy via lap fraction), and trains an XGBoost regressor. Monaco and Monza (2023) are held out as a validation set to test generalization to unseen circuits.

### Features used

| Feature | Description |
|---|---|
| `LapNumber` | Current lap of the race |
| `LapFrac` | Normalized lap number within a race (fuel-load proxy) |
| `TyreLife` | Laps on current tyre set |
| `Stint` | Stint number for the driver |
| `Compound_enc` | Encoded tyre compound (e.g. SOFT, MEDIUM, HARD) |
| `Driver_enc` | Encoded driver identifier |
| `GP_enc` | Encoded grand prix identifier |

## Quick start

```bash
pip install -r requirements.txt
python3 main.py
```

Outputs a residual plot (`residuals.png`) and a SHAP summary plot (`shap.png`).

## Project structure

```
main.py                  Pipeline orchestrator
src/
├── config.py            Constants (races, hyperparameters, paths)
├── data.py              fastf1 loading + lap cleaning
├── features.py          Encoding, stint detection, feature construction
├── models.py            Train/val split, model training, evaluation
└── viz.py               Residual plot, SHAP summary plot
```

## Key design decisions

- **Validation split by race**, not random — Monaco + Monza are unseen during training
- **Cache enabled** at `./f1_cache` to avoid re-downloading sessions (configurable in `src/config.py`)
- **SHAP sampling** at 500 rows to keep explainability fast
- Lap times filtered to 60–200s to remove outliers and invalid laps
