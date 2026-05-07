# AGENTS.md — F1 Lap Predictor

## Project Overview
Predicts F1 lap times using XGBoost on fastf1 telemetry data. Features include tyre compound, tyre life, stint number, fuel-load proxy, driver identity, and circuit identity.

## Architecture

```
main.py                  → Pipeline orchestrator
src/config.py            → All constants (races, hyperparams, paths)
src/data.py              → fastf1 loading + lap cleaning
src/features.py          → Encoding, stint detection, feature construction
src/models.py            → Train/val split, model training, evaluation
src/viz.py               → Residual plot, SHAP summary plot
```

## Running
```bash
pip install -r requirements.txt
python3 main.py
```

## Key Decisions
- **Validation split by race**, not random — Monaco + Monza are holdout
- **Cache enabled** at `./f1_cache` to avoid re-downloading sessions
- **SHAP sampling** at 500 rows to keep explainability fast
- Lap times filtered to 60–200s to remove outliers

## Conventions
- No docstrings or comments unless adding new logic
- All tunable values live in `src/config.py`
- Functions are pure where possible; side effects (plots, prints) in `viz.py` and `main.py`
- Type hints on all function signatures
- Modules are importable — `main.py` is the only entry point meant for direct execution
