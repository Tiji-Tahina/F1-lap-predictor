# F1 Lap Predictor — Codebase Guide

## What does this project do?

It predicts Formula 1 lap times using historical telemetry data from the `fastf1` library. An XGBoost regressor is trained on 4 races from the 2023 season (Bahrain, Australia, Spain, Silverstone) and validated on 2 unseen circuits (Monaco, Monza).

---

## How does data flow through the pipeline?

```
fastf1 API
    │
    ▼
load_race()          ── src/data.py
    │  Downloads session data (cached to disk)
    ▼
clean_laps()         ── src/data.py
    │  Removes pit laps, outliers, NaN rows
    ▼
add_stint()          ── src/features.py
    │  Detects when a driver changes tyres
    ▼
add_lap_frac()       ── src/features.py
    │  Creates a fuel-load proxy (0 → 1 across race)
    ▼
encode_features()    ── src/features.py
    │  Label-encodes Driver, Compound, GP
    ▼
train_val_split()    ── src/models.py
    │  Splits by race name (not random)
    ▼
build_model() + train()  ── src/models.py
    │  XGBoost with early stopping on validation set
    ▼
evaluate()           ── src/models.py
    │  MAE and RMSE on Monaco + Monza
    ▼
plot_residuals() + plot_shap()  ── src/viz.py
    │  Saves residuals.png + shap.png
```

---

## What does each module do?

### `src/config.py` — All knobs in one place

| Constant | What it controls |
|---|---|
| `CACHE_DIR` | Where fastf1 stores downloaded sessions |
| `RACES` | Which (year, grand-prix) pairs to load |
| `VAL_GP` | Which races are held out for validation |
| `FEATURES` | Column names used as X inputs |
| `KEY_COLS` | Columns that must not be NaN |
| `LAP_TIME_MIN / MAX` | Outlier filter bounds (seconds) |
| `MODEL_PARAMS` | XGBoost hyperparameters |
| `SHAP_SAMPLE_SIZE` | How many rows to use for SHAP (speed) |

### `src/data.py` — Loading and cleaning

- **`load_race(year, gp)`**: Calls `fastf1.get_session()`, loads the race session, extracts laps, tags them with GP name and year.
- **`load_all_races(races)`**: Concatenates multiple races into one DataFrame.
- **`clean_laps(laps)`**: Drops rows that are not personal bests, have pit-in/pit-out times, have lap times outside 60–200s, or have NaN in key columns.
- **`prepare_pipeline()`**: Enables the cache, runs load + clean.

### `src/features.py` — Turning raw laps into ML features

- **`add_stint(laps)`**: Sorts by GP → Driver → LapNumber, then detects tyre changes: whenever `TyreLife` drops (fresh tyres), a new stint begins. Stints are numbered 1, 2, 3, ...
- **`add_lap_frac(laps)`**: Normalizes LapNumber within each GP to [0, 1]. This acts as a proxy for fuel load (more fuel early → slower laps).
- **`encode_features(laps)`**: Uses `sklearn.preprocessing.LabelEncoder` to turn Driver, Compound, and GP strings into integers.
- **`build_features(laps)`**: Runs all three steps above, returns `X` (feature matrix), `y` (target: LapTime_s), and `encoders` (for later inverse-transform if needed).

### `src/models.py` — Train, validate, evaluate

- **`train_val_split(X, y, gps)`**: Splits data so that entire races are held out. The default holdout races are Monaco and Monza. All other races go to training.
- **`build_model()`**: Creates an `XGBRegressor` with parameters from `config.MODEL_PARAMS`.
- **`train(model, X_train, y_train, X_val, y_val)`**: Calls `model.fit()` with `eval_set` so XGBoost can print validation error every 50 rounds. (Early stopping is not explicitly set here — the model trains for the full `n_estimators` rounds.)
- **`evaluate(model, X_val, y_val)`**: Predicts on the validation set, returns predictions + MAE + RMSE.

### `src/viz.py` — Visual diagnostics

- **`plot_residuals(y_val, preds)`**: Scatter plot of actual vs predicted lap times. A perfect model would land all points on the diagonal red line.
- **`plot_shap(model, X_val)`**: Takes the first `SHAP_SAMPLE_SIZE` rows of the validation set, runs a TreeExplainer, and saves a SHAP summary plot. This shows which features matter most and how they affect predictions.

---

## Why validate by race instead of random split?

If you split randomly, the model could see laps from Monaco during training and also be tested on Monaco laps. That tells you how well the model memorizes — not how well it generalizes to new circuits. By holding out entire GPs, you get a realistic measure of performance on tracks the model has never seen.

---

## How does stint detection work?

```python
laps["Stint"] = (
    laps.groupby(["GP", "Driver"])["TyreLife"]
    .transform(lambda s: (s.diff() < 0).cumsum() + 1)
)
```

For each driver in each race, look at `TyreLife` lap-over-lap. When it drops (e.g. from 20 → 1), the driver pitted for fresh tyres. `s.diff() < 0` is True at that row. `cumsum()` counts how many times this has happened. `+ 1` makes stints start at 1.

---

## What is LapFrac?

It's a simple fuel-load proxy:

```python
laps["LapFrac"] = (LapNumber - min_LapNumber) / (max_LapNumber - min_LapNumber + 1e-6)
```

Early laps → low LapFrac (heavy fuel, slower). Late laps → high LapFrac (light fuel, faster). Adding `1e-6` prevents division by zero in short races.

---

## How to add a new race to the dataset?

Edit `src/config.py`:

```python
RACES = [
    (2023, "Bahrain"),
    (2023, "Australia"),
    (2023, "Monaco"),
    (2023, "Spain"),
    (2023, "Silverstone"),
    (2023, "Monza"),
    (2024, "Suzuka"),       # new
]
```

If you want to use it for validation instead of training, add "Suzuka" to `VAL_GP`.

---

## How to tune hyperparameters?

Edit `MODEL_PARAMS` in `src/config.py`. The current values:

```python
MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}
```

You could grid-search `max_depth`, `learning_rate`, and `subsample`, but the validation set is small (2 races), so be careful not to overfit to the validation split.

---

## How to understand the SHAP plot?

After running `main.py`, open `shap.png`. Each row is a feature. The horizontal position shows SHAP value (impact on prediction). Color shows feature value (red = high, blue = low). Features are sorted by importance.

For example, if `Compound_enc` is at the top with a wide spread, tyre compound is the strongest predictor of lap time. If high values (red) push SHAP values to the right, certain compounds are systematically slower.

---

## Conventions used in this codebase

| Rule | Why |
|---|---|
| No docstrings or comments unless adding new logic | Keeps code concise |
| All tunable values in `config.py` | Single source of truth |
| Pure functions where possible | Easier to test and reason about |
| Type hints on every function | Self-documenting signatures |
| Only `main.py` is meant to run directly | Modules stay importable |
