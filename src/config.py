# ── Paths & cache ─────────────────────────────────────────────────────────────
CACHE_DIR = "./f1-cache"

# ── Races to load ─────────────────────────────────────────────────────────────
RACES = [
    (2023, "Bahrain"),
    (2023, "Australia"),
    (2023, "Monaco"),
    (2023, "Spain"),
    (2023, "Silverstone"),
    (2023, "Monza"),
]

# ── Train / validation split ─────────────────────────────────────────────────
VAL_GP = ["Monaco", "Monza"]

# ── Feature columns ───────────────────────────────────────────────────────────
FEATURES = [
    "LapNumber",
    "LapFrac",
    "TyreLife",
    "Stint",
    "Compound_enc",
    "Driver_enc",
    "GP_enc",
]

# ── Key columns required for cleaning ─────────────────────────────────────────
KEY_COLS = ["LapTime_s", "Compound", "TyreLife", "LapNumber", "Driver"]

# ── Lap-time validity bounds (seconds) ────────────────────────────────────────
LAP_TIME_MIN = 60
LAP_TIME_MAX = 200

# ── XGBoost hype
# 
# parameters ──────────────────────────────────────────────────
MODEL_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1,
}

# ── Plot settings ────────────────────────────────────────────────────────────
DPI = 150
SHAP_SAMPLE_SIZE = 500
