

import fastf1
import pandas as pd 

from src.config import CACHE_DIR, KEY_COLS, LAP_TIME_MAX, LAP_TIME_MIN


def load_race(year: int, gp: str) -> pd.DataFrame:
    session = fastf1.get_session(year, gp, "R")
    session.load()
    laps = session.laps.copy()
    laps["GP"] = gp
    laps["Year"] = year
    return laps



def load_all_races(races: list[tuple[int, str]]) -> pd.DataFrame:
    return pd.concat(
        [load_race(y, gp) for y, gp in races], ignore_index=True
    )



def clean_laps(all_laps: pd.DataFrame) -> pd.DataFrame:
    laps = all_laps.copy()

    laps = laps[laps["IsPersonalBest"] | laps["LapTime"].notna()]
    laps = laps[~laps["PitInTime"].notna()]
    laps = laps[~laps["PitOutTime"].notna()]

    laps["LapTime_s"] = laps["LapTime"].dt.total_seconds()

    laps = laps[
        (laps["LapTime_s"] > LAP_TIME_MIN)
        & (laps["LapTime_s"] < LAP_TIME_MAX)
    ]

    laps = laps.dropna(subset=KEY_COLS)

    return laps


def prepare_pipeline() -> pd.DataFrame:
    fastf1.Cache.enable_cache(CACHE_DIR)
    all_laps = load_all_races(
        [(2023, "Bahrain"), (2023, "Australia"), (2023, "Monaco"),
         (2023, "Spain"), (2023, "Silverstone"), (2023, "Monza")]
    )
    return clean_laps(all_laps)
