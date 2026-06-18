"""
data_loader.py — StatsBomb open data download.

Public API
----------
load_matches()  -> pd.DataFrame   all match records for TARGET_COMPETITIONS
load_events()   -> pd.DataFrame   all events for every match in df_matches
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from statsbombpy import sb

from config import TARGET_COMPETITIONS


def load_matches() -> pd.DataFrame:
    """Download match records for every competition in config."""
    all_matches = []
    for comp in TARGET_COMPETITIONS:
        matches = sb.matches(
            competition_id=comp["competition_id"],
            season_id=comp["season_id"],
        )
        matches["competition_name"] = comp["name"]
        all_matches.append(matches)
        print(f"  {comp['name']:<35} -> {len(matches)} matches")

    df = pd.concat(all_matches, ignore_index=True)
    print(f"\nTotal matches loaded: {len(df)}")
    return df


def load_events(df_matches: pd.DataFrame) -> pd.DataFrame:
    """Download all events for every match. Prints OK / FAIL per match."""
    all_events, failed = [], []

    for _, row in df_matches.iterrows():
        mid  = int(row["match_id"])
        home = row["home_team"]
        away = row["away_team"]
        comp = row["competition_name"]
        try:
            ev = sb.events(match_id=mid)
            ev["match_id"]         = mid
            ev["competition_name"] = comp
            ev["home_team"]        = home
            ev["away_team"]        = away
            all_events.append(ev)
            print(f"  OK   {mid}  {home} vs {away}  ({len(ev)} events)")
        except Exception as e:
            failed.append(mid)
            print(f"  FAIL {mid}: {e}")

    df = pd.concat(all_events, ignore_index=True)
    df["total_seconds"] = (
        df["minute"] * 60 + df["second"].fillna(0).astype(float)
    )

    print(f"\nTotal events : {len(df):,}")
    print(f"Matches      : {df['match_id'].nunique()}")
    print(f"Failed       : {len(failed)}")
    return df


if __name__ == "__main__":
    print("=== Loading matches ===")
    df_m = load_matches()
    print("\n=== Loading events ===")
    df_e = load_events(df_m)
    print(df_e.head(3))
