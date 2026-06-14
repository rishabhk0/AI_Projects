"""
data_loader.py
Loads match records and events from StatsBomb open data.

Run standalone:
    python src/data_loader.py
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from statsbombpy import sb
from src.config import TARGET_COMPETITIONS


def load_matches() -> pd.DataFrame:
    """Download match records for all three target competitions."""
    all_matches = []
    for comp in TARGET_COMPETITIONS:
        matches = sb.matches(
            competition_id=comp["competition_id"],
            season_id=comp["season_id"]
        )
        matches["competition_name"] = comp["name"]
        all_matches.append(matches)
        print(f"  {comp['name']:<35} -> {len(matches)} matches")

    df_matches = pd.concat(all_matches, ignore_index=True)
    print(f"\nTotal matches: {len(df_matches)}")
    return df_matches


def load_events(df_matches: pd.DataFrame) -> pd.DataFrame:
    """
    Download every event for every match in df_matches.
    Adds total_seconds column for time-window filtering.
    Returns the combined events DataFrame.
    """
    all_events = []
    failed     = []

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

    df_events = pd.concat(all_events, ignore_index=True)

    # Convert (minute, second) to a single total_seconds float
    df_events["total_seconds"] = (
        df_events["minute"] * 60 +
        df_events["second"].fillna(0).astype(float)
    )

    print(f"\nTotal events : {len(df_events):,}")
    print(f"Matches      : {df_events['match_id'].nunique()}")
    print(f"Teams        : {df_events['team'].nunique()}")
    print(f"Failed       : {len(failed)}")
    return df_events


def load_all_events():
    """Convenience wrapper: returns (df_matches, df_events)."""
    print("Loading matches...")
    df_matches = load_matches()
    print("\nLoading events (takes several minutes)...")
    df_events  = load_events(df_matches)
    return df_matches, df_events


if __name__ == "__main__":
    df_matches, df_events = load_all_events()
    print(df_matches[["match_id", "competition_name",
                       "home_team", "away_team"]].head(5).to_string(index=False))
