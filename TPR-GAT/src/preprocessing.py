"""
preprocessing.py
Sliding window clip generation, ground-truth labeling for all three heads,
player position extraction, and label encoding.

Run standalone:
    python src/preprocessing.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import (
    WINDOW_SEC, FPS, PITCH_LENGTH, PITCH_WIDTH, HALFWAY_X,
    TACTIC_CLASSES, SUGGESTION_LOOKUP, RANDOM_SEED
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_bracket(diff: int) -> str:
    if diff >= 2:  return "winning_2+"
    if diff == 1:  return "winning_1"
    if diff == 0:  return "drawing"
    if diff == -1: return "losing_1"
    return "losing_2+"


def minute_bracket(minute: int) -> str:
    if minute < 30: return "early"
    if minute < 65: return "mid"
    return "late"


# ── HEAD 1: tactic_label ──────────────────────────────────────────────────────

def assign_tactic_label(row) -> str:
    """
    Rule-based labeling in priority order.

    Rule 1 — High Press   : 3+ pressures in opponent half (x > 52.5m)
    Rule 2 — Counter-Attack: ball recovery + 2+ progressive carries (>20m)
    Rule 3 — Low Block    : <=1 pressure + >=70% own-half passes + >=3 passes
    Rule 4 — Build-Up     : default
    """
    opp_p  = row["n_pressures_opp_half"]
    rec    = row["n_ball_recoveries"]
    prog_c = row["n_progressive_carries"]
    tot_p  = row["n_pressures"]
    n_pass = row["n_passes"]
    own_p  = row["n_passes_own_half"]

    if opp_p >= 3:
        return "high_press"
    if rec >= 1 and prog_c >= 2:
        return "counter_attack"
    own_ratio = own_p / max(n_pass, 1)
    if tot_p <= 1 and n_pass >= 3 and own_ratio >= 0.70:
        return "low_block"
    return "build_up"


# ── HEAD 2: adaptation_flag ───────────────────────────────────────────────────

def build_team_baselines(df_clips: pd.DataFrame) -> dict:
    """
    Compute the modal (most common) tactic for each team
    across ALL clips. Used as each team's 'normal' style.
    IMPORTANT: call this on training clips only to avoid leakage.
    """
    baselines = {}
    for team, group in df_clips.groupby("team"):
        baselines[team] = group["tactic_label"].mode()[0]
    return baselines


def assign_adaptation_flag(row, baselines: dict) -> int:
    """
    Returns 1 if this clip's tactic differs from the team's historical baseline
    (they are adapting to the opponent), 0 otherwise.
    """
    modal = baselines.get(row["team"], None)
    if modal is None:
        return 0
    return 1 if row["tactic_label"] != modal else 0


# ── HEAD 3: suggestion_label ──────────────────────────────────────────────────

def assign_suggestion_label(row) -> str:
    """Looks up the recommended tactic based on score state and match phase."""
    sb_ = score_bracket(row["score_differential"])
    mb_ = minute_bracket(row["match_minute"])
    return SUGGESTION_LOOKUP.get((sb_, mb_), "build_up")


# ── Sliding window clip generator ─────────────────────────────────────────────

def get_tactic_position_prior(tactic: str, flag: int, n: int) -> list:
    """
    Fills in players who had no events in the window with realistic
    pitch positions for the given tactic. flag=1 means labeled team.
    """
    noise = 6.0
    if flag == 1:
        centres = {
            "high_press":    [85,80,78,75,70,68,65,62,58,55,50],
            "counter_attack":[90,85,80,75,65,60,55,50,45,35,25],
            "low_block":     [15,20,22,25,28,30,32,35,38,40,42],
            "build_up":      [10,25,35,40,45,50,55,48,42,38,30],
        }.get(tactic, [52.5]*11)
    else:
        centres = {
            "high_press":    [8,15,18,22,25,28,32,35,38,42,45],
            "counter_attack":[12,20,30,40,50,55,60,65,70,75,80],
            "low_block":     [85,80,75,70,65,60,55,50,45,40,35],
            "build_up":      [90,80,70,65,60,55,50,45,40,35,25],
        }.get(tactic, [52.5]*11)

    y_centres = [34,15,55,25,45,34,20,48,30,40,34]
    priors = []
    for i in range(n):
        idx = i % len(centres)
        priors.append({
            "x":  float(np.clip(centres[idx] + np.random.normal(0,noise), 0, PITCH_LENGTH)),
            "y":  float(np.clip(y_centres[idx] + np.random.normal(0,noise), 0, PITCH_WIDTH)),
            "vx": float(np.random.normal(0, 0.5)),
            "vy": float(np.random.normal(0, 0.5)),
            "team_flag": float(flag)
        })
    return priors


def extract_player_positions(clip_row, df_events_all: pd.DataFrame,
                              n_players: int = 11) -> list:
    """
    Extracts approximate player positions from StatsBomb events within
    the clip's time window. For players with events: use actual location.
    For players without events: use tactic-informed position prior.
    Returns list of 22 dicts: {x, y, vx, vy, team_flag}.
    """
    team     = clip_row["team"]
    opponent = clip_row["opponent"]
    match_id = clip_row["match_id"]
    start_s  = clip_row["start_second"]
    end_s    = clip_row["end_second"]
    tactic   = clip_row["tactic_label"]

    mask = (
        (df_events_all["match_id"] == match_id) &
        (df_events_all["total_seconds"] >= start_s) &
        (df_events_all["total_seconds"] <  end_s)
    )
    window_ev = df_events_all[mask].copy()

    def positions_for_team(team_name, flag):
        team_ev = window_ev[
            (window_ev["team"] == team_name) &
            (window_ev["location"].notna())
        ]
        located = []
        for _, ev in team_ev.iterrows():
            loc = ev["location"]
            if isinstance(loc, list) and len(loc) == 2:
                x = float(loc[0]) * (PITCH_LENGTH / 120.0)
                y = float(loc[1]) * (PITCH_WIDTH  / 80.0)
                vx, vy = 0.0, 0.0
                if ev["type"] == "Carry" and isinstance(ev.get("carry"), dict):
                    el = ev["carry"].get("end_location")
                    dur = float(ev.get("duration", 1.0) or 1.0)
                    if el and isinstance(el, list):
                        vx = float(np.clip((el[0]*(PITCH_LENGTH/120)-x)/dur, -10, 10))
                        vy = float(np.clip((el[1]*(PITCH_WIDTH/80) -y)/dur, -10, 10))
                located.append({
                    "x": float(np.clip(x, 0, PITCH_LENGTH)),
                    "y": float(np.clip(y, 0, PITCH_WIDTH)),
                    "vx": vx, "vy": vy, "team_flag": float(flag)
                })
        # deduplicate by rounded metre
        seen, unique = set(), []
        for p in located:
            k = (round(p["x"]), round(p["y"]))
            if k not in seen:
                seen.add(k); unique.append(p)
        unique = unique[:n_players]
        needed = n_players - len(unique)
        if needed > 0:
            unique.extend(get_tactic_position_prior(tactic, flag, needed))
        return unique[:n_players]

    return positions_for_team(team, 1) + positions_for_team(opponent, 0)


def generate_clips(df_events_match: pd.DataFrame, match_id: int,
                   home_team: str, away_team: str) -> list:
    """
    Cut one match into 15-second non-overlapping windows.
    Returns list of clip dicts (two per window, one per team).
    """
    clips = []
    df = df_events_match.copy()
    df["total_seconds"] = df["minute"] * 60 + df["second"].fillna(0)
    max_s = int(df["total_seconds"].max())
    clip_idx = 0

    for start_s in range(0, max_s - WINDOW_SEC, WINDOW_SEC):
        end_s    = start_s + WINDOW_SEC
        window   = df[(df["total_seconds"] >= start_s) &
                      (df["total_seconds"] <  end_s)]
        if len(window) < 3:
            continue

        prev = df[df["total_seconds"] < start_s]
        home_g = len(prev[(prev["team"] == home_team) &
                           (prev["type"] == "Shot") &
                           (prev["shot_outcome"] == "Goal")])
        away_g = len(prev[(prev["team"] == away_team) &
                           (prev["type"] == "Shot") &
                           (prev["shot_outcome"] == "Goal")])
        minute = start_s // 60

        for team in [home_team, away_team]:
            opp        = away_team if team == home_team else home_team
            score_diff = (home_g - away_g) if team == home_team else (away_g - home_g)
            te         = window[window["team"] == team]

            press  = te[te["type"] == "Pressure"]
            passes = te[te["type"] == "Pass"]
            carries= te[te["type"] == "Carry"]
            recs   = te[te["type"] == "Ball Recovery"]

            opp_half_p = sum(
                1 for loc in press["location"].dropna()
                if isinstance(loc, list) and loc[0] > HALFWAY_X
            )
            prog_c = 0
            for _, crow in carries.iterrows():
                try:
                    e = crow["carry_end_location"]
                    s = crow["location"]
                    if e and s and np.sqrt((e[0]-s[0])**2+(e[1]-s[1])**2) > 20:
                        prog_c += 1
                except Exception:
                    pass

            own_p = sum(
                1 for loc in passes["location"].dropna()
                if isinstance(loc, list) and loc[0] < HALFWAY_X
            )

            clips.append({
                "clip_id":               f"M{match_id}_C{clip_idx:04d}_{team[:3].upper()}",
                "match_id":              match_id,
                "match_minute":          minute,
                "start_second":          start_s,
                "end_second":            end_s,
                "start_frame":           start_s * FPS,
                "end_frame":             end_s   * FPS,
                "window_seconds":        WINDOW_SEC,
                "team":                  team,
                "opponent":              opp,
                "home_team":             home_team,
                "away_team":             away_team,
                "team_is_home":          int(team == home_team),
                "home_goals":            home_g,
                "away_goals":            away_g,
                "score_differential":    score_diff,
                "n_events_team":         len(te),
                "n_pressures":           len(press),
                "n_pressures_opp_half":  opp_half_p,
                "n_passes":              len(passes),
                "n_passes_own_half":     own_p,
                "n_carries":             len(carries),
                "n_progressive_carries": prog_c,
                "n_ball_recoveries":     len(recs),
                "n_clearances":          len(te[te["type"] == "Clearance"]),
                "n_dribbles":            len(te[te["type"] == "Dribble"]),
            })
        clip_idx += 1

    return clips


def build_dataset(df_matches: pd.DataFrame, df_events: pd.DataFrame):
    """
    Full pipeline: clips -> label all three heads -> encode.
    Returns (df_clips, label_encoder, train_ids, val_ids, test_ids).
    """
    # 1. Generate clips
    all_clips = []
    for _, row in df_matches.iterrows():
        mid  = int(row["match_id"])
        home = row["home_team"]
        away = row["away_team"]
        comp = row["competition_name"]
        mev  = df_events[df_events["match_id"] == mid]
        if len(mev) < 50:
            continue
        clips = generate_clips(mev, mid, home, away)
        for c in clips:
            c["competition"] = comp
        all_clips.extend(clips)
        print(f"  {home} vs {away}  ->  {len(clips)} clips")

    df_clips = pd.DataFrame(all_clips)
    df_clips = df_clips[df_clips["n_events_team"] > 0].reset_index(drop=True)

    # 2. HEAD 1 — tactic_label
    df_clips["tactic_label"] = df_clips.apply(assign_tactic_label, axis=1)
    df_clips["score_bracket"]  = df_clips["score_differential"].apply(score_bracket)
    df_clips["minute_bracket"] = df_clips["match_minute"].apply(minute_bracket)

    # 3. Match split (by match to prevent leakage)
    all_ids = df_clips["match_id"].unique().copy()
    np.random.seed(RANDOM_SEED)
    np.random.shuffle(all_ids)
    n = len(all_ids)
    n_test = max(1, int(n * 0.15))
    n_val  = max(1, int(n * 0.15))
    train_ids = set(all_ids[:n-n_test-n_val])
    val_ids   = set(all_ids[n-n_test-n_val:n-n_test])
    test_ids  = set(all_ids[n-n_test:])

    # 4. HEAD 2 — adaptation_flag (from training data only)
    df_train = df_clips[df_clips["match_id"].isin(train_ids)]
    baselines = build_team_baselines(df_train)
    df_clips["adaptation_flag"] = df_clips.apply(
        lambda r: assign_adaptation_flag(r, baselines), axis=1
    )

    # 5. HEAD 3 — suggestion_label
    df_clips["suggestion_label"] = df_clips.apply(assign_suggestion_label, axis=1)

    # 6. Encode labels
    le = LabelEncoder().fit(TACTIC_CLASSES)
    df_clips["tactic_label_enc"]     = le.transform(df_clips["tactic_label"])
    df_clips["suggestion_label_enc"] = le.transform(df_clips["suggestion_label"])

    print(f"\nDataset: {len(df_clips):,} clips")
    print(df_clips["tactic_label"].value_counts().to_string())

    return df_clips, le, train_ids, val_ids, test_ids


def add_player_positions(df_clips: pd.DataFrame, df_events: pd.DataFrame):
    """
    Extract player positions for every clip and add as a new column.
    This is the most time-consuming step (~2 min for 240k clips).
    """
    if "total_seconds" not in df_events.columns:
        df_events["total_seconds"] = (
            df_events["minute"] * 60 +
            df_events["second"].fillna(0).astype(float)
        )

    all_pos, failed = [], 0
    for idx, row in df_clips.iterrows():
        try:
            pos = extract_player_positions(row, df_events, n_players=11)
            while len(pos) < 22:
                pos.append({"x": PITCH_LENGTH/2, "y": PITCH_WIDTH/2,
                             "vx": 0.0, "vy": 0.0,
                             "team_flag": float(len(pos) < 11)})
            all_pos.append(pos[:22])
        except Exception:
            failed += 1
            fallback = [{"x": PITCH_LENGTH/2 + np.random.normal(0,5),
                         "y": PITCH_WIDTH/2  + np.random.normal(0,5),
                         "vx": 0.0, "vy": 0.0, "team_flag": float(i<11)}
                        for i in range(22)]
            all_pos.append(fallback)
        if (idx + 1) % 5000 == 0:
            print(f"  Positions: {idx+1:,} / {len(df_clips):,}")

    df_clips = df_clips.copy()
    df_clips["player_positions"] = all_pos
    print(f"Positions done. Failed (fallback): {failed}")
    return df_clips


if __name__ == "__main__":
    from src.data_loader import load_all_events
    df_matches, df_events = load_all_events()
    df_clips, le, train_ids, val_ids, test_ids = build_dataset(df_matches, df_events)
    print(df_clips[["clip_id","team","tactic_label",
                    "adaptation_flag","suggestion_label"]].head(5).to_string(index=False))
