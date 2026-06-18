"""
preprocessing.py — Sliding windows, tactic labelling, player position extraction.

Public API
----------
generate_clips(df_events_match, match_id, home_team, away_team) -> list[dict]
build_clips_dataframe(df_matches, df_events)                     -> pd.DataFrame
assign_tactic_label(row)                                         -> str
build_team_baselines(df)                                         -> dict
assign_adaptation_flag(row, baselines)                           -> int
score_bracket(diff)                                              -> str
minute_bracket(minute)                                           -> str
assign_suggestion_label(row)                                     -> str
apply_all_labels(df_clips, train_match_ids)                      -> pd.DataFrame
extract_all_positions(df_clips, df_events)                       -> pd.DataFrame
run_label_unit_tests()
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import (
    WINDOW_SEC, FPS, HALFWAY_X, PITCH_LENGTH, PITCH_WIDTH,
    SUGGESTION_LOOKUP, TACTIC_CLASSES, RANDOM_SEED,
    TARGET_PER_CLASS, BUILD_UP_CAP,
)


# ══════════════════════════════════════════════════════════════════════════════
# SLIDING WINDOW CLIP GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_clips(df_events_match, match_id, home_team, away_team):
    """
    Slice one match into WINDOW_SEC-second non-overlapping windows.
    Returns one clip dict per (window, team) pair.
    """
    clips    = []
    df       = df_events_match.copy()
    df["total_seconds"] = df["minute"] * 60 + df["second"].fillna(0)
    max_second = int(df["total_seconds"].max())
    clip_idx   = 0

    for start_s in range(0, max_second - WINDOW_SEC, WINDOW_SEC):
        end_s     = start_s + WINDOW_SEC
        window_ev = df[(df["total_seconds"] >= start_s) &
                       (df["total_seconds"] <  end_s)]
        if len(window_ev) < 3:
            continue

        # Score at window start — count shots labelled Goal before this window
        prev   = df[df["total_seconds"] < start_s]
        home_g = len(prev[(prev["team"] == home_team) &
                           (prev["type"] == "Shot") &
                           (prev["shot_outcome"] == "Goal")])
        away_g = len(prev[(prev["team"] == away_team) &
                           (prev["type"] == "Shot") &
                           (prev["shot_outcome"] == "Goal")])
        minute = start_s // 60

        for team in [home_team, away_team]:
            opponent   = away_team if team == home_team else home_team
            score_diff = (home_g - away_g) if team == home_team else (away_g - home_g)
            te         = window_ev[window_ev["team"] == team]

            pressures_te  = te[te["type"] == "Pressure"]
            passes_te     = te[te["type"] == "Pass"]
            carries_te    = te[te["type"] == "Carry"]
            recoveries_te = te[te["type"] == "Ball Recovery"]

            # Pressures in opponent half (x > HALFWAY_X in StatsBomb 120-unit space)
            opp_half_p = sum(
                1 for loc in pressures_te["location"].dropna()
                if isinstance(loc, list) and loc[0] > HALFWAY_X
            )

            # Progressive carries: end location > 20 m further from start
            prog_carries = 0
            for _, crow in carries_te.iterrows():
                try:
                    end   = crow["carry_end_location"]
                    start = crow["location"]
                    if end and start:
                        if np.sqrt((end[0] - start[0]) ** 2 +
                                   (end[1] - start[1]) ** 2) > 20:
                            prog_carries += 1
                except Exception:
                    pass

            # Passes from own half
            own_half_p = sum(
                1 for loc in passes_te["location"].dropna()
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
                "team":                  team,
                "opponent":              opponent,
                "home_team":             home_team,
                "away_team":             away_team,
                "team_is_home":          int(team == home_team),
                "home_goals":            home_g,
                "away_goals":            away_g,
                "score_differential":    score_diff,
                "n_events_team":         len(te),
                "n_pressures":           len(pressures_te),
                "n_pressures_opp_half":  opp_half_p,
                "n_passes":              len(passes_te),
                "n_passes_own_half":     own_half_p,
                "n_carries":             len(carries_te),
                "n_progressive_carries": prog_carries,
                "n_ball_recoveries":     len(recoveries_te),
                "n_clearances":          len(te[te["type"] == "Clearance"]),
                "n_dribbles":            len(te[te["type"] == "Dribble"]),
            })
        clip_idx += 1

    return clips


def build_clips_dataframe(df_matches, df_events):
    """Run generate_clips over all matches and return combined DataFrame."""
    all_clips = []
    for _, row in df_matches.iterrows():
        mid  = int(row["match_id"])
        home = row["home_team"]
        away = row["away_team"]
        comp = row["competition_name"]
        match_events = df_events[df_events["match_id"] == mid].copy()
        if len(match_events) < 50:
            print(f"  SKIP {mid} — too few events")
            continue
        clips = generate_clips(match_events, mid, home, away)
        for c in clips:
            c["competition"] = comp
        all_clips.extend(clips)
        print(f"  {mid}  {home} vs {away}  ->  {len(clips)} clips")

    df = pd.DataFrame(all_clips)
    df = df[df["n_events_team"] > 0].reset_index(drop=True)
    print(f"\nTotal clips: {len(df):,}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# LABELLING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def assign_tactic_label(row) -> str:
    """
    HEAD 1 ground truth — priority-ordered rules.

    Rule 1 — high_press    : >= 2 pressures in opponent half
    Rule 2 — counter_attack: >= 1 ball recovery AND >= 1 progressive carry
    Rule 3 — low_block     : <= 1 total pressures, >= 3 passes, >= 70% own half
    Rule 4 — build_up      : default
    """
    opp_p  = row["n_pressures_opp_half"]
    rec    = row["n_ball_recoveries"]
    prog_c = row["n_progressive_carries"]
    tot_p  = row["n_pressures"]
    n_pass = row["n_passes"]
    own_p  = row["n_passes_own_half"]

    if opp_p >= 2:
        return "high_press"
    if rec >= 1 and prog_c >= 1:
        return "counter_attack"
    own_ratio = own_p / max(n_pass, 1)
    if tot_p <= 1 and n_pass >= 3 and own_ratio >= 0.70:
        return "low_block"
    return "build_up"


def build_team_baselines(df: pd.DataFrame) -> dict:
    """
    HEAD 2 helper — compute each team's modal tactic.
    Call on TRAINING DATA ONLY to prevent leakage.
    """
    return {
        team: group["tactic_label"].mode()[0]
        for team, group in df.groupby("team")
    }


def assign_adaptation_flag(row, baselines: dict) -> int:
    """HEAD 2 ground truth — 1 if team deviates from their modal tactic."""
    modal = baselines.get(row["team"])
    if modal is None:
        return 0
    return 1 if row["tactic_label"] != modal else 0


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


def assign_suggestion_label(row) -> str:
    """HEAD 3 ground truth — game-state lookup table."""
    return SUGGESTION_LOOKUP.get(
        (score_bracket(row["score_differential"]),
         minute_bracket(row["match_minute"])),
        "build_up",
    )


def run_label_unit_tests():
    """Sanity-check tactic labelling rules."""
    tests = [
        ({"n_pressures_opp_half": 2, "n_ball_recoveries": 0,
          "n_progressive_carries": 0, "n_pressures": 2,
          "n_passes": 5, "n_passes_own_half": 2},    "high_press"),
        ({"n_pressures_opp_half": 0, "n_ball_recoveries": 1,
          "n_progressive_carries": 1, "n_pressures": 2,
          "n_passes": 3, "n_passes_own_half": 1},    "counter_attack"),
        ({"n_pressures_opp_half": 0, "n_ball_recoveries": 0,
          "n_progressive_carries": 0, "n_pressures": 0,
          "n_passes": 6, "n_passes_own_half": 5},    "low_block"),
        ({"n_pressures_opp_half": 1, "n_ball_recoveries": 0,
          "n_progressive_carries": 0, "n_pressures": 3,
          "n_passes": 5, "n_passes_own_half": 2},    "build_up"),
    ]
    print("Unit tests — assign_tactic_label:")
    all_pass = True
    for row, expected in tests:
        got    = assign_tactic_label(row)
        status = "PASS" if got == expected else "FAIL"
        if status == "FAIL":
            all_pass = False
        print(f"  [{status}] expected={expected:<16} got={got}")
    print(f"  {'All passed.' if all_pass else 'FAILURES DETECTED.'}\n")
    return all_pass


def apply_all_labels(df_clips: pd.DataFrame, train_match_ids: set) -> pd.DataFrame:
    """Apply tactic, adaptation, and suggestion labels to df_clips."""
    df = df_clips.copy()

    # HEAD 1
    df["tactic_label"] = df.apply(assign_tactic_label, axis=1)

    # HEAD 2 — baselines from training set only
    df_train  = df[df["match_id"].isin(train_match_ids)]
    baselines = build_team_baselines(df_train)
    df["adaptation_flag"] = df.apply(
        lambda r: assign_adaptation_flag(r, baselines), axis=1
    )

    # HEAD 3
    df["score_bracket"]  = df["score_differential"].apply(score_bracket)
    df["minute_bracket"] = df["match_minute"].apply(minute_bracket)
    df["suggestion_label"] = df.apply(assign_suggestion_label, axis=1)

    print("Label distributions:")
    for col in ["tactic_label", "adaptation_flag", "suggestion_label"]:
        print(f"\n{col}:\n{df[col].value_counts().to_string()}")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# OVERSAMPLING RARE CLASSES (training set only)
# ══════════════════════════════════════════════════════════════════════════════

NOISE_COLS = [
    "n_pressures", "n_pressures_opp_half",
    "n_passes", "n_passes_own_half",
    "n_carries", "n_progressive_carries",
    "n_ball_recoveries", "n_clearances", "n_dribbles",
]


def oversample_and_cap(df_clips: pd.DataFrame,
                        train_match_ids: set,
                        le) -> pd.DataFrame:
    """
    1. Oversample counter_attack and high_press to TARGET_PER_CLASS.
    2. Cap build_up at BUILD_UP_CAP to reduce majority dominance.
    Val / test rows are never touched.
    """
    df_train = df_clips[df_clips["match_id"].isin(train_match_ids)].copy()
    df_other = df_clips[~df_clips["match_id"].isin(train_match_ids)].copy()

    # Step 1 — oversample rare classes
    np.random.seed(RANDOM_SEED)
    augmented = []
    for cls in ["counter_attack", "high_press"]:
        cls_clips = df_train[df_train["tactic_label"] == cls]
        needed    = max(0, TARGET_PER_CLASS - len(cls_clips))
        print(f"  {cls}: {len(cls_clips)} -> target {TARGET_PER_CLASS} "
              f"(adding {needed} synthetic clips)")
        if needed == 0:
            continue
        sampled = cls_clips.sample(n=needed, replace=True,
                                   random_state=RANDOM_SEED).copy()
        for col in NOISE_COLS:
            if col in sampled.columns:
                noise = np.random.normal(0, 0.5, size=len(sampled))
                sampled[col] = (sampled[col] + noise).clip(lower=0).round().astype(int)

        # Re-verify labels after noise
        sampled["tactic_label"]     = sampled.apply(assign_tactic_label, axis=1)
        sampled = sampled[sampled["tactic_label"] == cls].copy()
        sampled["tactic_label_enc"] = le.transform(sampled["tactic_label"])
        sampled["suggestion_label"] = sampled.apply(assign_suggestion_label, axis=1)
        sampled["suggestion_label_enc"] = le.transform(sampled["suggestion_label"])

        print(f"    -> {len(sampled)} synthetic clips retained after label check")
        augmented.append(sampled)

    if augmented:
        df_train = pd.concat([df_train] + augmented, ignore_index=True)

    # Step 2 — cap build_up
    df_bu    = df_train[df_train["tactic_label"] == "build_up"]
    df_rest  = df_train[df_train["tactic_label"] != "build_up"]
    df_bu_capped = df_bu.sample(n=min(BUILD_UP_CAP, len(df_bu)),
                                random_state=RANDOM_SEED)
    df_train = pd.concat([df_bu_capped, df_rest], ignore_index=True)

    print("\nTraining distribution after oversampling + capping:")
    total = len(df_train)
    for cls in TACTIC_CLASSES:
        n   = (df_train["tactic_label"] == cls).sum()
        print(f"  {cls:<20} {n:>8,}  ({n/total*100:.1f}%)")

    return pd.concat([df_train, df_other], ignore_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# PLAYER POSITION EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _get_tactic_position_prior(tactic, flag, n):
    noise = 6.0
    if flag == 1:
        xs = {
            "high_press":     [85, 80, 78, 75, 70, 68, 65, 62, 58, 55, 50],
            "counter_attack": [90, 85, 80, 75, 65, 60, 55, 50, 45, 35, 25],
            "low_block":      [15, 20, 22, 25, 28, 30, 32, 35, 38, 40, 42],
            "build_up":       [10, 25, 35, 40, 45, 50, 55, 48, 42, 38, 30],
        }.get(tactic, [52.5] * 11)
    else:
        xs = {
            "high_press":     [8,  15, 18, 22, 25, 28, 32, 35, 38, 42, 45],
            "counter_attack": [12, 20, 30, 40, 50, 55, 60, 65, 70, 75, 80],
            "low_block":      [85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35],
            "build_up":       [90, 80, 70, 65, 60, 55, 50, 45, 40, 35, 25],
        }.get(tactic, [52.5] * 11)
    ys = [34, 15, 55, 25, 45, 34, 20, 48, 30, 40, 34]
    priors = []
    for i in range(n):
        idx = i % len(xs)
        priors.append({
            "x":  float(np.clip(xs[idx] + np.random.normal(0, noise), 0, PITCH_LENGTH)),
            "y":  float(np.clip(ys[idx] + np.random.normal(0, noise), 0, PITCH_WIDTH)),
            "vx": float(np.random.normal(0, 0.5)),
            "vy": float(np.random.normal(0, 0.5)),
            "team_flag": float(flag),
        })
    return priors


def _extract_one_clip(clip_row, df_events_all, n_players=11):
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

    def positions_for(team_name, flag):
        te = window_ev[
            (window_ev["team"] == team_name) &
            (window_ev["location"].notna())
        ]
        located = []
        for _, ev in te.iterrows():
            loc = ev["location"]
            if not (isinstance(loc, list) and len(loc) == 2):
                continue
            x  = float(loc[0]) * (PITCH_LENGTH / 120.0)
            y  = float(loc[1]) * (PITCH_WIDTH  / 80.0)
            vx, vy = 0.0, 0.0
            if ev["type"] == "Carry" and isinstance(ev.get("carry"), dict):
                el  = ev["carry"].get("end_location")
                dur = float(ev.get("duration", 1.0) or 1.0)
                if el and isinstance(el, list):
                    vx = float(np.clip((el[0] * (PITCH_LENGTH / 120) - x) / dur, -10, 10))
                    vy = float(np.clip((el[1] * (PITCH_WIDTH  / 80)  - y) / dur, -10, 10))
            located.append({
                "x": float(np.clip(x, 0, PITCH_LENGTH)),
                "y": float(np.clip(y, 0, PITCH_WIDTH)),
                "vx": vx, "vy": vy, "team_flag": float(flag),
            })
        # Remove duplicates
        seen, unique = set(), []
        for p in located:
            k = (round(p["x"]), round(p["y"]))
            if k not in seen:
                seen.add(k); unique.append(p)
        unique = unique[:n_players]
        if len(unique) < n_players:
            unique.extend(_get_tactic_position_prior(tactic, flag,
                                                      n_players - len(unique)))
        return unique[:n_players]

    return positions_for(team, 1) + positions_for(opponent, 0)


def extract_all_positions(df_clips: pd.DataFrame,
                           df_events: pd.DataFrame) -> pd.DataFrame:
    """Add 'player_positions' column to df_clips."""
    all_positions, failed = [], 0

    for idx, row in df_clips.iterrows():
        try:
            pos = _extract_one_clip(row, df_events, n_players=11)
            while len(pos) < 22:
                pos.append({
                    "x": PITCH_LENGTH / 2, "y": PITCH_WIDTH / 2,
                    "vx": 0.0, "vy": 0.0,
                    "team_flag": float(len(pos) < 11),
                })
            all_positions.append(pos[:22])
        except Exception:
            failed += 1
            fallback = [
                {"x": PITCH_LENGTH / 2 + np.random.normal(0, 5),
                 "y": PITCH_WIDTH  / 2 + np.random.normal(0, 5),
                 "vx": 0.0, "vy": 0.0, "team_flag": float(i < 11)}
                for i in range(22)
            ]
            all_positions.append(fallback)

        if (idx + 1) % 5000 == 0:
            print(f"  {idx + 1:,} / {len(df_clips):,} clips processed")

    df_clips = df_clips.copy()
    df_clips["player_positions"] = all_positions
    print(f"Position extraction done. Fallback clips: {failed}")
    return df_clips


if __name__ == "__main__":
    run_label_unit_tests()
