import os
import time
import math
import warnings
from typing import Dict, Optional, List, Tuple

import random
import pandas as pd
import numpy as np
import joblib

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------
# 1. Draft class definition (60 players, hard-coded) – 2018 NBA Draft
# ---------------------------------------------------------------------

DRAFT_2018_PLAYERS = [
    (1, 1, "PHO", "Deandre Ayton", "Arizona"),
    (2, 1, "SAC", "Marvin Bagley III", "Duke"),
    (3, 1, "ATL", "Luka Doncic", ""),             # Real rights to DAL but ATL pick
    (4, 1, "MEM", "Jaren Jackson Jr.", "Michigan State"),
    (5, 1, "DAL", "Trae Young", "Oklahoma"),
    (6, 1, "ORL", "Mo Bamba", "Texas"),
    (7, 1, "CHI", "Wendell Carter Jr.", "Duke"),
    (8, 1, "CLE", "Collin Sexton", "Alabama"),
    (9, 1, "NYK", "Kevin Knox", "Kentucky"),
    (10, 1, "PHI", "Mikal Bridges", "Villanova"),
    (11, 1, "CHA", "Shai Gilgeous-Alexander", "Kentucky"),
    (12, 1, "LAC", "Miles Bridges", "Michigan State"),
    (13, 1, "LAC", "Jerome Robinson", "Boston College"),
    (14, 1, "DEN", "Michael Porter Jr.", "Missouri"),
    (15, 1, "WAS", "Troy Brown Jr.", "Oregon"),
    (16, 1, "PHI", "Zhaire Smith", "Texas Tech"),
    (17, 1, "MIL", "Donte DiVincenzo", "Villanova"),
    (18, 1, "SAS", "Lonnie Walker IV", "Miami (FL)"),
    (19, 1, "ATL", "Kevin Huerter", "Maryland"),
    (20, 1, "MIN", "Josh Okogie", "Georgia Tech"),
    (21, 1, "UTA", "Grayson Allen", "Duke"),
    (22, 1, "CHI", "Chandler Hutchison", "Boise State"),
    (23, 1, "IND", "Aaron Holiday", "UCLA"),
    (24, 1, "POR", "Anfernee Simons", ""),        # HS / prep, no NCAA
    (25, 1, "LAL", "Moritz Wagner", "Michigan"),
    (26, 1, "PHI", "Landry Shamet", "Wichita State"),
    (27, 1, "BOS", "Robert Williams", "Texas A&M"),
    (28, 1, "GSW", "Jacob Evans", "Cincinnati"),
    (29, 1, "BKN", "Dzanan Musa", ""),
    (30, 1, "ATL", "Omari Spellman", "Villanova"),

    (31, 2, "PHO", "Elie Okobo", "France"),       # pro, no NCAA
    (32, 2, "MEM", "Jevon Carter", "West Virginia"),
    (33, 2, "DAL", "Jalen Brunson", "Villanova"),
    (34, 2, "ATL", "Devonte' Graham", "Kansas"),
    (35, 2, "ORL", "Melvin Frazier", "Tulane"),
    (36, 2, "NYK", "Mitchell Robinson", ""),      # HS-only
    (37, 2, "SAC", "Gary Trent Jr.", "Duke"),
    (38, 2, "PHI", "Khyri Thomas", "Creighton"),
    (39, 2, "PHI", "Isaac Bonga", ""),
    (40, 2, "BKN", "Rodions Kurucs", ""),
    (41, 2, "ORL", "Jarred Vanderbilt", "Kentucky"),
    (42, 2, "DET", "Bruce Brown", "Miami (FL)"),
    (43, 2, "DEN", "Justin Jackson", "Maryland"),
    (44, 2, "WAS", "Issuf Sanon", ""),
    (45, 2, "BKN", "Hamidou Diallo", "Kentucky"),
    (46, 2, "HOU", "De'Anthony Melton", "USC"),
    (47, 2, "LAL", "Svi Mykhailiuk", "Kansas"),
    (48, 2, "MIN", "Keita Bates-Diop", "Ohio State"),
    (49, 2, "SAS", "Chimezie Metu", "USC"),
    (50, 2, "IND", "Alize Johnson", "Missouri State"),
    (51, 2, "NOP", "Tony Carr", "Penn State"),
    (52, 2, "UTA", "Vince Edwards", "Purdue"),
    (53, 2, "OKC", "Devon Hall", "Virginia"),
    (54, 2, "DAL", "Shake Milton", "SMU"),
    (55, 2, "CHA", "Arnoldas Kulboka", ""),
    (56, 2, "PHI", "Ray Spalding", "Louisville"),
    (57, 2, "OKC", "Kevin Hervey", "UT Arlington"),
    (58, 2, "DEN", "Thomas Welsh", "UCLA"),
    (59, 2, "PHX", "George King", "Colorado"),
    (60, 2, "PHI", "Kostas Antetokounmpo", "Dayton"),
]

# ---------------------------------------------------------------------
# 2. Feature engineering – same logic as training
# ---------------------------------------------------------------------

def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Minutes_Per_Game" in df.columns:
        minutes_factor = 40 / df["Minutes_Per_Game"].replace(0, 40)
        for stat in ["BLK", "STL", "TRB", "AST"]:
            if stat in df.columns:
                df[f"{stat}_per_40"] = df[stat] * minutes_factor
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Shooter features
    df["Three_Point_Rate"] = df["3PA"] / (df["FGA"] + 1e-6)
    df["Shot_Selection"] = df["eFG%"] / (df["USG%"] + 1e-6)
    df["Pure_Shooting"] = (df["3P%"] + df["FT%"]) / 2

    # Rebounder features
    df["Rebound_per_Minute"] = df["TRB"] / (df["Minutes_Per_Game"] + 1e-6)
    df["Rebound_Efficiency"] = df["TRB%"] * df["Minutes_Per_Game"] / 40
    df["Impact_Rebounding"] = df["TRB_per_40"] * df["WS/40"]

    # Playmaker features
    df["AST_to_TOV"] = df["AST"] / (df["TOV"] + 1e-6)
    df["Playmaking_Impact"] = df["AST%"] / (df["TOV%"] + 1e-6)
    df["Ball_Control"] = (df["AST_per_40"] * df["WS/40"]) / (df["TOV"] + 1e-6)

    # Defender features
    df["Stocks"] = df["STL"] + df["BLK"]
    df["Stocks_per_40"] = df["STL_per_40"] + df["BLK_per_40"]
    df["Defense_Impact"] = (df["STL%"] + df["BLK%"]) * df["DWS"]

    df["STL_BLK_Interaction"] = df["STL_per_40"] * df["BLK_per_40"]
    df["STL_BLK_Impact"] = df["STL%"] * df["BLK%"]

    return df

# ---------------------------------------------------------------------
# 2b. Scorer_Score computed per-player from df_player
# ---------------------------------------------------------------------

def _safe_min_max(series: pd.Series) -> pd.Series:
    s = series.astype(float).replace([np.inf, -np.inf], np.nan)
    mn, mx = s.min(), s.max()
    if pd.isna(mn) or pd.isna(mx) or mx == mn:
        return pd.Series(0.5, index=s.index)
    return (s - mn) / (mx - mn)

def _ensure_percent_0_1(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    if s.dropna().max() > 1.5:
        s = s / 100.0
    return s

def compute_scorer_score_for_df(df_player: pd.DataFrame) -> float:
    """
    Compute Scorer_Score (0–10) for a single-player df (1 row),
    using:
      - Volume: PTS_per_40, USG%
      - Efficiency: TS%, 3P%, FT%
      - Shot profile: 3PA_per_40, FTr
    """
    df = df_player.copy()

    # Per 40 using that player's MPG
    mpg = df["Minutes_Per_Game"].astype(float).replace(0, np.nan)
    pts_per_40 = df["PTS"].astype(float) * 40.0 / mpg
    threepa_per_40 = df["3PA"].astype(float) * 40.0 / mpg

    fga = df["FGA"].astype(float)
    fta = df["FTA"].astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ftr = np.where(fga > 0, fta / fga, np.nan)

    usg = df["USG%"].astype(float)

    ts = _ensure_percent_0_1(df["TS%"])
    three_p = _ensure_percent_0_1(df["3P%"])
    ft = _ensure_percent_0_1(df["FT%"])

    # Normalize by plausible ranges:
    norm_pts40   = (pts_per_40 / 30.0).clip(0, 1)          # 30 pts/40 ~ elite
    norm_usg     = (usg / 32.0).clip(0, 1)                 # 32% usage ~ very high
    norm_ts      = ((ts - 0.45) / (0.65 - 0.45)).clip(0, 1)   # 45–65% TS window
    norm_3p      = ((three_p - 0.25) / (0.45 - 0.25)).clip(0, 1)
    norm_ft      = ((ft - 0.60) / (0.90 - 0.60)).clip(0, 1)
    norm_3pa40   = (threepa_per_40 / 9.0).clip(0, 1)       # 9 3PA/40 ~ bomber
    norm_ftr     = (ftr / 0.6).clip(0, 1)                  # FTr 0.6 ~ foul magnet

    VOL = 0.6 * norm_pts40 + 0.4 * norm_usg
    EFF = 0.6 * norm_ts + 0.2 * norm_3p + 0.2 * norm_ft
    SHOT_PROFILE = 0.5 * norm_3pa40 + 0.5 * norm_ftr

    scorer_raw = 0.50 * EFF + 0.35 * VOL + 0.15 * SHOT_PROFILE
    return float(10.0 * scorer_raw.iloc[0])

# ---------------------------------------------------------------------
# 3. Post-processing: emphasize TRB for rebounding and 3PA for shooting
# ---------------------------------------------------------------------

def apply_skill_adjustments(df: pd.DataFrame,
                            alpha_reb: float = 0.18,
                            alpha_shoot: float = 0.20) -> pd.DataFrame:
    """
    Post-process skill E scores to emphasize:
    - TRB_per_40 for rebounding_E
    - 3PA_per_40 (shooting volume) for shooting_E
    """
    df = df.copy()

    # Derive 3PA_per_40 if we can
    if "Minutes_Per_Game" in df.columns:
        minutes_factor = 40 / df["Minutes_Per_Game"].replace(0, 40)
        if "3PA" in df.columns and "3PA_per_40" not in df.columns:
            df["3PA_per_40"] = df["3PA"] * minutes_factor
    else:
        if "3PA" in df.columns and "3PA_per_40" not in df.columns:
            df["3PA_per_40"] = df["3PA"]

    # Ensure TRB_per_40 and 3PA_per_40 exist
    for col in ["TRB_per_40", "3PA_per_40"]:
        if col not in df.columns:
            df[col] = 0.0

    # z-scores within this class
    for col in ["TRB_per_40", "3PA_per_40"]:
        mean = df[col].mean()
        std = df[col].std(ddof=0)
        if std == 0 or np.isnan(std):
            df[f"{col}_z"] = 0.0
        else:
            df[f"{col}_z"] = (df[col] - mean) / std

    # Adjust rebounding_E
    if "rebounding_E" in df.columns:
        z = df["TRB_per_40_z"]
        reb_factor = 1.0 + alpha_reb * z
        reb_factor = reb_factor.clip(lower=0.5, upper=1.5)
        df["rebounding_E_adj"] = df["rebounding_E"] * reb_factor
    else:
        df["rebounding_E_adj"] = df.get("rebounding_E", np.nan)

    # Adjust shooting_E
    if "shooting_E" in df.columns:
        z = df["3PA_per_40_z"]
        sh_factor = 1.0 + alpha_shoot * z
        sh_factor = sh_factor.clip(lower=0.5, upper=1.5)
        df["shooting_E_adj"] = df["shooting_E"] * sh_factor
    else:
        df["shooting_E_adj"] = df.get("shooting_E", np.nan)

    return df

# ---------------------------------------------------------------------
# 4. Top-N bonus for SVI
# ---------------------------------------------------------------------

def apply_top_n_bonuses(
    df: pd.DataFrame,
    categories: List[str],
    top_n: int = 5,
    bonus_per_category: float = 0.05,
    base_col: str = "SVI_total",
    out_col: str = "SVI_total_with_bonus",
) -> pd.DataFrame:
    """
    Give a bonus to players who finish in the top_n for each category.
    """
    df = df.copy()
    df[out_col] = df[base_col]

    for cat in categories:
        if cat not in df.columns:
            continue
        mask = df["has_ncaa_data"] & df[cat].notna()
        top_idx = df[mask].sort_values(cat, ascending=False).head(top_n).index
        df.loc[top_idx, out_col] += bonus_per_category

    return df

# ---------------------------------------------------------------------
# 5. Helpers for CSV-based stats
# ---------------------------------------------------------------------

def safe_float(x) -> float:
    if x is None or x == "" or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    try:
        return float(x)
    except ValueError:
        return 0.0

def build_raw_feature_row_from_csv(row_in: pd.Series) -> dict:
    """
    row_in is a single row from college_stats_final for one player.
    """
    row_dict = {}

    g = safe_float(row_in.get("Games"))
    row_dict["Games"] = g

    row_dict["Minutes_Per_Game"] = safe_float(row_in.get("Minutes_Per_Game"))

    row_dict["FG"] = safe_float(row_in.get("FG"))
    row_dict["FGA"] = safe_float(row_in.get("FGA"))
    row_dict["3P"] = safe_float(row_in.get("3P"))
    row_dict["3PA"] = safe_float(row_in.get("3PA"))
    row_dict["FT"] = safe_float(row_in.get("FT"))
    row_dict["FTA"] = safe_float(row_in.get("FTA"))
    row_dict["TRB"] = safe_float(row_in.get("TRB"))
    row_dict["AST"] = safe_float(row_in.get("AST"))
    row_dict["STL"] = safe_float(row_in.get("STL"))
    row_dict["BLK"] = safe_float(row_in.get("BLK"))
    row_dict["TOV"] = safe_float(row_in.get("TOV"))
    row_dict["PTS"] = safe_float(row_in.get("PTS"))

    if g > 0:
        row_dict["PPG"] = row_dict["PTS"] / g
    else:
        row_dict["PPG"] = 0.0

    row_dict["FG%"] = safe_float(row_in.get("FG%"))
    row_dict["3P%"] = safe_float(row_in.get("3P%"))
    row_dict["FT%"] = safe_float(row_in.get("FT%"))

    row_dict["TS%"] = safe_float(row_in.get("TS%"))
    row_dict["eFG%"] = safe_float(row_in.get("eFG%"))
    row_dict["TRB%"] = safe_float(row_in.get("TRB%"))
    row_dict["AST%"] = safe_float(row_in.get("AST%"))
    row_dict["STL%"] = safe_float(row_in.get("STL%"))
    row_dict["BLK%"] = safe_float(row_in.get("BLK%"))
    row_dict["TOV%"] = safe_float(row_in.get("TOV%"))
    row_dict["USG%"] = safe_float(row_in.get("USG%"))

    row_dict["OWS"] = safe_float(row_in.get("OWS"))
    row_dict["DWS"] = safe_float(row_in.get("DWS"))
    row_dict["WS"] = safe_float(row_in.get("WS"))
    row_dict["WS/40"] = safe_float(row_in.get("WS/40"))

    yp = row_in.get("YearsPlayed")
    row_dict["YearsPlayed"] = safe_float(yp) if yp is not None else 1.0

    last_conf = str(row_in.get("LastConf")) if "LastConf" in row_in else ""
    row_dict["LastConf"] = last_conf

    return row_dict

# ---------------------------------------------------------------------
# 6. Model loading and inference helpers
# ---------------------------------------------------------------------

def load_model_data(path: str) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    obj = joblib.load(path)

    if isinstance(obj, dict) and "model" in obj:
        if "features" not in obj:
            obj["features"] = []
        if "scaler" not in obj:
            obj["scaler"] = None
        return obj

    if isinstance(obj, dict):
        model = (
            obj.get("model")
            or obj.get("estimator")
            or obj.get("clf")
            or obj.get("classifier")
        )
        features = (
            obj.get("features")
            or obj.get("feature_names")
            or obj.get("feature_cols")
            or []
        )
        scaler = obj.get("scaler") or obj.get("standardizer") or None
        if model is None:
            model = obj
        if not features and hasattr(model, "feature_names_in_"):
            features = list(model.feature_names_in_)
        return {
            "model": model,
            "features": features,
            "scaler": scaler,
        }

    model = obj
    features: List[str] = []
    if hasattr(model, "feature_names_in_"):
        features = list(model.feature_names_in_)
    return {
        "model": model,
        "features": features,
        "scaler": None,
    }

def compute_skill_probabilities(model_data: dict, X: np.ndarray, temp: float = 0.85) -> Tuple[np.ndarray, List[int]]:
    """
    Compute class probabilities, with optional temperature sharpening.
    temp < 1.0 => sharper; >1.0 => flatter.
    """
    model = model_data["model"]
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if temp != 1.0:
            prob = np.power(prob, 1.0 / temp)
            prob = prob / prob.sum(axis=1, keepdims=True)
        classes_ = model.classes_.tolist()
        return prob, classes_
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        scores = np.array(scores)
        if scores.ndim == 1:
            scores = scores[:, None]
        scores = scores / temp
        exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
        prob = exp_scores / exp_scores.sum(axis=1, keepdims=True)
        return prob, list(range(prob.shape[1]))
    preds = model.predict(X)
    classes_ = np.unique(preds)
    prob = np.zeros((len(preds), len(classes_)))
    for i, c in enumerate(classes_):
        prob[preds == c, i] = 1.0
    return prob, classes_.tolist()

def expected_tier_value(prob_row: np.ndarray, classes_: List[int]) -> float:
    return float(np.dot(prob_row, np.array(classes_, dtype=float)))

def print_leaderboard_skill(df: pd.DataFrame, metric: str, title: str, n: int = 10):
    cols = [
        metric,
        "SVI_total_with_bonus",
        "pick",
        "round",
        "team",
        "player",
        "college",
        "has_ncaa_data",
    ]
    cols_present = [c for c in cols if c in df.columns]

    print(f"\n=== Top {n} {title} ({metric}) ===")
    print(
        df[df["has_ncaa_data"]]
        .sort_values(metric, ascending=False)
        [cols_present]
        .head(n)
        .to_string(index=False, float_format=lambda x: f"{x:0.3f}")
    )

# ---------------------------------------------------------------------
# 7. Main – build full feature CSV + predictions (2018, CSV-based)
# ---------------------------------------------------------------------

def main():
    root = os.path.dirname(os.path.abspath(__file__))

    stats_path = os.path.join(root, "college_stats_final.csv")
    if not os.path.exists(stats_path):
        raise FileNotFoundError(f"college_stats_final.csv not found at {stats_path}")

    college_df = pd.read_csv(stats_path)

    if "Player" not in college_df.columns:
        raise ValueError("CSV must have a 'Player' column (player names in col 0).")

    college_df["Player"] = college_df["Player"].astype(str).str.strip()

    model_paths = {
        "Defense":    os.path.join(root, "defense_best_model.joblib"),
        "Playmaking": os.path.join(root, "playmaking_best_model.joblib"),
        "Rebounding": os.path.join(root, "rebounding_best_model.joblib"),
        "Shooter":    os.path.join(root, "shooter_best_model.joblib"),
    }

    print("Loading skill models...")
    models = {k: load_model_data(p) for k, p in model_paths.items()}

    # Union of all features used by any model
    all_model_features = set()
    for name, m in models.items():
        feats = m.get("features")
        if isinstance(feats, list) and feats:
            all_model_features.update(feats)

        model = m.get("model")
        if model is not None and hasattr(model, "feature_names_in_"):
            all_model_features.update(list(model.feature_names_in_))

    all_model_features = sorted(all_model_features)
    print(f"Total distinct model features: {len(all_model_features)}")

    rows_for_df = []

    for pick, rnd, team, name, college in DRAFT_2018_PLAYERS:
        print(f"\n[INFO] 2018 Pick #{pick} – {name} ({team})")

        base_row = {
            "pick": pick,
            "round": rnd,
            "team": team,
            "player": name,
            "college": college,
            "has_ncaa_data": False,
            "Scorer_Score": math.nan,
        }

        skill_E = {
            "Shooter": math.nan,
            "Rebounding": math.nan,
            "Defense": math.nan,
            "Playmaking": math.nan,
        }
        SVI_total = math.nan

        feature_values = {feat: math.nan for feat in all_model_features}

        mask = college_df["Player"].str.lower() == name.lower()
        player_rows = college_df[mask]

        if not player_rows.empty:
            row_in = player_rows.iloc[-1]
            raw_row = build_raw_feature_row_from_csv(row_in)
            df_player = pd.DataFrame([raw_row])

            if "LastConf" in raw_row and isinstance(raw_row["LastConf"], str):
                conf_val = raw_row["LastConf"]
            else:
                conf_val = ""
            conf_col = f"Conf_{conf_val}" if conf_val else None

            df_player = add_advanced_features(df_player)
            df_player = engineer_features(df_player)

            # --- compute scorer score for this player ---
            try:
                scorer_score = compute_scorer_score_for_df(df_player)
            except Exception as e:
                print(f"[WARN] Scorer_Score computation failed for {name}: {e}")
                scorer_score = math.nan
            base_row["Scorer_Score"] = scorer_score
            # -------------------------------------------------

            for feat in all_model_features:
                if feat.startswith("Conf_") and feat not in df_player.columns:
                    df_player[feat] = 0.0
            if conf_col and conf_col in all_model_features:
                df_player[conf_col] = 1.0

            for feat in all_model_features:
                if feat not in df_player.columns:
                    df_player[feat] = 0.0

            for feat in all_model_features:
                feature_values[feat] = float(df_player.iloc[0][feat])

            for skill_name in ["Shooter", "Rebounding", "Defense", "Playmaking"]:
                mdata = models[skill_name]
                model = mdata["model"]
                scaler = mdata.get("scaler", None)

                if scaler is not None and hasattr(scaler, "feature_names_in_"):
                    model_feats = list(scaler.feature_names_in_)
                elif hasattr(model, "feature_names_in_"):
                    model_feats = list(model.feature_names_in_)
                else:
                    model_feats = mdata.get("features")
                    if not model_feats:
                        raise ValueError(
                            f"Model for {skill_name} does not have 'features', "
                            "'feature_names_in_', or scaler 'feature_names_in_'."
                        )

                for f in model_feats:
                    if f not in df_player.columns:
                        df_player[f] = 0.0

                X = df_player[model_feats].astype(float).values

                if scaler is not None:
                    try:
                        X_scaled = scaler.transform(X)
                    except Exception as e:
                        print(f"[WARN] {skill_name} scaler transform failed ({e}); using unscaled X.")
                        X_scaled = X
                else:
                    X_scaled = X

                if hasattr(model, "n_features_in_"):
                    expected = model.n_features_in_
                    if X_scaled.shape[1] != expected:
                        print(f"\nDEBUG {skill_name}: model expects {expected}, "
                              f"we are providing {X_scaled.shape[1]}")
                        print("DEBUG using features:", model_feats)
                        if hasattr(model, "feature_names_in_"):
                            fallback_feats = list(model.feature_names_in_)
                            for f in fallback_feats:
                                if f not in df_player.columns:
                                    df_player[f] = 0.0
                            X_fallback = df_player[fallback_feats].astype(float).values
                            X_scaled = X_fallback
                        else:
                            raise ValueError(
                                f"{skill_name} model expects {expected} features, "
                                f"but got {X_scaled.shape[1]}."
                            )

                prob, classes_ = compute_skill_probabilities(mdata, X_scaled, temp=0.85)
                E_val = expected_tier_value(prob[0], classes_)
                skill_E[skill_name] = E_val

            base_row["has_ncaa_data"] = True

            shooting_E = skill_E["Shooter"]
            rebounding_E = skill_E["Rebounding"]
            defense_E = skill_E["Defense"]
            playmaking_E = skill_E["Playmaking"]

            if not any(
                math.isnan(x)
                for x in [shooting_E, rebounding_E, defense_E, playmaking_E]
            ):
                SVI_total = shooting_E + rebounding_E + defense_E + playmaking_E
            else:
                SVI_total = math.nan

            base_row["shooting_E"] = shooting_E
            base_row["rebounding_E"] = rebounding_E
            base_row["defense_E"] = defense_E
            base_row["playmaking_E"] = playmaking_E
            base_row["SVI_total"] = SVI_total

        if not base_row["has_ncaa_data"]:
            base_row["shooting_E"] = math.nan
            base_row["rebounding_E"] = math.nan
            base_row["defense_E"] = math.nan
            base_row["playmaking_E"] = math.nan
            base_row["SVI_total"] = math.nan
            base_row["Scorer_Score"] = math.nan

        full_row = {**base_row, **feature_values}
        rows_for_df.append(full_row)

    df = pd.DataFrame(rows_for_df)

    # Apply TRB/3PA-based adjustments
    df = apply_skill_adjustments(df, alpha_reb=0.18, alpha_shoot=0.20)

    df["shooting_E_raw"] = df["shooting_E"]
    df["rebounding_E_raw"] = df["rebounding_E"]
    df["shooting_E"] = df["shooting_E_adj"]
    df["rebounding_E"] = df["rebounding_E_adj"]

    df["SVI_total"] = (
        df["shooting_E"] +
        df["rebounding_E"] +
        df["defense_E"] +
        df["playmaking_E"]
    )

    # Apply top-5 bonus per category
    df = apply_top_n_bonuses(
        df,
        categories=["shooting_E", "rebounding_E", "defense_E", "playmaking_E"],
        top_n=5,
        bonus_per_category=0.05,
        base_col="SVI_total",
        out_col="SVI_total_with_bonus",
    )

    df["rank_by_SVI"] = df["SVI_total_with_bonus"].rank(ascending=False, method="min")

    # Rank by Scorer_Score
    if "Scorer_Score" in df.columns:
        df["rank_by_Scorer"] = df["Scorer_Score"].rank(ascending=False, method="min")

    # Save to CSV
    out_path = os.path.join(root, "2018_case_study_full_features_and_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved 2018 full feature + prediction CSV to {out_path}\n")

    # Preview
    preview_cols = [
        "rank_by_SVI",
        "pick",
        "round",
        "team",
        "player",
        "college",
        "has_ncaa_data",
        "shooting_E",
        "rebounding_E",
        "defense_E",
        "playmaking_E",
        "SVI_total",
        "SVI_total_with_bonus",
        "Scorer_Score",
        "rank_by_Scorer",
    ]
    cols_to_show = [c for c in preview_cols if c in df.columns]
    print(
        df[cols_to_show]
        .sort_values("rank_by_SVI")
        .head(20)
        .to_string(index=False, float_format=lambda x: f"{x:0.3f}")
    )

    # Leaderboards
    print_leaderboard_skill(df, "shooting_E",    "Shooters")
    print_leaderboard_skill(df, "rebounding_E",  "Rebounders")
    print_leaderboard_skill(df, "playmaking_E",  "Playmakers")
    print_leaderboard_skill(df, "defense_E",     "Defenders")
    print_leaderboard_skill(df, "SVI_total_with_bonus", "Overall SVI (with top-5 bonus)")
    print_leaderboard_skill(df, "Scorer_Score",  "Pure Scorers")

if __name__ == "__main__":
    main()