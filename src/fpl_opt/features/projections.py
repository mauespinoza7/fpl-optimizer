from __future__ import annotations
import yaml
import pandas as pd
from typing import Dict, Any

def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def project_next_gw(players: pd.DataFrame, teams: pd.DataFrame, fixtures: pd.DataFrame,
                    weights_path: str = "configs/weights.yaml") -> pd.DataFrame:
    weights = load_yaml(weights_path)

    next_fx = (fixtures.sort_values(["event","fixture_id"]).dropna(subset=["event"]))
    team_diff = {}
    for _, row in next_fx.iterrows():
        team_diff.setdefault(row["team_h"], row["team_h_difficulty"])
        team_diff.setdefault(row["team_a"], row["team_a_difficulty"])

    df = players.copy()
    df["fixture_diff"] = df["team"].map(team_diff).fillna(3).astype(int)

    fixture_bump = weights["fixture_bump"]
    df["fixture_mult"] = df["fixture_diff"].map(lambda d: float(fixture_bump.get(int(d), 1.0)))

    status_minutes = weights["status_minutes"]
    df["exp_minutes"] = df["status"].map(lambda s: float(status_minutes.get(s, 0)))


# New (keeps it config-driven via two optional weights):
    df["ppg"]  = pd.to_numeric(df.get("points_per_game"), errors="coerce").fillna(0.0)
    df["form"] = pd.to_numeric(df.get("form"), errors="coerce").fillna(0.0)

    w_ppg  = float(weights.get("ppg_weight", 0.7))
    w_form = float(weights.get("form_weight", 0.3))
    # Convert appearance-based numbers to per-minute-ish signal
    df["per_min_est"] = (w_ppg * (df["ppg"] / 75.0)) + (w_form * (df["form"] / 75.0))

    pos_bias = weights["position_bps_bias"]
    df["pos_bias"] = df["position"].map(pos_bias).fillna(0)

    df["ep_next"] = (df["exp_minutes"] * df["per_min_est"] * df["fixture_mult"]) + df["pos_bias"]

    keep = ["element_id","web_name","team","position","price","status",
            "chance_of_playing_next_round","fixture_diff","exp_minutes","ppg","ep_next"]
    return df[keep].sort_values("ep_next", ascending=False).reset_index(drop=True)
