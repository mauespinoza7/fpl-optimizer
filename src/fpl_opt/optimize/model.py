from __future__ import annotations
import pandas as pd
from ortools.sat.python import cp_model

def build_squad(df: pd.DataFrame, budget_tenths: int = 1000, max_per_team: int = 3):
    """Pick a 15-man squad, legal XI, and a captain to maximize expected next-GW points."""
    model = cp_model.CpModel()

    n = len(df)
    price_tenths = (df["price"] * 10).round().astype(int).tolist()
    ep = df["ep_next"].tolist()
    teams = df["team"].tolist()
    pos = df["position"].tolist()

    x = [model.NewBoolVar(f"x_{i}") for i in range(n)]  # in 15-man squad
    s = [model.NewBoolVar(f"s_{i}") for i in range(n)]  # starter
    c = [model.NewBoolVar(f"c_{i}") for i in range(n)]  # captain

    # Link starters/captain to squad
    for i in range(n):
        model.Add(s[i] <= x[i])
        model.Add(c[i] <= s[i])

    # Budget
    model.Add(sum(x[i] * price_tenths[i] for i in range(n)) <= budget_tenths)

    # Squad composition
    model.Add(sum(x) == 15)
    for P, cnt in [("GK",2),("DEF",5),("MID",5),("FWD",3)]:
        idx = [i for i in range(n) if pos[i] == P]
        model.Add(sum(x[i] for i in idx) == cnt)

    # Per-team limit
    for t in set(teams):
        idx = [i for i in range(n) if teams[i] == t]
        model.Add(sum(x[i] for i in idx) <= max_per_team)

    # Valid XI
    model.Add(sum(s) == 11)
    model.Add(sum(s[i] for i in range(n) if pos[i] == "GK") == 1)
    model.Add(sum(s[i] for i in range(n) if pos[i] == "DEF") >= 3)
    model.Add(sum(s[i] for i in range(n) if pos[i] == "MID") >= 2)
    model.Add(sum(s[i] for i in range(n) if pos[i] == "FWD") >= 1)

    # Captain exactly one
    model.Add(sum(c) == 1)

    # Objective: starters + captain doubles
    model.Maximize(sum(s[i] * ep[i] for i in range(n)) + sum(c[i] * ep[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 15.0
    res = solver.Solve(model)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible solution")

    chosen_idx = [i for i in range(n) if solver.Value(x[i]) == 1]
    squad = df.iloc[chosen_idx].copy()
    squad["is_starter"] = [solver.Value(s[i]) == 1 for i in chosen_idx]
    squad["is_captain"] = [solver.Value(c[i]) == 1 for i in chosen_idx]
    return squad, solver.ObjectiveValue()


def pick_xi_from_squad(squad_df: pd.DataFrame):
    """
    Given a fixed 15-man squad (rows = those 15), choose a valid starting XI + captain
    to maximize expected points.
    """
    df = squad_df.reset_index(drop=True).copy()
    n = len(df)
    pos = df["position"].tolist()
    ep  = df["ep_next"].tolist()

    m = cp_model.CpModel()
    s = [m.NewBoolVar(f"s_{i}") for i in range(n)]  # starter
    c = [m.NewBoolVar(f"c_{i}") for i in range(n)]  # captain

    for i in range(n):
        m.Add(c[i] <= s[i])

    m.Add(sum(s) == 11)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "GK") == 1)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "DEF") >= 3)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "MID") >= 2)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "FWD") >= 1)
    m.Add(sum(c) == 1)

    m.Maximize(sum(s[i]*ep[i] for i in range(n)) + sum(c[i]*ep[i] for i in range(n)))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    res = solver.Solve(m)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No valid XI")

    out = df.copy()
    out["is_starter"] = [solver.Value(s[i]) == 1 for i in range(n)]
    out["is_captain"] = [solver.Value(c[i]) == 1 for i in range(n)]
    projected = solver.ObjectiveValue()
    return out, projected
