from __future__ import annotations
from typing import Iterable, Dict, Any, Set
import pandas as pd
from ortools.sat.python import cp_model

def _sell_price_tenths(buy_t: int, now_t: int) -> int:
    """FPL selling price in tenths.
    Profit: only half is realized, rounded down to nearest 0.1 (i.e., 5 tenths per 0.2 rise).
    Losses: fully realized.
    """
    if now_t <= buy_t:
        return now_t
    # profit in tenths
    prof = now_t - buy_t
    realized = (prof // 20) * 5  # each 0.2 ⇒ +0.1 realized
    return buy_t + int(realized)

def build_squad_with_transfers(
    df: pd.DataFrame,
    current_ids: Iterable[int],
    bank_tenths: int,
    purchases_tenths: Dict[int, int] | None = None,
    free_transfers: int = 1,
    max_extra_transfers: int = 3,
    max_per_team: int = 3,
) -> Dict[str, Any]:
    """
    Cash-flow-aware transfer optimization.

    Budget constraint (true FPL cash flow):
        sum(price of players you BUY) <= bank
                                     + sum(sell price of players you SELL)

    - Keeps are free (already owned).
    - SELL price uses FPL's 50% rule.
    - Objective: maximize EP of XI + captain - 4 * extra_transfers.
    """
    purchases_tenths = purchases_tenths or {}
    cur: Set[int] = set(int(x) for x in current_ids)

    n = len(df)
    ids = df["element_id"].astype(int).tolist()
    price_t = (df["price"] * 10).round().astype(int).tolist()
    ep = df["ep_next"].tolist()
    teams = df["team"].tolist()
    pos = df["position"].tolist()

    # Precompute per-player sell prices (only relevant for currently owned)
    id_to_now_t = {ids[i]: price_t[i] for i in range(n)}
    sell_price_map: Dict[int, int] = {}
    for pid in cur:
        buy_t = int(purchases_tenths.get(pid, id_to_now_t.get(pid, 0)))  # fallback: assume bought at now
        now_t = int(id_to_now_t.get(pid, buy_t))
        sell_price_map[pid] = _sell_price_tenths(buy_t, now_t)

    m = cp_model.CpModel()

    # Decision variables
    x = [m.NewBoolVar(f"x_{i}") for i in range(n)]  # in final 15
    s = [m.NewBoolVar(f"s_{i}") for i in range(n)]  # starter
    c = [m.NewBoolVar(f"c_{i}") for i in range(n)]  # captain

    # Linking
    for i in range(n):
        m.Add(s[i] <= x[i])
        m.Add(c[i] <= s[i])

    # Squad composition rules
    m.Add(sum(x) == 15)
    for P, cnt in [("GK", 2), ("DEF", 5), ("MID", 5), ("FWD", 3)]:
        idx = [i for i in range(n) if pos[i] == P]
        m.Add(sum(x[i] for i in idx) == cnt)

    # Club limit
    for t in set(teams):
        idx = [i for i in range(n) if teams[i] == t]
        m.Add(sum(x[i] for i in idx) <= max_per_team)

    # Valid XI & captain
    m.Add(sum(s) == 11)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "GK") == 1)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "DEF") >= 3)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "MID") >= 2)
    m.Add(sum(s[i] for i in range(n) if pos[i] == "FWD") >= 1)
    m.Add(sum(c) == 1)

    # Identify buy/sell decisions
    buys = []   # players not currently owned that we select (x=1)
    sells = []  # players currently owned that we do NOT select (x=0)
    sell_values = []
    buy_costs = []

    for i in range(n):
        pid = ids[i]
        if pid in cur:
            # selling boolean: current & not kept => sell
            o = m.NewBoolVar(f"sell_{i}")
            m.Add(o == 1 - x[i])
            sells.append(o)
            sell_values.append(sell_price_map.get(pid, 0))
        else:
            # buying boolean: not current & selected => buy
            b = m.NewBoolVar(f"buy_{i}")
            m.Add(b == x[i])
            buys.append(b)
            buy_costs.append(price_t[i])

    # Cash flow: sum(buy prices) <= bank + sum(sell prices)
    if buys:
        lhs = sum(buys[i] * buy_costs[i] for i in range(len(buys)))
    else:
        lhs = 0
    if sells:
        rhs = bank_tenths + sum(sells[i] * sell_values[i] for i in range(len(sells)))
    else:
        rhs = bank_tenths
    m.Add(lhs <= rhs)

    # Transfers counting & hit penalty
    transfers_out = m.NewIntVar(0, 15, "transfers_out")
    if sells:
        m.Add(transfers_out == sum(sells))
    else:
        m.Add(transfers_out == 0)

    extra = m.NewIntVar(0, 15, "extra_transfers")
    m.Add(extra >= transfers_out - free_transfers)
    m.Add(extra >= 0)

    # Objective: starters + captain doubles - 4 per extra transfer
    m.Maximize(sum(s[i] * ep[i] for i in range(n)) + sum(c[i] * ep[i] for i in range(n)) - 4.0 * extra)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 25.0
    res = solver.Solve(m)
    if res not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("No feasible transfer plan under cash-flow constraints")

    chosen_idx = [i for i in range(n) if solver.Value(x[i]) == 1]
    squad = df.iloc[chosen_idx].copy()
    squad["is_starter"] = [solver.Value(s[i]) == 1 for i in chosen_idx]
    squad["is_captain"] = [solver.Value(c[i]) == 1 for i in chosen_idx]

    final_ids = set(int(ids[i]) for i in chosen_idx)
    outs_list = sorted(list(cur - final_ids))
    ins_list  = sorted(list(final_ids - cur))

    # Compute final bank after the move
    spent = sum(price_t[i] for i in range(n) if ids[i] in ins_list)
    raised = sum(sell_price_map[pid] for pid in outs_list)
    final_bank = bank_tenths + raised - spent

    return {
        "squad": squad,
        "objective": solver.ObjectiveValue(),
        "transfers_out": outs_list,
        "transfers_in": ins_list,
        "transfers_out_count": solver.Value(transfers_out),
        "extra_transfers": solver.Value(extra),
        "final_bank_tenths": int(final_bank),
    }

