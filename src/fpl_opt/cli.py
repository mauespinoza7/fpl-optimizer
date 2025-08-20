from __future__ import annotations
import argparse, json, re
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

from rich.console import Console
from rich.table import Table

from .fplio.api import get_bootstrap_static, get_fixtures
from .fplio.normalize import players_table, teams_table, fixtures_table
from .features.projections import project_next_gw
from .optimize.model import build_squad, pick_xi_from_squad
from .optimize.transfers import build_squad_with_transfers

console = Console()

# ---------- utilities ----------

def _pretty_table(df, title: str) -> Table:
    t = Table(title=title)
    for col in ["web_name","position","price","status","fixture_diff","ep_next","is_captain"]:
        t.add_column(col)
    for _, r in df.sort_values(["position","ep_next"], ascending=[True,False]).iterrows():
        t.add_row(
            str(r.web_name), str(r.position),
            f"£{r.price:.1f}", str(r.status),
            str(int(r.fixture_diff)), f"{r.ep_next:.2f}",
            "C" if bool(r.is_captain) else ""
        )
    return t

def _load_current_team(path_str: str) -> Dict:
    p = Path(path_str)
    if not p.exists() or p.stat().st_size == 0:
        raise SystemExit(f"Current team file not found or empty: {p.resolve()}")
    try:
        cfg = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"Invalid JSON in {p.resolve()}: {e}")
    for key in ("element_ids","bank_tenths","free_transfers"):
        if key not in cfg: raise SystemExit(f"Missing '{key}' in {p.resolve()}")
    if not isinstance(cfg["element_ids"], list) or len(cfg["element_ids"]) != 15:
        raise SystemExit(f"'element_ids' must be a list of 15 IDs in {p.resolve()}")
    cfg["element_ids"] = [int(x) for x in cfg["element_ids"]]
    cfg["bank_tenths"] = int(cfg["bank_tenths"])
    cfg["free_transfers"] = int(cfg["free_transfers"])
    cfg["purchases_tenths"] = {int(k): int(v) for k,v in cfg.get("purchases_tenths",{}).items()}
    return cfg

def _name_to_id_map(bootstrap: dict) -> Dict[str,int]:
    lut = {}
    for e in bootstrap["elements"]:
        web = e["web_name"].lower()
        full = f'{e["first_name"]} {e["second_name"]}'.lower()
        lut[web] = e["id"]
        lut[full] = e["id"]
    return lut

def _parse_accept_list(s: str|None, lut: Dict[str,int]) -> List[int]:
    """Accept either 'id1;id2' or 'Name1; Name2'. Returns element_ids."""
    if not s: return []
    out: List[int] = []
    for token in [x.strip() for x in s.split(";") if x.strip()]:
        if re.fullmatch(r"\d+", token):
            out.append(int(token))
        else:
            key = token.lower()
            if key not in lut:
                raise SystemExit(f"Could not resolve '{token}' to an element_id. Try full name or web_name.")
            out.append(int(lut[key]))
    return out

def _sell_price_tenths(buy_t: int, now_t: int) -> int:
    if now_t <= buy_t:
        return now_t
    prof = now_t - buy_t
    realized = (prof // 20) * 5
    return buy_t + int(realized)

# ---------- main ops ----------

def run(
    current_team_path: str | None,
    show_current: bool,
    apply_path: str | None,
    accept_ins_raw: str | None,
    accept_outs_raw: str | None,
    max_extra_transfers: int,
    export_current_team: str | None,
):
    console.rule("[bold green]FPL Optimizer")

    # Live data
    bs = get_bootstrap_static()
    fx = get_fixtures()
    players = players_table(bs)
    teams = teams_table(bs)
    fixtures = fixtures_table(fx)
    proj = project_next_gw(players, teams, fixtures, "configs/weights.yaml")
    candidates = proj[proj["exp_minutes"] > 0].copy()

    id_to_name = {int(r.element_id): str(r.web_name) for _, r in candidates.iterrows()}
    name_lut   = _name_to_id_map(bs)

    # 1) SHOW CURRENT TEAM + XI + EP
    if show_current:
        if not current_team_path:
            raise SystemExit("--show-current requires --current-team <file>")
        cfg = _load_current_team(current_team_path)
        df15 = candidates[candidates["element_id"].isin(cfg["element_ids"])].copy()
        if len(df15) != 15:
            missing = set(cfg["element_ids"]) - set(df15["element_id"].astype(int))
            console.print(f"[yellow]Warning:[/yellow] Missing from candidate pool (status/minutes=0?): {sorted(missing)}")
        xi_df, xi_obj = pick_xi_from_squad(df15)
        console.print(_pretty_table(xi_df[xi_df["is_starter"]], "Best XI (from your 15)"))
        console.print(_pretty_table(xi_df[~xi_df["is_starter"]], "Bench"))
        console.print(f"[bold]Projected GW score with your team:[/bold] {xi_obj:.2f}")
        return

    # 2) APPLY SELECTED TRANSFERS (partial acceptance)
    if apply_path:
        # We expect you already ran a recommendation; but we can still apply against live prices now.
        cfg = _load_current_team(apply_path)
        accept_ins  = _parse_accept_list(accept_ins_raw, name_lut)
        accept_outs = _parse_accept_list(accept_outs_raw, name_lut)

        if not accept_ins and not accept_outs:
            raise SystemExit("Use --accept-ins/--accept-outs with IDs or names to apply transfers.")

        current = set(cfg["element_ids"])
        # If only ins are given, infer outs as arbitrary players to make space (same positions ideally).
        # Simple rule: pair outs with same-count as ins if user didn't specify outs.
        if accept_ins and not accept_outs:
            # Remove cheapest players first to free cash
            have_df = candidates[candidates["element_id"].isin(current)].copy()
            have_df = have_df.sort_values(["position","price"], ascending=[True, True])
            accept_outs = [int(x) for x in have_df["element_id"].head(len(accept_ins)).tolist() if x not in accept_ins]

        # Validate counts
        new_ids = list(current - set(accept_outs)) + accept_ins
        if len(new_ids) != 15:
            raise SystemExit(f"Applying these changes would leave {len(new_ids)} players; need exactly 15.")

        # Cash flow update
        # Build now_cost map (tenths)
        now_map = {int(r.element_id): int(round(r.price*10)) for _, r in candidates.iterrows()}
        purchases = cfg.get("purchases_tenths", {})
        raise_t = sum(_sell_price_tenths(int(purchases.get(pid, now_map.get(pid,0))), now_map.get(pid,0)) for pid in accept_outs)
        spend_t = sum(now_map.get(pid, 0) for pid in accept_ins)
        new_bank = cfg["bank_tenths"] + raise_t - spend_t
        if new_bank < 0:
            raise SystemExit(f"Insufficient funds: need {(-new_bank)/10:.1f} more. (Consider different outs/ins.)")

        # Update purchases: keep old buys; add buys at current price
        for pid in accept_ins:
            purchases[str(pid)] = int(now_map.get(pid,0))

        new_cfg = {
            "element_ids": [int(x) for x in new_ids],
            "bank_tenths": int(new_bank),
            "free_transfers": 1,  # after applying, typically reset to 1 for next week
            "purchases_tenths": purchases,
        }
        Path(apply_path).write_text(json.dumps(new_cfg, indent=2))
        ins_str  = ", ".join([f"{pid} ({id_to_name.get(pid, '?')})" for pid in accept_ins]) if accept_ins else "None"
        outs_str = ", ".join([f"{pid} ({id_to_name.get(pid, '?')})" for pid in accept_outs]) if accept_outs else "None"
        console.print(f"Ins:  {ins_str}")
        console.print(f"Outs: {outs_str}")
        return

    # 3) RECOMMEND TRANSFERS (weekly)
    if current_team_path:
        cfg = _load_current_team(current_team_path)
        current_ids = cfg["element_ids"]
        bank_tenths = cfg["bank_tenths"]
        free_transfers = cfg["free_transfers"]
        purchases = cfg.get("purchases_tenths", {})

        cand_ids = set(int(x) for x in candidates["element_id"])
        missing = [pid for pid in current_ids if pid not in cand_ids]
        if missing:
            console.print(f"[yellow]Warning:[/yellow] Not in pool (status/minutes=0?): {missing}")

        res = build_squad_with_transfers(
            df=candidates,
            current_ids=current_ids,
            bank_tenths=bank_tenths,
            purchases_tenths={int(k):int(v) for k,v in purchases.items()},
            free_transfers=free_transfers,
            max_extra_transfers=max_extra_transfers,
            max_per_team=3,
        )
        squad = res["squad"]
        starters = squad[squad["is_starter"]]
        bench = squad[~squad["is_starter"]]

        fmt = lambda ids: ", ".join(f"{pid} ({id_to_name.get(pid,'?')})" for pid in ids) if ids else "None"
        console.print(f"[bold]Transfers out ({res['transfers_out_count']}):[/bold] {fmt(res['transfers_out'])}")
        console.print(f"[bold]Transfers in:[/bold]  {fmt(res['transfers_in'])}")
        console.print(f"[bold]Extra transfers (hits):[/bold] {res['extra_transfers']} → penalty = {4*res['extra_transfers']} pts")
        console.print(f"[bold]Final bank if executed:[/bold] £{res['final_bank_tenths']/10:.1f}")
        console.print(_pretty_table(starters, "Starting XI (post-transfers)"))
        console.print(_pretty_table(bench, "Bench"))
        console.print(f"[bold]Projected GW score (net of hits):[/bold] {res['objective']:.2f}")
        console.print("\nTo apply some/all of these, run:")
        console.print("  python -m fpl_opt.cli --apply my_team.json --accept-ins \"Name1; Name2\" --accept-outs \"NameA; NameB\"")
        return

    # 4) FRESH-SQUAD (GW1) + auto-save my_team.json
    budget_tenths = 1000
    squad, projected = build_squad(candidates, budget_tenths=budget_tenths, max_per_team=3)
    starters = squad[squad["is_starter"]]
    bench = squad[~squad["is_starter"]]
    console.print(_pretty_table(starters, "Starting XI"))
    console.print(_pretty_table(bench, "Bench"))
    console.print(f"[bold]Projected GW score:[/bold] {projected:.2f}")

    out_path = Path(export_current_team) if export_current_team else Path("my_team.json")
    element_ids = [int(x) for x in squad["element_id"].tolist()]
    total_cost_t = int(round(squad["price"].sum() * 10))
    bank_t = max(0, budget_tenths - total_cost_t)
    team_blob = {
        "element_ids": element_ids,
        "bank_tenths": bank_t,
        "free_transfers": 1,
        "purchases_tenths": {str(pid): int(round(price*10)) for pid, price in zip(squad["element_id"], squad["price"])},
    }
    out_path.write_text(json.dumps(team_blob, indent=2))
    console.print(f"[bold green]Saved optimal GW1 squad to:[/bold green] {out_path} (bank £{bank_t/10:.1f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="FPL Weekly Optimizer")
    p.add_argument("--current-team", type=str, default=None, help="Path to my_team.json (to recommend transfers)")
    p.add_argument("--show-current", action="store_true", help="Show your current team, best XI & projected points")
    p.add_argument("--apply", type=str, default=None, help="Apply transfers and update this my_team.json")
    p.add_argument("--accept-ins", type=str, default=None, help="Names or IDs to buy (semicolon-separated)")
    p.add_argument("--accept-outs", type=str, default=None, help="Names or IDs to sell (semicolon-separated)")
    p.add_argument("--max-extra-transfers", type=int, default=3, help="Cap extra transfers (each costs -4)")
    p.add_argument("--export-current-team", type=str, default=None, help="When building fresh squad, save JSON here")
    args = p.parse_args()

    run(
        current_team_path=args.current_team,
        show_current=args.show_current,
        apply_path=args.apply,
        accept_ins_raw=args.accept_ins,
        accept_outs_raw=args.accept_outs,
        max_extra_transfers=args.max_extra_transfers,
        export_current_team=args.export_current_team,
    )
