from __future__ import annotations
import json, sys
from pathlib import Path
from typing import List
from fpl_opt.fplio.api import get_bootstrap_static

def normalize(s: str) -> str:
    return " ".join(s.lower().split())

def build_team_from_names(names: List[str], bank_tenths: int = 0, free_transfers: int = 1) -> dict:
    bs = get_bootstrap_static(save=False)
    elements = bs["elements"]
    # map web_name and "first last" to id
    lookup = {}
    for e in elements:
        lookup[normalize(e["web_name"])] = e["id"]
        full = normalize(f'{e["first_name"]} {e["second_name"]}')
        lookup[full] = e["id"]

    element_ids = []
    missing = []
    for raw in names:
        key = normalize(raw)
        if key in lookup:
            element_ids.append(int(lookup[key]))
        else:
            missing.append(raw)

    if missing:
        raise SystemExit(f"Could not resolve these names to IDs: {missing}")

    if len(element_ids) != 15:
        raise SystemExit(f"Need 15 players, got {len(element_ids)}")

    return {
        "element_ids": element_ids,
        "bank_tenths": int(bank_tenths),
        "free_transfers": int(free_transfers),
    }

def main():
    if len(sys.argv) < 3:
        print("Usage: python -m fpl_opt.tools.names_to_team 'Name1; Name2; ...; Name15' output.json [bank_tenths] [free_transfers]")
        sys.exit(1)
    names_str = sys.argv[1]
    out = Path(sys.argv[2])
    bank = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    fts = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    names = [n.strip() for n in names_str.split(";") if n.strip()]
    team = build_team_from_names(names, bank, fts)
    out.write_text(json.dumps(team, indent=2))
    print(f"Wrote {out} with {len(team['element_ids'])} players, bank £{team['bank_tenths']/10:.1f}, FT={team['free_transfers']}")

if __name__ == "__main__":
    main()
