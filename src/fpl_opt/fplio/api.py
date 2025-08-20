import json, pathlib
from datetime import datetime, timezone
import requests

BASE = "https://fantasy.premierleague.com/api"
RAW_DIR = pathlib.Path("data/raw")

def _save_json(name, payload):
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = RAW_DIR / f"{name}_{ts}.json"
    path.write_text(json.dumps(payload, indent=2))

def get_bootstrap_static(save=True):
    r = requests.get(f"{BASE}/bootstrap-static/")
    r.raise_for_status()
    data = r.json()
    if save: _save_json("bootstrap_static", data)
    return data

def get_fixtures(save=True):
    r = requests.get(f"{BASE}/fixtures/?future=1")
    r.raise_for_status()
    data = r.json()
    if save: _save_json("fixtures", data)
    return data
