import pandas as pd

def players_table(bootstrap):
    df = pd.DataFrame(bootstrap["elements"])
    df["price"] = df["now_cost"] / 10.0
    pos_map = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    df["position"] = df["element_type"].map(pos_map)
    keep = ["id","web_name","team","position","price","status",
            "chance_of_playing_next_round","form","points_per_game"]
    return df[keep].rename(columns={"id":"element_id"})

def teams_table(bootstrap):
    return pd.DataFrame(bootstrap["teams"])[["id","name","short_name"]] \
             .rename(columns={"id":"team_id"})

def fixtures_table(fixtures):
    keep = ["id","event","team_h","team_a","team_h_difficulty","team_a_difficulty","kickoff_time"]
    return pd.DataFrame(fixtures)[keep].rename(columns={"id":"fixture_id"})
