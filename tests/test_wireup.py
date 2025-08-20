from fpl_opt.fplio.api import get_bootstrap_static, get_fixtures
from fpl_opt.fplio.normalize import players_table, teams_table, fixtures_table

def test_end_to_end_shapes():
    bs = get_bootstrap_static(save=False)
    fx = get_fixtures(save=False)

    players = players_table(bs)
    teams = teams_table(bs)
    fixtures = fixtures_table(fx)

    # Players columns required later
    for col in ["element_id","web_name","team","position","price","status",
                "chance_of_playing_next_round","form","points_per_game"]:
        assert col in players.columns, f"Missing {col} in players"

    # Fixtures columns required later
    for col in ["fixture_id","event","team_h","team_a",
                "team_h_difficulty","team_a_difficulty","kickoff_time"]:
        assert col in fixtures.columns, f"Missing {col} in fixtures"
