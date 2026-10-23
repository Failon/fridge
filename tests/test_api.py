import pickle

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from football_predictor.features import FeatureEngine, build_features
from football_predictor.model import MatchPredictor


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    rng = np.random.default_rng(1)
    teams = [f"Team {c}" for c in "ABCDEFGH"]
    strength = dict(zip(teams, np.linspace(-0.5, 0.5, len(teams))))
    rows = []
    for year in range(2018, 2022):
        day = pd.Timestamp(f"{year}-08-10")
        for rnd in range(2):
            for h in teams:
                for a in teams:
                    if h != a:
                        rows.append(dict(date=day, league="premier-league", season=year, home=h, away=a,
                                         hg=rng.poisson(np.exp(0.3 + strength[h] - strength[a])),
                                         ag=rng.poisson(np.exp(0.05 + strength[a] - strength[h]))))
                        day += pd.Timedelta(days=1)
    m = pd.DataFrame(rows)
    for c in ["hs", "as", "hst", "ast", "hc", "ac", "odds_h", "odds_d", "odds_a"]:
        m[c] = np.nan
    m["result"] = np.select([m.hg > m.ag, m.hg == m.ag], [0, 1], 2)
    X, engine = build_features(m, FeatureEngine(dc_min_matches=50), verbose=False)
    model = MatchPredictor(use_market=False).fit(X, m, val_seasons=1, verbose=False)
    path = tmp_path_factory.mktemp("model") / "predictor.pkl"
    with open(path, "wb") as f:
        pickle.dump({"model": model, "engine": engine, "last_date": m["date"].max()}, f)

    import os
    os.environ["FOOTBALL_MODEL_PATH"] = str(path)
    from football_predictor.api import app
    with TestClient(app) as c:
        yield c


def test_health_and_teams(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["leagues"] == ["premier-league"]
    assert abs(sum(h["ensemble"].values()) - 1) < 1e-6
    teams = client.get("/leagues/premier-league/teams").json()
    assert teams[0]["name"] == "Team H"  # сильнейшая по построению
    assert client.get("/leagues/nope/teams").status_code == 404


def test_predict(client):
    r = client.post("/predict", json={"league": "premier-league", "home": "team h", "away": "Team A",
                                      "date": "2022-08-20", "odds": {"home": 1.5, "draw": 4.0, "away": 7.0}})
    assert r.status_code == 200, r.text
    p = r.json()
    assert p["home"] == "Team H"  # регистр исправлен
    assert abs(p["p_home"] + p["p_draw"] + p["p_away"] - 1) < 1e-6
    assert p["p_home"] > p["p_away"]
    assert len(p["value"]) == 3 and len(p["top_scores"]) == 5


def test_errors(client):
    r = client.post("/predict", json={"league": "premier-league", "home": "Barcelona", "away": "Team A"})
    assert r.status_code == 404
    r = client.post("/predict", json={"league": "premier-league", "home": "Team A", "away": "Team A"})
    assert r.status_code == 422
    r = client.post("/predict", json={"league": "premier-league", "home": "Team A", "away": "Team B",
                                      "odds": {"home": 0.5, "draw": 3, "away": 3}})
    assert r.status_code == 422


def test_batch_keeps_going_on_error(client):
    r = client.post("/predict/batch", json={"matches": [
        {"league": "premier-league", "home": "Team B", "away": "Team C"},
        {"league": "premier-league", "home": "Nobody", "away": "Team C"}]})
    items = r.json()
    assert items[0]["prediction"] is not None and items[0]["error"] is None
    assert items[1]["prediction"] is None and "Nobody" in items[1]["error"]
