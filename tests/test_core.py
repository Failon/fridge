import numpy as np
import pandas as pd

from football_predictor.dixon_coles import DixonColes, markets, score_matrix
from football_predictor.features import build_features
from football_predictor.metrics import rps
from football_predictor.model import matrix_1x2


def test_rps_perfect_and_uniform():
    y = np.array([0, 1, 2])
    assert rps(np.eye(3)[y], y) == 0
    assert np.isclose(rps(np.full((1, 3), 1 / 3), np.array([1])), 1 / 9)


def test_score_matrix_is_distribution():
    m = score_matrix(1.6, 1.1, -0.1)
    assert np.isclose(m.sum(), 1)
    mk = markets(m)
    assert np.isclose(mk["p_home"] + mk["p_draw"] + mk["p_away"], 1)
    vec = matrix_1x2(np.array([1.6]), np.array([1.1]), np.array([-0.1]))[0]
    assert np.allclose(vec, [mk["p_home"], mk["p_draw"], mk["p_away"]])


def test_dixon_coles_recovers_strengths():
    rng = np.random.default_rng(0)
    teams = [f"T{i}" for i in range(10)]
    att = np.linspace(-0.4, 0.4, 10)
    h, a, hg, ag = [], [], [], []
    for _ in range(30):
        for i in range(10):
            for j in range(10):
                if i != j:
                    h.append(teams[i]); a.append(teams[j])
                    hg.append(rng.poisson(np.exp(0.1 + 0.25 + att[i])))
                    ag.append(rng.poisson(np.exp(0.1 + att[j])))
    dc = DixonColes(xi=0, l2=0.1).fit(h, a, hg, ag, np.zeros(len(h)))
    est = np.array([dc.team_strength(t)[0] for t in teams])
    assert np.corrcoef(est, att)[0, 1] > 0.95
    assert abs(dc.params[-1] - 0.25) < 0.05
    assert dc.predict("T9", "T0")["p_home"] > 0.6


def test_features_have_no_leakage():
    """Признаки матча не должны зависеть от его собственного результата."""
    base = dict(league="L", hs=np.nan, **{"as": np.nan}, hst=np.nan, ast=np.nan, hc=np.nan, ac=np.nan,
                odds_h=np.nan, odds_d=np.nan, odds_a=np.nan)
    rows = [dict(base, date=pd.Timestamp("2020-08-01") + pd.Timedelta(days=7 * k),
                 home=f"T{k % 4}", away=f"T{(k + 1) % 4}", hg=k % 3, ag=1) for k in range(20)]
    df1 = pd.DataFrame(rows)
    df2 = df1.copy()
    df2.loc[19, ["hg", "ag"]] = [9, 0]
    X1, _ = build_features(df1, verbose=False)
    X2, _ = build_features(df2, verbose=False)
    pd.testing.assert_series_equal(X1.iloc[19], X2.iloc[19])
