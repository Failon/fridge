"""Последовательная генерация признаков без утечки будущего.

Матчи обрабатываются строго по времени: для каждого матча сначала считаются признаки
по состоянию «до матча», затем состояние обновляется его результатом. Тот же движок
после прохода по всей истории используется для прогноза будущих матчей.

Признаки:
  * Elo с поправкой на разницу мячей и откатом к среднему между сезонами;
  * pi-ratings (Constantinou & Fenton, 2013) — раздельные домашний/гостевой рейтинги;
  * Диксон–Коулз с затуханием по времени (переобучается раз в неделю на каждую лигу);
  * экспоненциально сглаженная форма (быстрая и медленная): голы, удары, удары в створ,
    угловые, очки; отдых, опыт в лиге, признак новичка;
  * рыночные вероятности из коэффициентов (если они есть в данных).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .dixon_coles import DixonColes

FORM_STATS = ["gf", "ga", "sf", "sa", "stf", "sta", "cf", "ca", "pts"]
ALPHAS = {"fast": 0.25, "slow": 0.07}


def _default_league_stats():
    return {"hg": 1.5, "ag": 1.1, "draw": 0.26}


@dataclass
class TeamState:
    elo: float
    pi_home: float
    pi_away: float
    season: int
    last_date: pd.Timestamp | None = None
    n_league: int = 0          # матчей в лиге за всё время
    n_season: int = 0
    new_in_league: bool = True
    form: dict = field(default_factory=dict)  # (stat, speed) -> EWMA


class FeatureEngine:
    def __init__(self, elo_k: float = 20.0, elo_hfa: float = 65.0, elo_revert: float = 0.2,
                 pi_lambda: float = 0.035, pi_gamma: float = 0.7, pi_c: float = 3.0,
                 dc_window_days: int = 3 * 365, dc_refit_days: int = 7, dc_min_matches: int = 250,
                 dc_xi: float = 0.0019):
        self.elo_k, self.elo_hfa, self.elo_revert = elo_k, elo_hfa, elo_revert
        self.pi_lambda, self.pi_gamma, self.pi_c = pi_lambda, pi_gamma, pi_c
        self.dc_window, self.dc_refit, self.dc_min, self.dc_xi = dc_window_days, dc_refit_days, dc_min_matches, dc_xi

        self.teams: dict[tuple[str, str], TeamState] = {}
        self.history: dict[str, list] = defaultdict(list)      # league -> [(date, h, a, hg, ag)]
        self.dc: dict[str, DixonColes] = {}
        self.dc_fit_date: dict[str, pd.Timestamp] = {}
        self.league_stats: dict[str, dict] = defaultdict(_default_league_stats)

    # ---------- рейтинги ----------
    def _league_teams(self, league, date):
        # только команды, игравшие в лиге за последний год (вылетевшие не в счёт)
        return [s for (lg, _), s in self.teams.items()
                if lg == league and s.last_date is not None and (date - s.last_date).days < 365]

    def _team(self, league: str, name: str, date: pd.Timestamp, season: int) -> TeamState:
        key = (league, name)
        st = self.teams.get(key)
        if st is None:
            others = self._league_teams(league, date)
            if len(others) >= 10:  # новичок лиги (как правило, промоутер) — слабее среднего
                elo = float(np.quantile([o.elo for o in others], 0.15))
                pih = float(np.quantile([o.pi_home for o in others], 0.15))
                pia = float(np.quantile([o.pi_away for o in others], 0.15))
            else:
                elo, pih, pia = 1500.0, 0.0, 0.0
            st = TeamState(elo=elo, pi_home=pih, pi_away=pia, season=season)
            self.teams[key] = st
        elif st.season != season:
            # новый сезон: откат Elo к среднему; вернувшийся после вылета — снова «новичок»
            others = self._league_teams(league, date)
            if others:
                mean_elo = float(np.mean([o.elo for o in others]))
                st.elo = st.elo * (1 - self.elo_revert) + mean_elo * self.elo_revert
            st.new_in_league = st.last_date is not None and (date - st.last_date).days > 150
            if st.new_in_league and len(others) >= 10:
                st.elo = min(st.elo, float(np.quantile([o.elo for o in others], 0.3)))
            st.season, st.n_season = season, 0
        return st

    def _pi_expected_gd(self, r: float) -> float:
        return np.sign(r) * (10 ** (abs(r) / self.pi_c) - 1)

    # ---------- Диксон–Коулз ----------
    def _maybe_refit_dc(self, league: str, date: pd.Timestamp):
        last = self.dc_fit_date.get(league)
        if last is not None and (date - last).days < self.dc_refit:
            return
        hist = [m for m in self.history[league] if (date - m[0]).days <= self.dc_window]
        if len(hist) < self.dc_min:
            return
        d, h, a, hg, ag = zip(*hist)
        days_ago = np.array([(date - x).days for x in d])
        self.dc[league] = DixonColes(xi=self.dc_xi).fit(h, a, hg, ag, days_ago, warm_start=self.dc.get(league))
        self.dc_fit_date[league] = date

    # ---------- признаки ----------
    def features(self, league: str, home: str, away: str, date: pd.Timestamp,
                 odds: tuple | None = None, refit: bool = True) -> dict:
        season = date.year if date.month >= 7 else date.year - 1
        if refit:
            self._maybe_refit_dc(league, date)
        h = self._team(league, home, date, season)
        a = self._team(league, away, date, season)

        f = {"league": league}
        f["elo_h"], f["elo_a"] = h.elo, a.elo
        f["elo_diff"] = h.elo + self.elo_hfa - a.elo
        f["elo_p_home"] = 1 / (1 + 10 ** (-f["elo_diff"] / 400))
        f["pi_h"], f["pi_a"] = h.pi_home, a.pi_away
        f["pi_gd"] = self._pi_expected_gd(h.pi_home) - self._pi_expected_gd(a.pi_away)
        f["pi_overall_diff"] = (h.pi_home + h.pi_away - a.pi_home - a.pi_away) / 2

        dc = self.dc.get(league)
        if dc is not None:
            p = dc.predict(home, away)
            f.update(dc_p_home=p["p_home"], dc_p_draw=p["p_draw"], dc_p_away=p["p_away"],
                     dc_lambda_h=p["lambda_home"], dc_lambda_a=p["lambda_away"],
                     dc_over_2_5=p["over_2_5"], dc_btts=p["btts"], dc_rho=dc.rho, dc_home_adv=dc.params[-1])
            ah, dh = dc.team_strength(home)
            aa, da = dc.team_strength(away)
            f.update(dc_att_h=ah, dc_def_h=dh, dc_att_a=aa, dc_def_a=da)
        else:
            for k in ["dc_p_home", "dc_p_draw", "dc_p_away", "dc_lambda_h", "dc_lambda_a",
                      "dc_over_2_5", "dc_btts", "dc_rho", "dc_home_adv", "dc_att_h", "dc_def_h", "dc_att_a", "dc_def_a"]:
                f[k] = np.nan

        for side, st in (("h", h), ("a", a)):
            for stat in FORM_STATS:
                for speed in ALPHAS:
                    f[f"{side}_{stat}_{speed}"] = st.form.get((stat, speed), np.nan)
            f[f"{side}_rest"] = min((date - st.last_date).days, 30) if st.last_date is not None else 30
            f[f"{side}_n_league"] = min(st.n_league, 400)
            f[f"{side}_n_season"] = st.n_season
            f[f"{side}_new"] = float(st.new_in_league)
        for stat in ["gf", "ga", "stf", "sta", "pts"]:
            for speed in ALPHAS:
                f[f"d_{stat}_{speed}"] = f[f"h_{stat}_{speed}"] - f[f"a_{stat}_{speed}"]
        f["d_rest"] = f["h_rest"] - f["a_rest"]
        f["season_progress"] = (h.n_season + a.n_season) / 2

        ls = self.league_stats[league]
        f["lg_home_goals"], f["lg_away_goals"], f["lg_draw_rate"] = ls["hg"], ls["ag"], ls["draw"]

        if odds is not None and all(o and o > 1 for o in odds):
            inv = np.array([1 / o for o in odds])
            f["mkt_p_home"], f["mkt_p_draw"], f["mkt_p_away"] = inv / inv.sum()  # снимаем маржу
            f["mkt_margin"] = inv.sum() - 1
        else:
            f["mkt_p_home"] = f["mkt_p_draw"] = f["mkt_p_away"] = f["mkt_margin"] = np.nan
        return f

    # ---------- обновление состояния ----------
    def update(self, m: dict) -> None:
        """m — словарь с полями матча (league, date, home, away, hg, ag, hs, as, ...)."""
        league, date = m["league"], m["date"]
        season = date.year if date.month >= 7 else date.year - 1
        h = self._team(league, m["home"], date, season)
        a = self._team(league, m["away"], date, season)
        hg, ag = int(m["hg"]), int(m["ag"])

        # Elo с множителем на разницу мячей (World Football Elo)
        gd = abs(hg - ag)
        mult = 1 if gd <= 1 else (1.5 if gd == 2 else (11 + gd) / 8)
        exp_h = 1 / (1 + 10 ** (-(h.elo + self.elo_hfa - a.elo) / 400))
        score_h = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        delta = self.elo_k * mult * (score_h - exp_h)
        h.elo += delta
        a.elo -= delta

        # pi-ratings
        pred_gd = self._pi_expected_gd(h.pi_home) - self._pi_expected_gd(a.pi_away)
        err = (hg - ag) - pred_gd
        psi = np.sign(err) * self.pi_c * np.log10(1 + abs(err))
        dh, da = psi * self.pi_lambda, -psi * self.pi_lambda
        h.pi_home += dh
        h.pi_away += dh * self.pi_gamma
        a.pi_away += da
        a.pi_home += da * self.pi_gamma

        # форма
        pts_h = 3 if hg > ag else (1 if hg == ag else 0)
        pts_a = 3 if ag > hg else (1 if hg == ag else 0)
        g = lambda k: m.get(k, np.nan)
        obs_h = {"gf": hg, "ga": ag, "sf": g("hs"), "sa": g("as"), "stf": g("hst"), "sta": g("ast"),
                 "cf": g("hc"), "ca": g("ac"), "pts": pts_h}
        obs_a = {"gf": ag, "ga": hg, "sf": g("as"), "sa": g("hs"), "stf": g("ast"), "sta": g("hst"),
                 "cf": g("ac"), "ca": g("hc"), "pts": pts_a}
        for st, obs in ((h, obs_h), (a, obs_a)):
            for stat, v in obs.items():
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    continue
                for speed, alpha in ALPHAS.items():
                    prev = st.form.get((stat, speed))
                    st.form[(stat, speed)] = v if prev is None else prev + alpha * (v - prev)
            st.last_date = date
            st.n_league += 1
            st.n_season += 1

        ls = self.league_stats[league]
        ls["hg"] += 0.005 * (hg - ls["hg"])
        ls["ag"] += 0.005 * (ag - ls["ag"])
        ls["draw"] += 0.005 * ((hg == ag) - ls["draw"])

        hist = self.history[league]
        hist.append((date, m["home"], m["away"], hg, ag))
        if len(hist) > 4000:  # держим только нужное окно
            del hist[:1000]


def build_features(matches: pd.DataFrame, engine: FeatureEngine | None = None,
                   verbose: bool = True) -> tuple[pd.DataFrame, FeatureEngine]:
    """Проходит по всем матчам по времени, возвращает таблицу признаков (по строке на матч)."""
    engine = engine or FeatureEngine()
    rows = []
    n = len(matches)
    for i, m in enumerate(matches.to_dict("records")):
        odds = (m["odds_h"], m["odds_d"], m["odds_a"]) if pd.notna(m.get("odds_h")) else None
        rows.append(engine.features(m["league"], m["home"], m["away"], m["date"], odds))
        engine.update(m)
        if verbose and (i + 1) % 10000 == 0:
            print(f"  признаки: {i + 1}/{n}")
    feats = pd.DataFrame(rows, index=matches.index)
    return feats, engine
