"""Стоимость составов и тренеры из transfermarkt-datasets (github.com/dcaribou/transfermarkt-datasets, CC0).

По каждому матчу 5 лиг (с 2012/13) считается «стоимость вышедшего состава»: рыночная стоимость
игроков на дату матча (последняя оценка Transfermarkt до неё), взвешенная по сыгранным минутам.
Так в модель попадает то, чего нет в счёте и ударах: бюджет, трансферы, ротация, травмы лидеров.
Плюс имя тренера — для признака «стаж тренера».

Датасет перестал обновляться в июле 2026 г.: для текущего сезона признаки берутся
по последним известным матчам (конец сезона 2025/26).
"""
from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from .understat import _name_map

BASE_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data"
TABLES = ["games", "appearances", "player_valuations", "game_lineups"]
COMPETITIONS = {"GB1": "premier-league", "ES1": "la-liga", "IT1": "serie-a", "L1": "bundesliga", "FR1": "ligue-1"}


def download(cache_dir="data/transfermarkt", force=False) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for t in TABLES:
        path = cache / f"{t}.csv.gz"
        if path.exists() and not force:
            continue
        req = urllib.request.Request(f"{BASE_URL}/{t}.csv.gz", headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=120) as resp:
            path.write_bytes(resp.read())
        print(f"  Transfermarkt {t}: {path.stat().st_size / 1e6:.0f} МБ")
    (cache / "match_values.pkl").unlink(missing_ok=True)  # пересчитать при новых данных


def match_values(cache_dir="data/transfermarkt") -> pd.DataFrame:
    """По матчу: стоимость вышедших составов хозяев/гостей (млн €) и тренеры. Кэшируется."""
    cache = Path(cache_dir)
    out_path = cache / "match_values.pkl"
    if out_path.exists():
        return pd.read_pickle(out_path)

    games = pd.read_csv(cache / "games.csv.gz", usecols=[
        "game_id", "competition_id", "season", "date", "home_club_id", "away_club_id", "home_club_goals",
        "away_club_goals", "home_club_name", "away_club_name", "home_club_manager_name", "away_club_manager_name"])
    games = games[games.competition_id.isin(COMPETITIONS)].dropna(subset=["home_club_goals", "away_club_goals"])
    games["date"] = pd.to_datetime(games["date"])

    app = pd.read_csv(cache / "appearances.csv.gz",
                      usecols=["game_id", "player_id", "player_club_id", "date", "minutes_played"])
    app = app[app.game_id.isin(games.game_id)].copy()
    app["date"] = pd.to_datetime(app["date"])

    val = pd.read_csv(cache / "player_valuations.csv.gz", usecols=["player_id", "date", "market_value_in_eur"])
    val["date"] = pd.to_datetime(val["date"])

    # последняя оценка игрока не позже даты матча
    app = pd.merge_asof(app.sort_values("date"), val.sort_values("date"), on="date", by="player_id",
                        direction="backward")
    app["value"] = app["market_value_in_eur"].fillna(0) / 1e6 * app["minutes_played"].clip(0, 90) / 90
    team_val = app.groupby(["game_id", "player_club_id"])["value"].sum()

    # стоимость стартового состава — известна за час до матча (её же видит букмекер в закрытии)
    xi_val = None
    if (cache / "game_lineups.csv.gz").exists():
        lu = pd.read_csv(cache / "game_lineups.csv.gz", usecols=["game_id", "player_id", "club_id", "date", "type"])
        lu = lu[(lu.type == "starting_lineup") & lu.game_id.isin(games.game_id)].copy()
        lu["date"] = pd.to_datetime(lu["date"])
        lu = pd.merge_asof(lu.sort_values("date"), val.sort_values("date"), on="date", by="player_id",
                           direction="backward")
        xi_val = (lu["market_value_in_eur"].fillna(0) / 1e6).groupby([lu.game_id, lu.club_id]).sum()

    g = games.set_index("game_id")
    hv = team_val.reindex(list(zip(g.index, g.home_club_id))).to_numpy()
    av = team_val.reindex(list(zip(g.index, g.away_club_id))).to_numpy()
    out = pd.DataFrame({
        "league": g.competition_id.map(COMPETITIONS).to_numpy(), "season": g.season.astype(int).to_numpy(),
        "date": g.date.to_numpy(), "home": g.home_club_name.to_numpy(), "away": g.away_club_name.to_numpy(),
        "hg": g.home_club_goals.astype(int).to_numpy(), "ag": g.away_club_goals.astype(int).to_numpy(),
        "h_value": hv, "a_value": av,
        "h_xi": xi_val.reindex(list(zip(g.index, g.home_club_id))).to_numpy() if xi_val is not None else np.nan,
        "a_xi": xi_val.reindex(list(zip(g.index, g.away_club_id))).to_numpy() if xi_val is not None else np.nan,
        "h_manager": g.home_club_manager_name.to_numpy(), "a_manager": g.away_club_manager_name.to_numpy(),
    })
    out.loc[out.h_value <= 0, "h_value"] = np.nan  # нет данных по составу
    out.loc[out.a_value <= 0, "a_value"] = np.nan
    out.loc[out.h_xi <= 0, "h_xi"] = np.nan
    out.loc[out.a_xi <= 0, "a_xi"] = np.nan
    out.to_pickle(out_path)
    return out


def attach(matches: pd.DataFrame, cache_dir="data/transfermarkt") -> pd.DataFrame:
    """Добавляет h_value/a_value/h_manager/a_manager; названия клубов сопоставляются по счёту и дате."""
    tm = match_values(cache_dir)
    names = _name_map(tm, matches)
    tm = tm.assign(home=[names.get((lg, t)) for lg, t in zip(tm.league, tm.home)],
                   away=[names.get((lg, t)) for lg, t in zip(tm.league, tm.away)])
    key = ["league", "season", "home", "away"]
    tm = tm.dropna(subset=["home", "away"]).drop_duplicates(key)
    cols = ["h_value", "a_value", "h_xi", "a_xi", "h_manager", "a_manager"]
    merged = matches.merge(tm[key + cols], on=key, how="left")
    merged.index = matches.index
    return merged
