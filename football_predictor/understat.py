"""xG по матчам с understat.com (5 топ-лиг с сезона 2014/15).

Understat отдаёт данные сезона JSON-ом по адресу getLeagueData/<лига>/<год>. Названия команд
там полные («Manchester United»), а в football-data — короткие («Man United»), поэтому
соответствие строится автоматически: по совпадению счёта и даты матчей одного сезона.
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from .data import LEAGUES, season_of

URL = "https://understat.com/getLeagueData/{league}/{year}"
LEAGUE_CODES = {"premier-league": "EPL", "la-liga": "La_liga", "serie-a": "Serie_A",
                "bundesliga": "Bundesliga", "ligue-1": "Ligue_1"}
FIRST_SEASON = 2014


def _fetch(league_code: str, year: int) -> dict:
    req = urllib.request.Request(URL.format(league=league_code, year=year), headers={
        "User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{league_code}/{year}"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        raw = resp.read()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def download(leagues=LEAGUES, cache_dir="data/understat", force=False, pause=1.0) -> None:
    """Сезоны в прошлом качаются один раз, текущий — всегда (он пополняется)."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    current = season_of(pd.Timestamp.today())
    for league in leagues:
        for year in range(FIRST_SEASON, current + 1):
            path = cache / f"{league}_{year}.json"
            if path.exists() and not force and year < current:
                continue
            try:
                dates = _fetch(LEAGUE_CODES[league], year)["dates"]
                path.write_text(json.dumps(dates))
                print(f"  xG {league} {year}/{(year + 1) % 100:02d}: {sum(d['isResult'] for d in dates)} матчей")
            except Exception as exc:
                print(f"  пропуск xG {league} {year}: {exc}")
            time.sleep(pause)  # не нагружаем сайт


def load(cache_dir="data/understat") -> pd.DataFrame:
    rows = []
    for path in sorted(Path(cache_dir).glob("*.json")):
        league, year = path.stem.rsplit("_", 1)
        for d in json.loads(path.read_text()):
            if not d.get("isResult"):
                continue
            rows.append({"league": league, "season": int(year), "date": pd.Timestamp(d["datetime"]).normalize(),
                         "home": d["h"]["title"], "away": d["a"]["title"],
                         "hg": int(d["goals"]["h"]), "ag": int(d["goals"]["a"]),
                         "hxg": float(d["xG"]["h"]), "axg": float(d["xG"]["a"])})
    return pd.DataFrame(rows)


def _name_map(us: pd.DataFrame, fd: pd.DataFrame) -> dict:
    """{(лига, имя understat): имя football-data} — голосованием по матчам с тем же счётом в ±1 день."""
    votes = defaultdict(Counter)
    fd_idx = defaultdict(list)
    for r in fd.itertuples():
        fd_idx[(r.league, r.season, r.hg, r.ag)].append(r)
    for r in us.itertuples():
        for c in fd_idx.get((r.league, r.season, r.hg, r.ag), []):
            if abs((c.date - r.date).days) <= 1:
                votes[(r.league, r.home)][c.home] += 1
                votes[(r.league, r.away)][c.away] += 1
    return {k: v.most_common(1)[0][0] for k, v in votes.items() if v.most_common(1)[0][1] >= 3}


def attach_xg(matches: pd.DataFrame, cache_dir="data/understat") -> pd.DataFrame:
    """Добавляет hxg/axg из Understat туда, где их нет (football-data даёт xG только с 2026/27)."""
    us = load(cache_dir)
    if us.empty:
        return matches
    names = _name_map(us, matches)
    us["home"] = [names.get((lg, t)) for lg, t in zip(us.league, us.home)]
    us["away"] = [names.get((lg, t)) for lg, t in zip(us.league, us.away)]
    us = us.dropna(subset=["home", "away"]).drop_duplicates(["league", "season", "home", "away"])
    key = ["league", "season", "home", "away"]
    merged = matches.merge(us[key + ["hxg", "axg"]], on=key, how="left", suffixes=("", "_us"))
    for c in ("hxg", "axg"):
        merged[c] = merged[c].fillna(merged.pop(f"{c}_us")) if c in matches else merged.pop(f"{c}_us")
    merged.index = matches.index
    return merged
