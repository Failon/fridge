"""Загрузка и нормализация данных о матчах.

Источник по умолчанию — открытый датасет github.com/datasets/football-datasets
(зеркало football-data.co.uk: 5 топ-лиг с сезона 1993/94, счёт, удары, угловые,
карточки). Коэффициенты букмекеров берутся из архива github.com/jokecamp/FootballData
(дампы football-data.co.uk, MIT): АПЛ 2002/03 – январь 2021, остальные лиги 2002/03 –
декабрь 2013. Можно подгрузить и свои CSV в формате football-data.co.uk.
"""
from __future__ import annotations

import io
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets"
ODDS_URL = "https://raw.githubusercontent.com/jokecamp/FootballData/master/football-data.co.uk"

LEAGUES = ["premier-league", "la-liga", "serie-a", "bundesliga", "ligue-1"]

# Коды football-data.co.uk -> наши названия лиг (для пользовательских CSV)
DIV_TO_LEAGUE = {"E0": "premier-league", "SP1": "la-liga", "I1": "serie-a",
                 "D1": "bundesliga", "F1": "ligue-1"}

FIRST_SEASON, LAST_SEASON = 1993, 2025


def season_code(start_year: int) -> str:
    """1993 -> '9394', 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_of(date: pd.Timestamp) -> int:
    """Год начала сезона (сезон начинается в июле)."""
    return date.year if date.month >= 7 else date.year - 1


def download(leagues=LEAGUES, first=FIRST_SEASON, last=LAST_SEASON,
             cache_dir="data/raw", force=False) -> None:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for league in leagues:
        for year in range(first, last + 1):
            path = cache / f"{league}_{season_code(year)}.csv"
            if path.exists() and not force:
                continue
            url = f"{BASE_URL}/{league}/season-{season_code(year)}.csv"
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    path.write_bytes(resp.read())
                print(f"  скачано {league} {season_code(year)}")
            except Exception as exc:  # сезона может не быть в датасете
                print(f"  пропуск {league} {season_code(year)}: {exc}")


# Файлы архива jokecamp: Англия разложена по папкам сезонов, остальные — "SP1 (N).csv"
_ODDS_COUNTRIES = {"la-liga": ("spain", "SP1"), "serie-a": ("italy", "I1"),
                   "bundesliga": ("germany", "D1"), "ligue-1": ("france", "F1")}


def _odds_candidates(league: str) -> list[str]:
    if league == "premier-league":
        folders = [f"{y}-{y + 1}" for y in range(2002, 2020)] + ["2020-2021 (Until Jan 22)"]
        return [f"england/{f}/Premier.csv" for f in folders]
    country, div = _ODDS_COUNTRIES[league]
    return [f"{country}/{div}.csv"] + [f"{country}/{div} ({k}).csv" for k in range(1, 46)]


def download_odds(leagues=LEAGUES, cache_dir="data/odds", force=False) -> None:
    """Скачивает сезоны с коэффициентами; сезон определяется по датам внутри файла."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for league in leagues:
        if not force and any(cache.glob(f"{league}_*.csv")):
            continue
        for rel in _odds_candidates(league):
            try:
                with urllib.request.urlopen(f"{ODDS_URL}/{urllib.parse.quote(rel)}", timeout=30) as resp:
                    raw = resp.read()
            except Exception:
                continue  # нумерация файлов с пропусками
            df = _normalize(pd.read_csv(io.BytesIO(raw), encoding="latin-1", on_bad_lines="skip"), league)
            if df.empty or df["odds_h"].notna().mean() < 0.5:
                continue  # сезоны до 2002/03 — без коэффициентов
            season = int(df["season"].mode()[0])
            (cache / f"{league}_{season_code(season)}.csv").write_bytes(raw)
            print(f"  коэффициенты {league} {season_code(season)}: {len(df)} матчей")


# Колонки с коэффициентами в порядке предпочтения: закрывающие Pinnacle — самые точные
_ODDS_SETS = [("PSCH", "PSCD", "PSCA"), ("PSH", "PSD", "PSA"), ("AvgH", "AvgD", "AvgA"),
              ("BbAvH", "BbAvD", "BbAvA"), ("B365H", "B365D", "B365A")]


def _normalize(df: pd.DataFrame, league: str | None = None) -> pd.DataFrame:
    df = df.dropna(subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]).copy()
    if df.empty:
        return pd.DataFrame()
    dates = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
    for fmt in ("%d/%m/%y", "%d/%m/%Y"):  # формат football-data.co.uk
        if dates.isna().any():
            dates = dates.fillna(pd.to_datetime(df["Date"], format=fmt, errors="coerce"))
    out = pd.DataFrame({
        "date": dates,
        "league": league if league else df["Div"].map(DIV_TO_LEAGUE).fillna(df["Div"]),
        "home": df["HomeTeam"].astype(str).str.strip(),
        "away": df["AwayTeam"].astype(str).str.strip(),
        "hg": df["FTHG"].astype(int),
        "ag": df["FTAG"].astype(int),
    })
    for src, dst in [("HS", "hs"), ("AS", "as"), ("HST", "hst"), ("AST", "ast"),
                     ("HC", "hc"), ("AC", "ac"), ("HR", "hr"), ("AR", "ar")]:
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df else np.nan

    out["odds_h"] = out["odds_d"] = out["odds_a"] = np.nan
    for h, d, a in _ODDS_SETS:
        if h in df:
            mask = out["odds_h"].isna()
            out.loc[mask, "odds_h"] = pd.to_numeric(df.loc[mask, h], errors="coerce")
            out.loc[mask, "odds_d"] = pd.to_numeric(df.loc[mask, d], errors="coerce")
            out.loc[mask, "odds_a"] = pd.to_numeric(df.loc[mask, a], errors="coerce")
    out = out.dropna(subset=["date"])
    out["season"] = out["date"].map(season_of)
    return out


def load_matches(cache_dir="data/raw", extra_csv: list[str] | None = None,
                 odds_dir="data/odds") -> pd.DataFrame:
    frames = []
    paths = sorted(Path(cache_dir).glob("*.csv")) + sorted(Path(odds_dir).glob("*.csv"))
    for path in paths:
        league = path.stem.rsplit("_", 1)[0]
        frames.append(_normalize(pd.read_csv(path, encoding="latin-1", on_bad_lines="skip"), league))
    for path in extra_csv or []:
        raw = pd.read_csv(path, encoding="latin-1")
        frames.append(_normalize(raw, None if "Div" in raw else Path(path).stem))
    if not frames:
        raise FileNotFoundError(f"Нет данных в {cache_dir}. Сначала выполните `download`.")
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)

    # Пользовательские CSV могут дублировать матчи из датасета — оставляем запись с коэффициентами
    df["_has_odds"] = df["odds_h"].notna()
    df = (df.sort_values(["date", "league", "home", "_has_odds"])
            .drop_duplicates(["date", "league", "home", "away"], keep="last")
            .drop(columns="_has_odds"))
    df["result"] = np.select([df.hg > df.ag, df.hg == df.ag], [0, 1], 2)  # 0=П1, 1=X, 2=П2
    return df.sort_values(["date", "league", "home"]).reset_index(drop=True)
