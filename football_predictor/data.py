"""Загрузка и нормализация данных о матчах.

Источник — football-data.co.uk: 5 топ-лиг с сезона 1993/94, включая текущий. Счёт, удары,
угловые, карточки, коэффициенты букмекеров (с ~2000 г.), в свежих сезонах ещё и xG.
Если сайт недоступен, берётся зеркало github.com/datasets/football-datasets (без коэффициентов).
Можно подгрузить и свои CSV в том же формате.
"""
from __future__ import annotations

import csv
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

FD_URL = "https://football-data.co.uk/mmz4281"
MIRROR_URL = "https://raw.githubusercontent.com/datasets/football-datasets/main/datasets"

LEAGUES = ["premier-league", "la-liga", "serie-a", "bundesliga", "ligue-1"]

# Коды дивизионов football-data.co.uk
DIV_TO_LEAGUE = {"E0": "premier-league", "SP1": "la-liga", "I1": "serie-a",
                 "D1": "bundesliga", "F1": "ligue-1"}
LEAGUE_TO_DIV = {v: k for k, v in DIV_TO_LEAGUE.items()}

FIRST_SEASON = 1993


def season_code(start_year: int) -> str:
    """1993 -> '9394', 2025 -> '2526'."""
    return f"{start_year % 100:02d}{(start_year + 1) % 100:02d}"


def season_of(date: pd.Timestamp) -> int:
    """Год начала сезона (сезон начинается в июле)."""
    return date.year if date.month >= 7 else date.year - 1


LAST_SEASON = season_of(pd.Timestamp.today())  # текущий сезон


def _fetch(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def download(leagues=LEAGUES, first=FIRST_SEASON, last=LAST_SEASON,
             cache_dir="data/raw", force=False) -> None:
    """Качает сезоны с football-data.co.uk; если сайт недоступен — с GitHub-зеркала (без коэффициентов).

    Текущий сезон перекачивается всегда: он пополняется каждую неделю.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    current = season_of(pd.Timestamp.today())
    for league in leagues:
        div = LEAGUE_TO_DIV[league]
        for year in range(first, last + 1):
            code = season_code(year)
            path = cache / f"{league}_{code}.csv"
            if path.exists() and not force and year < current:
                continue
            for url in (f"{FD_URL}/{code}/{div}.csv", f"{MIRROR_URL}/{league}/season-{code}.csv"):
                try:
                    path.write_bytes(_fetch(url))
                    print(f"  {league} {code} <- {url.split('/')[2]}")
                    break
                except Exception as exc:
                    err = exc
            else:
                print(f"  пропуск {league} {code}: {err}")


# Коэффициенты в порядке предпочтения: закрывающие (перед стартовым свистком) точнее всего,
# Pinnacle — самый «острый» букмекер, среднее по рынку — следующий по качеству вариант
_ODDS_SETS = [("PSCH", "PSCD", "PSCA"), ("AvgCH", "AvgCD", "AvgCA"), ("PSH", "PSD", "PSA"),
              ("AvgH", "AvgD", "AvgA"), ("BbAvH", "BbAvD", "BbAvA"),
              ("B365CH", "B365CD", "B365CA"), ("B365H", "B365D", "B365A")]


def read_csv(path) -> pd.DataFrame:
    """В части старых файлов строки длиннее заголовка (хвост пустых полей) — pandas их выкидывает.
    Читаем вручную и выравниваем строки по заголовку."""
    with open(path, encoding="latin-1", newline="") as f:
        rows = [r for r in csv.reader(f) if any(x.strip() for x in r)]
    if not rows:
        return pd.DataFrame()
    n = len(rows[0])
    body = [(r + [""] * n)[:n] for r in rows[1:]]
    return pd.DataFrame(body, columns=rows[0]).replace("", np.nan)


def _normalize(df: pd.DataFrame, league: str | None = None, season: int | None = None) -> pd.DataFrame:
    df.columns = df.columns.str.replace("ï»¿", "").str.strip()  # BOM в свежих файлах
    if "HomeTeam" not in df and "HT" in df:  # старые сезоны: HT/AT
        df = df.rename(columns={"HT": "HomeTeam", "AT": "AwayTeam"})
    if not {"HomeTeam", "AwayTeam", "FTHG", "FTAG"} <= set(df.columns):
        return pd.DataFrame()
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
        "hg": pd.to_numeric(df["FTHG"], errors="coerce"),
        "ag": pd.to_numeric(df["FTAG"], errors="coerce"),
    })
    for src, dst in [("HS", "hs"), ("AS", "as"), ("HST", "hst"), ("AST", "ast"),
                     ("HC", "hc"), ("AC", "ac"), ("HR", "hr"), ("AR", "ar"),
                     ("HxG", "hxg"), ("AxG", "axg")]:
        out[dst] = pd.to_numeric(df[src], errors="coerce") if src in df else np.nan

    out["odds_h"] = out["odds_d"] = out["odds_a"] = np.nan
    for h, d, a in _ODDS_SETS:
        if h in df:
            mask = out["odds_h"].isna()
            out.loc[mask, "odds_h"] = pd.to_numeric(df.loc[mask, h], errors="coerce")
            out.loc[mask, "odds_d"] = pd.to_numeric(df.loc[mask, d], errors="coerce")
            out.loc[mask, "odds_a"] = pd.to_numeric(df.loc[mask, a], errors="coerce")
    out = out.dropna(subset=["date", "hg", "ag"])
    out["hg"], out["ag"] = out["hg"].astype(int), out["ag"].astype(int)
    # сезон из имени файла надёжнее даты: сезон 2019/20 доигрывали в июле-августе 2020
    out["season"] = season if season is not None else out["date"].map(season_of)
    return out


def load_matches(cache_dir="data/raw", extra_csv: list[str] | None = None) -> pd.DataFrame:
    frames = []
    for path in sorted(Path(cache_dir).glob("*.csv")):
        league, code = path.stem.rsplit("_", 1)
        year = 1900 + int(code[:2]) if int(code[:2]) >= 90 else 2000 + int(code[:2])
        frames.append(_normalize(read_csv(path), league, year))
    for path in extra_csv or []:
        raw = read_csv(path)
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
