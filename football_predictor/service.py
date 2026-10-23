"""Прогноз по сохранённой модели — общий код для CLI и HTTP API."""
from __future__ import annotations

import difflib
import pickle
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from .dixon_coles import markets, score_matrix
from .model import rho_of

VALUE_EDGE = 0.03  # минимальное преимущество (EV), чтобы пометить ставку как value


class UnknownLeague(LookupError):
    def __init__(self, league: str, available: list[str]):
        super().__init__(f"Нет данных по лиге '{league}'. Доступны: {', '.join(available)}")
        self.available = available


class UnknownTeam(LookupError):
    def __init__(self, team: str, league: str, suggestions: list[str]):
        super().__init__(f"Команда '{team}' не найдена в {league}. Похожие: {', '.join(suggestions) or '—'}")
        self.suggestions = suggestions


class Predictor:
    def __init__(self, bundle: dict):
        self.model = bundle["model"]
        self.engine = bundle["engine"]
        self.last_date: pd.Timestamp = bundle["last_date"]
        # features() может менять состояние движка (переход сезона) — сериализуем доступ
        self._lock = threading.Lock()

    @classmethod
    def load(cls, path: str | Path) -> "Predictor":
        with open(path, "rb") as f:
            return cls(pickle.load(f))

    @property
    def uses_market(self) -> bool:
        return "market" in self.model.members

    def leagues(self) -> list[str]:
        return sorted({lg for lg, _ in self.engine.teams})

    def teams(self, league: str, active_days: int = 365) -> list[tuple[str, float]]:
        """Команды лиги, игравшие за последний год, по убыванию Elo."""
        if league not in self.leagues():
            raise UnknownLeague(league, self.leagues())
        last = max(s.last_date for s in self.engine.teams.values())
        rows = [(t, s.elo) for (lg, t), s in self.engine.teams.items()
                if lg == league and (last - s.last_date).days < active_days]
        return sorted(rows, key=lambda r: -r[1])

    def resolve_team(self, league: str, name: str) -> str:
        if league not in self.leagues():
            raise UnknownLeague(league, self.leagues())
        known = sorted({t for (lg, t) in self.engine.teams if lg == league})
        if name in known:
            return name
        lower = {t.lower(): t for t in known}
        if name.lower() in lower:
            return lower[name.lower()]
        close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
        if close:
            return close[0]
        active = [t for t, _ in self.teams(league)]  # подсказываем только действующие команды
        raise UnknownTeam(name, league, difflib.get_close_matches(name, active, 5, 0.3))

    def predict(self, league: str, home: str, away: str, date: pd.Timestamp | None = None,
                odds: tuple[float, float, float] | None = None) -> dict:
        date = date if date is not None else pd.Timestamp.today().normalize()
        home, away = self.resolve_team(league, home), self.resolve_team(league, away)
        if home == away:
            raise ValueError("Хозяева и гости — одна и та же команда")
        with self._lock:
            feats = self.engine.features(league, home, away, date, odds, refit=False)
        X = pd.DataFrame([feats])
        p = self.model.predict_proba(X)[0]
        lh, la = (float(v[0]) for v in self.model.predict_goals(X))
        mtx = score_matrix(lh, la, float(rho_of(X)[0]))
        mk = markets(mtx)
        top = sorted(((float(mtx[i, j]), f"{i}:{j}") for i in range(6) for j in range(6)), reverse=True)[:5]
        result = {
            "league": league, "home": home, "away": away, "date": date.date().isoformat(),
            "p_home": float(p[0]), "p_draw": float(p[1]), "p_away": float(p[2]),
            "fair_odds": {"home": 1 / p[0], "draw": 1 / p[1], "away": 1 / p[2]},
            "xg_home": lh, "xg_away": la,
            "over_1_5": mk["over_1_5"], "over_2_5": mk["over_2_5"], "over_3_5": mk["over_3_5"],
            "btts": mk["btts"], "top_scores": [{"score": s, "p": pr} for pr, s in top],
            "members": {m: [float(x) for x in v[0]] for m, v in self.model.member_probs(X).items()},
            "value": None,
        }
        if odds:
            result["value"] = value_bets((p[0], p[1], p[2]), odds)
        return result


def value_bets(probs, odds) -> list[dict]:
    """EV и размер ставки по 1/4 Келли (консервативно) для каждого исхода."""
    out = []
    for outcome, p, o in zip(("home", "draw", "away"), probs, odds):
        ev = float(p * o - 1)
        kelly = max(ev / (o - 1), 0.0) / 4 if o > 1 else 0.0
        out.append({"outcome": outcome, "odds": float(o), "ev": ev, "stake_quarter_kelly": float(kelly),
                    "is_value": ev > VALUE_EDGE})
    return out


def as_jsonable(obj, digits: int = 4):
    """numpy-типы -> обычные python-числа, округлённые (для JSON)."""
    if isinstance(obj, dict):
        return {k: as_jsonable(v, digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v, digits) for v in obj]
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float):
        return round(obj, digits)
    return obj
