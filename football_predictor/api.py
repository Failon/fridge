"""HTTP API прогнозов (FastAPI).

Запуск:  python -m football_predictor serve        (или uvicorn football_predictor.api:app)
Сайт: http://127.0.0.1:8000/   Документация API: http://127.0.0.1:8000/docs

Модель берётся из FOOTBALL_MODEL_PATH (по умолчанию models/predictor.pkl).
"""
from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from datetime import date as Date
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from . import data
from .service import Predictor, UnknownLeague, UnknownTeam, as_jsonable

MAX_BATCH = 200
FIXTURES_TTL = 30 * 60  # секунд: расписание и коэффициенты обновляются нечасто
WEB_DIR = Path(__file__).parent / "web"


# ---------- схемы ----------
class Odds(BaseModel):
    home: float = Field(gt=1, description="коэффициент на П1")
    draw: float = Field(gt=1, description="коэффициент на ничью")
    away: float = Field(gt=1, description="коэффициент на П2")


class MatchRequest(BaseModel):
    league: str = Field(examples=["premier-league"])
    home: str = Field(examples=["Arsenal"])
    away: str = Field(examples=["Man City"])
    date: Date | None = Field(None, description="дата матча, по умолчанию сегодня")
    odds: Odds | None = Field(None, description="коэффициенты букмекера — для расчёта value")

    @field_validator("league", "home", "away")
    @classmethod
    def not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("пустое значение")
        return v.strip()


class BatchRequest(BaseModel):
    matches: list[MatchRequest] = Field(min_length=1, max_length=MAX_BATCH)


class FairOdds(BaseModel):
    home: float
    draw: float
    away: float


class ScoreProb(BaseModel):
    score: str
    p: float


class ValueBet(BaseModel):
    outcome: str
    odds: float
    ev: float = Field(description="ожидаемая доходность ставки: p·k − 1")
    stake_quarter_kelly: float = Field(description="доля банка по 1/4 Келли")
    is_value: bool


class Prediction(BaseModel):
    league: str
    home: str = Field(description="название команды, как оно в данных")
    away: str
    date: Date
    p_home: float
    p_draw: float
    p_away: float
    fair_odds: FairOdds
    xg_home: float
    xg_away: float
    over_1_5: float
    over_2_5: float
    over_3_5: float
    btts: float
    top_scores: list[ScoreProb]
    members: dict[str, list[float]] = Field(description="П1/X/П2 от каждой модели ансамбля")
    value: list[ValueBet] | None


class BatchItem(BaseModel):
    request: MatchRequest
    prediction: Prediction | None = None
    error: str | None = None


class Team(BaseModel):
    name: str
    elo: float


class Fixture(BaseModel):
    league: str
    date: Date
    time: str | None
    home: str
    away: str
    odds: Odds | None
    prediction: Prediction | None = None
    error: str | None = None


class Health(BaseModel):
    status: str
    data_until: Date
    leagues: list[str]
    ensemble: dict[str, float]
    uses_market_features: bool


# ---------- приложение ----------
@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.environ.get("FOOTBALL_MODEL_PATH", "models/predictor.pkl")
    if not os.path.exists(path):
        raise RuntimeError(f"Модель {path} не найдена. Выполните `python -m football_predictor train`.")
    app.state.predictor = Predictor.load(path)
    app.state.fixtures = {"at": 0.0, "items": None}
    app.state.fixtures_lock = threading.Lock()
    yield


app = FastAPI(title="Football Predictor", version="1.0",
              description="Прогнозы футбольных матчей: П1/X/П2, голы, тоталы, точный счёт, value-ставки.",
              lifespan=lifespan)


def _predictor(request: Request) -> Predictor:
    return request.app.state.predictor


def _run(pred: Predictor, m: MatchRequest) -> dict:
    odds = (m.odds.home, m.odds.draw, m.odds.away) if m.odds else None
    date = pd.Timestamp(m.date) if m.date else None
    return as_jsonable(pred.predict(m.league, m.home, m.away, date, odds))


@app.get("/health", response_model=Health)
def health(request: Request):
    pred = _predictor(request)
    return Health(status="ok", data_until=pred.last_date.date(), leagues=pred.leagues(),
                  ensemble={m: round(float(w), 4) for m, w in zip(pred.model.members, pred.model.weights)},
                  uses_market_features=pred.uses_market)


@app.get("/leagues", response_model=list[str])
def leagues(request: Request):
    return _predictor(request).leagues()


@app.get("/leagues/{league}/teams", response_model=list[Team])
def teams(league: str, request: Request):
    try:
        return [Team(name=t, elo=round(e, 1)) for t, e in _predictor(request).teams(league)]
    except UnknownLeague as exc:
        raise HTTPException(404, str(exc))


@app.post("/predict", response_model=Prediction)
def predict(m: MatchRequest, request: Request):
    try:
        return _run(_predictor(request), m)
    except (UnknownLeague, UnknownTeam) as exc:
        raise HTTPException(404, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))


@app.post("/predict/batch", response_model=list[BatchItem])
def predict_batch(batch: BatchRequest, request: Request):
    """Ошибка в одном матче не роняет весь запрос — она попадает в поле error."""
    pred = _predictor(request)
    out = []
    for m in batch.matches:
        try:
            out.append(BatchItem(request=m, prediction=_run(pred, m)))
        except (LookupError, ValueError) as exc:
            out.append(BatchItem(request=m, error=str(exc)))
    return out


@app.get("/fixtures", response_model=list[Fixture])
def fixtures(request: Request):
    """Ближайшие матчи 5 лиг (football-data.co.uk) с коэффициентами и готовыми прогнозами."""
    st = request.app.state
    with st.fixtures_lock:
        if st.fixtures["items"] is not None and time.time() - st.fixtures["at"] < FIXTURES_TTL:
            return st.fixtures["items"]
        try:
            fx = data.fetch_fixtures()
        except Exception as exc:
            if st.fixtures["items"] is not None:  # сайт недоступен — отдаём последнее известное
                return st.fixtures["items"]
            raise HTTPException(503, f"Не удалось загрузить расписание: {exc}")
        pred, items = _predictor(request), []
        for f in fx.to_dict("records"):
            odds = (Odds(home=f["odds_h"], draw=f["odds_d"], away=f["odds_a"])
                    if pd.notna(f["odds_h"]) and min(f["odds_h"], f["odds_d"], f["odds_a"]) > 1 else None)
            m = MatchRequest(league=f["league"], home=f["home"], away=f["away"], date=f["date"].date(), odds=odds)
            kickoff = f["time"] if isinstance(f["time"], str) and f["time"].strip() else None  # пустое время -> NaN
            item = Fixture(league=m.league, date=m.date, time=kickoff, home=m.home, away=m.away, odds=odds)
            try:
                item.prediction = Prediction(**_run(pred, m))
            except (LookupError, ValueError) as exc:
                item.error = str(exc)
            items.append(item)
        st.fixtures = {"at": time.time(), "items": items}
        return items


@app.get("/", include_in_schema=False)
def site():
    return FileResponse(WEB_DIR / "index.html")
