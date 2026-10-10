from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

# Market keys
ML2 = "ML2"   # 2-way winner (home / away), incl. overtime
X12 = "1X2"   # 3-way full time (home / draw / away)

OUTCOMES = {ML2: ("home", "away"), X12: ("home", "draw", "away")}


@dataclass
class Price:
    odds: float                 # decimal odds, fees already applied
    raw: float | None = None    # source price (e.g. Polymarket probability)
    limit: float | None = None  # max stake/liquidity in USD, if known
    ref: str | None = None      # source-specific id (token id, outcome id)


@dataclass
class Event:
    book: str
    sport: str
    home: str
    away: str
    start: datetime | None
    league: str = ""
    url: str = ""
    event_id: str = ""
    markets: dict[str, dict[str, Price]] = field(default_factory=dict)

    @property
    def title(self) -> str:
        return f"{self.home} vs {self.away}"


@dataclass
class Leg:
    book: str
    outcome: str
    odds: float
    stake: float
    team: str
    url: str = ""
    limit: float | None = None
    ref: str | None = None


@dataclass
class Arb:
    sport: str
    title: str
    league: str
    start: datetime | None
    market: str
    profit_pct: float
    legs: list[Leg]
    bankroll: float

    @property
    def key(self) -> str:
        return f"{self.title}|{self.market}|" + "|".join(f"{l.book}:{l.outcome}" for l in self.legs)
