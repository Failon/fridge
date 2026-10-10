from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

from ..models import ML2, X12, Event, Price
from .base import Adapter

GAMES = [
    ("soccer", "Arsenal FC", "Chelsea", X12, (2.10, 3.50, 3.60)),
    ("soccer", "Real Madrid", "FC Barcelona", X12, (2.40, 3.60, 2.85)),
    ("basketball", "Los Angeles Lakers", "Boston Celtics", ML2, (2.05, 1.80)),
    ("ice-hockey", "Toronto Maple Leafs", "Montreal Canadiens", ML2, (1.70, 2.20)),
    ("tennis", "Carlos Alcaraz", "Jannik Sinner", ML2, (1.95, 1.90)),
]
ALT = {"Arsenal FC": "Arsenal", "FC Barcelona": "Barcelona", "Los Angeles Lakers": "Lakers",
       "Boston Celtics": "Celtics", "Toronto Maple Leafs": "Maple Leafs", "Montreal Canadiens": "Canadiens"}


class MockAdapter(Adapter):
    """Random-walk odds for demo/testing without network access."""

    async def fetch(self) -> list[Event]:
        rnd = random.Random()
        now = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
        out = []
        for i, (sport, home, away, mkt, base) in enumerate(GAMES):
            names = (ALT.get(home, home), ALT.get(away, away)) if self.book == "polymarket" else (home, away)
            odds = [round(o * rnd.uniform(0.9, 1.12), 2) for o in base]
            sides = dict(zip(("home", "draw", "away") if mkt == X12 else ("home", "away"), odds))
            if self.book == "polymarket":  # Polymarket lists the away team first for US sports
                names = names[::-1] if sport != "soccer" else names
                sides = {({"home": "away", "away": "home"}.get(k, k) if sport != "soccer" else k): v
                         for k, v in sides.items()}
            ev = Event(book=self.book, sport=sport, home=names[0], away=names[1],
                       start=now + timedelta(hours=5 + i), league="Mock League",
                       url=f"https://example.com/{self.book}/{i}", event_id=str(i))
            ev.markets[mkt] = {k: Price(odds=v, limit=rnd.choice([200, 500, 1000])) for k, v in sides.items()}
            out.append(ev)
        return out
