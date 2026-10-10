from __future__ import annotations

import asyncio
import json
import re

from ..matcher import team_score
from ..models import ML2, X12, Event, Price
from .base import Adapter, as_float, log, parse_time

GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"

# our sport slug -> Polymarket tag slugs
DEFAULT_TAGS = {
    "soccer": ["soccer"],
    "basketball": ["nba", "basketball"],
    "ice-hockey": ["nhl", "hockey"],
    "american-football": ["nfl", "cfb"],
    "baseball": ["mlb"],
    "tennis": ["tennis"],
    "mma": ["ufc", "mma"],
    "counter-strike": ["counter-strike-2", "cs2"],
    "league-of-legends": ["league-of-legends"],
    "dota-2": ["dota-2"],
}
VS = re.compile(r"\s+(?:vs\.?|v\.?|@)\s+", re.I)


def _list(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return []
    return v or []


class PolymarketAdapter(Adapter):
    """Public Gamma API (no auth). Prices = best ask, converted to decimal odds."""

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.fee = float(cfg.get("taker_fee", 0.0))  # fraction of payout lost to fees
        self.tags = {**DEFAULT_TAGS, **(cfg.get("tags") or {})}

    def odds(self, p: float | None) -> float | None:
        if p is None or not 0.01 <= p <= 0.99:
            return None
        return (1 - self.fee) / p

    async def fetch(self) -> list[Event]:
        jobs = [(sport, tag) for sport in self.sports for tag in self.tags.get(sport, [])]
        pages = await asyncio.gather(*(self._events(tag) for _, tag in jobs), return_exceptions=True)
        out, seen = [], set()
        for (sport, tag), res in zip(jobs, pages):
            if isinstance(res, Exception):
                log.warning("polymarket tag %s: %s", tag, res)
                continue
            for raw in res:
                if raw.get("id") in seen:
                    continue
                ev = self.parse_event(raw, sport)
                if ev:
                    seen.add(raw.get("id"))
                    out.append(ev)
        return out

    async def _events(self, tag: str) -> list[dict]:
        res, offset = [], 0
        while offset < 2000:
            page = await self.get_json(f"{GAMMA}/events", params={
                "tag_slug": tag, "active": "true", "closed": "false",
                "limit": 500, "offset": offset})
            res += page
            if len(page) < 500:
                break
            offset += 500
        return res

    def parse_event(self, raw: dict, sport: str) -> Event | None:
        teams = VS.split(raw.get("title", ""), maxsplit=1)
        if len(teams) != 2:
            return None  # futures / props, not a single game
        home, away = (t.strip() for t in teams)
        markets = [m for m in raw.get("markets", []) if m.get("active") and not m.get("closed")
                   and m.get("acceptingOrders", True)]
        start = parse_time(raw.get("startTime") or raw.get("gameStartTime")
                           or next((m.get("gameStartTime") for m in markets if m.get("gameStartTime")), None))
        ev = Event(book=self.book, sport=sport, home=home, away=away, start=start,
                   league=(raw.get("series") or [{}])[0].get("title", "") if raw.get("series") else "",
                   url=f"https://polymarket.com/event/{raw.get('slug', '')}", event_id=str(raw.get("id")))

        ml = self._moneyline(markets, home, away)
        if ml:
            ev.markets[ML2] = ml
        x12 = self._three_way(markets, home, away)
        if x12:
            ev.markets[X12] = x12
        return ev if ev.markets else None

    def _moneyline(self, markets, home, away):
        for m in markets:
            outs, toks = _list(m.get("outcomes")), _list(m.get("clobTokenIds"))
            if len(outs) != 2 or len(toks) != 2 or {o.lower() for o in outs} == {"yes", "no"}:
                continue
            if m.get("sportsMarketType") not in (None, "moneyline"):
                continue
            ask0, bid0 = as_float(m.get("bestAsk")), as_float(m.get("bestBid"))
            asks = [ask0, (1 - bid0) if bid0 else None]
            sides = {}
            for out, tok, ask in zip(outs, toks, asks):
                side = "home" if team_score(out, home) >= team_score(out, away) else "away"
                o = self.odds(ask)
                if o and max(team_score(out, home), team_score(out, away)) >= 70:
                    sides[side] = Price(odds=o, raw=ask, ref=tok)
            if len(sides) == 2:
                return sides
        return None

    def _three_way(self, markets, home, away):
        """Soccer: three Yes/No markets (home win, draw, away win)."""
        sides = {}
        for m in markets:
            outs, toks = _list(m.get("outcomes")), _list(m.get("clobTokenIds"))
            if [o.lower() for o in outs] != ["yes", "no"] or len(toks) != 2:
                continue
            label = m.get("groupItemTitle") or m.get("question", "")
            if label.lower().startswith("draw") or "end in a draw" in label.lower():
                side = "draw"
            else:
                sh, sa = team_score(label, home), team_score(label, away)
                if max(sh, sa) < 70:
                    continue
                side = "home" if sh >= sa else "away"
            o = self.odds(as_float(m.get("bestAsk")))
            if o:
                sides[side] = Price(odds=o, raw=as_float(m.get("bestAsk")), ref=toks[0])
        return sides if len(sides) == 3 else None

    async def book_ask(self, token_id: str) -> tuple[float, float] | None:
        """Live best ask and USD liquidity at that level from the CLOB order book."""
        book = await self.get_json(f"{CLOB}/book", params={"token_id": token_id})
        asks = [(float(a["price"]), float(a["size"])) for a in book.get("asks", [])]
        if not asks:
            return None
        price = min(p for p, _ in asks)
        usd = sum(p * s for p, s in asks if p <= price + 1e-9)
        return price, usd
