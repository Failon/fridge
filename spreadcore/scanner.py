from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .adapters import PolymarketAdapter, build
from .adapters.base import log
from .arb import find_arbs, stakes_for
from .matcher import group_events
from .models import Arb, Event, Price
from .notify import Telegram, format_arb

LIVE_TTL = 120  # websocket prices newer than this override the next REST snapshot


@dataclass
class State:
    arbs: list[Arb] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    streams: dict[str, str] = field(default_factory=dict)
    matched: int = 0
    ticks: int = 0
    updated: datetime | None = None


class Scanner:
    """REST snapshots discover events; websocket streams keep matched prices live.

    Every price tick re-evaluates only the match groups containing that price, so an arb
    is detected (and alerted) within milliseconds of the move instead of at the next poll.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg
        s = cfg.get("scan", {})
        self.interval = float(s.get("interval", 15))
        self.bankroll = float(s.get("bankroll", 100))
        self.min_profit = float(s.get("min_profit", 0.5))
        self.max_profit = float(s.get("max_profit", 15))  # higher is almost always a matching error
        self.require_pm = s.get("mode", "polymarket") == "polymarket"
        self.verify = bool(s.get("verify_polymarket", True))
        self.threshold = float(s.get("match_threshold", 86))
        self.window = float(s.get("time_window_hours", 3))
        self.cooldown = float(cfg.get("telegram", {}).get("cooldown", 600))
        sports = s.get("sports")
        self.adapters = []
        for name, acfg in (cfg.get("books") or {}).items():
            if acfg.get("enabled", True):
                if sports and "sports" not in acfg:
                    acfg["sports"] = sports
                self.adapters.append(build(name, acfg))
        self.tg = Telegram(cfg.get("telegram", {}))
        self.state = State()
        self.listeners: set[asyncio.Queue] = set()
        self._sent: dict[str, tuple[float, float]] = {}
        self._events: dict[str, list[Event]] = {}
        self._groups: list[list[Event]] = []
        self._index: dict[tuple[str, str], list[tuple[Price, int]]] = {}
        self._live: dict[tuple[str, str], tuple] = {}
        self._group_arbs: dict[int, list[Arb]] = {}
        self._dirty: set[int] = set()
        self._wake = asyncio.Event()

    @property
    def _pm(self) -> PolymarketAdapter | None:
        return next((a for a in self.adapters if isinstance(a, PolymarketAdapter)), None)

    # ---- snapshots ------------------------------------------------------------------------

    async def _fetch(self, a):
        t = time.monotonic()
        try:
            evs = await a.fetch()
            self.state.errors.pop(a.name, None)
            log.info("%-10s %4d events in %.1fs", a.name, len(evs), time.monotonic() - t)
            return evs
        except Exception as e:  # noqa: BLE001
            self.state.errors[a.name] = str(e)[:300]
            log.warning("%s failed: %s", a.name, e)
            return []

    def _regroup(self):
        events = [e for evs in self._events.values() for e in evs]
        groups = group_events(events, self.threshold, self.window)
        if self.require_pm:
            groups = [g for g in groups if any(e.book == "polymarket" for e in g)]
        now = time.time()
        index: dict[tuple[str, str], list[tuple[Price, int]]] = {}
        for gi, g in enumerate(groups):
            for e in g:
                for outs in e.markets.values():
                    for p in outs.values():
                        if p.ref:
                            index.setdefault((e.book, p.ref), []).append((p, gi))
                            live = self._live.get((e.book, p.ref))
                            if live and now - live[3] < LIVE_TTL:
                                self._apply(p, live)
        self._groups, self._index = groups, index
        self._group_arbs = {}
        self._dirty = set(range(len(groups)))
        self.state.counts = {name: len(evs) for name, evs in self._events.items()}
        self.state.matched = len(groups)
        self._wake.set()

    async def _poll(self, a):
        interval = float(a.cfg.get("interval", self.interval))
        while True:
            t = time.monotonic()
            self._events[a.name] = await self._fetch(a)
            self._regroup()
            await asyncio.sleep(max(1.0, interval - (time.monotonic() - t)))

    # ---- live updates ---------------------------------------------------------------------

    @staticmethod
    def _apply(p: Price, upd):
        odds, raw, limit = upd[:3]
        p.odds = odds or 0.0  # 0 = suspended / no liquidity, ignored by find_arbs
        if raw is not None:
            p.raw = raw
        if limit is not None:
            p.limit = limit
        p.live = True

    def _watched(self, book: str):
        return lambda: [p for (b, _), lst in self._index.items() if b == book for p, _ in lst]

    def _on_update(self, book: str):
        def handler(updates: dict):
            now = time.time()
            for ref, upd in updates.items():
                self._live[(book, ref)] = (*upd, now)
                for p, gi in self._index.get((book, ref), ()):
                    self._apply(p, upd)
                    self._dirty.add(gi)
            self.state.ticks += len(updates)
            self.state.streams[book] = datetime.now(timezone.utc).isoformat()
            if self._dirty:
                self._wake.set()
        return handler

    async def _stream(self, a):
        try:
            await a.stream(self._watched(a.book), self._on_update(a.book))
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            log.warning("%s stream stopped: %s", a.name, e)
        self.state.streams.pop(a.book, None)

    # ---- evaluation -----------------------------------------------------------------------

    async def _evaluate(self, verify: bool):
        dirty, self._dirty = self._dirty, set()
        for gi in dirty:
            if gi < len(self._groups):
                self._group_arbs[gi] = [a for a in find_arbs(self._groups[gi], self.bankroll, self.min_profit,
                                                             self.require_pm)
                                        if a.profit_pct <= self.max_profit]
        arbs = [a for lst in self._group_arbs.values() for a in lst]
        if verify:
            arbs = [a for a in await asyncio.gather(*(self._verify(a) for a in arbs)) if a]
        arbs.sort(key=lambda a: -a.profit_pct)
        self.state.arbs = arbs
        self.state.updated = datetime.now(timezone.utc)
        await self._notify(arbs)
        self._publish()
        return arbs

    async def _evaluator(self):
        while True:
            await self._wake.wait()
            await asyncio.sleep(0.05)  # coalesce bursts of ticks
            self._wake.clear()
            pm = self._pm
            # without a live Polymarket stream, confirm PM legs against the REST order book
            verify = self.verify and pm is not None and "polymarket" not in self.state.streams
            try:
                await self._evaluate(verify)
            except Exception:  # noqa: BLE001
                log.exception("evaluate failed")

    async def _verify(self, arb: Arb) -> Arb | None:
        """Re-price Polymarket legs from the live order book; drop arbs that vanished."""
        pm = self._pm
        legs = [l for l in arb.legs if l.book == "polymarket" and l.ref]
        if not pm or not legs:
            return arb
        try:
            for l in legs:
                live = await pm.book_ask(l.ref)
                if not live or not pm.odds(live[0]):
                    return None
                l.odds, l.limit = pm.odds(live[0]), live[1]
        except Exception as e:  # noqa: BLE001
            log.warning("verify %s: %s", arb.title, e)
            return arb
        profit, stakes = stakes_for([l.odds for l in arb.legs], arb.bankroll)
        if profit * 100 < self.min_profit:
            return None
        arb.profit_pct = round(profit * 100, 3)
        for l, s in zip(arb.legs, stakes):
            l.stake = round(s, 2)
        return arb

    async def _notify(self, arbs: list[Arb]):
        now = time.time()
        for a in arbs:
            prev = self._sent.get(a.key)
            if prev and now - prev[0] < self.cooldown and a.profit_pct < prev[1] + 0.5:
                continue
            self._sent[a.key] = (now, a.profit_pct)
            log.info("ARB %.2f%% %s %s %s", a.profit_pct, a.market, a.title,
                     " | ".join(f"{l.book}:{l.outcome}@{l.odds:.3f}" for l in a.legs))
            await self.tg.send(format_arb(a))

    # ---- dashboard feed -------------------------------------------------------------------

    def state_dict(self) -> dict:
        s = self.state
        return {
            "updated": s.updated.isoformat() if s.updated else None,
            "counts": s.counts, "errors": s.errors, "matched": s.matched,
            "streams": s.streams, "ticks": s.ticks,
            "arbs": [{
                "sport": a.sport, "title": a.title, "league": a.league, "market": a.market,
                "start": a.start.isoformat() if a.start else None, "profit": a.profit_pct,
                "legs": [{"book": l.book, "team": l.team, "odds": round(l.odds, 3),
                          "url": l.url, "limit": l.limit} for l in a.legs],
            } for a in s.arbs],
        }

    def _publish(self):
        if not self.listeners:
            return
        msg = json.dumps(self.state_dict(), ensure_ascii=False)
        for q in list(self.listeners):
            if q.full():  # slow client: drop the stale frame, keep the newest
                q.get_nowait()
            q.put_nowait(msg)

    # ---- entry points ---------------------------------------------------------------------

    async def scan_once(self) -> list[Arb]:
        results = await asyncio.gather(*(self._fetch(a) for a in self.adapters))
        self._events = {a.name: r for a, r in zip(self.adapters, results)}
        self._regroup()
        return await self._evaluate(self.verify)

    async def run(self):
        tasks = [self._evaluator()]
        for a in self.adapters:
            tasks.append(self._poll(a))
            if a.streaming:
                tasks.append(self._stream(a))
        await asyncio.gather(*tasks)

    async def close(self):
        for a in self.adapters:
            await a.close()
