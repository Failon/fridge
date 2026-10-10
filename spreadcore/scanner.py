from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .adapters import PolymarketAdapter, build
from .adapters.base import log
from .arb import find_arbs, stakes_for
from .matcher import group_events
from .models import Arb
from .notify import Telegram, format_arb


@dataclass
class State:
    arbs: list[Arb] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    errors: dict[str, str] = field(default_factory=dict)
    matched: int = 0
    updated: datetime | None = None


class Scanner:
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
        self._sent: dict[str, tuple[float, float]] = {}

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

    async def scan_once(self) -> list[Arb]:
        results = await asyncio.gather(*(self._fetch(a) for a in self.adapters))
        events = [e for evs in results for e in evs]
        self.state.counts = {a.name: len(r) for a, r in zip(self.adapters, results)}
        groups = group_events(events, self.threshold, self.window)
        arbs = [a for g in groups for a in find_arbs(g, self.bankroll, self.min_profit, self.require_pm)
                if a.profit_pct <= self.max_profit]
        if self.verify:
            arbs = [a for a in await asyncio.gather(*(self._verify(a) for a in arbs)) if a]
        arbs.sort(key=lambda a: -a.profit_pct)
        self.state.arbs, self.state.matched = arbs, len(groups)
        self.state.updated = datetime.now(timezone.utc)
        await self._notify(arbs)
        return arbs

    async def _verify(self, arb: Arb) -> Arb | None:
        """Re-price Polymarket legs from the live order book; drop arbs that vanished."""
        pm = next((a for a in self.adapters if isinstance(a, PolymarketAdapter)), None)
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

    async def run(self):
        while True:
            t = time.monotonic()
            try:
                await self.scan_once()
            except Exception:  # noqa: BLE001
                log.exception("scan failed")
            await asyncio.sleep(max(1.0, self.interval - (time.monotonic() - t)))

    async def close(self):
        for a in self.adapters:
            await a.close()
