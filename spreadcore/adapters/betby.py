from __future__ import annotations

from ..models import ML2, X12, Event, Price
from .base import Adapter, as_float, log, parse_time

# Betby white-label sportsbook feed (api-*.sptpub.com). Uses Sportradar market/outcome ids.
SPORT_IDS = {"soccer": "1", "basketball": "2", "baseball": "3", "ice-hockey": "4", "tennis": "5",
             "american-football": "16", "counter-strike": "109", "league-of-legends": "110",
             "dota-2": "111", "mma": "117"}
# market id -> (our key, {outcome id: side})
MARKETS = {
    "1": (X12, {"1": "home", "2": "draw", "3": "away"}),
    "186": (ML2, {"4": "home", "5": "away"}),    # winner
    "219": (ML2, {"4": "home", "5": "away"}),    # winner incl. overtime
    "251": (ML2, {"4": "home", "5": "away"}),    # winner incl. extra innings
    "406": (ML2, {"4": "home", "5": "away"}),    # winner incl. overtime and penalties
}


class BetbyAdapter(Adapter):
    """Config: api (e.g. https://api-xxxx.sptpub.com), brand_id, site_url. Find both in DevTools."""

    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.api = cfg.get("api", "").rstrip("/")
        self.brand = str(cfg.get("brand_id", ""))
        self.lang = cfg.get("lang", "en")
        self.site = cfg.get("site_url", "").rstrip("/")
        self.feeds = cfg.get("feeds", ["prematch", "live"])

    async def fetch(self) -> list[Event]:
        if not self.api or not self.brand:
            raise RuntimeError(f"{self.name}: set 'api' and 'brand_id' (see README, DevTools -> sptpub.com)")
        wanted = {SPORT_IDS[s]: s for s in self.sports if s in SPORT_IDS}
        out = []
        for feed in self.feeds:
            try:
                base = f"{self.api}/api/v3/{feed}/brand/{self.brand}/{self.lang}"
                head = await self.get_json(f"{base}/0")
                versions = [head.get("version")] + head.get("top_events_versions", []) + head.get("rest_events_versions", [])
                for v in dict.fromkeys(x for x in versions if x):
                    out += self.parse_feed(await self.get_json(f"{base}/{v}"), wanted)
            except Exception as e:  # noqa: BLE001
                log.warning("%s %s: %s", self.name, feed, e)
        return out

    def parse_feed(self, data: dict, wanted: dict[str, str]) -> list[Event]:
        tours = data.get("tournaments") or {}
        out = []
        for eid, raw in (data.get("events") or {}).items():
            desc = raw.get("desc") or {}
            sport = wanted.get(str(desc.get("sport")))
            comps = [c.get("name") for c in desc.get("competitors") or []]
            if not sport or len(comps) != 2:
                continue
            ev = Event(book=self.book, sport=sport, home=comps[0], away=comps[1],
                       start=parse_time(desc.get("scheduled")),
                       league=(tours.get(str(desc.get("tournament"))) or {}).get("name", ""),
                       url=f"{self.site}/?bt-path=/{desc.get('slug', eid)}" if self.site else "",
                       event_id=str(eid))
            for mid, spec in (raw.get("markets") or {}).items():
                if mid not in MARKETS:
                    continue
                key, sides_map = MARKETS[mid]
                if key in ev.markets:
                    continue
                outcomes = (spec or {}).get("") or {}  # "" = market without specifiers
                sides = {}
                for oid, o in outcomes.items():
                    odds = as_float((o or {}).get("k"))
                    if oid in sides_map and odds and odds > 1:
                        sides[sides_map[oid]] = Price(odds=odds, ref=f"{eid}:{mid}:{oid}")
                if len(sides) == len(sides_map):
                    ev.markets[key] = sides
            if ev.markets:
                out.append(ev)
        return out
