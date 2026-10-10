from __future__ import annotations

import asyncio
import re

from ..models import Event
from .base import Adapter, as_float, log, parse_time
from .outcomes import map_outcomes

# Default query for Stake's sportsbook GraphQL. If Stake changes its schema, copy the
# current request from DevTools (Network -> graphql -> "SportFixtureList") into config.
QUERY = """
query SportFixtureList($type: SportSearchEnum!, $sport: String!, $groups: String!, $offset: Int!, $limit: Int!) {
  slugSport(sport: $sport) {
    fixtureList(type: $type, limit: $limit, offset: $offset) {
      id slug status
      tournament { name slug category { slug } }
      data {
        __typename
        ... on SportFixtureDataMatch { startTime competitors { name } }
      }
      groups(groups: [$groups], status: [active]) {
        name
        templates(limit: 3, includeEmpty: true) {
          name
          markets(limit: 1) { id name status outcomes { id active odds name } }
        }
      }
    }
  }
}
"""

WINNER = re.compile(r"^(1x2|winner|match winner|moneyline|full ?time result|"
                    r"winner \(incl\. (overtime|extra innings|overtime and penalties)\))$", re.I)


class StakeAdapter(Adapter):
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.domain = cfg.get("domain", "stake.com")
        self.query = cfg.get("query", QUERY)
        self.group = cfg.get("group", "winner")
        self.types = cfg.get("types", ["upcoming", "live"])
        self.limit = int(cfg.get("limit", 50))
        self.pages = int(cfg.get("pages", 4))
        hdr = {"content-type": "application/json", "x-language": "en",
               "origin": f"https://{self.domain}", "referer": f"https://{self.domain}/sports/home"}
        if cfg.get("access_token"):
            hdr["x-access-token"] = cfg["access_token"]
        self.cfg["headers"] = {**hdr, **(cfg.get("headers") or {})}

    async def fetch(self) -> list[Event]:
        jobs = [(s, t, p) for s in self.sports for t in self.types for p in range(self.pages)]
        res = await asyncio.gather(*(self._page(*j) for j in jobs), return_exceptions=True)
        out, seen = [], set()
        for (sport, _, _), r in zip(jobs, res):
            if isinstance(r, Exception):
                log.warning("stake %s: %s", sport, r)
                continue
            for fx in r:
                if fx.get("id") not in seen:
                    seen.add(fx.get("id"))
                    ev = self.parse_fixture(fx, sport)
                    if ev:
                        out.append(ev)
        return out

    async def _page(self, sport, typ, page):
        data = await self.post_json(f"https://{self.domain}/_api/graphql", {
            "operationName": "SportFixtureList", "query": self.query,
            "variables": {"type": typ, "sport": sport, "groups": self.group,
                          "offset": page * self.limit, "limit": self.limit}})
        if data.get("errors"):
            raise RuntimeError(f"GraphQL errors: {data['errors'][:2]}")
        return ((data.get("data") or {}).get("slugSport") or {}).get("fixtureList") or []

    def parse_fixture(self, fx: dict, sport: str) -> Event | None:
        d = fx.get("data") or {}
        comps = [c.get("name") for c in d.get("competitors") or []]
        if len(comps) != 2 or fx.get("status") not in (None, "active", "live"):
            return None
        tour = fx.get("tournament") or {}
        url = f"https://{self.domain}/sports/{sport}"
        if tour.get("slug") and (tour.get("category") or {}).get("slug"):
            url += f"/{tour['category']['slug']}/{tour['slug']}/{fx.get('slug', '')}"
        ev = Event(book=self.book, sport=sport, home=comps[0], away=comps[1],
                   start=parse_time(d.get("startTime")), league=tour.get("name", ""),
                   url=url, event_id=str(fx.get("id")))
        for g in fx.get("groups") or []:
            for t in g.get("templates") or []:
                for m in t.get("markets") or []:
                    if m.get("status") not in (None, "active") or not WINNER.match((m.get("name") or "").strip()):
                        continue
                    items = [(o.get("name"), as_float(o.get("odds")), o.get("id"))
                             for o in m.get("outcomes") or [] if o.get("active", True)]
                    if any(odds is None for _, odds, _ in items):
                        continue
                    mapped = map_outcomes(items, ev.home, ev.away)
                    if mapped and mapped[0] not in ev.markets:
                        ev.markets[mapped[0]] = mapped[1]
        return ev if ev.markets else None
