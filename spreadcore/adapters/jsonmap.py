from __future__ import annotations

import re

from ..models import Event
from .base import Adapter, as_float, log, parse_time
from .outcomes import map_outcomes

TOKEN = re.compile(r"([^.\[\]]+)|\[(\*|-?\d+)\]")


def jget(obj, path: str) -> list:
    """Tiny JSONPath: 'a.b[0].c', 'events[*]', 'markets.*' (all dict values). Returns all matches."""
    if not path:
        return [obj]
    cur = [obj]
    for name, idx in TOKEN.findall(path):
        nxt = []
        for c in cur:
            if name == "*":
                nxt += list(c.values()) if isinstance(c, dict) else list(c) if isinstance(c, list) else []
            elif name:
                if isinstance(c, dict) and name in c:
                    nxt.append(c[name])
            elif idx == "*":
                nxt += list(c.values()) if isinstance(c, dict) else list(c) if isinstance(c, list) else []
            elif isinstance(c, list):
                try:
                    nxt.append(c[int(idx)])
                except IndexError:
                    pass
        cur = nxt
    return cur


def first(obj, path):
    if not path:
        return None
    v = jget(obj, path)
    return v[0] if v else None


class JsonMapAdapter(Adapter):
    """Generic adapter: any JSON endpoint + field mapping in config (see config.example.yaml)."""

    async def fetch(self) -> list[Event]:
        out = []
        for req in self.cfg.get("requests") or [self.cfg.get("request")]:
            if not req:
                raise RuntimeError(f"{self.name}: no 'request' configured")
            try:
                if req.get("method", "GET").upper() == "POST":
                    data = await self.post_json(req["url"], req.get("body"), params=req.get("params"))
                else:
                    data = await self.get_json(req["url"], params=req.get("params"))
            except Exception as e:  # noqa: BLE001
                log.warning("%s %s: %s", self.name, req.get("url"), e)
                continue
            out += self.parse(data, req.get("sport"))
        return out

    def parse(self, data, sport_hint: str | None = None) -> list[Event]:
        c = self.cfg
        sport_map = c.get("sport_map") or {}
        out = []
        for raw in jget(data, c["events"]):
            if not isinstance(raw, dict):
                continue
            sport = sport_hint or c.get("sport_fixed")
            if not sport:
                rs = first(raw, c.get("sport"))
                sport = sport_map.get(str(rs), str(rs) if rs is not None and not sport_map else None)
            if not sport or (self.sports and sport not in self.sports):
                continue
            home, away = first(raw, c["home"]), first(raw, c["away"])
            if not home or not away:
                continue
            eid = first(raw, c.get("id", "id"))
            ev = Event(book=self.book, sport=sport, home=str(home), away=str(away),
                       start=parse_time(first(raw, c.get("start"))),
                       league=str(first(raw, c.get("league")) or ""),
                       url=c.get("url_template", "").format_map(
                           {**{k: v for k, v in raw.items() if isinstance(v, (str, int))}, "id": eid}),
                       event_id=str(eid))
            for spec in c.get("markets") or []:
                for m in jget(raw, spec["path"]):
                    if not isinstance(m, dict):
                        continue
                    if any(str(first(m, k)) != str(v) for k, v in (spec.get("where") or {}).items()):
                        continue
                    items = []
                    for o in jget(m, spec["outcomes"]):
                        rawp = first(o, spec["odds"])
                        if spec.get("odds_format") == "american":
                            try:
                                a = float(rawp)
                                odds = 1 + (a / 100 if a > 0 else 100 / -a)
                            except (TypeError, ValueError, ZeroDivisionError):
                                odds = None
                        else:
                            odds = as_float(rawp)
                        items.append((str(first(o, spec.get("outcome_name", "name")) or ""), odds, None))
                    if not items or any(o is None or o <= 1 for _, o, _ in items):
                        continue
                    mapped = map_outcomes(items, ev.home, ev.away)
                    if mapped and (spec.get("key") in (None, mapped[0])) and mapped[0] not in ev.markets:
                        ev.markets[mapped[0]] = mapped[1]
            if ev.markets:
                out.append(ev)
        return out
