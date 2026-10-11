from __future__ import annotations

import re
import unicodedata
from dataclasses import replace
from datetime import timedelta

from rapidfuzz import fuzz

from .models import Event

STOP = {"fc", "cf", "sc", "afc", "ac", "fk", "sk", "bc", "club", "de", "the", "cd", "ca", "sv",
        "calcio", "football", "esports", "team"}
# tokens that change the identity of a team: never match "X" with "X Women"
MARKERS = {"women", "w", "u17", "u18", "u19", "u20", "u21", "u23", "ii", "b", "reserves", "youth"}
ALIASES = {
    "utd": "united", "man": "manchester", "st": "saint", "intl": "international",
    "psg": "paris saint germain", "spurs": "tottenham",
}


def norm(name: str) -> str:
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    s = re.sub(r"\(.*?\)", " ", s)
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [ALIASES.get(t, t) for t in s.split()]
    return " ".join(t for t in toks if t not in STOP)


def _markers(n: str) -> set[str]:
    return set(n.split()) & MARKERS


def team_score(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb or _markers(na) != _markers(nb):
        return 0.0
    return max(fuzz.token_set_ratio(na, nb), fuzz.ratio(na, nb))


def same_event(a: Event, b: Event, threshold: float, window: timedelta) -> str | None:
    """Return 'direct', 'swapped' or None."""
    if a.sport != b.sport:
        return None
    if a.start and b.start and abs(a.start - b.start) > window:
        return None
    direct = min(team_score(a.home, b.home), team_score(a.away, b.away))
    swapped = min(team_score(a.home, b.away), team_score(a.away, b.home))
    if max(direct, swapped) < threshold:
        return None
    return "direct" if direct >= swapped else "swapped"


def swap_sides(e: Event) -> Event:
    flip = {"home": "away", "away": "home", "draw": "draw"}
    markets = {m: {flip[o]: p for o, p in outs.items()} for m, outs in e.markets.items()}
    return replace(e, home=e.away, away=e.home, markets=markets)


def group_events(events: list[Event], threshold: float = 86, window_hours: float = 3) -> list[list[Event]]:
    """Cluster events from different books that describe the same match."""
    window = timedelta(hours=window_hours)
    groups: list[list[Event]] = []
    # bookmakers first so the anchor of a group carries bookmaker team order
    for ev in sorted(events, key=lambda e: e.book == "polymarket"):
        best, best_g, best_mode = 0.0, None, None
        for g in groups:
            if any(x.book == ev.book for x in g):
                continue
            mode = same_event(g[0], ev, threshold, window)
            if mode:
                anchor = g[0]
                score = (team_score(anchor.home, ev.home) + team_score(anchor.away, ev.away)
                         if mode == "direct" else
                         team_score(anchor.home, ev.away) + team_score(anchor.away, ev.home))
                if score > best:
                    best, best_g, best_mode = score, g, mode
        if best_g is None:
            groups.append([ev])
        else:
            best_g.append(swap_sides(ev) if best_mode == "swapped" else ev)
    return [g for g in groups if len(g) > 1]
