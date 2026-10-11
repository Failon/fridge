from __future__ import annotations

from .models import OUTCOMES, Arb, Event, Leg


def stakes_for(odds: list[float], bankroll: float) -> tuple[float, list[float]]:
    """Return (profit fraction, stakes) for a full-cover arbitrage."""
    inv = sum(1 / o for o in odds)
    stakes = [bankroll / (o * inv) for o in odds]
    return 1 / inv - 1, stakes


def find_arbs(group: list[Event], bankroll: float = 100.0, min_profit: float = 0.0,
              require_polymarket: bool = False) -> list[Arb]:
    """Search best price per outcome across a group of matched events (one per book)."""
    ref = next((e for e in group if e.book != "polymarket"), group[0])
    result = []
    for market, outcomes in OUTCOMES.items():
        best = []
        for out in outcomes:
            cands = [(e, e.markets[market][out]) for e in group
                     if market in e.markets and out in e.markets[market] and e.markets[market][out].odds > 1]
            if not cands:
                break
            best.append((out, *max(cands, key=lambda c: c[1].odds)))
        else:
            books = {e.book for _, e, _ in best}
            if len(books) < 2:
                continue  # a single book never offers a real arb; skip palpable errors
            if require_polymarket and "polymarket" not in books:
                continue
            profit, stakes = stakes_for([p.odds for _, _, p in best], bankroll)
            if profit * 100 < min_profit:
                continue
            legs = [Leg(book=e.book, outcome=out, odds=p.odds, stake=round(s, 2),
                        team={"home": ref.home, "away": ref.away}.get(out, "Draw"),
                        url=e.url, limit=p.limit, ref=p.ref)
                    for (out, e, p), s in zip(best, stakes)]
            result.append(Arb(sport=ref.sport, title=ref.title, league=ref.league, start=ref.start,
                              market=market, profit_pct=round(profit * 100, 3), legs=legs,
                              bankroll=bankroll))
    return result
