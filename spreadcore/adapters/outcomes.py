from __future__ import annotations

from ..matcher import team_score
from ..models import ML2, X12, Price

DRAW = {"draw", "x", "tie", "ничья"}


def map_outcomes(items: list[tuple[str, float, str | None]], home: str, away: str) -> tuple[str, dict] | None:
    """Map [(name, odds, ref)] of a winner market to ML2 / 1X2 sides by name, then by position."""
    if len(items) not in (2, 3):
        return None
    market = ML2 if len(items) == 2 else X12
    sides: dict[str, Price] = {}
    for i, (name, odds, ref) in enumerate(items):
        n = (name or "").strip().lower()
        if n in DRAW:
            side = "draw"
        elif n in ("1", "home", "w1", "п1"):
            side = "home"
        elif n in ("2", "away", "w2", "п2"):
            side = "away"
        else:
            sh, sa = team_score(name, home), team_score(name, away)
            if max(sh, sa) >= 70:
                side = "home" if sh >= sa else "away"
            else:  # unknown labels: home, (draw), away by position
                side = ["home", "away"][i] if market == ML2 else ["home", "draw", "away"][i]
        sides[side] = Price(odds=odds, ref=ref)
    expected = {"home", "away"} if market == ML2 else {"home", "draw", "away"}
    return (market, sides) if set(sides) == expected else None
