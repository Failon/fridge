from .base import Adapter
from .betby import BetbyAdapter
from .jsonmap import JsonMapAdapter
from .mock import MockAdapter
from .polymarket import PolymarketAdapter
from .stake import StakeAdapter

TYPES = {
    "polymarket": PolymarketAdapter,
    "stake": StakeAdapter,
    "betby": BetbyAdapter,
    "jsonmap": JsonMapAdapter,
    "mock": MockAdapter,
}


def build(name: str, cfg: dict) -> Adapter:
    kind = cfg.get("type", name)
    if kind not in TYPES:
        raise ValueError(f"{name}: unknown adapter type {kind!r}, expected one of {sorted(TYPES)}")
    return TYPES[kind](name, cfg)
