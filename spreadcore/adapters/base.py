from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Callable

from ..models import Event, Price

log = logging.getLogger("spreadcore")


def parse_time(v) -> datetime | None:
    if v in (None, ""):
        return None
    try:
        if isinstance(v, (int, float)) or (isinstance(v, str) and v.isdigit()):
            ts = float(v)
            if ts > 1e11:  # milliseconds
                ts /= 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except (ValueError, OSError):
        return None


def as_float(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f > 0 else None


class Adapter:
    """Base class. Subclasses implement `fetch()` returning normalized events."""

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.sports: list[str] = cfg.get("sports", ["soccer", "basketball", "ice-hockey", "tennis"])
        self._session = None

    @property
    def book(self) -> str:
        return self.cfg.get("book", self.name)

    async def session(self):
        if self._session is None:
            from curl_cffi.requests import AsyncSession
            # TLS fingerprint of a real Chrome helps against Cloudflare
            self._session = AsyncSession(
                impersonate=self.cfg.get("impersonate", "chrome"),
                proxy=self.cfg.get("proxy"),
                timeout=self.cfg.get("timeout", 20),
                headers=self.cfg.get("headers") or {},
                cookies=self._cookies(),
            )
        return self._session

    def _cookies(self) -> dict:
        raw = self.cfg.get("cookies") or {}
        if isinstance(raw, str):  # "a=1; b=2" copied from DevTools
            return dict(p.strip().split("=", 1) for p in raw.split(";") if "=" in p)
        return raw

    async def get_json(self, url: str, **kw):
        s = await self.session()
        r = await s.get(url, **kw)
        return self._decode(r, url)

    async def post_json(self, url: str, payload, **kw):
        s = await self.session()
        r = await s.post(url, json=payload, **kw)
        return self._decode(r, url)

    def _decode(self, r, url):
        if r.status_code != 200:
            hint = " (Cloudflare? set cookies/proxy in config)" if r.status_code in (403, 503) else ""
            raise RuntimeError(f"{self.name}: HTTP {r.status_code} for {url}{hint}: {r.text[:200]}")
        try:
            return r.json()
        except json.JSONDecodeError as e:
            raise RuntimeError(f"{self.name}: non-JSON response from {url}: {r.text[:200]}") from e

    async def fetch(self) -> list[Event]:
        raise NotImplementedError

    @property
    def streaming(self) -> bool:
        """True if this adapter pushes live price updates via `stream()`."""
        return False

    async def stream(self, watched: Callable[[], list[Price]], on_update: Callable[[dict], None]):
        """Run forever, calling on_update({ref: (odds, raw, limit)}) for prices returned by watched()."""
        raise NotImplementedError

    async def ws_connect(self, url: str, **kw):
        s = await self.session()
        return await s.ws_connect(url, **kw)

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None
