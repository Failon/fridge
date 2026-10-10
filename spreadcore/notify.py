from __future__ import annotations

from .adapters.base import log
from .models import Arb

BOOK_NAMES = {"stake": "Stake", "parimatch": "Parimatch", "goated": "Goated", "polymarket": "Polymarket"}


def format_arb(a: Arb) -> str:
    when = a.start.strftime("%d.%m %H:%M UTC") if a.start else "?"
    lines = [f"💰 <b>{a.profit_pct:.2f}%</b> | {a.sport} | {a.market}",
             f"<b>{a.title}</b> ({a.league}) — {when}"]
    for l in a.legs:
        lim = f" (лимит ~${l.limit:.0f})" if l.limit else ""
        lines.append(f"• {BOOK_NAMES.get(l.book, l.book)}: {l.team} @ {l.odds:.3f} → ${l.stake:.2f}{lim}"
                     + (f" <a href=\"{l.url}\">открыть</a>" if l.url else ""))
    lines.append(f"Банк ${a.bankroll:.0f} → прибыль ${a.bankroll * a.profit_pct / 100:.2f}")
    return "\n".join(lines)


class Telegram:
    def __init__(self, cfg: dict):
        self.token = cfg.get("bot_token")
        self.chat = cfg.get("chat_id")
        self.enabled = bool(self.token and self.chat)

    async def send(self, text: str):
        if not self.enabled:
            return
        from curl_cffi.requests import AsyncSession
        try:
            async with AsyncSession(timeout=15) as s:
                r = await s.post(f"https://api.telegram.org/bot{self.token}/sendMessage",
                                 json={"chat_id": self.chat, "text": text, "parse_mode": "HTML",
                                       "disable_web_page_preview": True})
                if r.status_code != 200:
                    log.warning("telegram: HTTP %s %s", r.status_code, r.text[:200])
        except Exception as e:  # noqa: BLE001
            log.warning("telegram: %s", e)
