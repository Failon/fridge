"""Find the odds API of a bookmaker site: python -m spreadcore.discover https://goated.com/sports

Opens a real Chromium (log in / pass Cloudflare by hand if needed), records every JSON
response while you click around the sportsbook, then ranks endpoints that look like odds feeds.
Requires: pip install playwright
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
from pathlib import Path

ODDS = re.compile(r'"(?:odds|price|k|coef|coefficient|value)"\s*:\s*"?([1-9]\d?\.\d{1,3})"?')
BETBY = re.compile(r"(https://api-[\w-]+\.sptpub\.com)/api/v\d/(?:prematch|live)/brand/(\d+)/")


async def run(url: str, seconds: int, out: Path):
    from playwright.async_api import async_playwright
    out.mkdir(parents=True, exist_ok=True)
    hits: dict[str, dict] = {}

    async def on_response(resp):
        if "json" not in (resp.headers.get("content-type") or ""):
            return
        try:
            body = await resp.text()
        except Exception:  # noqa: BLE001
            return
        m = BETBY.search(resp.url)
        if m:
            print(f"\n[Betby] api: {m.group(1)}  brand_id: {m.group(2)}  -> type: betby в config.yaml\n")
        n = len(ODDS.findall(body))
        if n < 5:
            return
        key = resp.url.split("?")[0]
        if n > hits.get(key, {}).get("odds", 0):
            fname = out / f"{len(hits):03d}.json"
            fname.write_text(body, encoding="utf-8")
            hits[key] = {"url": resp.url, "method": resp.request.method, "odds": n, "file": str(fname),
                         "post": resp.request.post_data}
            print(f"[{n:5d} odds] {resp.request.method} {resp.url[:150]}")

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(str(out / "profile"), headless=False)
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        page.on("response", lambda r: asyncio.ensure_future(on_response(r)))
        await page.goto(url)
        print(f"Кликайте по линии/лайву {seconds} сек. (Ctrl+C чтобы закончить раньше)…")
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            pass
        await ctx.close()

    ranked = sorted(hits.values(), key=lambda h: -h["odds"])
    (out / "endpoints.json").write_text(json.dumps(ranked, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nТоп эндпоинтов сохранён в {out / 'endpoints.json'}; тела ответов — в {out}/NNN.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--out", default="captures")
    a = ap.parse_args()
    try:
        asyncio.run(run(a.url, a.seconds, Path(a.out)))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
