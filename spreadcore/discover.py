"""Find the odds API of a bookmaker site: python -m spreadcore.discover https://goated.com/sports

Opens a real Chromium (log in / pass Cloudflare by hand if needed) and, while you click around
the sportsbook, records:
  * every JSON HTTP response that contains odds  -> captures/NNN.json
  * every WebSocket: all sent frames (subscriptions) and received frames -> captures/ws/NNN.jsonl
Then ranks both by how many odds they carry -> captures/endpoints.json
Requires: pip install playwright && playwright install chromium
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import json
import re
import time
import zlib
from pathlib import Path

ODDS = re.compile(r'"(?:odds|price|k|coef|coefficient|value)"\s*:\s*"?([1-9]\d?\.\d{1,3})"?')
LOOSE = re.compile(r"(?<![\d.])[1-9]\d?\.\d{2,3}(?![\d.])")  # bare decimals like 1.85 in arrays
BETBY = re.compile(r"(https://api-[\w-]+\.sptpub\.com)/api/v\d/(?:prematch|live)/brand/(\d+)/")
MAX_FRAMES = 5000  # received frames kept per socket (sent frames are always kept)


def decode_frame(payload) -> tuple[str, str]:
    """Return (text, encoding). Binary frames are tried as utf-8, gzip, deflate, else base64."""
    if isinstance(payload, str):
        return payload, "text"
    for name, fn in (("utf8", lambda b: b.decode("utf-8")),
                     ("gzip", lambda b: gzip.decompress(b).decode("utf-8")),
                     ("deflate", lambda b: zlib.decompress(b, -15).decode("utf-8")),
                     ("zlib", lambda b: zlib.decompress(b).decode("utf-8"))):
        try:
            return fn(payload), name
        except Exception:  # noqa: BLE001
            pass
    return base64.b64encode(payload).decode(), "base64"  # msgpack/protobuf: needs a custom decoder


class WsRecorder:
    def __init__(self, out: Path):
        self.dir = out / "ws"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sockets: list[dict] = []

    def attach(self, ws):
        info = {"url": ws.url, "file": str(self.dir / f"{len(self.sockets):03d}.jsonl"),
                "sent": 0, "recv": 0, "odds": 0, "binary": 0, "first_sent": [], "t0": time.time()}
        self.sockets.append(info)
        fh = open(info["file"], "w", encoding="utf-8")
        print(f"[ws open] {ws.url[:150]}")

        def frame(direction, payload):
            text, enc = decode_frame(payload)
            if enc == "base64":
                info["binary"] += 1
            if direction == "sent":
                info["sent"] += 1
                if len(info["first_sent"]) < 20:
                    info["first_sent"].append(text[:2000])
            else:
                info["recv"] += 1
                if info["recv"] > MAX_FRAMES:
                    return
                before = info["odds"]
                info["odds"] += len(ODDS.findall(text)) or len(LOOSE.findall(text))
                if before < 20 <= info["odds"]:
                    print(f"[ws odds] {ws.url[:150]} — похоже на поток коэффициентов")
            fh.write(json.dumps({"t": round(time.time() - info["t0"], 3), "dir": direction,
                                 "enc": enc, "data": text}, ensure_ascii=False) + "\n")
            fh.flush()

        ws.on("framesent", lambda p: frame("sent", p))
        ws.on("framereceived", lambda p: frame("recv", p))
        ws.on("close", lambda _: (fh.close(), print(f"[ws close] {ws.url[:150]}")))

    def ranked(self) -> list[dict]:
        return [{k: v for k, v in s.items() if k != "t0"}
                for s in sorted(self.sockets, key=lambda s: -s["odds"])]


async def run(url: str, seconds: int, out: Path, headless: bool = False, browser: str | None = None):
    from playwright.async_api import async_playwright
    out.mkdir(parents=True, exist_ok=True)
    hits: dict[str, dict] = {}
    wsrec = WsRecorder(out)

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

    def watch(page):
        page.on("response", lambda r: asyncio.ensure_future(on_response(r)))
        page.on("websocket", wsrec.attach)  # covers sockets opened by iframes of the page too

    async with async_playwright() as p:
        ctx = await p.chromium.launch_persistent_context(str(out / "profile"), headless=headless,
                                                         executable_path=browser)
        for pg in ctx.pages:
            watch(pg)
        ctx.on("page", watch)  # new tabs / popups
        page = ctx.pages[0] if ctx.pages else await ctx.new_page()
        await page.goto(url)
        print(f"Кликайте по линии/лайву {seconds} сек. (Ctrl+C чтобы закончить раньше)…")
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            pass
        await ctx.close()

    report = {"http": sorted(hits.values(), key=lambda h: -h["odds"]), "websockets": wsrec.ranked()}
    (out / "endpoints.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nHTTP: {len(report['http'])}, WebSocket: {len(report['websockets'])}")
    for s in report["websockets"][:5]:
        bin_note = f", бинарных {s['binary']}" if s["binary"] else ""
        print(f"  [{s['odds']:6d} odds] {s['url'][:120]}  ({s['sent']} отпр./{s['recv']} получ.{bin_note}) -> {s['file']}")
    print(f"Отчёт: {out / 'endpoints.json'}. Пришлите его, верхний captures/NNN.json и верхний captures/ws/NNN.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("--seconds", type=int, default=120)
    ap.add_argument("--out", default="captures")
    ap.add_argument("--headless", action="store_true", help="без окна (Cloudflare так чаще блокирует)")
    ap.add_argument("--browser", help="путь к Chrome/Chromium, если не хотите playwright install")
    a = ap.parse_args()
    try:
        asyncio.run(run(a.url, a.seconds, Path(a.out), a.headless, a.browser))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
