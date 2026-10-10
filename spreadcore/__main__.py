from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re

import yaml

from .notify import format_arb
from .scanner import Scanner

MOCK = {"scan": {"interval": 5, "min_profit": 0.1, "verify_polymarket": False},
        "books": {b: {"type": "mock"} for b in ("stake", "parimatch", "goated", "polymarket")}}


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    tg = cfg.setdefault("telegram", {})
    tg["bot_token"] = os.getenv("TG_BOT_TOKEN", tg.get("bot_token"))
    tg["chat_id"] = os.getenv("TG_CHAT_ID", tg.get("chat_id"))
    return cfg


async def main():
    ap = argparse.ArgumentParser(prog="spreadcore", description="Arb scanner: Stake/Parimatch/Goated vs Polymarket")
    ap.add_argument("-c", "--config", default="config.yaml")
    ap.add_argument("--mock", action="store_true", help="demo with fake odds, no network")
    ap.add_argument("--once", action="store_true", help="single scan, print and exit")
    ap.add_argument("--web", action="store_true", help="serve dashboard")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")

    scanner = Scanner(MOCK if args.mock else load_config(args.config))
    try:
        if args.once:
            arbs = await scanner.scan_once()
            print(f"\nСобытий: {scanner.state.counts}, пар: {scanner.state.matched}, вилок: {len(arbs)}")
            for name, err in scanner.state.errors.items():
                print(f"[!] {name}: {err}")
            for a in arbs:
                print("\n" + re.sub(r"<[^>]+>", "", format_arb(a)))
            return
        if args.web:
            import uvicorn
            from .web import create_app
            server = uvicorn.Server(uvicorn.Config(create_app(scanner), host=args.host, port=args.port,
                                                   log_level="warning"))
            logging.info("dashboard: http://%s:%d", args.host, args.port)
            await asyncio.gather(scanner.run(), server.serve())
        else:
            await scanner.run()
    finally:
        await scanner.close()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
