import asyncio
import json

import websockets

from spreadcore.adapters.polymarket import PolymarketAdapter
from spreadcore.adapters.stake import StakeAdapter
from spreadcore.models import Price


def test_polymarket_book_messages():
    pm = PolymarketAdapter("polymarket", {})
    upd = pm.handle_ws({"event_type": "book", "asset_id": "t1",
                        "asks": [{"price": "0.52", "size": "100"}, {"price": "0.50", "size": "40"}]})
    assert upd["t1"][1] == 0.50 and upd["t1"][2] == 20.0
    upd = pm.handle_ws({"event_type": "price_change", "price_changes": [
        {"asset_id": "t1", "price": "0.50", "size": "0", "side": "SELL", "best_ask": "0.52"}]})
    assert upd["t1"][1] == 0.52  # level removed -> next level is best


async def _run_stream(adapter, server_handler, prices):
    got = []
    async with websockets.serve(server_handler, "127.0.0.1", 0) as srv:
        port = srv.sockets[0].getsockname()[1]
        adapter.cfg["ws_url"] = f"ws://127.0.0.1:{port}"
        task = asyncio.create_task(adapter.stream(lambda: prices, got.append))
        for _ in range(50):
            await asyncio.sleep(0.1)
            if got:
                break
        task.cancel()
        await adapter.close()
    return got


def test_polymarket_stream_end_to_end():
    async def server(ws):
        sub = json.loads(await ws.recv())
        assert sub["assets_ids"] == ["tok"] and sub["type"] == "market"
        await ws.send(json.dumps([{"event_type": "book", "asset_id": "tok",
                                   "asks": [{"price": "0.40", "size": "250"}]}]))
        await asyncio.sleep(1)

    got = asyncio.run(_run_stream(PolymarketAdapter("polymarket", {}), server, [Price(odds=2, ref="tok")]))
    assert got and got[0]["tok"][1] == 0.40 and round(got[0]["tok"][0], 3) == 2.5


def test_stake_graphql_ws_end_to_end():
    async def server(ws):
        init = json.loads(await ws.recv())
        assert init["type"] == "connection_init"
        await ws.send(json.dumps({"type": "connection_ack"}))
        sub = json.loads(await ws.recv())
        assert sub["type"] == "subscribe" and sub["payload"]["variables"] == {"id": "m1"}
        await ws.send(json.dumps({"id": "m1", "type": "next", "payload": {"data": {"sportMarket": {
            "id": "m1", "status": "active",
            "outcomes": [{"id": "o1", "active": True, "odds": 2.3}, {"id": "o2", "active": False, "odds": 1.6}]}}}}))
        await asyncio.sleep(1)

    got = asyncio.run(_run_stream(StakeAdapter("stake", {}), server, [Price(odds=2, ref="o1", market_ref="m1")]))
    assert got[0] == {"o1": (2.3, None, None), "o2": (None, None, None)}


def test_tick_triggers_arb():
    from datetime import datetime, timezone

    from spreadcore.models import ML2, Event
    from spreadcore.scanner import Scanner

    async def main():
        sc = Scanner({"scan": {"min_profit": 0.1, "verify_polymarket": False}, "books": {}})
        t = datetime(2026, 10, 12, tzinfo=timezone.utc)
        sc._events = {
            "stake": [Event("stake", "tennis", "Carlos Alcaraz", "Jannik Sinner", t,
                            markets={ML2: {"home": Price(2.0, ref="s1"), "away": Price(1.8, ref="s2")}})],
            "polymarket": [Event("polymarket", "tennis", "Alcaraz", "Sinner", t,
                                 markets={ML2: {"home": Price(1.9, ref="p1"), "away": Price(1.9, ref="p2")}})]}
        sc._regroup()
        assert await sc._evaluate(False) == []
        sc._on_update("polymarket")({"p2": (2.2, 0.4545, 300.0)})  # Sinner price jumps on Polymarket
        arbs = await sc._evaluate(False)
        assert len(arbs) == 1 and {l.book for l in arbs[0].legs} == {"stake", "polymarket"}
        assert arbs[0].legs[1].limit == 300.0

    asyncio.run(main())
