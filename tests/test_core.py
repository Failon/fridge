from datetime import datetime, timezone

from spreadcore.adapters.betby import BetbyAdapter
from spreadcore.adapters.jsonmap import JsonMapAdapter, jget
from spreadcore.adapters.polymarket import PolymarketAdapter
from spreadcore.adapters.stake import StakeAdapter
from spreadcore.arb import find_arbs, stakes_for
from spreadcore.matcher import group_events, team_score
from spreadcore.models import ML2, X12, Event, Price

T = datetime(2026, 10, 12, 19, 0, tzinfo=timezone.utc)


def ev(book, home, away, market, odds, sport="soccer"):
    outs = ("home", "draw", "away") if market == X12 else ("home", "away")
    return Event(book=book, sport=sport, home=home, away=away, start=T,
                 markets={market: {o: Price(odds=x) for o, x in zip(outs, odds)}})


def test_stakes_equal_payout():
    profit, stakes = stakes_for([2.1, 2.1], 100)
    assert round(profit, 4) == 0.05
    assert abs(stakes[0] * 2.1 - stakes[1] * 2.1) < 1e-9


def test_team_matching():
    assert team_score("Manchester Utd", "Manchester United FC") >= 90
    assert team_score("Arsenal", "Arsenal Women") == 0
    assert team_score("Lakers", "Los Angeles Lakers") >= 86


def test_group_and_arb_with_swap():
    a = ev("stake", "Los Angeles Lakers", "Boston Celtics", ML2, (2.20, 1.70), "basketball")
    b = ev("polymarket", "Celtics", "Lakers", ML2, (2.00, 1.60), "basketball")  # swapped order
    c = ev("goated", "Chelsea", "Arsenal", ML2, (1.5, 2.5), "basketball")
    groups = group_events([a, b, c])
    assert len(groups) == 1
    arbs = find_arbs(groups[0], 100, 0, require_polymarket=True)
    assert len(arbs) == 1 and arbs[0].profit_pct > 0
    legs = {l.outcome: (l.book, l.odds) for l in arbs[0].legs}
    assert legs == {"home": ("stake", 2.20), "away": ("polymarket", 2.00)}


def test_no_arb_from_single_book():
    a = ev("stake", "A Team", "B Team", ML2, (2.2, 2.2))
    b = ev("parimatch", "A Team", "B Team", ML2, (1.1, 1.1))
    arbs = find_arbs([a, b], 100, 0)
    assert arbs == []  # all best prices come from stake only


def test_polymarket_parse():
    pm = PolymarketAdapter("polymarket", {})
    ml = pm.parse_event({"id": 1, "title": "Celtics vs. Lakers", "slug": "nba-bos-lal",
                         "startTime": "2026-10-12T23:00:00Z", "markets": [
        {"active": True, "closed": False, "outcomes": '["Celtics", "Lakers"]',
         "clobTokenIds": '["t1", "t2"]', "bestAsk": 0.55, "bestBid": 0.53, "sportsMarketType": "moneyline"}]},
        "basketball")
    assert ml.markets[ML2]["home"].ref == "t1"
    assert round(ml.markets[ML2]["home"].odds, 3) == round(1 / 0.55, 3)
    assert round(ml.markets[ML2]["away"].odds, 3) == round(1 / 0.47, 3)

    soc = pm.parse_event({"id": 2, "title": "Arsenal vs. Chelsea", "slug": "x", "markets": [
        {"active": True, "outcomes": '["Yes","No"]', "clobTokenIds": '["a","a2"]', "bestAsk": "0.48", "groupItemTitle": "Arsenal"},
        {"active": True, "outcomes": '["Yes","No"]', "clobTokenIds": '["d","d2"]', "bestAsk": "0.27", "groupItemTitle": "Draw (Arsenal vs. Chelsea)"},
        {"active": True, "outcomes": '["Yes","No"]', "clobTokenIds": '["c","c2"]', "bestAsk": "0.27", "groupItemTitle": "Chelsea"},
    ]}, "soccer")
    assert set(soc.markets[X12]) == {"home", "draw", "away"}
    assert soc.markets[X12]["draw"].ref == "d"


def test_stake_parse():
    st = StakeAdapter("stake", {})
    e = st.parse_fixture({"id": "f1", "status": "active", "slug": "ars-che",
        "tournament": {"name": "Premier League", "slug": "premier-league", "category": {"slug": "england"}},
        "data": {"startTime": 1791831600000,
                 "competitors": [{"name": "Arsenal FC"}, {"name": "Chelsea FC"}]},
        "groups": [{"name": "winner", "templates": [{"name": "1x2", "markets": [{"name": "1x2", "status": "active",
            "outcomes": [{"id": "o1", "name": "Arsenal FC", "odds": 2.1, "active": True},
                         {"id": "o2", "name": "draw", "odds": 3.4, "active": True},
                         {"id": "o3", "name": "Chelsea FC", "odds": 3.5, "active": True}]}]}]}]}, "soccer")
    assert e.markets[X12]["draw"].odds == 3.4
    assert e.url.endswith("/england/premier-league/ars-che")


def test_betby_parse():
    bb = BetbyAdapter("goated", {"api": "x", "brand_id": "1"})
    evs = bb.parse_feed({"events": {"e1": {"desc": {"sport": "2", "scheduled": 1791831600, "slug": "lal-bos",
        "competitors": [{"name": "LA Lakers"}, {"name": "Boston Celtics"}]},
        "markets": {"219": {"": {"4": {"k": "1.95"}, "5": {"k": "1.91"}}}}}}}, {"2": "basketball"})
    assert evs[0].markets[ML2]["away"].odds == 1.91


def test_jsonmap_parse():
    assert jget({"a": [{"b": 1}, {"b": 2}]}, "a[*].b") == [1, 2]
    jm = JsonMapAdapter("parimatch", {"sports": ["soccer"], "events": "data.events[*]", "sport": "sport",
        "sport_map": {"Football": "soccer"}, "home": "teams[0]", "away": "teams[1]", "start": "ts",
        "markets": [{"path": "markets[*]", "where": {"type": "1x2"}, "outcomes": "prices[*]",
                     "outcome_name": "n", "odds": "v"}]})
    evs = jm.parse({"data": {"events": [{"id": 5, "sport": "Football", "teams": ["Arsenal", "Chelsea"], "ts": 1791831600,
        "markets": [{"type": "total"}, {"type": "1x2", "prices": [{"n": "1", "v": 2}, {"n": "X", "v": 3.3}, {"n": "2", "v": 4}]}]}]}})
    assert evs[0].markets[X12]["away"].odds == 4
