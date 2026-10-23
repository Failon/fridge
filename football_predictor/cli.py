"""Командная строка: download / backtest / train / predict / teams."""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data, transfermarkt, understat
from .features import build_features
from .metrics import report
from .model import MatchPredictor
from .service import Predictor, UnknownLeague, UnknownTeam

FEATURES_CACHE = Path("data/features.pkl")
MODEL_PATH = Path(os.environ.get("FOOTBALL_MODEL_PATH", "models/predictor.pkl"))


def _load_features(args):
    """Матчи + признаки + состояние движка (кэшируется, т.к. проход по истории занимает минуты)."""
    matches = data.load_matches(args.data_dir, args.extra_csv)
    key = (len(matches), str(matches["date"].max()), int(matches["hxg"].notna().sum()),
           int(matches["h_value"].notna().sum()) if "h_value" in matches else 0,
           int(matches["h_xi"].notna().sum()) if "h_xi" in matches else 0, tuple(args.extra_csv or []))
    if FEATURES_CACHE.exists() and not args.rebuild:
        with open(FEATURES_CACHE, "rb") as f:
            cached = pickle.load(f)
        if cached["key"] == key:
            return matches, cached["X"], cached["engine"]
    print(f"Строю признаки для {len(matches)} матчей (один раз, затем кэш)...")
    X, engine = build_features(matches)
    FEATURES_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FEATURES_CACHE.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump({"key": key, "X": X, "engine": engine}, f)
    tmp.replace(FEATURES_CACHE)  # атомарно: прерванная запись не испортит кэш
    return matches, X, engine


def _filter(matches, X, args):
    mask = matches["season"] >= args.train_from
    if args.leagues:
        mask &= matches["league"].isin(args.leagues)
    return matches[mask], X[mask]


def cmd_download(args):
    data.download(args.leagues or data.LEAGUES, args.first, args.last, args.data_dir, args.force)
    if not args.no_xg:
        print("xG с understat.com...")
        understat.download(args.leagues or data.LEAGUES, force=args.force)
    if not args.no_tm:
        print("Составы и стоимость игроков (transfermarkt-datasets)...")
        transfermarkt.download(force=args.force)


def cmd_backtest(args):
    matches, X, _ = _load_features(args)
    matches, X = _filter(matches, X, args)
    seasons = sorted(matches["season"].unique())
    test_seasons = seasons[-args.seasons:]
    rows, all_probs, all_y, saved = [], {}, [], []

    for s in test_seasons:
        tr, te = (matches["season"] < s).to_numpy(), (matches["season"] == s).to_numpy()
        print(f"\n=== Тестовый сезон {data.season_code(s)}: обучение на {tr.sum()} матчах, тест {te.sum()}")
        model = MatchPredictor(use_market=False if args.no_market else None, calibration=args.calibration,
                               use_lineups=args.lineups).fit(X[tr], matches[tr], val_seasons=args.val_seasons)
        y = matches.loc[te, "result"].to_numpy()
        probs = model.member_probs(X[te])
        probs["ENSEMBLE"] = model.predict_proba(X[te])
        for k, p in model.predict_variants(X[te]).items():
            probs[f"ens_{k}"] = p
        base = np.bincount(matches.loc[tr, "result"], minlength=3) / tr.sum()
        probs["base_rate"] = np.tile(base, (te.sum(), 1))
        probs["market"] = X.loc[te, ["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy()  # рынок без маржи
        for name, p in probs.items():
            ok = ~np.isnan(p).any(1)
            if not ok.any():
                continue
            rows.append({"season": data.season_code(s), "model": name, **report(p[ok], y[ok])})
            all_probs.setdefault(name, []).append(p)
        all_y.append(y)
        if args.save_preds:
            out = matches.loc[te, ["season", "league", "date", "home", "away", "result"]].copy()
            for name in ("ENSEMBLE", "ens_none", "market"):
                out[[f"{name}_h", f"{name}_d", f"{name}_a"]] = probs[name]
            saved.append(out)

    df = pd.DataFrame(rows)
    y = np.concatenate(all_y)
    total = []
    for name, ps in all_probs.items():
        p = np.concatenate(ps)
        ok = ~np.isnan(p).any(1)
        if not ok.any():
            continue
        total.append({"model": name, "n": int(ok.sum()), **report(p[ok], y[ok])})
    total = pd.DataFrame(total).sort_values("RPS")
    pd.set_option("display.width", 160)
    print("\nПо сезонам (ансамбль):")
    print(df[df.model == "ENSEMBLE"].drop(columns="model").to_string(index=False, float_format="%.4f"))
    print(f"\nИТОГО за {len(test_seasons)} сезон(а/ов), walk-forward (меньше RPS/LogLoss/Brier — лучше):")
    print(total.to_string(index=False, float_format="%.4f"))
    if args.out:
        total.to_csv(args.out, index=False)
    if args.save_preds:
        pd.concat(saved).to_csv(args.save_preds, index=False, float_format="%.5f")


def cmd_train(args):
    matches, X, engine = _load_features(args)
    m, Xf = _filter(matches, X, args)
    print(f"Обучение на {len(m)} матчах (сезоны {data.season_code(m.season.min())}–{data.season_code(m.season.max())})")
    model = MatchPredictor(use_market=False if args.no_market else None, calibration=args.calibration,
                           use_lineups=args.lineups).fit(Xf, m, val_seasons=args.val_seasons)
    print("\nТоп признаков (LightGBM, доля gain):")
    print(model.feature_importance(15).to_string(float_format="%.3f"))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "engine": engine, "last_date": matches["date"].max()}, f)
    print(f"\nМодель сохранена в {MODEL_PATH}")


def _load_predictor() -> Predictor:
    if not MODEL_PATH.exists():
        sys.exit("Модель не найдена. Выполните `python -m football_predictor train`.")
    return Predictor.load(MODEL_PATH)


def _print_prediction(r):
    print(f"\n{r['home']} — {r['away']}  ({r['league']}, {r['date']})")
    print(f"  П1 {r['p_home']:6.1%}   X {r['p_draw']:6.1%}   П2 {r['p_away']:6.1%}")
    fo = r["fair_odds"]
    print(f"  справедливые коэф.:  {fo['home']:.2f} / {fo['draw']:.2f} / {fo['away']:.2f}")
    print(f"  ожидаемые голы: {r['xg_home']:.2f} — {r['xg_away']:.2f}")
    print(f"  ТБ1.5 {r['over_1_5']:.1%} | ТБ2.5 {r['over_2_5']:.1%} | ТБ3.5 {r['over_3_5']:.1%} | обе забьют {r['btts']:.1%}")
    print("  вероятные счета: " + ", ".join(f"{t['score']} ({t['p']:.1%})" for t in r["top_scores"]))
    print("  модели ансамбля: " + "; ".join(
        f"{m}: {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for m, v in r["members"].items()))
    for v, label in zip(r["value"] or [], ["П1", "X", "П2"]):
        mark = "  <-- модель выше рынка" if v["is_value"] else ""
        print(f"  {label} @ {v['odds']:.2f}: EV по модели {v['ev']:+.1%}{mark}")
    if r["value"]:
        print("  (в бэктесте ставки по сигналам «модель выше рынка» теряли 13–19% — рынок точнее модели)")


def _predict_or_exit(predictor, *a, **kw):
    try:
        return predictor.predict(*a, **kw)
    except (UnknownLeague, UnknownTeam, ValueError) as exc:
        sys.exit(str(exc))


def cmd_predict(args):
    predictor = _load_predictor()
    date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    if date <= predictor.last_date:
        print(f"Внимание: дата {date.date()} не позже последнего матча в данных ({predictor.last_date.date()}).")
    if args.fixtures:
        fx = pd.read_csv(args.fixtures)
        results = []
        for f in fx.to_dict("records"):
            d = pd.Timestamp(f["date"]) if "date" in f and pd.notna(f.get("date")) else date
            odds = tuple(f[k] for k in ("odds_h", "odds_d", "odds_a")) if "odds_h" in f and pd.notna(f["odds_h"]) else None
            r = _predict_or_exit(predictor, f["league"], f["home"], f["away"], d, odds)
            _print_prediction(r)
            results.append({k: v for k, v in r.items() if k not in ("top_scores", "members", "value", "fair_odds")})
        if args.out:
            pd.DataFrame(results).to_csv(args.out, index=False, float_format="%.4f")
            print(f"\nСохранено в {args.out}")
    else:
        if not (args.league and args.home and args.away):
            sys.exit("Укажите --league, --home и --away (или --fixtures файл.csv)")
        odds = tuple(args.odds) if args.odds else None
        _print_prediction(_predict_or_exit(predictor, args.league, args.home, args.away, date, odds))


def cmd_teams(args):
    predictor = _load_predictor()
    for league in args.leagues or data.LEAGUES:
        teams = predictor.teams(league)
        print(f"\n{league}: " + ", ".join(f"{t} ({e:.0f})" for t, e in teams))


def cmd_serve(args):
    import uvicorn

    os.environ.setdefault("FOOTBALL_MODEL_PATH", str(MODEL_PATH))
    uvicorn.run("football_predictor.api:app", host=args.host, port=args.port)


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data/raw")
    common.add_argument("--extra-csv", nargs="+", help="доп. CSV в формате football-data.co.uk (можно с коэффициентами)")
    common.add_argument("--leagues", nargs="+", help=f"подмножество лиг: {' '.join(data.LEAGUES)}")
    common.add_argument("--rebuild", action="store_true", help="пересчитать кэш признаков")
    common.add_argument("--no-market", action="store_true",
                        help="не использовать коэффициенты как признаки (чистая модель против рынка)")
    common.add_argument("--calibration", choices=["none", "temp", "dirichlet"], default="none",
                        help="калибровка вероятностей ансамбля (по умолчанию выключена — см. README)")
    common.add_argument("--lineups", action="store_true",
                        help="использовать стоимость объявленных стартовых составов (если они будут на прогнозе)")
    common.add_argument("--val-seasons", type=int, default=2, help="сезонов на валидацию (ранняя остановка, веса)")

    ap = argparse.ArgumentParser(prog="football_predictor", description="Прогнозирование футбольных матчей")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help):
        return sub.add_parser(name, help=help, parents=[common])

    p = add("download", "скачать исторические данные")
    p.add_argument("--first", type=int, default=data.FIRST_SEASON)
    p.add_argument("--last", type=int, default=data.LAST_SEASON)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-xg", action="store_true", help="не качать xG с understat.com")
    p.add_argument("--no-tm", action="store_true", help="не качать данные Transfermarkt (~60 МБ)")
    p.set_defaults(func=cmd_download)

    p = add("backtest", "честная walk-forward проверка по сезонам")
    p.add_argument("--seasons", type=int, default=5)
    p.add_argument("--train-from", type=int, default=1996)
    p.add_argument("--out")
    p.add_argument("--save-preds", help="CSV с прогнозами по каждому матчу")
    p.set_defaults(func=cmd_backtest)

    p = add("train", "обучить финальную модель на всей истории")
    p.add_argument("--train-from", type=int, default=1996)
    p.set_defaults(func=cmd_train)

    p = add("predict", "прогноз матча или списка матчей")
    p.add_argument("--league")
    p.add_argument("--home")
    p.add_argument("--away")
    p.add_argument("--date", help="дата матча YYYY-MM-DD (по умолчанию сегодня)")
    p.add_argument("--odds", type=float, nargs=3, metavar=("H", "D", "A"), help="коэффициенты для поиска value")
    p.add_argument("--fixtures", help="CSV: league,home,away[,date,odds_h,odds_d,odds_a]")
    p.add_argument("--out")
    p.set_defaults(func=cmd_predict)

    p = add("teams", "актуальные команды и их Elo")
    p.set_defaults(func=cmd_teams)

    p = add("serve", "HTTP API (FastAPI): документация на /docs")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
