"""Командная строка: download / backtest / train / predict / teams."""
from __future__ import annotations

import argparse
import difflib
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import data
from .dixon_coles import markets, score_matrix
from .features import build_features
from .metrics import report
from .model import MEMBERS, MatchPredictor, rho_of

FEATURES_CACHE = Path("data/features.pkl")
MODEL_PATH = Path("models/predictor.pkl")


def _load_features(args):
    """Матчи + признаки + состояние движка (кэшируется, т.к. проход по истории занимает минуты)."""
    matches = data.load_matches(args.data_dir, args.extra_csv, args.odds_dir)
    key = (len(matches), str(matches["date"].max()), tuple(args.extra_csv or []))
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
    if not args.no_odds:
        print("Коэффициенты букмекеров (архив jokecamp/FootballData)...")
        data.download_odds(args.leagues or data.LEAGUES, args.odds_dir, args.force)


def cmd_backtest(args):
    matches, X, _ = _load_features(args)
    matches, X = _filter(matches, X, args)
    seasons = sorted(matches["season"].unique())
    test_seasons = seasons[-args.seasons:]
    rows, all_probs, all_y = [], {}, []
    has_market = X["mkt_p_home"].notna().mean() > 0.5

    for s in test_seasons:
        tr, te = (matches["season"] < s).to_numpy(), (matches["season"] == s).to_numpy()
        print(f"\n=== Тестовый сезон {data.season_code(s)}: обучение на {tr.sum()} матчах, тест {te.sum()}")
        model = MatchPredictor().fit(X[tr], matches[tr], val_seasons=args.val_seasons)
        y = matches.loc[te, "result"].to_numpy()
        probs = model.member_probs(X[te])
        probs["ENSEMBLE"] = model.predict_proba(X[te])
        base = np.bincount(matches.loc[tr, "result"], minlength=3) / tr.sum()
        probs["base_rate"] = np.tile(base, (te.sum(), 1))
        if has_market:
            probs["market"] = X.loc[te, ["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy()
        for name, p in probs.items():
            ok = ~np.isnan(p).any(1)
            rows.append({"season": data.season_code(s), "model": name, **report(p[ok], y[ok])})
            all_probs.setdefault(name, []).append(p)
        all_y.append(y)

    df = pd.DataFrame(rows)
    y = np.concatenate(all_y)
    total = []
    for name, ps in all_probs.items():
        p = np.concatenate(ps)
        ok = ~np.isnan(p).any(1)
        total.append({"model": name, "n": int(ok.sum()), **report(p[ok], y[ok])})
    total = pd.DataFrame(total).sort_values("RPS")
    pd.set_option("display.width", 160)
    print("\nПо сезонам (ансамбль):")
    print(df[df.model == "ENSEMBLE"].drop(columns="model").to_string(index=False, float_format="%.4f"))
    print(f"\nИТОГО за {len(test_seasons)} сезон(а/ов), walk-forward (меньше RPS/LogLoss/Brier — лучше):")
    print(total.to_string(index=False, float_format="%.4f"))
    if args.out:
        total.to_csv(args.out, index=False)


def cmd_train(args):
    matches, X, engine = _load_features(args)
    m, Xf = _filter(matches, X, args)
    print(f"Обучение на {len(m)} матчах (сезоны {data.season_code(m.season.min())}–{data.season_code(m.season.max())})")
    model = MatchPredictor().fit(Xf, m, val_seasons=args.val_seasons)
    print("\nТоп признаков (LightGBM, доля gain):")
    print(model.feature_importance(15).to_string(float_format="%.3f"))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "engine": engine, "last_date": matches["date"].max()}, f)
    print(f"\nМодель сохранена в {MODEL_PATH}")


def _load_model():
    if not MODEL_PATH.exists():
        sys.exit("Модель не найдена. Выполните `python -m football_predictor train`.")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def _resolve_team(engine, league, name):
    known = sorted({t for (lg, t) in engine.teams if lg == league})
    if not known:
        sys.exit(f"Нет данных по лиге {league}. Доступны: {', '.join(sorted({lg for lg, _ in engine.teams}))}")
    if name in known:
        return name
    lower = {t.lower(): t for t in known}
    if name.lower() in lower:
        return lower[name.lower()]
    close = difflib.get_close_matches(name, known, n=1, cutoff=0.6)
    if close:
        print(f"  '{name}' -> '{close[0]}'")
        return close[0]
    sys.exit(f"Команда '{name}' не найдена в {league}. Похожие: {difflib.get_close_matches(name, known, 5, 0.3)}")


def _predict_one(bundle, league, home, away, date, odds=None):
    model, engine = bundle["model"], bundle["engine"]
    home, away = _resolve_team(engine, league, home), _resolve_team(engine, league, away)
    feats = engine.features(league, home, away, date, odds, refit=False)
    X = pd.DataFrame([feats])
    p = model.predict_proba(X)[0]
    lh, la = (v[0] for v in model.predict_goals(X))
    mtx = score_matrix(lh, la, float(rho_of(X)[0]))
    mk = markets(mtx)
    top = sorted(((mtx[i, j], f"{i}:{j}") for i in range(6) for j in range(6)), reverse=True)[:5]
    return {"league": league, "home": home, "away": away, "p_home": p[0], "p_draw": p[1], "p_away": p[2],
            "xg_home": lh, "xg_away": la, "over_1_5": mk["over_1_5"], "over_2_5": mk["over_2_5"],
            "over_3_5": mk["over_3_5"], "btts": mk["btts"], "top_scores": top,
            "members": {m: model.member_probs(X)[m][0] for m in MEMBERS}}


def _print_prediction(r, odds=None):
    print(f"\n{r['home']} — {r['away']}  ({r['league']})")
    print(f"  П1 {r['p_home']:6.1%}   X {r['p_draw']:6.1%}   П2 {r['p_away']:6.1%}")
    print(f"  справедливые коэф.:  {1 / r['p_home']:.2f} / {1 / r['p_draw']:.2f} / {1 / r['p_away']:.2f}")
    print(f"  ожидаемые голы: {r['xg_home']:.2f} — {r['xg_away']:.2f}")
    print(f"  ТБ1.5 {r['over_1_5']:.1%} | ТБ2.5 {r['over_2_5']:.1%} | ТБ3.5 {r['over_3_5']:.1%} | обе забьют {r['btts']:.1%}")
    print("  вероятные счета: " + ", ".join(f"{s} ({p:.1%})" for p, s in r["top_scores"]))
    print("  модели ансамбля: " + "; ".join(
        f"{m}: {v[0]:.2f}/{v[1]:.2f}/{v[2]:.2f}" for m, v in r["members"].items()))
    if odds:
        probs = [r["p_home"], r["p_draw"], r["p_away"]]
        for label, p, o in zip(["П1", "X", "П2"], probs, odds):
            ev = p * o - 1
            kelly = max(ev / (o - 1), 0) / 4  # четверть Келли — консервативный размер ставки
            mark = "  <-- value" if ev > 0.03 else ""
            print(f"  {label} @ {o:.2f}: EV {ev:+.1%}, ставка (1/4 Келли) {kelly:.1%} банка{mark}")


def cmd_predict(args):
    bundle = _load_model()
    date = pd.Timestamp(args.date) if args.date else pd.Timestamp.today().normalize()
    if date <= bundle["last_date"]:
        print(f"Внимание: дата {date.date()} не позже последнего матча в данных ({bundle['last_date'].date()}).")
    if args.fixtures:
        fx = pd.read_csv(args.fixtures)
        results = []
        for f in fx.to_dict("records"):
            d = pd.Timestamp(f["date"]) if "date" in f and pd.notna(f.get("date")) else date
            odds = tuple(f[k] for k in ("odds_h", "odds_d", "odds_a")) if "odds_h" in f and pd.notna(f["odds_h"]) else None
            r = _predict_one(bundle, f["league"], f["home"], f["away"], d, odds)
            _print_prediction(r, odds)
            results.append({k: v for k, v in r.items() if k not in ("top_scores", "members")})
        if args.out:
            pd.DataFrame(results).to_csv(args.out, index=False, float_format="%.4f")
            print(f"\nСохранено в {args.out}")
    else:
        if not (args.league and args.home and args.away):
            sys.exit("Укажите --league, --home и --away (или --fixtures файл.csv)")
        odds = tuple(args.odds) if args.odds else None
        _print_prediction(_predict_one(bundle, args.league, args.home, args.away, date, odds), odds)


def cmd_teams(args):
    engine = _load_model()["engine"]
    last = max(s.last_date for s in engine.teams.values())
    for league in args.leagues or data.LEAGUES:
        teams = sorted(((s.elo, t) for (lg, t), s in engine.teams.items()
                        if lg == league and (last - s.last_date).days < 365), reverse=True)
        print(f"\n{league}: " + ", ".join(f"{t} ({e:.0f})" for e, t in teams))


def main(argv=None):
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default="data/raw")
    common.add_argument("--odds-dir", default="data/odds")
    common.add_argument("--extra-csv", nargs="+", help="доп. CSV в формате football-data.co.uk (можно с коэффициентами)")
    common.add_argument("--leagues", nargs="+", help=f"подмножество лиг: {' '.join(data.LEAGUES)}")
    common.add_argument("--rebuild", action="store_true", help="пересчитать кэш признаков")
    common.add_argument("--val-seasons", type=int, default=2, help="сезонов на валидацию (ранняя остановка, веса)")

    ap = argparse.ArgumentParser(prog="football_predictor", description="Прогнозирование футбольных матчей")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, help):
        return sub.add_parser(name, help=help, parents=[common])

    p = add("download", "скачать исторические данные")
    p.add_argument("--first", type=int, default=data.FIRST_SEASON)
    p.add_argument("--last", type=int, default=data.LAST_SEASON)
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-odds", action="store_true", help="не качать архив коэффициентов")
    p.set_defaults(func=cmd_download)

    p = add("backtest", "честная walk-forward проверка по сезонам")
    p.add_argument("--seasons", type=int, default=5)
    p.add_argument("--train-from", type=int, default=1996)
    p.add_argument("--out")
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

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
