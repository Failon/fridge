"""Есть ли у модели преимущество на тоталах и в низших/небольших лигах?

Walk-forward: для каждого тестового сезона модель учится на всех лигах по прошлым сезонам
(без коэффициентов в признаках), затем прогнозирует исход (П1/X/П2) и тотал 2.5.
Сравнение:
  * точность — с закрывающей линией Pinnacle без маржи (log loss);
  * прибыль — ставка 1 ед., когда EV по модели выше порога, по коэффициентам Bet365 (одна реальная
    контора) и по лучшему коэффициенту рынка (Max — верхняя граница, в жизни недостижима целиком).

Запуск:  python -m football_predictor download --leagues all --first 2000
         python analysis/markets_backtest.py      (≈30–40 минут, результат — analysis/markets_backtest.txt)
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from football_predictor import data  # noqa: E402
from football_predictor.features import build_features  # noqa: E402
from football_predictor.model import MatchPredictor  # noqa: E402

TEST_SEASONS = [2021, 2022, 2023, 2024, 2025]
TRAIN_FROM = 2000
CACHE = ROOT / "data/features_all.pkl"
OUT = ROOT / "analysis/markets_backtest.txt"
THRESHOLDS = [0.0, 0.03, 0.05, 0.10]

ODDS_COLS = {  # имя -> варианты колонок по сезонам (берётся первая найденная)
    "b365": [("B365H", "B365D", "B365A")],
    "ps": [("PSH", "PSD", "PSA")],
    "psc": [("PSCH", "PSCD", "PSCA")],
    "max": [("MaxH", "MaxD", "MaxA"), ("BbMxH", "BbMxD", "BbMxA")],
    "b365_ou": [("B365>2.5", "B365<2.5")],
    "ps_ou": [("P>2.5", "P<2.5")],
    "psc_ou": [("PC>2.5", "PC<2.5")],
    "max_ou": [("Max>2.5", "Max<2.5"), ("BbMx>2.5", "BbMx<2.5")],
}


def load_odds() -> pd.DataFrame:
    frames = []
    for path in sorted((ROOT / "data/raw").glob("*.csv")):
        league, code = path.stem.rsplit("_", 1)
        year = 1900 + int(code[:2]) if int(code[:2]) >= 90 else 2000 + int(code[:2])
        if year not in TEST_SEASONS:
            continue
        df = data.read_csv(path)
        df.columns = df.columns.str.replace("ï»¿", "").str.strip()
        out = pd.DataFrame({"league": league, "date": pd.to_datetime(df["Date"], format="mixed", dayfirst=True),
                            "home": df["HomeTeam"].str.strip()})
        for name, variants in ODDS_COLS.items():
            cols = next((v for v in variants if all(c in df for c in v)), None)
            for i in range(len(variants[0])):
                out[f"{name}{i}"] = pd.to_numeric(df[cols[i]], errors="coerce") if cols else np.nan
        frames.append(out)
    return pd.concat(frames).drop_duplicates(["league", "date", "home"])


def fair(odds: np.ndarray) -> np.ndarray:
    inv = 1 / odds
    return inv / inv.sum(1, keepdims=True)


def valid(df: pd.DataFrame, *cols) -> pd.DataFrame:
    """Строки, где все нужные коэффициенты есть и > 1 (в файлах встречаются пустые и нули)."""
    ok = np.ones(len(df), bool)
    for c in cols:
        ok &= (pd.to_numeric(df[c], errors="coerce") > 1).to_numpy()
    return df[ok]


def logloss(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, None))))


def roi_table(prob, odds, y, seasons, label, lines):
    """Ставим на каждый исход, где prob·odds − 1 > порога."""
    n_out = prob.shape[1]
    for thr in THRESHOLDS:
        ok = ~np.isnan(odds).any(1)
        ev = prob * odds - 1
        sel = (ev > thr) & ok[:, None]
        win = np.arange(n_out)[None, :] == y[:, None]
        profit = np.where(win, odds - 1, -1.0)[sel]
        if len(profit) < 30:
            lines.append(f"    {label:<24} EV>{thr:>4.0%}: мало ставок ({len(profit)})")
            continue
        se = profit.std() / np.sqrt(len(profit))
        per = pd.Series(profit).groupby(np.broadcast_to(seasons[:, None], sel.shape)[sel]).mean()
        lines.append(f"    {label:<24} EV>{thr:>4.0%}: ставок {len(profit):6d}  ROI {profit.mean():+6.2%} "
                     f"(±{1.96 * se:.1%})  сезонов в плюсе {int((per > 0).sum())}/{len(per)}")


def main():
    if "--summary" in sys.argv:  # только пересчитать сводку по сохранённым прогнозам
        summarize(pd.read_csv(ROOT / "data/markets_preds.csv", parse_dates=["date"]))
        return
    matches = data.load_matches()
    if CACHE.exists() and pickle.load(open(CACHE, "rb"))["n"] == len(matches):
        X = pickle.load(open(CACHE, "rb"))["X"]
    else:
        print(f"признаки для {len(matches)} матчей всех лиг...")
        X, _ = build_features(matches)
        pickle.dump({"n": len(matches), "X": X}, open(CACHE, "wb"))

    preds = []
    for s in TEST_SEASONS:
        tr = ((matches.season >= TRAIN_FROM) & (matches.season < s)).to_numpy()
        te = (matches.season == s).to_numpy()
        print(f"сезон {s}: обучение {tr.sum()}, тест {te.sum()}")
        model = MatchPredictor(use_market=False).fit(X[tr], matches[tr], val_seasons=2)
        p = model.predict_proba(X[te])
        po = model.predict_over(X[te])
        t = matches.loc[te, ["league", "season", "date", "home", "away", "result", "hg", "ag"]].copy()
        t[["m0", "m1", "m2"]] = p
        t["m_over"] = po
        preds.append(t)
    pr = pd.concat(preds).merge(load_odds(), on=["league", "date", "home"], how="left")
    pr.to_csv(ROOT / "data/markets_preds.csv", index=False)
    summarize(pr)


def summarize(pr: pd.DataFrame):
    lines = []
    pr["group"] = np.where(pr.league.isin(data.LEAGUES), "топ-5", "остальные")
    for gname, g in [("ТОП-5 ЛИГ", pr[pr.group == "топ-5"]), ("ОСТАЛЬНЫЕ 15 ЛИГ", pr[pr.group == "остальные"])]:
        lines.append(f"\n=== {gname} ===")
        # ---- исходы 1X2
        g1 = valid(g, "psc0", "psc1", "psc2", "ps0", "ps1", "ps2")
        y = g1.result.to_numpy()
        m = g1[["m0", "m1", "m2"]].to_numpy()
        lines.append(f"  ИСХОДЫ П1/X/П2 ({len(g1)} матчей с закрытием Pinnacle)")
        lines.append(f"    log loss: модель {logloss(m, y):.4f} | Pinnacle ранние "
                     f"{logloss(fair(g1[['ps0', 'ps1', 'ps2']].to_numpy()), y):.4f} | Pinnacle закрытие "
                     f"{logloss(fair(g1[['psc0', 'psc1', 'psc2']].to_numpy()), y):.4f}")
        roi_table(m, g1[["b3650", "b3651", "b3652"]].to_numpy(), y, g1.season.to_numpy(), "модель → Bet365", lines)
        roi_table(m, g1[["max0", "max1", "max2"]].to_numpy(), y, g1.season.to_numpy(), "модель → лучший кэф", lines)
        roi_table(fair(g1[["ps0", "ps1", "ps2"]].to_numpy()), g1[["max0", "max1", "max2"]].to_numpy(), y,
                  g1.season.to_numpy(), "Pinnacle → лучший кэф", lines)

        # ---- тотал 2.5
        g2 = valid(g, "psc_ou0", "psc_ou1", "ps_ou0", "ps_ou1")
        yo = ((g2.hg + g2.ag) < 2.5).astype(int).to_numpy()  # 0 = больше, 1 = меньше
        mo = np.c_[g2.m_over, 1 - g2.m_over]
        lines.append(f"  ТОТАЛ 2.5 ({len(g2)} матчей с закрытием Pinnacle; больше 2.5 было в {np.mean(yo == 0):.1%})")
        lines.append(f"    log loss: модель {logloss(mo, yo):.4f} | Pinnacle ранние "
                     f"{logloss(fair(g2[['ps_ou0', 'ps_ou1']].to_numpy()), yo):.4f} | Pinnacle закрытие "
                     f"{logloss(fair(g2[['psc_ou0', 'psc_ou1']].to_numpy()), yo):.4f}")
        roi_table(mo, g2[["b365_ou0", "b365_ou1"]].to_numpy(), yo, g2.season.to_numpy(), "модель → Bet365", lines)
        roi_table(mo, g2[["max_ou0", "max_ou1"]].to_numpy(), yo, g2.season.to_numpy(), "модель → лучший кэф", lines)
        roi_table(fair(g2[["ps_ou0", "ps_ou1"]].to_numpy()), g2[["max_ou0", "max_ou1"]].to_numpy(), yo,
                  g2.season.to_numpy(), "Pinnacle → лучший кэф", lines)

    # отдельные лиги: точность исходов модель vs закрытие
    lines.append("\n=== ПО ЛИГАМ: log loss исходов, модель − закрытие Pinnacle (меньше 0 — модель точнее) ===")
    for lg, g in valid(pr, "psc0", "psc1", "psc2").groupby("league"):
        y = g.result.to_numpy()
        d1 = logloss(g[["m0", "m1", "m2"]].to_numpy(), y) - logloss(fair(g[["psc0", "psc1", "psc2"]].to_numpy()), y)
        g2 = valid(g, "psc_ou0", "psc_ou1")
        yo = ((g2.hg + g2.ag) < 2.5).astype(int).to_numpy()
        d2 = logloss(np.c_[g2.m_over, 1 - g2.m_over], yo) - logloss(fair(g2[["psc_ou0", "psc_ou1"]].to_numpy()), yo)
        lines.append(f"  {lg:<24} матчей {len(g):5d} | исходы {d1:+.4f} | тотал {d2:+.4f}")

    text = "\n".join(lines)
    OUT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
