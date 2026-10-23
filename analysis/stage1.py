"""Этап 1: разведка данных АПЛ за 10 сезонов и планка букмекера.

Запуск:  python analysis/stage1.py   (данные берутся из data/raw, см. `python -m football_predictor download`)
Результат: цифры в консоли + analysis/home_win_share.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

ROOT = Path(__file__).resolve().parent.parent
SEASONS = ["1415", "1516", "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324"]
COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "B365H", "B365D", "B365A"]
TEST = "2324"
LABELS = ["A", "D", "H"]  # порядок столбцов вероятностей для log_loss

# ---------- загрузка ----------
dfs = []
for s in SEASONS:
    d = pd.read_csv(ROOT / f"data/raw/premier-league_{s}.csv", encoding="latin-1")[COLS]
    d["Season"] = s
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True).dropna()
df["Date"] = pd.to_datetime(df["Date"], format="mixed", dayfirst=True)
df = df.sort_values("Date").reset_index(drop=True)

# ---------- Задание 1. Разведка ----------
print("=== Задание 1. Разведка ===")
print(f"1. Матчей: {len(df)}")
shares = df["FTR"].value_counts(normalize=True)
print(f"2. Исходы: H {shares['H']:.1%}, D {shares['D']:.1%}, A {shares['A']:.1%}")
print(f"3. Голов за матч: {(df.FTHG + df.FTAG).mean():.3f} "
      f"(хозяева {df.FTHG.mean():.3f}, гости {df.FTAG.mean():.3f})")

home_share = df.groupby("Season")["FTR"].apply(lambda r: (r == "H").mean())
print("4. Доля побед хозяев по сезонам:")
for s, v in home_share.items():
    print(f"   {s}: {v:.1%}" + ("   <- без зрителей" if s == "2021" else ""))

wins = pd.concat([df.loc[df.FTR == "H", "HomeTeam"], df.loc[df.FTR == "A", "AwayTeam"]]).value_counts()
print("5. Топ-5 по победам:")
for team, n in wins.head(5).items():
    print(f"   {team}: {n}")

fig, ax = plt.subplots(figsize=(9, 4.5))
labels = [f"{s[:2]}/{s[2:]}" for s in home_share.index]
colors = ["#d62728" if s == "2021" else "#1f77b4" for s in home_share.index]
bars = ax.bar(labels, home_share.values * 100, color=colors)
ax.axhline(home_share.mean() * 100, color="gray", ls="--", lw=1, label=f"среднее {home_share.mean():.1%}")
for b, v in zip(bars, home_share.values):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.5, f"{v:.0%}", ha="center", fontsize=9)
ax.set_ylim(0, 60)
ax.set_ylabel("Победы хозяев, %")
ax.set_title("АПЛ: доля побед хозяев по сезонам (красный — 2020/21 без зрителей)")
ax.legend(loc="upper right")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
out_png = ROOT / "analysis/home_win_share.png"
fig.savefig(out_png, dpi=120)
print(f"   график: {out_png.relative_to(ROOT)}")

# ---------- Задание 2. Вероятности букмекера ----------
print("\n=== Задание 2. Вероятности букмекера ===")
inv = 1 / df[["B365H", "B365D", "B365A"]]
df["margin"] = inv.sum(axis=1) - 1
df[["pH", "pD", "pA"]] = inv.div(inv.sum(axis=1), axis=0).to_numpy()
print(f"1. Средняя маржа Bet365: {df.margin.mean():.2%} (мин {df.margin.min():.2%}, макс {df.margin.max():.2%})")
print("   по сезонам: " + ", ".join(f"{s} {v:.1%}" for s, v in df.groupby('Season').margin.mean().items()))
print(f"2. Сумма pH+pD+pA = 1 во всех строках: {np.allclose(df[['pH', 'pD', 'pA']].sum(axis=1), 1)}")
fav = df[["pH", "pD", "pA"]].to_numpy().argmax(axis=1)
fav_label = np.array(["H", "D", "A"])[fav]
print(f"3. Фаворит букмекера побеждает в {(fav_label == df.FTR).mean():.1%} матчей "
      f"(ничья фаворитом: {(fav_label == 'D').sum()} раз)")

# ---------- Задание 3. Бенчмарк ----------
print(f"\n=== Задание 3. Бенчмарк log loss на сезоне {TEST} ===")
train, test = df[df.Season < TEST], df[df.Season == TEST]
y = test["FTR"]
naive = np.full((len(test), 3), 1 / 3)
freq = train["FTR"].value_counts(normalize=True)[LABELS].to_numpy()
bookie = test[["pA", "pD", "pH"]].to_numpy()
results = {
    "Наивный (1/3)": log_loss(y, naive, labels=LABELS),
    f"Частоты {SEASONS[0]}–{SEASONS[-2]}": log_loss(y, np.tile(freq, (len(test), 1)), labels=LABELS),
    "Букмекер (Bet365)": log_loss(y, bookie, labels=LABELS),
}
print(f"   частоты H/D/A из прошлых сезонов: {freq[2]:.3f} / {freq[1]:.3f} / {freq[0]:.3f}")
for name, v in results.items():
    print(f"   {name:<22} {v:.4f}")
