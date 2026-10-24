"""Модель Диксона–Коулза (1997) с затуханием по времени.

log λ_home = μ + h + att[home] + def[away]
log λ_away = μ     + att[away] + def[home]

Голы — независимые пуассоновские величины с поправкой τ(ρ) для счетов 0:0, 1:0, 0:1, 1:1
(в реальном футболе ничьих 0:0 и 1:1 больше, чем даёт чистый Пуассон).
Матчи взвешиваются как exp(-ξ·Δt), чтобы свежая форма весила больше.
Параметры команд регуляризуются L2 — это стабилизирует оценки при малом числе матчей.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10


def tau(hg, ag, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    t = np.where((hg == 0) & (ag == 0), 1 - lh * la * rho, t)
    t = np.where((hg == 0) & (ag == 1), 1 + lh * rho, t)
    t = np.where((hg == 1) & (ag == 0), 1 + la * rho, t)
    t = np.where((hg == 1) & (ag == 1), 1 - rho, t)
    return t


def score_matrix(lh: float, la: float, rho: float = 0.0, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Матрица вероятностей точного счёта P[i, j] = P(хозяева i, гости j)."""
    g = np.arange(max_goals + 1)
    m = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    m[0, 0] *= max(1 - lh * la * rho, 1e-6)
    m[0, 1] *= max(1 + lh * rho, 1e-6)
    m[1, 0] *= max(1 + la * rho, 1e-6)
    m[1, 1] *= max(1 - rho, 1e-6)
    return m / m.sum()


def markets(m: np.ndarray) -> dict:
    """Все основные рынки из матрицы счёта."""
    i, j = np.indices(m.shape)
    total = i + j
    return {
        "p_home": m[i > j].sum(), "p_draw": m[i == j].sum(), "p_away": m[i < j].sum(),
        "over_1_5": m[total > 1.5].sum(), "over_2_5": m[total > 2.5].sum(),
        "over_3_5": m[total > 3.5].sum(), "btts": m[(i > 0) & (j > 0)].sum(),
        "xg_home": (i * m).sum(), "xg_away": (j * m).sum(),
    }


class DixonColes:
    def __init__(self, xi: float = 0.0019, l2: float = 1.0):
        self.xi = xi      # затухание в день: 0.0019 -> полупериод ~1 год
        self.l2 = l2
        self.teams: dict[str, int] = {}
        self.params: np.ndarray | None = None
        self.rho = 0.0

    def fit(self, home, away, hg, ag, days_ago, warm_start: "DixonColes | None" = None, fit_rho: bool = True):
        """hg/ag могут быть и дробными (xG): пуассоновское правдоподобие от этого не ломается.
        Для xG поправка τ на низкие счета не имеет смысла — тогда fit_rho=False."""
        teams = sorted(set(home) | set(away))
        self.teams = {t: k for k, t in enumerate(teams)}
        n = len(teams)
        hi = np.array([self.teams[t] for t in home])
        ai = np.array([self.teams[t] for t in away])
        hg = np.asarray(hg, float)
        ag = np.asarray(ag, float)
        w = np.exp(-self.xi * np.asarray(days_ago, float))
        w = w / w.mean()

        x0 = np.zeros(2 * n + 2)
        x0[-2], x0[-1] = np.log(max(ag.mean(), 0.1)), 0.25
        if warm_start is not None and warm_start.params is not None:
            for t, k in self.teams.items():
                if t in warm_start.teams:
                    j = warm_start.teams[t]
                    x0[k], x0[n + k] = warm_start.params[j], warm_start.params[len(warm_start.teams) + j]
            x0[-2:] = warm_start.params[-2:]

        l2 = self.l2

        def nll(x):
            att, dfn, mu, h = x[:n], x[n:2 * n], x[-2], x[-1]
            llh = mu + h + att[hi] + dfn[ai]
            lla = mu + att[ai] + dfn[hi]
            lh, la = np.exp(llh), np.exp(lla)
            f = -np.sum(w * (hg * llh - lh + ag * lla - la)) + l2 * (att @ att + dfn @ dfn)
            rh, ra = w * (hg - lh), w * (ag - la)  # d(loglik)/d(log λ)
            g = np.empty_like(x)
            g[:n] = -(np.bincount(hi, rh, n) + np.bincount(ai, ra, n)) + 2 * l2 * att
            g[n:2 * n] = -(np.bincount(ai, rh, n) + np.bincount(hi, ra, n)) + 2 * l2 * dfn
            g[-2] = -(rh.sum() + ra.sum())
            g[-1] = -rh.sum()
            return f, g

        res = minimize(nll, x0, jac=True, method="L-BFGS-B")
        self.params = res.x

        if not fit_rho:
            self.rho = 0.0
            return self
        lh, la = self._lambdas(hi, ai)
        hg_i, ag_i = hg.astype(int), ag.astype(int)
        low = (hg_i <= 1) & (ag_i <= 1)  # τ влияет только на эти счета
        lim = 0.99 / max(1.0, float(np.max(lh[low] * la[low]))) if low.any() else 0.2
        r = minimize_scalar(
            lambda rho: -np.sum(w[low] * np.log(np.clip(
                tau(hg_i[low], ag_i[low], lh[low], la[low], rho), 1e-10, None))),
            bounds=(-min(0.3, lim), min(0.3, lim)), method="bounded")
        self.rho = float(r.x)
        return self

    def _lambdas(self, hi, ai):
        n = len(self.teams)
        att, dfn, mu, h = self.params[:n], self.params[n:2 * n], self.params[-2], self.params[-1]
        return np.exp(mu + h + att[hi] + dfn[ai]), np.exp(mu + att[ai] + dfn[hi])

    def team_strength(self, team: str, newcomer_q: float = 0.2) -> tuple[float, float]:
        """(атака, оборона). Для неизвестной (новой) команды — слабый квантиль лиги."""
        n = len(self.teams)
        att, dfn = self.params[:n], self.params[n:2 * n]
        if team in self.teams:
            k = self.teams[team]
            return att[k], dfn[k]
        return float(np.quantile(att, newcomer_q)), float(np.quantile(dfn, 1 - newcomer_q))

    def predict_lambdas(self, home: str, away: str, q_home: float = 0.2, q_away: float = 0.2) -> tuple[float, float]:
        """q_* — квантиль силы для команды, которой нет в выборке (новичок лиги)."""
        ah, dh = self.team_strength(home, q_home)
        aa, da = self.team_strength(away, q_away)
        mu, h = self.params[-2], self.params[-1]
        return float(np.exp(mu + h + ah + da)), float(np.exp(mu + aa + dh))

    def predict(self, home: str, away: str, q_home: float = 0.2, q_away: float = 0.2) -> dict:
        lh, la = self.predict_lambdas(home, away, q_home, q_away)
        out = markets(score_matrix(lh, la, self.rho))
        out.update(lambda_home=lh, lambda_away=la)
        return out
