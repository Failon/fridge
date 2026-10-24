"""Ансамбль моделей поверх признаков.

Члены ансамбля (все предсказывают вероятности П1/X/П2):
  1. LightGBM-классификатор (multiclass) — нелинейные взаимодействия признаков;
  2. мультиномиальная логистическая регрессия — устойчивый «линейный» взгляд;
  3. Пуассоновские LightGBM-регрессоры голов хозяев и гостей + поправка Диксона–Коулза ->
     матрица счёта -> П1/X/П2 (а также тоталы, обе забьют, точный счёт);
  4. чистый Диксон–Коулз.
Веса смешивания и калибровка подбираются на отложенных (самых свежих) сезонах минимизацией
log loss, после чего модели переобучаются на всех данных с найденным числом итераций.

Калибровка (опция): temp — температура и сдвиги (3 параметра), dirichlet — матрица 3×3 и сдвиги
над логарифмами вероятностей. По умолчанию выключена: в бэктесте 2022/23–2026/27 ансамбль и так
откалиброван (прогноз «75%» сбывается в ~77% случаев), а калибровка ухудшала log loss
(0.9782 → 0.9806) — сдвиги подстраивались под шум валидационных сезонов.
"""
from __future__ import annotations

import os
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message="X does not have valid feature names")

# Потоков LightGBM. Не «все ядра»: на мелких деревьях и в виртуалке (Docker на Windows/Mac)
# лишние потоки ждут друг друга, и обучение замедляется в десятки раз. Меняется через FP_THREADS.
N_THREADS = int(os.environ.get("FP_THREADS", min(4, os.cpu_count() or 1)))

CLF_PARAMS = dict(objective="multiclass", num_class=3, learning_rate=0.02, num_leaves=15,
                  min_child_samples=150, feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1,
                  lambda_l2=10.0, verbose=-1, n_jobs=N_THREADS)
REG_PARAMS = dict(objective="poisson", learning_rate=0.02, num_leaves=15, min_child_samples=150,
                  feature_fraction=0.6, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
                  verbose=-1, n_jobs=N_THREADS)
MEMBERS = ["lgbm", "logreg", "goals", "dc"]
CALIBRATIONS = ["none", "temp", "dirichlet"]
LINEUP_FEATURES = ["h_xi_dev", "a_xi_dev", "d_xi"]


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


class Calibrator:
    """Пересчитывает вероятности П1/X/П2: p' = softmax(W·log p + b)."""

    def __init__(self, kind: str = "temp", l2: float = 0.01):
        if kind not in CALIBRATIONS:
            raise ValueError(f"калибровка: {CALIBRATIONS}")
        self.kind, self.l2 = kind, l2
        self.W, self.b = np.eye(3), np.zeros(3)

    def _unpack(self, x):
        if self.kind == "temp":  # одна «температура» на всё + сдвиги ничьей и гостей
            return np.eye(3) * np.exp(x[0]), np.array([0.0, x[1], x[2]])
        return x[:9].reshape(3, 3), np.concatenate([[0.0], x[9:]])

    def fit(self, p: np.ndarray, y: np.ndarray) -> "Calibrator":
        if self.kind == "none":
            return self
        L = np.log(np.clip(p, 1e-12, None))

        def loss(x):
            W, b = self._unpack(x)
            q = _softmax(L @ W.T + b)
            reg = self.l2 * np.sum((W - np.eye(3)) ** 2) if self.kind == "dirichlet" else 0.0
            return -np.mean(np.log(np.clip(q[np.arange(len(y)), y], 1e-12, None))) + reg

        x0 = np.zeros(3) if self.kind == "temp" else np.concatenate([np.eye(3).ravel(), np.zeros(2)])
        self.W, self.b = self._unpack(minimize(loss, x0, method="L-BFGS-B").x)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if self.kind == "none":
            return p
        return _softmax(np.log(np.clip(p, 1e-12, None)) @ self.W.T + self.b)

    def describe(self) -> str:
        if self.kind == "temp":
            return f"температура {1 / self.W[0, 0]:.3f}, сдвиги X {self.b[1]:+.3f}, П2 {self.b[2]:+.3f}"
        return self.kind
DEFAULT_RHO = -0.05


def matrix_1x2(lh: np.ndarray, la: np.ndarray, rho: np.ndarray, max_goals: int = 10) -> np.ndarray:
    """Векторизованно: λ хозяев/гостей + ρ -> матрицы счёта Диксона–Коулза -> П1/X/П2."""
    g = np.arange(max_goals + 1)
    m = poisson.pmf(g[None, :], lh[:, None])[:, :, None] * poisson.pmf(g[None, :], la[:, None])[:, None, :]
    m[:, 0, 0] *= np.clip(1 - lh * la * rho, 1e-6, None)
    m[:, 0, 1] *= np.clip(1 + lh * rho, 1e-6, None)
    m[:, 1, 0] *= np.clip(1 + la * rho, 1e-6, None)
    m[:, 1, 1] *= np.clip(1 - rho, 1e-6, None)
    m /= m.sum((1, 2), keepdims=True)
    i, j = np.indices(m.shape[1:])
    return np.stack([m[:, i > j].sum(1), m[:, i == j].sum(1), m[:, i < j].sum(1)], 1)


def over_prob(lh: np.ndarray, la: np.ndarray, rho: np.ndarray, line: float = 2.5, max_goals: int = 10) -> np.ndarray:
    """P(голов больше line) по матрице счёта Диксона–Коулза (векторизованно)."""
    g = np.arange(max_goals + 1)
    m = poisson.pmf(g[None, :], lh[:, None])[:, :, None] * poisson.pmf(g[None, :], la[:, None])[:, None, :]
    m[:, 0, 0] *= np.clip(1 - lh * la * rho, 1e-6, None)
    m[:, 0, 1] *= np.clip(1 + lh * rho, 1e-6, None)
    m[:, 1, 0] *= np.clip(1 + la * rho, 1e-6, None)
    m[:, 1, 1] *= np.clip(1 - rho, 1e-6, None)
    m /= m.sum((1, 2), keepdims=True)
    i, j = np.indices(m.shape[1:])
    return m[:, (i + j) > line].sum(1)


def rho_of(X: pd.DataFrame) -> np.ndarray:
    return X["dc_rho"].fillna(DEFAULT_RHO).to_numpy(float)


class MatchPredictor:
    def __init__(self, seed: int = 0, use_market: bool | None = None, calibration: str = "none",
                 use_lineups: bool = False):
        """use_market: None — брать коэффициенты, если они есть у большинства матчей; False — никогда.
        calibration: none / temp / dirichlet (см. Calibrator).
        use_lineups: признаки стоимости объявленного старта. Включать, только если на прогнозе составы
        будут известны: модель, обученная с ними, без составов прогнозирует хуже (бэктест 2026/27)."""
        self.use_lineups = use_lineups
        if calibration not in CALIBRATIONS:
            raise ValueError(f"калибровка: {CALIBRATIONS}")
        self.seed = seed
        self.calibration = calibration
        self.calibrators: dict[str, Calibrator] = {}
        self.use_market = use_market
        self.feature_cols: list[str] = []
        self.leagues: list[str] = []
        self.members = list(MEMBERS)
        self.weights = np.full(len(MEMBERS), 1 / len(MEMBERS))
        self.best_iter = {"clf": 500, "hg": 500, "ag": 500}

    # ---------- подготовка ----------
    def _select_features(self, X: pd.DataFrame) -> list[str]:
        cols = [c for c in X.columns if c != "league"]
        # рыночные признаки берём, только если коэффициенты есть у заметной доли матчей
        market = self.use_market is not False and X["mkt_p_home"].notna().mean() >= 0.5
        if not market:
            cols = [c for c in cols if not c.startswith("mkt_")]
        if not self.use_lineups:
            cols = [c for c in cols if c not in LINEUP_FEATURES]
        self.members = MEMBERS + (["market"] if market else [])
        return cols

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        M = X[self.feature_cols].astype(float).copy()
        for lg in self.leagues:
            M[f"lg_{lg}"] = (X["league"] == lg).astype(float)
        return M

    # ---------- обучение ----------
    def _fit_members(self, X, y, hg, ag, iters=None, X_val=None, y_val=None, hg_val=None, ag_val=None):
        M = self._matrix(X)
        es = X_val is not None
        Mv = self._matrix(X_val) if es else None
        cb = [lgb.early_stopping(150, verbose=False)] if es else []
        it = iters or {"clf": 3000, "hg": 3000, "ag": 3000}

        self.clf = lgb.train({**CLF_PARAMS, "seed": self.seed}, lgb.Dataset(M, y), it["clf"],
                             valid_sets=[lgb.Dataset(Mv, y_val)] if es else None, callbacks=cb)
        self.reg_h = lgb.train({**REG_PARAMS, "seed": self.seed}, lgb.Dataset(M, hg), it["hg"],
                               valid_sets=[lgb.Dataset(Mv, hg_val)] if es else None, callbacks=cb)
        self.reg_a = lgb.train({**REG_PARAMS, "seed": self.seed}, lgb.Dataset(M, ag), it["ag"],
                               valid_sets=[lgb.Dataset(Mv, ag_val)] if es else None, callbacks=cb)
        if es:
            self.best_iter = {"clf": max(self.clf.best_iteration, 50), "hg": max(self.reg_h.best_iteration, 50),
                              "ag": max(self.reg_a.best_iteration, 50)}

        self.logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    LogisticRegression(C=0.05, max_iter=2000))
        self.logreg.fit(M, y)

    def member_probs(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        M = self._matrix(X)
        lh, la = self.reg_h.predict(M), self.reg_a.predict(M)
        p = {"lgbm": self.clf.predict(M), "logreg": self.logreg.predict_proba(M),
             "goals": matrix_1x2(lh, la, rho_of(X))}
        dc = X[["dc_p_home", "dc_p_draw", "dc_p_away"]].to_numpy(float, copy=True)
        missing = np.isnan(dc).any(1)
        dc[missing] = p["logreg"][missing]  # где DC ещё не обучен — подставляем логрег
        p["dc"] = dc
        if "market" in self.members:  # сам рынок — тоже член ансамбля
            mkt = X[["mkt_p_home", "mkt_p_draw", "mkt_p_away"]].to_numpy(float, copy=True)
            missing = np.isnan(mkt).any(1)
            mkt[missing] = p["lgbm"][missing]
            p["market"] = mkt
        return p

    def _blend(self, p: dict[str, np.ndarray], w: np.ndarray) -> np.ndarray:
        out = sum(wi * p[m] for wi, m in zip(w, self.members))
        return out / out.sum(1, keepdims=True)

    def _fit_weights(self, p: dict, y: np.ndarray) -> np.ndarray:
        def loss(z):
            w = np.exp(z) / np.exp(z).sum()
            pr = self._blend(p, w)
            return -np.mean(np.log(np.clip(pr[np.arange(len(y)), y], 1e-12, None)))
        res = minimize(loss, np.zeros(len(self.members)), method="Nelder-Mead", options={"maxiter": 2000})
        return np.exp(res.x) / np.exp(res.x).sum()

    def fit(self, X: pd.DataFrame, matches: pd.DataFrame, val_seasons: int = 1, verbose: bool = True):
        """X — признаки, matches — исходные матчи (те же индексы). Последний сезон(ы) — валидация."""
        self.feature_cols = self._select_features(X)
        self.leagues = sorted(X["league"].unique())
        y, hg, ag = matches["result"].to_numpy(), matches["hg"].to_numpy(), matches["ag"].to_numpy()
        seasons = matches["season"].to_numpy()
        val_from = np.sort(np.unique(seasons))[-val_seasons]
        tr, va = seasons < val_from, seasons >= val_from

        self._fit_members(X[tr], y[tr], hg[tr], ag[tr], X_val=X[va], y_val=y[va], hg_val=hg[va], ag_val=ag[va])
        p_va = self.member_probs(X[va])
        self.weights = self._fit_weights(p_va, y[va])
        blend_va = self._blend(p_va, self.weights)
        # все варианты калибровки — чтобы бэктест мог их сравнить; прогноз использует self.calibration
        self.calibrators = {k: Calibrator(k).fit(blend_va, y[va]) for k in CALIBRATIONS}
        if verbose:
            print("  итерации:", self.best_iter, "| веса:",
                  {m: round(float(w), 3) for m, w in zip(self.members, self.weights)},
                  "| калибровка:", self.calibrators["temp"].describe())
        # переобучение на всех данных (с небольшим запасом итераций под больший объём)
        iters = {k: int(v * 1.1) for k, v in self.best_iter.items()}
        self._fit_members(X, y, hg, ag, iters=iters)
        return self

    # ---------- прогноз ----------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw = self._blend(self.member_probs(X), self.weights)
        cal = getattr(self, "calibrators", {}).get(getattr(self, "calibration", "none"))  # старые модели — без
        return cal.transform(raw) if cal else raw

    def predict_variants(self, X: pd.DataFrame) -> dict[str, np.ndarray]:
        """Прогноз ансамбля при каждом варианте калибровки (для бэктеста)."""
        raw = self._blend(self.member_probs(X), self.weights)
        return {k: c.transform(raw) for k, c in self.calibrators.items()}

    def predict_goals(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        M = self._matrix(X)
        return self.reg_h.predict(M), self.reg_a.predict(M)

    def predict_over(self, X: pd.DataFrame, line: float = 2.5) -> np.ndarray:
        lh, la = self.predict_goals(X)
        return over_prob(lh, la, rho_of(X), line)

    def feature_importance(self, top: int = 20) -> pd.Series:
        imp = pd.Series(self.clf.feature_importance("gain"), index=self.clf.feature_name())
        return (imp / imp.sum()).sort_values(ascending=False).head(top)
